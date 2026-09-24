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

//! Configuration for `orangu-server`: a single `[orangu-server]` section
//! naming the models directory, and the address the HTTP server binds to.

use crate::engine::backend::DeviceRequest;
use crate::engine::placement::SplitMode;
use crate::engine::prefill_backend::PrefillBackend;
use anyhow::{Context, Result, anyhow, bail};
use orangu::config::parse_ini_sections;
use orangu::logging::LogTarget;
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

pub const SERVER_SECTION: &str = "orangu-server";

/// How many tokens a draft model proposes per verification when
/// `[orangu-server].draft_tokens` says nothing.
///
/// Four, matching the prompt-lookup path's own default, and for the same
/// reason: every drafted token past the first is only reached if all before it
/// were accepted, so the marginal value of a fifth is the pair agreeing five
/// times running. Raising it pays on highly predictable text and costs on
/// everything else.
pub const DEFAULT_DRAFT_TOKENS: usize = 4;

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

/// The web console's own section. **Its presence is what enables the
/// console** — a config with no `[web]` section binds no second listener at
/// all, which is what `--init` writes when the web console is declined.
pub const WEB_SECTION: &str = "web";

/// The dedicated Prometheus listener's own section, mirroring [`WEB_SECTION`]:
/// its *presence* is what enables the listener. A config with no
/// `[prometheus]` section binds no third listener at all.
pub const PROMETHEUS_SECTION: &str = "prometheus";

/// The `host` value meaning "every network interface on this machine" —
/// the default, and what `--init`'s `host` prompt offers first. `*` is
/// accepted as an alias for it, since that is the spelling most other
/// server config files use for the same idea.
pub const HOST_ALL: &str = "all";
pub const HOST_ALL_ALIAS: &str = "*";

pub fn default_host() -> String {
    HOST_ALL.to_string()
}

/// Turns a configured `host` into an address [`std::net::TcpListener::bind`]
/// actually understands: [`HOST_ALL`] (and its `*` alias) become the IPv4
/// wildcard `0.0.0.0`, so the listener answers on every interface rather
/// than only the loopback one; anything else — a literal interface address
/// such as `127.0.0.1` or `192.168.1.10` — is passed through untouched and
/// left for `bind` itself to reject if it isn't one of this machine's.
pub fn resolve_bind_host(host: &str) -> &str {
    let host = host.trim();
    if host.eq_ignore_ascii_case(HOST_ALL) || host == HOST_ALL_ALIAS {
        "0.0.0.0"
    } else {
        host
    }
}

pub fn default_port() -> u16 {
    8100
}

/// The resolved web-console port when there is no `[web]` section (and no
/// legacy `[orangu-server].web` either): `0`, meaning no second listener is
/// bound at all.
pub fn default_web() -> u16 {
    0
}

/// The port a `[web]` section that doesn't name one gets. Adjacent to the
/// API's own default so the pair reads as one server, and the value the
/// manual's example has always used.
pub fn default_web_port() -> u16 {
    8101
}

/// The resolved Prometheus-listener port when there is no `[prometheus]`
/// section: `0`, meaning no third listener is bound at all.
pub fn default_metrics() -> u16 {
    0
}

/// The port a `[prometheus]` section that doesn't name one gets. Adjacent to
/// the API's and web console's own defaults (`8100`/`8101`) so the trio reads
/// as one server, and the value `-i`/`--init` has always offered.
pub fn default_prometheus_port() -> u16 {
    8300
}

/// The address a bundled server binds when it was started with no config
/// file at all (see [`bundled_configuration`]) — the loopback interface,
/// not [`HOST_ALL`].
///
/// A bundle is one file somebody downloaded and ran, quite possibly on a
/// laptop on a network they don't administer. The ordinary `orangu-server`
/// default of every interface is a deliberate choice made in a config file
/// somebody wrote; it should not be what a binary does because it was
/// double-clicked. Writing an `orangu-server.conf` with `host = all` is all
/// it takes to opt back in.
pub const BUNDLED_HOST: &str = "127.0.0.1";

/// The web console port a bundled server takes, alongside
/// [`default_port`]'s `8100` for the API. Far enough from the API's port to
/// leave the usual `8101`, `8102`, … free for the other servers a machine
/// running several models ends up with.
pub fn bundled_web_port() -> u16 {
    8200
}

/// Where a bundle listens by default: whatever `bundle`'s own
/// `--host`/`--port`/`--web` were given, recorded in the bundle and read back
/// at startup.
///
/// Every field is optional, and an absent one means the built-in default
/// ([`BUNDLED_HOST`], [`default_port`], [`bundled_web_port`]) rather than
/// nothing — a bundle built before these existed, or built without them,
/// keeps exactly the behaviour it had.
///
/// This is the same idea as the bundle's role: a bundle is a server somebody
/// will run *without a config file*, so anything that would otherwise need
/// one has to be decidable when it is built. Without it, a bundle meant for a
/// LAN would need `--host all` typed at it on every start, on every machine.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BundledListen {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub web: Option<u16>,
    pub metrics: Option<u16>,
}

/// The configuration a bundled `orangu-server` runs on when it finds no
/// config file: `127.0.0.1:8100` for the API, `127.0.0.1:8200` for the web
/// console, and the role the bundle was built with — each overridden by
/// whatever `listen` records.
///
/// This is what "no configuration required" actually means — not a config
/// file written on first run (which would then have to be found, kept in
/// step with the binary, and explained), but a set of answers the binary
/// already has. A config file that *is* present still wins, in full and
/// unchanged: a bundle is a starting point, not a locked-down appliance.
///
/// `models` is still needed even though the served model is embedded: the
/// web console's model manager lists it, `download` fetches into it, and a
/// bundled server can be pointed at an ordinary model like any other. It
/// need not exist — an empty listing is the correct answer for a machine
/// that has only ever run the bundle.
pub fn bundled_configuration(
    models: PathBuf,
    role: Role,
    listen: &BundledListen,
) -> ServerConfiguration {
    let host = listen
        .host
        .clone()
        .unwrap_or_else(|| BUNDLED_HOST.to_string());
    ServerConfiguration {
        models,
        // A bundle carries one model and no companion projector, so there is
        // nothing for the NPU precompile step to find; leaving it on costs a
        // probe that fails immediately.
        npu_precompile: default_npu_precompile(),
        prefill_backend: PrefillBackend::Auto,
        npu_cache_gb: default_npu_cache_gb(),
        mlp_unroll: None,
        // The console follows the API's address, baked-in or default —
        // `bundle --host all` means "expose this bundle", not "expose half
        // of it". `--web 0` at build time, or at run time, is how a bundle
        // exposes only the API.
        web_host: host.clone(),
        metrics_host: host.clone(),
        host,
        port: listen.port.unwrap_or_else(default_port),
        slots: role.default_slots(),
        web: listen.web.unwrap_or_else(bundled_web_port),
        // Off unless `bundle` was told a port, same as `web`.
        metrics: listen.metrics.unwrap_or_else(default_metrics),
        // Nothing wrote a `[web].host` here, so `--host` at run time moves
        // the console along with the API — which is what makes `--host all`
        // on a bundle do the one thing somebody would reach for it to do.
        web_host_explicit: false,
        metrics_host_explicit: false,
        backend: default_backend(),
        // A bundle runs on a machine nobody configured, so it takes the
        // default rather than a lossy format nobody chose.
        kv_cache: KvCache::default(),
        read_size: DEFAULT_READ_SIZE,
        // A bundle serves whoever runs it; refusing on its owner's behalf is
        // not a decision it can make.
        queue_limit: 0,
        context: None,
        // A bundle carries one model. Pairing it with a draft would mean
        // embedding a second, which is a different product decision than
        // "the server and a model as one file".
        draft_model: None,
        draft_tokens: DEFAULT_DRAFT_TOKENS,
        // A bundle carries one model and no companions, so it can never be
        // a qwen_image model; these are never read.
        text_encoder: None,
        vision: None,
        vae: None,
        image_lora: ImageLora::Auto,
        image_lora_merge: true,
        vae_precision: crate::engine::image::vae::VaePrecision::default(),
        image_weights: crate::engine::image::transformer21::ImageWeights::default(),
        prompt_weights: crate::engine::prompt_weights::PromptWeights::default(),
        prefix_warmup: true,
        image_cache: crate::engine::image::stepcache::ImageCache::default(),
        image_reference: crate::engine::image::ReferenceCap::Source,
        image: crate::engine::image::ImageDefaults::default(),
        image_steps_set: false,
        image_cfg_scale_set: false,
        // A bundle carries no certificate; TLS is a per-deployment decision.
        tls: None,
        // A key baked into a distributed executable is a key everyone who has
        // the executable knows. `ORANGU_API_KEY` still applies at run time.
        api_key: std::env::var("ORANGU_API_KEY")
            .ok()
            .filter(|k| !k.is_empty()),
        // A bundle is built for a machine nobody will configure, so it takes
        // the ranking policy rather than an index that would only be right
        // on the box the bundle was built on.
        device: default_device(),
        device_split: default_device_split(),
        // A bundle runs on a machine nobody sized, so it takes rayon's own
        // choice rather than a count that was right where it was built.
        threads: None,
        // The bundle's own model is not a spec resolved against `models`, so
        // it is not this key — `main::prepare` reaches for it directly. Left
        // `None` so `--daemon` doesn't try to resolve a repo name against the
        // Hub for a model that is already in the file.
        model: None,
        delete: default_delete(),
        reexec: default_reexec(),
        role_key: Some(role),
        role,
        reasoning_effort: None,
        // A bundle is run by hand, so it says what it has to say where it
        // was started; a file log is a config-file decision like every other
        // deployment setting here.
        log: LogTarget::Console,
        mcp_servers: Vec::new(),
    }
}

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
            other => Err(anyhow!(
                "invalid role '{other}' (expected all, code, review, explorer, embedding, or image)"
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
        }
    }

    /// Whether this is the role a model's architecture decides — [`Role::
    /// Image`], which a `qwen_image` model always has and nothing else can
    /// have — as opposed to the five an operator picks between for a
    /// language model. A fixed role is never prompted for and never
    /// narrowed by an `x-orangu-role` header.
    pub fn fixed_by_model(&self) -> bool {
        matches!(self, Role::Image)
    }

    /// The role a model must be served in, from its `general.architecture`
    /// — `Some(Role::Image)` for a picture generator, `None` for a language
    /// model, whose role is the operator's choice.
    pub fn required_by(architecture: Option<&str>) -> Option<Role> {
        architecture
            .is_some_and(orangu::model_spec::is_image_architecture)
            .then_some(Role::Image)
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
            // orangu is a local, single-user AI, so generation defaults to one
            // slot. Concurrent decode is GPU/weight-bandwidth-bound — extra
            // slots don't raise throughput
            // (each token already streams the whole weight set), they only add
            // KV-cache memory. A multi-user deployment can still set `slots` in
            // the config file.
            //
            // **That reasoning is about one device.** On a machine with an
            // NPU the feed-forward moves off the GPU for decode, the two
            // slots stop queueing for the same weights, and a second slot
            // is worth 1.62x rather than 1.09x — see
            // `npu_tool::decode_enabled`, which switches the decode width
            // on precisely when `slots` is more than one. Still not the
            // default here: it is KV-cache memory spent on concurrency a
            // single user does not have.
            //
            // An image model draws one picture at a time (`engine::image::
            // Pipeline` holds a single busy lock), so a second slot would
            // only queue.
            Role::All | Role::Code | Role::Review | Role::Explorer | Role::Image => 1,
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
/// Only the Vulkan-family backends (Vulkan, Metal, DX12) have a GPU-side
/// mirror to store; on CPU, CUDA, OpenCL and ROCm this is inert.
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
    fn accepted() -> String {
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

    /// Parses one of the three names, or `None` for anything else.
    pub fn parse(raw: &str) -> Option<Self> {
        match raw.trim().to_lowercase().as_str() {
            "f16" => Some(KvCache::F16),
            "q8_0" | "q8" => Some(KvCache::Q8_0),
            "f32" => Some(KvCache::F32),
            _ => None,
        }
    }
}

/// Which `engine::backend::Backend` to run the forward pass on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BackendPreference {
    /// Tries every GPU backend compiled into this build, otherwise falls
    /// back to the CPU backend — no error either way. The order is
    /// Vulkan, CUDA, OpenCL, then — only if built with the `rocm` Cargo
    /// feature — ROCm, plus two platform-only entries: Metal ahead of
    /// everything on Apple targets (the only GPU API macOS ships) and DX12
    /// behind Vulkan on Windows. See `main.rs::select_backend`.
    #[default]
    Auto,
    Cpu,
    /// Fail to start (rather than silently falling back) if no Vulkan
    /// adapter is found — for when GPU inference was specifically asked
    /// for and silently running on the CPU instead would be surprising.
    Vulkan,
    /// Same fail-loudly contract as `Vulkan`, for an Apple Metal device.
    Metal,
    /// Same fail-loudly contract as `Vulkan`, for a Direct3D 12 device.
    /// The same `wgpu` engine and WGSL kernels as `Vulkan`, reached through
    /// `naga`'s HLSL output — for a Windows machine whose GPU has a D3D12
    /// driver but no Vulkan one.
    Dx12,
    /// Same fail-loudly contract as `Vulkan`, for an NVIDIA CUDA device.
    Cuda,
    /// Same fail-loudly contract as `Vulkan`, for an OpenCL device.
    OpenCl,
    /// Same fail-loudly contract as `Vulkan`, for an AMD ROCm/HIP device —
    /// also fails loudly if this binary wasn't compiled with the `rocm`
    /// Cargo feature.
    Rocm,
    /// The machine's NPU — **recognized but not yet runnable**.
    ///
    /// Accepted here so that asking for it is answered with an explanation
    /// of what is missing rather than `invalid value`, which would suggest
    /// a typo. `orangu::npu` detects the device and `orangu-server system`
    /// reports it, but no `engine::backend::Backend` runs on it: the vendor
    /// runtime executes whole graphs compiled ahead of time and exposes no
    /// per-operation entry point for `matmul` to call. See that module.
    ///
    /// Fails at startup like every other named backend, and for the same
    /// reason — someone who asked for the NPU should not silently get the
    /// CPU.
    Npu,
}

pub fn default_backend() -> BackendPreference {
    BackendPreference::Auto
}

/// The device the ranking policy picks — the highest-ranked hardware
/// device the selected backend reports. See `engine::backend::device`.
pub fn default_device() -> DeviceRequest {
    DeviceRequest::Auto
}

/// One device runs the whole model — what orangu did before it could do
/// anything else. See [`ServerConfiguration::device_split`] for why this,
/// and not `auto`, is the default.
pub fn default_device_split() -> SplitMode {
    SplitMode::Off
}

/// Whether the web console may load a different model — see
/// [`ServerConfiguration::reexec`]. On by default: the console is already
/// trusted with deleting models, and changing which one is served is the
/// less destructive of the two.
pub fn default_reexec() -> bool {
    true
}

/// Whether NPU precompilation runs at startup — see
/// [`Config::npu_precompile`].
/// [`Config::npu_cache_gb`].
///
/// 3 GiB: enough for every block of a model the size of Gemma 4 E4B (42 x
/// 55 MiB is 2.3 GiB) and a ceiling on anything larger, so a big model
/// degrades to a partial speedup instead of quietly claiming another ten
/// gigabytes of a shared-memory machine.
pub fn default_npu_cache_gb() -> f64 {
    /// The share of system memory compiled blocks may occupy.
    ///
    /// A quarter leaves the model's own weights, the KV cache and the rest
    /// of the machine three quarters, which held on every checkpoint in
    /// this project's sweep: an 8B `Q8_0` is 8.5 GiB of weights and 5.6 GiB
    /// of blocks on a 31 GiB machine.
    const SHARE: f64 = 0.25;
    /// Floors and ceilings the share, so a very small machine still gets a
    /// usable cache and a very large one does not reserve absurd amounts
    /// for a model that cannot use it.
    const RANGE: std::ops::RangeInclusive<f64> = 1.0..=16.0;

    static GB: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *GB.get_or_init(|| {
        let total = orangu::hardware::detect_cpu().total_memory_bytes as f64;
        let gb = total / (1u64 << 30) as f64 * SHARE;
        gb.clamp(*RANGE.start(), *RANGE.end())
    })
}

pub fn default_npu_precompile() -> bool {
    // **On**, and it took two fixes to get here. Blocks were being compiled
    // against a synthetic activation distribution whose tail is a ninth of
    // a real one's, which saturated the input quantizer and had them
    // returning output with no signal in it; they are now calibrated on the
    // model's own activations, captured from its first prompt. And each
    // activation tensor was held at one `uint8` scale that a handful of
    // outlier channels set for all of them; that range is now moved into
    // the weights, where a per-tensor scale can carry it
    // (`npu_ort::smoothing_scales`).
    //
    // Both models put through the whole path end to end, device off against
    // device on, a 720-token prompt, same history:
    //
    //   gemma 4 E2B Q8_0   4.44 -> 12.30 tok/s of prefill, correct answer
    //   gemma 4 E4B Q8_0   2.43 ->  7.00 tok/s of prefill, correct answer
    //
    // The output is not token-identical to the reference and will not be —
    // requantizing a matrix changes which token wins a close decision. What
    // it is now is an equally good answer rather than a worse one, which is
    // the bar this default is set against.
    //
    // A model that does not clear the accuracy gate is refused rather than
    // served badly (`npu_tool::MAX_BLOCK_ERROR`), and the
    // first prompt of a fresh model pays for the capture — measured at 4.49
    // tok/s against 4.44 with the device off, which is inside the noise.
    true
}

/// Whether the web console may delete models — see
/// [`ServerConfiguration::delete`]. On by default.
pub fn default_delete() -> bool {
    true
}

/// Parses a `yes`/`no`/`true`/`false`/`on`/`off`/`1`/`0` config value.
/// Every spelling a person might reasonably write for a switch, rather than
/// only the two Rust's own `bool` parser accepts — this is a hand-edited
/// `.ini`, not a serialized struct.
fn parse_bool(section: &str, key: &str, value: &str) -> Result<bool> {
    match value.trim().to_lowercase().as_str() {
        "yes" | "true" | "on" | "1" => Ok(true),
        "no" | "false" | "off" | "0" => Ok(false),
        other => Err(anyhow!(
            "invalid value for [{section}].{key}: '{other}' \
             (expected yes/no, true/false, on/off, or 1/0)"
        )),
    }
}

#[derive(Clone, Debug)]
pub struct ServerConfiguration {
    /// Directory a model spec is resolved against (and downloaded into, if
    /// it names a Hugging Face repo not already cached there).
    pub models: PathBuf,
    pub host: String,
    pub port: u16,
    /// Number of concurrent request slots (each with its own KV cache) the
    /// continuous-batching scheduler serves at once.
    pub slots: usize,
    /// `[web].port`: the port the web console listens on, bound alongside
    /// (not instead of) the API's own `port`. `0` — no `[web]` section, and
    /// no legacy `[orangu-server].web` either — disables it, and no second
    /// listener is bound.
    pub web: u16,
    /// `[web].host`: the address the web console binds, when it should
    /// differ from the API's. Defaults to [`host`](Self::host), so the two
    /// are reachable in the same places unless deliberately separated —
    /// which is the point of being able to set it: an API on `all` for the
    /// machines that consume it, with the console kept on `127.0.0.1`.
    pub web_host: String,
    /// Whether `[web].host` was set *explicitly*, as opposed to
    /// [`web_host`](Self::web_host) having fallen back to
    /// [`host`](Self::host).
    ///
    /// Only `--host` needs the distinction, and it needs it badly: that flag
    /// moves the console along with the API, since the two share an address
    /// unless something says otherwise — but a config that deliberately
    /// separated them (an API on the network, the console kept on loopback)
    /// must not have the console quietly dragged onto `0.0.0.0` by a flag
    /// aimed at the API. An explicit key stands; an inherited one follows.
    pub web_host_explicit: bool,
    /// `[prometheus].port`: the port a dedicated `/metrics` listener binds
    /// to, alongside (not instead of) the API's own `port`. `0` — the
    /// default, and what having no `[prometheus]` section resolves to —
    /// disables it entirely. See `http::native::render_metrics_text` for the
    /// response it serves.
    pub metrics: u16,
    /// `[prometheus].host`: the address the metrics listener binds, when it
    /// should differ from the API's. Defaults to [`host`](Self::host); same
    /// idea as [`web_host`](Self::web_host). Worth separating because the
    /// listener carries no API key check: an API on `all`, metrics on a
    /// loopback or private scrape address.
    pub metrics_host: String,
    /// Whether `[prometheus].host` was set *explicitly*; see
    /// [`web_host_explicit`](Self::web_host_explicit) for why `--host` needs
    /// the distinction.
    pub metrics_host_explicit: bool,
    /// Which `Backend` runs the forward pass — CPU, a named GPU API, or
    /// (the default) whichever GPU this platform finds first, falling back
    /// to CPU.
    pub backend: BackendPreference,
    /// `[orangu-server].mlp_unroll` — whether the GPU's block-unroll decode
    /// kernels are used.
    ///
    /// **`None` means decide at runtime**, which is the default and what
    /// almost everyone should leave it at: the backend cross-checks each
    /// quantized type against the CPU at startup and drops the tuned kernels
    /// for any type this device computes wrong (see
    /// `engine::backend::vulkan::VulkanBackend::disable_miscompiled_decode_kernels`).
    /// `Some(v)` overrides that decision for the unroll family and is
    /// obeyed — an operator who has measured their own hardware outranks a
    /// probe.
    pub mlp_unroll: Option<bool>,
    /// `[orangu-server].npu_precompile` — whether this model may use the
    /// NPU at all.
    ///
    /// On by default, and on a machine without an NPU it costs nothing to
    /// leave on: the check is a probe that fails immediately. Off is the
    /// operator's off switch for the device — no blocks are compiled *and*
    /// none are bound, including blocks a previous run already cached.
    ///
    /// The name says `precompile` and the switch does more than that, which
    /// is deliberate: an operator turning this off wants the device out of
    /// the picture, and a build that kept serving from a warm cache after
    /// they turned it off would be answering a question they did not ask.
    /// To keep the device but stop it growing the cache, set
    /// [`Config::npu_cache_gb`] to 0 instead.
    pub npu_precompile: bool,
    /// `[orangu-server].prefill_backend` — `auto` (the default: one
    /// prompt-shaped GEMM timed on each backend at load, the faster takes
    /// the prompts, so every machine decides for itself), `device` (prompts
    /// run where decode runs) or `cpu` (a multi-token pass runs on the CPU
    /// backend while single-token decode stays on the selected device). For
    /// a board whose device does a prompt's GEMMs slower than its cores do
    /// (the CIX P1's Mali: 1.6 against 9.8 tok/s on the 27B ternary model,
    /// 26 against 118 on `gemma-4-E2B`) and whose memory the two share.
    /// Honoured by the Gemma trunk (dense `gemma*`) and the Qwen 3.5-family
    /// trunk (`qwen35`, `qwen35moe`, `qwen3next`, `qwen4exp`);
    /// `ORANGU_HYBRID_PREFILL_CPU=1` is `cpu` from the environment.
    pub prefill_backend: PrefillBackend,
    /// `[orangu-server].npu_cache_gb` — how much compiled-block cache the
    /// NPU precompile may spend on one model.
    ///
    /// Every block compiled is that block's weights again in `uint8`
    /// alongside the model's own, about 55 MiB each for Gemma 4 E4B, and all
    /// of it is resident once bound. So this is a real memory budget and not
    /// a disk quota. A model that does not fit gets its first N layers on
    /// the device and the rest on the CPU or GPU, which is a partial
    /// speedup: 12 of 42 blocks measured 18.5 tok/s of prefill against 15.0
    /// with none, and all 42 measured 40.9.
    ///
    /// **Scaled to the machine, not a constant.** It was a flat 3 GiB, and
    /// that number quietly halved every model above about 4B: a sweep of 33
    /// checkpoints on a 31 GiB machine left 20 of granite 8B's 40 blocks
    /// off the device, 14 of 32 on Meta-Llama 3.1 8B, 14 of 32 on Mistral
    /// 7B, 15 of 36 on Qwen3 8B. Raising it so the whole model fits is
    /// worth more than anything else measured here — Meta-Llama 3.1 8B
    /// `Q8_0`, a 980-token prompt:
    ///
    /// | blocks on the device | prefill |
    /// |---|---|
    /// | 18 of 32 (3 GiB) | 2.52 tok/s |
    /// | 32 of 32 (8 GiB) | **5.36 tok/s** |
    ///
    /// A machine's memory is the thing that decides, so the default reads
    /// it rather than guessing. An operator who wants a different figure
    /// still sets `npu_cache_gb` and is obeyed.
    ///
    /// `0` disables precompiling without turning off
    /// [`Config::npu_precompile`], so blocks already cached still load.
    pub npu_cache_gb: f64,
    /// How the GPU-side KV mirror is stored — see [`KvCache`]. Overridden by
    /// `ORANGU_KV_CACHE`, so a sweep can vary it without editing the file.
    pub kv_cache: KvCache,
    /// `[orangu-server].read_size`: the granule an explicit read of a
    /// model file uses, in KiB.
    ///
    /// A span smaller than this is widened outward to it and the wanted bytes
    /// taken from the middle; a larger one is read in chunks of it. Only the
    /// explicit routes use it (`engine::expert_read`'s `pread`/`direct`) —
    /// the default `mmap` route leaves request size to the kernel's
    /// readahead.
    ///
    /// A key rather than a constant because the number that matters is the
    /// device's, not the format's: the request size at which a drive stops
    /// paying one round trip per read varies with the controller, the bus and
    /// the bridge in front of it. See [`DEFAULT_READ_SIZE`] for the
    /// measurement the default comes from.
    pub read_size: usize,
    /// `[orangu-server].tls_cert` / `tls_key`: PEM paths for serving HTTPS.
    ///
    /// Both or neither — one alone is a configuration error rather than a
    /// half-enabled server, because the failure it would otherwise produce is
    /// serving in the clear while the operator believes otherwise.
    pub tls: Option<(PathBuf, PathBuf)>,
    /// `[orangu-server].api_key`: the bearer token every request must carry.
    ///
    /// `None` — the default — leaves the server open, which is the behaviour
    /// before this key existed and is right for the loopback bind it also
    /// defaults to. It becomes load-bearing the moment `host` is widened.
    ///
    /// Overridden by `ORANGU_API_KEY`, which is the spelling a deployment
    /// wants: a secret in a config file is a secret on disk and in every
    /// backup of it.
    pub api_key: Option<String>,
    /// `[orangu-server].queue_limit`: how many requests may wait for a slot
    /// before the server starts refusing with `503`. `0` — the default —
    /// queues without bound, which is the behaviour before this key existed.
    pub queue_limit: usize,
    /// `[orangu-server].context`: the context, in tokens, one request must
    /// be able to hold on the device. Unset, the model's layers all go to
    /// the card when they fit and the context is whatever the card has
    /// left; set, layers move to the host until the card has room for this
    /// many tokens of KV cache beside the weights it keeps — the same
    /// trade the reference engines make to serve a long context on a small
    /// card, and slower per token for it, which is why it is a choice.
    pub context: Option<usize>,
    /// `[orangu-server].draft_model`: a second, smaller model whose guesses
    /// the served model verifies — speculative decoding. A model spec, the
    /// same shape as [`model`](Self::model).
    ///
    /// `None` (the default) decodes exactly as before. The pair must share a
    /// vocabulary, which is checked at startup rather than discovered as
    /// nonsense output.
    pub draft_model: Option<String>,
    /// `[orangu-server].text_encoder`: the `qwen2vl` GGUF a `qwen_image`
    /// model encodes prompts with — a model spec or path, the same shape as
    /// [`model`](Self::model). `None` (the default) finds the largest such
    /// file in the models directory. Ignored for every other architecture.
    pub text_encoder: Option<String>,
    /// `[orangu-server].vision`: the text encoder's vision projector (an
    /// `mmproj-*.gguf`) a `qwen_image_2_1` model reads a reference picture
    /// with — a path, absolute or relative to the models directory, or
    /// `none` to draw attached pictures over instead of editing them.
    /// `None` finds one beside the text encoder, or under `models`.
    pub vision: Option<String>,
    /// `[orangu-server].vae`: the Qwen-Image VAE (`.safetensors`) a
    /// `qwen_image` model decodes pictures with, absolute or relative to
    /// the models directory. `None` finds it by its tensors.
    pub vae: Option<String>,
    /// `[orangu-server].image_lora`: the low-rank adapter (`.safetensors`)
    /// applied to the picture transformer's linears — see [`ImageLora`].
    pub image_lora: ImageLora,
    /// `[orangu-server].image_lora_merge`: whether the adapter is folded
    /// into the weights at startup (the default — a minute or two once,
    /// then no cost per pass) or applied in `f32` on every pass (`no`:
    /// exact, 13–28% of a pass on the CPU).
    pub image_lora_merge: bool,
    /// `[orangu-server].vae_precision`: `int8` (the default — the VAE's
    /// convolutions as `Q6_K` weights against `int8` activations on the
    /// transformer's kernel, about three times faster on an `i8mm` core) or
    /// `f32` (as the file has them; exact).
    pub vae_precision: crate::engine::image::vae::VaePrecision,
    /// `[orangu-server].image_weights`: how a Qwen-Image 2.1 transformer's
    /// linears are held — `auto` (the default: per-row `int8` for the 8 × 8
    /// `smmla` tile when total memory is at least three times the 7 GB
    /// copy), `int8`, or `file` (the file's K-quants, no copy).
    pub image_weights: crate::engine::image::transformer21::ImageWeights,
    /// `[orangu-server].prompt_weights`: how the weights a prompt multiplies
    /// on the CPU are held — `auto` (the default: copies for the cores'
    /// matrix instructions — per-row `int8` for the K-quant projections,
    /// `bf16` for the unquantized ones — each kind when prompts run on the
    /// CPU, the CPU has its instruction, the copies take at most half the
    /// available memory, and the kind measures at least 10% faster at
    /// load), `copy` (the copies unmeasured), or `file` (the file's weights,
    /// no copy). See `engine::prompt_weights`.
    pub prompt_weights: crate::engine::prompt_weights::PromptWeights,
    /// `[orangu-server].prefix_warmup` — `on` (the default): remember the
    /// prompt prefixes requests reuse from the cache, per model, and prefill
    /// them again in the background after a restart, so a client's fixed
    /// opening (its system prompt and tools) is cached before its first
    /// request. `ORANGU_PREFIX_WARMUP` overrides it. See
    /// `engine::warm_prefixes`.
    pub prefix_warmup: bool,
    /// `[orangu-server].image_cache`: `easy` (the default) or
    /// `easy:<threshold>` (EasyCache — a step whose predicted change is
    /// small reuses the last passes' residual), or `off` (every step runs
    /// the transformer).
    pub image_cache: crate::engine::image::stepcache::ImageCache,
    /// `[orangu-server].image_reference_size`: `source` (the default: an
    /// edit reads its reference at the picture's area but no larger than
    /// the attached picture itself), `output` (the picture's area) or
    /// `WIDTHxHEIGHT`, the
    /// largest area it is read at — a smaller reference is a shorter
    /// prefix and fewer keys in every step.
    pub image_reference: crate::engine::image::ReferenceCap,
    /// `[orangu-server].image_*`: what a picture request gets when it does
    /// not say — size, steps, guidance, the negative prompt, and how far
    /// from an attached picture to start. The web console's chat sends none
    /// of these, so for it these *are* the settings.
    pub image: crate::engine::image::ImageDefaults,
    /// Whether `image_steps` and `image_cfg_scale` were written in the
    /// config. Under a step-distilled adapter the ones left out follow the
    /// adapter — its step count, guidance off — rather than Qwen-Image's
    /// release settings, which the adapter was made to replace; see
    /// [`ServerConfiguration::image_defaults_under`].
    pub image_steps_set: bool,
    pub image_cfg_scale_set: bool,
    /// `[orangu-server].draft_tokens`: how many tokens the draft model
    /// proposes per verification (default 4).
    ///
    /// The trade is direct: more drafted tokens means more decoded per
    /// verification when the pair agrees, and more wasted draft forwards when
    /// it does not. Overridden for one run by `ORANGU_SPEC_DRAFT`.
    pub draft_tokens: usize,
    /// *Which device* within [`backend`](Self::backend) — an enumeration
    /// index, a substring of the device's name, or (the default) the
    /// ranking policy in `engine::backend::device`.
    ///
    /// Separate from `backend` because they answer different questions and
    /// a machine can need both: `backend` picks the API, this picks the
    /// card. Overridden at run time by `ORANGU_DEVICE`.
    pub device: DeviceRequest,
    /// Whether to spread one model's layers across the selected devices,
    /// and how — `off` (the default), `auto`, `all`, or explicit
    /// proportions. See `engine::placement`.
    ///
    /// Off by default because a split model gives up every fused
    /// GPU-resident path in this engine (see `engine::backend::multi`), so
    /// it buys capacity at a real cost in speed. Overridden at run time by
    /// `ORANGU_DEVICE_SPLIT`.
    pub device_split: SplitMode,
    /// How many worker threads every CPU path in this process shares —
    /// `CpuBackend`'s matmul, the MoE expert loop, the per-expert fan-out.
    ///
    /// `None` (the default) leaves `rayon`'s own choice of one worker per
    /// logical core. Overridden at run time by `ORANGU_THREADS`.
    pub threads: Option<usize>,
    /// A model spec (local path, `NR`/`MODEL` label, or `<user>/<model>
    /// [:quant]` Hugging Face repo) — the same shape as the CLI's
    /// positional `model` argument. Only consulted in `--daemon` mode,
    /// where there is no attached terminal to pass a CLI argument to or
    /// prompt on interactively; ignored otherwise.
    pub model: Option<String>,
    /// `[web].delete`: whether the web console's model manager may delete
    /// models. `true` (the default) lets it; `false` removes the Delete
    /// button from every row — not merely disables it, since unlike the
    /// other switches there is nothing conditional about it to explain — and
    /// makes the endpoint behind it refuse.
    ///
    /// Worth its own key rather than riding on `reexec`: deleting a model is
    /// the one irreversible thing the console can do, and a deployment may
    /// well want to allow a model switch while keeping the models directory
    /// read-only.
    ///
    /// Models only — it says nothing about chat sessions. History's own
    /// delete controls are unconditional: a session is the console's own
    /// scratch data, not a file on disk something else put there.
    pub delete: bool,
    /// `[web].reexec`: whether the web console's model manager may load a
    /// different model into this server. `true` (the default) lets it;
    /// `false` disables the panel's Load button and makes the endpoint
    /// behind it refuse.
    ///
    /// Loading a model re-executes this process (see `main::reexec`), which
    /// is exactly what makes it worth a switch: a deployment behind a
    /// supervisor, or one where a specific model is the point of the
    /// process, wants the server it started to stay the server it started.
    pub reexec: bool,
    /// The config file's own `role` key, parsed, whatever mode this is —
    /// as opposed to [`role`](Self::role), which is the *resolved* role and
    /// still ignores this outside `--daemon`. Kept apart so the interactive
    /// startup prompt can pre-select what the config names without that
    /// silently becoming the role of a run that never reaches the prompt.
    pub role_key: Option<Role>,
    /// The resolved [`Role`] — whichever CLI flag (`--all`/`--code`/
    /// `--review`/`--explorer`/`--embedding`) was passed to
    /// [`load_server_configuration`]; or, in `--daemon` mode only (same
    /// reasoning as `model`: no attached terminal to pass a CLI flag to),
    /// the config file's own `role` key; or, failing both, [`Role::All`].
    pub role: Role,
    /// `[orangu-server].reasoning_effort`: how hard a reasoning model is
    /// asked to think, passed straight into the chat template as the
    /// same-named variable and otherwise left undefined.
    ///
    /// Undefined is not the same as "normal". A template that reads this
    /// picks its own default when the variable is absent, and Qwen3.x's
    /// picks the *most* expensive one — `reasoning_effort|default('xhigh')`,
    /// which prepends "Reasoning effort is set to xhigh. Please think
    /// carefully through the task, validate key assumptions, consider
    /// plausible alternatives…" as a system message. Measured on
    /// `Qwen3.8-27B`, that is not a small tax: "implement a doubly linked
    /// list in C" spent over 8000 tokens inside `<think>`, drafting the
    /// program three times before writing any of the answer.
    ///
    /// Left `None` by default all the same, because the levels are the
    /// template's vocabulary rather than this server's — Qwen3.x accepts
    /// `xhigh`/`medium`/`low` and raises on anything else, others spell
    /// theirs `high`/`medium`/`low` — and imposing one model family's word
    /// on every other template would turn an unset knob into a 400.
    pub reasoning_effort: Option<String>,
    /// `[orangu-server].log_type` / `log_path`: where this server's output
    /// goes — the banner, every request's completion line, and every note
    /// or warning it produces while serving. The console by default, exactly
    /// as before the keys existed; `file` appends all of it to `log_path`
    /// instead, and drops the once-a-second progress a request writes to a
    /// terminal, which a file has no use for. See `orangu::logging`.
    ///
    /// What a `--daemon` run needs: detached, its stdout is `/dev/null`, so
    /// without this a daemon serves in silence.
    pub log: LogTarget,
    /// Read-only HTTP MCP profiles exposed by the web console. Changing this
    /// list requires restarting `orangu-server`.
    pub mcp_servers: Vec<McpConfiguration>,
}

#[derive(Clone, Debug)]
pub struct McpConfiguration {
    pub name: String,
    pub endpoint: String,
    pub enabled: bool,
    pub approval_mode: String,
}

/// Expands a leading `~` or `~/` to the user's home directory — a config
/// value is otherwise taken literally, but a models directory is the one
/// place a user is likely to type a `~`-relative path, same as a shell
/// would accept.
fn expand_tilde(path: &str) -> PathBuf {
    match path.strip_prefix('~') {
        Some(rest) => match home::home_dir() {
            Some(home) => home.join(rest.trim_start_matches('/')),
            None => PathBuf::from(path),
        },
        None => PathBuf::from(path),
    }
}

pub fn default_server_config_path() -> Option<PathBuf> {
    let cwd_path = std::env::current_dir().ok()?.join("orangu-server.conf");
    if cwd_path.exists() {
        return Some(cwd_path);
    }

    let config_path = home::home_dir()?.join(".orangu/orangu-server.conf");
    config_path.exists().then_some(config_path)
}

/// `cli_role` is whichever of `--all`/`--code`/`--review`/`--explorer`/
/// `--embedding` was passed on the command line, already resolved by the
/// caller — `Some` only when a flag was actually given, so this can tell
/// "explicitly `--all`" apart from "no role flag at all". `daemon` gates
/// whether the config file's own `role` key is even consulted as a
/// fallback for the latter case — same reasoning as the `model` key: in
/// an attached run, a missing CLI flag just means `Role::All`, exactly
/// like before this key existed; only `--daemon` (no attached terminal to
/// pass a flag to) falls back to the config.
pub fn load_server_configuration(
    path: &Path,
    cli_role: Option<Role>,
    daemon: bool,
) -> Result<ServerConfiguration> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read configuration {}", path.display()))?;
    let mut sections = parse_ini_sections(&contents)
        .with_context(|| format!("failed to parse configuration {}", path.display()))?;

    let section = sections.remove(SERVER_SECTION).unwrap_or_default();

    let models = section
        .get("models")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow!("[{SERVER_SECTION}].models must be set to a models directory"))?;

    let host = section
        .get("host")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(default_host);

    let port = match section.get("port") {
        Some(value) => value
            .trim()
            .parse::<u16>()
            .map_err(|err| anyhow!("invalid value for [{SERVER_SECTION}].port: {err}"))?,
        None => default_port(),
    };

    // Parsed unconditionally, so a bad value is an error in every mode
    // rather than only under `--daemon` — and so the interactive prompt has
    // something to pre-select.
    let role_key = match section.get("role") {
        Some(value) => Some(
            Role::parse(value)
                .map_err(|err| anyhow!("invalid value for [{SERVER_SECTION}].role: {err}"))?,
        ),
        None => None,
    };

    let role = match cli_role {
        Some(role) => role,
        None if daemon => role_key.unwrap_or_default(),
        None => Role::default(),
    };

    let slots = match section.get("slots") {
        Some(value) => {
            let slots = value
                .trim()
                .parse::<usize>()
                .map_err(|err| anyhow!("invalid value for [{SERVER_SECTION}].slots: {err}"))?;
            if slots == 0 {
                return Err(anyhow!("[{SERVER_SECTION}].slots must be at least 1"));
            }
            slots
        }
        None => role.default_slots(),
    };

    // The web console lives in its own `[web]` section, and *having* one is
    // what turns the console on. `[orangu-server].web` is the spelling that
    // shipped before that section existed and is still honored — a config
    // written against it goes on working untouched — but only when there is
    // no `[web]` section to take precedence over it.
    let web_section = sections.remove(WEB_SECTION);
    let (web, web_host, web_host_explicit, reexec, delete) = match web_section {
        Some(web_section) => {
            let port = match web_section.get("port") {
                Some(value) => value
                    .trim()
                    .parse::<u16>()
                    .map_err(|err| anyhow!("invalid value for [{WEB_SECTION}].port: {err}"))?,
                None => default_web_port(),
            };
            // Only worth spelling out when the console should be reachable
            // somewhere the API isn't; unset, the two share an address.
            let explicit = web_section
                .get("host")
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty());
            let web_host = explicit.clone().unwrap_or_else(|| host.clone());
            let reexec = match web_section.get("reexec") {
                Some(value) => parse_bool(WEB_SECTION, "reexec", value)?,
                None => default_reexec(),
            };
            let delete = match web_section.get("delete") {
                Some(value) => parse_bool(WEB_SECTION, "delete", value)?,
                None => default_delete(),
            };
            (port, web_host, explicit.is_some(), reexec, delete)
        }
        None => {
            let port = match section.get("web") {
                Some(value) => value
                    .trim()
                    .parse::<u16>()
                    .map_err(|err| anyhow!("invalid value for [{SERVER_SECTION}].web: {err}"))?,
                None => default_web(),
            };
            // The pre-section spelling has no `[web].host` to be explicit
            // with, so the console has always followed the API's address.
            (
                port,
                host.clone(),
                false,
                default_reexec(),
                default_delete(),
            )
        }
    };

    // The dedicated Prometheus listener lives in its own `[prometheus]`
    // section, the same shape as `[web]` above: *having* one is what enables
    // it.
    let (metrics, metrics_host, metrics_host_explicit) = match sections.remove(PROMETHEUS_SECTION) {
        Some(prometheus_section) => {
            let port = match prometheus_section.get("port") {
                Some(value) => value.trim().parse::<u16>().map_err(|err| {
                    anyhow!("invalid value for [{PROMETHEUS_SECTION}].port: {err}")
                })?,
                None => default_prometheus_port(),
            };
            let explicit = prometheus_section
                .get("host")
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty());
            let metrics_host = explicit.clone().unwrap_or_else(|| host.clone());
            (port, metrics_host, explicit.is_some())
        }
        None => (default_metrics(), host.clone(), false),
    };

    let model = section
        .get("model")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    let tls = match (section.get("tls_cert"), section.get("tls_key")) {
        (Some(cert), Some(key)) => Some((expand_tilde(cert), expand_tilde(key))),
        (None, None) => None,
        (Some(_), None) => {
            return Err(anyhow!(
                "[{SERVER_SECTION}].tls_cert is set but tls_key is not — both are needed, \
                 and starting without TLS here would serve in the clear while looking configured"
            ));
        }
        (None, Some(_)) => {
            return Err(anyhow!(
                "[{SERVER_SECTION}].tls_key is set but tls_cert is not — both are needed, \
                 and starting without TLS here would serve in the clear while looking configured"
            ));
        }
    };

    // The environment wins, so a key never has to be written down.
    let api_key = std::env::var("ORANGU_API_KEY")
        .ok()
        .or_else(|| section.get("api_key").cloned())
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty());

    let queue_limit = match section.get("queue_limit") {
        Some(value) => value
            .trim()
            .parse::<usize>()
            .map_err(|err| anyhow!("invalid value for [{SERVER_SECTION}].queue_limit: {err}"))?,
        None => 0,
    };

    let context = match section.get("context") {
        Some(value) => {
            let context = value
                .trim()
                .parse::<usize>()
                .map_err(|err| anyhow!("invalid value for [{SERVER_SECTION}].context: {err}"))?;
            if context == 0 {
                return Err(anyhow!("[{SERVER_SECTION}].context must be at least 1"));
            }
            Some(context)
        }
        None => None,
    };

    let draft_model = section
        .get("draft_model")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    let draft_tokens = match section.get("draft_tokens") {
        Some(value) => {
            let tokens = value.trim().parse::<usize>().map_err(|err| {
                anyhow!("invalid value for [{SERVER_SECTION}].draft_tokens: {err}")
            })?;
            if tokens == 0 {
                // `0` would be a draft of nothing verified by a forward pass
                // that could have decoded a token on its own — strictly worse
                // than not speculating, and not what anyone means by it.
                return Err(anyhow!(
                    "invalid value for [{SERVER_SECTION}].draft_tokens: 0 (leave draft_model \
                     out to turn speculation off)"
                ));
            }
            tokens
        }
        None => DEFAULT_DRAFT_TOKENS,
    };

    let text_encoder = section
        .get("text_encoder")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let vision = section
        .get("vision")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let vae = section
        .get("vae")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let image_lora = match section.get("image_lora").map(|value| value.trim()) {
        None | Some("") | Some("auto") => ImageLora::Auto,
        Some("none") => ImageLora::None,
        Some(spec) => ImageLora::Spec(spec.to_string()),
    };
    let image_steps_set = section.contains_key("image_steps");
    let image_cfg_scale_set = section.contains_key("image_cfg_scale");
    let image_lora_merge = match section.get("image_lora_merge") {
        Some(value) => parse_bool(SERVER_SECTION, "image_lora_merge", value)?,
        None => true,
    };
    let vae_precision = match section.get("vae_precision") {
        Some(value) => crate::engine::image::vae::VaePrecision::parse(value).ok_or_else(|| {
            anyhow!(
                "invalid value for [{SERVER_SECTION}].vae_precision: '{}' (expected int8 or f32)",
                value.trim()
            )
        })?,
        None => crate::engine::image::vae::VaePrecision::default(),
    };
    let image_weights = match section.get("image_weights") {
        Some(value) => {
            crate::engine::image::transformer21::ImageWeights::parse(value).ok_or_else(|| {
                anyhow!(
                    "invalid value for [{SERVER_SECTION}].image_weights: '{}' (expected auto, \
                     int8 or file)",
                    value.trim()
                )
            })?
        }
        None => crate::engine::image::transformer21::ImageWeights::default(),
    };
    let prompt_weights = match section.get("prompt_weights") {
        Some(value) => {
            crate::engine::prompt_weights::PromptWeights::parse(value).ok_or_else(|| {
                anyhow!(
                    "invalid value for [{SERVER_SECTION}].prompt_weights: '{}' (expected auto, \
                 copy or file)",
                    value.trim()
                )
            })?
        }
        None => crate::engine::prompt_weights::PromptWeights::default(),
    };
    let prefix_warmup = match section.get("prefix_warmup") {
        Some(value) => parse_bool(SERVER_SECTION, "prefix_warmup", value)?,
        None => true,
    };
    let image_cache = match section.get("image_cache") {
        Some(value) => crate::engine::image::stepcache::ImageCache::parse(value)
            .map_err(|e| anyhow!("invalid value for [{SERVER_SECTION}].image_cache: {e}"))?,
        None => crate::engine::image::stepcache::ImageCache::default(),
    };
    let image_reference = match section.get("image_reference_size") {
        None => crate::engine::image::ReferenceCap::Source,
        Some(value) => crate::engine::image::ReferenceCap::parse(value).ok_or_else(|| {
            anyhow!(
                "invalid value for [{SERVER_SECTION}].image_reference_size: '{}' (expected \
                 output, source or WIDTHxHEIGHT)",
                value.trim()
            )
        })?,
    };
    let image = parse_image_defaults(&section)?;

    let kv_cache = match section.get("kv_cache") {
        Some(value) => KvCache::parse(value).ok_or_else(|| {
            anyhow!(
                "invalid value for [{SERVER_SECTION}].kv_cache: '{}' (expected {})",
                value.trim(),
                KvCache::accepted()
            )
        })?,
        None => KvCache::default(),
    };

    // Rejected rather than clamped, and page-aligned rather than rounded: a
    // read granule that is not a multiple of the page size cannot satisfy
    // `O_DIRECT`, and silently rounding a stated number would make a sweep's
    // arms differ from the values it thinks it set.
    let read_size = match section.get("read_size") {
        Some(value) => {
            let kib = value
                .trim()
                .parse::<usize>()
                .map_err(|err| anyhow!("invalid value for [{SERVER_SECTION}].read_size: {err}"))?;
            if kib == 0 {
                return Err(anyhow!(
                    "[{SERVER_SECTION}].read_size must be at least 4 (one page)"
                ));
            }
            if !kib.is_multiple_of(4) {
                return Err(anyhow!(
                    "[{SERVER_SECTION}].read_size must be a multiple of 4 (the page size),                      got {kib}"
                ));
            }
            kib
        }
        None => DEFAULT_READ_SIZE,
    };

    // Trimmed, and an empty value read as unset: a template that validates
    // its levels would raise on "" exactly as it would on a typo, and
    // `reasoning_effort =` with nothing after it plainly means "no opinion".
    let reasoning_effort = section
        .get("reasoning_effort")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let mlp_unroll = match section.get("mlp_unroll") {
        Some(value) => Some(parse_bool(SERVER_SECTION, "mlp_unroll", value)?),
        None => None,
    };
    let npu_cache_gb = match section.get("npu_cache_gb") {
        Some(value) => value.parse::<f64>().ok().filter(|v| *v >= 0.0).ok_or_else(|| {
            anyhow::anyhow!(
                "[{SERVER_SECTION}].npu_cache_gb must be a non-negative number of GiB, got '{value}'"
            )
        })?,
        None => default_npu_cache_gb(),
    };
    let npu_precompile = match section.get("npu_precompile") {
        Some(value) => parse_bool(SERVER_SECTION, "npu_precompile", value)?,
        None => default_npu_precompile(),
    };
    let prefill_backend = match section.get("prefill_backend") {
        Some(value) => match value.trim().to_lowercase().as_str() {
            "device" => PrefillBackend::Device,
            "cpu" => PrefillBackend::Cpu,
            "auto" => PrefillBackend::Auto,
            other => {
                return Err(anyhow!(
                    "invalid value for [{SERVER_SECTION}].prefill_backend: {other:?} (expected device, cpu or auto)"
                ));
            }
        },
        None => PrefillBackend::Auto,
    };

    let backend = match section.get("backend") {
        Some(value) => match value.trim().to_lowercase().as_str() {
            "auto" => BackendPreference::Auto,
            "cpu" => BackendPreference::Cpu,
            "vulkan" => BackendPreference::Vulkan,
            "metal" => BackendPreference::Metal,
            "dx12" => BackendPreference::Dx12,
            "cuda" => BackendPreference::Cuda,
            "opencl" => BackendPreference::OpenCl,
            "rocm" => BackendPreference::Rocm,
            "npu" => BackendPreference::Npu,
            other => {
                return Err(anyhow!(
                    "invalid value for [{SERVER_SECTION}].backend: '{other}' \
                     (expected auto, cpu, vulkan, metal, dx12, cuda, opencl, rocm, or npu)"
                ));
            }
        },
        None => default_backend(),
    };

    // No validation here, deliberately. Whether `device = 2` or
    // `device = navi` names anything is a question only the enumerated
    // device list can answer, and that list doesn't exist until a backend
    // has been chosen and its driver asked — so the error belongs at
    // startup, where it can print the devices that *do* exist alongside it,
    // not here, where it could only say "that isn't a number".
    let device = section
        .get("device")
        .map(|value| DeviceRequest::parse(value))
        .unwrap_or_else(default_device);

    // Unlike `device`, this one *is* validated here: `auto`/`all`/`off` and
    // a ratio list are decidable without asking a driver anything, so a
    // typo should stop the server at the config file rather than silently
    // becoming "off" and quietly running the shape nobody asked for.
    let threads = match section.get("threads") {
        Some(value) => {
            let threads = value
                .trim()
                .parse::<usize>()
                .map_err(|err| anyhow!("invalid value for [{SERVER_SECTION}].threads: {err}"))?;
            if threads == 0 {
                return Err(anyhow!(
                    "invalid value for [{SERVER_SECTION}].threads: 0 (leave the key out for \
                     one worker per logical core)"
                ));
            }
            Some(threads)
        }
        None => None,
    };

    let device_split = match section.get("device_split") {
        Some(value) => SplitMode::parse(value).map_err(|err| {
            anyhow!("invalid value for [{SERVER_SECTION}].device_split: {err} (expected off, auto, all, or a list such as 3,1)")
        })?,
        None => default_device_split(),
    };

    let log = LogTarget::from_keys(
        SERVER_SECTION,
        section.get("log_type").map(String::as_str),
        section.get("log_path").map(String::as_str),
    )?;

    let mut mcp_servers = sections
        .into_iter()
        .map(|(name, values)| parse_mcp_configuration(name, values))
        .collect::<Result<Vec<_>>>()?;
    mcp_servers.sort_by(|left, right| left.name.cmp(&right.name));

    Ok(ServerConfiguration {
        models: expand_tilde(&models),
        host,
        port,
        metrics,
        metrics_host,
        metrics_host_explicit,
        model,
        role_key,
        role,
        reasoning_effort,
        slots,
        web,
        web_host,
        web_host_explicit,
        backend,
        mlp_unroll,
        npu_cache_gb,
        npu_precompile,
        prefill_backend,
        kv_cache,
        read_size,
        queue_limit,
        context,
        draft_model,
        draft_tokens,
        text_encoder,
        vision,
        vae,
        image_lora,
        image_lora_merge,
        vae_precision,
        image_weights,
        prompt_weights,
        prefix_warmup,
        image_cache,
        image_reference,
        image,
        image_steps_set,
        image_cfg_scale_set,
        api_key,
        tls,
        device,
        device_split,
        threads,
        reexec,
        delete,
        log,
        mcp_servers,
    })
}

/// `[orangu-server].image_lora`. Qwen-Image's own settings — fifty guided
/// steps — are hours on a CPU, and its publishers' Lightning distillation
/// (an adapter that makes a picture in eight unguided steps) is fetched
/// with the model; so absent, or `auto`, the adapter found under the
/// models directory is served, `none` runs the base model, and anything
/// else names an adapter: a path, absolute or relative to the models
/// directory, or the `<user>/<repo>:<file>` reference `download` fetched.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ImageLora {
    #[default]
    Auto,
    None,
    Spec(String),
}

impl ServerConfiguration {
    /// The picture defaults for a `variant` model with `adapter` in place:
    /// `image_steps` and `image_cfg_scale` as configured, or — left out —
    /// the model's own release settings (Qwen-Image's fifty steps at
    /// guidance 4, Qwen-Image 2.1's forty unguided ones), except under a
    /// Lightning adapter, where they are the step count its name carries
    /// and guidance off, which is what it was trained for (the base model's
    /// fifty guided steps under it, or eight unguided steps without it, are
    /// noise).
    pub fn image_defaults_under(
        &self,
        variant: crate::engine::image::Variant,
        adapter: Option<&std::path::Path>,
    ) -> crate::engine::image::ImageDefaults {
        let mut defaults = self.image.clone();
        let (release_steps, release_cfg) = variant.release_steps_and_cfg();
        if !self.image_steps_set {
            defaults.steps = release_steps;
        }
        if !self.image_cfg_scale_set {
            defaults.cfg_scale = release_cfg;
        }
        let steps = adapter
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .and_then(orangu::model_spec::lightning_steps);
        if let Some(steps) = steps {
            if !self.image_steps_set {
                defaults.steps = steps;
            }
            if !self.image_cfg_scale_set {
                defaults.cfg_scale = 1.0;
            }
        }
        defaults
    }
}

/// `[orangu-server].image_size`, `image_steps`, `image_cfg_scale`,
/// `image_negative_prompt` and `image_strength` — each optional, each
/// checked here so a typo is a startup error rather than a picture that
/// silently came out at the default.
fn parse_image_defaults(
    section: &HashMap<String, String>,
) -> Result<crate::engine::image::ImageDefaults> {
    let mut image = crate::engine::image::ImageDefaults::default();
    if let Some(value) = section.get("image_size") {
        let (width, height) = parse_image_size(value).ok_or_else(|| {
            anyhow!(
                "invalid value for [{SERVER_SECTION}].image_size: '{}' (expected WIDTHxHEIGHT, \
                 both multiples of 16)",
                value.trim()
            )
        })?;
        image.width = width;
        image.height = height;
    }
    if let Some(value) = section.get("image_steps") {
        image.steps = value
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|steps| *steps > 0)
            .ok_or_else(|| {
                anyhow!(
                    "invalid value for [{SERVER_SECTION}].image_steps: '{}' (expected a positive \
                     integer)",
                    value.trim()
                )
            })?;
    }
    if let Some(value) = section.get("image_cfg_scale") {
        image.cfg_scale = value
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|scale| scale.is_finite() && *scale >= 0.0)
            .ok_or_else(|| {
                anyhow!(
                    "invalid value for [{SERVER_SECTION}].image_cfg_scale: '{}' (expected a \
                     non-negative number; 1 turns guidance off)",
                    value.trim()
                )
            })?;
    }
    if let Some(value) = section.get("image_negative_prompt") {
        image.negative_prompt = value.trim().to_string();
    }
    if let Some(value) = section.get("image_strength") {
        image.strength = value
            .trim()
            .parse::<f32>()
            .ok()
            .filter(|s| (0.0..=1.0).contains(s))
            .ok_or_else(|| {
                anyhow!(
                    "invalid value for [{SERVER_SECTION}].image_strength: '{}' (expected a number \
                     from 0 to 1)",
                    value.trim()
                )
            })?;
    }
    if let Some(value) = section.get("image_format") {
        image.format = crate::engine::image::ImageFormat::parse(value).ok_or_else(|| {
            anyhow!(
                "invalid value for [{SERVER_SECTION}].image_format: '{}' (expected png, jpeg, gif, \
                 webp or svg)",
                value.trim()
            )
        })?;
    }
    Ok(image)
}

/// `WIDTHxHEIGHT` (or `WIDTH` alone for a square), both multiples of 16 —
/// the unit a picture is generated in: the VAE's 8 pixels per latent cell
/// times the transformer's 2x2 patch.
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
    (w >= 16 && h >= 16 && w.is_multiple_of(16) && h.is_multiple_of(16)).then_some((w, h))
}

/// The `approval_mode` values an MCP section may carry — orangu's own
/// `McpApprovalMode` spellings, checked here so the console's editor
/// refuses what the client would refuse.
pub const MCP_APPROVAL_MODES: [&str; 4] = ["auto", "prompt", "writes", "deny"];

impl McpConfiguration {
    /// Refuses what the file could not hold or the client could not read:
    /// a section name that is empty, is one of the two reserved sections,
    /// or would not survive the INI line it is written on; an empty
    /// endpoint; an approval mode orangu does not know.
    pub fn validate(&self) -> Result<()> {
        let name = self.name.trim();
        if name.is_empty() {
            bail!("an MCP server needs a name");
        }
        if name == SERVER_SECTION || name == WEB_SECTION {
            bail!("'{name}' is the server's own section, not an MCP server");
        }
        if name.contains(['[', ']', '\n', '\r', '=', '#', ';']) {
            bail!("an MCP server's name cannot contain [ ] = # ; or a line break");
        }
        let endpoint = self.endpoint.trim();
        if endpoint.is_empty() {
            bail!("[{name}].endpoint must be set for an MCP server");
        }
        if endpoint.contains(['\n', '\r']) {
            bail!("[{name}].endpoint cannot contain a line break");
        }
        if !MCP_APPROVAL_MODES.contains(&self.approval_mode.as_str()) {
            bail!(
                "invalid [{name}].approval_mode '{}'; expected auto, prompt, writes, or deny",
                self.approval_mode
            );
        }
        Ok(())
    }

    /// The section as the file holds it: the header, the endpoint, and the
    /// two other keys only where they differ from their defaults — the
    /// same rule `--init` writes by.
    pub fn render_section(&self) -> String {
        let mut out = format!(
            "[{}]\nendpoint = {}\n",
            self.name.trim(),
            self.endpoint.trim()
        );
        if !self.enabled {
            out.push_str("enabled = no\n");
        }
        if self.approval_mode != "writes" {
            out.push_str(&format!("approval_mode = {}\n", self.approval_mode));
        }
        out
    }
}

/// Rewrites one MCP section of a configuration file in place — replaced
/// by `replacement`, or removed when that is `None` — and leaves every
/// other line as it was, comments included: the file is the operator's,
/// and the console's editor touches only the section it was asked about.
/// A section is its header line through the line before the next header
/// (or the end of the file); a section that is not there is appended.
/// Written to a temporary file beside the original and renamed over it,
/// so a crash mid-write leaves the file as it was.
pub fn rewrite_mcp_section(
    path: &Path,
    name: &str,
    replacement: Option<&McpConfiguration>,
) -> Result<()> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read configuration {}", path.display()))?;
    let is_header = |line: &str| {
        let line = line.trim();
        line.strip_prefix('[')
            .and_then(|rest| rest.strip_suffix(']'))
            .map(|inner| inner.trim() == name)
            .unwrap_or(false)
    };
    let lines: Vec<&str> = contents.lines().collect();
    let start = lines.iter().position(|line| is_header(line));
    let end = start.map(|start| {
        lines[start + 1..]
            .iter()
            .position(|line| line.trim().starts_with('['))
            .map_or(lines.len(), |offset| start + 1 + offset)
    });
    let mut out = String::new();
    match (start, end) {
        (Some(start), Some(end)) => {
            for line in &lines[..start] {
                out.push_str(line);
                out.push('\n');
            }
            if let Some(mcp) = replacement {
                out.push_str(&mcp.render_section());
                // Keep the section's own trailing blank line, if it had one,
                // so the file keeps its spacing.
                if end < lines.len() && lines[end - 1].trim().is_empty() {
                    out.push('\n');
                }
            }
            for line in &lines[end..] {
                out.push_str(line);
                out.push('\n');
            }
        }
        _ => {
            out.push_str(&contents);
            if let Some(mcp) = replacement {
                if !out.is_empty() && !out.ends_with('\n') {
                    out.push('\n');
                }
                if !out.is_empty() && !out.ends_with("\n\n") {
                    out.push('\n');
                }
                out.push_str(&mcp.render_section());
            }
        }
    }
    let temp = path.with_extension("conf.tmp");
    std::fs::write(&temp, &out).with_context(|| format!("failed to write {}", temp.display()))?;
    std::fs::rename(&temp, path)
        .with_context(|| format!("failed to replace {}", path.display()))?;
    Ok(())
}

/// The MCP sections of a configuration file, as [`load_server_configuration`]
/// reads them — for the console to show what the file holds after an edit.
pub fn load_mcp_servers(path: &Path) -> Result<Vec<McpConfiguration>> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read configuration {}", path.display()))?;
    let mut sections = parse_ini_sections(&contents)?;
    sections.remove(SERVER_SECTION);
    sections.remove(WEB_SECTION);
    let mut mcp_servers = sections
        .into_iter()
        .map(|(name, values)| parse_mcp_configuration(name, values))
        .collect::<Result<Vec<_>>>()?;
    mcp_servers.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(mcp_servers)
}

fn parse_mcp_configuration(
    name: String,
    values: HashMap<String, String>,
) -> Result<McpConfiguration> {
    let endpoint = values
        .get("endpoint")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow!("[{name}].endpoint must be set for an MCP server"))?;
    let enabled = values
        .get("enabled")
        .map(|value| parse_bool(&name, "enabled", value))
        .transpose()?
        .unwrap_or(true);
    let approval_mode = values
        .get("approval_mode")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "writes".to_string());
    Ok(McpConfiguration {
        name,
        endpoint,
        enabled,
        approval_mode,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::image::Variant;
    use std::io::Write;

    #[test]
    fn loads_models_directory_with_defaults() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "[orangu-server]\nmodels = /srv/models\n").unwrap();

        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(conf.models, PathBuf::from("/srv/models"));
        assert_eq!(conf.host, "all");
        assert_eq!(conf.port, 8100);
        // orangu is local/single-user: generation defaults to one slot.
        assert_eq!(conf.slots, 1);
        assert_eq!(conf.web, 0);
        assert_eq!(conf.model, None);
        assert_eq!(conf.backend, BackendPreference::Auto);
        assert_eq!(conf.role, Role::All);
        // Both on by default.
        assert!(conf.reexec);
        assert!(conf.delete);
        // And the terminal, as it always was.
        assert_eq!(conf.log, LogTarget::Console);
    }

    /// `log_type = file` sends the server's output to `log_path`; the loader
    /// stores the resolved file so a `~` or a relative path is settled
    /// before a daemon moves to `/`. The console needs no path and ignores
    /// one it is given, so a config can keep a path written down while
    /// switched back.
    #[test]
    fn loads_the_log_keys() {
        let load = |lines: &str| {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            writeln!(file, "[orangu-server]\nmodels = /srv/models\n{lines}").unwrap();
            load_server_configuration(file.path(), None, false)
        };
        // Spelled from the current directory so the path is absolute on
        // every platform — a bare `/var/...` has no drive on Windows.
        let log_path = std::env::current_dir()
            .unwrap()
            .join("var")
            .join("orangu-server.log");
        assert_eq!(
            load(&format!(
                "log_type = file\nlog_path = {}\n",
                log_path.display()
            ))
            .unwrap()
            .log,
            LogTarget::File(log_path.clone())
        );
        assert_eq!(
            load(&format!(
                "log_type = console\nlog_path = {}\n",
                log_path.display()
            ))
            .unwrap()
            .log,
            LogTarget::Console
        );

        // A file with no path named is `orangu-server.log` where the server
        // was started; a type that is neither is an error rather than a
        // silent console.
        assert_eq!(
            load("log_type = file\n").unwrap().log,
            LogTarget::File(std::env::current_dir().unwrap().join("orangu-server.log"))
        );
        let err = load("log_type = syslog\n").unwrap_err().to_string();
        assert!(err.contains("[orangu-server].log_type"), "{err}");
        assert!(err.contains("console or file"), "{err}");
    }

    /// A bundle has no config file to have asked for a file log, and is run
    /// by hand: it logs to the console.
    #[test]
    fn a_bundle_logs_to_the_console() {
        assert_eq!(
            bundled_configuration(PathBuf::new(), Role::All, &BundledListen::default()).log,
            LogTarget::Console
        );
    }

    /// No `[prometheus]` section means disabled.
    #[test]
    fn no_prometheus_section_means_disabled() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "[orangu-server]\nmodels = /srv/models\n").unwrap();
        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(conf.metrics, 0);
    }

    #[test]
    fn reads_the_metrics_port_from_its_own_section() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\n\n[prometheus]\nport = 8300\n"
        )
        .unwrap();
        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(conf.metrics, 8300);
    }

    /// Having a `[prometheus]` section is what turns the listener on — a bare
    /// section with no `port` key still takes the default port, the same way
    /// a bare `[web]` section does.
    #[test]
    fn a_bare_prometheus_section_enables_the_listener_on_the_default_port() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\n\n[prometheus]\n"
        )
        .unwrap();
        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(conf.metrics, default_prometheus_port());
    }

    #[test]
    fn rejects_a_non_numeric_prometheus_port() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\n\n[prometheus]\nport = not-a-port\n"
        )
        .unwrap();
        let err = load_server_configuration(file.path(), None, false).unwrap_err();
        assert!(
            err.to_string()
                .contains("invalid value for [prometheus].port"),
            "{err:#}"
        );
    }

    #[test]
    fn the_metrics_listener_inherits_the_api_host_when_it_names_none() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\nhost = 192.168.1.10\n\n[prometheus]\nport = 8300\n"
        )
        .unwrap();
        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(conf.metrics_host, "192.168.1.10");
        assert!(!conf.metrics_host_explicit);
    }

    #[test]
    fn the_metrics_listener_can_bind_a_different_host_than_the_api() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\nhost = all\n\n[prometheus]\nport = 8300\nhost = 127.0.0.1\n"
        )
        .unwrap();
        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(conf.host, "all");
        assert_eq!(conf.metrics_host, "127.0.0.1");
        assert!(conf.metrics_host_explicit);
    }

    /// The whole promise of a bundle: no config file, and it still comes up
    /// somewhere reachable — on the loopback interface only, and with the
    /// web console on.
    #[test]
    fn a_bundle_needs_no_config_file_to_know_where_to_listen() {
        let conf = bundled_configuration(
            PathBuf::from("/srv/models"),
            Role::Code,
            &BundledListen::default(),
        );

        assert_eq!(conf.host, "127.0.0.1");
        assert_eq!(conf.port, 8100);
        assert_eq!(conf.web_host, "127.0.0.1");
        assert_eq!(conf.web, 8200);
        // Not the wildcard: a binary somebody downloaded and ran should not
        // put itself on every interface of a network it knows nothing about.
        assert_eq!(resolve_bind_host(&conf.host), "127.0.0.1");
        assert_eq!(resolve_bind_host(&conf.web_host), "127.0.0.1");
    }

    /// The role a bundle was built with is the role it serves in, and it
    /// carries through to the settings a role decides.
    #[test]
    fn a_bundles_role_decides_its_slot_count_like_any_other() {
        assert_eq!(
            bundled_configuration(PathBuf::new(), Role::Embedding, &BundledListen::default()).slots,
            Role::Embedding.default_slots()
        );
        let conf = bundled_configuration(PathBuf::new(), Role::Review, &BundledListen::default());
        assert_eq!(conf.role, Role::Review);
        assert_eq!(conf.role_key, Some(Role::Review));
        assert_eq!(conf.slots, Role::Review.default_slots());
        // The embedded model is not a spec to resolve against `models`, so
        // `--daemon` must not find one here and go looking for it.
        assert_eq!(conf.model, None);
    }

    /// `image_size` is `WIDTHxHEIGHT` or one side for a square, in the
    /// 16-pixel unit a picture is generated in; anything else is refused
    /// at startup rather than rounded.
    #[test]
    fn image_size_parses_pairs_and_squares_in_sixteens() {
        assert_eq!(parse_image_size("1024x768"), Some((1024, 768)));
        assert_eq!(parse_image_size(" 512 X 512 "), Some((512, 512)));
        assert_eq!(parse_image_size("640*400"), Some((640, 400)));
        assert_eq!(parse_image_size("768"), Some((768, 768)));
        assert_eq!(parse_image_size("1000x1000"), None);
        assert_eq!(parse_image_size("8"), None);
        assert_eq!(parse_image_size("wide"), None);
    }

    #[test]
    fn image_defaults_are_read_and_checked() {
        let mut section = HashMap::new();
        section.insert("image_size".to_string(), "512x256".to_string());
        section.insert("image_steps".to_string(), "20".to_string());
        section.insert("image_cfg_scale".to_string(), "1".to_string());
        section.insert("image_negative_prompt".to_string(), "blurry".to_string());
        section.insert("image_strength".to_string(), "0.8".to_string());
        section.insert("image_format".to_string(), "webp".to_string());
        let image = parse_image_defaults(&section).unwrap();
        assert_eq!((image.width, image.height), (512, 256));
        assert_eq!(image.steps, 20);
        assert_eq!(image.cfg_scale, 1.0);
        assert_eq!(image.negative_prompt, "blurry");
        assert_eq!(image.strength, 0.8);
        assert_eq!(image.format, crate::engine::image::ImageFormat::Webp);

        let defaults = parse_image_defaults(&HashMap::new()).unwrap();
        assert_eq!((defaults.width, defaults.height), (1024, 1024));
        assert_eq!(defaults.steps, 50);

        for (key, value) in [
            ("image_steps", "0"),
            ("image_cfg_scale", "-1"),
            ("image_strength", "2"),
            ("image_size", "100x100"),
            ("image_format", "bmp"),
        ] {
            let mut section = HashMap::new();
            section.insert(key.to_string(), value.to_string());
            let err = parse_image_defaults(&section).unwrap_err().to_string();
            assert!(err.contains(key), "{key}: {err}");
        }
    }

    /// A hand-edited `.ini` gets every spelling of a switch a person might
    /// reasonably write, not only the two Rust's `bool` parser accepts.
    #[test]
    fn parses_every_reexec_spelling() {
        for (value, expected) in [
            ("yes", true),
            ("YES", true),
            ("true", true),
            ("on", true),
            ("1", true),
            ("no", false),
            ("No", false),
            ("false", false),
            ("off", false),
            ("0", false),
        ] {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            writeln!(
                file,
                "[orangu-server]\nmodels = /srv/models\n\n[web]\nport = 8101\nreexec = {value}\n"
            )
            .unwrap();

            let conf = load_server_configuration(file.path(), None, false).unwrap();
            assert_eq!(conf.reexec, expected, "reexec = {value}");
        }
    }

    /// The editor rewrites one section and nothing else: comments, spacing
    /// and the other sections come through untouched; a missing section is
    /// appended; removal takes the section and its trailing blank line.
    #[test]
    fn an_mcp_section_is_rewritten_in_place_and_nothing_else_moves() {
        let file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            "# my server\n[orangu-server]\nmodels = /srv/models\n\n[docs]\nendpoint = http://a/\n\
             enabled = no\n\n[web]\nport = 8101 # console\n",
        )
        .unwrap();
        let docs = McpConfiguration {
            name: "docs".into(),
            endpoint: "http://b/".into(),
            enabled: true,
            approval_mode: "deny".into(),
        };
        rewrite_mcp_section(file.path(), "docs", Some(&docs)).unwrap();
        assert_eq!(
            std::fs::read_to_string(file.path()).unwrap(),
            "# my server\n[orangu-server]\nmodels = /srv/models\n\n[docs]\nendpoint = http://b/\n\
             approval_mode = deny\n\n[web]\nport = 8101 # console\n"
        );
        let tools = McpConfiguration {
            name: "tools".into(),
            endpoint: "http://c/".into(),
            enabled: false,
            approval_mode: "writes".into(),
        };
        rewrite_mcp_section(file.path(), "tools", Some(&tools)).unwrap();
        assert!(std::fs::read_to_string(file.path()).unwrap().ends_with(
            "[web]\nport = 8101 # console\n\n[tools]\nendpoint = http://c/\nenabled = no\n"
        ));
        rewrite_mcp_section(file.path(), "docs", None).unwrap();
        let after = std::fs::read_to_string(file.path()).unwrap();
        assert!(!after.contains("[docs]"));
        assert!(after.contains("[orangu-server]\nmodels = /srv/models\n\n[web]\n"));
        let servers = load_mcp_servers(file.path()).unwrap();
        assert_eq!(servers.len(), 1);
        assert_eq!(servers[0].name, "tools");
        assert!(!servers[0].enabled);

        let bad = McpConfiguration {
            name: "web".into(),
            ..docs.clone()
        };
        assert!(bad.validate().is_err());
        let bad = McpConfiguration {
            approval_mode: "maybe".into(),
            ..docs.clone()
        };
        assert!(bad.validate().is_err());
        assert!(docs.validate().is_ok());
    }

    /// `image_lora` absent or `auto` serves the adapter found under the
    /// models directory, `none` the base model, anything else that file;
    /// and under a Lightning adapter the steps and guidance left unset
    /// follow it — an explicit key is kept as written.
    #[test]
    fn image_lora_defaults_to_auto_and_the_adapter_sets_the_schedule() {
        let load = |keys: &str| {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            writeln!(file, "[orangu-server]\nmodels = /srv/models\n{keys}").unwrap();
            load_server_configuration(file.path(), None, false).unwrap()
        };
        let eight = Path::new("/srv/models/Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors");

        let conf = load("");
        assert_eq!(conf.image_lora, ImageLora::Auto);
        assert!(!conf.image_steps_set && !conf.image_cfg_scale_set);
        let under = conf.image_defaults_under(Variant::QwenImage, Some(eight));
        assert_eq!((under.steps, under.cfg_scale), (8, 1.0));
        let base = conf.image_defaults_under(Variant::QwenImage, None);
        assert_eq!((base.steps, base.cfg_scale), (50, 4.0));
        let other = conf.image_defaults_under(
            Variant::QwenImage,
            Some(Path::new("/srv/models/style.safetensors")),
        );
        assert_eq!((other.steps, other.cfg_scale), (50, 4.0));

        assert_eq!(load("image_lora = none").image_lora, ImageLora::None);
        assert_eq!(load("image_lora = auto").image_lora, ImageLora::Auto);
        assert_eq!(
            load("image_lora = lora/style.safetensors").image_lora,
            ImageLora::Spec("lora/style.safetensors".to_string())
        );

        let conf = load("image_steps = 4\nimage_cfg_scale = 2");
        assert!(conf.image_steps_set && conf.image_cfg_scale_set);
        let under = conf.image_defaults_under(Variant::QwenImage, Some(eight));
        assert_eq!((under.steps, under.cfg_scale), (4, 2.0));
        let v21 = conf.image_defaults_under(Variant::QwenImage21, None);
        assert_eq!((v21.steps, v21.cfg_scale), (4, 2.0));
    }

    /// `image_weights` defaults to `auto`, takes `int8` and `file`, and
    /// refuses anything else at startup.
    #[test]
    fn image_weights_parses_its_three_values() {
        use crate::engine::image::transformer21::ImageWeights;
        let load = |keys: &str| {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            writeln!(file, "[orangu-server]\nmodels = /srv/models\n{keys}").unwrap();
            load_server_configuration(file.path(), None, false)
        };
        assert_eq!(load("").unwrap().image_weights, ImageWeights::Auto);
        assert_eq!(
            load("image_weights = int8").unwrap().image_weights,
            ImageWeights::Int8
        );
        assert_eq!(
            load("image_weights = File").unwrap().image_weights,
            ImageWeights::File
        );
        assert!(load("image_weights = q4").is_err());
    }

    /// `prefix_warmup` defaults to on and takes the usual booleans.
    #[test]
    fn prefix_warmup_defaults_on() {
        let load = |keys: &str| {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            writeln!(file, "[orangu-server]\nmodels = /srv/models\n{keys}").unwrap();
            load_server_configuration(file.path(), None, false)
        };
        assert!(load("").unwrap().prefix_warmup);
        assert!(!load("prefix_warmup = off").unwrap().prefix_warmup);
        assert!(load("prefix_warmup = maybe").is_err());
    }

    /// `prompt_weights` defaults to `auto`, takes `copy` and `file`, and
    /// refuses anything else at startup.
    #[test]
    fn prompt_weights_parses_its_three_values() {
        use crate::engine::prompt_weights::PromptWeights;
        let load = |keys: &str| {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            writeln!(file, "[orangu-server]\nmodels = /srv/models\n{keys}").unwrap();
            load_server_configuration(file.path(), None, false)
        };
        assert_eq!(load("").unwrap().prompt_weights, PromptWeights::Auto);
        assert_eq!(
            load("prompt_weights = COPY").unwrap().prompt_weights,
            PromptWeights::Copy
        );
        assert_eq!(
            load("prompt_weights = file").unwrap().prompt_weights,
            PromptWeights::File
        );
        assert!(load("prompt_weights = q4").is_err());
        assert!(load("prompt_weights = int8").is_err());
    }

    /// `image_cache` defaults to `easy`, takes a threshold after it, and
    /// `off`.
    #[test]
    fn image_cache_parses() {
        use crate::engine::image::stepcache::ImageCache;
        let load = |keys: &str| {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            writeln!(file, "[orangu-server]\nmodels = /srv/models\n{keys}").unwrap();
            load_server_configuration(file.path(), None, false)
        };
        assert_eq!(
            load("").unwrap().image_cache,
            ImageCache::Easy(ImageCache::EASY)
        );
        assert_eq!(
            load("image_cache = off").unwrap().image_cache,
            ImageCache::Off
        );
        assert_eq!(
            load("image_cache = easy:0.1").unwrap().image_cache,
            ImageCache::Easy(0.1)
        );
        assert!(load("image_cache = fast").is_err());
    }

    /// `image_reference_size` is `source` by default, `output`, or an area
    /// cap.
    #[test]
    fn image_reference_size_parses() {
        let load = |keys: &str| {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            writeln!(file, "[orangu-server]\nmodels = /srv/models\n{keys}").unwrap();
            load_server_configuration(file.path(), None, false)
        };
        use crate::engine::image::ReferenceCap;
        assert_eq!(load("").unwrap().image_reference, ReferenceCap::Source);
        assert_eq!(
            load("image_reference_size = Output")
                .unwrap()
                .image_reference,
            ReferenceCap::Output
        );
        assert_eq!(
            load("image_reference_size = 512x512")
                .unwrap()
                .image_reference,
            ReferenceCap::Area(512 * 512)
        );
        assert!(load("image_reference_size = big").is_err());
    }

    /// Qwen-Image 2.1's own settings — forty steps, no guidance — are what
    /// it gets when the config leaves them out.
    #[test]
    fn qwen_image_2_1_defaults_to_its_release_settings() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "[orangu-server]\nmodels = /srv/models\n").unwrap();
        let conf = load_server_configuration(file.path(), None, false).unwrap();
        let d = conf.image_defaults_under(Variant::QwenImage21, None);
        assert_eq!((d.steps, d.cfg_scale), (40, 1.0));
        assert_eq!((d.width, d.height), (1024, 1024));
    }

    /// NPU precompilation is on unless it is turned off, and a start with
    /// no opinion about it gets the default rather than nothing.
    ///
    /// This default has moved twice: off when the device was first measured
    /// against a real prompt rather than a sixteen-token one, and back on
    /// once the calibration and the per-channel smoothing made a real
    /// prompt come back with a real answer — see [`default_npu_precompile`].
    /// `prefill_backend` is `auto` unless the file says `cpu` or `device`;
    /// anything else stops the server at the file.
    #[test]
    fn prefill_backend_is_auto_unless_the_file_names_one() {
        use crate::engine::prefill_backend::PrefillBackend;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "[orangu-server]\nmodels = /srv/models\n").unwrap();
        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(conf.prefill_backend, PrefillBackend::Auto);
        for (value, expected) in [
            ("cpu", PrefillBackend::Cpu),
            ("device", PrefillBackend::Device),
            ("CPU", PrefillBackend::Cpu),
            ("auto", PrefillBackend::Auto),
        ] {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            writeln!(
                file,
                "[orangu-server]\nmodels = /srv/models\nprefill_backend = {value}\n"
            )
            .unwrap();
            let conf = load_server_configuration(file.path(), None, false).unwrap();
            assert_eq!(conf.prefill_backend, expected, "prefill_backend = {value}");
        }
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\nprefill_backend = gpu\n"
        )
        .unwrap();
        assert!(load_server_configuration(file.path(), None, false).is_err());
    }

    #[test]
    fn npu_precompile_defaults_on_and_can_be_turned_off() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "[orangu-server]\nmodels = /srv/models\n").unwrap();
        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert!(conf.npu_precompile, "unset should mean on");

        for (value, expected) in [("off", false), ("no", false), ("on", true), ("1", true)] {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            writeln!(
                file,
                "[orangu-server]\nmodels = /srv/models\nnpu_precompile = {value}\n"
            )
            .unwrap();
            let conf = load_server_configuration(file.path(), None, false).unwrap();
            assert_eq!(conf.npu_precompile, expected, "npu_precompile = {value}");
        }
    }

    /// A misspelling is an error rather than a silent "off" — the same
    /// reason `reexec` refuses one.
    #[test]
    fn rejects_an_invalid_npu_precompile_value() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\nnpu_precompile = sometimes\n"
        )
        .unwrap();
        let err = load_server_configuration(file.path(), None, false).unwrap_err();
        assert!(err.to_string().contains("npu_precompile"), "{err}");
        assert!(err.to_string().contains("sometimes"), "{err}");
    }

    /// A misspelling must not quietly read as "off" — that would silently
    /// take away a button the config was trying to keep.
    #[test]
    fn rejects_an_invalid_reexec_value() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\n\n[web]\nreexec = maybe\n"
        )
        .unwrap();

        let err = load_server_configuration(file.path(), None, false).unwrap_err();
        assert!(err.to_string().contains("[web].reexec"), "{err}");
        assert!(err.to_string().contains("maybe"), "{err}");
    }

    /// Having a `[web]` section is what turns the console on — so one that
    /// names nothing at all still gets a working port.
    #[test]
    fn a_bare_web_section_enables_the_console_on_the_default_port() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "[orangu-server]\nmodels = /srv/models\n\n[web]\n").unwrap();

        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(conf.web, default_web_port());
        assert!(conf.reexec);
        assert!(conf.delete);
    }

    #[test]
    fn parses_every_delete_spelling() {
        for (value, expected) in [
            ("yes", true),
            ("true", true),
            ("on", true),
            ("1", true),
            ("no", false),
            ("NO", false),
            ("false", false),
            ("off", false),
            ("0", false),
        ] {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            writeln!(
                file,
                "[orangu-server]\nmodels = /srv/models\n\n[web]\nport = 8101\ndelete = {value}\n"
            )
            .unwrap();

            let conf = load_server_configuration(file.path(), None, false).unwrap();
            assert_eq!(conf.delete, expected, "delete = {value}");
        }
    }

    /// The two console switches are independent: a deployment may well want
    /// a model switch allowed while the models directory stays read-only.
    #[test]
    fn delete_and_reexec_are_set_independently() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\n\n[web]\nport = 8101\nreexec = yes\ndelete = no\n"
        )
        .unwrap();

        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert!(conf.reexec);
        assert!(!conf.delete);
    }

    #[test]
    fn rejects_an_invalid_delete_value() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\n\n[web]\ndelete = sometimes\n"
        )
        .unwrap();

        let err = load_server_configuration(file.path(), None, false).unwrap_err();
        assert!(err.to_string().contains("[web].delete"), "{err}");
    }

    /// No `[web]` section means no console and no second listener — which is
    /// what `--init` writes when it is declined.
    #[test]
    fn no_web_section_means_no_console() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "[orangu-server]\nmodels = /srv/models\n").unwrap();

        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(conf.web, 0);
    }

    /// The console inherits the API's address unless it says otherwise, so
    /// the ordinary config names one host and both listeners use it.
    #[test]
    fn the_web_console_inherits_the_api_host_when_it_names_none() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\nhost = 192.168.1.10\n\n[web]\nport = 8101\n"
        )
        .unwrap();

        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(conf.web_host, "192.168.1.10");
        assert_eq!(conf.host, conf.web_host);
    }

    /// `--host` moves the console along with the API when the console was
    /// only following it anyway, and leaves it alone when a config put it
    /// somewhere on purpose. Exposing the API must never be a way to expose
    /// the console by accident, which is the whole reason this flag is
    /// recorded rather than inferred from the two addresses matching.
    #[test]
    fn only_an_inherited_web_host_is_flagged_as_following_the_api() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\nhost = 127.0.0.1\n\n[web]\nport = 8101\n"
        )
        .unwrap();
        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(conf.web_host, "127.0.0.1");
        assert!(!conf.web_host_explicit);

        // The same two addresses, but one of them was asked for by name.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\nhost = 127.0.0.1\n\n[web]\nport = 8101\nhost = 127.0.0.1\n"
        )
        .unwrap();
        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(conf.web_host, "127.0.0.1");
        assert!(conf.web_host_explicit);

        // No `[web]` section at all, and the legacy `[orangu-server].web`
        // spelling, both predate there being a key to be explicit with.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "[orangu-server]\nmodels = /srv/models\nweb = 8200\n").unwrap();
        assert!(
            !load_server_configuration(file.path(), None, false)
                .unwrap()
                .web_host_explicit
        );

        // And a bundle, which has no config file to have written one.
        assert!(
            !bundled_configuration(PathBuf::new(), Role::All, &BundledListen::default())
                .web_host_explicit
        );
    }

    /// And overrides it when it does — the case this exists for: an API
    /// reachable from the network, with the console kept off it.
    #[test]
    fn the_web_console_can_bind_a_different_host_than_the_api() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\nhost = all\n\n[web]\nport = 8101\nhost = 127.0.0.1\n"
        )
        .unwrap();

        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(conf.host, "all");
        assert_eq!(conf.web_host, "127.0.0.1");
        assert_eq!(resolve_bind_host(&conf.host), "0.0.0.0");
        assert_eq!(resolve_bind_host(&conf.web_host), "127.0.0.1");
    }

    #[test]
    fn reads_the_web_port_from_its_own_section() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\n\n[web]\nport = 8200\n"
        )
        .unwrap();

        assert_eq!(
            load_server_configuration(file.path(), None, false)
                .unwrap()
                .web,
            8200
        );
    }

    /// `[orangu-server].web` is the spelling that shipped before `[web]`
    /// existed. A config written against it has to go on working untouched —
    /// silently disabling somebody's console because a key moved would be
    /// the worst possible way to introduce a section.
    #[test]
    fn the_pre_section_web_key_still_works() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "[orangu-server]\nmodels = /srv/models\nweb = 8200\n").unwrap();

        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(conf.web, 8200);
        assert!(conf.reexec, "the legacy spelling gets the default");
        assert!(conf.delete, "the legacy spelling gets the default");
        assert_eq!(conf.web_host, conf.host);
    }

    /// ...but only until there is a `[web]` section, which is the one that
    /// means anything once it exists.
    #[test]
    fn a_web_section_takes_precedence_over_the_legacy_key() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\nweb = 9999\n\n[web]\nport = 8200\n"
        )
        .unwrap();

        assert_eq!(
            load_server_configuration(file.path(), None, false)
                .unwrap()
                .web,
            8200
        );
    }

    #[test]
    fn parses_each_backend_value_case_insensitively() {
        for (value, expected) in [
            ("cpu", BackendPreference::Cpu),
            ("CPU", BackendPreference::Cpu),
            ("vulkan", BackendPreference::Vulkan),
            ("metal", BackendPreference::Metal),
            ("METAL", BackendPreference::Metal),
            ("dx12", BackendPreference::Dx12),
            ("cuda", BackendPreference::Cuda),
            ("CUDA", BackendPreference::Cuda),
            ("opencl", BackendPreference::OpenCl),
            ("rocm", BackendPreference::Rocm),
            ("npu", BackendPreference::Npu),
            ("NPU", BackendPreference::Npu),
            ("auto", BackendPreference::Auto),
        ] {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            writeln!(
                file,
                "[orangu-server]\nmodels = /srv/models\nbackend = {value}\n"
            )
            .unwrap();

            let conf = load_server_configuration(file.path(), None, false).unwrap();
            assert_eq!(conf.backend, expected, "backend = {value}");
        }
    }

    /// `device` reads as an index, a name, or the policy — and its absence
    /// is the policy, so every config written before the key existed keeps
    /// choosing a device the same way.
    #[test]
    fn loads_the_device_key_in_each_of_its_three_forms() {
        for (value, expected) in [
            ("auto", DeviceRequest::Auto),
            ("1", DeviceRequest::Index(1)),
            ("RX 7900", DeviceRequest::Name("RX 7900".to_string())),
        ] {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            writeln!(
                file,
                "[orangu-server]\nmodels = /srv/models\ndevice = {value}\n"
            )
            .unwrap();

            let conf = load_server_configuration(file.path(), None, false).unwrap();
            assert_eq!(conf.device, expected, "device = {value}");
        }
    }

    #[test]
    fn an_absent_device_key_is_the_ranking_policy() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "[orangu-server]\nmodels = /srv/models\n").unwrap();

        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(conf.device, DeviceRequest::Auto);
    }

    /// A device that doesn't exist is *not* rejected here. It cannot be:
    /// nothing at config-parse time has asked a driver what the machine
    /// has, and rejecting `device = 9` without being able to print the
    /// devices that do exist would be a worse error than the one startup
    /// gives.
    #[test]
    fn a_nonexistent_device_is_not_a_config_error() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "[orangu-server]\nmodels = /srv/models\ndevice = 9\n").unwrap();

        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(conf.device, DeviceRequest::Index(9));
    }

    #[test]
    fn loads_every_form_of_device_split() {
        for (value, expected) in [
            ("off", SplitMode::Off),
            ("auto", SplitMode::Auto),
            ("ALL", SplitMode::All),
            ("3,1", SplitMode::Ratios(vec![3.0, 1.0])),
        ] {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            writeln!(
                file,
                "[orangu-server]\nmodels = /srv/models\ndevice_split = {value}\n"
            )
            .unwrap();

            let conf = load_server_configuration(file.path(), None, false).unwrap();
            assert_eq!(conf.device_split, expected, "device_split = {value}");
        }
    }

    /// `context` is a token count, absent by default, and zero is refused
    /// rather than read as "none".
    #[test]
    fn loads_the_context_key() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\ncontext = 8192\n"
        )
        .unwrap();
        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(conf.context, Some(8192));

        let mut absent = tempfile::NamedTempFile::new().unwrap();
        writeln!(absent, "[orangu-server]\nmodels = /srv/models\n").unwrap();
        assert_eq!(
            load_server_configuration(absent.path(), None, false)
                .unwrap()
                .context,
            None
        );

        let mut zero = tempfile::NamedTempFile::new().unwrap();
        writeln!(zero, "[orangu-server]\nmodels = /srv/models\ncontext = 0\n").unwrap();
        assert!(load_server_configuration(zero.path(), None, false).is_err());
    }

    /// One device runs the model unless something says otherwise — a config
    /// written before this key existed must keep the behaviour it had.
    #[test]
    fn an_absent_device_split_key_keeps_the_model_on_one_device() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "[orangu-server]\nmodels = /srv/models\n").unwrap();

        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(conf.device_split, SplitMode::Off);
    }

    /// Unlike `device`, this one *is* decidable at parse time — so a typo
    /// must stop the server rather than becoming a silent `off`, which is
    /// precisely the outcome somebody setting the key is trying to avoid.
    #[test]
    fn rejects_an_invalid_device_split_value() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\ndevice_split = sideways\n"
        )
        .unwrap();

        let err = load_server_configuration(file.path(), None, false).unwrap_err();
        let message = err.to_string();
        assert!(message.contains("device_split"), "{message}");
        assert!(message.contains("off, auto, all"), "{message}");
    }

    #[test]
    fn loads_the_threads_key_and_rejects_zero() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "[orangu-server]\nmodels = /srv/models\nthreads = 6\n").unwrap();
        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(conf.threads, Some(6));

        // Absent means rayon's own choice, not "no threads".
        let mut bare = tempfile::NamedTempFile::new().unwrap();
        writeln!(bare, "[orangu-server]\nmodels = /srv/models\n").unwrap();
        assert_eq!(
            load_server_configuration(bare.path(), None, false)
                .unwrap()
                .threads,
            None
        );

        // Zero is a mistake rather than a way to say "default": rayon reads
        // `num_threads(0)` as the default, which would make a typo silently
        // mean the opposite of what it looks like.
        let mut zero = tempfile::NamedTempFile::new().unwrap();
        writeln!(zero, "[orangu-server]\nmodels = /srv/models\nthreads = 0\n").unwrap();
        let err = load_server_configuration(zero.path(), None, false).unwrap_err();
        assert!(err.to_string().contains("threads"), "{err}");
    }

    #[test]
    fn cpu_is_a_device_split_mode() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\ndevice_split = cpu\n"
        )
        .unwrap();
        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(conf.device_split, SplitMode::Cpu);
    }

    #[test]
    fn rejects_an_invalid_backend_value() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\nbackend = quantum\n"
        )
        .unwrap();

        let err = load_server_configuration(file.path(), None, false).unwrap_err();
        assert!(
            err.to_string().contains("backend"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn loads_the_model_key_for_daemon_mode() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\nmodel = unsloth/gemma-4-E2B-it-GGUF:Q4_K_M\n"
        )
        .unwrap();

        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(
            conf.model.as_deref(),
            Some("unsloth/gemma-4-E2B-it-GGUF:Q4_K_M")
        );
    }

    #[test]
    fn overrides_host_port_slots_and_web() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\nhost = 0.0.0.0\nport = 9090\nslots = 4\nweb = 8081\n"
        )
        .unwrap();

        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(conf.host, "0.0.0.0");
        assert_eq!(conf.port, 9090);
        assert_eq!(conf.slots, 4);
        assert_eq!(conf.web, 8081);
    }

    /// `all`/`*` are the only two values rewritten before binding — spelled
    /// any way, since the config file is hand-edited — and a real address is
    /// handed to `bind` exactly as written.
    #[test]
    fn resolves_only_the_all_host_to_the_wildcard_address() {
        for value in ["all", "ALL", " All ", "*", " * "] {
            assert_eq!(resolve_bind_host(value), "0.0.0.0", "host = {value}");
        }
        for value in ["127.0.0.1", "0.0.0.0", "192.168.1.10", "::1"] {
            assert_eq!(resolve_bind_host(value), value, "host = {value}");
        }
    }

    #[test]
    fn requires_models_key() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "[orangu-server]\n").unwrap();

        let err = load_server_configuration(file.path(), None, false).unwrap_err();
        assert!(
            err.to_string().contains("models"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn rejects_zero_slots() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "[orangu-server]\nmodels = /srv/models\nslots = 0\n").unwrap();

        let err = load_server_configuration(file.path(), None, false).unwrap_err();
        assert!(
            err.to_string().contains("slots"),
            "unexpected error: {err:#}"
        );
    }

    /// The key exists, parses, defaults to 8 MiB, and rejects the two values
    /// that would produce an unaligned read — a config that does not mention
    /// it behaves exactly as it did before the key existed.
    #[test]
    fn the_read_size_key_parses_defaults_and_rejects_unaligned_values() {
        let load = |line: &str| {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            writeln!(file, "[orangu-server]\nmodels = /tmp\n{line}").unwrap();
            load_server_configuration(file.path(), None, false)
        };

        assert_eq!(load("").unwrap().read_size, DEFAULT_READ_SIZE);
        assert_eq!(
            load("").unwrap().read_size,
            8192,
            "the default is 8 MiB — measured +36% decode on a cold mixture-of-experts model"
        );
        assert_eq!(load("read_size = 4\n").unwrap().read_size, 4);
        assert_eq!(load("read_size = 65536\n").unwrap().read_size, 65536);

        // Zero is not "no widening", it is a read of nothing.
        let err = load("read_size = 0\n").unwrap_err().to_string();
        assert!(err.contains("read_size"), "unexpected error: {err}");

        // `O_DIRECT` needs page alignment, so a granule that is not a whole
        // number of pages is refused rather than quietly rounded — a swept
        // value that does not survive to the read is an arm that secretly
        // ran the control.
        let err = load("read_size = 6\n").unwrap_err().to_string();
        assert!(err.contains("multiple of 4"), "unexpected error: {err}");

        let err = load("read_size = banana\n").unwrap_err().to_string();
        assert!(err.contains("read_size"), "unexpected error: {err}");
    }

    /// The key exists, parses all three values, and defaults to `f16` — the
    /// behaviour the environment variable had before it was a key at all, so
    /// adding the key changes nothing for a config that does not mention it.
    #[test]
    fn the_kv_cache_key_parses_and_defaults_to_f16() {
        let load = |line: &str| {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            writeln!(file, "[orangu-server]\nmodels = /tmp\n{line}").unwrap();
            load_server_configuration(file.path(), None, false)
        };
        assert_eq!(load("").unwrap().kv_cache, KvCache::F16);
        assert_eq!(load("kv_cache = f16\n").unwrap().kv_cache, KvCache::F16);
        assert_eq!(load("kv_cache = q8_0\n").unwrap().kv_cache, KvCache::Q8_0);
        assert_eq!(load("kv_cache = f32\n").unwrap().kv_cache, KvCache::F32);
    }

    /// A misspelled value is an error naming the alternatives, never a silent
    /// fall back to the default: `kv_cache` chooses between a lossless and a
    /// lossy format, so a typo that quietly kept `f16` would be an operator
    /// asking for less memory and getting none of it, with nothing said.
    #[test]
    fn an_unknown_kv_cache_value_is_rejected_with_the_alternatives() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "[orangu-server]\nmodels = /tmp\nkv_cache = int8\n").unwrap();
        let err = load_server_configuration(file.path(), None, false).unwrap_err();
        let text = err.to_string();
        assert!(text.contains("kv_cache"), "{text}");
        assert!(text.contains("int8"), "{text}");
        assert!(text.contains("q8_0"), "{text}");
    }

    #[test]
    fn expands_leading_tilde() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(file, "[orangu-server]\nmodels = ~/models\n").unwrap();

        let conf = load_server_configuration(file.path(), None, false).unwrap();
        let home = home::home_dir().unwrap();
        assert_eq!(conf.models, home.join("models"));
    }

    /// The config file's `role` key is only ever consulted in `--daemon`
    /// mode — same as `model` (see its own doc comment). `daemon: true`
    /// here is what actually exercises it.
    #[test]
    fn parses_each_role_value_case_insensitively_from_the_config_file_in_daemon_mode() {
        for (value, expected) in [
            ("all", Role::All),
            ("ALL", Role::All),
            ("code", Role::Code),
            ("review", Role::Review),
            ("explorer", Role::Explorer),
            ("embedding", Role::Embedding),
            ("image", Role::Image),
        ] {
            let mut file = tempfile::NamedTempFile::new().unwrap();
            writeln!(
                file,
                "[orangu-server]\nmodels = /srv/models\nrole = {value}\n"
            )
            .unwrap();

            let conf = load_server_configuration(file.path(), None, true).unwrap();
            assert_eq!(conf.role, expected, "role = {value}");
        }
    }

    /// `image` is the one role a file's architecture decides: a `qwen_image`
    /// model requires it, nothing else may take it, and it is never asked
    /// for.
    #[test]
    fn the_image_role_is_fixed_by_a_qwen_image_architecture() {
        assert_eq!(Role::required_by(Some("qwen_image")), Some(Role::Image));
        assert_eq!(Role::required_by(Some("qwen_image_2_1")), Some(Role::Image));
        assert_eq!(Role::required_by(Some("llama")), None);
        assert_eq!(Role::required_by(None), None);
        assert!(Role::Image.fixed_by_model());
        for role in [
            Role::All,
            Role::Code,
            Role::Review,
            Role::Explorer,
            Role::Embedding,
        ] {
            assert!(!role.fixed_by_model(), "{}", role.label());
        }
        assert!(Role::Image.allows_generation());
        assert_eq!(Role::Image.default_slots(), 1);
    }

    /// Outside `--daemon` mode, the config file's `role` key isn't even
    /// looked at — a missing CLI flag always means `Role::All`, exactly
    /// as if the key (however it's spelled, valid or not) weren't there.
    #[test]
    fn config_files_role_key_is_ignored_outside_daemon_mode() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\nrole = embedding\n"
        )
        .unwrap();

        let conf = load_server_configuration(file.path(), None, false).unwrap();
        assert_eq!(conf.role, Role::All);
    }

    #[test]
    fn rejects_an_invalid_role_value_in_daemon_mode() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\nrole = summarizer\n"
        )
        .unwrap();

        let err = load_server_configuration(file.path(), None, true).unwrap_err();
        assert!(
            err.to_string().contains("role"),
            "unexpected error: {err:#}"
        );
    }

    /// An explicit CLI role flag overrides the config file's own `role`
    /// key — `--daemon` mode is the one case where a CLI flag and a
    /// config-file `role` key could genuinely both be present at once
    /// (e.g. a saved daemon config defaulting to `embedding`, started
    /// once with `--review` to override it for a single run).
    #[test]
    fn cli_role_overrides_the_config_files_role_key_in_daemon_mode() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\nrole = embedding\n"
        )
        .unwrap();

        let conf = load_server_configuration(file.path(), Some(Role::Review), true).unwrap();
        assert_eq!(conf.role, Role::Review);
    }

    /// `Role::Embedding`'s higher default slot count only applies when
    /// `slots` isn't set explicitly in the config file — an explicit
    /// `slots` value always wins, for every role. Uses `daemon: true` so
    /// the config's `role = embedding` is actually picked up.
    #[test]
    fn embedding_role_defaults_slots_to_eight_unless_overridden() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\nrole = embedding\n"
        )
        .unwrap();
        let conf = load_server_configuration(file.path(), None, true).unwrap();
        assert_eq!(conf.slots, 8);

        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-server]\nmodels = /srv/models\nrole = embedding\nslots = 3\n"
        )
        .unwrap();
        let conf = load_server_configuration(file.path(), None, true).unwrap();
        assert_eq!(conf.slots, 3);
    }
}
