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
use crate::tenant::{TenantConfig, TenantLimits};
use anyhow::{Context, Result, anyhow};
use orangu::config::parse_ini_sections;
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

/// The web console's own section. **Its presence is what enables the
/// console** — a config with no `[web]` section binds no second listener at
/// all, which is what `--init` writes when the web console is declined.
pub const WEB_SECTION: &str = "web";

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
        // The console follows the API's address, baked-in or default —
        // `bundle --host all` means "expose this bundle", not "expose half
        // of it". `--web 0` at build time, or at run time, is how a bundle
        // exposes only the API.
        web_host: host.clone(),
        host,
        port: listen.port.unwrap_or_else(default_port),
        slots: role.default_slots(),
        web: listen.web.unwrap_or_else(bundled_web_port),
        // Nothing wrote a `[web].host` here, so `--host` at run time moves
        // the console along with the API — which is what makes `--host all`
        // on a bundle do the one thing somebody would reach for it to do.
        web_host_explicit: false,
        backend: default_backend(),
        // A bundle runs on a machine nobody configured, so it takes the
        // default rather than a lossy format nobody chose.
        kv_cache: KvCache::default(),
        // A bundle serves whoever runs it; refusing on its owner's behalf is
        // not a decision it can make.
        queue_limit: 0,
        // A bundle carries one model. Pairing it with a draft would mean
        // embedding a second, which is a different product decision than
        // "the server and a model as one file".
        draft_model: None,
        draft_tokens: DEFAULT_DRAFT_TOKENS,
        // A bundle carries no certificate; TLS is a per-deployment decision.
        tls: None,
        // A key baked into a distributed executable is a key everyone who has
        // the executable knows. `ORANGU_API_KEY` still applies at run time.
        api_key: std::env::var("ORANGU_API_KEY")
            .ok()
            .filter(|k| !k.is_empty()),
        // Tenants come from config sections, and a bundle has no config file.
        tenants: Vec::new(),
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
}

impl Role {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_lowercase().as_str() {
            "all" => Ok(Role::All),
            "code" => Ok(Role::Code),
            "review" => Ok(Role::Review),
            "explorer" => Ok(Role::Explorer),
            "embedding" => Ok(Role::Embedding),
            other => Err(anyhow!(
                "invalid role '{other}' (expected all, code, review, explorer, or embedding)"
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
            // orangu is a local, single-user AI, so generation defaults to one
            // slot. Concurrent decode is GPU/weight-bandwidth-bound — extra
            // slots don't raise throughput
            // (each token already streams the whole weight set), they only add
            // KV-cache memory. A multi-user deployment can still set `slots` in
            // the config file.
            Role::All | Role::Code | Role::Review | Role::Explorer => 1,
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
    /// Which `Backend` runs the forward pass — CPU, a named GPU API, or
    /// (the default) whichever GPU this platform finds first, falling back
    /// to CPU.
    pub backend: BackendPreference,
    /// How the GPU-side KV mirror is stored — see [`KvCache`]. Overridden by
    /// `ORANGU_KV_CACHE`, so a sweep can vary it without editing the file.
    pub kv_cache: KvCache,
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
    /// `[tenant:<name>]` sections: named keys, each with its own limits.
    ///
    /// Empty by default, which is every deployment that has not asked for
    /// this. A non-empty list turns authentication on by itself, with or
    /// without [`api_key`](Self::api_key): declaring who may call is a
    /// statement that not everyone may, and a server that accepted anonymous
    /// requests alongside metered ones would meter nothing.
    pub tenants: Vec<TenantConfig>,
    /// `[orangu-server].queue_limit`: how many requests may wait for a slot
    /// before the server starts refusing with `503`. `0` — the default —
    /// queues without bound, which is the behaviour before this key existed.
    pub queue_limit: usize,
    /// `[orangu-server].draft_model`: a second, smaller model whose guesses
    /// the served model verifies — speculative decoding. A model spec, the
    /// same shape as [`model`](Self::model).
    ///
    /// `None` (the default) decodes exactly as before. The pair must share a
    /// vocabulary, which is checked at startup rather than discovered as
    /// nonsense output.
    pub draft_model: Option<String>,
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
            other => {
                return Err(anyhow!(
                    "invalid value for [{SERVER_SECTION}].backend: '{other}' \
                     (expected auto, cpu, vulkan, metal, dx12, cuda, opencl, or rocm)"
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

    // Taken out before the rest of the file is read as MCP servers, which is
    // what every remaining section means. Without this a `[tenant:alice]`
    // section would be reported as an MCP server missing an `endpoint` — an
    // error about a key the operator never wrote.
    let tenant_names: Vec<String> = sections
        .keys()
        .filter(|name| name.starts_with(TENANT_PREFIX))
        .cloned()
        .collect();
    let mut tenants = Vec::new();
    for section_name in tenant_names {
        let values = sections
            .remove(&section_name)
            .expect("the name came from this map");
        tenants.push(parse_tenant(&section_name, &values)?);
    }
    tenants.sort_by(|left: &TenantConfig, right: &TenantConfig| left.name.cmp(&right.name));
    reject_ambiguous_keys(&tenants, api_key.as_deref())?;

    let mut mcp_servers = sections
        .into_iter()
        .map(|(name, values)| parse_mcp_configuration(name, values))
        .collect::<Result<Vec<_>>>()?;
    mcp_servers.sort_by(|left, right| left.name.cmp(&right.name));

    Ok(ServerConfiguration {
        models: expand_tilde(&models),
        host,
        port,
        model,
        role_key,
        role,
        slots,
        web,
        web_host,
        web_host_explicit,
        backend,
        kv_cache,
        queue_limit,
        draft_model,
        draft_tokens,
        api_key,
        tenants,
        tls,
        device,
        device_split,
        threads,
        reexec,
        delete,
        mcp_servers,
    })
}

/// What marks a section as a tenant rather than an MCP server.
///
/// A prefix rather than a `[tenants]` block with one key per name, because a
/// tenant carries four values and an INI file has no nesting to express that
/// in. It is the same shape the client config already uses for a named server.
pub const TENANT_PREFIX: &str = "tenant:";

fn parse_tenant(section: &str, values: &HashMap<String, String>) -> Result<TenantConfig> {
    let name = section
        .strip_prefix(TENANT_PREFIX)
        .expect("caller filtered on the prefix")
        .trim()
        .to_string();
    if name.is_empty() {
        return Err(anyhow!(
            "[{section}] has no name — the section header is [{TENANT_PREFIX}<name>], \
             and the name is what appears in /metrics and in refusals"
        ));
    }

    // Either the key itself or the name of a variable holding it, never both:
    // two spellings of one secret is a configuration whose effective value
    // depends on a precedence rule nobody should have to know.
    let literal = values
        .get("api_key")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let from_env = values
        .get("api_key_env")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let api_key = match (literal, from_env) {
        (Some(_), Some(_)) => {
            return Err(anyhow!(
                "[{section}] sets both api_key and api_key_env — use one"
            ));
        }
        (Some(key), None) => key,
        // The variable must be set *now*. Falling back to "no key" would leave
        // a tenant nobody can authenticate as, and falling back to open would
        // publish the server; a typo in a variable name should stop the
        // server, naming the variable.
        (None, Some(variable)) => std::env::var(&variable)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .ok_or_else(|| {
                anyhow!("[{section}].api_key_env names {variable}, which is unset or empty")
            })?,
        (None, None) => {
            return Err(anyhow!(
                "[{section}] must set api_key (or api_key_env) — a tenant is identified by \
                 the key it presents, so one without a key can never be recognised"
            ));
        }
    };

    let limits = TenantLimits {
        max_concurrent: tenant_number::<usize>(section, values, "max_concurrent")?.unwrap_or(0),
        requests_per_minute: tenant_number::<u64>(section, values, "requests_per_minute")?
            .unwrap_or(0),
        tokens_per_minute: tenant_number::<u64>(section, values, "tokens_per_minute")?.unwrap_or(0),
    };

    // Anything else is a typo in a key that silently does nothing — and a
    // limit that silently does nothing is the failure this whole feature is
    // supposed to prevent.
    const KNOWN: [&str; 5] = [
        "api_key",
        "api_key_env",
        "max_concurrent",
        "requests_per_minute",
        "tokens_per_minute",
    ];
    let mut unknown: Vec<&str> = values
        .keys()
        .map(String::as_str)
        .filter(|key| !KNOWN.contains(key))
        .collect();
    unknown.sort_unstable();
    if let Some(key) = unknown.first() {
        return Err(anyhow!(
            "[{section}] has no key '{key}' (expected {})",
            KNOWN.join(", ")
        ));
    }

    Ok(TenantConfig {
        name,
        api_key,
        limits,
    })
}

fn tenant_number<T>(section: &str, values: &HashMap<String, String>, key: &str) -> Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    values
        .get(key)
        .map(|value| {
            value
                .trim()
                .parse::<T>()
                .map_err(|err| anyhow!("invalid value for [{section}].{key}: {err}"))
        })
        .transpose()
}

/// Two tenants sharing a key, or a tenant sharing the server's own key.
///
/// A startup error rather than a first-match win: the same token would
/// authenticate as two identities, so every limit and every usage figure would
/// be attributed to whichever one the lookup happened to reach — which is not
/// a policy, it is a coin flip that looks like one.
fn reject_ambiguous_keys(tenants: &[TenantConfig], api_key: Option<&str>) -> Result<()> {
    for (index, tenant) in tenants.iter().enumerate() {
        if Some(tenant.api_key.as_str()) == api_key {
            return Err(anyhow!(
                "[{TENANT_PREFIX}{}] uses the same key as [{SERVER_SECTION}].api_key — \
                 a request presenting it could be either",
                tenant.name
            ));
        }
        if let Some(other) = tenants[..index]
            .iter()
            .find(|other| other.api_key == tenant.api_key)
        {
            return Err(anyhow!(
                "[{TENANT_PREFIX}{}] and [{TENANT_PREFIX}{}] share an api_key — \
                 a request presenting it could be either",
                other.name,
                tenant.name
            ));
        }
    }
    Ok(())
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

    /// A config file, written and parsed, so the section-name handling is
    /// exercised rather than the tenant parser in isolation.
    fn load(contents: &str) -> Result<ServerConfiguration> {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        write!(file, "{contents}").unwrap();
        load_server_configuration(file.path(), None, false)
    }

    #[test]
    fn tenants_are_read_with_their_limits_and_sorted_by_name() {
        let conf = load(
            "[orangu-server]\nmodels = /srv/models\n\n\
             [tenant:web]\napi_key = web-key\nmax_concurrent = 2\n\
             requests_per_minute = 60\ntokens_per_minute = 20000\n\n\
             [tenant:ci]\napi_key = ci-key\n",
        )
        .unwrap();
        assert_eq!(
            conf.tenants
                .iter()
                .map(|t| t.name.as_str())
                .collect::<Vec<_>>(),
            vec!["ci", "web"]
        );
        // A tenant declared with nothing but a key is authentication only.
        assert_eq!(conf.tenants[0].api_key, "ci-key");
        assert_eq!(conf.tenants[0].limits, TenantLimits::default());
        assert_eq!(
            conf.tenants[1].limits,
            TenantLimits {
                max_concurrent: 2,
                requests_per_minute: 60,
                tokens_per_minute: 20000,
            }
        );
    }

    /// Every section that is not `[orangu-server]` or `[web]` is read as an
    /// MCP server, so a tenant section has to be taken out first. Without
    /// that, declaring a tenant fails with "endpoint must be set for an MCP
    /// server" — an error about a key the operator never wrote.
    #[test]
    fn a_tenant_section_is_not_mistaken_for_an_mcp_server() {
        let conf = load(
            "[orangu-server]\nmodels = /srv/models\n\n\
             [tenant:alice]\napi_key = alice-key\n\n\
             [weather]\nendpoint = http://localhost:9000\n",
        )
        .unwrap();
        assert_eq!(conf.tenants.len(), 1);
        assert_eq!(conf.mcp_servers.len(), 1);
        assert_eq!(conf.mcp_servers[0].name, "weather");
    }

    #[test]
    fn a_tenant_without_a_key_is_a_startup_error() {
        let err =
            load("[orangu-server]\nmodels = /srv/models\n\n[tenant:alice]\nmax_concurrent = 1\n")
                .unwrap_err()
                .to_string();
        assert!(err.contains("[tenant:alice]"), "{err}");
        assert!(err.contains("api_key"), "{err}");
    }

    /// A typo in a limit is a limit that silently does nothing — which is the
    /// failure the whole key exists to prevent.
    #[test]
    fn an_unknown_key_in_a_tenant_section_names_itself() {
        let err = load(
            "[orangu-server]\nmodels = /srv/models\n\n\
             [tenant:alice]\napi_key = k\nmax_concurent = 2\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("max_concurent"), "{err}");
        assert!(err.contains("max_concurrent"), "{err}");
    }

    #[test]
    fn a_limit_that_is_not_a_number_names_the_key() {
        let err = load(
            "[orangu-server]\nmodels = /srv/models\n\n\
             [tenant:alice]\napi_key = k\ntokens_per_minute = lots\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("[tenant:alice].tokens_per_minute"), "{err}");
    }

    /// Two identities behind one token is not a policy, it is a coin flip —
    /// every limit and every usage figure would land on whichever tenant the
    /// lookup happened to reach.
    #[test]
    fn two_tenants_may_not_share_a_key() {
        let err = load(
            "[orangu-server]\nmodels = /srv/models\n\n\
             [tenant:alice]\napi_key = same\n\n[tenant:bob]\napi_key = same\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("alice") && err.contains("bob"), "{err}");
    }

    #[test]
    fn a_tenant_may_not_share_the_servers_own_key() {
        let err = load(
            "[orangu-server]\nmodels = /srv/models\napi_key = shared\n\n\
             [tenant:alice]\napi_key = shared\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("[orangu-server].api_key"), "{err}");
    }

    /// Multi-tenancy must not force secrets onto disk — the property
    /// `ORANGU_API_KEY` exists to preserve for the single-key case.
    #[test]
    fn a_tenant_key_can_come_from_the_environment() {
        // Scoped to this test's own variable name so it cannot collide with
        // another test's environment.
        let variable = "ORANGU_TEST_TENANT_KEY_ALICE";
        // SAFETY: single-threaded setup for this test's own variable.
        unsafe { std::env::set_var(variable, "from-the-env") };
        let conf = load(&format!(
            "[orangu-server]\nmodels = /srv/models\n\n\
             [tenant:alice]\napi_key_env = {variable}\n"
        ))
        .unwrap();
        assert_eq!(conf.tenants[0].api_key, "from-the-env");

        unsafe { std::env::remove_var(variable) };
        let err = load(&format!(
            "[orangu-server]\nmodels = /srv/models\n\n\
             [tenant:alice]\napi_key_env = {variable}\n"
        ))
        .unwrap_err()
        .to_string();
        // An unset variable stops the server naming it, rather than leaving a
        // tenant nobody can authenticate as or a server nobody has to.
        assert!(err.contains(variable), "{err}");
    }

    #[test]
    fn a_tenant_may_not_spell_its_key_twice() {
        let err = load(
            "[orangu-server]\nmodels = /srv/models\n\n\
             [tenant:alice]\napi_key = k\napi_key_env = SOMEWHERE\n",
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("api_key_env"), "{err}");
    }

    #[test]
    fn a_config_with_no_tenant_section_declares_none() {
        let conf = load("[orangu-server]\nmodels = /srv/models\n").unwrap();
        assert!(conf.tenants.is_empty());
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
