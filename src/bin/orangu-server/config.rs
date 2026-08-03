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

use anyhow::{Context, Result, anyhow};
use orangu::config::parse_ini_sections;
use std::path::{Path, PathBuf};

pub const SERVER_SECTION: &str = "orangu-server";

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

/// Which `engine::backend::Backend` to run the forward pass on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BackendPreference {
    /// Tries every GPU backend compiled into this build, otherwise falls
    /// back to the CPU backend — no error either way. The order is
    /// Vulkan, CUDA, OpenCL, then — only if built with the `rocm` Cargo
    /// feature — ROCm, except on Apple targets, where Metal is tried
    /// first: it is the only GPU API macOS actually ships, and it reaches
    /// the same kernels (see `main.rs::select_backend`).
    #[default]
    Auto,
    Cpu,
    /// Fail to start (rather than silently falling back) if no Vulkan
    /// adapter is found — for when GPU inference was specifically asked
    /// for and silently running on the CPU instead would be surprising.
    Vulkan,
    /// Same fail-loudly contract as `Vulkan`, for an Apple Metal device.
    Metal,
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
    /// Which `Backend` runs the forward pass — CPU, a named GPU API, or
    /// (the default) whichever GPU this platform finds first, falling back
    /// to CPU.
    pub backend: BackendPreference,
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
    let (web, web_host, reexec, delete) = match web_section {
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
            let web_host = web_section
                .get("host")
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| host.clone());
            let reexec = match web_section.get("reexec") {
                Some(value) => parse_bool(WEB_SECTION, "reexec", value)?,
                None => default_reexec(),
            };
            let delete = match web_section.get("delete") {
                Some(value) => parse_bool(WEB_SECTION, "delete", value)?,
                None => default_delete(),
            };
            (port, web_host, reexec, delete)
        }
        None => {
            let port = match section.get("web") {
                Some(value) => value
                    .trim()
                    .parse::<u16>()
                    .map_err(|err| anyhow!("invalid value for [{SERVER_SECTION}].web: {err}"))?,
                None => default_web(),
            };
            (port, host.clone(), default_reexec(), default_delete())
        }
    };

    let model = section
        .get("model")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    let backend = match section.get("backend") {
        Some(value) => match value.trim().to_lowercase().as_str() {
            "auto" => BackendPreference::Auto,
            "cpu" => BackendPreference::Cpu,
            "vulkan" => BackendPreference::Vulkan,
            "metal" => BackendPreference::Metal,
            "cuda" => BackendPreference::Cuda,
            "opencl" => BackendPreference::OpenCl,
            "rocm" => BackendPreference::Rocm,
            other => {
                return Err(anyhow!(
                    "invalid value for [{SERVER_SECTION}].backend: '{other}' \
                     (expected auto, cpu, vulkan, metal, cuda, opencl, or rocm)"
                ));
            }
        },
        None => default_backend(),
    };

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
        backend,
        reexec,
        delete,
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
