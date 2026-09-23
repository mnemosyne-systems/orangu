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

//! Configuration for `orangu-coordinator`: a single `[orangu-coordinator]`
//! client section (which also names the shared models directory every
//! profile's `orangu-server` is started against) plus one section per
//! `orangu-server`-backed profile, mirroring the shape of `orangu.conf`'s
//! server sections.

use anyhow::{Context, Result, anyhow};
use orangu::config::parse_ini_sections;
use orangu::logging::LogTarget;
use std::{collections::HashMap, path::Path, path::PathBuf};

pub const CLIENT_SECTION: &str = "orangu-coordinator";

/// The `host` value meaning "every network interface on this machine" — the
/// default, and what `--init`'s `host` prompts offer first. `*` is accepted
/// as an alias for it. Mirrors `orangu-server`'s own `config::HOST_ALL`/
/// `HOST_ALL_ALIAS` (duplicated for the same reason [`default_profile_port`]
/// is: each binary's config module stands on its own) so a coordinator
/// config's `host` accepts exactly the spellings an `orangu-server` config's
/// does — including in the per-profile `host` this coordinator writes
/// straight into the `orangu-server.conf` it generates.
pub const HOST_ALL: &str = "all";
pub const HOST_ALL_ALIAS: &str = "*";

/// Whether `host` is the wildcard — [`HOST_ALL`] or its `*` alias.
fn is_host_all(host: &str) -> bool {
    let host = host.trim();
    host.eq_ignore_ascii_case(HOST_ALL) || host == HOST_ALL_ALIAS
}

/// Turns a configured `host` into an address [`std::net::TcpListener::bind`]
/// actually understands: the wildcard becomes `0.0.0.0`, so the listener
/// answers on every interface rather than only the loopback one; anything
/// else is passed through untouched and left for `bind` itself to reject if
/// it isn't one of this machine's. Same rule — and the same spelling —
/// `orangu-server`'s own `config::resolve_bind_host` applies.
pub fn resolve_bind_host(host: &str) -> &str {
    if is_host_all(host) {
        "0.0.0.0"
    } else {
        host.trim()
    }
}

/// Turns a configured `host` into an address a *client* can connect to,
/// which the wildcard is not: `all` (or `*`) says where a profile's
/// `orangu-server` listens, not where to reach it, so a URL built from it
/// has to name a concrete address. Loopback is that address — the
/// coordinator and the `orangu-server` it spawns are always the same
/// machine — and a wildcard listener answers on it by definition. Anything
/// else is already connectable and passes through untouched.
pub fn resolve_connect_host(host: &str) -> &str {
    if is_host_all(host) {
        "127.0.0.1"
    } else {
        host.trim()
    }
}

/// The conventional roles `orangu.conf` itself documents, in the order they
/// are listed there. Used both to report a model for every role in
/// [`CoordinatorConfiguration::models_by_role`] (not just the ones a given
/// `orangu-coordinator.conf` happens to define profiles for) and to
/// validate a profile's `role` key at load time — a role has to map to one
/// of `orangu-server`'s own `--all`/`--code`/`--review`/`--explorer`/
/// `--embedding` flags (see [`role_server_flag`]) for the profile to be
/// startable at all, so an unrecognized role is a load-time error.
pub const KNOWN_ROLES: &[&str] = &["all", "code", "review", "explorer", "embeddings"];

/// The `orangu-server` CLI flag a coordinator-profile `role` maps to. Note
/// the vocabulary mismatch this bridges: coordinator profiles (like
/// `orangu.conf` itself) use `embeddings` (plural), while `orangu-server`'s
/// own flag is `--embedding` (singular) — every other role's flag matches
/// its role name exactly. Returns `None` for anything not in
/// [`KNOWN_ROLES`].
/// Whether two roles can be served by one `orangu-server` process.
///
/// The four generation roles can: what separates them is sampling defaults
/// and reasoning suppression, and both travel with the request. `embeddings`
/// cannot be one of them, in either direction, however identical the rest of
/// the profile is — an `--embedding` server refuses the generation endpoints
/// outright, so sharing a process with a chat role would answer chat requests
/// with `501` instead of swapping to something that can serve them.
///
/// An unknown role only shares with itself. Config loading rejects those
/// (`parse_llm_profiles`), so this is the conservative reading of a value that
/// should not exist rather than a case with behaviour to define.
pub fn roles_share_a_process(a: &str, b: &str) -> bool {
    // `review` is **not** in this set, by operator decision: a review is run
    // by an `orangu-server --review`, a real process in that role, not by
    // another process answering a request labelled `review`. The two are
    // behaviourally the same — every effect of the role travels per request —
    // but which of them is running is a thing an operator reads off the
    // startup banner's `Mode` row and expects to be true.
    //
    // What it costs is a model load each way when a session alternates chat
    // and review. `all`, `code` and `explorer` still share, so ordinary work
    // never pays it.
    //
    // `embeddings` is excluded for a stronger reason: an `--embedding` server
    // refuses the generation endpoints outright, so sharing a process with a
    // chat role would answer chat with `501` instead of swapping.
    const SHARED: [&str; 3] = ["all", "code", "explorer"];
    a == b || (SHARED.contains(&a) && SHARED.contains(&b))
}

pub fn role_server_flag(role: &str) -> Option<&'static str> {
    match role {
        "all" => Some("--all"),
        "code" => Some("--code"),
        "review" => Some("--review"),
        "explorer" => Some("--explorer"),
        "embeddings" => Some("--embedding"),
        _ => None,
    }
}

#[derive(Clone, Debug)]
pub struct CoordinatorConfiguration {
    /// Host the proxy listens on: [`HOST_ALL`] (the default), its `*`
    /// alias, or a literal interface address such as `127.0.0.1`. See
    /// [`resolve_bind_host`], which is what [`Self::listen_addr`] binds.
    pub host: String,
    /// Port the proxy listens on.
    pub port: u16,
    /// Models directory forwarded to every profile's own `orangu-server`
    /// (its `[orangu-server].models` key) — one shared directory for every
    /// profile, matching how a single machine typically has one model
    /// cache regardless of how many roles are configured.
    pub models: PathBuf,
    /// How long to wait for a newly started `orangu-server` to answer
    /// `GET /v1/models` before giving up and reporting an error to the
    /// caller.
    pub startup_timeout_seconds: u64,
    /// Request/response body size cap in bytes.
    pub max_body_bytes: usize,
    /// How long to wait before unloading an idle model. Defaults to None (disabled).
    pub idle_timeout_seconds: Option<u64>,
    /// Shared secret required to use `GET /v1/coordinator/shutdown`. When
    /// absent the endpoint is disabled entirely.
    pub shutdown_token: Option<String>,
    /// `log_type`/`log_path`: where this coordinator's own output goes —
    /// the console (the default) or a file. Forwarded into every profile's
    /// generated `orangu-server.conf` too (see `process::write_server_config`),
    /// so a coordinator logging to a file has its servers logging to the
    /// same one rather than into a pipe nobody reads. Read once at startup:
    /// the logger is installed before the listener is even bound, so a
    /// reload that changes these keys takes effect on the next start.
    pub log: LogTarget,
    pub llms: HashMap<String, CoordinatorLlmEntry>,
    /// Name of the section whose `role` is `all`; used whenever a request's
    /// `model` field is absent or matches no configured entry.
    pub default_entry: String,
}

impl CoordinatorConfiguration {
    /// The `host:port` string to bind the proxy's listener to, with
    /// [`HOST_ALL`] (and its `*` alias) resolved to the wildcard address
    /// `bind` understands.
    pub fn listen_addr(&self) -> String {
        format!("{}:{}", resolve_bind_host(&self.host), self.port)
    }

    /// The model each of [`KNOWN_ROLES`] resolves to: the model of the
    /// lexicographically-first profile tagged with that role (matching how
    /// [`crate::process::Coordinator::resolve_entry`] breaks ties), or, when
    /// no profile defines that role, the `all`-role default's model.
    pub fn models_by_role(&self) -> Vec<(&str, &str)> {
        let default_model = self.llms[&self.default_entry].model.as_str();
        KNOWN_ROLES
            .iter()
            .map(|&role| {
                let mut candidates: Vec<&CoordinatorLlmEntry> = self
                    .llms
                    .values()
                    .filter(|entry| entry.role == role)
                    .collect();
                candidates.sort_unstable_by(|a, b| a.name.cmp(&b.name));
                let model = candidates
                    .first()
                    .map(|entry| entry.model.as_str())
                    .unwrap_or(default_model);
                (role, model)
            })
            .collect()
    }
}

/// One configured profile: which role it serves, which model, and where its
/// own `orangu-server` should listen. Every field here is an explicit,
/// individually-parsed and -validated config key — `orangu-coordinator`
/// builds `orangu-server`'s own argv itself from these (see
/// `process::Coordinator::start`), rather than scraping a model id or a
/// host/port back out of a free-form command line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoordinatorLlmEntry {
    pub name: String,
    pub role: String,
    /// A model spec in the same shape `orangu-server`'s own positional
    /// `MODEL` CLI argument accepts: a local `.gguf` path, an `NR`/`MODEL`
    /// label already under the shared `models` directory, or a
    /// `<user>/<model>[:quant]` Hugging Face repo (fetched on first start if
    /// not already cached). This is the model id a client request must
    /// carry (in its JSON `model` field) to be routed to this entry.
    pub model: String,
    /// Host this profile's `orangu-server` will listen on — written
    /// verbatim into the `orangu-server.conf` this coordinator generates for
    /// it, so it accepts every spelling that file's own `host` does.
    /// Defaults to [`HOST_ALL`] when absent, the same default
    /// `orangu-server` itself applies. [`Self::origin`] resolves it back to
    /// a connectable address.
    pub host: String,
    /// Port this profile's `orangu-server` will listen on. Defaults to
    /// `8100` — the same default `orangu-server` itself uses — when absent.
    /// Unlike a real multi-process setup, profiles sharing a host and port
    /// is fine even though only one is ever active at a time: at most one
    /// `orangu-server` is alive under a coordinator (this project's whole
    /// invariant), and `process::Coordinator::ensure_active` always fully
    /// stops whichever one is currently running — awaited, not just
    /// signaled — before starting a different profile's own, so the new
    /// process never races the old one for the same port.
    pub port: u16,
    /// Forwarded verbatim to this profile's generated `[orangu-server].
    /// backend` key when set — `orangu-server` itself validates the value
    /// (`auto`/`cpu`/`vulkan`/`metal`/`cuda`/`opencl`/`rocm`), so this isn't
    /// re-validated here. `None` leaves `orangu-server`'s own default
    /// (`auto`) in place by simply omitting the key.
    pub backend: Option<String>,
    /// Forwarded to this profile's generated `[orangu-server].slots` key
    /// when set. `None` leaves `orangu-server`'s own role-based default in
    /// place (see `orangu-server`'s `config::Role::default_slots`).
    pub slots: Option<usize>,
    /// Forwarded to this profile's generated `[web].port` key when
    /// set. `None` leaves `orangu-server`'s own default (`0`, disabled) in
    /// place — a coordinator-managed profile has no obvious single "the"
    /// web UI port across every role, so this is opt-in per profile only.
    pub web: Option<u16>,
}

impl CoordinatorLlmEntry {
    /// Whether an `orangu-server` started for `other` is already serving what
    /// this profile asks for — the same model file under the same listener and
    /// the same device settings, differing at most in `role`.
    ///
    /// A role is not a property of the process it needs to be. It selects
    /// sampling defaults and whether reasoning is suppressed, both of which
    /// `orangu-server` will take per request (`x-orangu-role`). Everything
    /// else here *is* baked into the process: the model it loaded, the address
    /// it bound, the backend it chose, the slots it sized its pools for.
    ///
    /// Moving between two such profiles keeps the running process: no
    /// identical weights are re-read, and its cached prefixes survive.
    pub fn serves_same_process_as(&self, other: &Self) -> bool {
        roles_share_a_process(&self.role, &other.role)
            && self.model == other.model
            && self.host == other.host
            && self.port == other.port
            && self.backend == other.backend
            && self.slots == other.slots
            && self.web == other.web
    }

    /// The origin (`http://host:port`) requests are proxied to once this
    /// entry's `orangu-server` is active, with [`HOST_ALL`] (and its `*`
    /// alias) resolved to loopback — see [`resolve_connect_host`] for why a
    /// wildcard can't be dialed as written.
    pub fn origin(&self) -> String {
        format!("http://{}:{}", resolve_connect_host(&self.host), self.port)
    }
}

pub(crate) fn default_startup_timeout() -> u64 {
    180
}

pub(crate) fn default_max_body_bytes() -> usize {
    64 * 1024 * 1024
}

/// The `host` both the coordinator's own listener and every profile's
/// `orangu-server` fall back to — [`HOST_ALL`], the same default
/// `orangu-server`'s own `config::default_host` uses, so a coordinator
/// config that says nothing about `host` is reachable from exactly where an
/// `orangu-server` config that says nothing about it is.
pub(crate) fn default_host() -> String {
    HOST_ALL.to_string()
}

pub(crate) fn default_port() -> u16 {
    9000
}

/// Default `port` for a profile's own `orangu-server` — the same default
/// `orangu-server`'s own `config::default_port` uses, not to be confused
/// with [`default_port`] above (the coordinator's own listen port).
pub(crate) fn default_profile_port() -> u16 {
    8100
}

/// Parses an optional `u16` key, returning `None` — never an error — for
/// both a genuinely absent key *and* one present but left blank (`port = `
/// with nothing after the `=`): `parse_ini_sections` inserts a blank value
/// into its map same as any other, so `values.get(name)` alone can't tell
/// "not set" apart from "set to nothing" — every other optional key in this
/// file (`host`, `role`, `backend`, `shutdown_token`, ...) already treats
/// them the same via `.filter(|v| !v.is_empty())`; this is that same
/// guarantee for numeric keys, so a stray trailing `=` falls back to
/// whatever default the caller applies instead of surfacing as a confusing
/// "invalid digit found in string" parse error.
fn parse_port(values: &HashMap<String, String>, name: &str, section: &str) -> Result<Option<u16>> {
    match values.get(name).map(|v| v.trim()).filter(|v| !v.is_empty()) {
        Some(value) => value
            .parse::<u16>()
            .map(Some)
            .map_err(|err| anyhow!("invalid value for [{section}].{name}: {err}")),
        None => Ok(None),
    }
}

/// Expands a leading `~` or `~/` to the user's home directory — same
/// convenience `orangu-server`'s own config applies to its `models` key,
/// mirrored here so a coordinator config can use the same shorthand.
fn expand_tilde(path: &str) -> PathBuf {
    match path.strip_prefix('~') {
        Some(rest) => match home::home_dir() {
            Some(home) => home.join(rest.trim_start_matches('/')),
            None => PathBuf::from(path),
        },
        None => PathBuf::from(path),
    }
}

pub fn default_coordinator_config_path() -> Option<PathBuf> {
    let cwd_path = std::env::current_dir()
        .ok()?
        .join("orangu-coordinator.conf");
    if cwd_path.exists() {
        return Some(cwd_path);
    }

    let config_path = home::home_dir()?.join(".orangu/orangu-coordinator.conf");
    config_path.exists().then_some(config_path)
}

pub fn load_coordinator_configuration(path: &Path) -> Result<CoordinatorConfiguration> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read configuration {}", path.display()))?;
    let mut sections = parse_ini_sections(&contents)
        .with_context(|| format!("failed to parse configuration {}", path.display()))?;

    let client = sections.remove(CLIENT_SECTION).unwrap_or_default();

    let host = client
        .get("host")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(default_host);
    let port = parse_port(&client, "port", CLIENT_SECTION)?.unwrap_or_else(default_port);

    let models = client
        .get("models")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .ok_or_else(|| anyhow!("[{CLIENT_SECTION}].models must be set to a models directory"))?;

    // `.filter(!is_empty())` on every one of these (matching every string
    // key elsewhere in this function, e.g. `host`/`models` above) makes a
    // present-but-blank value (`startup_timeout = ` with nothing after the
    // `=`) fall back to its default exactly like an absent key would,
    // rather than surfacing as a confusing parse error — see `parse_port`'s
    // own doc comment for why `parse_ini_sections` makes this distinction
    // necessary to handle explicitly.
    let startup_timeout_seconds = match client
        .get("startup_timeout")
        .map(|v| v.trim())
        .filter(|v| !v.is_empty())
    {
        Some(value) => value.parse::<u64>().map_err(|err| {
            anyhow!("invalid value for [{CLIENT_SECTION}].startup_timeout: {err}")
        })?,
        None => default_startup_timeout(),
    };

    let max_body_bytes = match client
        .get("max_body_bytes")
        .map(|v| v.trim())
        .filter(|v| !v.is_empty())
    {
        Some(value) => value
            .parse::<usize>()
            .map_err(|err| anyhow!("invalid value for [{CLIENT_SECTION}].max_body_bytes: {err}"))?,
        None => default_max_body_bytes(),
    };

    let idle_timeout_seconds =
        match client
            .get("idle_timeout")
            .map(|v| v.trim())
            .filter(|v| !v.is_empty())
        {
            Some(value) => Some(value.parse::<u64>().map_err(|err| {
                anyhow!("invalid value for [{CLIENT_SECTION}].idle_timeout: {err}")
            })?),
            None => None,
        };

    let shutdown_token = client
        .get("shutdown_token")
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());

    let log = LogTarget::from_keys(
        CLIENT_SECTION,
        client.get("log_type").map(String::as_str),
        client.get("log_path").map(String::as_str),
    )?;

    if sections.is_empty() {
        return Err(anyhow!("At least one named LLM profile must be defined"));
    }

    let llms = parse_llm_profiles(sections)?;

    let default_entry = {
        let mut all_entries: Vec<&str> = llms
            .values()
            .filter(|entry| entry.role == "all")
            .map(|entry| entry.name.as_str())
            .collect();
        all_entries.sort_unstable();
        all_entries.first().map(|name| name.to_string()).ok_or_else(|| {
            anyhow!(
                "At least one profile must specify (or default to) role = all, to serve as the fallback when a request's model doesn't match a specific profile"
            )
        })?
    };

    Ok(CoordinatorConfiguration {
        host,
        port,
        models: expand_tilde(&models),
        startup_timeout_seconds,
        max_body_bytes,
        idle_timeout_seconds,
        shutdown_token,
        log,
        llms,
        default_entry,
    })
}

fn parse_llm_profiles(
    sections: HashMap<String, HashMap<String, String>>,
) -> Result<HashMap<String, CoordinatorLlmEntry>> {
    sections
        .into_iter()
        .map(|(name, values)| {
            // A section named after a role (`[code]`) is that role; any other
            // name takes its `role` key, defaulting to `all`.
            let named_role = KNOWN_ROLES
                .iter()
                .find(|role| name.eq_ignore_ascii_case(role))
                .map(|role| role.to_string());
            let role_key = values
                .get("role")
                .map(|value| value.trim().to_lowercase())
                .filter(|value| !value.is_empty());
            let role = match (named_role, role_key) {
                (Some(named), Some(key)) if named != key => {
                    return Err(anyhow!(
                        "[{name}].role '{key}' contradicts the section name; remove the role key or rename the section"
                    ));
                }
                (Some(named), _) => named,
                (None, Some(key)) => key,
                (None, None) => "all".to_string(),
            };
            if role_server_flag(&role).is_none() {
                return Err(anyhow!(
                    "[{name}].role '{role}' is not a known role (expected one of: {})",
                    KNOWN_ROLES.join(", ")
                ));
            }

            let model = values
                .get("model")
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .ok_or_else(|| anyhow!("[{name}].model must not be empty"))?;

            let host = values
                .get("host")
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(default_host);

            let port = parse_port(&values, "port", &name)?.unwrap_or_else(default_profile_port);

            let backend = values
                .get("backend")
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty());

            let slots = match values
                .get("slots")
                .map(|v| v.trim())
                .filter(|v| !v.is_empty())
            {
                Some(value) => Some(
                    value
                        .parse::<usize>()
                        .map_err(|err| anyhow!("invalid value for [{name}].slots: {err}"))?,
                ),
                None => None,
            };

            let web = parse_port(&values, "web", &name)?;

            Ok((
                name.clone(),
                CoordinatorLlmEntry {
                    name,
                    role,
                    model,
                    host,
                    port,
                    backend,
                    slots,
                    web,
                },
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn loads_minimal_configuration_with_defaults() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nmodels = /srv/models\n\n[main]\nrole = all\nmodel = org/gemma\nport = 8100\n"
        )
        .unwrap();

        let conf = load_coordinator_configuration(file.path()).unwrap();
        // `host` is omitted above, so it falls back to `all` — resolved
        // to the wildcard address `bind` understands.
        assert_eq!(conf.listen_addr(), "0.0.0.0:9000");
        assert_eq!(conf.models, PathBuf::from("/srv/models"));
        assert_eq!(conf.startup_timeout_seconds, 180);
        assert_eq!(conf.default_entry, "main");
        assert_eq!(conf.llms["main"].host, HOST_ALL);
        // A wildcard is where it listens, not somewhere to dial: the origin
        // resolves it back to loopback.
        assert_eq!(conf.llms["main"].origin(), "http://127.0.0.1:8100");
        assert_eq!(conf.llms["main"].model, "org/gemma");
        assert_eq!(conf.llms["main"].backend, None);
        assert_eq!(conf.llms["main"].slots, None);
        assert_eq!(conf.llms["main"].web, None);
    }

    /// Every optional key in both the `[orangu-coordinator]` client section
    /// and a profile section, left out entirely — not just one at a time
    /// like the other tests in this module each check, but all of them
    /// simultaneously — still loads and falls back to every documented
    /// default, with nothing left unset that shouldn't be.
    #[test]
    fn every_optional_key_defaults_when_the_whole_file_omits_them() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nmodels = /srv/models\n\n[main]\nmodel = org/gemma\n"
        )
        .unwrap();

        let conf = load_coordinator_configuration(file.path()).unwrap();
        assert_eq!(conf.host, HOST_ALL);
        assert_eq!(conf.port, 9000);
        assert_eq!(conf.startup_timeout_seconds, 180);
        assert_eq!(conf.max_body_bytes, 64 * 1024 * 1024);
        assert_eq!(conf.idle_timeout_seconds, None);
        assert_eq!(conf.shutdown_token, None);
        assert_eq!(conf.log, LogTarget::Console);

        let main = &conf.llms["main"];
        assert_eq!(main.role, "all");
        assert_eq!(main.host, HOST_ALL);
        assert_eq!(main.port, 8100);
        assert_eq!(main.backend, None);
        assert_eq!(main.slots, None);
        assert_eq!(main.web, None);
    }

    /// A key that's *present* but left blank (`key = ` with nothing after
    /// the `=`, as opposed to the key being absent entirely) must fall back
    /// to the exact same default an absent key would — not surface as an
    /// "invalid digit found in string" parse error. `parse_ini_sections`
    /// stores a blank value the same as any other, so this has to be
    /// handled explicitly (see `parse_port`'s own doc comment) — covers
    /// every numeric key across both sections, since `port`/`slots`/
    /// `startup_timeout`/`max_body_bytes`/`idle_timeout` each had their own
    /// separate parsing before being unified under the same guard.
    #[test]
    fn a_blank_but_present_value_defaults_the_same_as_an_absent_key() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nmodels = /srv/models\nport = \nstartup_timeout = \nmax_body_bytes = \nidle_timeout = \n\n[main]\nrole = all\nmodel = org/gemma\nport = \nslots = \n"
        )
        .unwrap();

        let conf = load_coordinator_configuration(file.path()).unwrap();
        assert_eq!(conf.port, 9000);
        assert_eq!(conf.startup_timeout_seconds, 180);
        assert_eq!(conf.max_body_bytes, 64 * 1024 * 1024);
        assert_eq!(conf.idle_timeout_seconds, None);
        assert_eq!(conf.llms["main"].port, 8100);
        assert_eq!(conf.llms["main"].slots, None);
    }

    /// `log_type = file` names a file for the coordinator's own output;
    /// what the loader stores is the resolved target, so a `~` and a
    /// relative path are already usable by the time a daemon has moved to
    /// `/`.
    #[test]
    fn loads_the_log_keys() {
        // Spelled from the current directory so the path is absolute on
        // every platform — a bare `/var/...` has no drive on Windows.
        let log_path = std::env::current_dir()
            .unwrap()
            .join("var")
            .join("orangu-coordinator.log");
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nmodels = /srv/models\nlog_type = file\nlog_path = {}\n\n[main]\nmodel = org/gemma\n",
            log_path.display()
        )
        .unwrap();

        let conf = load_coordinator_configuration(file.path()).unwrap();
        assert_eq!(conf.log, LogTarget::File(log_path.clone()));

        // The console needs no path, and keeps one that happens to be there.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nmodels = /srv/models\nlog_type = console\nlog_path = {}\n\n[main]\nmodel = org/gemma\n",
            log_path.display()
        )
        .unwrap();
        assert_eq!(
            load_coordinator_configuration(file.path()).unwrap().log,
            LogTarget::Console
        );
    }

    /// `log_type = file` with no `log_path` logs to `orangu-coordinator.log`
    /// in the directory the coordinator was started from; a `log_type` that
    /// is neither of the two is a config error — the point of the key is to
    /// have the output somewhere in particular, so a silent console is the
    /// one answer that must not happen.
    #[test]
    fn a_file_log_defaults_its_path_and_an_unknown_log_type_is_rejected() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nmodels = /srv/models\nlog_type = file\n\n[main]\nmodel = org/gemma\n"
        )
        .unwrap();
        assert_eq!(
            load_coordinator_configuration(file.path()).unwrap().log,
            LogTarget::File(
                std::env::current_dir()
                    .unwrap()
                    .join("orangu-coordinator.log")
            )
        );

        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nmodels = /srv/models\nlog_type = syslog\n\n[main]\nmodel = org/gemma\n"
        )
        .unwrap();
        let err = load_coordinator_configuration(file.path()).unwrap_err();
        assert!(
            err.to_string()
                .contains("invalid value for [orangu-coordinator].log_type"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn requires_models_key() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\n\n[main]\nrole = all\nmodel = org/gemma\nport = 8100\n"
        )
        .unwrap();

        let err = load_coordinator_configuration(file.path()).unwrap_err();
        assert!(
            err.to_string().contains("models"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn expands_leading_tilde_in_models_path() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nmodels = ~/models\n\n[main]\nrole = all\nmodel = org/gemma\nport = 8100\n"
        )
        .unwrap();

        let conf = load_coordinator_configuration(file.path()).unwrap();
        let home = home::home_dir().unwrap();
        assert_eq!(conf.models, home.join("models"));
    }

    #[test]
    fn requires_at_least_one_all_role() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nmodels = /srv/models\n\n[explorer]\nrole = explorer\nmodel = org/qwen\nport = 8200\n"
        )
        .unwrap();

        let err = load_coordinator_configuration(file.path()).unwrap_err();
        assert!(
            err.to_string().contains("role = all"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn role_defaults_to_all() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nmodels = /srv/models\n\n[main]\nmodel = org/gemma\nport = 8100\n"
        )
        .unwrap();

        let conf = load_coordinator_configuration(file.path()).unwrap();
        assert_eq!(conf.llms["main"].role, "all");
        assert_eq!(conf.default_entry, "main");
    }

    fn load(contents: &str) -> Result<CoordinatorConfiguration> {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        write!(file, "{contents}").unwrap();
        load_coordinator_configuration(file.path())
    }

    #[test]
    fn a_section_named_after_a_role_is_that_role() {
        let conf = load(
            "[orangu-coordinator]\nmodels = /srv/models\n\n[all]\nmodel = org/gemma\n\n[code]\nmodel = org/gemma\n\n[Embeddings]\nmodel = org/embed\n\n[qwen]\nrole = explorer\nmodel = org/qwen\n",
        )
        .unwrap();
        assert_eq!(conf.llms["all"].role, "all");
        assert_eq!(conf.llms["code"].role, "code");
        assert_eq!(conf.llms["Embeddings"].role, "embeddings");
        assert_eq!(conf.llms["qwen"].role, "explorer");
        assert_eq!(conf.default_entry, "all");
    }

    #[test]
    fn a_role_key_matching_its_section_name_is_accepted() {
        let conf = load(
            "[orangu-coordinator]\nmodels = /srv/models\n\n[all]\nmodel = org/gemma\n\n[code]\nrole = code\nmodel = org/gemma\n",
        )
        .unwrap();
        assert_eq!(conf.llms["code"].role, "code");
    }

    #[test]
    fn a_role_key_contradicting_its_section_name_is_rejected() {
        let err = load(
            "[orangu-coordinator]\nmodels = /srv/models\n\n[all]\nmodel = org/gemma\n\n[code]\nrole = review\nmodel = org/gemma\n",
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("contradicts the section name"),
            "{err}"
        );
    }

    #[test]
    fn rejects_an_unknown_role() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nmodels = /srv/models\n\n[main]\nrole = all\nmodel = org/gemma\nport = 8100\n\n[weird]\nrole = summarizer\nmodel = org/qwen\nport = 8200\n"
        )
        .unwrap();

        let err = load_coordinator_configuration(file.path()).unwrap_err();
        assert!(
            err.to_string().contains("not a known role"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn allows_profiles_sharing_a_model() {
        // Two profiles may reference the same model (e.g. one per role, even
        // when the underlying model happens to be identical); this is not an
        // error, just an ambiguity `resolve_entry` breaks deterministically.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nmodels = /srv/models\n\n[a]\nrole = all\nmodel = org/gemma\nport = 8100\n\n[b]\nrole = explorer\nmodel = org/gemma\nport = 8200\n"
        )
        .unwrap();

        let conf = load_coordinator_configuration(file.path()).unwrap();
        assert_eq!(conf.llms["a"].model, "org/gemma");
        assert_eq!(conf.llms["b"].model, "org/gemma");
    }

    #[test]
    fn rejects_a_profile_without_a_model() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nmodels = /srv/models\n\n[main]\nrole = all\nport = 8100\n"
        )
        .unwrap();

        let err = load_coordinator_configuration(file.path()).unwrap_err();
        assert!(
            err.to_string().contains("model"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn port_defaults_to_8100_when_a_profile_omits_it() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nmodels = /srv/models\n\n[main]\nrole = all\nmodel = org/gemma\n"
        )
        .unwrap();

        let conf = load_coordinator_configuration(file.path()).unwrap();
        assert_eq!(conf.llms["main"].port, 8100);
    }

    /// Two profiles sharing the same default `host`/`port` is fine, not an
    /// error at load time — see `CoordinatorLlmEntry::port`'s own doc
    /// comment for why (only one `orangu-server` is ever active, and
    /// swapping always fully stops the old one first).
    #[test]
    fn two_profiles_may_share_the_default_host_and_port() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nmodels = /srv/models\n\n[main]\nrole = all\nmodel = org/gemma\n\n[explorer]\nrole = explorer\nmodel = org/qwen\n"
        )
        .unwrap();

        let conf = load_coordinator_configuration(file.path()).unwrap();
        assert_eq!(conf.llms["main"].origin(), conf.llms["explorer"].origin());
    }

    #[test]
    fn rejects_an_invalid_port() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nmodels = /srv/models\n\n[main]\nrole = all\nmodel = org/gemma\nport = not-a-port\n"
        )
        .unwrap();

        let err = load_coordinator_configuration(file.path()).unwrap_err();
        assert!(
            err.to_string().contains("invalid value for [main].port"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn host_defaults_when_a_profile_omits_it() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nmodels = /srv/models\n\n[main]\nrole = all\nmodel = org/gemma\nport = 8100\n"
        )
        .unwrap();

        let conf = load_coordinator_configuration(file.path()).unwrap();
        assert_eq!(conf.llms["main"].host, HOST_ALL);
    }

    /// `all`, its `*` alias, and any casing of either all mean the same
    /// listener — and none of them is a spelling anything can dial, so a
    /// URL built from one names loopback instead. An ordinary address is
    /// untouched by both.
    #[test]
    fn resolves_the_wildcard_for_binding_and_for_connecting() {
        for wildcard in [HOST_ALL, HOST_ALL_ALIAS, "ALL", " all "] {
            assert_eq!(resolve_bind_host(wildcard), "0.0.0.0");
            assert_eq!(resolve_connect_host(wildcard), "127.0.0.1");
        }
        assert_eq!(resolve_bind_host("192.168.1.10"), "192.168.1.10");
        assert_eq!(resolve_connect_host("192.168.1.10"), "192.168.1.10");
    }

    /// A config that spells its `host` `*` binds and dials exactly as one
    /// that spells it `all` does — including for a profile, whose `host`
    /// travels on into the generated `orangu-server.conf`.
    #[test]
    fn the_star_alias_behaves_exactly_like_all() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nhost = *\nmodels = /srv/models\n\n[main]\nrole = all\nmodel = org/gemma\nhost = *\n"
        )
        .unwrap();

        let conf = load_coordinator_configuration(file.path()).unwrap();
        assert_eq!(conf.listen_addr(), "0.0.0.0:9000");
        // Written through verbatim — `orangu-server` accepts `*` too.
        assert_eq!(conf.llms["main"].host, "*");
        assert_eq!(conf.llms["main"].origin(), "http://127.0.0.1:8100");
    }

    #[test]
    fn parses_multiple_roles_with_distinct_hosts_and_optional_overrides() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nhost = 0.0.0.0\nport = 9100\nmodels = /srv/models\nstartup_timeout = 30\n\n[main]\nrole = all\nmodel = org/gemma\nport = 8100\n\n[explorer]\nrole = explorer\nmodel = org/qwen\nhost = 192.168.1.20\nport = 8200\nbackend = vulkan\nslots = 4\nweb = 8281\n"
        )
        .unwrap();

        let conf = load_coordinator_configuration(file.path()).unwrap();
        assert_eq!(conf.listen_addr(), "0.0.0.0:9100");
        assert_eq!(conf.startup_timeout_seconds, 30);
        assert_eq!(conf.llms.len(), 2);
        assert_eq!(conf.llms["main"].host, HOST_ALL);
        assert_eq!(conf.llms["explorer"].origin(), "http://192.168.1.20:8200");
        assert_eq!(conf.llms["explorer"].model, "org/qwen");
        assert_eq!(conf.llms["explorer"].backend.as_deref(), Some("vulkan"));
        assert_eq!(conf.llms["explorer"].slots, Some(4));
        assert_eq!(conf.llms["explorer"].web, Some(8281));
    }

    #[test]
    fn models_by_role_falls_back_to_all_for_undefined_roles() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nmodels = /srv/models\n\n[main]\nrole = all\nmodel = org/gemma\nport = 8100\n\n[explorer]\nrole = explorer\nmodel = org/qwen\nport = 8200\n"
        )
        .unwrap();

        let conf = load_coordinator_configuration(file.path()).unwrap();
        let models: std::collections::HashMap<&str, &str> =
            conf.models_by_role().into_iter().collect();
        assert_eq!(models.len(), KNOWN_ROLES.len());
        assert_eq!(models["all"], "org/gemma");
        assert_eq!(models["explorer"], "org/qwen");
        // code/review/embeddings have no profile of their own — fall back to
        // the `all`-role default's model.
        assert_eq!(models["code"], "org/gemma");
        assert_eq!(models["review"], "org/gemma");
        assert_eq!(models["embeddings"], "org/gemma");
    }

    /// The distinction the coordinator turns a model reload on: two
    /// profiles that differ only in role are one process, and anything that
    /// changes what the process *is* — the model above all — is not.
    #[test]
    fn only_the_role_may_differ_for_two_profiles_to_share_a_process() {
        let base = CoordinatorLlmEntry {
            name: "all".to_string(),
            role: "all".to_string(),
            model: "org/model:Q4_K_M".to_string(),
            host: "all".to_string(),
            port: 8100,
            backend: None,
            slots: None,
            web: None,
        };

        // `all` and `code` differ in nothing a process can express, so they
        // are one.
        let code = CoordinatorLlmEntry {
            name: "code".to_string(),
            role: "code".to_string(),
            ..base.clone()
        };
        assert!(base.serves_same_process_as(&code));
        assert!(code.serves_same_process_as(&base), "and both ways round");

        // `review` is not, by operator decision: a review runs on an
        // `orangu-server --review`, which is what the banner then says.
        let review = CoordinatorLlmEntry {
            name: "review".to_string(),
            role: "review".to_string(),
            ..base.clone()
        };
        assert!(!base.serves_same_process_as(&review));
        assert!(!review.serves_same_process_as(&base));

        // An embedding server refuses the generation endpoints, so it is not
        // interchangeable with a chat role however identical the rest is.
        let embeddings = CoordinatorLlmEntry {
            name: "embeddings".to_string(),
            role: "embeddings".to_string(),
            ..base.clone()
        };
        assert!(!base.serves_same_process_as(&embeddings));
        assert!(!embeddings.serves_same_process_as(&base));
        assert!(
            embeddings.serves_same_process_as(&CoordinatorLlmEntry {
                name: "embeddings-2".to_string(),
                ..embeddings.clone()
            }),
            "two embedding profiles on one model are still one process"
        );

        // Everything else is a different process.
        for different in [
            CoordinatorLlmEntry {
                model: "org/other:Q4_K_M".to_string(),
                ..code.clone()
            },
            CoordinatorLlmEntry {
                port: 8101,
                ..code.clone()
            },
            CoordinatorLlmEntry {
                backend: Some("cpu".to_string()),
                ..code.clone()
            },
            CoordinatorLlmEntry {
                slots: Some(4),
                ..code.clone()
            },
            CoordinatorLlmEntry {
                web: Some(8200),
                ..code.clone()
            },
        ] {
            assert!(
                !base.serves_same_process_as(&different),
                "'{}' differs in more than its role",
                different.name
            );
        }
    }

    #[test]
    fn role_server_flag_bridges_the_embeddings_plural_vs_embedding_singular_mismatch() {
        assert_eq!(role_server_flag("all"), Some("--all"));
        assert_eq!(role_server_flag("code"), Some("--code"));
        assert_eq!(role_server_flag("review"), Some("--review"));
        assert_eq!(role_server_flag("explorer"), Some("--explorer"));
        assert_eq!(role_server_flag("embeddings"), Some("--embedding"));
        assert_eq!(role_server_flag("nonexistent"), None);
    }
}
