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

//! Loading a different model by replacing this process with a new one.
//!
//! The web console's model manager can load a model that isn't the one this
//! server started on. Rather than swapping the model *inside* the running
//! process, it `execve`s this binary again with the new model in `argv` —
//! which means the model is loaded by `main::prepare`, the same code that
//! already runs at startup, and there is never a second load path to keep in
//! step with it.
//!
//! Three properties make that a handover rather than a restart:
//!
//! - **The listening sockets survive.** Both are already bound before the
//!   exec, and their `FD_CLOEXEC` flag is cleared so they stay open across
//!   it; the new image picks them back up from their descriptors
//!   ([`INHERIT_FDS_VAR`]) instead of binding. The port is therefore never
//!   unbound, so nothing can take it in between and a client's next
//!   connection is simply queued in the listen backlog until the new image
//!   starts accepting.
//! - **The process identity survives.** `execve` replaces the image but
//!   keeps the pid, so a supervisor (systemd, a `--daemon` launcher, a shell
//!   job) goes on tracking the same process. A `--daemon` server stays
//!   detached, since the exec inherits its already-detached session.
//! - **A failed load falls back.** The previous model spec travels in
//!   [`FALLBACK_MODEL_VAR`]; if the new image can't load its model it execs
//!   once more with that, and only then gives up. Bounded to a single retry,
//!   so a genuinely broken pair can't loop.
//!
//! What does *not* survive is anything held only in memory: requests in
//! flight are cut off (which is why the handover refuses while any slot is
//! busy), and unsaved slot KV-caches are gone. Chat sessions are on disk and
//! come back untouched.
//!
//! Unix only — `execve` and descriptor inheritance have no Windows
//! equivalent here. [`supported`] says so, and the endpoint refuses rather
//! than pretending.

use anyhow::{Result, anyhow, bail};
use std::{
    ffi::OsString,
    path::{Path, PathBuf},
    sync::OnceLock,
};

use crate::config::Role;

/// Names the already-bound listening sockets handed to the new image, as
/// `api:<fd>` or `api:<fd>,web:<fd>`. Read exactly once, at the top of
/// `main` (see [`take_inherited`]), and removed from the environment there
/// so nothing downstream — a child process, a second handover — can act on a
/// stale value.
pub const INHERIT_FDS_VAR: &str = "ORANGU_INHERIT_FDS";

/// The model spec to fall back to when the newly exec'd image cannot load
/// the model it was given. Set by [`Handover::exec`] to whatever was loaded
/// before it, and *not* set on the fallback attempt itself, which is what
/// bounds the retry to one.
pub const FALLBACK_MODEL_VAR: &str = "ORANGU_FALLBACK_MODEL";

/// Whether this build can hand over at all. `false` on non-Unix, where the
/// model manager's Load button is disabled for the same reason
/// `[orangu-server].reexec = no` disables it.
pub const fn supported() -> bool {
    cfg!(unix)
}

/// Listening sockets inherited from the process this one replaced — raw
/// descriptors, not [`std::net::TcpListener`]s, so a [`prepare`] that fails
/// before it gets to them leaves them open for the fallback exec to hand on
/// again.
///
/// [`prepare`]: crate::prepare
#[derive(Clone, Copy, Debug, Default)]
pub struct InheritedFds {
    pub api: Option<i32>,
    pub web: Option<i32>,
}

static INHERITED: OnceLock<InheritedFds> = OnceLock::new();
static FALLBACK: OnceLock<Option<String>> = OnceLock::new();

/// Reads [`INHERIT_FDS_VAR`] and [`FALLBACK_MODEL_VAR`] and removes both
/// from the environment.
///
/// **Must be called from the very top of `main`,** on the only thread that
/// exists yet and before anything else could read the environment — the same
/// condition that makes `main`'s own `set_var("RUST_BACKTRACE", ...)` sound.
/// Everything afterwards reads the parsed values through [`inherited`] and
/// [`fallback_model`], never the environment again.
///
/// # Safety
///
/// The caller must guarantee no other thread exists.
pub unsafe fn take_inherited() {
    let fds = std::env::var(INHERIT_FDS_VAR)
        .ok()
        .map(|value| parse_inherit_fds(&value))
        .unwrap_or_default();
    let fallback = std::env::var(FALLBACK_MODEL_VAR)
        .ok()
        .filter(|value| !value.is_empty());
    unsafe {
        std::env::remove_var(INHERIT_FDS_VAR);
        std::env::remove_var(FALLBACK_MODEL_VAR);
    }
    let _ = INHERITED.set(fds);
    let _ = FALLBACK.set(fallback);
}

pub fn inherited() -> InheritedFds {
    INHERITED.get().copied().unwrap_or_default()
}

/// The model to try when the one this image was given fails to load, or
/// `None` — which is both "this is an ordinary start" and "this *is* already
/// the fallback attempt".
pub fn fallback_model() -> Option<&'static str> {
    FALLBACK.get().and_then(|f| f.as_deref())
}

fn parse_inherit_fds(value: &str) -> InheritedFds {
    let mut fds = InheritedFds::default();
    for part in value.split(',') {
        let Some((name, number)) = part.split_once(':') else {
            continue;
        };
        let Ok(fd) = number.trim().parse::<i32>() else {
            continue;
        };
        match name.trim() {
            "api" => fds.api = Some(fd),
            "web" => fds.web = Some(fd),
            _ => {}
        }
    }
    fds
}

/// A listener for `addr`: the inherited descriptor when there is a usable
/// one, otherwise a fresh bind.
///
/// The descriptor is checked with `F_GETFD` before it is adopted rather than
/// trusted: a handover that got as far as binding and then failed leaves a
/// closed number behind, and adopting *that* would hand out a listener on
/// whatever the number has since been recycled for.
#[cfg(unix)]
pub fn adopt_or_bind(fd: Option<i32>, addr: &str) -> Result<std::net::TcpListener> {
    use std::os::fd::FromRawFd;

    if let Some(fd) = fd
        && unsafe { libc::fcntl(fd, libc::F_GETFD) } != -1
    {
        // Safe: the descriptor was verified open just above, it was a
        // listening socket in the process that handed it over, and this is
        // the only place that adopts it (`INHERITED` is a `OnceLock` read
        // once per listener).
        return Ok(unsafe { std::net::TcpListener::from_raw_fd(fd) });
    }
    std::net::TcpListener::bind(addr).map_err(|err| anyhow!("failed to bind {addr}: {err}"))
}

#[cfg(not(unix))]
pub fn adopt_or_bind(_fd: Option<i32>, addr: &str) -> Result<std::net::TcpListener> {
    std::net::TcpListener::bind(addr).map_err(|err| anyhow!("failed to bind {addr}: {err}"))
}

/// Everything needed to start this server again with a different model:
/// captured once, in `serve`, while the values are all still in hand.
///
/// The flags are re-derived from what this process actually *resolved*, not
/// from the `argv` it was given — the workspace as an absolute path (a
/// `--daemon` process has since moved to `/`, so a relative one would mean
/// something else), and the role as an explicit flag (it may have been
/// answered at an interactive prompt rather than passed at all).
pub struct Handover {
    /// This binary. `current_exe`, not `argv[0]`, which a caller can set to
    /// anything and a `PATH` lookup may not resolve the same way twice.
    exe: PathBuf,
    /// `--config`, when one was given. Omitted otherwise so the new image
    /// runs the same default-path search this one did.
    config: Option<PathBuf>,
    /// `--workspace`, always passed, always absolute.
    workspace: PathBuf,
    /// `--all`/`--code`/`--review`/`--explorer`/`--embedding`.
    role: Role,
    /// The model this process is serving — what the new image falls back to
    /// if it cannot load the one it is being given.
    current_model: String,
    fds: InheritedFds,
}

impl Handover {
    pub fn new(
        config: Option<PathBuf>,
        workspace: PathBuf,
        role: Role,
        current_model: String,
        fds: InheritedFds,
    ) -> Result<Self> {
        Ok(Self {
            exe: std::env::current_exe()?,
            config,
            workspace,
            role,
            current_model,
            fds,
        })
    }

    /// The model this process is currently serving.
    pub fn current_model(&self) -> &str {
        &self.current_model
    }

    fn argv(&self, model: &str) -> Vec<OsString> {
        let mut argv: Vec<OsString> = vec![model.into()];
        if let Some(config) = &self.config {
            argv.push("--config".into());
            argv.push(config.into());
        }
        argv.push("--workspace".into());
        argv.push((&self.workspace).into());
        argv.push(role_flag(self.role).into());
        // Deliberately not `--daemon`: this process has already detached, and
        // the exec inherits that. Passing it again would fork a *second*
        // time, orphaning the pid a supervisor is watching.
        argv
    }

    fn inherit_value(&self) -> Option<String> {
        let mut parts = Vec::new();
        if let Some(fd) = self.fds.api {
            parts.push(format!("api:{fd}"));
        }
        if let Some(fd) = self.fds.web {
            parts.push(format!("web:{fd}"));
        }
        (!parts.is_empty()).then(|| parts.join(","))
    }

    /// Replaces this process with one serving `model`. Returns only on
    /// failure — on success there is no "after" to return to.
    ///
    /// `fallback` is the spec the new image should try if it cannot load
    /// `model`; `None` on a fallback attempt itself, which is what stops the
    /// retry from looping.
    #[cfg(unix)]
    pub fn exec(&self, model: &str, fallback: Option<&str>) -> anyhow::Error {
        use std::os::unix::process::CommandExt;

        // Rust opens every socket with `SOCK_CLOEXEC`, so without this the
        // kernel would close both listeners the instant the image is
        // replaced — and the new one would find nothing to adopt and have to
        // bind again, reopening exactly the window this whole approach
        // exists to avoid.
        for fd in [self.fds.api, self.fds.web].into_iter().flatten() {
            if unsafe { libc::fcntl(fd, libc::F_SETFD, 0) } == -1 {
                return anyhow!(
                    "failed to clear FD_CLOEXEC on descriptor {fd}: {}",
                    std::io::Error::last_os_error()
                );
            }
        }

        let mut command = std::process::Command::new(&self.exe);
        command.args(self.argv(model));
        match self.inherit_value() {
            Some(value) => command.env(INHERIT_FDS_VAR, value),
            None => command.env_remove(INHERIT_FDS_VAR),
        };
        match fallback {
            Some(fallback) => command.env(FALLBACK_MODEL_VAR, fallback),
            None => command.env_remove(FALLBACK_MODEL_VAR),
        };

        // `exec` replaces this image; anything it returns is the reason it
        // could not.
        anyhow!(
            "failed to re-execute {}: {}",
            self.exe.display(),
            command.exec()
        )
    }

    #[cfg(not(unix))]
    pub fn exec(&self, _model: &str, _fallback: Option<&str>) -> anyhow::Error {
        anyhow!("loading a different model needs execve, which this platform does not have")
    }
}

fn role_flag(role: Role) -> &'static str {
    match role {
        Role::All => "--all",
        Role::Code => "--code",
        Role::Review => "--review",
        Role::Explorer => "--explorer",
        Role::Embedding => "--embedding",
    }
}

/// Rejects a model that this build plainly cannot load, from its header
/// alone — before anything irreversible happens.
///
/// This is the check that makes a handover safe to attempt: once the image
/// is replaced there is no going back to the model that was working, so the
/// obvious failures are caught while it still is. It is the same judgement
/// the `SUPPORTED` column reports
/// (`engine::loader::model_load_support`), so a row the manager shows as
/// loadable is exactly a row this accepts.
///
/// It is *not* exhaustive, which is what [`FALLBACK_MODEL_VAR`] is for: a
/// GPU backend with no kernel for one of the model's tensor types, or a
/// model too large for the machine, can only be discovered by loading it.
pub fn precheck(path: &Path) -> Result<()> {
    let gguf = orangu::gguf::GgufFile::open(path)?;
    let (architecture, unsupported_quant) = crate::engine::loader::model_load_support(&gguf);
    let Some(architecture) = architecture else {
        bail!("{} names no architecture", path.display());
    };
    if let Some(quant) = unsupported_quant {
        bail!("this build has no dequantizer for tensor type {quant}");
    }
    crate::engine::loader::resolve_arch_family(&architecture)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_both_descriptors_and_ignores_anything_else() {
        let fds = parse_inherit_fds("api:3,web:4");
        assert_eq!(fds.api, Some(3));
        assert_eq!(fds.web, Some(4));

        // Web UI disabled: only the API listener is handed over.
        let fds = parse_inherit_fds("api:3");
        assert_eq!(fds.api, Some(3));
        assert_eq!(fds.web, None);

        // Anything unparseable is dropped rather than guessed at — the
        // caller then binds, which is correct behavior, not a failure.
        let fds = parse_inherit_fds("api:notanumber,junk,web:");
        assert_eq!(fds.api, None);
        assert_eq!(fds.web, None);
        assert_eq!(parse_inherit_fds("").api, None);
    }

    /// The new image has to be given back everything this one *resolved*,
    /// not everything it was told: a `--daemon` process has moved to `/`, so
    /// its workspace only survives as an absolute path, and a role answered
    /// at an interactive prompt was never a flag in the first place.
    #[test]
    fn argv_carries_the_resolved_workspace_and_role_and_never_daemonizes_again() {
        let handover = Handover {
            exe: PathBuf::from("/usr/bin/orangu-server"),
            config: Some(PathBuf::from("/etc/orangu-server.conf")),
            workspace: PathBuf::from("/srv/project"),
            role: Role::Review,
            current_model: "user/old".to_string(),
            fds: InheritedFds {
                api: Some(3),
                web: Some(4),
            },
        };

        let argv: Vec<String> = handover
            .argv("user/new:Q4_K_M")
            .into_iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();

        assert_eq!(
            argv,
            [
                "user/new:Q4_K_M",
                "--config",
                "/etc/orangu-server.conf",
                "--workspace",
                "/srv/project",
                "--review",
            ]
        );
        assert!(!argv.iter().any(|arg| arg == "--daemon"));
        assert_eq!(handover.inherit_value().as_deref(), Some("api:3,web:4"));
    }

    /// A server started with no `--config` must not gain one: it did its own
    /// default-path search, and the new image has to do the same one.
    #[test]
    fn argv_omits_a_config_flag_that_was_never_given() {
        let handover = Handover {
            exe: PathBuf::from("/usr/bin/orangu-server"),
            config: None,
            workspace: PathBuf::from("/srv"),
            role: Role::All,
            current_model: "user/old".to_string(),
            fds: InheritedFds {
                api: Some(3),
                web: None,
            },
        };

        let argv: Vec<String> = handover
            .argv("user/new")
            .into_iter()
            .map(|arg| arg.to_string_lossy().into_owned())
            .collect();

        assert!(!argv.iter().any(|arg| arg == "--config"), "{argv:?}");
        // Web UI disabled — only the API descriptor travels.
        assert_eq!(handover.inherit_value().as_deref(), Some("api:3"));
    }
}
