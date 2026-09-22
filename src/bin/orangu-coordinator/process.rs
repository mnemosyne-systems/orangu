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

//! Owns the single `orangu-server` child process orangu-coordinator manages:
//! which configured entry (if any) is currently running, and the
//! start/stop/health machinery to swap it for a different one on demand.

use crate::config::{CoordinatorConfiguration, CoordinatorLlmEntry, role_server_flag};
use anyhow::{Context, Result, anyhow};
use std::{
    collections::VecDeque,
    path::{Path, PathBuf},
    process::Stdio,
    sync::Arc,
    sync::Mutex as StdMutex,
    sync::atomic::{AtomicU32, Ordering},
    time::Duration,
};
use tokio::{
    io::{AsyncBufReadExt, AsyncRead, BufReader},
    process::Command,
    sync::Mutex,
    time::Instant,
};

/// A running `orangu-server` process for one configured entry.
/// The `orangu-server` a coordinator is currently routing to, and how it came
/// to be there.
///
/// Two ways, because starting one is not the only way for one to exist. A
/// server may already be listening on the profile's address when the
/// coordinator looks — an operator's own, or one left by an earlier
/// coordinator — and starting a second is not merely wasteful: it cannot bind,
/// so it loads a model for nothing and exits, and the coordinator is left
/// talking to a process it does not think it has. Adopting the one that is
/// there is both cheaper and more truthful.
enum ServerHandle {
    /// Started by this coordinator, and reaped by it.
    Spawned(tokio::process::Child),
    /// Already serving this profile when the coordinator looked. Held by pid
    /// alone: there is no `Child` to wait on for a process this one did not
    /// fork, so liveness is a signal probe and stopping is a signal.
    Adopted { pid: u32 },
}

struct ActiveProcess {
    entry_name: String,
    child: ServerHandle,
    /// The whole entry that was used to start this process, so hot-reload
    /// can detect when a profile's settings changed (any field, not just
    /// its model — a `port`/`backend`/`slots` change matters just as much).
    entry_at_start: CoordinatorLlmEntry,
    /// Kept for the process's whole lifetime (not just while starting) so a
    /// crash discovered later — e.g. while it was actively serving a
    /// request — can still be reported with its own diagnostic output
    /// attached, the same as a startup failure.
    tail: OutputTail,
}

/// An `orangu-server` found already listening on a profile's address.
struct Occupant {
    pid: u32,
    /// The model it reports serving (`/props`), compared *exactly* against the
    /// spec a profile names: a match that has to be guessed at is not a match.
    model: String,
    /// The role it came up in (`/props`), which decides whether it can answer
    /// for a profile at all — an `--embedding` server cannot serve a chat one
    /// however identical the weights.
    role: String,
}

/// Whether `pid` is an `orangu-server` process, as far as the OS is asked.
///
/// The pid comes from a `/health` answer, which is a *network* fact, and it is
/// about to be signalled. A server that reported someone else's pid — broken,
/// or hostile on a port it should not be on — would otherwise have this
/// coordinator kill an unrelated process for it. Asking the kernel what the
/// pid actually is costs one small read and turns that into nothing.
///
/// `false` when it cannot be established, which includes every platform
/// without `/proc`: the takeover it guards is a convenience, and declining to
/// perform it leaves the operator with a clear message, where performing it on
/// an unverified pid could leave them with a stopped database.
#[cfg(target_os = "linux")]
fn pid_is_orangu_server(pid: u32) -> bool {
    std::fs::read_to_string(format!("/proc/{pid}/comm"))
        .map(|comm| comm.trim() == "orangu-server")
        .unwrap_or(false)
}

#[cfg(not(target_os = "linux"))]
fn pid_is_orangu_server(_pid: u32) -> bool {
    false
}

impl Occupant {
    /// Whether this server is already doing what `entry` asks for, so that
    /// starting one would produce exactly it.
    fn serves(&self, entry: &CoordinatorLlmEntry) -> bool {
        self.model == entry.model && crate::config::roles_share_a_process(&self.role, &entry.role)
    }
}

impl ServerHandle {
    /// Whether the process has gone, in the shape [`tokio::process::Child`]
    /// answers it — `Ok(None)` while it is still there.
    ///
    /// An adopted process cannot be waited on (it is not this process's
    /// child), so it is asked the only way a stranger can be: signal `0`,
    /// which delivers nothing and fails exactly when the pid is gone. The
    /// exit status it reports is therefore a stand-in — nobody reaped it, so
    /// there is no real one — and it is only ever used to say *that* the
    /// process ended, never how.
    fn try_wait(&mut self) -> std::io::Result<Option<std::process::ExitStatus>> {
        match self {
            Self::Spawned(child) => child.try_wait(),
            #[cfg(unix)]
            Self::Adopted { pid } => {
                // SAFETY: `kill` with signal 0 sends nothing; it only reports
                // whether the pid can be signalled.
                let alive = unsafe { libc::kill(*pid as libc::pid_t, 0) } == 0;
                Ok((!alive).then(|| {
                    use std::os::unix::process::ExitStatusExt;
                    std::process::ExitStatus::from_raw(0)
                }))
            }
            #[cfg(not(unix))]
            Self::Adopted { .. } => Ok(None),
        }
    }
}

/// Number of most-recent stdout/stderr lines kept per starting/active
/// process, so a crash or a stuck health check can be reported with
/// `orangu-server`'s own diagnostic output attached instead of just a bare
/// exit signal or "timed out".
const OUTPUT_TAIL_LINES: usize = 20;

/// How long a failed startup waits for the output-capture tasks to finish
/// draining before it gives up and reports whatever it has. They normally
/// finish immediately — the pipes are at EOF once the process is gone —
/// so this only bounds the case where something else inherited a pipe and
/// is holding it open.
const OUTPUT_DRAIN_TIMEOUT: Duration = Duration::from_secs(5);

/// The exit status `orangu-server` uses when it gives up on a GPU device
/// the driver reset out from under it (its own `device_lost::EXIT_CODE`,
/// `75`/`EX_TEMPFAIL` — duplicated here rather than shared for the same
/// reason `config::default_profile_port` is: each binary stands on its own).
///
/// A child that exits with it has not crashed: it detected a lost device,
/// told its caller so in one sentence, and stepped aside precisely so this
/// coordinator would start it again on a device that works. Recognizing the
/// number is what lets that be reported as the recovery it is.
const SERVER_EXIT_DEVICE_LOST: i32 = 75;

/// Whether a stopped child stepped aside over a lost GPU device
/// ([`SERVER_EXIT_DEVICE_LOST`]) rather than failing. A signal-killed child
/// has no exit code at all, which is correctly not this.
fn is_device_lost_exit(status: &std::process::ExitStatus) -> bool {
    status.code() == Some(SERVER_EXIT_DEVICE_LOST)
}

/// Rolling tail of a process's combined stdout/stderr output.
type OutputTail = Arc<Mutex<VecDeque<String>>>;

pub struct Coordinator {
    /// RwLock to allow hot-reloading config.
    config: std::sync::RwLock<CoordinatorConfiguration>,
    http_client: reqwest::Client,
    active: Mutex<Option<ActiveProcess>>,
    /// PID of whatever `orangu-server` process is currently starting or
    /// active, if any. Set the instant a process is spawned — before its
    /// (possibly slow) health check even begins — and cleared once it's
    /// known to have stopped. `active` is held locked for an entire start
    /// sequence to serialize concurrent swaps, which can take up to
    /// `startup_timeout`; `shutdown` must not have to wait on that same
    /// lock just to kill a still-starting process, so this is tracked
    /// separately and only ever locked briefly.
    current_pid: AtomicU32,
    /// When the coordinator was last accessed by a request (for idle timeout).
    last_accessed: StdMutex<Instant>,
    /// Explicit override for which `orangu-server` executable to spawn,
    /// read once from `ORANGU_COORDINATOR_SERVER_BIN` by `main` at startup
    /// (never re-read per spawn, and never a bare global env lookup inside
    /// `start` itself — this is also what lets tests point at a stand-in
    /// executable deterministically, without racing real parallel test
    /// threads over a shared process-wide environment variable). `None`
    /// (the common case) falls back to [`Coordinator::resolve_server_binary`]'s
    /// sibling-binary/`PATH` search.
    server_binary_override: Option<PathBuf>,
}

/// Per-attempt timeout for the `/v1/models` health-check probe only — kept
/// short so a genuinely stuck/unreachable process is detected quickly and
/// retried. This must never be the client's default timeout: the same
/// client also forwards real requests to the active backend, and real
/// generation can legitimately take far longer than a health check without
/// being stuck.
const HEALTH_CHECK_TIMEOUT: Duration = Duration::from_secs(5);

impl Coordinator {
    pub fn new(
        config: CoordinatorConfiguration,
        server_binary_override: Option<PathBuf>,
    ) -> Result<Self> {
        // No default timeout: this client also proxies real requests to
        // whichever backend is active, and those must be allowed to run for
        // as long as generation actually takes. A fixed default here would
        // silently cut off any response slower than it, tearing down the
        // connection to `orangu-server` mid-stream — which surfaces to the
        // client as a bare "unexpected EOF", not a clear timeout error. The
        // health check applies its own short, explicit per-request timeout
        // instead (see `HEALTH_CHECK_TIMEOUT`).
        let http_client = reqwest::Client::builder()
            .build()
            .context("failed to build HTTP client")?;
        Ok(Self {
            config: std::sync::RwLock::new(config),
            http_client,
            active: Mutex::new(None),
            current_pid: AtomicU32::new(0),
            last_accessed: StdMutex::new(Instant::now()),
            server_binary_override,
        })
    }

    /// The shared HTTP client used both for readiness probes and, by the
    /// proxy handler, for forwarding requests to the active `orangu-server`.
    pub fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }

    /// The model each conventional role resolves to, for `GET
    /// /v1/coordinator` — see [`CoordinatorConfiguration::models_by_role`].
    pub fn models_by_role(&self) -> Vec<(String, String)> {
        let config = self.config.read().unwrap();
        config
            .models_by_role()
            .into_iter()
            .map(|(r, m)| (r.to_string(), m.to_string()))
            .collect()
    }

    /// Every role a profile is configured for, deduplicated and sorted —
    /// what `GET /v1/coordinator` reports as `roles`.
    ///
    /// Distinct from [`Self::models_by_role`], which answers "what would this
    /// role route to" for *every* conventional role because routing always
    /// falls back to `all`. This answers "is this role configured", which is
    /// the question a client has when it wants to know whether a capability
    /// exists rather than where a request would go.
    pub fn configured_roles(&self) -> Vec<String> {
        let config = self.config.read().unwrap();
        let mut roles: Vec<String> = config.llms.values().map(|e| e.role.clone()).collect();
        roles.sort();
        roles.dedup();
        roles
    }

    /// Matches `hint` against a real model id, then a role name — see
    /// [`match_hint`]. Used by `POST /v1/coordinator/activate`, which
    /// (unlike ordinary request routing) has no "currently active" or
    /// `all`-role fallback: an explicit activation request that names
    /// nothing configured is a caller error, not something to paper over.
    pub fn match_hint(&self, hint: &str) -> Option<CoordinatorLlmEntry> {
        let config = self.config.read().unwrap();
        match_hint(&config.llms, hint).cloned()
    }

    pub fn idle_timeout(&self) -> Option<u64> {
        self.config.read().unwrap().idle_timeout_seconds
    }

    pub fn shutdown_token(&self) -> Option<String> {
        self.config.read().unwrap().shutdown_token.clone()
    }

    pub fn default_entry(&self) -> CoordinatorLlmEntry {
        let config = self.config.read().unwrap();
        config.llms[&config.default_entry].clone()
    }

    pub fn reload_config(&self, new_config: CoordinatorConfiguration) {
        *self.config.write().unwrap() = new_config;
    }

    /// After a config reload, stop the active process if its profile was
    /// removed or any of its settings changed — otherwise it would keep
    /// running with stale settings indefinitely.
    pub async fn stop_if_stale(&self) {
        // Snapshot the active identity without holding the async mutex while
        // reading the (blocking) config lock.
        let (entry_name, entry_at_start) = {
            let guard = self.active.lock().await;
            let Some(active) = guard.as_ref() else {
                return;
            };
            (active.entry_name.clone(), active.entry_at_start.clone())
        };

        let should_stop = {
            let config = self.config.read().unwrap();
            match config.llms.get(&entry_name) {
                None => true, // profile was removed
                Some(entry) => *entry != entry_at_start,
            }
        };

        if !should_stop {
            return;
        }

        // Re-lock and verify the active process is still the same one we
        // snapshotted — a concurrent request could have swapped it.
        let mut guard = self.active.lock().await;
        if let Some(active) = guard.as_ref()
            && active.entry_name == entry_name
            && active.entry_at_start == entry_at_start
        {
            let active = guard.take().expect("checked above");
            log::info!(
                "stopping '{}' — profile was removed or changed in reloaded config",
                active.entry_name
            );
            self.current_pid.store(0, Ordering::Relaxed);
            Self::stop(active).await;
        }
    }

    /// Picks the configured entry a request should be routed to, trying each
    /// of the following in order and falling through when a step finds
    /// nothing:
    ///
    /// 1. The entry whose `model` matches `model_hint`.
    /// 2. The entry whose `role` matches `model_hint` — this is what lets
    ///    orangu's own config stay entirely coordinator-agnostic: a server
    ///    section behind a coordinator can just set `model` to the role name
    ///    itself (`all`, `code`, `review`, `explorer`, `embeddings`) instead
    ///    of duplicating the real backend model id.
    /// 3. The entry whose `role` matches `implied_role` — the role a request
    ///    *type* itself implies regardless of what `model` it named or
    ///    didn't (currently just `/v1/embeddings` implying `embeddings`; see
    ///    [`crate::proxy::implied_role_for_path`]). This outranks "currently
    ///    active" on purpose: a stale or absent `model` field must not send
    ///    an embeddings request to whatever chat model happens to be loaded.
    /// 4. Whichever entry is currently active, if any — this is what makes
    ///    the `orangu-server`-native endpoints (`/v1/models`, `/health`,
    ///    `/props`, `/slots`, `/metrics`), which carry no `model` field to
    ///    route on, report on whatever is actually running instead of
    ///    silently forcing a swap back to `all` — e.g. `/information`
    ///    probing a server's `/health` would otherwise itself knock out
    ///    whatever role a real request had just switched to.
    /// 5. The `all`-role default entry.
    pub async fn resolve_entry(
        &self,
        model_hint: Option<&str>,
        implied_role: Option<&str>,
    ) -> CoordinatorLlmEntry {
        self.touch_last_accessed();
        let active_entry_name = self
            .active
            .lock()
            .await
            .as_ref()
            .map(|active| active.entry_name.clone());
        let config = self.config.read().unwrap();
        select_entry(
            &config.llms,
            &config.default_entry,
            model_hint,
            implied_role,
            active_entry_name.as_deref(),
        )
        .clone()
    }

    /// Ensures `entry`'s `orangu-server` is the active process, starting it
    /// (and stopping whatever else was active) if it isn't already, then
    /// returns the origin requests should be proxied to.
    ///
    /// Swapping to a *different* profile always fully stops (`Self::stop`,
    /// which awaits the child's actual exit, not just signals it) whatever
    /// was running **before** starting the new one — never the other way
    /// around, and never concurrently. This is what makes it safe for
    /// multiple profiles to share the same `host`/`port` (the default for
    /// every role, since `CoordinatorLlmEntry::host`/`port` both fall back
    /// to `all`/`8100` when a profile's own config omits them): by
    /// the time the new `orangu-server` tries to bind that address, the old
    /// one's listening socket has already been released, not merely asked
    /// to release it.
    pub async fn ensure_active(&self, entry: &CoordinatorLlmEntry) -> Result<String> {
        let mut guard = self.active.lock().await;

        if let Some(active) = guard.as_mut() {
            // The same profile, or a different one this process already
            // serves: `serves_same_process_as` is true exactly when the only
            // difference is the role, which travels per request instead
            // (`proxy::ROLE_HEADER`). Restarting for that would reload the
            // identical model file and throw away every cached prefix with
            // the old process.
            let running = active.entry_name.clone();
            if active.entry_name == entry.name
                || active.entry_at_start.serves_same_process_as(entry)
            {
                // Already the active model — but confirm the process is
                // still alive; a crashed backend must be restarted rather
                // than silently proxied into.
                match active.child.try_wait() {
                    Ok(None) => return Ok(entry.origin()),
                    Ok(Some(status)) => {
                        // A device-loss exit is expected, self-inflicted,
                        // and fixed by the restart this function is about
                        // to do — so it's reported as what it is rather
                        // than as an unexplained crash, and without the
                        // output tail, whose last lines are the same
                        // message `orangu-server` already printed.
                        if is_device_lost_exit(&status) {
                            log::warn!(
                                "'{running}' exited after losing its GPU device (a driver \
                                 reset); restarting it on a fresh device"
                            );
                        } else {
                            log::warn!(
                                "warning: '{running}' exited unexpectedly while active \
                                 (status: {status}){}",
                                format_output_tail(&active.tail).await
                            );
                        }
                    }
                    _ => {}
                }
            }
            // Clear `current_pid` before reaping: once `stop` returns, that
            // pid is gone and the OS is free to recycle it, so it must not
            // linger as a stale value a concurrent `shutdown` could kill.
            self.current_pid.store(0, Ordering::Relaxed);
            Self::stop(guard.take().expect("checked above")).await;
        }

        // Before starting one, see what is already on the address — starting
        // a second server there cannot bind, so it would load a model for
        // nothing and exit, leaving the coordinator talking to a process it
        // does not think it has.
        //
        // An `orangu-server` on this address is under this coordinator's
        // management whoever started it: the address is the coordinator's by
        // configuration, and the only question is whether the server there is
        // already doing the job.
        if let Some(occupant) = self.occupant(entry).await {
            if occupant.serves(entry) {
                log::info!(
                    "adopting the orangu-server already serving '{}' at {} (process {})",
                    entry.name,
                    entry.origin(),
                    occupant.pid
                );
                self.current_pid.store(occupant.pid, Ordering::Relaxed);
                *guard = Some(ActiveProcess {
                    entry_name: entry.name.clone(),
                    entry_at_start: entry.clone(),
                    child: ServerHandle::Adopted { pid: occupant.pid },
                    // Nothing was captured from a process this one did not
                    // start; its output belongs to whoever did.
                    tail: Arc::new(Mutex::new(VecDeque::new())),
                });
                return Ok(entry.origin());
            }
            // Serving something else. This is a *swap*, and the fact that
            // this coordinator did not start the incumbent changes nothing
            // about it: the same act is already performed on every adopted
            // server whose profile is swapped away from. Refusing it instead
            // — which this used to do — left a leftover from an earlier run
            // able to block every profile indefinitely, with the operator
            // told to go and find it by hand.
            //
            // Only against a pid the kernel agrees is an `orangu-server`.
            // Everything else falls through to the start below, which reports
            // the address as taken rather than signalling a stranger.
            if pid_is_orangu_server(occupant.pid) {
                log::info!(
                    "taking {} for '{}': process {} is serving {} in {} mode, which this profile \
                     does not ask for",
                    entry.origin(),
                    entry.name,
                    occupant.pid,
                    occupant.model,
                    occupant.role,
                );
                Self::stop(ActiveProcess {
                    entry_name: format!("(unmanaged, process {})", occupant.pid),
                    entry_at_start: entry.clone(),
                    child: ServerHandle::Adopted { pid: occupant.pid },
                    tail: Arc::new(Mutex::new(VecDeque::new())),
                })
                .await;
            }
        }

        let (child, tail) = self.start(entry).await?;
        *guard = Some(ActiveProcess {
            entry_name: entry.name.clone(),
            entry_at_start: entry.clone(),
            child: ServerHandle::Spawned(child),
            tail,
        });
        Ok(entry.origin())
    }

    /// What is already listening at `entry`'s address, when it is an
    /// `orangu-server` that can be identified — `None` for an empty address,
    /// and for anything that cannot say what it is.
    ///
    /// The distinction that matters is *identifiable*, not *ours*. A server
    /// this coordinator did not start is still an `orangu-server` on an
    /// address this coordinator is configured to own, and what to do about it
    /// follows from what it is serving, not from who started it. Something
    /// that answers neither `/props` nor `/health` — a different program, or
    /// one whose `api_key` locks this coordinator out — is not identifiable
    /// and is never touched.
    async fn occupant(&self, entry: &CoordinatorLlmEntry) -> Option<Occupant> {
        let origin = entry.origin();
        let props: serde_json::Value = self
            .http_client
            .get(format!("{origin}/props"))
            .timeout(HEALTH_CHECK_TIMEOUT)
            .send()
            .await
            .ok()?
            .json()
            .await
            .ok()?;
        let pid = self
            .http_client
            .get(format!("{origin}/health"))
            .timeout(HEALTH_CHECK_TIMEOUT)
            .send()
            .await
            .ok()?
            .json::<serde_json::Value>()
            .await
            .ok()?
            .get("pid")?
            .as_u64()? as u32;
        Some(Occupant {
            pid,
            model: props.get("model")?.as_str()?.to_string(),
            role: props.get("role")?.as_str()?.to_string(),
        })
    }

    /// Makes sure `entry`'s `orangu-server` is *answering*, not merely
    /// believed to be running, and returns the origin to (re)send a request
    /// to. Used by the proxy after a forwarded request failed to reach the
    /// child at all.
    ///
    /// [`Self::ensure_active`] asks `try_wait` whether the child is alive,
    /// which is the right question everywhere except here. That answer
    /// lags: a `SIGKILL`ed child is *gone* immediately but is not reported
    /// as exited until tokio's `SIGCHLD` handling has run, so a retry that
    /// consults it milliseconds after the connection failed is told the
    /// dead process is fine and sends the request straight back into the
    /// same closed port. (Measured, not theorized: killing a live profile
    /// mid-request did exactly that.)
    ///
    /// So this asks the only question that cannot lag — can it be reached —
    /// with the same short probe a startup health check uses. An answer
    /// means the failure was transient and nothing should be restarted (a
    /// restart would throw away a working process and every other request
    /// on it). No answer means the child is gone whatever `try_wait`
    /// currently believes, and it is replaced.
    pub async fn ensure_reachable(&self, entry: &CoordinatorLlmEntry) -> Result<String> {
        let probe = self
            .http_client
            .get(format!("{}/v1/models", entry.origin()))
            .timeout(HEALTH_CHECK_TIMEOUT)
            .send()
            .await;
        // **Any** HTTP response proves the process is there — including one
        // that refuses. Requiring `is_success` conflates reachability with
        // authorization, and the moment the server is given an
        // `[orangu-server].api_key` this unauthenticated probe starts getting
        // `401`, is read as "stopped answering", and restarts a perfectly
        // healthy child on every request. The question here is whether the
        // child is alive, and a `401` answers it.
        if probe.is_ok() {
            return Ok(entry.origin());
        }

        let mut guard = self.active.lock().await;
        if let Some(active) = guard.take() {
            log::warn!(
                "'{}' stopped answering on {}; restarting it",
                entry.name,
                entry.origin()
            );
            // Same ordering rule as `ensure_active`: clear the pid before
            // reaping, so a concurrent `shutdown` can't signal a number the
            // OS has already handed to something else.
            self.current_pid.store(0, Ordering::Relaxed);
            Self::stop(active).await;
        }
        let (child, tail) = self.start(entry).await?;
        *guard = Some(ActiveProcess {
            entry_name: entry.name.clone(),
            entry_at_start: entry.clone(),
            child: ServerHandle::Spawned(child),
            tail,
        });
        Ok(entry.origin())
    }

    /// Stops whatever `orangu-server` process is currently starting or
    /// active, if any. Called on coordinator shutdown so no orphaned
    /// process is left running — including one still mid-startup (spawned,
    /// but not yet confirmed healthy), which `current_pid` catches and
    /// `active` alone would miss, since `active` isn't populated until the
    /// health check succeeds.
    pub async fn shutdown(&self) {
        self.current_pid.store(0, Ordering::Relaxed);
        let mut guard = self.active.lock().await;
        if let Some(active) = guard.take() {
            Self::stop(active).await;
        }
        // If no active process exists but a stale PID was recorded (still
        // mid-startup), we intentionally do NOT signal it here: the startup
        // code path still holds the Child handle, and `kill_on_drop(true)`
        // ensures the process is cleaned up when that handle is dropped.
        // Signalling a bare PID risks hitting an unrelated process if the
        // OS has already recycled it.
    }

    pub fn touch_last_accessed(&self) {
        *self.last_accessed.lock().unwrap() = Instant::now();
    }

    pub async fn unload_if_idle(&self, timeout_secs: u64) {
        let last_accessed = *self.last_accessed.lock().unwrap();
        if last_accessed.elapsed().as_secs() >= timeout_secs {
            let mut guard = self.active.lock().await;
            let last_accessed = *self.last_accessed.lock().unwrap();
            if last_accessed.elapsed().as_secs() >= timeout_secs
                && let Some(active) = guard.take()
            {
                log::info!(
                    "unloading active profile '{}' due to idle timeout",
                    active.entry_name
                );
                self.current_pid.store(0, Ordering::Relaxed);
                Self::stop(active).await;
            }
        }
    }

    async fn stop(active: ActiveProcess) {
        match active.child {
            ServerHandle::Spawned(mut child) => {
                #[cfg(unix)]
                {
                    if let Some(pid) = child.id() {
                        kill_pid(pid);
                    }
                    // Wait up to 5 seconds for graceful shutdown
                    if tokio::time::timeout(Duration::from_secs(5), child.wait())
                        .await
                        .is_err()
                    {
                        let _ = child.start_kill();
                        let _ = child.wait().await;
                    }
                }
                #[cfg(not(unix))]
                {
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                }
            }
            // Not this process's child, so there is nothing to reap and
            // nothing to wait on: signal it and watch the pid instead.
            //
            // Stopped at all, rather than left alone, because the port is the
            // coordinator's by configuration and a swap needs it. A profile
            // whose server must not be touched belongs on its own `port`,
            // where no swap will ever ask for it.
            ServerHandle::Adopted { pid } => {
                #[cfg(unix)]
                {
                    kill_pid(pid);
                    let deadline = Instant::now() + Duration::from_secs(5);
                    // SAFETY: signal 0 delivers nothing and only reports
                    // whether the pid is still there.
                    while unsafe { libc::kill(pid as libc::pid_t, 0) } == 0 {
                        if Instant::now() >= deadline {
                            // SAFETY: same call, with the signal that cannot
                            // be caught or ignored.
                            unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) };
                            break;
                        }
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
                #[cfg(not(unix))]
                let _ = pid;
            }
        }
    }

    /// Which `orangu-server` executable to spawn: `server_binary_override`
    /// if one was given (see its own doc comment), otherwise a sibling
    /// `orangu-server` next to this coordinator's own executable (the
    /// common case — both binaries come from the same build/install), then
    /// falling back to a bare `orangu-server` resolved via `PATH`.
    fn resolve_server_binary(&self) -> PathBuf {
        if let Some(path) = &self.server_binary_override {
            return path.clone();
        }
        let binary_name = if cfg!(windows) {
            "orangu-server.exe"
        } else {
            "orangu-server"
        };
        if let Ok(mut path) = std::env::current_exe() {
            path.set_file_name(binary_name);
            if path.is_file() {
                return path;
            }
        }
        PathBuf::from(binary_name)
    }

    async fn start(
        &self,
        entry: &CoordinatorLlmEntry,
    ) -> Result<(tokio::process::Child, OutputTail)> {
        let (models_dir, log) = {
            let config = self.config.read().unwrap();
            (config.models.clone(), config.log.clone())
        };
        let server_config_path =
            write_server_config(entry, &models_dir, &log).with_context(|| {
                format!("failed to write orangu-server config for '{}'", entry.name)
            })?;
        let program = self.resolve_server_binary();
        // Already validated at config-load time (`config::parse_llm_profiles`
        // rejects any role `role_server_flag` doesn't recognize), so this can
        // only fail if a config was somehow constructed bypassing that check.
        let role_flag = role_server_flag(&entry.role)
            .ok_or_else(|| anyhow!("[{}] has an unknown role '{}'", entry.name, entry.role))?;

        // No `--workspace` here: the backend takes its own, which without
        // the flag is the directory it is started in — the coordinator's own
        // working directory, inherited by the child. The coordinator has no
        // workspace of its own; it is a pass-through.
        let mut command = Command::new(&program);
        command
            .arg("--config")
            .arg(&server_config_path)
            .arg(role_flag)
            .arg(&entry.model)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        die_with_parent(&mut command);
        let mut child = command.spawn().with_context(|| {
            format!(
                "failed to start orangu-server for '{}' ({})",
                entry.name,
                program.display()
            )
        })?;

        // Record the PID *before* the (possibly long) health-check wait
        // below, so a concurrent `shutdown` can always kill this process,
        // no matter how long it takes to become ready.
        self.current_pid
            .store(child.id().unwrap_or(0), Ordering::Relaxed);

        let tail: OutputTail = Arc::new(Mutex::new(VecDeque::new()));
        // Kept so a startup failure can wait for these to drain before it
        // reports what the process said — see `wait_until_healthy`.
        let mut capture = Vec::new();
        if let Some(stdout) = child.stdout.take() {
            capture.push(spawn_output_capture(stdout, tail.clone()));
        }
        if let Some(stderr) = child.stderr.take() {
            capture.push(spawn_output_capture(stderr, tail.clone()));
        }

        if let Err(err) = self
            .wait_until_healthy(entry, &mut child, &tail, capture)
            .await
        {
            let _ = child.start_kill();
            let _ = child.wait().await;
            self.current_pid.store(0, Ordering::Relaxed);
            return Err(err);
        }

        Ok((child, tail))
    }

    async fn wait_until_healthy(
        &self,
        entry: &CoordinatorLlmEntry,
        child: &mut tokio::process::Child,
        tail: &OutputTail,
        capture: Vec<tokio::task::JoinHandle<()>>,
    ) -> Result<()> {
        let startup_timeout = self.config.read().unwrap().startup_timeout_seconds;
        let deadline = Instant::now() + Duration::from_secs(startup_timeout);
        // `/health`, not `/v1/models`: it is open on an authenticated server
        // (so a `200` needs no key) and it names the process answering, which
        // is the question this loop is actually asking. See `answering_pid`.
        let probe_url = format!("{}/health", entry.origin());
        let spawned_pid = child.id();

        loop {
            if let Ok(Some(status)) = child.try_wait() {
                // `try_wait` sees the exit the instant it happens, but the
                // tasks draining stdout/stderr into `tail` are separate —
                // on a loaded machine they may not have run yet. Reporting
                // straight away is how a crash could still surface as a
                // bare status with no diagnostic, which is the very thing
                // capturing the output was meant to prevent. The pipes are
                // at EOF now that the process is gone, so these finish on
                // their own; the timeout only guards against a stray child
                // inheriting a pipe and holding it open.
                let _ = tokio::time::timeout(OUTPUT_DRAIN_TIMEOUT, async {
                    for task in capture {
                        let _ = task.await;
                    }
                })
                .await;
                return Err(anyhow!(
                    "orangu-server for '{}' exited before becoming ready (status: {status}){}",
                    entry.name,
                    format_output_tail(tail).await
                ));
            }

            let request = self
                .http_client
                .get(&probe_url)
                .timeout(HEALTH_CHECK_TIMEOUT);
            // An answer means *a* listener is up on that address. Whether it
            // is the one just spawned is a different question, and it used to
            // go unasked: a leftover `orangu-server` still holding the port
            // answers instantly, so the coordinator called the swap done while
            // its own child was still loading, recorded it as active, and
            // proxied every request to a process it did not start — serving
            // whatever model *that* one had. The child then failed to bind and
            // exited, and the next request found it dead, restarted it, and
            // did the whole thing again. Nothing in that loop is visible as an
            // error; the model is simply not the one that was asked for.
            if let Ok(response) = request.send().await {
                match (answering_pid(response).await, spawned_pid) {
                    // Someone else's listener. Retrying cannot help — the port
                    // is taken by a process this coordinator does not manage —
                    // so this is reported rather than waited out.
                    (Some(answering), Some(spawned)) if answering != spawned => {
                        return Err(anyhow!(
                            "'{}' cannot serve {}: process {answering} is already listening there, \
                             and it is not the orangu-server just started (process {spawned}). \
                             Stop it — an orangu-server or orangu-coordinator left running from \
                             earlier is the usual cause — or give this profile its own `port`.",
                            entry.name,
                            entry.origin(),
                        ));
                    }
                    // Its own child, or a server too old to say (`pid` was
                    // added alongside this check): ready, as before.
                    _ => return Ok(()),
                }
            }

            if Instant::now() >= deadline {
                return Err(anyhow!(
                    "timed out after {}s waiting for orangu-server ('{}') to become ready at {}{}",
                    startup_timeout,
                    entry.name,
                    probe_url,
                    format_output_tail(tail).await
                ));
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
    }
}

/// Asks the kernel to signal this child if the coordinator goes away, so an
/// `orangu-server` can never outlive the process that owns its lifecycle.
///
/// `kill_on_drop` and the shutdown path cover every exit the coordinator gets
/// to *run*. They cover nothing about a `SIGKILL`, and what they leave behind
/// is the worst kind of leftover: a server still holding the port, still
/// answering, and belonging to nobody — so the next coordinator's children
/// cannot bind, exit, and are replaced forever while requests are served by
/// the ghost with whatever model it happened to have.
///
/// `PR_SET_PDEATHSIG` is per-child and set between fork and exec, so it says
/// nothing about a server started by hand: only one spawned here dies with its
/// parent. The `getppid` re-check closes the one race in it — a parent that
/// died in the window before the `prctl` would otherwise never send anything.
///
/// A no-op where the kernel has no such facility; there the shutdown path is
/// all there is, as before.
#[cfg(target_os = "linux")]
fn die_with_parent(command: &mut Command) {
    let parent = std::process::id();
    // SAFETY: async-signal-safe calls only (`prctl`, `getppid`, `_exit`), as
    // required between `fork` and `exec`; no allocation, no locks.
    unsafe {
        command.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL) != 0 {
                // Not fatal: the server still works, it just outlives a
                // killed coordinator the way it always did.
                return Ok(());
            }
            // Reparented before the `prctl` landed — the signal it asks for
            // will never come, so leave now rather than become the ghost.
            if libc::getppid() as u32 != parent {
                libc::_exit(1);
            }
            Ok(())
        });
    }
}

#[cfg(not(target_os = "linux"))]
fn die_with_parent(_command: &mut Command) {}

/// The `pid` a `GET /health` answer names, or `None` when the answer carries
/// none — an older `orangu-server`, or something else entirely on that port.
///
/// `None` is deliberately not a failure. It cannot distinguish "an old build"
/// from "a stranger", and refusing to start against the first would break an
/// in-place upgrade for a check that exists to catch the second. The pid is
/// what makes the strong statement; its absence leaves the old, weaker one.
async fn answering_pid(response: reqwest::Response) -> Option<u32> {
    response
        .json::<serde_json::Value>()
        .await
        .ok()?
        .get("pid")?
        .as_u64()
        .map(|pid| pid as u32)
}

/// Reads `stream` line by line for as long as the process keeps it open,
/// keeping the last [`OUTPUT_TAIL_LINES`] in `tail` and echoing each line
/// into the coordinator's own log as it arrives (which `--quiet` silences on
/// the console, like everything else) — preserving today's visible behavior
/// (e.g. model-download progress) for anyone watching the coordinator's
/// console, while still letting a later crash or stuck health check report
/// the same output inline.
///
/// Under `log_type = file` a profile's `orangu-server` is handed the same
/// file (see [`write_server_config`]) and writes its own log lines straight
/// into it, so what arrives here is only what it prints *around* its log:
/// the `error:` line it exits on, most usefully.
fn spawn_output_capture(
    stream: impl AsyncRead + Send + Unpin + 'static,
    tail: OutputTail,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut lines = BufReader::new(stream).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            log::info!("{line}");
            let mut tail = tail.lock().await;
            if tail.len() >= OUTPUT_TAIL_LINES {
                tail.pop_front();
            }
            tail.push_back(line);
        }
    })
}

/// Formats a process's captured output tail as an error-message suffix:
/// empty when there's nothing captured (yet), otherwise a labeled, indented
/// block ready to append directly to an `anyhow!` message.
async fn format_output_tail(tail: &OutputTail) -> String {
    let tail = tail.lock().await;
    if tail.is_empty() {
        return String::new();
    }
    let lines = tail
        .iter()
        .map(|line| format!("  {line}"))
        .collect::<Vec<_>>()
        .join("\n");
    format!("\nlast output:\n{lines}")
}

/// Writes `entry`'s own `orangu-server.conf` — the `[orangu-server]`
/// section (`models`, `host`, `port`, whichever of `backend`/`slots` were
/// set, and the coordinator's own `log_type`/`log_path` when that is a
/// file), plus a `[web]` section when the profile asks for a web console
/// — to `~/.orangu/coordinator/servers/<name>.conf`,
/// overwriting any previous contents — `orangu-server` itself reads this
/// file once at its own startup, so a stale file from a previous run is
/// never an issue, and this path doubles as a debugging aid: exactly what a
/// profile was last started with is always inspectable on disk.
///
/// The log keys travel because the server's output has to end up where the
/// coordinator's does. Left to the console, a profile would print into the
/// pipe this coordinator reads — every second of a request's progress
/// included, `\r` and all — and the coordinator would copy that into its
/// file as one long line. Handed the file, the server appends its own
/// completed lines, each stamped, and prints no progress at all.
fn write_server_config(
    entry: &CoordinatorLlmEntry,
    models_dir: &Path,
    log: &orangu::logging::LogTarget,
) -> Result<PathBuf> {
    let dir = home::home_dir()
        .context("failed to resolve home directory")?
        .join(".orangu/coordinator/servers");
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("failed to create directory {}", dir.display()))?;
    let path = dir.join(format!("{}.conf", entry.name));

    let mut contents = format!(
        "[orangu-server]\nmodels = {}\nhost = {}\nport = {}\n",
        models_dir.display(),
        entry.host,
        entry.port
    );
    if let Some(backend) = &entry.backend {
        contents.push_str(&format!("backend = {backend}\n"));
    }
    if let Some(slots) = entry.slots {
        contents.push_str(&format!("slots = {slots}\n"));
    }
    if let Some(path) = log.path() {
        contents.push_str(&format!(
            "log_type = {}\nlog_path = {}\n",
            log.log_type(),
            path.display()
        ));
    }
    // Its own section, which is what `orangu-server` reads today: the
    // section's presence is what turns the console on. A profile with no
    // `web` gets no section, and so no second listener.
    if let Some(web) = entry.web {
        contents.push_str(&format!("\n[web]\nport = {web}\n"));
    }

    std::fs::write(&path, contents)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(path)
}

/// Pure entry-selection logic behind [`Coordinator::resolve_entry`]: prefers
/// an entry whose `model` matches `model_hint`, then whichever entry
/// `active_entry_name` names, then the `all`-role default. Kept free of any
/// locking so the routing policy itself is directly unit-testable.
///
/// When more than one entry shares the requested model, the
/// lexicographically smallest name wins, so the choice is stable across runs
/// rather than depending on hash map iteration order.
fn select_entry<'a>(
    llms: &'a std::collections::HashMap<String, CoordinatorLlmEntry>,
    default_entry: &str,
    model_hint: Option<&str>,
    implied_role: Option<&str>,
    active_entry_name: Option<&str>,
) -> &'a CoordinatorLlmEntry {
    if let Some(hint) = model_hint {
        if let Some(entry) = match_hint(llms, hint) {
            return entry;
        }
        // An explicit hint that matched nothing configured is a deliberate
        // request for something specific — falling through to "currently
        // active" would silently substitute whatever unrelated role a prior
        // request happened to leave running (e.g. `code` requested but not
        // configured, while `review` is still active from earlier). Skip
        // straight to the implied role (if any) and then the `all` default,
        // which is deterministic and doesn't depend on session history.
        if let Some(role) = implied_role
            && let Some(entry) = best_match(llms, |entry| entry.role == role)
        {
            return entry;
        }
        return &llms[default_entry];
    }
    // The request type itself implies a role (currently just
    // /v1/embeddings → embeddings), regardless of what `model` named or
    // didn't. This outranks "currently active": a stale or absent `model`
    // field must not send an embeddings request to whatever chat model
    // happens to be loaded.
    if let Some(role) = implied_role
        && let Some(entry) = best_match(llms, |entry| entry.role == role)
    {
        return entry;
    }
    if let Some(name) = active_entry_name
        && let Some(entry) = llms.get(name)
    {
        return entry;
    }
    &llms[default_entry]
}

/// The entry matching `predicate` with the lexicographically smallest name,
/// so ties (more than one profile sharing a model, or a role) resolve the
/// same stable way regardless of hash map iteration order.
fn best_match(
    llms: &std::collections::HashMap<String, CoordinatorLlmEntry>,
    predicate: impl Fn(&CoordinatorLlmEntry) -> bool,
) -> Option<&CoordinatorLlmEntry> {
    let mut matches: Vec<&CoordinatorLlmEntry> =
        llms.values().filter(|entry| predicate(entry)).collect();
    matches.sort_unstable_by(|a, b| a.name.cmp(&b.name));
    matches.into_iter().next()
}

/// Matches `hint` against an entry's real model id first, then, failing
/// that, against an entry's role name — so orangu.conf can just set `model
/// = explorer` (the role) instead of duplicating the real backend model id;
/// the coordinator alone owns which actual model that role maps to.
fn match_hint<'a>(
    llms: &'a std::collections::HashMap<String, CoordinatorLlmEntry>,
    hint: &str,
) -> Option<&'a CoordinatorLlmEntry> {
    best_match(llms, |entry| entry.model == hint)
        .or_else(|| best_match(llms, |entry| entry.role == hint))
}

/// Sends an immediate, unconditional kill to a bare PID — used by
/// [`Coordinator::shutdown`] to terminate a still-starting `orangu-server`
/// process that has no live `tokio::process::Child` handle left to call
/// `start_kill()` on (see `current_pid`'s doc comment). Best-effort: an
/// already-gone PID is simply a no-op. Its two callers are gated the same
/// way, so there is no other-platform arm to leave unused.
#[cfg(unix)]
fn kill_pid(pid: u32) {
    unsafe {
        libc::kill(pid as libc::pid_t, libc::SIGINT);
    }
}

/// Writes a `#!/bin/sh` script with `body` as its content to a fresh
/// temp file, makes it executable, and leaks it (never auto-deleted) so
/// the returned path stays valid for as long as a test needs to exec
/// it — standing in for "a real `orangu-server`-shaped executable"
/// wherever a test needs `orangu-coordinator` to spawn specific,
/// controlled behavior (hang, crash with a specific stderr, echo an
/// argument back, etc.) without needing a real `orangu-server` binary
/// or model. Pointed at via `Coordinator::new`'s
/// `server_binary_override`, never a shared environment variable — see
/// that field's own doc comment for why (parallel test threads would
/// otherwise race on a process-wide env var).
///
/// Unix-only, and so is every test that spawns one: a shell script is not an
/// executable on Windows. Shared with `proxy`'s tests, which need the same
/// "a server that just stays alive" stand-in.
#[cfg(all(test, unix))]
pub(crate) fn fake_server_script(body: &str) -> PathBuf {
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;
    let mut file = tempfile::NamedTempFile::new().unwrap();
    write!(file, "#!/bin/sh\n{body}").unwrap();
    let path = file.into_temp_path().keep().unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    path
}

/// Spawns one of these scripts, waiting out a kernel that still considers
/// the file to be open for writing.
///
/// A test binary is many threads and every `Command::spawn` forks the lot of
/// them: a child forked while this script was still being written inherits a
/// duplicate of that write descriptor and keeps it until it execs, and an
/// `exec` of the file during that window fails with `ETXTBSY` however
/// carefully the writing thread closed its own handle. `O_CLOEXEC` does not
/// close it either — it closes at `exec`, which is the far end of the
/// window. So it is a race against whatever else the suite is doing, not a
/// broken script, and the answer is to try again rather than to report it.
/// Seen once on a CI runner and not on this project's own machines, which is
/// the shape of a race that is scheduling-dependent.
#[cfg(test)]
pub(crate) fn spawn_fake_server(path: &Path) -> std::process::Child {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        match std::process::Command::new(path).spawn() {
            Ok(child) => return child,
            Err(e)
                if e.kind() == std::io::ErrorKind::ExecutableFileBusy
                    && std::time::Instant::now() < deadline =>
            {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(e) => panic!("spawning {}: {e}", path.display()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::load_coordinator_configuration;
    use std::collections::HashMap;
    use std::io::Write;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    fn minimal_config(extra_client: &str, profiles: &str) -> CoordinatorConfiguration {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nmodels = /srv/models\n{extra_client}\n{profiles}"
        )
        .unwrap();
        load_coordinator_configuration(file.path()).unwrap()
    }

    #[tokio::test]
    async fn http_client_has_no_default_timeout_for_proxied_requests() {
        // Regression test: the coordinator's shared HTTP client used to
        // have a hardcoded 5s timeout meant only for the health-check probe
        // in `wait_until_healthy`, but the same client also proxies real
        // requests to the active backend — any generation slower than 5s
        // got its connection killed mid-stream, surfacing to the caller as
        // a bare "unexpected EOF during chunk size line" rather than a
        // clear timeout. The client itself must have no default timeout;
        // only the health check applies one explicitly (`HEALTH_CHECK_TIMEOUT`).
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf).await;
            tokio::time::sleep(Duration::from_secs(6)).await;
            let body = b"ok";
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = stream.write_all(response.as_bytes()).await;
            let _ = stream.write_all(body).await;
            let _ = stream.shutdown().await;
        });

        let config = minimal_config("", "[main]\nrole = all\nmodel = org/gemma\nport = 8100\n");
        let coordinator = Coordinator::new(config, None).unwrap();

        let result = coordinator
            .http_client()
            .get(format!("http://{addr}"))
            .send()
            .await;
        assert!(
            result.is_ok(),
            "a 6s response must not be cut off by a default client timeout: {result:?}"
        );
    }

    #[tokio::test]
    async fn resolve_entry_falls_back_to_default_when_model_hint_is_absent_or_unknown() {
        let config = minimal_config("", "[main]\nrole = all\nmodel = org/gemma\nport = 8100\n");
        let coordinator = Coordinator::new(config, None).unwrap();

        assert_eq!(coordinator.resolve_entry(None, None).await.name, "main");
        assert_eq!(
            coordinator.resolve_entry(Some("unknown"), None).await.name,
            "main"
        );
        assert_eq!(
            coordinator
                .resolve_entry(Some("org/gemma"), None)
                .await
                .name,
            "main"
        );
    }

    #[tokio::test]
    async fn resolve_entry_breaks_ties_between_profiles_sharing_a_model_by_name() {
        // Profiles may share a model (not an error, see config.rs); the match
        // must still be deterministic rather than depend on hash map order.
        let config = minimal_config(
            "",
            "[zeta]\nrole = explorer\nmodel = org/gemma\nport = 8200\n\n[alpha]\nrole = all\nmodel = org/gemma\nport = 8100\n",
        );
        let coordinator = Coordinator::new(config, None).unwrap();

        assert_eq!(
            coordinator
                .resolve_entry(Some("org/gemma"), None)
                .await
                .name,
            "alpha"
        );
    }

    fn test_llms() -> HashMap<String, CoordinatorLlmEntry> {
        let mut llms = HashMap::new();
        llms.insert(
            "all".to_string(),
            CoordinatorLlmEntry {
                name: "all".to_string(),
                role: "all".to_string(),
                model: "org/gemma".to_string(),
                host: "127.0.0.1".to_string(),
                port: 8100,
                backend: None,
                slots: None,
                web: None,
            },
        );
        llms.insert(
            "explorer".to_string(),
            CoordinatorLlmEntry {
                name: "explorer".to_string(),
                role: "explorer".to_string(),
                model: "org/qwen".to_string(),
                host: "127.0.0.1".to_string(),
                port: 8200,
                backend: None,
                slots: None,
                web: None,
            },
        );
        llms
    }

    #[test]
    fn select_entry_prefers_the_active_entry_when_no_model_hint_is_given() {
        // A bodyless request (GET /v1/models, /health, /props, /slots,
        // /metrics) must report on whatever is actually running, not force a
        // swap back to `all` just because it carries no model to match on.
        let llms = test_llms();
        let entry = select_entry(&llms, "all", None, None, Some("explorer"));
        assert_eq!(entry.name, "explorer");
    }

    #[test]
    fn select_entry_falls_back_to_default_when_nothing_is_active() {
        let llms = test_llms();
        let entry = select_entry(&llms, "all", None, None, None);
        assert_eq!(entry.name, "all");
    }

    #[test]
    fn select_entry_falls_back_to_default_when_active_entry_is_unknown() {
        let llms = test_llms();
        let entry = select_entry(&llms, "all", None, None, Some("stale-removed-entry"));
        assert_eq!(entry.name, "all");
    }

    #[test]
    fn select_entry_prefers_an_explicit_model_hint_over_the_active_entry() {
        // A real client request naming a model always wins, even if a
        // different role happens to be active right now.
        let llms = test_llms();
        let entry = select_entry(&llms, "all", Some("org/qwen"), None, Some("all"));
        assert_eq!(entry.name, "explorer");
    }

    #[test]
    fn select_entry_falls_back_to_default_rather_than_active_when_hint_is_unmatched() {
        // An explicit hint that matches nothing configured (e.g. `code`
        // requested but no dedicated profile exists) is a deliberate ask for
        // something specific — it must not silently inherit whatever
        // unrelated role a prior request left active (here, `explorer`).
        // The deterministic `all` default is used instead.
        let llms = test_llms();
        let entry = select_entry(&llms, "all", Some("code"), None, Some("explorer"));
        assert_eq!(entry.name, "all");
    }

    #[test]
    fn select_entry_matches_a_role_name_when_the_hint_is_not_a_real_model_id() {
        // Lets orangu.conf skip knowing the real backend model id entirely:
        // a server section can just set `model = explorer` (the role) and
        // share the coordinator's endpoint; the coordinator alone decides
        // what model that role actually loads.
        let llms = test_llms();
        let entry = select_entry(&llms, "all", Some("explorer"), None, None);
        assert_eq!(entry.name, "explorer");
    }

    #[test]
    fn select_entry_prefers_a_real_model_id_match_over_a_role_name_match() {
        // If a hint happens to match both an entry's model and another
        // entry's role, the exact model id takes priority — it's the more
        // specific, unambiguous signal.
        let mut llms = test_llms();
        llms.insert(
            "literally-named-explorer".to_string(),
            CoordinatorLlmEntry {
                name: "literally-named-explorer".to_string(),
                role: "code".to_string(),
                model: "explorer".to_string(),
                host: "127.0.0.1".to_string(),
                port: 8300,
                backend: None,
                slots: None,
                web: None,
            },
        );
        let entry = select_entry(&llms, "all", Some("explorer"), None, None);
        assert_eq!(entry.name, "literally-named-explorer");
    }

    #[test]
    fn select_entry_prefers_the_implied_role_over_the_active_entry() {
        // /v1/embeddings (or any other request whose type implies a role)
        // must not be sent to whatever chat model happens to be active —
        // e.g. a coordinator mid-conversation on the `explorer` role must
        // still route a stray embeddings request to `embeddings`.
        let mut llms = test_llms();
        llms.insert(
            "embeddings".to_string(),
            CoordinatorLlmEntry {
                name: "embeddings".to_string(),
                role: "embeddings".to_string(),
                model: "org/embed".to_string(),
                host: "127.0.0.1".to_string(),
                port: 8400,
                backend: None,
                slots: None,
                web: None,
            },
        );
        let entry = select_entry(&llms, "all", None, Some("embeddings"), Some("explorer"));
        assert_eq!(entry.name, "embeddings");
    }

    #[test]
    fn select_entry_falls_back_when_implied_role_has_no_matching_profile() {
        let llms = test_llms();
        let entry = select_entry(&llms, "all", None, Some("embeddings"), Some("explorer"));
        assert_eq!(entry.name, "explorer");
    }

    #[test]
    fn select_entry_prefers_model_hint_over_implied_role() {
        // An explicit, matching model choice still wins over the request
        // type's implied role.
        let llms = test_llms();
        let entry = select_entry(&llms, "all", Some("org/qwen"), Some("embeddings"), None);
        assert_eq!(entry.name, "explorer");
    }

    #[test]
    fn match_hint_finds_by_model_id_or_role_name() {
        let llms = test_llms();
        assert_eq!(match_hint(&llms, "org/qwen").unwrap().name, "explorer");
        assert_eq!(match_hint(&llms, "explorer").unwrap().name, "explorer");
        assert!(match_hint(&llms, "nonexistent").is_none());
    }

    #[tokio::test]
    async fn coordinator_match_hint_returns_none_for_an_activation_hint_matching_nothing() {
        // Unlike ordinary routing, an explicit activation request has no
        // "currently active"/`all` fallback to paper over an unmatched hint.
        let config = minimal_config("", "[main]\nrole = all\nmodel = org/gemma\nport = 8100\n");
        let coordinator = Coordinator::new(config, None).unwrap();

        assert_eq!(coordinator.match_hint("org/gemma").unwrap().name, "main");
        assert_eq!(coordinator.match_hint("all").unwrap().name, "main");
        assert!(coordinator.match_hint("nonexistent-role").is_none());
    }

    #[test]
    fn resolve_server_binary_uses_the_override_when_given() {
        let config = minimal_config("", "[main]\nrole = all\nmodel = org/gemma\nport = 8100\n");
        let coordinator =
            Coordinator::new(config, Some(PathBuf::from("/opt/fake/orangu-server"))).unwrap();
        assert_eq!(
            coordinator.resolve_server_binary(),
            PathBuf::from("/opt/fake/orangu-server")
        );
    }

    #[test]
    fn write_server_config_includes_optional_overrides_only_when_set() {
        let entry = CoordinatorLlmEntry {
            name: "test-profile".to_string(),
            role: "all".to_string(),
            model: "org/gemma".to_string(),
            host: "127.0.0.1".to_string(),
            port: 8100,
            backend: Some("vulkan".to_string()),
            slots: Some(4),
            web: Some(8181),
        };
        let path = write_server_config(
            &entry,
            Path::new("/srv/models"),
            &orangu::logging::LogTarget::Console,
        )
        .unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("[orangu-server]"));
        assert!(contents.contains("models = /srv/models"));
        assert!(contents.contains("host = 127.0.0.1"));
        assert!(contents.contains("port = 8100"));
        assert!(contents.contains("backend = vulkan"));
        assert!(contents.contains("slots = 4"));
        assert!(contents.contains("[web]\nport = 8181"), "{contents}");
        // The console is the server's own default, so nothing is written
        // for it.
        assert!(!contents.contains("log_"), "{contents}");
        std::fs::remove_file(&path).ok();

        let minimal_entry = CoordinatorLlmEntry {
            backend: None,
            slots: None,
            web: None,
            ..entry
        };
        let path = write_server_config(
            &minimal_entry,
            Path::new("/srv/models"),
            &orangu::logging::LogTarget::Console,
        )
        .unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(!contents.contains("backend"));
        assert!(!contents.contains("slots"));
        assert!(!contents.contains("web ="));
        std::fs::remove_file(&path).ok();
    }

    /// A coordinator logging to a file hands the same file to the server it
    /// starts — otherwise the server would log into the pipe this
    /// coordinator reads, progress updates and all, and the file would get
    /// that as one line per request.
    #[test]
    fn write_server_config_forwards_a_file_log_to_the_server() {
        let entry = CoordinatorLlmEntry {
            name: "test-profile-logged".to_string(),
            role: "all".to_string(),
            model: "org/gemma".to_string(),
            host: "127.0.0.1".to_string(),
            port: 8100,
            backend: None,
            slots: None,
            web: None,
        };
        let path = write_server_config(
            &entry,
            Path::new("/srv/models"),
            &orangu::logging::LogTarget::File(PathBuf::from("/var/log/orangu.log")),
        )
        .unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(
            contents.contains("log_type = file\nlog_path = /var/log/orangu.log\n"),
            "{contents}"
        );
        std::fs::remove_file(&path).ok();
    }

    /// Whether a PID still refers to a live process, via signal 0 (sends no
    /// actual signal, just checks deliverability/existence).
    #[cfg(unix)]
    fn process_is_alive(pid: u32) -> bool {
        unsafe { libc::kill(pid as libc::pid_t, 0) == 0 }
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn shutdown_kills_a_process_still_waiting_on_its_health_check() {
        // Regression test: a real `orangu-server` that takes a long time to
        // load was leaked (orphaned, still running) if the coordinator was
        // shut down while `ensure_active` was still awaiting its health
        // check — `active` isn't populated until that check succeeds, so
        // `shutdown`'s old `active`-only cleanup had nothing to kill. The
        // `sleep 30` here stands in for a slow model load: nothing ever
        // listens on the configured port, so the health check keeps
        // failing (not timing out) until `shutdown` intervenes.
        let config = minimal_config(
            "startup_timeout = 30",
            "[main]\nrole = all\nmodel = org/gemma\nport = 65535\n",
        );
        let fake_bin = fake_server_script("sleep 30");
        let coordinator =
            std::sync::Arc::new(Coordinator::new(config, Some(fake_bin.clone())).unwrap());

        let entry = coordinator.resolve_entry(None, None).await.clone();
        let ensure_active_coordinator = coordinator.clone();
        let handle =
            tokio::spawn(async move { ensure_active_coordinator.ensure_active(&entry).await });

        // Let the health-check loop actually start (and record the pid)
        // before shutting down.
        let pid = loop {
            let pid = coordinator.current_pid.load(Ordering::Relaxed);
            if pid != 0 {
                break pid;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        };
        assert!(
            process_is_alive(pid),
            "test process should be running before shutdown"
        );

        coordinator.shutdown().await;

        let result = tokio::time::timeout(Duration::from_secs(5), handle).await;
        assert!(
            result.is_ok(),
            "ensure_active did not return after shutdown killed its process"
        );
        assert!(
            !process_is_alive(pid),
            "process {pid} leaked: still alive after shutdown"
        );
        std::fs::remove_file(&fake_bin).ok();
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn start_error_includes_captured_output_when_the_process_crashes() {
        // Regression coverage for a real report: a backend aborting used to
        // surface only a bare "status: signal: 6 (SIGABRT)" with no way to
        // tell why. The process's own stderr/stdout is now captured and
        // appended, so the actual diagnostic ends up in the same error.
        let config = minimal_config(
            "startup_timeout = 5",
            "[main]\nrole = all\nmodel = org/gemma\nport = 65534\n",
        );
        let fake_bin = fake_server_script("echo GGML_ASSERT failed >&2; exit 1");
        let coordinator = Coordinator::new(config, Some(fake_bin.clone())).unwrap();
        let entry = coordinator.resolve_entry(None, None).await.clone();

        let err = coordinator.ensure_active(&entry).await.unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("exited before becoming ready"),
            "{message}"
        );
        assert!(message.contains("GGML_ASSERT failed"), "{message}");
        std::fs::remove_file(&fake_bin).ok();
    }

    /// Answers `GET /health` with a pid that is not the caller's child — a
    /// leftover `orangu-server` still holding the port, which is what this is
    /// standing in for.
    #[cfg(unix)]
    async fn stranger_on_the_port(listener: tokio::net::TcpListener, pid: u32) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        while let Ok((mut stream, _)) = listener.accept().await {
            let mut scratch = [0u8; 1024];
            let _ = stream.read(&mut scratch).await;
            let body = format!("{{\"status\":\"ok\",\"pid\":{pid}}}");
            let _ = stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await;
        }
    }

    /// A stand-in `orangu-server`: answers `/props` with `model` and `role`,
    /// and `/health` with `pid`. Everything else gets `404`, which is what a
    /// coordinator asking anything else would deserve here.
    #[cfg(unix)]
    async fn standin_server(
        listener: tokio::net::TcpListener,
        model: String,
        role: String,
        pid: u32,
    ) {
        while let Ok((mut stream, _)) = listener.accept().await {
            let mut scratch = [0u8; 1024];
            let read = stream.read(&mut scratch).await.unwrap_or(0);
            let request = String::from_utf8_lossy(&scratch[..read]).to_string();
            let body = if request.contains("/props") {
                format!("{{\"model\":\"{model}\",\"role\":\"{role}\"}}")
            } else if request.contains("/health") {
                format!("{{\"status\":\"ok\",\"pid\":{pid}}}")
            } else {
                "{}".to_string()
            };
            let _ = stream
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\
                         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await;
        }
    }

    /// A server already serving what a profile asks for is used, not competed
    /// with.
    ///
    /// Starting a second one on the same address cannot work — it fails to
    /// bind after loading a whole model — so the only question is whether the
    /// coordinator finds that out by trying or by looking first. The marker
    /// file is what makes "did not try" checkable: the stand-in binary writes
    /// it when invoked, and adoption means it never is.
    #[tokio::test]
    #[cfg(unix)]
    async fn a_server_already_serving_this_profile_is_adopted_rather_than_restarted() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(standin_server(
            listener,
            "org/gemma".to_string(),
            // Not the profile's own role: `all` and `code` are one process,
            // and adoption has to know that or it restarts for nothing.
            "all".to_string(),
            4242,
        ));

        let marker = std::env::temp_dir().join(format!("orangu-adopt-{}", std::process::id()));
        std::fs::remove_file(&marker).ok();
        // Two profiles on one model and one port, which is the shape this is
        // about: `all` is required as the fallback, and `code` is what the
        // request resolves to.
        let config = minimal_config(
            "startup_timeout = 5",
            &format!(
                "[main]\nrole = all\nmodel = org/gemma\nport = {port}\n\
                 [coder]\nrole = code\nmodel = org/gemma\nport = {port}\n"
            ),
        );
        let fake_bin = fake_server_script(&format!("touch {}; sleep 30", marker.display()));
        let coordinator = Coordinator::new(config, Some(fake_bin.clone())).unwrap();
        let entry = coordinator.resolve_entry(Some("code"), None).await.clone();
        assert_eq!(
            entry.role, "code",
            "the request resolved to the code profile"
        );

        let origin = coordinator.ensure_active(&entry).await.unwrap();
        assert_eq!(origin, entry.origin());
        assert!(
            !marker.exists(),
            "the coordinator started a second orangu-server for a profile already being served"
        );

        std::fs::remove_file(&fake_bin).ok();
        std::fs::remove_file(&marker).ok();
    }

    /// A server on the profile's address that is *not* serving it has the
    /// address taken from it, rather than being reported as an obstacle.
    ///
    /// The same act as swapping away from an adopted server, and refusing it
    /// on the grounds that this coordinator did not start the incumbent is
    /// what left a leftover from an earlier run able to block every profile
    /// indefinitely.
    ///
    /// The incumbent here is a real child process whose pid the stand-in
    /// reports, because the takeover signals that pid — a fabricated one would
    /// have this test signalling a stranger.
    #[tokio::test]
    #[cfg(unix)]
    async fn a_server_serving_something_else_has_the_address_taken_from_it() {
        // `pid_is_orangu_server` is what gates the takeover, and it reads
        // `/proc`; on a platform without one there is nothing to assert.
        if !cfg!(target_os = "linux") {
            return;
        }
        // Named `orangu-server` so the kernel agrees it is one — the guard is
        // there so a `/health` answer cannot nominate an arbitrary victim.
        let incumbent_bin = fake_server_script("sleep 30");
        let renamed = incumbent_bin.with_file_name("orangu-server");
        std::fs::rename(&incumbent_bin, &renamed).unwrap();
        let mut incumbent = spawn_fake_server(&renamed);
        let pid = incumbent.id();

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(standin_server(
            listener,
            "org/something-else".to_string(),
            "all".to_string(),
            pid,
        ));

        let config = minimal_config(
            "startup_timeout = 5",
            &format!("[main]\nrole = all\nmodel = org/gemma\nport = {port}\n"),
        );
        let fake_bin = fake_server_script("sleep 30");
        let coordinator = Coordinator::new(config, Some(fake_bin.clone())).unwrap();
        let entry = coordinator.resolve_entry(None, None).await.clone();

        // Whether the *start* then succeeds is not what this is about — the
        // stand-in is still holding the socket, which no real incumbent would
        // be once stopped.
        let _ = coordinator.ensure_active(&entry).await;

        let deadline = Instant::now() + Duration::from_secs(10);
        let stopped = loop {
            match incumbent.try_wait() {
                Ok(Some(_)) => break true,
                _ if Instant::now() >= deadline => break false,
                _ => tokio::time::sleep(Duration::from_millis(50)).await,
            }
        };
        let _ = incumbent.kill();
        std::fs::remove_file(&renamed).ok();
        std::fs::remove_file(&fake_bin).ok();
        assert!(
            stopped,
            "the server holding this profile's address was left running"
        );
    }

    /// A port already held by something else must not read as "the child I
    /// just started is up".
    ///
    /// It did, and the consequence was not a failed start but a *silent wrong
    /// answer*: the health probe was satisfied by the leftover, the swap was
    /// recorded as done, and every request went to a process the coordinator
    /// never started — serving whatever model that one had loaded. The child
    /// meanwhile failed to bind and exited, so the next request found it dead
    /// and started the whole cycle again, one model load at a time, forever.
    #[tokio::test]
    #[cfg(unix)]
    async fn a_stranger_holding_the_port_is_reported_not_mistaken_for_the_child() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        // A pid no child of this test can have.
        tokio::spawn(stranger_on_the_port(listener, u32::MAX));

        let config = minimal_config(
            "startup_timeout = 5",
            &format!("[main]\nrole = all\nmodel = org/gemma\nport = {port}\n"),
        );
        // Alive and quiet: it never binds, exactly like a real server whose
        // own bind failed would be during the window this races.
        let fake_bin = fake_server_script("sleep 30");
        let coordinator = Coordinator::new(config, Some(fake_bin.clone())).unwrap();
        let entry = coordinator.resolve_entry(None, None).await.clone();

        let err = coordinator.ensure_active(&entry).await.unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("is already listening there"),
            "the leftover has to be named as the problem: {message}"
        );
        assert!(
            message.contains(&u32::MAX.to_string()),
            "and named by pid, so it can be found and stopped: {message}"
        );
        std::fs::remove_file(&fake_bin).ok();
    }

    #[tokio::test]
    #[cfg(unix)]
    async fn start_passes_the_config_path_role_flag_and_model_to_the_spawned_process() {
        // Confirms the argv shape `start` builds: `--config <generated
        // path> <role flag> <model>` — the marker script here echoes its
        // own argv back via stderr so the test can inspect exactly what
        // orangu-coordinator invoked it with.
        let config = minimal_config(
            "startup_timeout = 5",
            "[main]\nrole = all\nmodel = org/all-model\nport = 65531\n\n[marker]\nrole = explorer\nmodel = org/marker-model\nport = 65532\n",
        );
        let fake_bin = fake_server_script("echo \"argv=$*\" >&2; exit 1");
        let coordinator = Coordinator::new(config, Some(fake_bin.clone())).unwrap();
        let entry = coordinator
            .resolve_entry(Some("org/marker-model"), None)
            .await
            .clone();

        let err = coordinator.ensure_active(&entry).await.unwrap_err();
        let message = format!("{err:#}");
        assert!(message.contains("--config"), "{message}");
        assert!(message.contains("--explorer"), "{message}");
        assert!(message.contains("org/marker-model"), "{message}");
        std::fs::remove_file(&fake_bin).ok();
    }

    /// The exit status `orangu-server` uses for a lost GPU device is
    /// recognized as the deliberate step-aside it is — so the restart that
    /// follows is reported as a recovery rather than as a crash. Uses a real
    /// exited process rather than a hand-built `ExitStatus`, since that
    /// type's only honest constructor is a process that actually ran.
    #[tokio::test]
    async fn a_device_lost_exit_is_told_apart_from_a_crash() {
        let lost = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(format!("exit {SERVER_EXIT_DEVICE_LOST}"))
            .status()
            .await
            .unwrap();
        assert!(is_device_lost_exit(&lost));

        // Every other way out is not this: an ordinary failure, a clean
        // exit, and a signal (which has no exit code at all).
        for other in ["exit 1", "exit 0", "kill -TERM $$"] {
            let status = tokio::process::Command::new("sh")
                .arg("-c")
                .arg(other)
                .status()
                .await
                .unwrap();
            assert!(!is_device_lost_exit(&status), "{other} must not match");
        }
    }
}
