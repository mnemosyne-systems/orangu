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
struct ActiveProcess {
    entry_name: String,
    child: tokio::process::Child,
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

/// Number of most-recent stdout/stderr lines kept per starting/active
/// process, so a crash or a stuck health check can be reported with
/// `orangu-server`'s own diagnostic output attached instead of just a bare
/// exit signal or "timed out".
const OUTPUT_TAIL_LINES: usize = 20;

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
    /// Suppresses echoing a starting/active process's stdout/stderr to the
    /// coordinator's own output, mirroring `--quiet`. The lines are still
    /// captured into each process's tail regardless, for error reporting.
    quiet: bool,
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
        quiet: bool,
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
            quiet,
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
            if !self.quiet {
                println!(
                    "stopping '{}' — profile was removed or changed in reloaded config",
                    active.entry_name
                );
            }
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
    /// to `127.0.0.1`/`8100` when a profile's own config omits them): by
    /// the time the new `orangu-server` tries to bind that address, the old
    /// one's listening socket has already been released, not merely asked
    /// to release it.
    pub async fn ensure_active(&self, entry: &CoordinatorLlmEntry) -> Result<String> {
        let mut guard = self.active.lock().await;

        if let Some(active) = guard.as_mut() {
            if active.entry_name == entry.name {
                // Already the active model — but confirm the process is
                // still alive; a crashed backend must be restarted rather
                // than silently proxied into.
                match active.child.try_wait() {
                    Ok(None) => return Ok(entry.origin()),
                    Ok(Some(status)) if !self.quiet => {
                        eprintln!(
                            "warning: '{}' exited unexpectedly while active (status: {status}){}",
                            entry.name,
                            format_output_tail(&active.tail).await
                        );
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

        let (child, tail) = self.start(entry).await?;
        *guard = Some(ActiveProcess {
            entry_name: entry.name.clone(),
            entry_at_start: entry.clone(),
            child,
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
                if !self.quiet {
                    println!(
                        "unloading active profile '{}' due to idle timeout",
                        active.entry_name
                    );
                }
                self.current_pid.store(0, Ordering::Relaxed);
                Self::stop(active).await;
            }
        }
    }

    async fn stop(mut active: ActiveProcess) {
        #[cfg(unix)]
        {
            if let Some(pid) = active.child.id() {
                kill_pid(pid);
            }
            // Wait up to 5 seconds for graceful shutdown
            if tokio::time::timeout(Duration::from_secs(5), active.child.wait())
                .await
                .is_err()
            {
                let _ = active.child.start_kill();
                let _ = active.child.wait().await;
            }
        }
        #[cfg(not(unix))]
        {
            let _ = active.child.start_kill();
            let _ = active.child.wait().await;
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
        let models_dir = self.config.read().unwrap().models.clone();
        let server_config_path = write_server_config(entry, &models_dir).with_context(|| {
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
        let mut child = Command::new(&program)
            .arg("--config")
            .arg(&server_config_path)
            .arg(role_flag)
            .arg(&entry.model)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| {
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
        if let Some(stdout) = child.stdout.take() {
            spawn_output_capture(stdout, tail.clone(), self.quiet);
        }
        if let Some(stderr) = child.stderr.take() {
            spawn_output_capture(stderr, tail.clone(), self.quiet);
        }

        if let Err(err) = self.wait_until_healthy(entry, &mut child, &tail).await {
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
    ) -> Result<()> {
        let startup_timeout = self.config.read().unwrap().startup_timeout_seconds;
        let deadline = Instant::now() + Duration::from_secs(startup_timeout);
        let probe_url = format!("{}/v1/models", entry.origin());

        loop {
            if let Ok(Some(status)) = child.try_wait() {
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
            if let Ok(response) = request.send().await
                && response.status().is_success()
            {
                return Ok(());
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

/// Reads `stream` line by line for as long as the process keeps it open,
/// keeping the last [`OUTPUT_TAIL_LINES`] in `tail` and, unless `quiet`,
/// echoing each line to the coordinator's own stdout as it arrives —
/// preserving today's visible behavior (e.g. model-download progress) for
/// anyone watching the coordinator's console, while still letting a later
/// crash or stuck health check report the same output inline.
fn spawn_output_capture(
    stream: impl AsyncRead + Send + Unpin + 'static,
    tail: OutputTail,
    quiet: bool,
) {
    tokio::spawn(async move {
        let mut lines = BufReader::new(stream).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if !quiet {
                println!("{line}");
            }
            let mut tail = tail.lock().await;
            if tail.len() >= OUTPUT_TAIL_LINES {
                tail.pop_front();
            }
            tail.push_back(line);
        }
    });
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

/// Writes `entry`'s own `orangu-server.conf` (just the `[orangu-server]`
/// section: `models`, `host`, `port`, and whichever of `backend`/`slots`/
/// `web` were set) to `~/.orangu/coordinator/servers/<name>.conf`,
/// overwriting any previous contents — `orangu-server` itself reads this
/// file once at its own startup, so a stale file from a previous run is
/// never an issue, and this path doubles as a debugging aid: exactly what a
/// profile was last started with is always inspectable on disk.
fn write_server_config(entry: &CoordinatorLlmEntry, models_dir: &Path) -> Result<PathBuf> {
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
    if let Some(web) = entry.web {
        contents.push_str(&format!("web = {web}\n"));
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
/// already-gone PID is simply a no-op.
#[cfg(unix)]
fn kill_pid(pid: u32) {
    unsafe {
        libc::kill(pid as libc::pid_t, libc::SIGINT);
    }
}

#[cfg(not(unix))]
fn kill_pid(_pid: u32) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::load_coordinator_configuration;
    use std::collections::HashMap;
    use std::io::Write;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

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
    #[cfg(unix)]
    fn fake_server_script(body: &str) -> PathBuf {
        use std::os::unix::fs::PermissionsExt;
        let mut file = tempfile::NamedTempFile::new().unwrap();
        write!(file, "#!/bin/sh\n{body}").unwrap();
        let path = file.into_temp_path().keep().unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path
    }

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
        let coordinator = Coordinator::new(config, false, None).unwrap();

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
        let coordinator = Coordinator::new(config, false, None).unwrap();

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
        let coordinator = Coordinator::new(config, false, None).unwrap();

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
        let coordinator = Coordinator::new(config, false, None).unwrap();

        assert_eq!(coordinator.match_hint("org/gemma").unwrap().name, "main");
        assert_eq!(coordinator.match_hint("all").unwrap().name, "main");
        assert!(coordinator.match_hint("nonexistent-role").is_none());
    }

    #[test]
    fn resolve_server_binary_uses_the_override_when_given() {
        let config = minimal_config("", "[main]\nrole = all\nmodel = org/gemma\nport = 8100\n");
        let coordinator = Coordinator::new(
            config,
            false,
            Some(PathBuf::from("/opt/fake/orangu-server")),
        )
        .unwrap();
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
        let path = write_server_config(&entry, Path::new("/srv/models")).unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.contains("[orangu-server]"));
        assert!(contents.contains("models = /srv/models"));
        assert!(contents.contains("host = 127.0.0.1"));
        assert!(contents.contains("port = 8100"));
        assert!(contents.contains("backend = vulkan"));
        assert!(contents.contains("slots = 4"));
        assert!(contents.contains("web = 8181"));
        std::fs::remove_file(&path).ok();

        let minimal_entry = CoordinatorLlmEntry {
            backend: None,
            slots: None,
            web: None,
            ..entry
        };
        let path = write_server_config(&minimal_entry, Path::new("/srv/models")).unwrap();
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(!contents.contains("backend"));
        assert!(!contents.contains("slots"));
        assert!(!contents.contains("web ="));
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
            std::sync::Arc::new(Coordinator::new(config, false, Some(fake_bin.clone())).unwrap());

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
        let coordinator = Coordinator::new(config, true, Some(fake_bin.clone())).unwrap();
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
        let coordinator = Coordinator::new(config, true, Some(fake_bin.clone())).unwrap();
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
}
