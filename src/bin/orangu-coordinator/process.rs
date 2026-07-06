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

//! Owns the single llama.cpp child process orangu-coordinator manages: which
//! configured entry (if any) is currently running, and the start/stop/health
//! machinery to swap it for a different one on demand.

use crate::config::{CoordinatorConfiguration, CoordinatorLlmEntry};
use anyhow::{Context, Result, anyhow};
use std::{collections::HashMap, collections::VecDeque, process::Stdio, sync::Arc, time::Duration};
use tokio::{
    io::{AsyncBufReadExt, AsyncRead, BufReader},
    process::Command,
    sync::Mutex,
    time::Instant,
};

/// A running llama.cpp process for one configured entry.
struct ActiveProcess {
    entry_name: String,
    child: tokio::process::Child,
    /// Kept for the process's whole lifetime (not just while starting) so a
    /// crash discovered later — e.g. while it was actively serving a
    /// request — can still be reported with its own diagnostic output
    /// attached, the same as a startup failure.
    tail: OutputTail,
}

/// Number of most-recent stdout/stderr lines kept per starting/active
/// process, so a crash or a stuck health check can be reported with
/// llama.cpp's own diagnostic output attached instead of just a bare exit
/// signal or "timed out".
const OUTPUT_TAIL_LINES: usize = 20;

/// Rolling tail of a process's combined stdout/stderr output.
type OutputTail = Arc<Mutex<VecDeque<String>>>;

pub struct Coordinator {
    config: CoordinatorConfiguration,
    http_client: reqwest::Client,
    active: Mutex<Option<ActiveProcess>>,
    /// PID of whatever llama.cpp process is currently starting or active, if
    /// any. Set the instant a process is spawned — before its (possibly
    /// slow) health check even begins — and cleared once it's known to have
    /// stopped. `active` is held locked for an entire start sequence to
    /// serialize concurrent swaps, which can take up to `startup_timeout`;
    /// `shutdown` must not have to wait on that same lock just to kill a
    /// still-starting process, so this is tracked separately and only ever
    /// locked briefly.
    current_pid: Mutex<Option<u32>>,
    /// Suppresses echoing a starting/active process's stdout/stderr to the
    /// coordinator's own output, mirroring `--quiet`. The lines are still
    /// captured into each process's tail regardless, for error reporting.
    quiet: bool,
}

/// Per-attempt timeout for the `/v1/models` health-check probe only — kept
/// short so a genuinely stuck/unreachable process is detected quickly and
/// retried. This must never be the client's default timeout: the same
/// client also forwards real requests to the active backend, and real
/// generation can legitimately take far longer than a health check without
/// being stuck.
const HEALTH_CHECK_TIMEOUT: Duration = Duration::from_secs(5);

impl Coordinator {
    pub fn new(config: CoordinatorConfiguration, quiet: bool) -> Result<Self> {
        // No default timeout: this client also proxies real requests to
        // whichever backend is active, and those must be allowed to run for
        // as long as generation actually takes. A fixed default here would
        // silently cut off any response slower than it, tearing down the
        // connection to llama.cpp mid-stream — which surfaces to the client
        // as a bare "unexpected EOF", not a clear timeout error. The health
        // check applies its own short, explicit per-request timeout instead
        // (see `HEALTH_CHECK_TIMEOUT`).
        let http_client = reqwest::Client::builder()
            .build()
            .context("failed to build HTTP client")?;
        Ok(Self {
            config,
            http_client,
            active: Mutex::new(None),
            current_pid: Mutex::new(None),
            quiet,
        })
    }

    /// The shared HTTP client used both for readiness probes and, by the
    /// proxy handler, for forwarding requests to the active llama.cpp.
    pub fn http_client(&self) -> &reqwest::Client {
        &self.http_client
    }

    /// The model each conventional role resolves to, for `GET
    /// /v1/coordinator` — see [`CoordinatorConfiguration::models_by_role`].
    pub fn models_by_role(&self) -> Vec<(&str, &str)> {
        self.config.models_by_role()
    }

    /// Matches `hint` against a real model id, then a role name — see
    /// [`match_hint`]. Used by `POST /v1/coordinator/activate`, which
    /// (unlike ordinary request routing) has no "currently active" or
    /// `all`-role fallback: an explicit activation request that names
    /// nothing configured is a caller error, not something to paper over.
    pub fn match_hint(&self, hint: &str) -> Option<CoordinatorLlmEntry> {
        match_hint(&self.config.llms, hint).cloned()
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
    ///    the llama.cpp-native endpoints (`/v1/models`, `/health`, `/props`,
    ///    `/slots`, `/metrics`), which carry no `model` field to route on,
    ///    report on whatever is actually running instead of silently forcing
    ///    a swap back to `all` — e.g. `/information` probing a server's
    ///    `/health` would otherwise itself knock out whatever role a real
    ///    request had just switched to.
    /// 5. The `all`-role default entry.
    pub async fn resolve_entry(
        &self,
        model_hint: Option<&str>,
        implied_role: Option<&str>,
    ) -> &CoordinatorLlmEntry {
        let active_entry_name = self
            .active
            .lock()
            .await
            .as_ref()
            .map(|active| active.entry_name.clone());
        select_entry(
            &self.config.llms,
            &self.config.default_entry,
            model_hint,
            implied_role,
            active_entry_name.as_deref(),
        )
    }

    /// Ensures `entry`'s llama.cpp is the active process, starting it (and
    /// stopping whatever else was active) if it isn't already, then returns
    /// the origin requests should be proxied to.
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
            *self.current_pid.lock().await = None;
            Self::stop(guard.take().expect("checked above")).await;
        }

        let (child, tail) = self.start(entry).await?;
        *guard = Some(ActiveProcess {
            entry_name: entry.name.clone(),
            child,
            tail,
        });
        Ok(entry.origin())
    }

    /// Stops whatever llama.cpp process is currently starting or active, if
    /// any. Called on coordinator shutdown so no orphaned process is left
    /// running — including one still mid-startup (spawned, but not yet
    /// confirmed healthy), which `current_pid` catches and `active` alone
    /// would miss, since `active` isn't populated until the health check
    /// succeeds.
    pub async fn shutdown(&self) {
        if let Some(pid) = self.current_pid.lock().await.take() {
            kill_pid(pid);
        }
        let mut guard = self.active.lock().await;
        if let Some(mut active) = guard.take() {
            // Already signalled above (if it was this same process); this
            // just reaps it so it doesn't linger as a zombie.
            let _ = active.child.wait().await;
        }
    }

    async fn stop(mut active: ActiveProcess) {
        let _ = active.child.start_kill();
        let _ = active.child.wait().await;
    }

    async fn start(
        &self,
        entry: &CoordinatorLlmEntry,
    ) -> Result<(tokio::process::Child, OutputTail)> {
        let argv = shell_words::split(&entry.llamacpp)
            .with_context(|| format!("invalid llamacpp command for '{}'", entry.name))?;
        let (env_vars, argv) = split_leading_env_assignments(argv);
        let argv: Vec<String> = argv.iter().map(|token| expand_tilde(token)).collect();
        let (program, args) = argv
            .split_first()
            .ok_or_else(|| anyhow!("llamacpp command for '{}' is empty", entry.name))?;

        let mut child = Command::new(program)
            .args(args)
            .envs(env_vars)
            .stdin(std::process::Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| {
                format!(
                    "failed to start llama.cpp for '{}' ({})",
                    entry.name, entry.llamacpp
                )
            })?;

        // Record the PID *before* the (possibly long) health-check wait
        // below, so a concurrent `shutdown` can always kill this process,
        // no matter how long it takes to become ready.
        *self.current_pid.lock().await = child.id();

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
            *self.current_pid.lock().await = None;
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
        let deadline = Instant::now() + Duration::from_secs(self.config.startup_timeout_seconds);
        let probe_url = format!("{}/v1/models", entry.origin());

        loop {
            if let Ok(Some(status)) = child.try_wait() {
                return Err(anyhow!(
                    "llama.cpp for '{}' exited before becoming ready (status: {status}){}",
                    entry.name,
                    format_output_tail(tail).await
                ));
            }

            let mut request = self
                .http_client
                .get(&probe_url)
                .timeout(HEALTH_CHECK_TIMEOUT);
            if let Some(api_key) = &entry.api_key {
                request = request.bearer_auth(api_key);
            }
            if let Ok(response) = request.send().await
                && response.status().is_success()
            {
                return Ok(());
            }

            if Instant::now() >= deadline {
                return Err(anyhow!(
                    "timed out after {}s waiting for llama.cpp ('{}') to become ready at {}{}",
                    self.config.startup_timeout_seconds,
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
/// preserving today's visible behavior (e.g. `-hf` download progress) for
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

/// Splits a `llamacpp` argv into leading `KEY=VALUE` environment assignments
/// — the shell convention for setting one-off variables before a command,
/// e.g. `LLAMA_CACHE=/models llama-server ...` — and the command that
/// follows them. `llamacpp` is never run through a real shell (see the
/// module docs: that's what lets `shutdown` kill the exact process it
/// spawned), so this convention has to be recognized explicitly instead;
/// otherwise `LLAMA_CACHE=/models` would be tried as the program itself and
/// fail with "No such file or directory".
fn split_leading_env_assignments(mut argv: Vec<String>) -> (Vec<(String, String)>, Vec<String>) {
    let split_at = argv
        .iter()
        .position(|token| !is_env_assignment(token))
        .unwrap_or(argv.len());
    let command_argv = argv.split_off(split_at);
    let env_vars = argv
        .into_iter()
        .map(|token| {
            let (key, value) = token
                .split_once('=')
                .expect("is_env_assignment guarantees a '='");
            (key.to_string(), expand_tilde(value))
        })
        .collect();
    (env_vars, command_argv)
}

/// Expands a leading `~` or `~/...` to the user's home directory — a
/// convenience a real shell provides that `llamacpp` would otherwise lose,
/// since it's run directly and never through one (see
/// `split_leading_env_assignments`'s doc comment for why). `~otheruser`
/// forms are left untouched (rare, and not worth a passwd lookup). Falls
/// back to the token unchanged if the home directory can't be resolved.
fn expand_tilde(token: &str) -> String {
    let Some(home) = home::home_dir() else {
        return token.to_string();
    };
    if token == "~" {
        home.display().to_string()
    } else if let Some(rest) = token.strip_prefix("~/") {
        home.join(rest).display().to_string()
    } else {
        token.to_string()
    }
}

/// Whether `token` looks like a shell-style `KEY=VALUE` environment
/// assignment: text before the first `=` is a valid identifier (starts with
/// a letter or underscore, followed only by letters, digits, or
/// underscores). A flag like `--foo=bar` never matches, since `-` isn't a
/// valid identifier start, so it's safe to check every leading token this
/// way without risk of misreading a real argument as an assignment.
fn is_env_assignment(token: &str) -> bool {
    let Some((key, _)) = token.split_once('=') else {
        return false;
    };
    let mut chars = key.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
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
    llms: &'a HashMap<String, CoordinatorLlmEntry>,
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
    llms: &HashMap<String, CoordinatorLlmEntry>,
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
    llms: &'a HashMap<String, CoordinatorLlmEntry>,
    hint: &str,
) -> Option<&'a CoordinatorLlmEntry> {
    best_match(llms, |entry| entry.model == hint)
        .or_else(|| best_match(llms, |entry| entry.role == hint))
}

/// Sends an immediate, unconditional kill to a bare PID — used by
/// [`Coordinator::shutdown`] to terminate a still-starting llama.cpp process
/// that has no live `tokio::process::Child` handle left to call
/// `start_kill()` on (see `current_pid`'s doc comment). Best-effort: an
/// already-gone PID is simply a no-op.
#[cfg(unix)]
fn kill_pid(pid: u32) {
    unsafe {
        libc::kill(pid as libc::pid_t, libc::SIGKILL);
    }
}

#[cfg(not(unix))]
fn kill_pid(_pid: u32) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::load_coordinator_configuration;
    use std::io::Write;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

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

        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\n\n[main]\nrole = all\nllamacpp = llama-server -hf org/gemma --port 8100\n"
        )
        .unwrap();
        let config = load_coordinator_configuration(file.path()).unwrap();
        let coordinator = Coordinator::new(config, false).unwrap();

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

    #[test]
    fn is_env_assignment_recognizes_key_value_pairs_but_not_flags() {
        assert!(is_env_assignment("LLAMA_CACHE=/models"));
        assert!(is_env_assignment("_foo=bar"));
        assert!(is_env_assignment("FOO_2=bar"));
        // A flag's `--name=value` form must never be mistaken for an
        // assignment: `-` isn't a valid identifier start.
        assert!(!is_env_assignment("--hf-repo=org/gemma"));
        assert!(!is_env_assignment("-hf=org/gemma"));
        assert!(!is_env_assignment("llama-server"));
        assert!(!is_env_assignment("2FOO=bar")); // identifiers can't start with a digit
        assert!(!is_env_assignment(""));
    }

    #[test]
    fn split_leading_env_assignments_separates_them_from_the_command() {
        let argv = shell_words::split(
            "LLAMA_CACHE=/models LLAMA_ARG_N_GPU_LAYERS=999 llama-server -hf org/gemma --port 8100",
        )
        .unwrap();
        let (env_vars, command_argv) = split_leading_env_assignments(argv);
        assert_eq!(
            env_vars,
            vec![
                ("LLAMA_CACHE".to_string(), "/models".to_string()),
                ("LLAMA_ARG_N_GPU_LAYERS".to_string(), "999".to_string()),
            ]
        );
        assert_eq!(
            command_argv,
            vec!["llama-server", "-hf", "org/gemma", "--port", "8100"]
        );
    }

    #[test]
    fn split_leading_env_assignments_is_a_no_op_without_any() {
        let argv = shell_words::split("llama-server -hf org/gemma --port 8100").unwrap();
        let (env_vars, command_argv) = split_leading_env_assignments(argv.clone());
        assert!(env_vars.is_empty());
        assert_eq!(command_argv, argv);
    }

    #[test]
    fn expand_tilde_replaces_a_leading_tilde_with_the_home_directory() {
        // Regression test: `llamacpp` is never run through a real shell (see
        // `split_leading_env_assignments`'s doc comment), so `~` in an
        // argument like `--slot-save-path ~/.orangu/llama-slots` was passed
        // to llama-server literally, which doesn't do tilde expansion
        // itself and failed with "not a directory: ~/.orangu/llama-slots".
        let home = home::home_dir().expect("test environment has a home directory");
        assert_eq!(expand_tilde("~"), home.display().to_string());
        assert_eq!(
            expand_tilde("~/.orangu/llama-slots"),
            home.join(".orangu/llama-slots").display().to_string()
        );
    }

    #[test]
    fn expand_tilde_leaves_other_tokens_untouched() {
        assert_eq!(expand_tilde("--port"), "--port");
        assert_eq!(expand_tilde("/models"), "/models");
        // `~otheruser` forms are intentionally left alone (rare, and would
        // need a passwd lookup to resolve correctly).
        assert_eq!(expand_tilde("~otheruser/models"), "~otheruser/models");
    }

    #[test]
    fn split_leading_env_assignments_expands_a_tilde_in_the_value() {
        let argv = shell_words::split(
            "LLAMA_CACHE=~/.cache/llama.cpp llama-server -hf org/gemma --port 8100",
        )
        .unwrap();
        let (env_vars, _) = split_leading_env_assignments(argv);
        let home = home::home_dir().expect("test environment has a home directory");
        assert_eq!(
            env_vars,
            vec![(
                "LLAMA_CACHE".to_string(),
                home.join(".cache/llama.cpp").display().to_string()
            )]
        );
    }

    #[tokio::test]
    async fn resolve_entry_falls_back_to_default_when_model_hint_is_absent_or_unknown() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\n\n[main]\nrole = all\nllamacpp = llama-server -hf org/gemma --port 8100\n"
        )
        .unwrap();
        let config = load_coordinator_configuration(file.path()).unwrap();
        let coordinator = Coordinator::new(config, false).unwrap();

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
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\n\n[zeta]\nrole = explorer\nllamacpp = llama-server -hf org/gemma --port 8200\n\n[alpha]\nrole = all\nllamacpp = llama-server -hf org/gemma --port 8100\n"
        )
        .unwrap();
        let config = load_coordinator_configuration(file.path()).unwrap();
        let coordinator = Coordinator::new(config, false).unwrap();

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
                llamacpp: "llama-server -hf org/gemma --port 8100".to_string(),
                api_key: None,
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
                llamacpp: "llama-server -hf org/qwen --port 8200".to_string(),
                api_key: None,
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
                llamacpp: "llama-server -hf org/explorer --port 8300".to_string(),
                api_key: None,
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
                llamacpp: "llama-server -hf org/embed --port 8400".to_string(),
                api_key: None,
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
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\n\n[main]\nrole = all\nllamacpp = llama-server -hf org/gemma --port 8100\n"
        )
        .unwrap();
        let config = load_coordinator_configuration(file.path()).unwrap();
        let coordinator = Coordinator::new(config, false).unwrap();

        assert_eq!(coordinator.match_hint("org/gemma").unwrap().name, "main");
        assert_eq!(coordinator.match_hint("all").unwrap().name, "main");
        assert!(coordinator.match_hint("nonexistent-role").is_none());
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
        // Regression test: a real llama.cpp process that takes a long time to
        // load was leaked (orphaned, still running) if the coordinator was
        // shut down while `ensure_active` was still awaiting its health
        // check — `active` isn't populated until that check succeeds, so
        // `shutdown`'s old `active`-only cleanup had nothing to kill. The
        // `sleep 30` here stands in for a slow model load: nothing ever
        // listens on the configured port, so the health check keeps
        // failing (not timing out) until `shutdown` intervenes.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nstartup_timeout = 30\n\n[main]\nrole = all\nllamacpp = sh -c \"sleep 30\" -hf org/gemma --port 65535\n"
        )
        .unwrap();
        let config = load_coordinator_configuration(file.path()).unwrap();
        let coordinator = std::sync::Arc::new(Coordinator::new(config, false).unwrap());

        let entry = coordinator.resolve_entry(None, None).await.clone();
        let ensure_active_coordinator = coordinator.clone();
        let handle =
            tokio::spawn(async move { ensure_active_coordinator.ensure_active(&entry).await });

        // Let the health-check loop actually start (and record the pid)
        // before shutting down.
        let pid = loop {
            if let Some(pid) = *coordinator.current_pid.lock().await {
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
    }

    #[tokio::test]
    async fn start_error_includes_captured_output_when_the_process_crashes() {
        // Regression coverage for a real report: llama.cpp aborting (e.g.
        // SIGABRT on an assertion failure) used to surface only a bare
        // "status: signal: 6 (SIGABRT)" with no way to tell why. The
        // process's own stderr/stdout is now captured and appended, so the
        // actual diagnostic (here standing in for e.g. a GGML assertion
        // message) ends up in the same error.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nstartup_timeout = 5\n\n[main]\nrole = all\nllamacpp = sh -c \"echo GGML_ASSERT failed >&2; exit 1\" -hf org/gemma --port 65534\n"
        )
        .unwrap();
        let config = load_coordinator_configuration(file.path()).unwrap();
        let coordinator = Coordinator::new(config, true).unwrap();
        let entry = coordinator.resolve_entry(None, None).await.clone();

        let err = coordinator.ensure_active(&entry).await.unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("exited before becoming ready"),
            "{message}"
        );
        assert!(message.contains("GGML_ASSERT failed"), "{message}");
    }

    #[tokio::test]
    async fn start_expands_a_tilde_in_llamacpp_arguments() {
        // Regression test for a real report: `--slot-save-path
        // ~/.orangu/llama-slots` failed with "not a directory:
        // ~/.orangu/llama-slots" because llama.cpp itself doesn't expand
        // `~` — only a real shell does, and `llamacpp` is never run through
        // one. `$1` here echoes back exactly what the spawned process
        // received, proving the expansion happened on our side before the
        // process ever saw the argument.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        writeln!(
            file,
            "[orangu-coordinator]\nstartup_timeout = 5\n\n[main]\nrole = all\nllamacpp = sh -c 'echo \"marker=$1\" >&2; exit 1' _ ~/orangu-tilde-test-marker -hf org/gemma --port 65533\n"
        )
        .unwrap();
        let config = load_coordinator_configuration(file.path()).unwrap();
        let coordinator = Coordinator::new(config, true).unwrap();
        let entry = coordinator.resolve_entry(None, None).await.clone();

        let err = coordinator.ensure_active(&entry).await.unwrap_err();
        let message = format!("{err:#}");
        let home = home::home_dir().expect("test environment has a home directory");
        assert!(
            message.contains(&format!(
                "marker={}/orangu-tilde-test-marker",
                home.display()
            )),
            "{message}"
        );
        assert!(!message.contains("marker=~"), "{message}");
    }
}
