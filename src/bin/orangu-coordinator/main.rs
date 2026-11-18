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

//! `orangu-coordinator` is a small HTTP proxy that sits in front of a single
//! local `orangu-server` process. Rather than requiring one already-running
//! `orangu-server` per role (as plain `orangu.conf` does), it starts and
//! stops `orangu-server` on demand: every incoming request's JSON `model`
//! field picks which configured entry should be active, and the coordinator
//! swaps the running process (via that entry's own role flag — `--all`/
//! `--code`/`--review`/`--explorer`/`--embedding` — and model) if a
//! different one is needed before forwarding the request unchanged.
//!
//! `ORANGU_COORDINATOR_SERVER_BIN`, if set, overrides which `orangu-server`
//! executable is spawned (see `process::Coordinator::resolve_server_binary`)
//! — otherwise a sibling `orangu-server` next to this binary's own
//! executable is used, falling back to `PATH`.
//!
//! Everything this process says while running goes through `log` — see
//! `orangu::logging` for where that ends up: the console unless
//! `[orangu-coordinator].log_type = file` sends it to `log_path`, which is
//! what a `--daemon` run wants. The fatal `error:` lines `main` exits on stay
//! on stderr regardless — they are for the terminal that started this.

mod config;
mod crashes;
mod init;
mod mover;
mod process;
mod proxy;
mod shell;

use anyhow::{Context, Result};
use axum::{Router, extract::DefaultBodyLimit};
use clap::Parser;
use config::{
    CoordinatorConfiguration, default_coordinator_config_path, load_coordinator_configuration,
};
use orangu::shell_completions;
use process::Coordinator;
use std::{path::PathBuf, process::ExitCode, sync::Arc};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

const TERMINAL_TITLE: &str = "orangu-coordinator";

/// Sets the terminal window/tab title via the standard OSC 0 escape
/// sequence (supported by essentially every modern terminal emulator), and
/// restores it (clears it back) on drop. Mirrors `orangu`'s own
/// `TerminalTitleGuard` — skipped entirely in daemon mode, where there is no
/// attached terminal to write to.
struct TerminalTitleGuard;

impl TerminalTitleGuard {
    fn new(title: &str) -> Self {
        print!("\x1b]0;{title}\x07");
        Self
    }
}

impl Drop for TerminalTitleGuard {
    fn drop(&mut self) {
        print!("\x1b]0;\x07");
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "orangu-coordinator",
    version = orangu::build_info::id(),
    about = "Start orangu-server instances on demand and route to them"
)]
struct Args {
    /// Path to orangu-coordinator.conf. Defaults to ./orangu-coordinator.conf,
    /// then ~/.orangu/orangu-coordinator.conf.
    #[arg(short, long)]
    config: Option<PathBuf>,
    /// Interactively create ~/.orangu/orangu-coordinator.conf and exit.
    #[arg(short, long)]
    init: bool,
    /// Suppress all console output (the startup banner, profile list, and
    /// shutdown message). A log file is unaffected.
    #[arg(short, long)]
    quiet: bool,
    /// Detach from the terminal and run in the background. Unix-only. With
    /// the console as the log there is nothing left to print to, so it
    /// implies --quiet; with log_type = file the log is written as usual.
    #[arg(short, long)]
    daemon: bool,
    /// Print the shell completion script for the detected shell and exit.
    #[arg(short = 's', long = "shell-completions")]
    shell_completions: bool,
}

/// This binary's three completion scripts, for the shared
/// `-s`/`--shell-completions` detection in `orangu::shell_completions`.
const COMPLETION_SCRIPTS: shell_completions::Scripts = shell_completions::Scripts {
    bash: shell::BASH,
    zsh: shell::ZSH,
    fish: shell::FISH,
    powershell: shell::POWERSHELL,
};

fn main() -> ExitCode {
    let args = Args::parse();

    if args.shell_completions {
        return match shell_completions::print("orangu-coordinator", &COMPLETION_SCRIPTS) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("error: {err:#}");
                ExitCode::FAILURE
            }
        };
    }

    if args.init {
        return match build_runtime().block_on(init::run_init()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("error: {err:#}");
                ExitCode::FAILURE
            }
        };
    }

    // Resolve the config and bind the listener synchronously, before either
    // daemonizing or starting the async runtime: daemonizing forks the
    // process, which is only safe to do before any additional OS threads
    // exist (the tokio runtime below spawns a pool of them), and doing it
    // before the listener is bound would send any "address already in use"
    // or bad-config error to `/dev/null` instead of the caller's terminal.
    let config_path = match args.config.clone().or_else(default_coordinator_config_path) {
        Some(path) => path,
        None => {
            eprintln!(
                "error: Missing config file; pass --config or add ./orangu-coordinator.conf or ~/.orangu/orangu-coordinator.conf"
            );
            return ExitCode::FAILURE;
        }
    };
    let config = match load_coordinator_configuration(&config_path) {
        Ok(config) => config,
        Err(err) => {
            eprintln!("error: {err:#}");
            return ExitCode::FAILURE;
        }
    };
    // The logger, as soon as there is a config to say where it goes and
    // before anything worth logging happens. Both flags are about the
    // console: a `--daemon` with the console as its log has nowhere to
    // write and logs nothing at all, `--quiet` keeps the errors — while a
    // log file is written exactly as an attached run would write it, which
    // is the point of having one. An unwritable `log_path` fails here, on
    // the terminal.
    let console = if args.daemon {
        orangu::logging::Console::Nothing
    } else if args.quiet {
        orangu::logging::Console::ErrorsOnly
    } else {
        orangu::logging::Console::Everything
    };
    if let Err(err) = orangu::logging::install(&config.log, console, "orangu_coordinator") {
        eprintln!("error: {err:#}");
        return ExitCode::FAILURE;
    }
    let listen = config.listen_addr();
    let std_listener = match std::net::TcpListener::bind(&listen) {
        Ok(listener) => listener,
        Err(err) => {
            eprintln!("error: failed to bind {listen}: {err}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(err) = std_listener.set_nonblocking(true) {
        eprintln!("error: failed to configure listener on {listen}: {err}");
        return ExitCode::FAILURE;
    }

    if args.daemon
        && let Err(err) = daemonize()
    {
        eprintln!("error: failed to start as a daemon: {err:#}");
        return ExitCode::FAILURE;
    }

    match build_runtime().block_on(run(args, config, std_listener, config_path)) {
        Ok(()) => ExitCode::SUCCESS,
        // Through the logger, not `eprintln!`: by now a daemon's stderr is
        // `/dev/null`, and a serving failure is exactly the line a log file
        // exists to hold. On the console it lands on stderr as before.
        Err(err) => {
            log::error!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn build_runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime")
}

/// Detach from the controlling terminal and continue running in the
/// background. Only the final, fully-detached process returns `Ok(())`; the
/// original (and an intermediate) process exit here and never return.
#[cfg(unix)]
fn daemonize() -> Result<()> {
    daemonize::Daemonize::new()
        .start()
        .map_err(|err| anyhow::anyhow!(err))
}

#[cfg(not(unix))]
fn daemonize() -> Result<()> {
    Err(anyhow::anyhow!(
        "--daemon is only supported on Unix-like platforms"
    ))
}

async fn run(
    args: Args,
    config: CoordinatorConfiguration,
    std_listener: std::net::TcpListener,
    config_path: PathBuf,
) -> Result<()> {
    let _terminal_title_guard = (!args.daemon).then(|| TerminalTitleGuard::new(TERMINAL_TITLE));
    let listen = config.listen_addr();
    let max_body_bytes = config.max_body_bytes;

    let mut profile_summary: Vec<(String, String)> = config
        .llms
        .values()
        .map(|entry| (entry.name.clone(), entry.model.clone()))
        .collect();
    profile_summary.sort();

    let (shutdown_tx, mut shutdown_rx) = tokio::sync::mpsc::channel::<()>(1);
    let server_binary_override =
        std::env::var_os("ORANGU_COORDINATOR_SERVER_BIN").map(PathBuf::from);
    let coordinator = Arc::new(Coordinator::new(config, server_binary_override)?);

    // Eagerly activate the `all`-role profile so the default model is
    // already loaded (or loading) by the time the first request arrives,
    // instead of every cold start paying that latency. Run it in the
    // background rather than awaiting it here: the listener below must
    // start accepting connections right away — `GET /v1/coordinator` in
    // particular is meant to answer instantly, even while a model is still
    // loading. A real request for a different role races this harmlessly:
    // `ensure_active` serializes on the same lock either way.
    let startup_coordinator = coordinator.clone();
    tokio::spawn(async move {
        let entry = startup_coordinator.resolve_entry(None, None).await.clone();
        if let Err(err) = startup_coordinator.ensure_active(&entry).await {
            log::warn!(
                "warning: failed to start default profile '{}' at startup: {err:#}",
                entry.name
            );
        }
    });

    let background_coordinator = coordinator.clone();
    let config_path_clone = config_path.clone();
    tokio::spawn(async move {
        let mut last_modified = tokio::fs::metadata(&config_path_clone)
            .await
            .ok()
            .and_then(|m| m.modified().ok());
        loop {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;

            // Check config
            if let Ok(metadata) = tokio::fs::metadata(&config_path_clone).await
                && let Ok(modified) = metadata.modified()
                && Some(modified) != last_modified
            {
                last_modified = Some(modified);
                if let Ok(new_config) = load_coordinator_configuration(&config_path_clone) {
                    log::info!(
                        "reloaded configuration from {}",
                        config_path_clone.display()
                    );
                    background_coordinator.reload_config(new_config);
                    // Reconcile: if the active profile was removed or its
                    // command changed, stop the now-stale process.
                    background_coordinator.stop_if_stale().await;
                } else {
                    log::warn!("warning: failed to reload configuration; keeping previous state");
                }
            }

            // Check idle
            if let Some(timeout_secs) = background_coordinator.idle_timeout() {
                background_coordinator.unload_if_idle(timeout_secs).await;
            }
        }
    });

    // With a `[workers]` section, moves this machine to the model its
    // channel wants (`mover`); idle otherwise, so a reload can add one.
    tokio::spawn(mover::run(coordinator.clone()));

    let app = Router::new()
        .route(
            "/v1/coordinator",
            axum::routing::get(proxy::coordinator_info),
        )
        .route(
            "/v1/coordinator/activate",
            axum::routing::post(proxy::activate),
        )
        .route(
            "/v1/coordinator/shutdown",
            axum::routing::get({
                let tx = shutdown_tx.clone();
                let shutdown_coordinator = coordinator.clone();
                move |connect_info: axum::extract::ConnectInfo<std::net::SocketAddr>,
                      query: axum::extract::Query<std::collections::HashMap<String, String>>| async move {
                    // Defense-in-depth: reject non-loopback even with a valid token.
                    if !connect_info.0.ip().is_loopback() {
                        return (axum::http::StatusCode::FORBIDDEN, "shutdown is only available from localhost\n");
                    }
                    // Require a matching shutdown_token from config.
                    let Some(expected) = shutdown_coordinator.shutdown_token() else {
                        return (axum::http::StatusCode::NOT_FOUND, "shutdown endpoint is disabled; set shutdown_token in config to enable\n");
                    };
                    let provided = query.get("token").map(|s| s.as_str()).unwrap_or("");
                    if provided != expected {
                        return (axum::http::StatusCode::FORBIDDEN, "invalid shutdown token\n");
                    }
                    let _ = tx.send(()).await;
                    (axum::http::StatusCode::OK, "shutting down...\n")
                }
            }),
        )
        .fallback(proxy::proxy)
        .layer(DefaultBodyLimit::max(max_body_bytes))
        .with_state(coordinator.clone());

    let listener = tokio::net::TcpListener::from_std(std_listener)
        .context("failed to hand the bound listener off to the async runtime")?;

    log::info!(
        "orangu-coordinator {} listening on {listen}",
        orangu::build_info::id()
    );
    for (name, model) in profile_summary {
        log::info!("  {name}: {model}");
    }

    let shutdown_coordinator = coordinator.clone();
    let result = tokio::select! {
        result = axum::serve(listener, app.into_make_service_with_connect_info::<std::net::SocketAddr>()) => result.context("server error"),
        _ = terminated() => {
            log::info!("shutting down...");
            Ok(())
        },
        _ = shutdown_rx.recv() => {
            log::info!("shutting down via API...");
            Ok(())
        }
    };

    shutdown_coordinator.shutdown().await;
    result
}

/// Resolves when this process is asked to stop — `Ctrl+C` anywhere, and
/// `SIGTERM` as well on Unix.
///
/// `SIGTERM` is what `kill` sends by default, what a service manager sends,
/// and what a closing terminal leads to. Waiting only on `Ctrl+C` meant every
/// one of those killed the coordinator *without* running the shutdown that
/// stops its `orangu-server` — and a server that outlives its coordinator goes
/// on holding the port, so the next coordinator's children cannot bind and are
/// restarted forever while requests are answered by the leftover.
async fn terminated() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        // A handler that cannot be installed is not a reason to refuse to
        // run: fall back to `Ctrl+C` alone, which is where this started.
        let Ok(mut term) = signal(SignalKind::terminate()) else {
            let _ = tokio::signal::ctrl_c().await;
            return;
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = term.recv() => {},
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    /// Every flag clap parses is offered by all three hand-written
    /// completion scripts — see `orangu::shell_completions::unoffered`.
    #[test]
    fn every_flag_is_offered_by_every_completion_script() {
        let missing = shell_completions::unoffered(&Args::command(), &COMPLETION_SCRIPTS);
        assert!(missing.is_empty(), "not offered: {missing:?}");
    }

    #[test]
    fn completion_script_detects_each_supported_shell() {
        for (shell, marker) in [
            ("/bin/bash", "bash completion for orangu-coordinator"),
            ("/usr/bin/zsh", "#compdef orangu-coordinator"),
            (
                "/usr/local/bin/fish",
                "fish completion for orangu-coordinator",
            ),
            ("pwsh", "PowerShell completion for orangu-coordinator"),
            (
                r"C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe",
                "PowerShell completion for orangu-coordinator",
            ),
        ] {
            let script =
                shell_completions::script("orangu-coordinator", shell, false, &COMPLETION_SCRIPTS)
                    .expect(shell);
            assert!(script.contains(marker), "{shell} -> {marker}");
        }
    }
}
