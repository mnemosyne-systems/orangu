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
//! local llama.cpp process. Rather than requiring one already-running
//! `llama-server` per role (as plain `orangu.conf` does), it starts and stops
//! `llama-server` on demand: every incoming request's JSON `model` field
//! picks which configured entry should be active, and the coordinator swaps
//! the running process if a different one is needed before forwarding the
//! request unchanged.

mod config;
mod init;
mod process;
mod proxy;

use anyhow::{Context, Result};
use axum::{Router, extract::DefaultBodyLimit};
use clap::Parser;
use config::{
    CoordinatorConfiguration, default_coordinator_config_path, load_coordinator_configuration,
};
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
#[command(version = VERSION)]
struct Args {
    /// Path to orangu-coordinator.conf. Defaults to ./orangu-coordinator.conf,
    /// then ~/.orangu/orangu-coordinator.conf.
    #[arg(short, long)]
    config: Option<PathBuf>,
    /// Interactively create ~/.orangu/orangu-coordinator.conf and exit.
    #[arg(short, long)]
    init: bool,
    /// Suppress all output (the startup banner, profile list, and shutdown
    /// message).
    #[arg(short, long)]
    quiet: bool,
    /// Detach from the terminal and run in the background. Implies --quiet:
    /// once detached there is no terminal left to print to. Unix-only.
    #[arg(short, long)]
    daemon: bool,
}

fn main() -> ExitCode {
    let mut args = Args::parse();

    if args.init {
        return match build_runtime().block_on(init::run_init()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("error: {err:#}");
                ExitCode::FAILURE
            }
        };
    }

    if args.daemon {
        args.quiet = true;
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
        Err(err) => {
            eprintln!("error: {err:#}");
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
    let coordinator = Arc::new(Coordinator::new(config, args.quiet)?);

    // Eagerly activate the `all`-role profile so the default model is
    // already loaded (or loading) by the time the first request arrives,
    // instead of every cold start paying that latency. Run it in the
    // background rather than awaiting it here: the listener below must
    // start accepting connections right away — `GET /v1/coordinator` in
    // particular is meant to answer instantly, even while a model is still
    // loading. A real request for a different role races this harmlessly:
    // `ensure_active` serializes on the same lock either way.
    let startup_coordinator = coordinator.clone();
    let quiet = args.quiet;
    tokio::spawn(async move {
        let entry = startup_coordinator.resolve_entry(None, None).await.clone();
        if let Err(err) = startup_coordinator.ensure_active(&entry).await
            && !quiet
        {
            eprintln!(
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
                    if !quiet {
                        println!(
                            "reloaded configuration from {}",
                            config_path_clone.display()
                        );
                    }
                    background_coordinator.reload_config(new_config);
                    // Reconcile: if the active profile was removed or its
                    // command changed, stop the now-stale process.
                    background_coordinator.stop_if_stale().await;
                } else if !quiet {
                    eprintln!("warning: failed to reload configuration; keeping previous state");
                }
            }

            // Check idle
            if let Some(timeout_secs) = background_coordinator.idle_timeout() {
                background_coordinator.unload_if_idle(timeout_secs).await;
            }
        }
    });

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

    if !args.quiet {
        println!("orangu-coordinator {VERSION} listening on {listen}");
        for (name, model) in profile_summary {
            println!("  {name}: {model}");
        }
    }

    let shutdown_coordinator = coordinator.clone();
    let result = tokio::select! {
        result = axum::serve(listener, app.into_make_service_with_connect_info::<std::net::SocketAddr>()) => result.context("server error"),
        _ = tokio::signal::ctrl_c() => {
            if !args.quiet {
                println!("shutting down...");
            }
            Ok(())
        },
        _ = shutdown_rx.recv() => {
            if !args.quiet {
                println!("shutting down via API...");
            }
            Ok(())
        }
    };

    shutdown_coordinator.shutdown().await;
    result
}
