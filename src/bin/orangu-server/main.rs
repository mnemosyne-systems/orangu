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

//! `orangu-server <model>`: loads a GGUF model and serves a llama.cpp-
//! compatible HTTP API. Also the machine's one-stop GGUF inventory tool —
//! `system`/`suggest`/`list`/`show`/`download` answer the questions that
//! matter when *getting* and *choosing* a model to run, before any serving
//! starts (formerly the separate `orangu-gguf` binary, folded in here so
//! there's one tool, one config file, and one shell-completion script to
//! keep in sync with the models directory convention both jobs share).

// The `Send`/`Sync` proof for the web UI's SSE stream walks the whole wgpu
// type graph the Vulkan backend pulls in, which sits right at the default
// 128-step auto-trait recursion limit — deep enough that adding a single
// field to `web::WebState` overflows it (in the test build only, where the
// stream type is monomorphized again). Raising it is rustc's own suggested
// remedy and costs nothing but a deeper proof search.
#![recursion_limit = "256"]

mod config;
mod engine;
mod http;
mod init;
mod panic_capture;
mod prune;
mod shell;
mod suggest;
mod web;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand};
use config::{
    BackendPreference, ServerConfiguration, default_server_config_path, load_server_configuration,
};
use engine::arch::ModelForward;
use engine::arch::gemma::GemmaModel;
use engine::arch::llama::LlamaModel;
use engine::arch::qwen35::Qwen35Model;
use engine::arch::qwen35moe::Qwen35MoeModel;
use engine::backend::{Backend, CpuBackend, CudaBackend, VulkanBackend};
use engine::generate::Engine;
use engine::loader::ArchFamily;
use engine::loader::LoadedModel;
use engine::scheduler::SlotPool;
use engine::tokenizer::Tokenizer;
use orangu::gguf::{GgufFile, GgufValue, ggml_type_name};
use std::{
    io::{IsTerminal, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    sync::Arc,
};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Metadata arrays longer than this print a truncated preview instead of
/// every element — `tokenizer.ggml.tokens` routinely holds 100,000+ entries.
/// Pass `--full` to disable the cap.
const DEFAULT_ARRAY_PREVIEW: usize = 8;

const TERMINAL_TITLE: &str = "orangu-server";

/// Sets the terminal window/tab title via the standard OSC 0 escape
/// sequence (supported by essentially every modern terminal emulator), and
/// restores it (clears it back) on drop. Mirrors `orangu`'s and
/// `orangu-coordinator`'s own `TerminalTitleGuard`.
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
    version = VERSION,
    about = "Serve a GGUF model over a llama.cpp-compatible HTTP API",
    long_about = "Serve a GGUF model over a llama.cpp-compatible HTTP API.",
    group(clap::ArgGroup::new("role").args(["all", "code", "review", "explorer", "embedding"]).multiple(false))
)]
struct Args {
    /// A local .gguf path, an NR/MODEL label already under the configured
    /// models directory, or a <user>/<model>[:quant] Hugging Face repo
    /// (fetched first if not already cached). Omit it to list the models
    /// under the configured models directory and pick one interactively.
    /// Ignored when a subcommand is given.
    model: Option<String>,
    /// Path to orangu-server.conf. Defaults to ./orangu-server.conf, then
    /// ~/.orangu/orangu-server.conf.
    #[arg(short, long)]
    config: Option<PathBuf>,
    /// Root directory this server is allowed to operate in. Defaults to the
    /// current working directory.
    #[arg(short, long)]
    workspace: Option<PathBuf>,
    /// Interactively create ~/.orangu/orangu-server.conf.
    #[arg(short, long)]
    init: bool,
    /// Print the shell completion script for the detected shell and exit.
    #[arg(short = 's', long = "shell-completions")]
    shell_completions: bool,
    /// Run in the background, detached from the terminal.
    #[arg(short, long)]
    daemon: bool,
    /// General-purpose. The default role.
    #[arg(long)]
    all: bool,
    /// Coding.
    #[arg(long)]
    code: bool,
    /// Code review — suppresses reasoning.
    #[arg(long)]
    review: bool,
    /// Exploration — tuned for broader, more varied output.
    #[arg(long)]
    explorer: bool,
    /// Embedding only.
    #[arg(long)]
    embedding: bool,
    #[command(subcommand)]
    command: Option<Command>,
}

/// GGUF-inventory subcommands: everything that matters when *getting* and
/// *choosing* a model, before serving one — no model is loaded, no HTTP
/// listener is bound. Serving itself isn't one of these; it stays the
/// struct's own positional `model` argument (with or without no subcommand
/// at all), exactly as before this enum existed, so `orangu-server
/// <model>` keeps working unchanged. The one collision this admits: a local
/// `.gguf` file whose bare name is exactly `system`/`suggest`/`list`/
/// `show`/`download` would be parsed as that subcommand instead of a model
/// spec — resolvable by passing a path (`./system`) instead of the bare
/// name.
#[derive(Subcommand, Debug)]
enum Command {
    /// Detect the machine's CPU and GPU(s) and print their statistics.
    System,
    /// Suggest a GGUF model size (not yet a specific model) likely to run
    /// comfortably on this machine's detected hardware.
    Suggest,
    /// List every .gguf file found under the configured models directory.
    List,
    /// Print a GGUF file's full metadata.
    Show {
        /// A path to a .gguf file, a bare name resolved against the
        /// configured models directory, an NR from `list`'s first column, or
        /// a MODEL name from its second. Omit it to pick one interactively
        /// from the same table `list` prints.
        file: Option<String>,
        /// Print every array element instead of a truncated preview.
        #[arg(long)]
        full: bool,
        /// Also list each tensor's name, shape, type, and offset.
        #[arg(long)]
        tensors: bool,
    },
    /// Download a GGUF model from Hugging Face into the configured models
    /// directory.
    Download {
        /// A Hugging Face repo, `<user>/<model>[:quant]`. Without `:quant`,
        /// prefers Q4_K_M then Q8_0, falling back to the first GGUF file
        /// found.
        repo: String,
    },
    /// Delete a GGUF model (every shard) from the configured models
    /// directory, reclaiming its Hugging Face hub-cache blob(s) too when
    /// nothing else still references them.
    Delete {
        /// A path to a .gguf file, a bare name resolved against the
        /// configured models directory, an NR from `list`'s first column, or
        /// a MODEL name from its second. Omit it to pick one interactively
        /// from the same table `list` prints.
        model: Option<String>,
        /// Skip the confirmation prompt.
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Delete chat sessions from ~/.orangu/server/sessions/. Every
    /// invocation, regardless of its own argument, first removes any
    /// non-active session with an empty chat history.
    Prune {
        /// An NR from this command's own listing, a full session id, or
        /// "all" for every non-active session. Omit it to list sessions and
        /// pick one interactively. A session currently in use by a running
        /// orangu-server is never pruned, even if named explicitly.
        identifier: Option<String>,
        /// Skip the confirmation prompt.
        #[arg(short = 'y', long)]
        yes: bool,
    },
}

impl Args {
    /// The CLI role flag that was actually given, if any — `None` when
    /// none of `--all`/`--code`/`--review`/`--explorer`/`--embedding` was
    /// passed, letting the caller fall back to the config file's own
    /// `role` key rather than silently assuming `--all` was meant.
    fn role(&self) -> Option<config::Role> {
        if self.all {
            Some(config::Role::All)
        } else if self.code {
            Some(config::Role::Code)
        } else if self.review {
            Some(config::Role::Review)
        } else if self.explorer {
            Some(config::Role::Explorer)
        } else if self.embedding {
            Some(config::Role::Embedding)
        } else {
            None
        }
    }
}

fn print_shell_completions() -> Result<()> {
    let shell = std::env::var("SHELL").unwrap_or_default();
    let script = if shell.ends_with("/bash") || shell == "bash" {
        shell::BASH
    } else if shell.ends_with("/zsh") || shell == "zsh" {
        shell::ZSH
    } else if shell.ends_with("/fish") || shell == "fish" {
        shell::FISH
    } else {
        return Err(anyhow!(
            "could not detect shell from $SHELL ({shell:?}).\n\
             Supported shells: bash, zsh, fish.\n\
             \n\
             Usage:\n\
             \x20 bash: eval \"$(orangu-server -s)\"\n\
             \x20 zsh:  orangu-server -s > ~/.zsh/completions/_orangu-server\n\
             \x20 fish: orangu-server -s > ~/.config/fish/completions/orangu-server.fish"
        ));
    };
    print!("{script}");
    Ok(())
}

fn main() -> ExitCode {
    panic_capture::install();
    // Backtraces normally need `RUST_BACKTRACE=1` from whoever launched
    // the process — set unconditionally instead, so both a captured panic
    // (`panic_capture`) and every `anyhow::Error` created from here on
    // (`?`/`anyhow!`/`bail!`, which capture a backtrace themselves when
    // this is set) carry one regardless of how the server was started.
    // Safe here specifically: this is the very first statement in `main`,
    // on the only thread that exists yet, before any other code — this
    // process's own or a dependency's — could read the environment
    // concurrently.
    unsafe {
        std::env::set_var("RUST_BACKTRACE", "1");
    }

    let mut args = Args::parse();

    if args.shell_completions {
        return match print_shell_completions() {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("error: {err:#}");
                ExitCode::FAILURE
            }
        };
    }

    if args.init {
        return match init::run_init() {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("error: {err:#}");
                ExitCode::FAILURE
            }
        };
    }

    if let Some(command) = args.command.take() {
        return match run_command(args.config, command) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("error: {err:#}");
                ExitCode::FAILURE
            }
        };
    }

    let prepared = match prepare(args) {
        Ok(prepared) => prepared,
        Err(err) => {
            eprintln!("error: {err:#}");
            return ExitCode::FAILURE;
        }
    };

    let runtime = match tokio::runtime::Runtime::new() {
        Ok(rt) => rt,
        Err(err) => {
            eprintln!("error: failed to start async runtime: {err}");
            return ExitCode::FAILURE;
        }
    };

    match runtime.block_on(serve(prepared)) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

/// Everything [`serve`] needs, resolved synchronously in [`prepare`] —
/// config, model, and both listeners are all bound *before* daemonizing (see
/// [`prepare`]'s doc comment), so `serve` itself only ever converts already-
/// bound `std` listeners to their `tokio` counterparts and runs the request
/// loop.
struct Prepared {
    engine: Arc<Engine>,
    model_label: String,
    architecture: String,
    backend_label: String,
    /// Absolute, normalized root directory the server operates in — from
    /// `-w`/`--workspace`, else the current working directory. Resolved in
    /// [`prepare`], i.e. *before* `--daemon` detaches (daemonizing moves the
    /// process to `/`), so a relative value still means what it meant in the
    /// launching shell.
    workspace: PathBuf,
    conf: ServerConfiguration,
    api_listener: std::net::TcpListener,
    web_listener: Option<std::net::TcpListener>,
    daemon: bool,
}

/// Resolves the config and model, builds the engine, and binds both
/// listeners — all synchronously (no tokio runtime yet) and, when
/// `--daemon` is set, all *before* [`daemonize`] detaches from the
/// terminal. Mirrors `orangu-coordinator --daemon`'s own reasoning: a bad
/// config, an unresolvable model, or a "address already in use" bind error
/// needs to reach the invoking terminal, not vanish into a detached daemon
/// with its stdout/stderr redirected to `/dev/null`.
fn prepare(args: Args) -> Result<Prepared> {
    let cli_role = args.role();
    let conf = load_config(args.config, cli_role, args.daemon)?;
    let mut role = conf.role;
    let workspace = resolve_workspace(args.workspace.clone())?;

    let (path, model_label) = if args.daemon {
        let spec = conf.model.clone().ok_or_else(|| {
            anyhow!(
                "--daemon requires [{}].model to be set in the config file (see --init); \
                 there is no attached terminal to prompt on",
                config::SERVER_SECTION
            )
        })?;
        let path = orangu::model_spec::resolve_or_fetch_model(&conf.models, &spec)
            .with_context(|| format!("resolving model '{spec}'"))?;
        (path, spec)
    } else {
        match args.model {
            Some(spec) => {
                let path = orangu::model_spec::resolve_or_fetch_model(&conf.models, &spec)
                    .with_context(|| format!("resolving model '{spec}'"))?;
                (path, spec)
            }
            None => {
                let selected = select_model_interactively(&conf.models)?;
                // Only when no `--all`/`--code`/`--review`/`--explorer`/
                // `--embedding` flag was given — an explicit flag already
                // settled `role`, and shouldn't be second-guessed by a
                // prompt. `conf.slots` was already resolved against
                // `Role::default()` by `load_config` above (role isn't
                // known interactively until now), so a role picked here
                // that has a different `default_slots()` than `all`'s
                // won't retroactively change `slots` unless `slots` is
                // also set explicitly in the config — the same scoping
                // `--code`/`--review`/etc. already have when combined with
                // an interactively-prompted model.
                if cli_role.is_none() {
                    role = init::prompt_role(&format!(
                        "role [{}]: ",
                        config::Role::default().label()
                    ))?;
                }
                selected
            }
        }
    };

    let gguf = GgufFile::open(&path)?;
    let tokenizer = Arc::new(Tokenizer::from_gguf(&gguf).context("building tokenizer")?);
    let chat_template_source = metadata_string(&gguf, "tokenizer.chat_template");

    let loaded = LoadedModel::open(&path).context("loading model weights")?;
    let (backend, backend_label): (Arc<dyn Backend>, String) = select_backend(conf.backend)?;
    let architecture = loaded.config.architecture.clone();
    let model: Arc<dyn ModelForward> = match engine::loader::resolve_arch_family(&architecture)? {
        ArchFamily::LlamaStyle => Arc::new(
            LlamaModel::load_with_backend(&loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::Gemma => Arc::new(
            GemmaModel::load_with_backend(&loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::Qwen35Moe => Arc::new(
            Qwen35MoeModel::load_with_backend(&loaded, backend.clone())
                .context("building model")?,
        ),
        ArchFamily::Qwen35 => Arc::new(
            Qwen35Model::load_with_backend(&loaded, backend.clone()).context("building model")?,
        ),
    };

    let slots = SlotPool::new(conf.slots);
    // Cross-sequence GEMM batching, off by default: a real, reproducible
    // concurrent-load measurement (`ORANGU_BATCH_DECODE=1` vs. without,
    // same `slots` count, 4 concurrent 100-token generations) showed it
    // 25-55% *slower*, not faster, even with the batched path fully
    // GPU-resident — see `engine::generate::Engine::batch_coordinator`'s
    // own doc comment for the numbers and the likely cause. Only built at
    // all when `slots > 1` (nothing to batch across otherwise) *and* the
    // env var is set.
    let batch_coordinator = (conf.slots > 1 && std::env::var_os("ORANGU_BATCH_DECODE").is_some())
        .then(|| engine::batch::BatchCoordinator::new(slots.clone()));
    // Cross-request KV-cache prefix reuse (`engine::prefix_cache`),
    // **off by default; opt in with `ORANGU_PREFIX_CACHE=1`**. Unlike
    // every other opt-in-then-promoted flag in this codebase (`wide_load`,
    // `packed_dot_f16`, `subgroup_reduce`), the risk here isn't a modest
    // performance regression on some adapter — a bug in prefix matching
    // or reuse would silently produce a *wrong* generation, not just a
    // slow one, so this starts opt-in on general principle even though
    // nothing has actually been measured to regress. `PREFIX_CACHE_
    // ENTRIES` is a small fixed pool size, not exposed as its own env
    // var, the same way `ATTN_SPLIT_K`/`ARGMAX_SPLIT_N`
    // (`engine/backend/vulkan.rs`) are fixed constants rather than
    // per-deployment tuning knobs — each entry holds a whole `KvCache`'s
    // worth of `f32` K/V buffers (easily hundreds of MB at real context
    // lengths), so this is sized to stay well within ordinary system RAM,
    // not tuned per-deployment.
    const PREFIX_CACHE_ENTRIES: usize = 4;
    let prefix_cache = std::env::var_os("ORANGU_PREFIX_CACHE")
        .is_some()
        .then(|| Arc::new(engine::prefix_cache::PrefixCache::new(PREFIX_CACHE_ENTRIES)));

    // Durable per-slot KV-cache persistence (`engine::slot_store`), backing
    // the `POST /slots/{id}?action=save|restore` endpoints. **On by default**;
    // set `ORANGU_NO_SLOT_SAVE` to disable it. Reuse of a restored prefix goes
    // through the same `copy_prefix_from` path the (opt-in) `ORANGU_PREFIX_CACHE`
    // pool uses; the opt-out exists for the same reason that pool is cautious —
    // a bug in prefix matching would silently produce a *wrong* generation, not
    // just a slow one — but persistence is only ever exercised when a client
    // explicitly saves/restores a slot, so it stays dormant unless used. The
    // `<fingerprint>` directory component ties every saved file to this exact
    // model architecture, label, and KV structure, so a snapshot can never be
    // restored into a different model. `None` here (disabled, or `$HOME`
    // unresolvable) makes the endpoints report "not supported," matching a
    // llama.cpp server started without `--slot-save-path` — which is exactly
    // what the orangu client already degrades against.
    let slot_store = std::env::var_os("ORANGU_NO_SLOT_SAVE")
        .is_none()
        .then(|| {
            let structure_tag = model.new_kv_cache(1).structure_tag();
            let fingerprint = engine::slot_store::SlotStore::fingerprint(
                &architecture,
                &model_label,
                &structure_tag,
            );
            engine::slot_store::SlotStore::new(slots.total(), fingerprint).map(Arc::new)
        })
        .flatten();

    let engine = Arc::new(Engine {
        model,
        tokenizer,
        chat_template_source,
        slots,
        batch_coordinator,
        prefix_cache,
        slot_store,
        role,
    });

    let api_addr = format!("{}:{}", conf.host, conf.port);
    let api_listener = std::net::TcpListener::bind(&api_addr)
        .with_context(|| format!("failed to bind {api_addr}"))?;
    api_listener
        .set_nonblocking(true)
        .with_context(|| format!("failed to configure listener on {api_addr}"))?;

    let web_listener = if conf.web != 0 {
        let web_addr = format!("{}:{}", conf.host, conf.web);
        let listener = std::net::TcpListener::bind(&web_addr)
            .with_context(|| format!("failed to bind web UI to {web_addr}"))?;
        listener
            .set_nonblocking(true)
            .with_context(|| format!("failed to configure web UI listener on {web_addr}"))?;
        Some(listener)
    } else {
        None
    };

    if args.daemon {
        daemonize().context("failed to start as a daemon")?;
    }

    Ok(Prepared {
        engine,
        model_label,
        architecture,
        backend_label,
        workspace,
        conf,
        api_listener,
        web_listener,
        daemon: args.daemon,
    })
}

/// The root directory the server operates in: the `-w`/`--workspace`
/// argument, or (`None`) the current working directory — the same default
/// `orangu`'s own `--workspace` has, resolved by the same shared
/// [`orangu::workspaces::resolve_workspace_root`]: made absolute against the
/// current directory and normalized.
///
/// Unlike `orangu` — which resolves a workspace it is about to open a session
/// on, and would report a bad path at its first tool call anyway — a server
/// hands its root to requests it hasn't received yet, so a typo would only
/// surface much later, in whatever feature happens to use it first. Hence the
/// directory check here, at startup, while there's still a terminal to report
/// it on.
fn resolve_workspace(cli: Option<PathBuf>) -> Result<PathBuf> {
    let workspace = orangu::workspaces::resolve_workspace_root(cli)?;
    if !workspace.is_dir() {
        bail!(
            "workspace {} does not exist or is not a directory",
            workspace.display()
        );
    }
    Ok(workspace)
}

/// Detach from the controlling terminal and continue running in the
/// background. Only the final, fully-detached process returns `Ok(())`; the
/// original (and an intermediate) process exit here and never return.
/// Mirrors `orangu-coordinator`'s own `daemonize`.
#[cfg(unix)]
fn daemonize() -> Result<()> {
    daemonize::Daemonize::new()
        .start()
        .map_err(|err| anyhow!(err))
}

#[cfg(not(unix))]
fn daemonize() -> Result<()> {
    Err(anyhow!("--daemon is only supported on Unix-like platforms"))
}

async fn serve(prepared: Prepared) -> Result<()> {
    let Prepared {
        engine,
        model_label,
        architecture,
        backend_label,
        workspace,
        conf,
        api_listener,
        web_listener,
        daemon,
    } = prepared;

    let _terminal_title_guard = (!daemon).then(|| TerminalTitleGuard::new(TERMINAL_TITLE));

    let listener = tokio::net::TcpListener::from_std(api_listener)
        .context("failed to attach listener to the async runtime")?;
    let web_listener = match web_listener {
        Some(l) => Some(
            tokio::net::TcpListener::from_std(l)
                .context("failed to attach web UI listener to the async runtime")?,
        ),
        None => None,
    };

    let (shutdown_tx, mut shutdown_rx) = tokio::sync::mpsc::channel::<()>(1);
    let state = Arc::new(http::AppState {
        engine: engine.clone(),
        model_label: model_label.clone(),
        workspace: workspace.clone(),
        started_at: std::time::Instant::now(),
        shutdown_tx,
    });
    let app = http::build_router(state);

    if !daemon {
        let cpu = orangu::hardware::detect_cpu();
        let gpus = orangu::hardware::detect_gpus(cpu.total_memory_bytes);
        print!("{}", orangu::hardware::format_report(&cpu, &gpus));
        println!();
        println!(
            "Model      {model_label} ({architecture} arch, {backend_label}, {} layers, {} ctx)",
            engine.model.config().n_layer,
            engine.model.config().n_ctx_train,
        );
        match &web_listener {
            Some(l) => println!("UI         http://{}", l.local_addr()?),
            None => println!("UI         disabled"),
        }
        println!("API        http://{}:{}", conf.host, conf.port);
        println!("Workspace  {}", workspace.display());
        for advisory in orangu::hardware::performance_advisories() {
            println!("Note       {advisory}");
        }
    }

    if let Some(web_listener) = web_listener {
        let web_state = Arc::new(web::WebState {
            engine,
            model_label,
            architecture,
            backend_label,
            workspace,
            version: VERSION,
        });
        let web_app = web::build_router(web_state);
        // Not joined: when `serve` returns (any shutdown path below), the
        // tokio Runtime it's driven by is dropped right after in `main`,
        // which cancels every still-running spawned task, this one
        // included — the same abrupt-stop behavior the primary API
        // listener gets from losing the `tokio::select!` race below.
        tokio::spawn(async move {
            let _ = axum::serve(web_listener, web_app).await;
        });
    }

    tokio::select! {
        result = axum::serve(listener, app.into_make_service_with_connect_info::<std::net::SocketAddr>()) => {
            result.context("server error")?;
        }
        _ = tokio::signal::ctrl_c() => {
            if !daemon {
                println!("shutting down");
            }
        }
        _ = shutdown_rx.recv() => {
            if !daemon {
                println!("received shutdown request, shutting down");
            }
        }
        // A real terminal Ctrl+C also delivers SIGINT, so this branch races
        // tokio::signal::ctrl_c() above for the exact same event — tokio::
        // select! picks whichever's ready essentially at random, so this
        // must print the same message rather than staying silent, or the
        // "shutting down" line only shows up on half of all Ctrl+Cs.
        _ = wait_for_sigint() => {
            if !daemon {
                println!("shutting down");
            }
        }
    }

    Ok(())
}

fn load_config(
    explicit: Option<PathBuf>,
    cli_role: Option<config::Role>,
    daemon: bool,
) -> Result<ServerConfiguration> {
    let path = explicit.or_else(default_server_config_path).ok_or_else(|| {
        anyhow!(
            "Missing config file; pass --config or add ./orangu-server.conf or ~/.orangu/orangu-server.conf (see --init)"
        )
    })?;
    load_server_configuration(&path, cli_role, daemon)
        .with_context(|| format!("loading {}", path.display()))
}

fn metadata_string(gguf: &GgufFile, key: &str) -> Option<String> {
    gguf.metadata.iter().find_map(|(k, v)| {
        (k == key).then_some(v).and_then(|v| match v {
            GgufValue::String(s) => Some(s.clone()),
            _ => None,
        })
    })
}

/// The architecture, and whether this build can load it, for each group in
/// `groups` (aligned by index) — judged from each group's representative
/// file's own header (cheap: metadata + tensor directory, no tensor data),
/// for the `SUPPORTED` column of the `list` table and the model pickers.
/// `engine::loader::model_load_support` owns the actual judgement — which
/// can be stricter than the architecture string alone when a header-
/// detectable blocker exists — and this only maps its result onto the
/// lib-side `ModelSupport`
/// that `format_groups` renders. A file that can't even be opened is reported
/// unsupported with no architecture.
fn model_support(
    groups: &[orangu::model_spec::ModelGroup],
) -> Vec<orangu::model_spec::ModelSupport> {
    groups
        .iter()
        .map(|group| {
            let (architecture, supported) = GgufFile::open(&group.representative_path)
                .ok()
                .map(|gguf| engine::loader::model_load_support(&gguf))
                .unwrap_or((None, false));
            orangu::model_spec::ModelSupport {
                architecture,
                supported,
            }
        })
        .collect()
}

/// Runs one of the GGUF-inventory subcommands (`system`/`suggest`/`list`/
/// `show`/`download`) to completion and returns — none of these load a
/// model or bind a listener, so there's no `tokio` runtime involved, unlike
/// [`serve`]. `system`/`suggest` don't even need a config file (they only
/// ever look at the local machine's own hardware); `list`/`show`/`download`
/// resolve against the same `[orangu-server].models` directory the serving
/// path uses, via the same [`load_config`] — `cli_role`/`daemon` are passed
/// as `None`/`false` since neither matters to a subcommand that never
/// serves anything.
fn run_command(config_arg: Option<PathBuf>, command: Command) -> Result<()> {
    match command {
        Command::System => {
            let cpu = orangu::hardware::detect_cpu();
            let gpus = orangu::hardware::detect_gpus(cpu.total_memory_bytes);
            print!("{}", orangu::hardware::format_report(&cpu, &gpus));
            Ok(())
        }
        Command::Suggest => {
            let cpu = orangu::hardware::detect_cpu();
            let gpus = orangu::hardware::detect_gpus(cpu.total_memory_bytes);
            print!("{}", suggest::format_suggestion(&cpu, &gpus));
            Ok(())
        }
        Command::List => {
            let conf = load_config(config_arg, None, false)?;
            let models = orangu::model_spec::scan_models_dir(&conf.models)?;
            let groups = orangu::model_spec::group_models(&models);
            let repos: Vec<String> = groups
                .iter()
                .filter_map(|g| g.hf_repo.clone())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            let latest_commits = orangu::model_download::latest_commits(&repos);
            let support = model_support(&groups);
            print!(
                "{}",
                orangu::model_spec::format_groups(
                    &groups,
                    &conf.models,
                    &latest_commits,
                    &support,
                    std::io::stdout().is_terminal(),
                )
            );
            Ok(())
        }
        Command::Show {
            file,
            full,
            tensors,
        } => {
            let conf = load_config(config_arg, None, false)?;
            let path = match file {
                Some(spec) => orangu::model_spec::resolve_show_target(&conf.models, &spec)?,
                None => select_model_for_show(&conf.models)?,
            };
            let gguf = GgufFile::open(&path)?;
            print!("{}", format_show(&gguf, full, tensors));
            Ok(())
        }
        Command::Download { repo } => {
            let conf = load_config(config_arg, None, false)?;
            let path = orangu::model_download::download_model(&conf.models, &repo)?;
            println!("Downloaded to {}", path.display());
            Ok(())
        }
        Command::Delete { model, yes } => {
            let conf = load_config(config_arg, None, false)?;
            let group = match model {
                Some(spec) => orangu::model_spec::resolve_delete_target(&conf.models, &spec)?,
                None => select_model_for_deletion(&conf.models)?,
            };
            let plural = if group.paths.len() == 1 { "" } else { "s" };
            if !yes {
                let confirmed = confirm(&format!(
                    "Delete '{}' ({} file{plural}, {}) from {}? [y/N]: ",
                    group.label,
                    group.paths.len(),
                    orangu::format::format_bytes(group.size_bytes),
                    conf.models.display(),
                ))?;
                if !confirmed {
                    println!("Aborted. Nothing deleted.");
                    return Ok(());
                }
            }
            orangu::model_spec::delete_model(&conf.models, &group)?;
            println!(
                "Deleted '{}' ({} file{plural}, {})",
                group.label,
                group.paths.len(),
                orangu::format::format_bytes(group.size_bytes),
            );
            Ok(())
        }
        Command::Prune { identifier, yes } => prune::run(identifier, yes),
    }
}

/// Lists every `.gguf` model under `models_dir` (the same table `list`
/// prints) and prompts for an `NR`, for `show` invoked with no file
/// argument. Returns the chosen model's representative path — the same one
/// `resolve_show_target`'s own NR resolution would give an explicit `NR`
/// argument, so both paths into `show` end up looking at exactly the same
/// file.
fn select_model_for_show(models_dir: &Path) -> Result<PathBuf> {
    let models = orangu::model_spec::scan_models_dir(models_dir)
        .with_context(|| format!("scanning {}", models_dir.display()))?;
    let groups = orangu::model_spec::group_models(&models);
    if groups.is_empty() {
        bail!("no .gguf models found under {}", models_dir.display());
    }
    print!(
        "{}",
        orangu::model_spec::format_groups(
            &groups,
            models_dir,
            &Default::default(),
            &model_support(&groups),
            std::io::stdout().is_terminal(),
        )
    );

    print!("\nSelect a model (NR): ");
    std::io::stdout().flush().ok();
    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .context("failed to read model selection")?;
    let nr: usize = input
        .trim()
        .parse()
        .with_context(|| format!("'{}' is not a number", input.trim()))?;
    let count = groups.len();
    nr.checked_sub(1)
        .and_then(|index| groups.into_iter().nth(index))
        .map(|group| group.representative_path)
        .ok_or_else(|| anyhow!("no model with NR {nr} ({count} model(s) listed)"))
}

/// Lists every `.gguf` model under `models_dir` (the same table `list`
/// prints) and prompts for an `NR`, for `delete` invoked with no model
/// argument. Returns the chosen model's full `ModelGroup` — every shard,
/// not just the representative one — so the caller can delete all of them
/// atomically.
fn select_model_for_deletion(models_dir: &Path) -> Result<orangu::model_spec::ModelGroup> {
    let models = orangu::model_spec::scan_models_dir(models_dir)
        .with_context(|| format!("scanning {}", models_dir.display()))?;
    let groups = orangu::model_spec::group_models(&models);
    if groups.is_empty() {
        bail!("no .gguf models found under {}", models_dir.display());
    }
    print!(
        "{}",
        orangu::model_spec::format_groups(
            &groups,
            models_dir,
            &Default::default(),
            &model_support(&groups),
            std::io::stdout().is_terminal(),
        )
    );

    print!("\nSelect a model to delete (NR): ");
    std::io::stdout().flush().ok();
    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .context("failed to read model selection")?;
    let nr: usize = input
        .trim()
        .parse()
        .with_context(|| format!("'{}' is not a number", input.trim()))?;
    let count = groups.len();
    nr.checked_sub(1)
        .and_then(|index| groups.into_iter().nth(index))
        .ok_or_else(|| anyhow!("no model with NR {nr} ({count} model(s) listed)"))
}

/// Reads a Yes/No confirmation from stdin, defaulting to No on an empty
/// entry or unrecognized input — `delete` (and `prune`, `crate::prune`) is
/// destructive, so anything but an explicit "y"/"yes" leaves the model(s)/
/// session(s) untouched. A closed stdin (EOF) also reads as an empty line
/// here, so a non-interactive invocation without `--yes` safely deletes
/// nothing rather than hanging or guessing.
pub(crate) fn confirm(prompt: &str) -> Result<bool> {
    print!("{prompt}");
    std::io::stdout().flush().ok();
    let mut line = String::new();
    std::io::stdin()
        .read_line(&mut line)
        .context("failed to read confirmation")?;
    Ok(matches!(line.trim().to_lowercase().as_str(), "y" | "yes"))
}

fn format_show(gguf: &GgufFile, full: bool, tensors: bool) -> String {
    let preview_limit = if full {
        usize::MAX
    } else {
        DEFAULT_ARRAY_PREVIEW
    };

    let mut out = String::new();
    out.push_str(&format!("GGUF version   : {}\n", gguf.version));
    out.push_str(&format!("Metadata pairs : {}\n", gguf.metadata.len()));
    out.push_str(&format!("Tensors        : {}\n", gguf.tensors.len()));
    out.push_str(&format!("Alignment      : {} bytes\n", gguf.alignment));
    out.push_str(&format!("Data offset    : {} bytes\n", gguf.data_offset));

    out.push_str("\nMetadata\n");
    let key_width = gguf
        .metadata
        .iter()
        .map(|(k, _)| k.len())
        .max()
        .unwrap_or(0);
    for (key, value) in &gguf.metadata {
        out.push_str(&format!(
            "  {key:<key_width$} = {}\n",
            value.display(preview_limit)
        ));
    }

    if tensors {
        out.push_str("\nTensors\n");
        let name_width = gguf.tensors.iter().map(|t| t.name.len()).max().unwrap_or(0);
        let type_width = gguf
            .tensors
            .iter()
            .map(|t| ggml_type_name(t.ggml_type).len())
            .max()
            .unwrap_or(0);
        for tensor in &gguf.tensors {
            out.push_str(&format!(
                "  {:<name_width$}  {:<type_width$}  {}  (offset {})\n",
                tensor.name,
                ggml_type_name(tensor.ggml_type),
                tensor.shape(),
                tensor.offset
            ));
        }
    }

    out
}

/// Lists every `.gguf` model under `models_dir` (the same table
/// `orangu-server list` prints, `SUPPORTED` column and all — models this
/// build can't load are shown greyed rather than hidden: a user can still
/// pick one and will hit the same clear "not yet supported" error `prepare`
/// gives for any other unsupported model) and prompts for an `NR`, for
/// `orangu-server` invoked with no model argument. Returns the chosen
/// model's file path and its display label.
fn select_model_interactively(models_dir: &Path) -> Result<(PathBuf, String)> {
    let models = orangu::model_spec::scan_models_dir(models_dir)
        .with_context(|| format!("scanning {}", models_dir.display()))?;
    let groups = orangu::model_spec::group_models(&models);
    if groups.is_empty() {
        bail!(
            "no .gguf models found under {}; download one first (e.g. `orangu-server download <user>/<model>`) or pass one directly: orangu-server <model>",
            models_dir.display()
        );
    }

    print!(
        "{}",
        orangu::model_spec::format_groups(
            &groups,
            models_dir,
            &Default::default(),
            &model_support(&groups),
            std::io::stdout().is_terminal(),
        )
    );

    print!("\nSelect a model (NR): ");
    std::io::stdout().flush().ok();

    let mut input = String::new();
    std::io::stdin()
        .read_line(&mut input)
        .context("failed to read model selection")?;
    let nr: usize = input
        .trim()
        .parse()
        .with_context(|| format!("'{}' is not a number", input.trim()))?;
    let group = nr
        .checked_sub(1)
        .and_then(|index| groups.get(index))
        .ok_or_else(|| anyhow!("no model with NR {nr} ({} model(s) listed)", groups.len()))?;
    Ok((group.representative_path.clone(), group.label.clone()))
}

/// Ctrl+C (`tokio::signal::ctrl_c`) already covers `SIGINT` on Unix in
/// practice, but this listens for it explicitly too so a plain `kill
/// -INT <pid>` (not delivered via a controlling terminal) is unambiguously
/// covered on every platform this binary ships for.
#[cfg(unix)]
async fn wait_for_sigint() {
    use tokio::signal::unix::{SignalKind, signal};
    match signal(SignalKind::interrupt()) {
        Ok(mut sig) => {
            sig.recv().await;
        }
        Err(_) => std::future::pending::<()>().await,
    }
}

#[cfg(not(unix))]
async fn wait_for_sigint() {
    std::future::pending::<()>().await
}

/// Picks the `Backend` the forward pass runs on, per `[orangu-server].
/// backend` (`auto`/`cpu`/`vulkan`/`cuda`/`opencl`/`rocm`, see
/// `config::BackendPreference`), and a label for the startup banner (e.g.
/// `"CPU/AVX2"` or `"Vulkan/AMD Radeon RX 5500M (RADV NAVI14)"`). `auto`
/// tries every GPU backend compiled into this build, preferring the most
/// mature one first (`VulkanBackend`, the only one with real fused/GPU-
/// resident optimizations — see its module doc), then falls back to the
/// CPU backend if none found one; every other named backend fails loudly
/// instead of falling back, since GPU inference was asked for explicitly.
/// `rocm` additionally fails loudly (a clear "rebuild with `--features
/// rocm`" message, not a panic) when this binary wasn't built with that
/// Cargo feature — see `engine::backend::rocm`'s module doc for why it's
/// the one opt-in backend (`cuda`/`opencl`/`vulkan` are always compiled
/// in).
/// `VulkanBackend::try_init` can transiently return `None` — its
/// `request_adapter`/`request_device` fail intermittently right after a prior
/// process released the GPU (the driver hasn't finished tearing the previous
/// context down), which surfaces as a flaky "no usable Vulkan adapter" at
/// startup. Retry a few times with a short backoff (silently) so a transient
/// race doesn't sink the whole server; the caller prints a single error if it
/// still returns `None`. A genuine absence of a Vulkan device also returns
/// `None`, only a little later — each attempt is fast when there's no adapter.
fn init_vulkan_with_retry() -> Option<VulkanBackend> {
    const ATTEMPTS: usize = 4;
    for attempt in 1..=ATTEMPTS {
        if let Some(backend) = VulkanBackend::try_init() {
            return Some(backend);
        }
        if attempt < ATTEMPTS {
            std::thread::sleep(std::time::Duration::from_millis(700));
        }
    }
    None
}

fn select_backend(preference: BackendPreference) -> Result<(Arc<dyn Backend>, String)> {
    let cpu = || -> (Arc<dyn Backend>, String) {
        let label = if is_x86_feature_detected() {
            "CPU/AVX2"
        } else {
            "CPU"
        };
        (Arc::new(CpuBackend), label.to_string())
    };
    match preference {
        BackendPreference::Cpu => Ok(cpu()),
        BackendPreference::Vulkan => {
            let backend = init_vulkan_with_retry().ok_or_else(|| {
                anyhow!(
                    "[{}].backend = vulkan, but no usable Vulkan adapter was found",
                    config::SERVER_SECTION
                )
            })?;
            let label = format!("Vulkan/{}", backend.adapter_name);
            Ok((Arc::new(backend), label))
        }
        BackendPreference::Cuda => {
            let backend = CudaBackend::try_init().ok_or_else(|| {
                anyhow!(
                    "[{}].backend = cuda, but no usable CUDA device was found",
                    config::SERVER_SECTION
                )
            })?;
            let label = format!("CUDA/{}", backend.device_name);
            Ok((Arc::new(backend), label))
        }
        BackendPreference::OpenCl => {
            let backend = engine::backend::OpenClBackend::try_init().ok_or_else(|| {
                anyhow!(
                    "[{}].backend = opencl, but no usable OpenCL device was found",
                    config::SERVER_SECTION
                )
            })?;
            let label = format!("OpenCL/{}", backend.device_name);
            Ok((Arc::new(backend), label))
        }
        BackendPreference::Rocm => {
            #[cfg(feature = "rocm")]
            {
                let backend = engine::backend::RocmBackend::try_init().ok_or_else(|| {
                    anyhow!(
                        "[{}].backend = rocm, but no usable ROCm/HIP device was found",
                        config::SERVER_SECTION
                    )
                })?;
                let label = format!("ROCm/{}", backend.device_name);
                Ok((Arc::new(backend), label))
            }
            #[cfg(not(feature = "rocm"))]
            {
                Err(anyhow!(
                    "[{}].backend = rocm, but this build of orangu-server was compiled without \
                     the \"rocm\" Cargo feature (rebuild with `--features rocm`)",
                    config::SERVER_SECTION
                ))
            }
        }
        BackendPreference::Auto => {
            if let Some(backend) = init_vulkan_with_retry() {
                let label = format!("Vulkan/{}", backend.adapter_name);
                return Ok((Arc::new(backend), label));
            }
            if let Some(backend) = CudaBackend::try_init() {
                let label = format!("CUDA/{}", backend.device_name);
                return Ok((Arc::new(backend), label));
            }
            if let Some(backend) = engine::backend::OpenClBackend::try_init() {
                let label = format!("OpenCL/{}", backend.device_name);
                return Ok((Arc::new(backend), label));
            }
            #[cfg(feature = "rocm")]
            if let Some(backend) = engine::backend::RocmBackend::try_init() {
                let label = format!("ROCm/{}", backend.device_name);
                return Ok((Arc::new(backend), label));
            }
            Ok(cpu())
        }
    }
}

fn is_x86_feature_detected() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_workspace;

    #[test]
    fn workspace_defaults_to_the_current_directory() {
        let current_dir = std::env::current_dir().expect("current directory");

        assert_eq!(resolve_workspace(None).expect("workspace"), current_dir);
    }

    /// The argument is resolved the same way `orangu`'s own is: made
    /// absolute against the current directory and normalized.
    #[test]
    fn the_argument_is_normalized() {
        let dir = tempfile::tempdir().expect("workspace");
        let with_detour = dir.path().join("sub").join("..");

        let resolved = resolve_workspace(Some(with_detour)).expect("workspace");

        assert_eq!(resolved, dir.path());
    }

    #[test]
    fn a_workspace_that_is_not_a_directory_is_rejected() {
        let dir = tempfile::tempdir().expect("workspace");
        let file = dir.path().join("orangu-server.conf");
        std::fs::write(&file, "[orangu-server]\n").expect("write file");

        let err = resolve_workspace(Some(file)).expect_err("not a directory");
        assert!(
            err.to_string().contains("not a directory"),
            "unexpected error: {err:#}"
        );

        let err =
            resolve_workspace(Some(dir.path().join("missing"))).expect_err("missing directory");
        assert!(
            err.to_string().contains("does not exist"),
            "unexpected error: {err:#}"
        );
    }
}
