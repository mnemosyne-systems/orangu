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
//! `system`/`suggest`/`list`/`show`/`download`/`delete`/`refresh` answer the
//! questions that matter when *getting*, *choosing*, and *keeping current* a
//! model to run, before any serving starts (formerly the separate
//! `orangu-gguf` binary, folded in here so
//! there's one tool, one config file, and one shell-completion script to
//! keep in sync with the models directory convention both jobs share).

// The `Send`/`Sync` proof for the web UI's SSE stream walks the whole wgpu
// type graph the Vulkan backend pulls in, which sits right at the default
// 128-step auto-trait recursion limit — deep enough that adding a single
// field to `web::WebState` overflows it (in the test build only, where the
// stream type is monomorphized again). Raising it is rustc's own suggested
// remedy and costs nothing but a deeper proof search.
#![recursion_limit = "256"]

mod bundle;
mod config;
mod device_lost;
mod engine;
mod http;
mod init;
mod panic_capture;
mod prune;
mod reexec;
mod refresh;
mod shell;
mod suggest;
mod tls;
mod web;

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand, ValueEnum};
use config::{
    BackendPreference, ServerConfiguration, default_server_config_path, load_server_configuration,
};
use engine::arch::ModelForward;
use engine::arch::bailingmoe::BailingMoeModel;
use engine::arch::deepseek4::Deepseek4Model;
use engine::arch::dflash::DFlashModel;
use engine::arch::gemma::GemmaModel;
use engine::arch::glm::GlmModel;
use engine::arch::glm5::Glm5Model;
use engine::arch::inkling::InklingModel;
use engine::arch::kimi3::Kimi3Model;
use engine::arch::llama::LlamaModel;
use engine::arch::mistral::MistralModel;
use engine::arch::muse::MuseModel;
use engine::arch::nemotron::NemotronModel;
use engine::arch::phi::PhiModel;
use engine::arch::qwen3next::Qwen3NextModel;
use engine::arch::qwen4exp::Qwen4ExpModel;
use engine::arch::qwen35::Qwen35Model;
use engine::arch::qwen35moe::Qwen35MoeModel;
use engine::backend::device;
use engine::backend::device::DeviceClass;
use engine::backend::vulkan_shaders::KvStorage;
use engine::backend::{
    Backend, CpuBackend, CudaBackend, DeviceCandidate, DeviceError, DeviceErrorKind, DeviceRequest,
    MetalBackend, MultiDeviceBackend, VulkanBackend,
};
use engine::footprint::DeviceFootprint;
use engine::generate::Engine;
use engine::kv_cache::KvCache;
use engine::loader::ArchFamily;
use engine::loader::LoadedModel;
use engine::placement::{self, SplitMode, SplitPlan};
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

/// The terminal title for one mode of this binary: `orangu-server <mode>`
/// — `orangu-server download` while a model is being fetched,
/// `orangu-server prune` while sessions are being cleaned up, and so on. A
/// single binary that both serves and manages an inventory of models is
/// otherwise indistinguishable from itself in a tab bar; the mode is what a
/// glance at a backgrounded terminal actually needs to answer. Serving
/// itself has no mode word — it isn't a subcommand — and keeps the plain
/// [`TERMINAL_TITLE`].
fn terminal_title(mode: &str) -> String {
    format!("{TERMINAL_TITLE} {mode}")
}

/// Sets the terminal window/tab title via the standard OSC 0 escape
/// sequence (supported by essentially every modern terminal emulator), and
/// restores it (clears it back) on drop. Mirrors `orangu`'s and
/// `orangu-coordinator`'s own `TerminalTitleGuard`.
struct TerminalTitleGuard;

impl TerminalTitleGuard {
    /// `None` — nothing printed, nothing to restore — when stdout isn't a
    /// terminal: `orangu-server list > models.txt` and `... show | grep`
    /// would otherwise take the raw escape bytes into the file or the pipe,
    /// and there's no title to set in the first place.
    fn new(title: &str) -> Option<Self> {
        if !std::io::stdout().is_terminal() {
            return None;
        }
        print!("\x1b]0;{title}\x07");
        // The sequence ends in BEL, not a newline, so a line-buffered stdout
        // would otherwise hold the title back until the command's own first
        // line of output — which for `download` is only after the fetch it
        // was meant to announce.
        let _ = std::io::stdout().flush();
        Some(Self)
    }
}

impl Drop for TerminalTitleGuard {
    fn drop(&mut self) {
        print!("\x1b]0;\x07");
        let _ = std::io::stdout().flush();
    }
}

/// The five mutually exclusive role flags, as one reusable block.
///
/// Shared rather than written twice because `bundle` needs exactly the same
/// five: the role a bundle is built with is the role its server comes up in,
/// so it is chosen the same way and spelled the same way. Flattening keeps
/// them a set clap enforces the exclusivity of, in both places at once.
#[derive(clap::Args, Debug, Default)]
#[group(required = false, multiple = false)]
struct RoleFlags {
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
}

impl RoleFlags {
    /// The flag that was actually given, if any — `None` when none was, so
    /// the caller can fall back to a config file's own `role` key (or a
    /// bundle's) rather than silently assuming `--all` was meant.
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

/// Where to listen, as three flags — the other block shared between serving
/// and `bundle`, for the same reason [`RoleFlags`] is.
///
/// Serving, they override the config file for this run. Bundling, they are
/// recorded *in* the bundle and become its defaults, exactly as the role is:
/// a bundle is a server somebody will run without a config file, so the
/// address it comes up on has to be decidable when it is built, not only when
/// it is started.
#[derive(clap::Args, Debug, Default, Clone)]
struct ListenFlags {
    /// Address to bind: "all" (or "*") for every network interface, or a
    /// literal address such as 0.0.0.0 or 127.0.0.1. Serving, it overrides
    /// [orangu-server].host (and [web].host unless that is set explicitly);
    /// on `bundle`, it is the address the bundle binds by default.
    #[arg(long)]
    host: Option<String>,
    /// Port the HTTP API listens on. Serving, it overrides
    /// [orangu-server].port; on `bundle`, it is the bundle's own default
    /// (8100 either way).
    #[arg(short, long)]
    port: Option<u16>,
    /// Port the web console listens on, or 0 to disable it. Serving, it
    /// overrides [web].port; on `bundle`, it is the bundle's own default
    /// (8200).
    #[arg(long)]
    web: Option<u16>,
}

impl ListenFlags {
    /// These flags as the record a bundle stores, or `None` per field where
    /// nothing was given.
    fn bundled(&self) -> config::BundledListen {
        config::BundledListen {
            host: self.host.clone(),
            port: self.port,
            web: self.web,
        }
    }

    /// `self` where it names something, falling back to `other` field by
    /// field — how a flag after the subcommand wins over the same flag
    /// before it without either one having to be all-or-nothing.
    fn or(&self, other: &ListenFlags) -> ListenFlags {
        ListenFlags {
            host: self.host.clone().or_else(|| other.host.clone()),
            port: self.port.or(other.port),
            web: self.web.or(other.web),
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "orangu-server",
    version = VERSION,
    about = "Serve a GGUF model over a llama.cpp-compatible HTTP API",
    // Without this, `--help` promotes the `Command` enum's doc comment — an
    // internal note about subcommand parsing — into the top-level long help.
    long_about = None
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
    #[command(flatten)]
    listen: ListenFlags,
    /// Root directory this server is allowed to operate in. Defaults to the
    /// current working directory.
    #[arg(short, long)]
    workspace: Option<PathBuf>,
    /// GPU device to use exclusively: an index, part of its name, or `auto`.
    #[arg(long, value_name = "DEVICE")]
    device: Option<String>,
    /// Spread the model's layers across the selected devices: off, auto, all, cpu, or shares like 3,1.
    #[arg(long = "device-split", value_name = "MODE")]
    device_split: Option<String>,
    /// Worker threads for every CPU path. Defaults to one per logical core.
    #[arg(long, value_name = "N")]
    threads: Option<String>,
    /// Interactively create ~/.orangu/orangu-server.conf.
    #[arg(short, long)]
    init: bool,
    /// Print the shell completion script for the detected shell and exit.
    #[arg(short = 's', long = "shell-completions")]
    shell_completions: bool,
    /// Run in the background, detached from the terminal.
    #[arg(short, long)]
    daemon: bool,
    #[command(flatten)]
    roles: RoleFlags,
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
/// `show`/`download`/`refresh`/`bundle` would be parsed as that subcommand
/// instead of a model spec — resolvable by passing a path (`./system`)
/// instead of the bare name.
///
/// `bundle` is the odd one out: it neither loads a model nor binds a
/// listener either, but what it produces is a *server* — see
/// [`crate::bundle`].
#[derive(Subcommand, Debug)]
enum Command {
    /// Detect the machine's CPU and GPU(s) and print their statistics.
    System,
    /// Suggest a GGUF model size (not yet a specific model) likely to run
    /// comfortably on this machine's detected hardware.
    Suggest,
    /// List every .gguf file found under the configured models directory.
    List {
        /// Order rows by size (largest first) or last use (most recent first).
        /// NR remains the model's number in the default alphabetical listing.
        #[arg(long, value_enum, value_name = "FIELD")]
        sort: Option<ListSort>,
    },
    /// Report what a model would need to run here — **without loading it**.
    ///
    /// Reads only the GGUF tensor tables, so a plan for a model far larger
    /// than this machine costs seconds rather than a thirty-minute load.
    Plan {
        /// A path to a .gguf file, a bare name resolved against the
        /// configured models directory, an NR from `list`, or a MODEL name.
        file: Option<String>,
        /// Also check that every shard is present and readable, and that the
        /// architecture is one this build implements.
        #[arg(long)]
        deep: bool,
    },
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
    /// directory, planning it against this machine first.
    Download {
        /// A Hugging Face repo, `<user>/<model>[:quant]`. Without `:quant`,
        /// prefers Q4_K_M then Q8_0, falling back to the first GGUF file
        /// found.
        repo: String,
        /// Download without confirming a model this machine cannot run.
        #[arg(short = 'y', long)]
        yes: bool,
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
    /// Delete a GGUF model and download it again, picking up a newer
    /// revision of its Hugging Face repo.
    Refresh {
        /// An NR from `list`, a MODEL name (with `:QUANT` when the repo has
        /// more than one on disk), a bare name, or a path. Omit it to pick
        /// one interactively.
        #[arg(conflicts_with = "all")]
        model: Option<String>,
        /// Refresh every model that is behind its Hugging Face repository.
        #[arg(long)]
        all: bool,
        /// Accepted for compatibility; refresh no longer prompts.
        #[arg(short = 'y', long)]
        yes: bool,
    },
    /// Write a single executable carrying both this server and a model,
    /// which then runs with no models directory and no configuration file.
    Bundle {
        /// The model to embed: a local .gguf path, an NR/MODEL label from
        /// `list`, or a <user>/<model>[:quant] Hugging Face repo (fetched
        /// first if not already cached). Omit it to pick one interactively.
        model: Option<String>,
        /// Where to write the bundle. Defaults to
        /// ./orangu-server-bundle-<ARCH>, naming the architecture the
        /// bundled binary was built for (plus .exe for a Windows one).
        #[arg(short, long)]
        output: Option<PathBuf>,
        /// The executable to bundle the model into. Defaults to this one;
        /// pass a build for another platform to bundle that instead.
        #[arg(long)]
        binary: Option<PathBuf>,
        /// Skip the confirmation prompt, and take the default role rather
        /// than asking for one.
        #[arg(short = 'y', long)]
        yes: bool,
        /// The role the bundle comes up in. Also accepted before the
        /// subcommand (`orangu-server --code bundle ...`); omit both to be
        /// asked for one.
        #[command(flatten)]
        roles: RoleFlags,
        /// Where the bundle listens by default, baked into it. Also accepted
        /// before the subcommand (`orangu-server --host all bundle ...`).
        #[command(flatten)]
        listen: ListenFlags,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum ListSort {
    Size,
    LastUsed,
}

impl Command {
    /// This subcommand's name as the user typed it, for the terminal title
    /// (see [`terminal_title`]). Kept in step with the variants' clap names
    /// by [`terminal_title_names_every_subcommand`].
    fn mode(&self) -> &'static str {
        match self {
            Command::System => "system",
            Command::Suggest => "suggest",
            Command::List { .. } => "list",
            Command::Plan { .. } => "plan",
            Command::Show { .. } => "show",
            Command::Download { .. } => "download",
            Command::Delete { .. } => "delete",
            Command::Refresh { .. } => "refresh",
            Command::Bundle { .. } => "bundle",
            Command::Prune { .. } => "prune",
        }
    }
}

impl Args {
    /// The CLI role flag that was actually given, if any — see
    /// [`RoleFlags::role`].
    fn role(&self) -> Option<config::Role> {
        self.roles.role()
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
        // Same window, same reason: `reexec` reads the two variables a
        // handover sets and takes them back out of the environment, so
        // nothing later — a child process, a second handover — can act on a
        // stale value. Everything afterwards reads the parsed results.
        reexec::take_inherited();
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
        let _terminal_title_guard = TerminalTitleGuard::new(&terminal_title("init"));
        return match init::run_init() {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("error: {err:#}");
                ExitCode::FAILURE
            }
        };
    }

    if let Some(command) = args.command.take() {
        let _terminal_title_guard = TerminalTitleGuard::new(&terminal_title(command.mode()));
        // Only `bundle` reads these — it is the one subcommand that decides
        // how the server it writes will start, so the role and the address
        // are settings it records rather than settings for this run.
        let cli_role = args.role();
        let cli_listen = args.listen.clone();
        return match run_command(args.config, cli_role, &cli_listen, command) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("error: {err:#}");
                ExitCode::FAILURE
            }
        };
    }

    // Serving is not a subcommand — it's the bare `orangu-server <model>`
    // invocation — so it keeps the plain binary name as its title. Installed
    // here rather than inside `serve`, so the title is already up during the
    // slowest part of starting up: resolving, possibly downloading, and
    // loading the model. Skipped under `--daemon`: the title would outlive
    // this process's hold on the terminal, since `prepare` detaches before
    // returning and the guard's restore would then run against the daemon's
    // own (redirected) stdout.
    let _terminal_title_guard = (!args.daemon)
        .then(|| TerminalTitleGuard::new(TERMINAL_TITLE))
        .flatten();

    // `config`/`workspace` are needed again below if this start fails and a
    // fallback has to be exec'd, and `prepare` consumes `args`.
    let config_arg = args.config.clone();
    let workspace_arg = args.workspace.clone();
    let role_arg = args.role().unwrap_or_default();
    let listen_arg = reexec::Listen {
        host: args.listen.host.clone(),
        api: args.listen.port,
        web: args.listen.web,
    };
    let prepared = match prepare(args) {
        Ok(prepared) => prepared,
        Err(err) => {
            eprintln!("error: {err:#}");
            // This image was exec'd by a handover whose model has just
            // turned out not to load — a GPU backend with no kernel for one
            // of its tensor types, a model too large for the machine, or
            // anything else only a real load can discover. Go back to the
            // model that *was* working rather than leaving the pid dead with
            // its port still bound.
            if let Some(fallback) = reexec::fallback_model() {
                eprintln!("falling back to '{fallback}'");
                match reexec::Handover::new(
                    config_arg,
                    listen_arg,
                    workspace_arg.unwrap_or_else(|| PathBuf::from(".")),
                    role_arg,
                    fallback.to_string(),
                    reexec::inherited(),
                ) {
                    // `None` as the fallback of the fallback: one retry, so a
                    // pair of models that both fail can't loop forever.
                    Ok(handover) => eprintln!("error: {:#}", handover.exec(fallback, None)),
                    Err(err) => eprintln!("error: {err:#}"),
                }
            }
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
    /// `[orangu-server].api_key` / `ORANGU_API_KEY`, resolved once — the
    /// bearer token `http::require_api_key` checks. `None` leaves the server
    /// open.
    api_key: Option<String>,
    /// `[orangu-server].tls_cert`/`tls_key`, resolved once. `None` serves
    /// plain HTTP.
    tls: Option<(PathBuf, PathBuf)>,
    /// Where to write the prefix-cache snapshot on the way out, and the model
    /// identity that snapshot is only valid for. `None` unless
    /// `ORANGU_PREFIX_CACHE_DIR` asked for it — carried here rather than
    /// re-derived at shutdown so the fingerprint that reads a snapshot back is
    /// provably the one that wrote it.
    prefix_cache_snapshot: Option<(PathBuf, String)>,
    model_label: String,
    /// The quantization the resolved file is stored at
    /// ([`orangu::model_spec::quantization_for_file`]), for the startup
    /// banner's `MODEL:QUANT` line. `None` when `model_label` already carries
    /// a `:tag` of its own (the label was named that way), or when the file
    /// says nothing about its scheme.
    quantization: Option<String>,
    architecture: String,
    backend_label: String,
    /// The `.gguf` this server loaded — the first shard, for a multi-part
    /// model. The web UI's model manager marks this row as the loaded one,
    /// and refuses to delete it: the weights are mapped by the running
    /// engine, so removing the file would leave this process reading
    /// something that no longer has a name.
    model_path: PathBuf,
    /// `[orangu-server].models`, the directory the model manager lists,
    /// downloads into and deletes from.
    models_dir: PathBuf,
    /// `--config` as given, or `None` when the default search found it —
    /// see [`reexec::Handover`], which has to reproduce the same choice.
    config_path: Option<PathBuf>,
    /// `--host`/`--port`/`--web` as given, for the same reason `config_path`
    /// is kept: a handover has to come back up on the same addresses this run
    /// chose, not on the ones its config file names.
    listen_override: reexec::Listen,
    /// The role this process actually resolved to, flag or prompt.
    role: config::Role,
    /// `[web].reexec`: whether the web console may load a different model
    /// into this server.
    reexec: bool,
    /// `[web].delete`: whether the web console may delete models.
    delete: bool,
    /// The model inside this executable, when that is what is being served
    /// — for the startup banner, and for naming this model to a handover
    /// that has to fall back to it (see [`bundle::EMBEDDED_SPEC`]). `None`
    /// both for an ordinary build and for a bundled one that was pointed at
    /// a model on disk instead.
    bundle: Option<&'static bundle::Bundle>,
    /// MCP profiles captured at startup for the web console's read-only view.
    mcp_servers: Vec<config::McpConfiguration>,
    /// The GPU kernel/tuning selection this device came up with, and its
    /// one-line form for the startup banner — see [`AppState::gpu_tuning`]
    /// and `VulkanBackend::tuning_report`. Both `None` only for
    /// `CpuBackend`: a GPU backend with no kernel selection to report
    /// carries `Backend::reduced_surface` here instead, so neither the
    /// banner nor `/props` goes silent on a device that is running a
    /// narrower path than the one every published number was taken on.
    /// Captured in [`prepare`], where the concrete backend is still in hand
    /// rather than a `dyn Backend`.
    gpu_tuning: Option<serde_json::Value>,
    gpu_tuning_summary: Option<String>,
    /// The backend, when it is the `wgpu` engine — see
    /// [`http::AppState::wgpu_backend`].
    wgpu_backend: Option<Arc<dyn Backend>>,
    /// Absolute, normalized root directory the server operates in — from
    /// `-w`/`--workspace`, else the current working directory. Resolved in
    /// [`prepare`], i.e. *before* `--daemon` detaches (daemonizing moves the
    /// process to `/`), so a relative value still means what it meant in the
    /// launching shell.
    workspace: PathBuf,
    api_listener: std::net::TcpListener,
    web_listener: Option<std::net::TcpListener>,
    daemon: bool,
}

/// Which model this process is about to serve, and where its bytes are.
///
/// The two cases differ in exactly one place — the file that gets mapped and
/// how far into it the model starts — so everything past [`prepare`] is
/// written against the result rather than against this.
enum ModelSource {
    /// A `.gguf` on disk: a path given on the command line, named by the
    /// config file, or picked at the prompt.
    File(PathBuf),
    /// The model inside this executable. See [`crate::bundle`].
    Embedded(&'static bundle::Bundle),
}

impl ModelSource {
    /// The file the weights are read out of — the model itself, or the
    /// executable carrying it. This is what the web console shows as the
    /// loaded model's path and refuses to delete; for a bundle that is this
    /// binary, which is both true and the right thing to refuse.
    fn path(&self) -> &Path {
        match self {
            ModelSource::File(path) => path,
            ModelSource::Embedded(bundle) => &bundle.exe,
        }
    }

    /// The model's header: metadata, tokenizer, and tensor directory.
    fn gguf(&self) -> Result<GgufFile> {
        match self {
            ModelSource::File(path) => GgufFile::open(path),
            ModelSource::Embedded(bundle) => {
                let offset = *bundle
                    .shard_offsets
                    .first()
                    .ok_or_else(|| anyhow!("the embedded model has no shards"))?;
                GgufFile::open_at(&bundle.exe, offset)
            }
        }
    }

    /// The mapped weights.
    fn load(&self) -> Result<LoadedModel> {
        match self {
            ModelSource::File(path) => LoadedModel::open(path),
            ModelSource::Embedded(bundle) => {
                LoadedModel::open_bundled(&bundle.exe, &bundle.shard_offsets)
            }
        }
    }
}

/// Resolves a model spec — whatever was passed on the command line, or read
/// from `[orangu-server].model` — to the model it names.
///
/// [`bundle::EMBEDDED_SPEC`] is the one spec not resolved against the models
/// directory: it names the model already in this file. In a binary with no
/// bundle it isn't reserved at all and resolves like any other name, which
/// is the honest outcome — there is nothing embedded for it to mean.
///
/// The label handed back is the resolved group's own `MODEL` name, not the
/// spec as typed: an `NR` from `list`'s first column is a position in a
/// listing, and carrying it forward would make `orangu-server 84` call
/// itself `84` in the banner, on `/v1/models`, in every response's `model`
/// field, and in the web console's header. A spec naming nothing on disk
/// (a Hugging Face repo to fetch) keeps the spelling it was given.
fn resolve_model_spec(
    models_dir: &Path,
    spec: &str,
    bundled: Option<&'static bundle::Bundle>,
) -> Result<(ModelSource, String)> {
    if spec == bundle::EMBEDDED_SPEC
        && let Some(bundle) = bundled
    {
        return Ok((ModelSource::Embedded(bundle), bundle.model.clone()));
    }
    let (path, label) = orangu::model_spec::resolve_load_target(models_dir, spec)
        .with_context(|| format!("resolving model '{spec}'"))?;
    Ok((ModelSource::File(path), label))
}

fn auto_pair_dflash_target(
    models_dir: &Path,
    source: ModelSource,
    label: String,
) -> Result<(ModelSource, String)> {
    let ModelSource::File(path) = &source else {
        return Ok((source, label));
    };
    let gguf = GgufFile::open(path)?;
    if metadata_string(&gguf, "general.architecture").as_deref() != Some("dflash") {
        return Ok((source, label));
    }
    let repo = orangu::model_spec::hf_repo_for_path(path).ok_or_else(|| {
        anyhow!(
            "dflash draft model {} is not under a Hugging Face cache repo, so its paired target model cannot be resolved automatically",
            path.display()
        )
    })?;
    if let Ok(models) = orangu::model_spec::scan_models_dir(models_dir) {
        let groups = orangu::model_spec::group_models(&models);
        if let Some(target) = groups.iter().find_map(|group| {
            (group.hf_repo.as_deref() == Some(repo.as_str()) && group.representative_path != *path)
                .then_some(group.representative_path.clone())
        }) {
            return Ok((ModelSource::File(target), repo));
        }
    }
    let target = orangu::model_download::download_model(models_dir, &repo).with_context(|| {
        format!(
            "selected dflash draft sidecar {}, but failed to fetch its paired target model from {repo}",
            path.display()
        )
    })?;
    Ok((ModelSource::File(target), repo))
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
    let config_path = args.config.clone();
    // Read before `args` is consumed below; applied at `select_backend`,
    // where the device list it names actually exists.
    let device_flag = args.device.clone();
    let split_flag = args.device_split.clone();
    let threads_flag = args.threads.clone();
    let conf = load_config(args.config, cli_role, args.daemon)?;
    let mut role = conf.role;
    let reasoning_effort = conf.reasoning_effort.clone();
    let workspace = resolve_workspace(args.workspace.clone())?;
    let bundled = bundle::embedded();

    let (source, model_label) = if args.daemon {
        // The positional argument first, then `[orangu-server].model`: a
        // spec that was typed is not a spec to ignore, and a bundled
        // `--daemon` start with neither would otherwise have no way to be
        // pointed at a different model at all.
        match (args.model.clone().or_else(|| conf.model.clone()), bundled) {
            (Some(spec), _) => resolve_model_spec(&conf.models, &spec, bundled)?,
            // A bundle answers the question `--daemon` otherwise has no way
            // to answer: which model, with no terminal to ask on and no
            // config file required to have been written.
            (None, Some(bundle)) => (ModelSource::Embedded(bundle), bundle.model.clone()),
            (None, None) => bail!(
                "--daemon requires [{}].model to be set in the config file (see --init); \
                 there is no attached terminal to prompt on",
                config::SERVER_SECTION
            ),
        }
    } else {
        match args.model {
            Some(spec) => resolve_model_spec(&conf.models, &spec, bundled)?,
            // A bundled binary serves what it carries. There is nothing to
            // choose between — that is what was downloaded — so nothing is
            // asked, which is what lets a bundle start on a double-click.
            // Naming a model on the command line still overrides it, above.
            None if bundled.is_some() => {
                let bundle = bundled.expect("checked in the guard");
                (ModelSource::Embedded(bundle), bundle.model.clone())
            }
            None => {
                let selected = select_model_interactively(&conf.models, conf.model.as_deref())?;
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
                    // Pre-selected from `[orangu-server].role` when the
                    // config names one, so a configuration that already says
                    // how this server is meant to run is one Enter away
                    // rather than something to retype. An explicit CLI flag
                    // never reaches here at all.
                    // Pre-selected as the ghost alone, matching the model
                    // prompt just above it — no `[review]` after the label
                    // as well, which would print the same value twice.
                    let default = conf.role_key.unwrap_or_default();
                    role = init::prompt_role("Role: ", default)?;
                    init::echo_answer("Role: ", role.label());
                }
                let (path, label) = selected;
                (ModelSource::File(path), label)
            }
        }
    };

    // A bundled server comes up in the role its bundle was built with,
    // unless this run says otherwise — an explicit flag (handled by
    // `load_config`, which never reaches here), or the config file's own
    // `role` key, which a bundle-only start has as `Some(bundle.role)`
    // anyway (`config::bundled_configuration`). A config file that names a
    // role therefore still wins over the bundle, which is the right way
    // round: one was written for this machine, the other was baked in
    // wherever the bundle was made.
    if let ModelSource::Embedded(bundle) = &source
        && cli_role.is_none()
    {
        role = conf.role_key.unwrap_or(bundle.role);
    }
    let (source, model_label) = auto_pair_dflash_target(&conf.models, source, model_label)?;
    let path = source.path().to_path_buf();

    let gguf = source.gguf()?;
    let quantization = match &source {
        // Recorded when the bundle was made, from the file it was made from
        // — the executable it now lives in says nothing about quantization,
        // and there is no file name left to read one off.
        ModelSource::Embedded(bundle) => bundle.quantization.clone(),
        // Only when the label doesn't already name a tag itself (`-m
        // user/model:Q4_K_M`), so the banner never reads `...:Q4_K_M:Q4_K_M`.
        ModelSource::File(path) => (!label_carries_tag(&model_label))
            .then(|| orangu::model_spec::quantization_for_file(path, &gguf))
            .flatten(),
    };
    let tokenizer = Arc::new(Tokenizer::from_gguf(&gguf).context("building tokenizer")?);
    let chat_template_source = metadata_string(&gguf, "tokenizer.chat_template");

    // Before anything parallel runs — the loader itself reaches for rayon,
    // and `build_global` can only be called once.
    let threads = configure_cpu_threads(threads_flag.as_deref(), conf.threads)?;
    // Before the model is opened, because the loader is the first thing that
    // can read a weight through an explicit route.
    engine::expert_read::set_read_size(conf.read_size);
    let mut loaded = source.load().context("loading model weights")?;
    // Before the first device comes up, because bringing one up compiles the
    // attention shaders against whichever storage this names — the choice
    // cannot be revisited afterwards without rebuilding them.
    engine::backend::vulkan::set_kv_cache_preference(conf.kv_cache);
    let (backend, backend_label): (Arc<dyn Backend>, String) = select_backend(
        conf.backend,
        &requested_device(device_flag.as_deref(), &conf.device),
    )?;
    // Before any prompt is prefilled: the chunker's sizer aims at a driver
    // timeout, and on a backend that has none it reads the clock as a
    // per-token rate that a streamed model does not have.
    engine::generate::set_chunk_policy(backend.has_submission_timeout());
    // The split decision needs only the *weight* side of the footprint,
    // which is readable straight from the loaded tensor table — the KV side
    // needs a built model, and the model cannot be built until placement is
    // decided, since building it is what stamps each tensor's device.
    let (weights_device_bytes, _) =
        engine::backend::device_resident_split(loaded.resident_tensor_sizes());
    let per_layer_bytes =
        engine::footprint::DeviceFootprint::weights_per_layer(&loaded, loaded.config.n_layer);
    // What device 0 is charged with no matter how the layers are placed:
    // `LoadedModel::device_for_tensor` pins the embeddings, the output norm
    // and `lm_head` there. On a large-vocabulary model at `BF16` that alone
    // is gigabytes, and if it exceeds the head device nothing a placement can
    // do will make the model fit a GPU.
    let non_layer_bytes = weights_device_bytes.saturating_sub(per_layer_bytes.iter().sum::<u64>());
    // **The last rung of the ladder: dGPU, then iGPU, then the CPU.** The
    // embeddings, the output norm and `lm_head` are pinned to device 0 by
    // `LoadedModel::device_for_tensor` whatever the placement, so if they
    // alone exceed device 0 no split rescues the model — the GPU path will
    // over-commit the card and run out of memory mid-request. Run it on the
    // host instead, which is bounded by RAM and the page cache rather than by
    // a card.
    //
    // Checked here, before the split, because afterwards the backend is a
    // `MultiDeviceBackend` and `as_wgpu` is `None` by design — there is no
    // head device left to ask.
    //
    // Loud rather than silent, and only where the device choice was
    // automatic: an explicit `--device`/`device` is a decision to respect.
    // A tensor larger than the device's maximum buffer size cannot be
    // uploaded at all — not "slowly", not "paged", not at all: `wgpu` fails
    // `create_buffer` validation and the request panics. Bigger than any
    // capacity question, because no split and no eviction changes it.
    let oversized_tensor = backend.as_wgpu().and_then(|wgpu| {
        let limit = wgpu.max_buffer_size();
        loaded
            .resident_tensor_sizes()
            .filter(|&(name, bytes)| {
                let (device_bytes, _) =
                    engine::backend::device_resident_split(std::iter::once((name, bytes)));
                device_bytes > limit
            })
            .map(|(name, bytes)| (name.to_string(), bytes))
            .max_by_key(|&(_, bytes)| bytes)
            .map(|(name, bytes)| (name, bytes, limit))
    });
    if let Some((name, bytes, limit)) = &oversized_tensor {
        eprintln!(
            "orangu-server: tensor `{name}` is {} and this device will not create a buffer \
             larger than {} — no placement changes that, so the model runs on the CPU.",
            orangu::format::format_bytes(*bytes),
            orangu::format::format_bytes(*limit),
        );
    }

    let head_cannot_hold_pinned = backend
        .as_wgpu()
        .and_then(|wgpu| {
            wgpu.device_in_use()
                .vram_total_bytes
                .map(|total| (wgpu.device_in_use().name.clone(), total))
        })
        .filter(|(_, total)| non_layer_bytes > *total)
        .filter(|_| {
            matches!(
                requested_device(device_flag.as_deref(), &conf.device),
                engine::backend::device::DeviceRequest::Auto
            )
        });
    let (backend, backend_label) = match head_cannot_hold_pinned {
        _ if oversized_tensor.is_some() => {
            let cpu: Arc<dyn Backend> = Arc::new(engine::backend::CpuBackend);
            let label = if is_x86_feature_detected() {
                "CPU/AVX2"
            } else {
                "CPU"
            };
            (cpu, label.to_string())
        }
        Some((name, total)) => {
            eprintln!(
                "orangu-server: {name} has {total_h} and this model pins {pinned} of embeddings \
                 and output weights to it before a single layer is placed — no split can make \
                 that fit, so the model runs on the CPU. Choose a device explicitly with \
                 `--device` to override.",
                total_h = orangu::format::format_bytes(total),
                pinned = orangu::format::format_bytes(non_layer_bytes),
            );
            let cpu: Arc<dyn Backend> = Arc::new(engine::backend::CpuBackend);
            let label = if is_x86_feature_detected() {
                "CPU/AVX2"
            } else {
                "CPU"
            };
            (cpu, label.to_string())
        }
        None => (backend, backend_label),
    };

    // **Every model has to run, even slowly.** When nothing was asked for and
    // the weights will not fit the selected device, fill the devices in their
    // ranked order — discrete first, then integrated — and run the remainder
    // on the CPU, rather than loading a model this device cannot serve.
    //
    // Escalating here rather than refusing later is the difference between a
    // slow server and no server. The alternative was measured: a 57.18 GiB
    // model on a 4.00 GiB card uploaded weights into an arena that never
    // evicts until `radv` could not allocate for a command submission, lost
    // the device, and exited 75 — which a supervisor restarts into the same
    // wall on the next request.
    //
    // Only when the caller expressed no preference. An explicit
    // `device_split` — including `off` — is a decision, and this must not
    // quietly overrule it.
    let requested = requested_split(split_flag.as_deref(), &conf.device_split)?;
    let split_mode = if requested.is_off()
        && split_flag.is_none()
        && conf.device_split.is_off()
        && overflows_selected_device(backend.as_ref(), weights_device_bytes)
    {
        eprintln!(
            "orangu-server: the weights are larger than the selected device — spreading \
             them across every device in order and running the remainder on the CPU \
             (`device_split = cpu`). Set `device_split` explicitly to choose differently."
        );
        SplitMode::Cpu
    } else {
        requested
    };
    let (backend, backend_label, split) = apply_device_split(
        backend,
        backend_label,
        &split_mode,
        &per_layer_bytes,
        weights_device_bytes,
    )?;

    eprintln!(
        "{}",
        cpu_inventory(
            match &split {
                Some(split) if split.plan.runs_on_host() => "— overflow tier",
                _ if backend.as_wgpu().is_none() && split.is_none() => "<- in use",
                _ => "— not running layers",
            },
            threads
        )
    );
    // Before the model is built: `LoadedModel::matrix` is what stamps each
    // tensor's device, and every architecture calls it during construction.
    if let Some(split) = &split {
        loaded.set_layer_devices(split.plan.layer_device.clone());
    }
    // After the model is built — `LoadedModel::matrix` has stamped every
    // tensor by then — and before the first token, so the sweep sees a
    // complete table on its first pass.
    engine::dense_residency::register(loaded.layer_weight_spans(loaded.config.n_layer));
    // Two caveats belong in the message rather than in a doc nobody reads at
    // 2am. The release hook lives in `CpuBackend::matmul_into` and nowhere
    // else, so on layers a device executes this knob does exactly nothing —
    // silence there would read as "it is working". And it was measured at
    // three windows and never won: stating that here is cheaper than an
    // operator re-deriving it.
    if engine::dense_residency::enabled() {
        eprintln!(
            "orangu-server: dense residency: releasing each layer's weights once the sweep is \
             past it (ORANGU_DENSE_WINDOW). Applies only to layers the CPU executes. \
             Measured at windows 1, 4 and 32: none beat leaving residency to the kernel."
        );
    }
    let expert_tier_active = plan_expert_tier(&mut loaded, backend.as_ref(), weights_device_bytes);
    // Captured here, while the concrete backend is still in hand, because the
    // `wgpu` engine is the only thing that knows which of its kernels feature
    // negotiation and the `ORANGU_*` flags actually left live — and that
    // answer travels with every benchmark result through `/props`. `None` for
    // `CpuBackend`; on the CUDA/OpenCL/ROCm backends, which have no such
    // selection to report, a `surface` naming what they run instead. See
    // `VulkanBackend::tuning_report` and `Backend::reduced_surface`.
    let mut gpu_tuning = backend
        .as_wgpu()
        .map(engine::backend::VulkanBackend::tuning_report)
        .or_else(|| {
            // Not a `tuning_report`: these backends have no flags, no
            // geometry and no per-type kernel table to report. What they do
            // have is a surface, and `/props.gpu` reading `null` for them
            // makes a CUDA run indistinguishable from a CPU one to anything
            // reading this endpoint — `orangu-bench`'s header included.
            backend
                .reduced_surface()
                .map(|surface| serde_json::json!({ "surface": surface, "kernels": null }))
        });
    // The banner reports the kernels *this* model's own weights decode
    // through, most-common type first. A file named for a K-quant is often
    // mostly something else — `unsloth/Qwen3.8-27B-GGUF:IQ2_XXS` has no
    // `Q4_K` tensor at all — so naming a fixed pair of types would answer a
    // question nobody asked. Floats are excluded: they carry the norms and
    // biases, never the weight bytes decode throughput is made of.
    let dominant_types = dominant_tensor_types(loaded.tensor_types());
    // On a `wgpu` backend, the kernels this model's weights actually decode
    // through. On `CudaBackend`/`RocmBackend`/`OpenClBackend`, which have no
    // kernel *selection* to report, what they have instead of one — see
    // `Backend::reduced_surface` for why an absent row was the wrong answer
    // there.
    let gpu_tuning_summary = backend
        .as_wgpu()
        .map(|v| v.tuning_summary_for(&dominant_types))
        .or_else(|| backend.reduced_surface().map(str::to_string));
    // The backend itself, when it is a `wgpu` one, so `/gpu-timings` can drain
    // its accumulated timestamp breakdown. Kept as the `dyn Backend` the state
    // already holds rather than a second concrete handle: `as_wgpu` is how
    // every other GPU-specific path reaches through, and one way in is easier
    // to keep honest than two.
    let wgpu_backend: Option<Arc<dyn Backend>> =
        backend.as_wgpu().is_some().then(|| backend.clone());
    // Every GPU backend covers fewer `ggml_type`s than `engine::quant` reads
    // on the CPU, so a file this build can decode can still be one the
    // selected device has no kernel for. Caught here, against the tensor
    // directory alone, rather than as a panic from inside `matmul` partway
    // through the first request.
    let unsupported = engine::backend::unsupported_tensor_types(loaded.tensor_types(), &*backend);
    if !unsupported.is_empty() {
        bail!(
            "backend {backend_label} has no kernel for tensor type(s) {}; only backend = cpu \
             reads every type this build supports, so re-run with that (or pick a \
             quantization of this model without those types)",
            unsupported.join(", ")
        );
    }
    let architecture = loaded.config.architecture.clone();
    let model = build_model(&loaded, &backend)?;

    // What this model puts on the chosen device, against what that device
    // has — reported here, where both are finally known, and before the
    // first request rather than after somebody notices the throughput.
    //
    // The KV geometry comes from a one-token probe cache, the same trick
    // the slot-persistence fingerprint below uses: the per-layer shape is
    // fixed model geometry, so it can be read from a cache that allocates
    // nothing and then scaled to a context far too large to build.
    // A split model has no single device to measure, so it reports what
    // each device holds instead — the same question, answered per device.
    if let Some(split) = &split {
        // One probe cache for every device: the shape is the model's, and
        // each device's share of it is selected by layer inside
        // `for_split_device`.
        let footprints = split.footprints(&loaded, &model.new_kv_cache(1), conf.slots);
        for line in split.lines(&footprints) {
            eprintln!("{line}");
        }
        // `as_wgpu` is `None` on a split, so the ordinary tuning report was
        // never built. Standing this in its place keeps `/props` describing
        // the run rather than going silent on the configuration that most
        // needs describing.
        let (_, weights_host_bytes) =
            engine::backend::device_resident_split(loaded.resident_tensor_sizes());
        gpu_tuning = Some(split.to_json(&footprints, weights_host_bytes));
    }
    let footprint = backend.as_wgpu().map(|wgpu| {
        engine::footprint::DeviceFootprint::measure(
            &loaded,
            &model.new_kv_cache(1),
            Some(wgpu.kv_storage()),
            conf.slots,
        )
    });
    if let (Some(footprint), Some(wgpu)) = (&footprint, backend.as_wgpu()) {
        for line in footprint.report(wgpu.api_tag(), wgpu.device_in_use()) {
            eprintln!("{line}");
        }
        // Beside the tuning report rather than in it: `tuning_report` is a
        // property of the device and its kernels, this is a property of the
        // device *and this model*, and a reader of `/props` wants both in
        // one place regardless.
        if let Some(object) = gpu_tuning.as_mut().and_then(|value| value.as_object_mut()) {
            object.insert(
                "footprint".to_string(),
                footprint.to_json(wgpu.device_in_use()),
            );
        }
        // On a MoE model, what a device expert tier in that headroom would
        // be worth. A projection, printed because the question otherwise
        // can only be answered by building the tier first — see
        // `engine::expert_tier`.
        for line in expert_tier_projection(&loaded, footprint, wgpu, expert_tier_active) {
            eprintln!("{line}");
        }
        // No refusal here, deliberately. A device with no headroom left is a
        // reason to place layers somewhere else — which the overflow split
        // above has already done — not a reason to decline the model. See
        // `DeviceFootprint::serves_no_tokens_on`, which still exists so the
        // report can say plainly what the numbers are.
    }

    let slots = SlotPool::with_queue_limit(conf.slots, conf.queue_limit);
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
    let prefix_cache = crate::engine::env::flag_on("ORANGU_PREFIX_CACHE")
        .then(|| Arc::new(engine::prefix_cache::PrefixCache::new(PREFIX_CACHE_ENTRIES)));

    // The paged KV pool (`ORANGU_PAGED_KV=1`), and the index that gives its
    // pages a content identity. Both or neither: a pool without an index pages
    // without sharing, which is the cost of paging for none of the benefit.
    //
    // Sized from the same headroom figure the startup report already prints,
    // but **not divided by the slot count** — that division is what a shared
    // pool exists to remove. See `KvPool::sized_for`.
    /// Host memory the paged KV pool may take, unless `ORANGU_KV_POOL_BYTES`
    /// says otherwise.
    ///
    /// A flat ceiling rather than a fraction of free RAM. What is free at
    /// startup is not this process's to spend — the page cache holding the
    /// model is counted as free and is doing real work — and a pool sized from
    /// it would grow on an idle machine and shrink on a busy one, which makes
    /// two runs of the same benchmark incomparable for reasons nothing reports.
    const DEFAULT_HOST_KV_POOL_BYTES: u64 = 2 * 1024 * 1024 * 1024;
    // On unless disabled — and not built at all where the selected attention
    // kernels have no paged form, because a paged cache under those would keep
    // the shared pages *and* the per-request mirror.
    let paged_supported = backend
        .as_wgpu()
        .is_none_or(engine::backend::vulkan::VulkanBackend::supports_paged_kv);
    let paged_kv = (engine::kv_pool::paged_kv_enabled() && paged_supported)
        .then(|| {
            let page_tokens = engine::kv_pool::page_tokens();
            let probe = model.new_kv_cache(1);
            let layers = engine::kv_pool::LayerGeometry::of(&probe);
            // **Host** bytes, and only host bytes. This used to read the
            // device headroom, on the reasoning that VRAM is what a KV cache
            // competes for — which is true of the device *mirror* and false of
            // this pool. The pool holds `f32` rows in system memory; the mirror
            // is a separate, per-request allocation the pool does not replace.
            // Sizing one from the other's budget gave a plausible number for
            // the wrong reason, and would have been wrong in either direction
            // on a machine whose card and RAM are differently proportioned.
            let budget = engine::backend::env_tuning_value(
                "ORANGU_KV_POOL_BYTES",
                DEFAULT_HOST_KV_POOL_BYTES,
                "a positive byte count",
                |v: u64| v > 0,
            );
            // Bounded by *both* budgets. The pool holds `f32` on the host and
            // the backend's KV width on the device, and it allocates the device
            // side up front where the per-request mirror grew on demand — so a
            // page count derived from host bytes alone will happily claim
            // headroom the weights need.
            //
            // A **quarter** of the headroom, and the fraction is measured
            // rather than chosen. The startup report's "room for N tokens of
            // KV" describes memory the mirror would have taken *as it grew*;
            // the pool takes its share at once and keeps it, so the same figure
            // is not available to it. On a 4 GiB card with a 1.9 GiB model
            // (2.13 GiB headroom), device pages of 1.06 GiB cost 47% of decode
            // with a spread fourteen times the contiguous path's, while 0.54
            // GiB and below ran at parity — so the usable share is under half
            // and a quarter clears it with margin.
            let device_budget = backend.as_wgpu().and_then(|wgpu| {
                footprint
                    .as_ref()?
                    .headroom_on(wgpu.device_in_use())
                    .map(|h| h / 4)
            });
            // `F32` when there is no device: the width only matters for the
            // device bound, which is `None` in that case and never consulted.
            let storage = backend.as_wgpu().map_or(
                engine::backend::vulkan_shaders::KvStorage::F32,
                |w| w.kv_storage(),
            );
            let pages = engine::kv_pool::KvPool::pages_within(
                budget,
                device_budget,
                &layers,
                page_tokens,
                storage,
            );
            if pages == 0 {
                return None;
            }
            let mut pool = engine::kv_pool::KvPool::with_policy(
                pages,
                page_tokens,
                layers,
                engine::kv_pool::policy_from_env(),
            );
            // Device pages, once for the process. This is what makes a shared
            // prefix cost one mirror instead of one per request — the per-slot
            // mirrors it replaces were the reason a four-slot server advertised
            // a quarter of the context.
            //
            // The block table is sized for every slot holding a full pool's
            // worth of pages at once, which cannot happen (they draw on one
            // pool) but is the bound that cannot be exceeded, and it is four
            // bytes an entry.
            if let Some(wgpu) = backend.as_wgpu() {
                let (device, _) = wgpu.device_and_queue();
                let entries = pool.num_pages() * conf.slots;
                if pool.attach_device(device, wgpu.kv_storage(), entries) {
                    println!(
                        "orangu-server: [kv] device pages — {:.2} GiB across {} layers",
                        pool.device_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
                        pool.layers().len(),
                    );
                }
            }
            println!(
                "orangu-server: [kv] paged cache — {} pages of {page_tokens} tokens                  ({} tokens total, {:.2} GiB host), policy {:?}",
                pool.num_pages(),
                pool.token_capacity(),
                pool.host_bytes() as f64 / (1024.0 * 1024.0 * 1024.0),
                pool.policy(),
            );
            let index = engine::prefix_index::PrefixIndex::new(page_tokens);
            Some((Arc::new(pool), Arc::new(index)))
        })
        .flatten();
    if engine::kv_pool::paged_kv_enabled() && !paged_supported {
        println!(
            "orangu-server: [kv] paged cache off — this device's attention \
             kernels (GQA or flash split) have no paged form yet"
        );
    } else if engine::kv_pool::paged_kv_enabled() && paged_kv.is_none() {
        // Said out loud rather than silently falling back: an operator who
        // asked for the pool and did not get one should learn it here, not
        // from a hit rate that never moves.
        eprintln!(
            "orangu-server: [kv] ORANGU_PAGED_KV is set but the memory budget              does not hold one page; running with per-request caches"
        );
    }

    // Durable version of that pool (`ORANGU_PREFIX_CACHE_DIR=<dir>`), so a
    // conversation survives a restart instead of re-prefilling. On a model
    // whose experts stream, a re-prefill is not just recomputed attention: it
    // re-reads every expert the replayed positions touched, which is the
    // dominant cost. Separately opt-in from the pool itself because a snapshot
    // is a whole `KvCache` per entry — hundreds of MB at real context lengths,
    // and nobody should discover that by finding their disk full.
    //
    // The fingerprint is `slot_store`'s, so a snapshot can never be read back
    // for a different model: that would match on token ids and answer from
    // another model's state, which is plausible-looking rather than obviously
    // broken.
    let prefix_cache_dir = std::env::var_os("ORANGU_PREFIX_CACHE_DIR")
        .filter(|_| prefix_cache.is_some())
        .map(std::path::PathBuf::from);
    let prefix_fingerprint = engine::slot_store::SlotStore::fingerprint(
        &architecture,
        &model_label,
        &model.new_kv_cache(1).structure_tag(),
    );
    if let (Some(pool), Some(dir)) = (prefix_cache.as_ref(), prefix_cache_dir.as_ref()) {
        let loaded = pool.load_from(dir, &prefix_fingerprint);
        if loaded > 0 {
            eprintln!(
                "orangu-server: prefix cache restored {loaded} entries from {}",
                dir.display()
            );
        }
    }

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
    let slot_store = (!crate::engine::env::flag_on("ORANGU_NO_SLOT_SAVE"))
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

    let draft = match &conf.draft_model {
        Some(spec) => {
            let draft = load_draft_model(
                &conf.models,
                spec,
                draft_tokens(conf.draft_tokens),
                &tokenizer,
                &backend,
                &backend_label,
            )
            .with_context(|| format!("loading draft model '{spec}'"))?;
            // Both halves are probed before anything is served, because
            // `forward_all_logits` is optional on `ModelForward` and the
            // architectures that implement it are a minority. Discovering
            // that on the first request would mean a server that starts,
            // reports a draft on its banner, and then fails every generation.
            supports_multi_position(model.as_ref(), &architecture)?;
            supports_multi_position(draft.model.as_ref(), &draft.label)?;
            Some(Arc::new(draft))
        }
        None => None,
    };

    let engine = Arc::new(Engine {
        paged_kv: paged_kv.clone(),
        metrics: Arc::new(engine::metrics::ServerMetrics::new()),
        model,
        draft,
        tokenizer,
        chat_template_source,
        slots,
        prefix_cache,
        slot_store,
        role,
        reasoning_effort,
    });

    // `all` (the default) and its `*` alias become `0.0.0.0` here — see
    // `config::resolve_bind_host`; every other value is a literal address
    // `bind` gets as written.
    // Bound here, *after* the model is loaded, which is what lets a failed
    // handover fall back: the overwhelmingly common failure — this model
    // cannot be loaded — happens above, with the descriptors handed over by
    // the previous image still untouched and ready to be passed on again.
    //
    // `adopt_or_bind`, not `bind`: on a handover these sockets are already
    // listening and were kept open across the exec, so the port is never
    // released and nothing can take it in between. On an ordinary start
    // there is nothing to adopt and it binds.
    let inherited = reexec::inherited();
    // `--host`/`--port`/`--web` override whatever the config file (or, for a
    // bundle, `config::bundled_configuration`) resolved to. Where to listen is
    // the setting that is routinely per-*run* rather than per-machine — a
    // second server alongside one already on 8100, a port a firewall happens
    // to allow, a bundle that should be reachable from the LAN for one
    // afternoon — and for a bundle there may be no config file to edit at all.
    let host = args.listen.host.as_deref().unwrap_or(&conf.host);
    // The console follows `--host` only when it was following the API's
    // address anyway. A config that deliberately put it somewhere else keeps
    // it there — see `ServerConfiguration::web_host_explicit`; exposing the
    // API must never be a way to expose the console by accident.
    let web_host = match conf.web_host_explicit {
        true => conf.web_host.as_str(),
        false => host,
    };
    let bind_host = config::resolve_bind_host(host);
    let api_port = args.listen.port.unwrap_or(conf.port);
    let web_port = args.listen.web.unwrap_or(conf.web);
    let api_addr = format!("{bind_host}:{api_port}");
    let api_listener = reexec::adopt_or_bind(inherited.api, &api_addr)?;
    api_listener
        .set_nonblocking(true)
        .with_context(|| format!("failed to configure listener on {api_addr}"))?;

    let web_listener = if web_port != 0 {
        // `[web].host` when it names one, else the API's own — see
        // `ServerConfiguration::web_host`.
        let web_addr = format!("{}:{web_port}", config::resolve_bind_host(web_host));
        let listener = reexec::adopt_or_bind(inherited.web, &web_addr)?;
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

    if let ModelSource::File(path) = &source {
        // Only a completely prepared server counts as use. Resolution alone
        // (`show`, `plan`, shell completion), or a load that later fails to
        // build its backend or bind its listener, must not change the date.
        if let Err(err) = orangu::model_registry::record_used(&model_label, path) {
            eprintln!("warning: could not update ~/.orangu/models: {err:#}");
        }
    }

    Ok(Prepared {
        api_key: conf.api_key.clone(),
        tls: conf.tls.clone(),
        engine,
        prefix_cache_snapshot: prefix_cache_dir.map(|dir| (dir, prefix_fingerprint)),
        model_label,
        quantization,
        architecture,
        backend_label,
        model_path: path,
        models_dir: conf.models,
        config_path,
        listen_override: reexec::Listen {
            host: args.listen.host.clone(),
            api: args.listen.port,
            web: args.listen.web,
        },
        role,
        reexec: conf.reexec,
        delete: conf.delete,
        bundle: match source {
            ModelSource::Embedded(bundle) => Some(bundle),
            ModelSource::File(_) => None,
        },
        mcp_servers: conf.mcp_servers,
        gpu_tuning,
        gpu_tuning_summary,
        wgpu_backend,
        workspace,
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

/// Whether a model label already names a tag of its own — the
/// `<user>/<model>:<quant>` spelling `--model` accepts (see
/// [`orangu::model_spec::ModelGroup::matches_label`]) — so the startup banner
/// appends the resolved quantization to a bare label only, never producing
/// `user/model:Q4_K_M:Q4_K_M`. Looked for in the last path segment rather
/// than the whole string, so a `:` somewhere in a directory name along a
/// model *path* isn't read as a tag.
fn label_carries_tag(label: &str) -> bool {
    label
        .rsplit(['/', '\\'])
        .next()
        .is_some_and(|last| last.contains(':'))
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

/// A bound listener's raw descriptor, for [`reexec::Handover`]. Zero on a
/// platform with no descriptors — where `Handover` is never built anyway,
/// since [`reexec::supported`] is `false` there.
fn listener_fd(listener: &tokio::net::TcpListener) -> i32 {
    #[cfg(unix)]
    {
        use std::os::fd::AsRawFd;
        listener.as_raw_fd()
    }
    #[cfg(not(unix))]
    {
        let _ = listener;
        0
    }
}

/// How many tokens the draft proposes per verification: the config key,
/// overridden for one run by `ORANGU_SPEC_DRAFT` — the variable the
/// prompt-lookup path already used, kept so a sweep of drafting depth reads
/// the same whichever drafter is in play.
fn draft_tokens(configured: usize) -> usize {
    std::env::var("ORANGU_SPEC_DRAFT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(configured)
}

/// Loads the draft half of a speculative pair.
///
/// The vocabulary check is the whole reason this is not two lines. A draft
/// model proposes *token ids*, and the target verifies them against its own
/// logits — so two models whose id 1234 means different text do not fail, they
/// agree at chance and produce a slower version of the right answer, or agree
/// spuriously and produce a wrong one. There is no symptom to notice. Checked
/// once here, against the token strings rather than only the vocabulary size,
/// because two tokenizers of the same size that disagree about what is in them
/// is exactly the case a size comparison passes.
fn load_draft_model(
    models_dir: &Path,
    spec: &str,
    tokens: usize,
    target_tokenizer: &Tokenizer,
    backend: &Arc<dyn Backend>,
    backend_label: &str,
) -> Result<engine::generate::DraftModel> {
    let (path, label) = orangu::model_spec::resolve_load_target(models_dir, spec)
        .with_context(|| format!("resolving draft model '{spec}'"))?;
    let gguf = GgufFile::open(&path)?;
    let draft_tokenizer = Tokenizer::from_gguf(&gguf).context("building draft tokenizer")?;
    if let Some(mismatch) = vocabulary_mismatch(target_tokenizer, &draft_tokenizer) {
        bail!(
            "draft model {label} does not share the served model's vocabulary ({mismatch}). \
             Speculation compares token ids between the two, so a pair that disagrees about \
             what an id means produces wrong or needlessly slow output with nothing to see"
        );
    }
    let loaded = engine::loader::LoadedModel::open(&path).context("loading draft weights")?;
    // The same backend as the target: a draft is only worth having if it runs
    // where the target runs, and a pair split across a device boundary would
    // pay a transfer per drafted token. No device split either — a draft small
    // enough to be worth drafting with is small enough to place whole.
    let unsupported = engine::backend::unsupported_tensor_types(loaded.tensor_types(), &**backend);
    if !unsupported.is_empty() {
        bail!(
            "backend {backend_label} has no kernel for the draft model's tensor type(s) {}",
            unsupported.join(", ")
        );
    }
    let model = build_model(&loaded, backend).context("building draft model")?;
    Ok(engine::generate::DraftModel {
        model,
        tokens,
        label,
    })
}

/// Refuses a model that cannot run a multi-position forward.
///
/// Probed rather than declared, because `ModelForward::forward_all_logits` is
/// a defaulted trait method — there is nothing to ask a type about, only
/// something to try. One token through it costs a single forward at startup
/// and answers the question for certain, which beats a hand-kept list of which
/// architectures implement it going quietly out of date.
fn supports_multi_position(model: &dyn ModelForward, label: &str) -> Result<()> {
    let mut cache = model.new_kv_cache(1);
    // Token 0 exists in every vocabulary; what comes back is discarded.
    model
        .forward_all_logits(&mut cache, &[0], 0, 0)
        .with_context(|| {
            format!(
                "{label} cannot be used in a speculative pair: speculation verifies several \
                 positions in one forward, and this architecture has no multi-position path"
            )
        })?;
    Ok(())
}

/// How two tokenizers differ, or `None` when they agree.
///
/// Compares the token strings themselves, not a count: the failure worth
/// catching is two vocabularies of identical size whose contents diverge,
/// which is what a same-family model at a different scale can easily be.
fn vocabulary_mismatch(target: &Tokenizer, draft: &Tokenizer) -> Option<String> {
    if target.vocab_size() != draft.vocab_size() {
        return Some(format!(
            "{} tokens against the served model's {}",
            draft.vocab_size(),
            target.vocab_size()
        ));
    }
    let differing = (0..target.vocab_size() as u32)
        .find(|&id| target.token_text(id) != draft.token_text(id))?;
    Some(format!(
        "same size, but token {differing} is {:?} in the draft and {:?} in the served model",
        draft.token_text(differing).unwrap_or_default(),
        target.token_text(differing).unwrap_or_default()
    ))
}

/// Builds the architecture's `ModelForward` from a loaded checkpoint.
///
/// One function rather than an inline `match`, because a speculative pair
/// loads two models and the second must go through exactly the same
/// construction as the first — an architecture reachable as the served model
/// but not as a draft would be a gap nothing announced.
/// One startup-banner deployment gate: `Yes` when it is configured, `No`
/// when it is not.
///
/// Just the value, on every start, whatever the bind — the banner is a
/// table of what this server resolved, and a row that sometimes carries
/// advice is not a table. What to do about a `No` is in the manual, under
/// `api_key` and `tls_cert`/`tls_key`; the line directly above these two is
/// the bound address, which is the other half of the question and is
/// already on screen.
fn gate(configured: bool) -> &'static str {
    if configured { "Yes" } else { "No" }
}

fn build_model(
    loaded: &engine::loader::LoadedModel,
    backend: &Arc<dyn Backend>,
) -> Result<Arc<dyn ModelForward>> {
    let architecture = loaded.config.architecture.clone();
    let model: Arc<dyn ModelForward> = match engine::loader::resolve_arch_family(&architecture)? {
        ArchFamily::LlamaStyle => Arc::new(
            LlamaModel::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::Gemma => Arc::new(
            GemmaModel::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::Qwen35Moe => Arc::new(
            Qwen35MoeModel::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::Qwen35 => Arc::new(
            Qwen35Model::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::Qwen3Next => Arc::new(
            Qwen3NextModel::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::Qwen4Exp => Arc::new(
            Qwen4ExpModel::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::DFlash => {
            Arc::new(DFlashModel::load_with_backend(loaded).context("building model")?)
        }
        ArchFamily::Deepseek4 => Arc::new(
            Deepseek4Model::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::GlmDsa => Arc::new(
            GlmModel::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::Glm5Next => Arc::new(
            Glm5Model::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::KimiK3 => Arc::new(
            Kimi3Model::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::Phi3 => Arc::new(
            PhiModel::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::Mistral3 => Arc::new(
            MistralModel::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::Muse => Arc::new(
            MuseModel::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::Inkling => Arc::new(
            InklingModel::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::NemotronHMoe => Arc::new(
            NemotronModel::load_with_backend(loaded, backend.clone()).context("building model")?,
        ),
        ArchFamily::BailingMoe3 => Arc::new(
            BailingMoeModel::load_with_backend(loaded, backend.clone())
                .context("building model")?,
        ),
    };
    Ok(model)
}

async fn serve(prepared: Prepared) -> Result<()> {
    let Prepared {
        api_key,
        tls,
        engine,
        prefix_cache_snapshot,
        model_label,
        quantization,
        architecture,
        backend_label,
        model_path,
        models_dir,
        config_path,
        listen_override,
        role,
        reexec: reexec_allowed,
        delete: delete_allowed,
        bundle,
        mcp_servers,
        gpu_tuning,
        gpu_tuning_summary,
        wgpu_backend,
        workspace,
        api_listener,
        web_listener,
        daemon,
    } = prepared;

    // A handle to the pool taken before `engine` is moved into the router's
    // state, so the snapshot can still be written once serving has stopped.
    let prefix_cache_for_snapshot = engine.prefix_cache.clone();

    let listener = tokio::net::TcpListener::from_std(api_listener)
        .context("failed to attach listener to the async runtime")?;
    // Loaded before anything is served, so a bad certificate is a startup
    // failure naming the file rather than a server that came up in the clear.
    let tls_config = match &tls {
        Some((cert, key)) => Some(tls::server_config(&tls::TlsPaths {
            cert: cert.clone(),
            key: key.clone(),
        })?),
        None => None,
    };
    let scheme = if tls_config.is_some() {
        "https"
    } else {
        "http"
    };
    let web_listener = match web_listener {
        Some(l) => Some(
            tokio::net::TcpListener::from_std(l)
                .context("failed to attach web UI listener to the async runtime")?,
        ),
        None => None,
    };

    // How the model is *named* to a human — `MODEL:QUANT`, so which of a
    // repo's quantizations is actually loaded is visible at a glance. Kept
    // apart from `model_label`, which is this server's model *id* on the API
    // (`/v1/models`, every response's `model` field) and in the slot-store
    // fingerprint, and so has to stay exactly the string it was resolved
    // from. Shared by the startup banner and the web UI, which say the same
    // thing about the same process.
    let model_display = match &quantization {
        Some(quant) => format!("{model_label}:{quant}"),
        None => model_label.clone(),
    };

    // Captured before `api_key` moves into `AppState`, for the exposure note
    // further down.
    let has_api_key = api_key.is_some();
    let (shutdown_tx, mut shutdown_rx) = tokio::sync::mpsc::channel::<()>(1);
    let state = Arc::new(http::AppState {
        engine: engine.clone(),
        api_key,
        model_label: model_label.clone(),
        backend_label: backend_label.clone(),
        gpu_tuning,
        wgpu_backend: wgpu_backend.clone(),
        workspace: workspace.clone(),
        started_at: std::time::Instant::now(),
        shutdown_tx,
    });
    let app = http::build_router(state);

    if !daemon {
        let os = orangu::os::detect();
        let cpu = orangu::hardware::detect_cpu();
        let gpus = orangu::hardware::detect_gpus(cpu.total_memory_bytes);
        let power = orangu::hardware::detect_power();
        print!(
            "{}",
            orangu::hardware::format_report(&os, &cpu, &gpus, &power)
        );
        println!();
        println!(
            "Model      {model_display} ({architecture} arch, {backend_label}, {} layers, {} ctx)",
            // Trunk layers, not `block_count`. On a file carrying a
            // multi-token-prediction block the two differ, and the banner
            // must name what the forward pass runs — `Qwen3.8-27B` declares
            // 65 blocks and runs 64.
            engine.model.n_trunk_layer(),
            engine.model.config().n_ctx_train,
        );
        // Speculation changes how fast an answer arrives and never what it
        // says, so it belongs on the banner rather than in a note: it is a
        // property of this server worth seeing beside the model, and a pair
        // whose acceptance is poor is a configuration to notice, not a fault
        // to warn about.
        if let Some(draft) = &engine.draft {
            println!(
                "Draft      {} ({} arch, {} layers, {} tokens/step)",
                draft.label,
                draft.model.config().architecture,
                draft.model.n_trunk_layer(),
                draft.tokens,
            );
        }
        // Which kernels the device actually got, on the line under the device
        // that got them. A decode number is only comparable against another
        // one taken with the same kernels, and on a GPU whose defaults were
        // tuned elsewhere that is not something to assume — see
        // `VulkanBackend::tuning_report` for the full picture `/props`
        // carries.
        //
        // On a backend with no kernel selection to make, the same row says
        // what it runs instead of one (`Backend::reduced_surface`). The row
        // is never silent on a GPU: an absent line is what let a `--device
        // cuda:0` run look like a full-path run in every number taken from
        // it.
        if let Some(summary) = &gpu_tuning_summary {
            println!("Kernels    {summary}");
        }
        // Where the weights came from, when the answer isn't a file somebody
        // can point at: this binary is the model. Worth a line of its own —
        // it explains both why no models directory was needed and why the
        // executable is the size it is.
        if let Some(bundle) = bundle {
            println!(
                "Bundled    {} embedded in {}",
                orangu::format::format_bytes(bundle.bytes),
                bundle.exe.display()
            );
        }
        match &web_listener {
            Some(l) => println!("UI         {scheme}://{}", l.local_addr()?),
            None => println!("UI         disabled"),
        }
        // The bound address, not the configured `host`: `all` says nothing
        // about where to point a client, `0.0.0.0:8100` does.
        println!("API        {scheme}://{}", listener.local_addr()?);
        // The two deployment gates, on the two lines under the address they
        // are gates on, and printed on *every* start rather than only on the
        // ones where they are missing. A row that always has a value is a
        // thing a reader can check; a warning that appears conditionally is
        // a thing they learn to expect the absence of. What a `No` costs and
        // how to answer it is documented, not reprinted here — see `gate`.
        println!("API key    {}", gate(has_api_key));
        println!("TLS        {}", gate(tls_config.is_some()));
        println!("Workspace  {}", workspace.display());
        // The governor is a *state*, not a finding: it has an answer on
        // every machine, and `Performance` is as worth seeing as anything
        // else — it is what makes the throughput numbers below it
        // comparable. See `hardware::cpu_governor`.
        if let Some(governor) = orangu::hardware::cpu_governor() {
            println!("Frequency  {governor}");
        }
        // What is left as a `Note` is *conditions* — on battery, or already
        // near a critical temperature. Neither has a command as an answer,
        // and both explain a slow number that would otherwise look like the
        // engine's fault. The machine *settings* that used to print here are
        // documented instead: the CPU governor is the `Frequency` row above,
        // and GPU power levels are in the manual, where a line per card does
        // not have to be reprinted on every start.
        for advisory in orangu::hardware::power_advisories(&power) {
            println!("Note       {advisory}");
        }
    }

    if let Some(web_listener) = web_listener {
        // Captured here because this is the last point where both listeners
        // and every resolved setting are in hand at once. Raw descriptor
        // numbers only — the listeners themselves are about to be moved into
        // `axum::serve` and stay open for the life of the process, which is
        // exactly as long as a handover could need them.
        //
        // `None` when the config turned it off, or on a platform with no
        // `execve`; the model manager reads that and disables its Load
        // button rather than offering something that would only refuse.
        let handover = (reexec_allowed && reexec::supported())
            .then(|| {
                reexec::Handover::new(
                    config_path,
                    listen_override.clone(),
                    workspace.clone(),
                    role,
                    // The spec this process would be started with again. For
                    // an embedded model that is the reserved
                    // `bundle::EMBEDDED_SPEC`, not its label: the label names
                    // a Hugging Face repo, and a fallback that went to the
                    // network for a model already inside the file it is
                    // falling back into would fail exactly when the network
                    // is what's missing.
                    match bundle {
                        Some(_) => bundle::EMBEDDED_SPEC.to_string(),
                        None => model_label.clone(),
                    },
                    reexec::InheritedFds {
                        api: Some(listener_fd(&listener)),
                        web: Some(listener_fd(&web_listener)),
                    },
                )
                .map(Arc::new)
                .map_err(|err| {
                    if !daemon {
                        println!(
                            "Note       model loading from the web console is unavailable: {err:#}"
                        );
                    }
                })
                .ok()
            })
            .flatten();
        // Resolved here, once, from the tree this server was rooted at: a
        // code block downloaded out of a reply is a file for *that* project,
        // so it carries that project's licence or none. See
        // `WebState::project_licence` for why it is not read per render.
        let project_licence = orangu::license::Project::detect(&workspace);
        let web_state = Arc::new(web::WebState {
            engine,
            project_licence,
            model_display,
            architecture,
            backend_label,
            model_path,
            models_dir,
            workspace,
            version: VERSION,
            jobs: Default::default(),
            catalog: Default::default(),
            handover,
            can_delete: delete_allowed,
            bundled: bundle.is_some(),
            loading: Default::default(),
            mcp_servers,
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

    // One `select!` and one shutdown path either way — the TLS listener is an
    // `axum::serve::Listener` like the plain one, so only the listener changes.
    let service = app.into_make_service_with_connect_info::<std::net::SocketAddr>();
    let serve_api = async move {
        match tls_config {
            Some(config) => axum::serve(
                tls::TlsListener::new(listener, config).with_connect_info(),
                service,
            )
            .await
            .context("server error"),
            None => axum::serve(listener, service).await.context("server error"),
        }
    };
    tokio::select! {
        result = serve_api => {
            result?;
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

    // Written on the way out rather than after every turn: a snapshot is a
    // whole `KvCache` per entry, and paying that on each request would cost
    // more than the re-prefill it saves on any model small enough to hold.
    // The trade-off inverts on a streaming model, and if it ever needs to be
    // taken per turn, `save_to` is already crash-safe (temp file, rename).
    if let (Some(pool), Some((dir, fingerprint))) = (
        prefix_cache_for_snapshot.as_ref(),
        prefix_cache_snapshot.as_ref(),
    ) {
        match pool.save_to(dir, fingerprint) {
            Ok(n) if !daemon => println!("prefix cache saved {n} entries to {}", dir.display()),
            Ok(_) => {}
            // A snapshot that could not be written costs a cold start next
            // time, which is exactly the status quo — never a failed exit.
            Err(err) => eprintln!("orangu-server: prefix cache not saved: {err}"),
        }
    }

    Ok(())
}

/// The configuration for this run: the file `--config` names, else the one
/// the default search finds, else — **only for a bundled binary** — the
/// built-in answers [`config::bundled_configuration`] holds.
///
/// That last case is the whole point of a bundle. An ordinary
/// `orangu-server` has nothing to serve and nowhere to look for one without
/// being told, so a missing config file is a plain error and always has
/// been; a bundled one is carrying its model, and needs a config file for
/// nothing but the things it already has defaults for.
fn load_config(
    explicit: Option<PathBuf>,
    cli_role: Option<config::Role>,
    daemon: bool,
) -> Result<ServerConfiguration> {
    let path = match explicit.or_else(default_server_config_path) {
        Some(path) => path,
        None => {
            let bundle = bundle::embedded().ok_or_else(|| {
                anyhow!(
                    "Missing config file; pass --config or add ./orangu-server.conf or ~/.orangu/orangu-server.conf (see --init)"
                )
            })?;
            return Ok(config::bundled_configuration(
                default_models_dir(),
                cli_role.unwrap_or(bundle.role),
                &bundle.listen,
            ));
        }
    };
    load_server_configuration(&path, cli_role, daemon)
        .with_context(|| format!("loading {}", path.display()))
}

/// Where models live for a run with no config file to say: the Hugging Face
/// hub cache, the same directory `--init` offers first and the one a
/// `huggingface-cli`/`llama.cpp` download already went to. `./models` only
/// when there is no home directory to hang it off, which is a broken
/// environment rather than a supported layout — an empty listing there is
/// still a better answer than refusing to start.
fn default_models_dir() -> PathBuf {
    init::huggingface_cache_dir().unwrap_or_else(|| PathBuf::from("models"))
}

/// `[orangu-server].models` when a config file can be found and read, and
/// [`default_models_dir`] otherwise — for `bundle`, which needs somewhere to
/// resolve and fetch a model spec but no configuration of its own.
fn models_dir_or_default(config_arg: Option<PathBuf>) -> PathBuf {
    load_config(config_arg, None, false)
        .map(|conf| conf.models)
        .unwrap_or_else(|_| default_models_dir())
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
pub(crate) fn model_support(
    groups: &[orangu::model_spec::ModelGroup],
) -> Vec<orangu::model_spec::ModelSupport> {
    groups
        .iter()
        .map(|group| {
            // Every shard, not just the representative: a split model's
            // later shards carry their own tensor directory, and can use a
            // quantization shard 1 never does. Headers only — no tensor
            // data is read, so this stays cheap even for a many-shard model.
            let mut architecture = None;
            let mut unsupported_quant = None;
            for path in &group.paths {
                let Ok(gguf) = GgufFile::open(path) else {
                    continue;
                };
                let (arch, bad_quant) = engine::loader::model_load_support(&gguf);
                architecture = architecture.or(arch);
                unsupported_quant = unsupported_quant.or(bad_quant);
            }
            let supported = architecture
                .as_deref()
                .is_some_and(|arch| engine::loader::resolve_arch_family(arch).is_ok());
            orangu::model_spec::ModelSupport {
                architecture,
                supported,
                unsupported_quant,
            }
        })
        .collect()
}

/// Runs one of the GGUF-inventory subcommands (`system`/`suggest`/`list`/
/// `show`/`download`) to completion and returns — none of these load a
/// model or bind a listener, so there's no `tokio` runtime involved, unlike
/// [`serve`]. `system`/`suggest` don't even need a config file (they only
/// ever look at the local machine's own hardware, plus — for `system` — the
/// models directory a config names, if there is one); `list`/`show`/
/// `download` resolve against the same `[orangu-server].models` directory the
/// serving path uses, via the same [`load_config`] — `cli_role`/`daemon` are
/// passed as `None`/`false` since neither matters to a subcommand that never
/// serves anything.
fn run_command(
    config_arg: Option<PathBuf>,
    cli_role: Option<config::Role>,
    cli_listen: &ListenFlags,
    command: Command,
) -> Result<()> {
    match command {
        Command::System => {
            let mut os = orangu::os::detect();
            // Best-effort, unlike every other subcommand's `load_config?`:
            // `system` reports the machine, and has to keep working on one
            // with no config file at all (the first thing run after an
            // install, and what a bug report is asked for). A config that
            // is there adds the models directory's disk use and the room
            // left for the next download; one that isn't just leaves those
            // lines out.
            os.models = load_config(config_arg, None, false)
                .ok()
                .and_then(|conf| orangu::os::detect_model_storage(&conf.models));
            let cpu = orangu::hardware::detect_cpu();
            let gpus = orangu::hardware::detect_gpus(cpu.total_memory_bytes);
            let power = orangu::hardware::detect_power();
            print!(
                "{}",
                orangu::hardware::format_report(&os, &cpu, &gpus, &power)
            );
            Ok(())
        }
        Command::Suggest => {
            let mut os = orangu::os::detect();
            // Same best-effort load as `system`, and for the same reason:
            // a machine with no config file still gets a suggestion, and
            // one with a config gets the models directory measured. Without
            // this the report claimed the machine had no models directory
            // while `list` was printing the models sitting in it.
            os.models = load_config(config_arg, None, false)
                .ok()
                .and_then(|conf| orangu::os::detect_model_storage(&conf.models));
            let cpu = orangu::hardware::detect_cpu();
            let gpus = orangu::hardware::detect_gpus(cpu.total_memory_bytes);
            print!("{}", suggest::format_suggestion(&os, &cpu, &gpus));
            Ok(())
        }
        Command::List { sort } => {
            let conf = load_config(config_arg, None, false)?;
            let models = orangu::model_spec::scan_models_dir(&conf.models)?;
            let groups = orangu::model_spec::group_models(&models);
            let latest_commits = check_for_updates(&groups);
            let support = model_support(&groups);
            let last_used = orangu::model_registry::last_used_for(
                groups.iter().map(|group| group.paths.as_slice()),
            );
            let order = list_order(&groups, &last_used, sort);
            print!(
                "{}",
                orangu::model_spec::format_groups_with_last_used_in_order(
                    &groups,
                    &conf.models,
                    &latest_commits,
                    &support,
                    dimming(orangu::model_spec::Dimming::Unsupported),
                    Some(&last_used),
                    &order,
                )
            );
            Ok(())
        }
        Command::Plan { file, deep } => {
            let conf = load_config(config_arg, None, false)?;
            let path = match file {
                Some(spec) => orangu::model_spec::resolve_show_target(&conf.models, &spec)?,
                None => select_model_for_show(&conf.models)?,
            };
            let plan = engine::plan::analyze(&path)?;
            let cpu = orangu::hardware::detect_cpu();
            let gpus = orangu::hardware::detect_gpus(cpu.total_memory_bytes);
            print!(
                "{}",
                engine::plan::format_plan(
                    &plan,
                    cpu.available_memory_bytes,
                    plan_device(&gpus).as_ref().map(|(n, v)| (n.as_str(), *v)),
                )
            );
            if deep {
                // Everything the plan assumed, checked. A plan is only worth
                // acting on if the files it described are actually there and
                // this build can actually read them.
                let gguf = GgufFile::open(&path)?;
                let shards = engine::loader::shard_paths(&path, &gguf)?;
                let mut problems = Vec::new();
                for shard in &shards {
                    match std::fs::metadata(shard) {
                        Ok(meta) if meta.len() > 0 => {}
                        Ok(_) => problems.push(format!("{} is empty", shard.display())),
                        Err(err) => problems.push(format!("{}: {err}", shard.display())),
                    }
                }
                if let Err(err) = engine::loader::resolve_arch_family(&plan.architecture) {
                    problems.push(format!("architecture: {err}"));
                }
                if problems.is_empty() {
                    println!(
                        "Check      {} shard(s) readable, architecture supported",
                        shards.len()
                    );
                } else {
                    for problem in &problems {
                        println!("Problem    {problem}");
                    }
                    anyhow::bail!("{} problem(s) found", problems.len());
                }
            }
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
        Command::Download { repo, yes } => {
            let conf = load_config(config_arg, None, false)?;
            if !plan_before_download(&repo, yes)? {
                println!("Nothing downloaded.");
                return Ok(());
            }
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
            // The quantization is named explicitly, not just the label: two
            // quantizations of one repo now share a `MODEL` cell (`QUANT` is
            // what tells them apart in `list`), so the label alone wouldn't
            // say which of them this irreversible step is about to remove.
            let quant = group
                .quantization
                .as_deref()
                .map(|quant| format!("{quant}, "))
                .unwrap_or_default();
            if !yes {
                let confirmed = confirm(&format!(
                    "Delete '{}' ({quant}{} file{plural}, {}) from {}? [y/N]: ",
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
                "Deleted '{}' ({quant}{} file{plural}, {})",
                group.label,
                group.paths.len(),
                orangu::format::format_bytes(group.size_bytes),
            );
            Ok(())
        }
        Command::Refresh { model, all, yes } => {
            let conf = load_config(config_arg, None, false)?;
            refresh::run(&conf.models, model, all, yes)
        }
        Command::Bundle {
            model,
            output,
            binary,
            yes,
            roles,
            listen,
        } => bundle::run(bundle::Request {
            // Unlike every other subcommand, a missing config file is not an
            // error here: `bundle` exists to produce a server for a machine
            // that has none, and is quite reasonably run on one. The models
            // directory is only where a spec is resolved and downloaded, and
            // the default download location is a fine answer for that.
            models_dir: models_dir_or_default(config_arg),
            model,
            // After the subcommand first, since that is where it reads most
            // naturally (`bundle <model> --code`); before it still works, so
            // a habit formed on the serving flags carries over.
            role: roles.role().or(cli_role),
            // Same precedence as the role: after the subcommand first, then
            // before it — so `bundle --host all` and `--host all bundle`
            // both bake the address in.
            listen: listen.or(cli_listen).bundled(),
            output,
            binary,
            yes,
        }),
        Command::Prune { identifier, yes } => prune::run(identifier, yes),
    }
}

/// Display order for `list`. Indices always refer to the canonical,
/// alphabetically grouped inventory, which is also the order numeric model
/// resolution uses. A sorted table can therefore move a row without changing
/// its `NR`.
fn list_order(
    groups: &[orangu::model_spec::ModelGroup],
    last_used: &[Option<u64>],
    sort: Option<ListSort>,
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..groups.len()).collect();
    match sort {
        None => {}
        Some(ListSort::Size) => {
            order.sort_by_key(|&index| std::cmp::Reverse(groups[index].size_bytes));
        }
        Some(ListSort::LastUsed) => {
            order.sort_by(|&left, &right| last_used[right].cmp(&last_used[left]));
        }
    }
    order
}

/// The latest downloadable state for every distinct Hugging Face repo
/// `groups` came from - deduped by repo id, so a repo with several `:quant`
/// rows costs one lookup, not one per row. Shared by `list` (which marks
/// rows `(Refresh)` only when that row's own files changed) and `refresh`
/// (which greys the ones that did not), so the two always agree on what is
/// stale. Failures are swallowed per repo by
/// [`orangu::model_download::latest_repo_updates`] itself: an unreachable
/// Hub yields an empty map, never an error.
pub(crate) fn check_for_updates(
    groups: &[orangu::model_spec::ModelGroup],
) -> std::collections::HashMap<String, orangu::model_download::RepoUpdateInfo> {
    let repos: Vec<String> = groups
        .iter()
        .filter_map(|g| g.hf_repo.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    orangu::model_download::latest_repo_updates(&repos)
}

/// `mode` when stdout is a terminal, [`Dimming::Off`] when it isn't — every
/// caller that prints `list`'s table wants exactly this. Redirected or piped
/// output stays escape-free, which is what the shell-completion scripts
/// parsing it by column depend on.
///
/// [`Dimming::Off`]: orangu::model_spec::Dimming::Off
pub(crate) fn dimming(mode: orangu::model_spec::Dimming) -> orangu::model_spec::Dimming {
    if std::io::stdout().is_terminal() {
        mode
    } else {
        orangu::model_spec::Dimming::Off
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
            dimming(orangu::model_spec::Dimming::Unsupported),
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
            dimming(orangu::model_spec::Dimming::Unsupported),
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

/// Plans a `download` target against this machine **before** fetching it,
/// and answers whether the download should go ahead.
///
/// A GGUF file states what it needs in its tensor table, and the table sits
/// at the front of the file, so `engine::plan` can answer "what would this
/// cost to run here" from a few hundred kilobytes of each shard's header
/// rather than from the model. That is the whole reason to do this before
/// the download instead of after it: the answer arrives while it can still
/// change the decision, in seconds, against a repo that may be hundreds of
/// gigabytes.
///
/// Only a model that **cannot run here** stops to ask. The test is
/// `Plan::dense_fits_in` — the same one the printed verdict is written from,
/// so the prompt can never contradict the report above it. A model whose
/// experts do not fit does *not* prompt: those stream from disk, so that
/// model is slow rather than broken, and a warning there would be crying
/// wolf on the case orangu is specifically built to handle.
///
/// Planning is a courtesy and never a gate. Every failure — no network, a
/// rate limit, a private repo, a header this build cannot parse — returns
/// `true` and lets the download proceed. The one thing this must not do is
/// turn a working download into a failure because the advisory step ahead
/// of it did not work.
/// The `ggml_type`s the banner names as the kernels this model decodes
/// through, most common first.
///
/// Floats are excluded **when the file has anything else**, because on a
/// quantized model they carry the norms and biases and would otherwise
/// outnumber the type the weight bytes are actually in.
///
/// They are counted when they are all there is. A `BF16` or `F16` file has no
/// quantized tensors at all, so the unconditional exclusion left the set empty
/// and the banner read `Kernels none` — which says "this build has no kernel
/// for your model" to a reader, when the truth is the opposite. The rule was
/// right about its own case and wrong about the case it did not consider.
///
/// Ordered by count and then by type id, so two runs of the same file print
/// the same line rather than following `HashMap` iteration order.
fn dominant_tensor_types<'a>(tensors: impl Iterator<Item = (&'a str, u32)>) -> Vec<u32> {
    let is_float = |ty: u32| {
        ty == engine::quant::GGML_TYPE_F32
            || ty == engine::quant::GGML_TYPE_F16
            || ty == engine::quant::GGML_TYPE_BF16
    };
    let mut quantized: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    let mut floats: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for (_, ty) in tensors {
        *if is_float(ty) {
            &mut floats
        } else {
            &mut quantized
        }
        .entry(ty)
        .or_default() += 1;
    }
    let counts = if quantized.is_empty() {
        floats
    } else {
        quantized
    };
    let mut types: Vec<(u32, usize)> = counts.into_iter().collect();
    types.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    types.into_iter().map(|(ty, _)| ty).collect()
}

fn plan_before_download(repo: &str, yes: bool) -> Result<bool> {
    // Resolution failures are silent on purpose. Everything that can go
    // wrong here — the repo missing, the token rejected, no file matching
    // the `:quant` — is something `download_model` is about to hit while
    // making the very same two Hub calls, and reporting it twice tells the
    // user nothing the second line didn't. A *plan* failure below is
    // different: the download will sail past a header this build cannot
    // parse, so nothing else would ever mention it.
    let Ok(model) = orangu::model_download::resolve_remote_model(repo) else {
        return Ok(true);
    };
    let plan = match engine::plan::analyze_shards(model.headers()) {
        Ok(plan) => plan,
        Err(err) => {
            eprintln!("orangu-server: could not plan {repo} before downloading: {err}");
            return Ok(true);
        }
    };

    let cpu = orangu::hardware::detect_cpu();
    let gpus = orangu::hardware::detect_gpus(cpu.total_memory_bytes);
    println!(
        "Download   {} · {} · {}",
        repo,
        model.commit.chars().take(7).collect::<String>(),
        orangu::format::format_bytes(model.total_bytes()),
    );
    print!(
        "{}",
        engine::plan::format_plan(
            &plan,
            cpu.available_memory_bytes,
            plan_device(&gpus).as_ref().map(|(n, v)| (n.as_str(), *v)),
        )
    );

    if plan.dense_fits_in(cpu.available_memory_bytes) || yes {
        return Ok(true);
    }
    confirm(
        "\nThis model is larger than this machine's RAM and will stream from disk, which is slow. Download anyway? [y/N]: ",
    )
}

/// The GPU a plan should be judged against — the one the server would load
/// onto — as `(name, capacity)`, or `None` on a machine with no dedicated
/// card.
///
/// The largest *dedicated* GPU, which is `suggest`'s own
/// `largest_dedicated_gpu` rather than a second rule, and which agrees with
/// what `engine::backend::device`'s preference order picks: discrete
/// outranks integrated, so on a laptop with both, the 4 GiB discrete card is
/// the target even though the integrated one reports far more.
///
/// That last point is the whole reason this is not a one-liner. The obvious
/// spelling — the largest `vram_total_bytes` over every GPU — reads an
/// integrated GPU's *shared* pool, which is all of system RAM, and so
/// reports 62 GiB of "VRAM" on a machine whose real ceiling is 4 GiB. It
/// also double-counts: that memory is already on the RAM line beside it.
fn plan_device(gpus: &[orangu::hardware::GpuInfo]) -> Option<(String, u64)> {
    let gpu = suggest::largest_dedicated_gpu(gpus)?;
    Some((gpu.name.clone(), gpu.vram_total_bytes?))
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

pub(crate) fn format_show(gguf: &GgufFile, full: bool, tensors: bool) -> String {
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
/// `orangu-server list` prints, `LAST_USED` and `SUPPORTED` columns and all
/// — models this build can't load are shown greyed rather than hidden: a
/// user can still pick one and will hit the same clear "not yet supported"
/// error `prepare` gives for any other unsupported model) and prompts for an
/// `NR`, for `orangu-server` invoked with no model argument. Returns the
/// chosen model's file path and its display label.
///
/// The one thing not carried over from `list` is the `(Refresh)` marker:
/// that costs a Hugging Face round trip per repo, and this table is on the
/// path to starting a server, not to maintaining the models directory.
///
/// A directory holding exactly one model ([`init::sole_model`], shared with
/// the `--init` wizard's own `model` prompt) skips the `NR` prompt entirely
/// and goes straight on to the caller's role prompt.
fn select_model_interactively(
    models_dir: &Path,
    configured: Option<&str>,
) -> Result<(PathBuf, String)> {
    let models = orangu::model_spec::scan_models_dir(models_dir)
        .with_context(|| format!("scanning {}", models_dir.display()))?;
    let groups = orangu::model_spec::group_models(&models);
    if groups.is_empty() {
        bail!(
            "no .gguf models found under {}; download one first (e.g. `orangu-server download <user>/<model>`) or pass one directly: orangu-server <model>",
            models_dir.display()
        );
    }

    let last_used =
        orangu::model_registry::last_used_for(groups.iter().map(|group| group.paths.as_slice()));
    print!(
        "{}",
        orangu::model_spec::format_groups_with_last_used(
            &groups,
            models_dir,
            &Default::default(),
            &model_support(&groups),
            dimming(orangu::model_spec::Dimming::Unsupported),
            Some(&last_used),
        )
    );

    // A single listed model is not a choice — take it and move straight on
    // to the role prompt, rather than asking for the only NR on offer. The
    // table above is still printed: it's what names the model being taken,
    // and whether this build supports it at all. Echoed in the same
    // `key: value` shape as the `role [all]: ` prompt that follows, matching
    // how `--init` echoes the same decision.
    if let Some(only) = init::sole_model(&groups) {
        println!("\nmodel: {}", only.label);
        return Ok((only.representative_path.clone(), only.label.clone()));
    }

    // `[orangu-server].model`, as the `NR` of the row it names — the prompt
    // asks for a number, so that is the form to pre-select it in. A config
    // naming a model that isn't installed has no row to point at; its spec
    // is offered as written instead, and Enter fetches it exactly as
    // `orangu-server <spec>` would.
    let default = configured.map(|spec| match init::nr_of(&groups, spec) {
        Some(nr) => nr.to_string(),
        None => spec.to_string(),
    });
    let answer = init::prompt_model_nr(&groups, default.as_deref())?;
    // Enter on the ghost leaves the line blank; put the value that was
    // actually taken back on it, so the transcript says what was chosen.
    init::echo_answer("Model: ", &answer);

    // An `NR` is answered straight out of the table already in hand. Anything
    // else — a label, a path, a repo still to fetch — goes through the same
    // resolution the positional argument uses, so the prompt accepts
    // everything that does.
    if let Ok(nr) = answer.parse::<usize>() {
        let group = nr
            .checked_sub(1)
            .and_then(|index| groups.get(index))
            .ok_or_else(|| anyhow!("no model with NR {nr} ({} model(s) listed)", groups.len()))?;
        return Ok((group.representative_path.clone(), group.label.clone()));
    }
    orangu::model_spec::resolve_load_target(models_dir, &answer)
        .with_context(|| format!("resolving model '{answer}'"))
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

/// Retries `init` a few times with a short backoff before giving up.
///
/// `VulkanBackend::try_init` can transiently return `None` — its
/// `request_adapter`/`request_device` fail intermittently right after a prior
/// process released the GPU (the driver hasn't finished tearing the previous
/// context down), which surfaces as a flaky "no usable Vulkan adapter" at
/// startup. Retrying (silently) means a transient race doesn't sink the whole
/// server; the caller prints a single error if this still returns `None`. A
/// genuine absence of a device also returns `None`, only a little later — each
/// attempt is fast when there's no adapter.
///
/// Generic over the constructor rather than hardcoding `VulkanBackend` because
/// `MetalBackend` comes up through the same `wgpu` machinery and so has the
/// same transient-failure window.
fn init_gpu_with_retry<B>(init: impl Fn() -> Option<B>) -> Option<B> {
    const ATTEMPTS: usize = 4;
    for attempt in 1..=ATTEMPTS {
        if let Some(backend) = init() {
            return Some(backend);
        }
        if attempt < ATTEMPTS {
            std::thread::sleep(std::time::Duration::from_millis(700));
        }
    }
    None
}

/// The CPU's own name, for a device list that has to name it beside the
/// GPUs — the brand string when `sysinfo` has one, `"CPU"` otherwise.
fn cpu_label() -> String {
    let brand = orangu::hardware::detect_cpu().brand;
    if brand.trim().is_empty() {
        "CPU".to_string()
    } else {
        brand
    }
}

/// The CPU's line in the startup device inventory, so one block covers
/// every processor in the machine rather than only the GPUs.
///
/// `role` says what it is doing in this run — running the model, holding
/// the layers that did not fit a device, or nothing. The last is worth a
/// line too: on a machine whose GPU is doing the work, the CPU's core count
/// and instruction set are still what the tokenizer, the sampler and (on a
/// split model) attention run on.
fn cpu_inventory(role: &str, threads: Option<usize>) -> String {
    let cpu = orangu::hardware::detect_cpu();
    let mut detail = Vec::new();
    match cpu.physical_cores {
        Some(cores) => detail.push(format!("{cores} cores / {} threads", cpu.logical_cores)),
        None => detail.push(format!("{} threads", cpu.logical_cores)),
    }
    // The widest instruction set `engine::vecdot` will actually dispatch to,
    // which is what decides the CPU matmul's speed — not the full feature
    // list, which belongs in the `system` report.
    detail.push(
        if cpu.features.avx512f {
            "AVX-512"
        } else if cpu.features.avx2 {
            "AVX2"
        } else if cpu.features.sse4_2 {
            "SSE4.2"
        } else {
            "scalar"
        }
        .to_string(),
    );
    detail.push(format!(
        "{} RAM",
        orangu::format::format_bytes(cpu.total_memory_bytes)
    ));
    detail.push(match threads {
        Some(threads) => format!("{threads} worker threads"),
        None => format!("{} worker threads (default)", cpu.logical_cores),
    });
    format!(
        "orangu-server: [cpu] {} [{}] {role}",
        cpu_label(),
        detail.join(", ")
    )
}

/// Sizes the worker pool every CPU path in this process shares.
///
/// One pool, set once, before anything parallel runs: `CpuBackend`'s
/// matmul, the MoE expert loop (`engine::arch::project_expert`) and the
/// per-expert fan-out all go through `rayon`'s global pool, so the knob
/// belongs there rather than on any one of them.
///
/// `None` — the default — leaves rayon's own choice (one worker per logical
/// core) untouched, so a config that says nothing about threads keeps
/// exactly the behaviour it had. Returns what was applied, for the
/// inventory line.
fn configure_cpu_threads(flag: Option<&str>, configured: Option<usize>) -> Result<Option<usize>> {
    let requested = match flag {
        Some(raw) => Some(
            raw.trim()
                .parse::<usize>()
                .map_err(|_| anyhow!("--threads {raw:?} is not a number"))?,
        ),
        None => match std::env::var("ORANGU_THREADS") {
            Ok(raw) => Some(
                raw.trim()
                    .parse::<usize>()
                    .map_err(|_| anyhow!("ORANGU_THREADS={raw:?} is not a number"))?,
            ),
            Err(_) => configured,
        },
    };
    let Some(threads) = requested else {
        return Ok(None);
    };
    if threads == 0 {
        bail!("threads must be at least 1 (leave it unset for one worker per logical core)");
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .map_err(|err| anyhow!("could not size the worker pool to {threads} threads: {err}"))?;
    Ok(Some(threads))
}

/// Chooses which routed experts a device holds, and records it on the model
/// so `LoadedModel::expert_matrix` can stamp each tensor.
///
/// Only when `ORANGU_GPU_EXPERTS=1` asked for the tier at all — without it
/// every expert stays on the host path and this does nothing.
///
/// **This is what bounds the tier.** `VulkanBackend::weight_buffer`'s arena
/// never evicts, so an expert that reaches the device stays there; choosing
/// the resident set up front is the only thing standing between a tier and
/// unbounded VRAM growth. `engine::expert_tier::plan` does the choosing —
/// whole experts, hottest first, fastest device first — and heat is uniform
/// here because a routing profile belongs to a session and this runs before
/// the first token. That makes this the floor a tier achieves, which is
/// what the startup projection has always reported.
/// Returns whether a tier was actually stamped onto the model. Every early
/// return here is a reason there is no tier, and the projection printed later
/// has to agree with that — the two lines sit three screens apart in the
/// startup output and a reader who believes the wrong one debugs the wrong
/// half of the engine.
fn plan_expert_tier(
    loaded: &mut engine::loader::LoadedModel,
    backend: &dyn Backend,
    weights_device_bytes: u64,
) -> bool {
    if !engine::arch::gpu_experts() {
        return false;
    }
    // Streaming makes a resident tier redundant *and* harmful: it holds
    // whatever this call needs in a bounded region that rewinds, so a fixed
    // subset admitted up front only takes VRAM the streaming region and the
    // KV cache then cannot have. Measured with both on, the card sat at
    // capacity.
    if engine::arch::expert_streaming() {
        eprintln!(
            "orangu-server: [{}] expert weights stream per batch; no resident tier is planned",
            backend.as_wgpu().map_or("cpu", |w| w.api_tag()),
        );
        return false;
    }
    let Some(wgpu) = backend.as_wgpu() else {
        return false;
    };
    let Some(total) = wgpu.device_in_use().vram_total_bytes else {
        // A device that will not say how big it is cannot be given a
        // budget, and guessing one is how an arena silently overruns.
        return false;
    };
    let tensors = loaded.expert_tensors();
    if tensors.is_empty() {
        return false;
    }
    // Half the headroom after the dense weights, leaving the rest for the
    // KV cache and the transient arenas — the same reservation the startup
    // projection reports against, and for the same reason it cannot be
    // computed exactly here.
    let budget = total.saturating_sub(weights_device_bytes) / 2;

    // Real routing heat when a previous session left one
    // (`ORANGU_EXPERT_USAGE`), by size otherwise. The difference is not
    // marginal: colibri measured the same tier 3-5x apart depending on
    // which of the two filled it, and this tier holds only about half the
    // experts, so the choice of *which* half is most of its value.
    let learned = engine::expert_store::learned_heat();
    let mut heat = Vec::new();
    for (name, n_expert, bytes) in &tensors {
        let key: std::sync::Arc<str> = std::sync::Arc::from(name.as_str());
        for expert in 0..*n_expert {
            heat.push(engine::expert_tier::ExpertHeat {
                bytes: *bytes,
                // Zero for an expert no history mentions. `expert_tier::
                // plan` still admits those once the hot ones are placed —
                // unused VRAM serves nobody — they simply cannot displace
                // one routing actually asked for.
                heat: learned.get(&(key.clone(), expert)).copied().unwrap_or(0) as u64,
            });
        }
    }
    let profiled = heat.iter().any(|e| e.heat > 0);
    let plan = engine::expert_tier::plan(&heat, &[budget]);

    let mut residency = std::collections::HashMap::new();
    let mut at = 0usize;
    for (name, n_expert, _) in &tensors {
        let flags: std::sync::Arc<[bool]> = plan.device_of[at..at + n_expert]
            .iter()
            .map(Option::is_some)
            .collect();
        residency.insert(name.clone(), flags);
        at += n_expert;
    }
    // Before anything is stamped onto the model: a tier that cannot pay is
    // not merely useless, it is VRAM taken from the KV cache and the
    // transient arenas, which on a small card is the ceiling everything else
    // runs into. Declining is a decision worth printing with its number, not
    // a silent no-op.
    let floor = engine::expert_tier::coverage_floor();
    if !engine::expert_tier::worth_building(plan.coverage(), floor) {
        eprintln!(
            "orangu-server: [{}] expert tier declined: the budget of {} would cover only \
             {:.1}% of recorded routing, under the {:.0}% floor — that VRAM goes to the KV \
             cache and the arenas instead. Set ORANGU_EXPERT_TIER_FLOOR=<percent> to override.",
            wgpu.api_tag(),
            orangu::format::format_bytes(plan.resident_bytes()),
            plan.coverage().unwrap_or(0.0) * 100.0,
            floor * 100.0,
        );
        return false;
    }
    eprintln!(
        "orangu-server: [{}] expert tier: {} of {} experts on device ({}), filled by {}",
        wgpu.api_tag(),
        plan.resident_count(),
        heat.len(),
        orangu::format::format_bytes(plan.resident_bytes()),
        // Which of the two filled it is the thing that decides what the
        // tier is worth, so it is on the line rather than inferable.
        match (profiled, plan.coverage()) {
            (true, Some(coverage)) => format!(
                "measured routing heat ({:.1}% of recorded selections)",
                coverage * 100.0
            ),
            _ => "size (no routing profile — set ORANGU_EXPERT_USAGE)".to_string(),
        },
    );
    loaded.set_expert_residency(residency);
    true
}

/// What a device-resident expert tier in this device's free VRAM would
/// hold — empty for a dense model, which has no routed experts at all.
///
/// Sizes every expert from the model's own `*_exps.weight` tensors and the
/// architecture's `expert_count`, so the figure is this model's rather than
/// a class of model's. Heat is uniform: orangu's routing profile lives in
/// `engine::expert_store`'s sidecar and belongs to a *session*, while this
/// runs before the first token — so the projection reports the floor a tier
/// would achieve, and says that a real profile does better.
fn expert_tier_projection(
    loaded: &engine::loader::LoadedModel,
    footprint: &engine::footprint::DeviceFootprint,
    wgpu: &VulkanBackend,
    active: bool,
) -> Vec<String> {
    let expert_bytes = footprint.weights_host_bytes;
    if expert_bytes == 0 {
        return Vec::new();
    }
    let n_expert = loaded.metadata_u64("expert_count").unwrap_or(0) as usize;
    // The unit is `(tensor, expert)`, matching `engine::expert_store`'s own
    // granularity: GGUF keeps one layer's gate/up/down experts in three
    // separate stacked tensors, so a "seat" in a tier is one expert of one
    // of them. Sized from each tensor's real `expert_bytes` rather than by
    // dividing a total, since the three stacks of a layer are not the same
    // size as each other.
    let stacks: Vec<String> = loaded
        .tensor_sizes()
        .filter(|(name, _)| name.ends_with("_exps.weight"))
        .map(|(name, _)| name.to_string())
        .collect();
    if n_expert == 0 || stacks.is_empty() {
        return Vec::new();
    }
    let mut heat = Vec::with_capacity(n_expert * stacks.len());
    for name in &stacks {
        let Ok(stacked) = loaded.expert_matrix(name) else {
            return Vec::new();
        };
        heat.extend(std::iter::repeat_n(
            engine::expert_tier::ExpertHeat {
                bytes: stacked.expert_bytes(),
                heat: 1,
            },
            stacked.n_expert,
        ));
    }
    let headroom = footprint
        .headroom_on(wgpu.device_in_use())
        .unwrap_or(0)
        // The same headroom the KV cache and the transient arenas draw on,
        // so a tier cannot be projected into memory the model is already
        // going to need. Half is a deliberately conservative split, and it
        // is a projection either way.
        / 2;
    let slots = heat.len();
    let plan = engine::expert_tier::plan(&heat, &[headroom]);
    // Whether the tier this projects is the one actually running — passed in
    // from `plan_expert_tier`'s own outcome rather than re-derived from
    // `gpu_experts()`. The knob being on is no longer sufficient: a tier
    // under the coverage floor is declined with the knob set, and reading the
    // knob here printed "the tier above is active" directly beneath "expert
    // tier declined".
    engine::expert_tier::projection(wgpu.api_tag(), &plan, slots, true, active)
}

/// Spreads the model across the selected devices when asked to, wrapping
/// the already-built head backend rather than replacing it.
///
/// Runs *after* `select_backend`, not inside it, and the split is decided
/// from what that returned: the chosen backend still holds the device set
/// it selected from, so nothing has to be enumerated twice or threaded
/// through. The head device's backend is reused as-is — element 0 of the
/// wrapper — so a split costs one extra bring-up per additional device and
/// no rework of the first.
///
/// Only the `wgpu` backends can be split. `CudaBackend`/`OpenClBackend`/
/// `RocmBackend` are matmul-only implementations that no NVIDIA/OpenCL/ROCm
/// hardware has verified during development; giving them an untested
/// multi-device path would be a worse answer than not offering one, so a
/// split asked for on those is refused with that said out loud.
/// Whether the weights will not fit the device the backend selected.
///
/// The trigger for the automatic overflow split above. `false` for a non-wgpu
/// backend (nothing to overflow *from* — the CPU path already holds whatever
/// RAM and the page cache can between them) and `false` for a device that does
/// not report its size, which is the "unknown is not zero" rule the rest of
/// the capacity code follows.
fn overflows_selected_device(backend: &dyn Backend, weights_bytes: u64) -> bool {
    backend
        .as_wgpu()
        .and_then(|wgpu| wgpu.device_in_use().vram_total_bytes)
        .is_some_and(|total| weights_bytes > total)
}

fn apply_device_split(
    backend: Arc<dyn Backend>,
    label: String,
    mode: &SplitMode,
    per_layer_bytes: &[u64],
    weights_bytes: u64,
) -> Result<(Arc<dyn Backend>, String, Option<SplitReport>)> {
    /// The share of a device's memory a fill-in-order placement will put
    /// weights into, leaving the rest for the KV cache and the transient
    /// compute buffers.
    ///
    /// A heuristic, and the only one here. It cannot be computed: the KV
    /// geometry needs a built model, and the model cannot be built until
    /// placement is decided, since building it is what stamps each tensor's
    /// device. Explicit ratios exist for anyone who wants to set the
    /// boundary exactly, and the footprint report says afterwards how much
    /// headroom the choice actually left.
    const WEIGHTS_SHARE_OF_DEVICE: f64 = 0.8;

    if mode.is_off() {
        return Ok((backend, label, None));
    }
    let Some(wgpu) = backend.as_wgpu() else {
        bail!(
            "device_split = {mode}, but {label} cannot be split — only the wgpu backends \
             (vulkan, metal, dx12) place layers across devices. Use device_split = off, or \
             a backend that can."
        );
    };
    let set = wgpu.device_set();
    // What each device actually has, for the report — as distinct from the
    // reduced budget the fill plans against below. Showing the budget would
    // make a device look over-subscribed when it is merely holding back
    // room for the KV cache.
    let reported_capacities: Vec<Option<u64>> = set.iter().map(|c| c.vram_total_bytes).collect();
    let mut device_classes: Vec<DeviceClass> = set.iter().map(|c| c.class).collect();
    let mut capacities: Vec<Option<u64>> = set
        .iter()
        .map(|c| {
            c.vram_total_bytes
                .map(|total| (total as f64 * WEIGHTS_SHARE_OF_DEVICE) as u64)
        })
        .collect();
    // The first device pays for every tensor outside a numbered layer —
    // token embeddings, the output norm, `lm_head` — because that is where
    // `LoadedModel::device_for_tensor` puts them. Charging it only for the
    // layers it was given would overfill it by their size, which on a
    // large-vocabulary model is gigabytes: a live Kimi-K3 fill put 4.96 GiB
    // of weights on a card budgeted for 3.20 GiB before this line existed.
    let non_layer_bytes = weights_bytes.saturating_sub(per_layer_bytes.iter().sum::<u64>());
    if let Some(head) = capacities.first_mut().and_then(Option::as_mut) {
        *head = head.saturating_sub(non_layer_bytes);
    }
    let backends_bit = wgpu.backends_bit();
    let api = wgpu.api_tag().to_string();
    // Indices and names first, then drop the borrow: the head backend is
    // about to be moved into the wrapper, and `set` borrows from it.
    let indices: Vec<usize> = set.iter().map(|c| c.index).collect();
    let mut names: Vec<String> = set.iter().map(|c| c.name.clone()).collect();
    drop(set);

    // The host as the last device: unbounded, because its memory is system
    // RAM and the weights are mapped there already. It therefore takes
    // exactly what the devices could not hold, which is the whole point of
    // `SplitMode::Cpu` and is why it is a fill rather than a share.
    let overflow_to_cpu = matches!(mode, SplitMode::Cpu);
    let mut reported_capacities = reported_capacities;
    if overflow_to_cpu {
        capacities.push(None);
        reported_capacities.push(None);
        names.push(cpu_label());
        // The host tier is not a GPU at all; classed as software so the
        // placement note never mistakes it for a card worth preferring.
        device_classes.push(DeviceClass::Software);
    }

    let Some(plan) = placement::plan(mode, per_layer_bytes, &capacities, weights_bytes) else {
        return Ok((backend, label, None));
    };

    let mut devices: Vec<Arc<dyn Backend>> = Vec::with_capacity(capacities.len());
    // Captured per device as each one is brought up. After the loop the
    // backends are behind the wrapper and `as_wgpu` answers `None`, so this is
    // the only point at which each device can still be asked how it stores a
    // KV cache — which is what turns a layer count into a byte figure.
    let mut kv_storage: Vec<Option<KvStorage>> = vec![Some(wgpu.kv_storage())];
    devices.push(backend);
    for (position, &index) in indices.iter().enumerate().skip(1) {
        // `indices` holds only the wgpu devices; the host entry appended to
        // `capacities`/`names` above has no adapter to bring up.

        // A device the plan gave no layers to is still brought up: the
        // wrapper indexes by position, and skipping one would shift every
        // later device's weights onto the wrong card.
        let extra =
            init_gpu_with_retry(|| VulkanBackend::try_init_selected(backends_bit, &[index]))
                .ok_or_else(|| {
                    anyhow!(
                        "device_split = {mode}, but device {index} ({}) could not be brought up \
                     (device or pipeline creation failed)",
                        names[position]
                    )
                })?;
        kv_storage.push(Some(extra.kv_storage()));
        devices.push(Arc::new(extra));
    }

    if overflow_to_cpu {
        // The host tier has no GPU KV mirror, which is a different answer
        // from "not measured" and is what `None` says here.
        kv_storage.push(None);
        devices.push(Arc::new(CpuBackend));
    }
    let label = format!(
        "{label} + {} more ({} devices, split)",
        devices.len() - 1,
        devices.len()
    );
    Ok((
        Arc::new(MultiDeviceBackend::new(devices)),
        label,
        Some(SplitReport {
            api,
            plan,
            device_names: names,
            device_capacities: reported_capacities,
            device_classes,
            device_kv_storage: kv_storage,
        }),
    ))
}

/// Everything a split run has to be able to say about itself afterwards.
///
/// Carried out of `apply_device_split` because nothing else can reconstruct
/// it: once the per-device backends are behind a `MultiDeviceBackend`,
/// `Backend::as_wgpu` answers `None` by design and the device metadata is no
/// longer reachable. Without this, splitting a model would *lose* the
/// device reporting that P0 exists to provide.
struct SplitReport {
    api: String,
    plan: SplitPlan,
    device_names: Vec<String>,
    device_capacities: Vec<Option<u64>>,
    /// What each device *is* — discrete, integrated, software. Carried
    /// alongside the capacities because shares are proportional to reported
    /// memory, and an integrated device reports system RAM: the two numbers
    /// are not the same quantity, and comparing them is how the largest share
    /// ends up on the slowest card. See [`Self::placement_note`].
    device_classes: Vec<DeviceClass>,
    /// How each device stores its KV mirror, captured while the concrete
    /// backends were still in hand. `None` for the host overflow tier, which
    /// has no GPU mirror at all.
    ///
    /// Read per device rather than once from the head, because it is a
    /// property of each device's own negotiated features — and taken here for
    /// the same reason the capacities are: after the wrapper is built there is
    /// nothing left to ask.
    device_kv_storage: Vec<Option<KvStorage>>,
}

impl SplitReport {
    /// A warning when the split handed its largest share to an integrated
    /// device while a discrete one was in the set.
    ///
    /// Shares are proportional to each device's *reported* memory, which is
    /// the right rule for a set of discrete cards and a trap on a laptop: an
    /// integrated GPU reports system RAM, so it can advertise five times a
    /// discrete card's VRAM while being several times slower. The result is a
    /// split that assigns most of the model to the slowest device — and
    /// nothing about the layer ranges says so, because "20 layers on Radeon
    /// Graphics" looks like a bigger card getting more work.
    ///
    /// Measured on this shape (24-layer model, 4 GiB discrete card beside an
    /// integrated GPU reporting 21 GiB): the memory-proportional split gave
    /// 4 layers to the discrete card and 20 to the integrated one, and
    /// reversing that with an explicit ratio was 1.4x faster at one stream
    /// and 2.0x at four. That is larger than anything the split itself buys
    /// back, so it is worth a line rather than a doc paragraph nobody reads
    /// at the moment it applies.
    ///
    /// Named as a *possible* improvement, not a misconfiguration: the
    /// integrated device may genuinely be the one with room, which is the
    /// whole reason the split exists.
    fn placement_note(&self) -> Option<String> {
        let (largest, _) = self
            .plan
            .per_device_layers
            .iter()
            .enumerate()
            .max_by_key(|&(index, layers)| (layers, std::cmp::Reverse(index)))?;
        if self.device_classes.get(largest) != Some(&DeviceClass::Integrated) {
            return None;
        }
        let discrete = self
            .device_classes
            .iter()
            .position(|class| *class == DeviceClass::Discrete)?;
        Some(format!(
            "orangu-server: [{}] most layers ({}) went to {}, which is integrated — shares are \
             proportional to reported memory, and an integrated device reports system RAM rather \
             than dedicated VRAM. If {} has the room, an explicit device_split ratio favouring it \
             is usually faster.",
            self.api,
            self.plan
                .per_device_layers
                .get(largest)
                .copied()
                .unwrap_or(0),
            self.device_names
                .get(largest)
                .cloned()
                .unwrap_or_else(|| format!("device {largest}")),
            self.device_names
                .get(discrete)
                .cloned()
                .unwrap_or_else(|| format!("device {discrete}")),
        ))
    }

    /// The startup lines: the layer ranges, what each device holds against
    /// what it has, and what the split costs.
    ///
    /// The cost is stated on the same screen as the split itself, because
    /// it is large and counter-intuitive: spreading a model across two
    /// cards makes it *slower* per token, not faster, and the reason to do
    /// it is that a model too big for one card runs at all.
    fn lines(&self, footprints: &[DeviceFootprint]) -> Vec<String> {
        let api = &self.api;
        let mut lines = vec![format!(
            "orangu-server: [{api}] split: {}",
            self.plan.describe(&self.device_names)
        )];
        for (device, footprint) in footprints.iter().enumerate() {
            let total = self.device_capacities.get(device).copied().flatten();
            let capacity = match total {
                Some(total) => format!(" of {}", orangu::format::format_bytes(total)),
                None => String::new(),
            };
            lines.push(format!(
                "orangu-server: [{api}] {}: {} weights{capacity}, {} layer{}",
                self.device_names
                    .get(device)
                    .cloned()
                    .unwrap_or_else(|| format!("device {device}")),
                orangu::format::format_bytes(footprint.weights_device_bytes),
                self.plan
                    .per_device_layers
                    .get(device)
                    .copied()
                    .unwrap_or(0),
                if self.plan.per_device_layers.get(device) == Some(&1) {
                    ""
                } else {
                    "s"
                },
            ));
            // What is left on that card, and what it buys — the question a
            // split is chosen to answer, and the one a layer count alone
            // cannot: two cards holding the same number of layers can have
            // wildly different KV shares, and a card can be full of weights
            // with no room left for the context the operator wants.
            if let Some(headroom) = footprint.headroom_in(total) {
                let mut line = format!(
                    "orangu-server: [{api}] {}: {} free after weights",
                    self.device_names
                        .get(device)
                        .cloned()
                        .unwrap_or_else(|| format!("device {device}")),
                    orangu::format::format_bytes(headroom)
                );
                if let (Some(tokens), Some(storage)) =
                    (footprint.kv_tokens_in(headroom), footprint.kv_storage)
                {
                    let ceiling = footprint.n_ctx_train.saturating_mul(footprint.slots);
                    let layers = self
                        .plan
                        .per_device_layers
                        .get(device)
                        .copied()
                        .unwrap_or(0);
                    let plural = if layers == 1 { "" } else { "s" };
                    // Two devices whose headroom both exceeds the ceiling
                    // would otherwise print the *same* capped number and look
                    // alike, when what they have is 918k tokens of room
                    // against 724k. Saying the ceiling binds is both shorter
                    // and the thing an operator can act on: a device that
                    // holds the whole context needs no further arithmetic.
                    if tokens >= ceiling {
                        line.push_str(&format!(
                            " — room for the full {ceiling}-token context in {storage:?} KV \
                             for its {layers} layer{plural}"
                        ));
                    } else {
                        line.push_str(&format!(
                            " — about {tokens} tokens of {storage:?} KV for its \
                             {layers} layer{plural}"
                        ));
                    }
                }
                lines.push(line);
            }
            if let Some(shortfall) = footprint.shortfall_in(total) {
                lines.push(format!(
                    "orangu-server: [{api}] {}: the weights placed here are {} larger than \
                     the device — the driver will page them on every token. Give this device \
                     a smaller share (device_split = <ratios>) or add a device.",
                    self.device_names
                        .get(device)
                        .cloned()
                        .unwrap_or_else(|| format!("device {device}")),
                    orangu::format::format_bytes(shortfall)
                ));
            }
        }
        lines.push(format!(
            "orangu-server: [{api}] a split model keeps its per-layer GPU work — fused \
             attention, fused FFN, the device-side KV cache — but gives up the whole-step \
             decode submission, which cannot span devices, and the hidden state crosses \
             the bus {} time{} per token. It buys capacity, not speed.",
            self.plan.boundaries(),
            if self.plan.boundaries() == 1 { "" } else { "s" }
        ));
        lines.extend(self.placement_note());
        lines
    }

    /// The same thing for `/props`, standing in for the tuning report a
    /// split run has no single device to produce.
    ///
    /// Each device carries a `footprint` under the same field names the
    /// single-device report uses (`DeviceFootprint::to_json_in`), so a reader
    /// parses one shape whether the run used one card or four.
    fn to_json(
        &self,
        footprints: &[DeviceFootprint],
        weights_host_bytes: u64,
    ) -> serde_json::Value {
        serde_json::json!({
            "api": self.api,
            "split": true,
            "boundaries_per_token": self.plan.boundaries(),
            // Once, at the top: routed experts are in system RAM and on no
            // device, so per-device footprints report zero for them and this
            // is where the total belongs.
            "weights_host_bytes": weights_host_bytes,
            "devices": self
                .device_names
                .iter()
                .enumerate()
                .map(|(device, name)| serde_json::json!({
                    "name": name,
                    "total_bytes": self.device_capacities.get(device).copied().flatten(),
                    "weights_bytes": footprints
                        .get(device)
                        .map_or(0, |f| f.weights_device_bytes),
                    "layers": self.plan.per_device_layers.get(device).copied().unwrap_or(0),
                    "footprint": footprints.get(device).map(|f| {
                        f.to_json_in(self.device_capacities.get(device).copied().flatten())
                    }),
                }))
                .collect::<Vec<_>>(),
        })
    }

    /// One [`DeviceFootprint`] per device: what the plan placed there, and
    /// what that device's own layers cost in KV.
    fn footprints(
        &self,
        model: &LoadedModel,
        probe: &KvCache,
        slots: usize,
    ) -> Vec<DeviceFootprint> {
        let weights = DeviceFootprint::weights_per_device(model, self.device_names.len());
        (0..self.device_names.len())
            .map(|device| {
                DeviceFootprint::for_split_device(
                    &model.config,
                    probe,
                    self.device_kv_storage.get(device).copied().flatten(),
                    slots,
                    weights.get(device).copied().unwrap_or(0),
                    &self.plan.layer_device,
                    device,
                )
            })
            .collect()
    }
}

/// How the model should be spread: `--device-split` if given, then
/// `ORANGU_DEVICE_SPLIT`, then `[orangu-server].device_split`.
///
/// Same precedence as `requested_device`, and validated the same way — a
/// value that isn't a mode or a ratio list is an error rather than a silent
/// `off`, since silently not splitting is exactly the outcome somebody
/// setting this is trying to avoid.
fn requested_split(flag: Option<&str>, configured: &SplitMode) -> Result<SplitMode> {
    let raw = match flag {
        Some(raw) => Some(raw.to_string()),
        None => std::env::var("ORANGU_DEVICE_SPLIT").ok(),
    };
    match raw {
        Some(raw) => {
            SplitMode::parse(&raw).map_err(|err| anyhow!("device split {raw:?} is invalid: {err}"))
        }
        None => Ok(configured.clone()),
    }
}

/// The device the operator asked for: `ORANGU_DEVICE` if set, otherwise
/// `[orangu-server].device`, otherwise the ranking policy.
///
/// The environment wins so a benchmark sweep can walk a machine's cards
/// without rewriting the config file between runs — the same precedence
/// every other `ORANGU_*` tuning knob has. A malformed value can't be
/// rejected here (see `config`'s own note on the key): "device 2" is only
/// wrong once a driver has been asked what it has.
fn requested_device(flag: Option<&str>, configured: &DeviceRequest) -> DeviceRequest {
    if let Some(raw) = flag {
        return DeviceRequest::parse(raw);
    }
    match std::env::var("ORANGU_DEVICE") {
        Ok(raw) => DeviceRequest::parse(&raw),
        Err(_) => configured.clone(),
    }
}

/// Enumeration under the same short retry [`init_gpu_with_retry`] applies
/// to bring-up.
///
/// A `wgpu` driver that has just had a context torn down by an exiting
/// process can briefly report *no* adapters, not merely fail to create a
/// device on one — and enumeration used to sit inside the retried call
/// (`request_adapter` both chose and created). Splitting selection out
/// ahead of bring-up would have quietly dropped that protection, turning a
/// restart race into "no device was found" on a machine that has one.
///
/// Costs nothing where there is genuinely no driver beyond the wait the
/// previous code already paid there: four fast failures and 2.1s of
/// backoff.
fn devices_with_retry(enumerate: impl Fn() -> Vec<DeviceCandidate>) -> Vec<DeviceCandidate> {
    init_gpu_with_retry(|| {
        let devices = enumerate();
        (!devices.is_empty()).then_some(devices)
    })
    .unwrap_or_default()
}

/// Applies `request` to one backend's enumerated devices and prints the
/// inventory — returning the selected *enumeration indices*, best first.
///
/// Under `auto` that is every hardware device the backend reported, ranked;
/// under an explicit index or name it is exactly one. One device runs the
/// model either way (the head), which is why the inventory says which of
/// the selected ones is idle rather than implying all of them are working.
///
/// Printing happens here, unconditionally, rather than at the call sites:
/// these lines are what make a later measurement attributable to a device,
/// and they are worth nothing if they have to be switched on before the run
/// they describe. They are also the only place a second, idle card on the
/// machine becomes visible.
///
/// Shared by all five backends — the policy is the same whether the device
/// list came from `wgpu`, CUDA, OpenCL or HIP, and each of those knowing
/// only how to *enumerate* is what keeps it that way.
fn choose_device(
    api: &str,
    candidates: &[DeviceCandidate],
    request: &DeviceRequest,
) -> std::result::Result<Vec<usize>, DeviceError> {
    match device::select_all(candidates, request) {
        Ok(selected) => {
            for line in device::inventory(api, candidates, &selected) {
                eprintln!("{line}");
            }
            let head = selected[0];
            if candidates[head].class.is_software() {
                eprintln!(
                    "orangu-server: [{api}] {} is a software rasterizer — this runs the GPU \
                     code path on the CPU, and orangu's own CPU backend is faster. It was \
                     asked for explicitly, so it is being used.",
                    candidates[head].name
                );
            }
            Ok(selected.iter().map(|&p| candidates[p].index).collect())
        }
        // A machine with no driver for this API is the ordinary case under
        // `auto` and says nothing. A machine that has devices and still
        // can't offer one — the software-rasterizer-only case — is
        // surprising, and stays silent forever unless it is said here.
        Err(err) => {
            if !candidates.is_empty() && err.kind == DeviceErrorKind::Absent {
                eprintln!("orangu-server: [{api}] {err}");
            }
            Err(err)
        }
    }
}

/// Picks the `Backend` the forward pass runs on, per `[orangu-server].
/// backend` (`auto`/`cpu`/`vulkan`/`metal`/`cuda`/`opencl`/`rocm`, see
/// `config::BackendPreference`) and `device` (`[orangu-server].device` or
/// `ORANGU_DEVICE`, see `engine::backend::device`), and a label for the
/// startup banner (e.g.
/// `"CPU/AVX2"`, `"Vulkan/AMD Radeon RX 5500M (RADV NAVI14)"` or
/// `"Metal/Apple M1 Pro (Metal)"`). `auto`
/// tries every GPU backend compiled into this build, preferring the most
/// mature one first (`VulkanBackend`, the one with real fused/GPU-
/// resident optimizations — see its module doc), then falls back to the
/// CPU backend if none found one; every other named backend fails loudly
/// instead of falling back, since GPU inference was asked for explicitly.
/// `rocm` additionally fails loudly (a clear "rebuild with `--features
/// rocm`" message, not a panic) when this binary wasn't built with that
/// Cargo feature — see `engine::backend::rocm`'s module doc for why it's
/// the one opt-in backend (`cuda`/`opencl`/`vulkan`/`metal` are always
/// compiled in).
///
/// On Apple targets `auto` tries **Metal first**. Not a preference: macOS
/// ships no Vulkan driver at all, so leading with Vulkan there is four
/// retry rounds of guaranteed failure (2.1s of startup latency) before
/// reaching the API the machine actually has — and `MetalBackend` is the
/// same engine and the same kernels as `VulkanBackend`, so nothing is
/// given up by preferring it. Vulkan stays in the chain behind it for a
/// Mac running MoltenVK.
///
/// On Windows `auto` additionally tries **DX12 behind Vulkan**. It is the
/// same `wgpu` engine and the same WGSL kernels again (`naga` emits HLSL
/// instead of SPIR-V), so it reaches every fused path `engine::arch` asks
/// for through `Backend::as_wgpu` — which is why it goes ahead of the
/// matmul-only CUDA and OpenCL backends rather than after them. Behind
/// Vulkan because that is the API this engine was tuned on. Its value is
/// the machine with a working D3D12 driver and no Vulkan ICD, which until
/// now ran on the CPU without ever saying why.
///
/// `device` is applied to *whichever* backend answers, and a backend can
/// only answer by satisfying it — so a named device is never silently
/// swapped for another one. Under `auto`, a request no backend in the
/// chain could satisfy is an error at the end of the chain rather than a
/// fall-back to the CPU.
fn select_backend(
    preference: BackendPreference,
    device: &DeviceRequest,
) -> Result<(Arc<dyn Backend>, String)> {
    // Metal is an Apple API and `wgpu` compiles its Metal backend only for
    // Apple targets, so `Backends::METAL` matches nothing anywhere else.
    // Elsewhere, `auto` therefore skips it rather than paying an adapter
    // request that cannot succeed, and an explicit `backend = metal` says
    // *that* rather than "no device found" after 2.1s of retry backoff for
    // a device that was never going to appear.
    const HAS_METAL: bool = cfg!(target_vendor = "apple");
    // The same argument for Direct3D 12, which `wgpu` compiles only for
    // Windows.
    const HAS_DX12: bool = cfg!(windows);
    const VULKAN: wgpu::Backends = wgpu::Backends::VULKAN;
    const DX12: wgpu::Backends = wgpu::Backends::DX12;

    let cpu = || -> (Arc<dyn Backend>, String) {
        let label = if is_x86_feature_detected() {
            "CPU/AVX2"
        } else {
            "CPU"
        };
        (Arc::new(CpuBackend), label.to_string())
    };
    // Turns a device-selection failure into the message an explicitly-named
    // backend fails with. The `DeviceError` already carries the whole device
    // list, which is the part that makes the message actionable.
    let named = |api: &str, err: DeviceError| {
        anyhow!(
            "[{}].backend = {api}, but {err}",
            config::SERVER_SECTION,
            err = err
        )
    };
    // Bring-up failed *after* a device was successfully chosen: the driver
    // or pipeline creation refused, which is a different problem from not
    // finding a device and must not be reported as one.
    let unusable = |api: &str, index: usize| {
        anyhow!(
            "[{}].backend = {api}, but device {index} could not be brought up \
             (device or pipeline creation failed)",
            config::SERVER_SECTION
        )
    };

    match preference {
        BackendPreference::Cpu => Ok(cpu()),
        BackendPreference::Vulkan => {
            let selected = choose_device(
                "vulkan",
                &devices_with_retry(|| VulkanBackend::devices(VULKAN)),
                device,
            )
            .map_err(|err| named("vulkan", err))?;
            let backend =
                init_gpu_with_retry(|| VulkanBackend::try_init_selected(VULKAN, &selected))
                    .ok_or_else(|| unusable("vulkan", selected[0]))?;
            let label = format!("Vulkan/{}", backend.adapter_name);
            Ok((Arc::new(backend), label))
        }
        BackendPreference::Metal if !HAS_METAL => Err(anyhow!(
            "[{}].backend = metal, but Metal is an Apple API and this build is not \
             running on macOS",
            config::SERVER_SECTION
        )),
        BackendPreference::Metal => {
            let selected =
                choose_device("metal", &devices_with_retry(MetalBackend::devices), device)
                    .map_err(|err| named("metal", err))?;
            let backend = init_gpu_with_retry(|| MetalBackend::try_init_selected(&selected))
                .ok_or_else(|| unusable("metal", selected[0]))?;
            let label = format!("Metal/{}", backend.device_name());
            Ok((Arc::new(backend), label))
        }
        BackendPreference::Dx12 if !HAS_DX12 => Err(anyhow!(
            "[{}].backend = dx12, but Direct3D 12 is a Windows API and this build is \
             not running on Windows",
            config::SERVER_SECTION
        )),
        BackendPreference::Dx12 => {
            let selected = choose_device(
                "dx12",
                &devices_with_retry(|| VulkanBackend::devices(DX12)),
                device,
            )
            .map_err(|err| named("dx12", err))?;
            let backend = init_gpu_with_retry(|| VulkanBackend::try_init_selected(DX12, &selected))
                .ok_or_else(|| unusable("dx12", selected[0]))?;
            let label = format!("DX12/{}", backend.adapter_name);
            Ok((Arc::new(backend), label))
        }
        BackendPreference::Cuda => {
            let index = choose_device("cuda", &CudaBackend::devices(), device)
                .map_err(|err| named("cuda", err))?[0];
            let backend =
                CudaBackend::try_init_index(index).ok_or_else(|| unusable("cuda", index))?;
            let label = format!("CUDA/{}", backend.device_name);
            Ok((Arc::new(backend), label))
        }
        BackendPreference::OpenCl => {
            use engine::backend::OpenClBackend;
            let index = choose_device("opencl", &OpenClBackend::devices(), device)
                .map_err(|err| named("opencl", err))?[0];
            let backend =
                OpenClBackend::try_init_index(index).ok_or_else(|| unusable("opencl", index))?;
            let label = format!("OpenCL/{}", backend.device_name);
            Ok((Arc::new(backend), label))
        }
        BackendPreference::Rocm => {
            #[cfg(feature = "rocm")]
            {
                use engine::backend::RocmBackend;
                let index = choose_device("rocm", &RocmBackend::devices(), device)
                    .map_err(|err| named("rocm", err))?[0];
                let backend =
                    RocmBackend::try_init_index(index).ok_or_else(|| unusable("rocm", index))?;
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
            // An explicitly requested device that an API in this chain
            // *has* devices for but doesn't offer. Remembered rather than
            // raised on the spot: the next API along may be the one that
            // has it, and only the end of the chain knows that none did.
            //
            // What this must never become is a silent fall-through. A
            // backend can only succeed below by satisfying `device`, so
            // reaching the CPU fallback with a request outstanding means
            // the named device was nowhere — which is an error, not a
            // reason to quietly measure something else.
            let mut rejected: Option<DeviceError> = None;
            let mut remember = |err: DeviceError| {
                if err.kind == DeviceErrorKind::Rejected && rejected.is_none() {
                    rejected = Some(err);
                }
            };

            // Ahead of Vulkan, and only where it can succeed at all — see
            // this function's doc comment for both halves of that.
            if HAS_METAL {
                match choose_device("metal", &devices_with_retry(MetalBackend::devices), device) {
                    Ok(selected) => {
                        if let Some(backend) =
                            init_gpu_with_retry(|| MetalBackend::try_init_selected(&selected))
                        {
                            let label = format!("Metal/{}", backend.device_name());
                            return Ok((Arc::new(backend), label));
                        }
                    }
                    Err(err) => remember(err),
                }
            }
            match choose_device(
                "vulkan",
                &devices_with_retry(|| VulkanBackend::devices(VULKAN)),
                device,
            ) {
                Ok(selected) => {
                    if let Some(backend) =
                        init_gpu_with_retry(|| VulkanBackend::try_init_selected(VULKAN, &selected))
                    {
                        let label = format!("Vulkan/{}", backend.adapter_name);
                        return Ok((Arc::new(backend), label));
                    }
                }
                Err(err) => remember(err),
            }
            // Behind Vulkan, and only on Windows — see this function's doc
            // comment. Before CUDA/OpenCL because it reaches the same fused
            // `wgpu` engine those two don't have.
            if HAS_DX12 {
                match choose_device(
                    "dx12",
                    &devices_with_retry(|| VulkanBackend::devices(DX12)),
                    device,
                ) {
                    Ok(selected) => {
                        if let Some(backend) = init_gpu_with_retry(|| {
                            VulkanBackend::try_init_selected(DX12, &selected)
                        }) {
                            let label = format!("DX12/{}", backend.adapter_name);
                            return Ok((Arc::new(backend), label));
                        }
                    }
                    Err(err) => remember(err),
                }
            }
            match choose_device("cuda", &CudaBackend::devices(), device) {
                Ok(selected) => {
                    if let Some(backend) = CudaBackend::try_init_index(selected[0]) {
                        let label = format!("CUDA/{}", backend.device_name);
                        return Ok((Arc::new(backend), label));
                    }
                }
                Err(err) => remember(err),
            }
            {
                use engine::backend::OpenClBackend;
                match choose_device("opencl", &OpenClBackend::devices(), device) {
                    Ok(selected) => {
                        if let Some(backend) = OpenClBackend::try_init_index(selected[0]) {
                            let label = format!("OpenCL/{}", backend.device_name);
                            return Ok((Arc::new(backend), label));
                        }
                    }
                    Err(err) => remember(err),
                }
            }
            #[cfg(feature = "rocm")]
            {
                use engine::backend::RocmBackend;
                match choose_device("rocm", &RocmBackend::devices(), device) {
                    Ok(selected) => {
                        if let Some(backend) = RocmBackend::try_init_index(selected[0]) {
                            let label = format!("ROCm/{}", backend.device_name);
                            return Ok((Arc::new(backend), label));
                        }
                    }
                    Err(err) => remember(err),
                }
            }
            if let Some(err) = rejected {
                // The qualifier goes in front: `DeviceError`'s message ends
                // with the multi-line device listing, and anything appended
                // after it reads as part of the last device's line.
                return Err(anyhow!(
                    "no backend on this machine could satisfy device = {device}: {err}",
                    device = device,
                    err = err
                ));
            }
            if !device.is_auto() {
                return Err(anyhow!(
                    "device = {device}, but no GPU backend on this machine reported any \
                     device to match it against",
                ));
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
    use super::{ListSort, dominant_tensor_types, list_order};
    use crate::engine::quant::{GGML_TYPE_BF16, GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K};
    use clap::Parser;
    use orangu::model_spec::ModelGroup;
    use std::path::PathBuf;

    fn listed_group(label: &str, size_bytes: u64) -> ModelGroup {
        ModelGroup {
            label: label.to_string(),
            size_bytes,
            quantization: None,
            errors: Vec::new(),
            representative_path: PathBuf::from(format!("{label}.gguf")),
            paths: Vec::new(),
            hf_repo: None,
            local_commit: None,
        }
    }

    #[test]
    fn list_sort_orders_descending_without_changing_canonical_indices() {
        let groups = [
            listed_group("a", 20),
            listed_group("b", 30),
            listed_group("c", 10),
        ];
        let last_used = [Some(10), None, Some(20)];

        assert_eq!(
            list_order(&groups, &last_used, Some(ListSort::Size)),
            vec![1, 0, 2]
        );
        assert_eq!(
            list_order(&groups, &last_used, Some(ListSort::LastUsed)),
            vec![2, 0, 1]
        );
        assert_eq!(list_order(&groups, &last_used, None), vec![0, 1, 2]);
    }

    #[test]
    fn clap_accepts_both_list_sort_fields() {
        for (value, expected) in [("size", ListSort::Size), ("last-used", ListSort::LastUsed)] {
            let args = Args::try_parse_from(["orangu-server", "list", "--sort", value]).unwrap();
            assert!(matches!(
                args.command,
                Some(Command::List { sort: Some(actual) }) if actual == expected
            ));
        }
    }

    #[test]
    fn clap_parses_refresh_all_as_a_refresh_option() {
        let args = Args::try_parse_from(["orangu-server", "refresh", "--all"]).unwrap();
        assert!(matches!(
            args.command,
            Some(Command::Refresh {
                model: None,
                all: true,
                yes: false,
            })
        ));
    }

    #[test]
    fn clap_rejects_refresh_all_with_a_model() {
        assert!(Args::try_parse_from(["orangu-server", "refresh", "1", "--all"]).is_err());
    }

    /// On a quantized file the floats are norms and biases and must not
    /// outnumber the type the weight bytes are in.
    #[test]
    fn floats_are_excluded_while_the_file_has_quantized_tensors() {
        let tensors = vec![
            ("blk.0.attn_norm.weight", GGML_TYPE_F32),
            ("blk.1.attn_norm.weight", GGML_TYPE_F32),
            ("blk.2.attn_norm.weight", GGML_TYPE_F32),
            ("blk.0.ffn_down.weight", GGML_TYPE_Q4_K),
            ("blk.1.ffn_down.weight", GGML_TYPE_Q4_K),
            ("output.weight", GGML_TYPE_Q6_K),
        ];
        assert_eq!(
            dominant_tensor_types(tensors.into_iter()),
            vec![GGML_TYPE_Q4_K, GGML_TYPE_Q6_K],
            "the three F32 norms must not lead the line"
        );
    }

    /// The reported bug: a `BF16` file has no quantized tensor at all, so
    /// excluding floats unconditionally emptied the set and the banner read
    /// `Kernels none` on a model the backend decodes perfectly well.
    #[test]
    fn a_float_only_model_reports_its_float_type_rather_than_nothing() {
        let tensors = vec![
            ("blk.0.attn_norm.weight", GGML_TYPE_F32),
            ("blk.0.ffn_down.weight", GGML_TYPE_BF16),
            ("blk.1.ffn_down.weight", GGML_TYPE_BF16),
            ("token_embd.weight", GGML_TYPE_BF16),
        ];
        assert_eq!(
            dominant_tensor_types(tensors.into_iter()),
            vec![GGML_TYPE_BF16, GGML_TYPE_F32],
            "a BF16 model decodes through BF16 kernels, not through none"
        );
    }

    /// An empty file is still empty — the fallback adds a case, it does not
    /// invent a type.
    #[test]
    fn no_tensors_means_no_types() {
        assert!(dominant_tensor_types(std::iter::empty()).is_empty());
    }

    /// Ties break on type id so the banner is byte-identical across runs.
    #[test]
    fn equal_counts_order_by_type_id() {
        let tensors = vec![("a", GGML_TYPE_Q6_K), ("b", GGML_TYPE_Q4_K)];
        let out = dominant_tensor_types(tensors.into_iter());
        assert_eq!(
            out,
            vec![
                GGML_TYPE_Q4_K.min(GGML_TYPE_Q6_K),
                GGML_TYPE_Q4_K.max(GGML_TYPE_Q6_K)
            ]
        );
    }

    use super::{
        Args, Command, DeviceClass, SplitReport, gate, label_carries_tag, resolve_model_spec,
        resolve_workspace, terminal_title,
    };
    use crate::engine::placement::SplitPlan;

    /// A deployment-gate row is a value and nothing else — the banner is a
    /// table, and a cell that sometimes grows a sentence of advice is what
    /// stops it being one. The advice lives in the manual.
    #[test]
    fn a_deployment_gate_row_is_only_yes_or_no() {
        assert_eq!(gate(true), "Yes");
        assert_eq!(gate(false), "No");
    }

    fn report(layers: &[usize], classes: &[DeviceClass]) -> SplitReport {
        SplitReport {
            api: "vulkan".to_string(),
            plan: SplitPlan {
                per_device_layers: layers.to_vec(),
                layer_device: layers
                    .iter()
                    .enumerate()
                    .flat_map(|(device, n)| std::iter::repeat_n(device, *n))
                    .collect(),
            },
            device_names: (0..classes.len()).map(|i| format!("dev{i}")).collect(),
            device_capacities: vec![None; classes.len()],
            device_classes: classes.to_vec(),
            device_kv_storage: vec![None; classes.len()],
        }
    }

    /// The trap the note exists for: shares are proportional to reported
    /// memory, an integrated GPU reports system RAM, and so the largest share
    /// lands on the slowest device while the layer ranges look unremarkable.
    #[test]
    fn the_placement_note_fires_when_the_integrated_device_took_the_most_layers() {
        let note = report(&[4, 20], &[DeviceClass::Discrete, DeviceClass::Integrated])
            .placement_note()
            .expect("integrated device holds the most layers");
        assert!(note.contains("integrated"), "{note}");
        assert!(
            note.contains("dev1"),
            "names the device that took them: {note}"
        );
        assert!(
            note.contains("dev0"),
            "names the discrete alternative: {note}"
        );
    }

    /// Silent when the split already favours the discrete card — the note is
    /// advice, and advice that fires when nothing is wrong is noise that
    /// teaches operators to ignore the line.
    #[test]
    fn the_placement_note_is_silent_when_the_split_already_favours_the_discrete_card() {
        assert!(
            report(&[20, 4], &[DeviceClass::Discrete, DeviceClass::Integrated])
                .placement_note()
                .is_none()
        );
    }

    /// Silent with no discrete device to move work to: on a machine whose
    /// only GPU is integrated, "prefer the discrete one" is not advice.
    #[test]
    fn the_placement_note_is_silent_without_a_discrete_alternative() {
        assert!(
            report(
                &[12, 12],
                &[DeviceClass::Integrated, DeviceClass::Integrated]
            )
            .placement_note()
            .is_none()
        );
    }

    /// Every subcommand clap parses has a `mode()` name for the terminal
    /// title, spelled exactly the way the user typed it — the title is only
    /// useful if `orangu-server download` says `download`.
    #[test]
    fn terminal_title_names_every_subcommand() {
        use clap::CommandFactory;

        let modes = [
            Command::System.mode(),
            Command::Suggest.mode(),
            Command::List { sort: None }.mode(),
            Command::Plan {
                file: None,
                deep: false,
            }
            .mode(),
            Command::Show {
                file: None,
                full: false,
                tensors: false,
            }
            .mode(),
            Command::Download {
                repo: String::new(),
                yes: false,
            }
            .mode(),
            Command::Delete {
                model: None,
                yes: false,
            }
            .mode(),
            Command::Refresh {
                model: None,
                all: false,
                yes: false,
            }
            .mode(),
            Command::Bundle {
                model: None,
                output: None,
                binary: None,
                yes: false,
                roles: Default::default(),
                listen: Default::default(),
            }
            .mode(),
            Command::Prune {
                identifier: None,
                yes: false,
            }
            .mode(),
        ];
        let parsed: Vec<String> = Args::command()
            .get_subcommands()
            .map(|sub| sub.get_name().to_string())
            .collect();

        assert_eq!(
            modes.len(),
            parsed.len(),
            "mode() covers {modes:?}, clap parses {parsed:?}"
        );
        for name in &parsed {
            assert!(
                modes.contains(&name.as_str()),
                "subcommand {name} has no matching mode(): {modes:?}"
            );
        }
    }

    /// The subcommand names a completion script actually *offers*, as
    /// opposed to merely mentions.
    ///
    /// The distinction is the whole point of the test below. A description
    /// string can contain a subcommand's name — `--deep`'s reads "Also
    /// verify plan's shards and architecture" — so a substring search over
    /// the script passes while Tab still offers nothing, which is precisely
    /// the bug being guarded against. Each of these parses the one
    /// construct that puts a word in front of the user.
    fn offered_subcommands(shell: &str, script: &str) -> Vec<String> {
        match shell {
            // One `compgen -W "$(_orangu_server_models) <names...>"` list.
            "bash" => {
                let marker = "_orangu_server_models) ";
                let start = script.find(marker).expect("bash subcommand list") + marker.len();
                let rest = &script[start..];
                let end = rest.find('"').expect("unterminated bash word list");
                rest[..end].split_whitespace().map(String::from).collect()
            }
            // `_values` entries, each `'<name>[description]'`.
            "zsh" => script
                .lines()
                .filter_map(|line| {
                    let line = line.trim_start().strip_prefix('\'')?;
                    Some(line[..line.find('[')?].to_string())
                })
                .collect(),
            // `complete ... '__fish_use_subcommand' -a <name> -d '...'`.
            "fish" => script
                .lines()
                .filter(|line| line.contains("'__fish_use_subcommand'"))
                .filter_map(|line| {
                    let after = line.split(" -a ").nth(1)?;
                    Some(after.split_whitespace().next()?.to_string())
                })
                .collect(),
            other => panic!("no parser for {other}"),
        }
    }

    /// Every subcommand clap parses is offered by all three completion
    /// scripts.
    ///
    /// Written because `plan` was not: it was added to clap and to the
    /// terminal title, and `shell.rs` — three hand-written scripts with no
    /// generator behind them — was never updated, so the one command whose
    /// entire purpose is answering a question *before* you commit to a
    /// download was the one command Tab would not offer. Nothing else in the
    /// tree connects clap to those scripts, and a shell script has no
    /// compiler to notice.
    #[test]
    fn every_subcommand_is_offered_by_every_completion_script() {
        use clap::CommandFactory;

        let parsed: Vec<String> = Args::command()
            .get_subcommands()
            .map(|sub| sub.get_name().to_string())
            .collect();

        for (shell, script) in [
            ("bash", crate::shell::BASH),
            ("zsh", crate::shell::ZSH),
            ("fish", crate::shell::FISH),
        ] {
            let offered = offered_subcommands(shell, script);
            for name in &parsed {
                assert!(
                    offered.contains(name),
                    "the {shell} completion script does not offer `{name}` — it offers {offered:?}"
                );
            }
        }
    }

    /// The parsers above must be able to fail, or the test they serve is
    /// decoration. Each is handed its own script with one subcommand's
    /// *offer* removed while every mention of the name stays put, which is
    /// exactly the shape the real bug had.
    #[test]
    fn the_completion_parsers_notice_a_missing_offer() {
        for (shell, script, remove) in [
            ("bash", crate::shell::BASH, "show plan download"),
            (
                "zsh",
                crate::shell::ZSH,
                "'plan[Report what a model needs to run here, without loading it]' \\",
            ),
            (
                "fish",
                crate::shell::FISH,
                "complete -c orangu-server -n '__fish_use_subcommand' -a plan     -d 'Report what a model needs to run here, without loading it'",
            ),
        ] {
            assert!(
                script.contains(remove),
                "{shell}: the text this test removes is no longer in the script"
            );
            let replacement = if shell == "bash" { "show download" } else { "" };
            let crippled = script.replace(remove, replacement);
            // The name survives elsewhere — in `--deep`'s description, and
            // in the argument-completion lines — so a substring search would
            // still pass here. The parser must not.
            assert!(
                crippled.contains("plan"),
                "{shell}: nothing left to fool a substring search"
            );
            assert!(
                !offered_subcommands(shell, &crippled).contains(&"plan".to_string()),
                "{shell}: the parser still reports `plan` as offered after its offer was removed"
            );
        }
    }

    #[test]
    fn the_title_is_the_binary_then_the_mode() {
        assert_eq!(
            terminal_title(
                Command::Download {
                    repo: String::new(),
                    yes: false,
                }
                .mode()
            ),
            "orangu-server download"
        );
        assert_eq!(terminal_title("prune"), "orangu-server prune");
    }

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

    /// The banner appends the resolved quantization to a bare label, but a
    /// label that already names a tag keeps exactly the spelling it was
    /// started with.
    #[test]
    fn only_a_bare_label_gets_the_quantization_appended() {
        assert!(!label_carries_tag("unsloth/gemma-4-E2B-it-GGUF"));
        assert!(!label_carries_tag("gemma-4-E2B-it-Q4_K_M.gguf"));
        assert!(label_carries_tag("unsloth/gemma-4-E2B-it-GGUF:Q4_K_M"));
        // A `:` above the file itself is part of a directory name, not a tag.
        assert!(!label_carries_tag("/mnt/models:old/gemma-4-E2B-it.gguf"));
    }

    /// `orangu-server 84` names a *position* in `list`'s output, not a model.
    /// The label resolution keeps has to be the model's own `MODEL` name, or
    /// the number is what the startup banner, `/v1/models`, every response's
    /// `model` field and the web console header all report as the model.
    #[test]
    fn an_nr_spec_is_labelled_with_the_models_own_name() {
        let dir = tempfile::tempdir().expect("models directory");
        write_minimal_gguf(&dir.path().join("Llama-3.2-3B-Instruct-Q4_K_M.gguf"));

        let (source, label) = resolve_model_spec(dir.path(), "1", None).expect("resolved");

        assert_eq!(label, "Llama-3.2-3B-Instruct-Q4_K_M");
        assert!(
            source.path().ends_with("Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
            "{:?}",
            source.path()
        );
    }

    /// A `MODEL` name is already the id it resolves to, so it survives
    /// unchanged — the fix for the `NR` case must not rewrite what was
    /// spelled correctly to begin with.
    #[test]
    fn a_model_name_spec_is_kept_as_written() {
        let dir = tempfile::tempdir().expect("models directory");
        write_minimal_gguf(&dir.path().join("Llama-3.2-3B-Instruct-Q4_K_M.gguf"));

        let (_, label) =
            resolve_model_spec(dir.path(), "Llama-3.2-3B-Instruct-Q4_K_M", None).expect("resolved");

        assert_eq!(label, "Llama-3.2-3B-Instruct-Q4_K_M");
    }

    /// Writes a minimal GGUF — enough header for the models-directory scan
    /// behind `resolve_model_spec` to count it as a model.
    fn write_minimal_gguf(path: &std::path::Path) {
        use std::io::Write;

        let architecture = "llama";
        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&1u64.to_le_bytes()); // metadata_kv_count
        let key = "general.architecture";
        buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
        buf.extend_from_slice(key.as_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes()); // STRING
        buf.extend_from_slice(&(architecture.len() as u64).to_le_bytes());
        buf.extend_from_slice(architecture.as_bytes());
        std::fs::File::create(path)
            .expect("create gguf")
            .write_all(&buf)
            .expect("write gguf");
    }
}
