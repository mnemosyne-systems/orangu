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

//! `orangu-gguf` is a small standalone inventory tool for local LLM
//! inference: it reports the CPU/GPU hardware available to run a model
//! (`system`), and reads GGUF model files directly off disk — no llama.cpp
//! server required — to list what's under a models directory (`list`) or
//! dump one file's full metadata (`show`).

mod config;
mod download;
mod gguf;
mod init;
mod models;
mod prompt;
mod roles;
mod shell;
mod suggest;
mod system;

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use config::{default_gguf_config_path, load_gguf_configuration};
use gguf::{GgufFile, ggml_type_name};
use std::{path::PathBuf, process::ExitCode};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Metadata arrays longer than this print a truncated preview instead of
/// every element — `tokenizer.ggml.tokens` routinely holds 100,000+ entries.
/// Pass `--full` to disable the cap.
const DEFAULT_ARRAY_PREVIEW: usize = 8;

#[derive(Parser, Debug)]
#[command(
    version = VERSION,
    about = "Inspect machine CPU/GPU hardware and local GGUF model files",
    long_about = "Inspect machine CPU/GPU hardware and local GGUF model files.\n\nRun with no subcommand for an interactive wizard: pick a role (all/code/review/explorer/embeddings) and a model, and get a llama-server command line tuned for that combination."
)]
struct Args {
    /// Path to orangu-gguf.conf. Defaults to ./orangu-gguf.conf, then
    /// ~/.orangu/orangu-gguf.conf. Needed by `list`/`show`, and by the
    /// interactive role wizard run with no subcommand.
    #[arg(short, long)]
    config: Option<PathBuf>,
    /// Interactively create ~/.orangu/orangu-gguf.conf and exit.
    #[arg(short, long)]
    init: bool,
    /// Print the shell completion script for the detected shell and exit.
    ///
    /// Detects the current shell from $SHELL. Pipe into your shell's eval or
    /// drop the output into the appropriate completions directory:
    ///
    ///   bash: eval "$(orangu-gguf -s)"
    ///   zsh:  orangu-gguf -s > ~/.zsh/completions/_orangu-gguf
    ///   fish: orangu-gguf -s > ~/.config/fish/completions/orangu-gguf.fish
    #[arg(short = 's', long = "shell-completions")]
    shell_completions: bool,
    #[command(subcommand)]
    command: Option<Commands>,
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
             \x20 bash: eval \"$(orangu-gguf -s)\"\n\
             \x20 zsh:  orangu-gguf -s > ~/.zsh/completions/_orangu-gguf\n\
             \x20 fish: orangu-gguf -s > ~/.config/fish/completions/orangu-gguf.fish"
        ));
    };
    print!("{script}");
    Ok(())
}

#[derive(Subcommand, Debug)]
enum Commands {
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
        /// a MODEL name from its second.
        file: String,
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
}

fn main() -> ExitCode {
    let args = Args::parse();

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

    let Some(command) = args.command else {
        return match load_config(args.config).and_then(|conf| roles::run_wizard(&conf)) {
            Ok(()) => ExitCode::SUCCESS,
            Err(err) => {
                eprintln!("error: {err:#}");
                ExitCode::FAILURE
            }
        };
    };

    match run(args.config, command) {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(config_arg: Option<PathBuf>, command: Commands) -> Result<()> {
    match command {
        Commands::System => {
            let cpu = system::detect_cpu();
            let gpus = system::detect_gpus(cpu.total_memory_bytes);
            print!("{}", system::format_report(&cpu, &gpus));
            Ok(())
        }
        Commands::Suggest => {
            let cpu = system::detect_cpu();
            let gpus = system::detect_gpus(cpu.total_memory_bytes);
            print!("{}", suggest::format_suggestion(&cpu, &gpus));
            Ok(())
        }
        Commands::List => {
            let conf = load_config(config_arg)?;
            let models = models::scan_models_dir(&conf.models)?;
            print!("{}", models::format_list(&models, &conf.models));
            Ok(())
        }
        Commands::Show {
            file,
            full,
            tensors,
        } => {
            let conf = load_config(config_arg)?;
            let path = models::resolve_show_target(&conf.models, &file)?;
            let gguf = GgufFile::open(&path)?;
            print!("{}", format_show(&gguf, full, tensors));
            Ok(())
        }
        Commands::Download { repo } => {
            let conf = load_config(config_arg)?;
            download::run_download(&conf, &repo)
        }
    }
}

fn load_config(explicit: Option<PathBuf>) -> Result<config::GgufConfiguration> {
    let path = explicit.or_else(default_gguf_config_path).ok_or_else(|| {
        anyhow!(
            "Missing config file; pass --config or add ./orangu-gguf.conf or ~/.orangu/orangu-gguf.conf (see --init)"
        )
    })?;
    load_gguf_configuration(&path).with_context(|| format!("loading {}", path.display()))
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

/// Formats a byte count as a human-readable size (e.g. `4.92 GiB`), shared
/// by the `system` (RAM/VRAM) and `list` (file size) output.
pub(crate) fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.2} {}", UNITS[unit])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_byte_sizes() {
        assert_eq!(format_bytes(0), "0 B");
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(1024), "1.00 KiB");
        assert_eq!(format_bytes(1024 * 1024 * 5), "5.00 MiB");
        assert_eq!(format_bytes(4_929_003_520), "4.59 GiB");
    }
}
