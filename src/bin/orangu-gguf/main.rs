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

//! `orangu-gguf` — builds a model file.
//!
//! Two jobs, one binary, because they are two ends of the same pipeline:
//!
//! - Given a **manifest** of permissively-licensed repositories, train a
//!   model from random weights and write it as a GGUF file. Clone, train a
//!   tokenizer, pack the corpus, pretrain, export — every stage on disk in
//!   a work directory, so a run that stops picks up where it left off.
//! - Given an existing **model**, write it out at a different weight
//!   format: a K-quant or one of the two wide I-quants, from the
//!   full-precision file.
//!
//! What comes out is an ordinary GGUF file that `orangu-server` serves
//! natively, with no conversion step in between and nothing to install
//! beside this binary.
//!
//! The training is honest but it is not small. Everything runs on the CPU
//! in `f32`; the sizes and what each one costs are in the manual. `smoke`
//! finishes in minutes and exists to prove the pipeline end to end, and the
//! sizes above it are measured in days.

use anyhow::{Context, Result, bail};
use clap::Parser;
use orangu::gguf::GgufValue;
use orangu::profiling::profile;
use orangu::shell_completions;
use rayon::prelude::*;
use std::{
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

mod aligned;
mod auto;
mod choose;
mod corpus;
mod gpu;
mod hf;
mod imatrix;
mod inspect;
mod jobs;
mod kl;
mod manifest;
mod model;
mod model4;
mod node;
mod optimize;
mod pack;
mod quant;
mod shards;
mod shell;
mod speed;
mod spread;
mod stages;
mod train;
mod vocab;
mod wikipedia;
mod write;

use manifest::Manifest;
use model::{Config, Model};
use quant::Ftype;

/// The architecture written into every model this tool trains.
///
/// The help text, with the two-letter short options in the column they
/// belong in. `{about-with-newline}`, `{usage-heading}`, `{usage}` and
/// `{positionals}` are filled in by the parser; the options are not.
const HELP: &str = "\
{about-with-newline}
{usage-heading} {usage}

Arguments:
{positionals}

Options:
  -m,  --model <PATH>          Convert this GGUF file, checkpoint directory or HF repo instead of training
  -ts, --training-size <SIZE>  Training size for this run, overriding the manifest
  -q,  --quantization <QUANT>  Weight format written, overriding the manifest
  -cs, --context-size <N>      Context length the model declares, overriding the manifest
  -o,  --output <FILE>         Where the model is written, overriding the manifest
       --imatrix <FILE>        Fit the quantization to this importance matrix file or HF repo
       --calibration <FILE>    Measure an importance matrix over this text and fit the quantization to it
       --imatrix-out <FILE>    With --calibration, write the measured importance matrix here and stop
       --list-quantizations    List the weight formats a manifest's quantization accepts
       --max-size <SIZE>       With -q auto, the most bytes of weights the file may take
       --max-kl <NATS>         With -q auto, the most KL from the source over the calibration text
       --tensor-type <PATTERN=TYPE>
                               Write tensors whose name matches PATTERN as TYPE (repeatable)
       --dump <FILE>           Print a model file's metadata and tensor table
       --hash <FILE>           Print a content hash of every tensor in a model file
       --compare <A> <B>       Compare two model files' metadata and tensors
       --split-max-size <SIZE> Write the model as shards of at most SIZE each
       --split-max-tensors <N> Write the model as shards of at most N tensors each
       --keep-split            Write a shard set as the source set is cut
       --merge <FILE>          Join a shard set, named by its first shard, into one file
       --set <KEY=TYPE:VALUE>  Set a metadata key, in place when no format is given (repeatable)
       --delete <KEY>          Delete a metadata key, in place when no format is given (repeatable)
       --override-kv <KEY=TYPE:VALUE>
                               Write this metadata key into the output (repeatable)
       --pure                  Quantize every tensor that may be to the format's base type
       --allow-requantize      Accept a quantized source, rounding its tensors a second time
       --worker                Join the [workers] channel as a node that takes a share of building a model
       --channel               List the nodes of the [workers] channel, servers and gguf nodes
       --config <FILE>         Read [workers] from this file instead of orangu-gguf.conf
       --perplexity <TEXT>     Print the perplexity of the -m model over this text
       --flamegraph <PATH>     Record a CPU flamegraph of the run and render it here
       --flamegraph-freq <HZ>  Sampling frequency in Hz for --flamegraph [default: 999]
       --flamegraph-call-graph <MODE>
                               Call-graph mode for --flamegraph: fp or dwarf [default: fp]
       --flamegraph-png        Also render a PNG beside the flamegraph SVG
  -s,  --shell-completions     Print shell completion script for the detected shell and exit
  -h,  --help                  Print help
  -V,  --version               Print version
";

#[derive(Parser, Debug)]
#[command(
    name = "orangu-gguf",
    version = orangu::build_info::id(),
    about = "Build a model from a manifest, or convert one to another weight format",
    // Nothing sensible happens with no arguments — there is no manifest to
    // read and no model to convert — so show what the arguments are rather
    // than a one-line complaint about their absence.
    arg_required_else_help = true,
    // The options block is written out rather than generated, for one
    // reason: `-ts` and `-cs` are two letters behind a single dash, and an
    // argument parser's flags column has room for one. They are what this
    // tool is documented to take, so they are what its help shows.
    // `every_option_appears_in_the_help` is what keeps this block honest.
    help_template = HELP
)]
struct Args {
    /// JSON manifest describing the model and the repositories to train on.
    #[arg(value_name = "MANIFEST")]
    manifest: Option<PathBuf>,

    /// Convert this GGUF file, checkpoint directory or HF repo instead of training.
    #[arg(short = 'm', long, value_name = "PATH")]
    model: Option<PathBuf>,

    /// Training size for this run, overriding the manifest.
    #[arg(long = "training-size", value_name = "SIZE")]
    training_size: Option<String>,

    /// Weight format written, overriding the manifest.
    #[arg(short = 'q', long, value_name = "QUANT")]
    quantization: Option<String>,

    /// Context length the model declares, overriding the manifest.
    #[arg(long = "context-size", value_name = "N")]
    context_size: Option<String>,

    /// Where the model is written, overriding the manifest.
    #[arg(short = 'o', long, value_name = "FILE")]
    output: Option<PathBuf>,

    /// Fit the quantization to this importance matrix file or HF repo.
    #[arg(long, value_name = "FILE")]
    imatrix: Option<PathBuf>,

    /// Measure an importance matrix over this text and fit the quantization to it.
    #[arg(long, value_name = "FILE")]
    calibration: Option<PathBuf>,

    /// With --calibration, write the measured importance matrix here and stop.
    #[arg(long = "imatrix-out", value_name = "FILE", requires = "calibration")]
    imatrix_out: Option<PathBuf>,

    /// List the weight formats a manifest's `quantization` accepts and exit.
    #[arg(long = "list-quantizations")]
    list_quantizations: bool,

    /// With -q auto, the most bytes of weights the file may take.
    #[arg(long = "max-size", value_name = "SIZE")]
    max_size: Option<String>,

    /// With -q auto, the most KL from the source over the calibration text.
    #[arg(long = "max-kl", value_name = "NATS")]
    max_kl: Option<f64>,

    /// Write tensors whose name matches PATTERN as TYPE (repeatable).
    #[arg(long = "tensor-type", value_name = "PATTERN=TYPE")]
    tensor_type: Vec<String>,

    /// Print a model file's metadata and tensor table.
    #[arg(long, value_name = "FILE")]
    dump: Option<PathBuf>,

    /// Print a content hash of every tensor in a model file.
    #[arg(long, value_name = "FILE")]
    hash: Option<PathBuf>,

    /// Compare two model files' metadata and tensors.
    #[arg(long, num_args = 2, value_names = ["A", "B"])]
    compare: Option<Vec<PathBuf>>,

    /// Write the model as shards of at most SIZE each.
    #[arg(long = "split-max-size", value_name = "SIZE")]
    split_max_size: Option<String>,

    /// Write the model as shards of at most N tensors each.
    #[arg(long = "split-max-tensors", value_name = "N")]
    split_max_tensors: Option<usize>,

    /// Write a shard set as the source set is cut.
    #[arg(long = "keep-split")]
    keep_split: bool,

    /// Join a shard set, named by its first shard, into one file.
    #[arg(long, value_name = "FILE")]
    merge: Option<PathBuf>,

    /// Set a metadata key, in place when no format is given (repeatable).
    #[arg(long, value_name = "KEY=TYPE:VALUE")]
    set: Vec<String>,

    /// Delete a metadata key, in place when no format is given (repeatable).
    #[arg(long, value_name = "KEY")]
    delete: Vec<String>,

    /// Write this metadata key into the output (repeatable).
    #[arg(long = "override-kv", value_name = "KEY=TYPE:VALUE")]
    override_kv: Vec<String>,

    /// Quantize every tensor that may be to the format's base type.
    #[arg(long)]
    pure: bool,

    /// Accept a quantized source, rounding its tensors a second time.
    #[arg(long = "allow-requantize")]
    allow_requantize: bool,

    /// Join the [workers] channel as a node that takes a share of building a model.
    #[arg(long)]
    worker: bool,

    /// List the nodes of the [workers] channel, servers and gguf nodes.
    #[arg(long)]
    channel: bool,

    /// Read [workers] from this file instead of orangu-gguf.conf.
    #[arg(long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Print the perplexity of the -m model over this text.
    #[arg(long, value_name = "TEXT", requires = "model")]
    perplexity: Option<PathBuf>,

    /// Record a CPU flamegraph of the run and render it here.
    #[arg(long, value_name = "PATH")]
    flamegraph: Option<PathBuf>,

    /// Sampling frequency in Hz for `--flamegraph`.
    #[arg(long = "flamegraph-freq", default_value_t = 999, value_name = "HZ")]
    flamegraph_freq: u32,

    /// Call-graph mode for `--flamegraph`: `fp` or `dwarf`.
    #[arg(
        long = "flamegraph-call-graph",
        default_value = "fp",
        value_name = "MODE"
    )]
    flamegraph_call_graph: String,

    /// Also render a PNG beside the flamegraph SVG.
    #[arg(long = "flamegraph-png", default_value_t = false)]
    flamegraph_png: bool,

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

fn main() -> Result<()> {
    let args = Args::parse_from(normalize(std::env::args_os()));

    if args.shell_completions {
        return shell_completions::print("orangu-gguf", &COMPLETION_SCRIPTS);
    }

    stages::init();

    if args.list_quantizations {
        for ftype in Ftype::ALL {
            println!("{:<8} {}", ftype.name(), ftype.description());
        }
        return Ok(());
    }
    node::configure(args.config.clone());
    if let Some(text) = &args.perplexity {
        let model = listed(
            args.model
                .as_deref()
                .context("--perplexity needs -m MODEL")?,
        )?;
        let links = jobs::claim_for("perplexity");
        let tally = jobs::perplexity(&model, text, kl::CHUNK, links)?;
        let (perplexity, error) = tally.perplexity();
        println!(
            "perplexity {perplexity:.4} +/- {error:.5}, {} tokens scored in windows of {}",
            tally.count,
            kl::CHUNK
        );
        return Ok(());
    }
    if args.worker || args.channel {
        let settings = node::Settings::load(args.config.as_deref())?.ok_or_else(|| {
            anyhow::anyhow!(
                "no channel to join: give orangu-gguf.conf a [workers] section with a name \
                 (and the secret, when the channel has one)"
            )
        })?;
        return if args.worker {
            node::Worker::serve(settings)
        } else {
            node::list(&settings)
        };
    }
    if let Some(path) = &args.dump {
        return inspect::dump(&listed(path)?);
    }
    if let Some(path) = &args.hash {
        return inspect::hash(&listed(path)?);
    }
    if let Some(pair) = &args.compare {
        return inspect::compare(&listed(&pair[0])?, &listed(&pair[1])?);
    }
    let layout = layout_of(&args)?;
    if let Some(path) = &args.merge {
        return merge(&listed(path)?, args.output.as_deref(), &layout);
    }

    match (&args.manifest, &args.model) {
        (Some(manifest), None) => build(&args, manifest),
        (None, Some(model)) => {
            // Nothing here has a manifest to take a default from, so the
            // command line keeps one of its own.
            // `-q auto` chooses within the standard `Q4_K_M`, planned first.
            let requested = args.quantization.as_deref().unwrap_or("bf16");
            let auto = requested.eq_ignore_ascii_case("auto").then_some(());
            let options = optimize::Options {
                max_bytes: args
                    .max_size
                    .as_deref()
                    .map(optimize::parse_size)
                    .transpose()?,
                max_kl: args.max_kl,
                text: args.calibration.clone(),
                overrides: args
                    .tensor_type
                    .iter()
                    .map(|t| optimize::parse_override(t))
                    .collect::<Result<_>>()?,
            };
            if auto.is_none() && (options.max_bytes.is_some() || options.max_kl.is_some()) {
                bail!("--max-size and --max-kl go with -q auto");
            }
            match (auto, &args.imatrix, &args.calibration) {
                (Some(()), _, None) => bail!(
                    "-q auto needs --calibration TEXT: it measures each choice's KL over that text"
                ),
                (None, Some(_), Some(_)) => bail!(
                    "--imatrix and --calibration are alternatives, except with -q auto, where \
                     the matrix comes from --imatrix and the KL is measured over --calibration"
                ),
                _ => {}
            }
            let ftype = if auto.is_some() {
                Ftype::Q4KM
            } else {
                Ftype::parse(requested)?
            };
            let matrix = args
                .imatrix
                .as_deref()
                .map(imatrix::Loaded::open)
                .transpose()?;
            // A matrix given with --imatrix is used as it is; the text is
            // then only what -q auto measures KL over.
            let calibrated = |source: &Path| -> Result<Option<imatrix::Loaded>> {
                args.calibration
                    .as_deref()
                    .filter(|_| args.imatrix.is_none())
                    .map(|text| imatrix::Loaded::open(&imatrix::measure(source, text, None)?))
                    .transpose()
            };
            // `--imatrix-out`: measure into that file and stop.
            let measure_only = |source: &Path| -> Result<Option<()>> {
                let (Some(out), Some(text)) = (&args.imatrix_out, &args.calibration) else {
                    return Ok(None);
                };
                let written = imatrix::measure(source, text, Some(out))?;
                println!("wrote {}", written.display());
                Ok(Some(()))
            };
            // A file or checkpoint directory here; then a model the server
            // lists, by name or NR; then a Hugging Face repository.
            let file = if model.is_file() {
                Some(model.clone())
            } else if !model.exists() {
                listed(model).ok()
            } else {
                None
            };
            if let Some(path) = file {
                // No format: the tensors are copied as stored.
                if args.quantization.is_none()
                    && (layout.split.is_set() || layout.keep_split || !layout.edits.is_empty())
                {
                    return rewrite(&path, args.output.as_deref(), &layout);
                }
                if measure_only(&path)?.is_some() {
                    return Ok(());
                }
                let measured = calibrated(&path)?;
                return convert(
                    args.output.as_deref(),
                    &path,
                    ftype,
                    measured.as_ref().or(matrix.as_ref()),
                    &options,
                    auto.is_some(),
                    &layout,
                );
            }
            match hf::locate(model)? {
                Some(checkpoint) => {
                    // A checkpoint is measured as the BF16 file it converts
                    // to, which is kept beside the result.
                    let measured = match &args.calibration {
                        Some(_) => {
                            let source = hf::run(&checkpoint, None, Ftype::Bf16, None)?;
                            if measure_only(&source)?.is_some() {
                                return Ok(());
                            }
                            calibrated(&source)?
                        }
                        None => None,
                    };
                    let matrix = measured.as_ref().or(matrix.as_ref());
                    if auto.is_some() {
                        // Chosen from the BF16 file the checkpoint converts to.
                        let source = hf::run(&checkpoint, None, Ftype::Bf16, None)?;
                        return convert(
                            args.output.as_deref(),
                            &source,
                            ftype,
                            matrix,
                            &options,
                            true,
                            &layout,
                        );
                    }
                    if layout.pure || layout.requantize || layout.keep_split {
                        bail!(
                            "--pure, --allow-requantize and --keep-split apply to a GGUF source; \
                             convert the checkpoint first"
                        );
                    }
                    let written = hf::run(&checkpoint, args.output.clone(), ftype, matrix)?;
                    finish_checkpoint(&written, &layout)
                }
                None => bail!(
                    "{} is not a model file, a checkpoint or a listed model",
                    model.display()
                ),
            }
        }
        (Some(_), Some(_)) => bail!(
            "give a manifest to train from, or --model to convert an existing file — not both"
        ),
        (None, None) => bail!(
            "nothing to do: pass a manifest to train from, or --model to convert an \
             existing file — orangu-gguf --help lists both"
        ),
    }
}

/// The layout and metadata options of the command line.
fn layout_of(args: &Args) -> Result<Layout> {
    let mut edits = Vec::new();
    for text in args.set.iter().chain(&args.override_kv) {
        edits.push(shards::parse_set(text)?);
    }
    for key in &args.delete {
        edits.push(shards::Edit::Delete(key.clone()));
    }
    if args.split_max_tensors == Some(0) {
        bail!("--split-max-tensors 0 would put no tensor anywhere");
    }
    let max_bytes = args
        .split_max_size
        .as_deref()
        .map(optimize::parse_size)
        .transpose()?;
    if max_bytes == Some(0) {
        bail!("--split-max-size 0 would put no tensor anywhere");
    }
    Ok(Layout {
        split: shards::Split {
            max_bytes,
            max_tensors: args.split_max_tensors,
        },
        keep_split: args.keep_split,
        pure: args.pure,
        requantize: args.allow_requantize,
        edits,
    })
}

/// Writes `source` again with every tensor as stored: cut into new shards,
/// or with its metadata changed — in place, when no output is named and
/// nothing is re-cut.
fn rewrite(source: &Path, output: Option<&Path>, layout: &Layout) -> Result<()> {
    if layout.pure || layout.requantize {
        bail!("--pure and --allow-requantize go with -q: without a format nothing is quantized");
    }
    if layout.keep_split && layout.split.is_set() {
        bail!("--keep-split keeps the source's shards; --split-max-* cuts new ones — give one");
    }
    let set = shards::Source::open(source)?;
    if !layout.split.is_set() && output.is_none() {
        if layout.edits.is_empty() {
            bail!("nothing to change: name an output, a shard limit or a metadata edit");
        }
        let first = &set.paths[0];
        let in_place = shards::edit(first, &layout.edits)?;
        println!(
            "{} {}",
            if in_place {
                "edited in place:"
            } else {
                "the header changed size, so the file was rewritten:"
            },
            first.display()
        );
        return Ok(());
    }
    let mut metadata = set.metadata.clone();
    shards::apply(&mut metadata, &layout.edits)?;
    let groups = if layout.split.is_set() {
        copy_groups(&set, &metadata, &layout.split)?
    } else {
        set.shard_groups()
    };
    let output = output
        .map(Path::to_path_buf)
        .unwrap_or_else(|| source.with_file_name(format!("{}.gguf", shards::set_stem(source))));
    write_copy(&set, &output, &metadata, &groups)
}

/// The shards a byte-for-byte copy of `set` is cut into.
fn copy_groups(
    set: &shards::Source,
    metadata: &[(String, GgufValue)],
    split: &shards::Split,
) -> Result<Vec<Vec<usize>>> {
    let sizes: Vec<u64> = (0..set.tensors.len())
        .map(|i| write::aligned(set.bytes(i).len() as u64, set.alignment))
        .collect();
    let records: Vec<(&str, &[u64], u32)> = set
        .tensors
        .iter()
        .map(|t| (t.name.as_str(), t.dims.as_slice(), t.ggml_type))
        .collect();
    Ok(shards::groups(
        &sizes,
        split,
        shards::overheads(metadata, &records, set.alignment)?,
    ))
}

/// `--merge`: a shard set into one file, or re-cut to new limits.
fn merge(source: &Path, output: Option<&Path>, layout: &Layout) -> Result<()> {
    if layout.pure || layout.requantize || layout.keep_split {
        bail!(
            "--merge copies tensors as stored: --pure, --allow-requantize and --keep-split do not apply"
        );
    }
    let set = shards::Source::open(source)?;
    if !set.is_split() {
        bail!(
            "{} is one file, not the first shard of a set",
            source.display()
        );
    }
    let mut metadata = set.metadata.clone();
    shards::apply(&mut metadata, &layout.edits)?;
    let groups = copy_groups(&set, &metadata, &layout.split)?;
    let output = output
        .map(Path::to_path_buf)
        .unwrap_or_else(|| source.with_file_name(format!("{}.gguf", shards::set_stem(source))));
    write_copy(&set, &output, &metadata, &groups)
}

fn write_copy(
    set: &shards::Source,
    output: &Path,
    metadata: &[(String, GgufValue)],
    groups: &[Vec<usize>],
) -> Result<()> {
    let started = Instant::now();
    println!(
        "source      {} ({} tensors in {} file{})",
        set.paths[0].display(),
        set.tensors.len(),
        set.paths.len(),
        if set.paths.len() == 1 { "" } else { "s" }
    );
    let count = groups.len();
    println!(
        "output      {} ({count} file{})\n",
        shards::shard_path(output, 0, count).display(),
        if count == 1 { "" } else { "s" }
    );
    let written = shards::copy(set, output, metadata, groups)?;
    println!(
        "\nwrote {} in {count} file{} in {}",
        bytes(written),
        if count == 1 { "" } else { "s" },
        train::format_duration(started.elapsed().as_secs_f64())
    );
    Ok(())
}

/// A checkpoint's converted file, cut into shards or edited as asked.
fn finish_checkpoint(written: &Path, layout: &Layout) -> Result<()> {
    if layout.split.is_set() {
        let set = shards::Source::open(written)?;
        let mut metadata = set.metadata.clone();
        shards::apply(&mut metadata, &layout.edits)?;
        let groups = copy_groups(&set, &metadata, &layout.split)?;
        if groups.len() > 1 {
            write_copy(&set, written, &metadata, &groups)?;
            drop(set);
            fs::remove_file(written).with_context(|| format!("removing {}", written.display()))?;
            return Ok(());
        }
    }
    if !layout.edits.is_empty() {
        shards::edit(written, &layout.edits)?;
    }
    Ok(())
}

/// A model file named by path, or by the name or NR `orangu-server list`
/// shows for it in the configured models directory.
fn listed(model: &Path) -> Result<PathBuf> {
    if model.is_file() {
        return Ok(model.to_path_buf());
    }
    let models = hf::models_dir().with_context(|| {
        format!(
            "{} is not a file, and no models directory is configured to look it up in",
            model.display()
        )
    })?;
    orangu::model_spec::resolve_show_target(&models, &model.to_string_lossy())
}

/// Rewrites the two multi-letter short options into their long spellings.
///
/// `-ts` and `-cs` are what the tool is documented to take, and a single
/// dash followed by two letters is a *group of two short flags* to any
/// ordinary argument parser — `-ts` would be `-t -s`. Rewriting them here
/// keeps both spellings working without inventing a second parser.
fn normalize(args: impl Iterator<Item = OsString>) -> Vec<OsString> {
    args.map(|arg| {
        let Some(text) = arg.to_str() else { return arg };
        let (name, rest) = match text.split_once('=') {
            Some((name, value)) => (name, Some(value)),
            None => (text, None),
        };
        let long = match name {
            "-ts" => "--training-size",
            "-cs" => "--context-size",
            _ => return arg,
        };
        match rest {
            Some(value) => OsString::from(format!("{long}={value}")),
            None => OsString::from(long),
        }
    })
    .collect()
}

/// Parses a context length, which is conventionally written in thousands
/// (`256k`) rather than as the 262144 it means.
fn parse_context(value: &str) -> Result<usize> {
    let text = value.trim().to_ascii_lowercase();
    let (digits, multiplier) = match text.strip_suffix('k') {
        Some(digits) => (digits, 1024),
        None => match text.strip_suffix('m') {
            Some(digits) => (digits, 1024 * 1024),
            None => (text.as_str(), 1),
        },
    };
    let count: f64 = digits
        .parse()
        .with_context(|| format!("{value:?} is not a context length"))?;
    let tokens = (count * multiplier as f64) as usize;
    if tokens == 0 {
        bail!("a context length of zero is not a context length");
    }
    Ok(tokens)
}

/// Checks that the tokenizer can reproduce corpus text exactly.
///
/// A byte-level vocabulary has no excuse for losing anything: every byte
/// has a token. A round trip that does not come back identical means the
/// merge table and the encoder disagree, which shows up later only as a
/// model that was trained on token sequences no prompt will ever produce.
fn verify_round_trip(encoder: &vocab::Encoder<'_>, files: &[PathBuf]) -> Result<()> {
    let sampled: Vec<&PathBuf> = files
        .iter()
        .step_by((files.len() / 16).max(1))
        .take(16)
        .collect();
    // Sixteen independent documents, and reading one is I/O — there is no
    // reason for them to take turns.
    let failed: Vec<&PathBuf> = sampled
        .par_iter()
        .filter(|path| {
            let Some(text) = corpus::read_document(path) else {
                return false;
            };
            let sample: String = text.chars().take(4000).collect();
            encoder.decode(&encoder.encode(&sample)) != sample
        })
        .copied()
        .collect();
    if let Some(path) = failed.first() {
        bail!(
            "the tokenizer does not reproduce {} — the vocabulary and the merge table disagree",
            path.display()
        );
    }
    println!("tokenizer: round-trips {} sampled documents", sampled.len());
    Ok(())
}

/// The full pipeline: corpus, tokenizer, packed tokens, training, export.
/// Starts sampling this process, if the run asked for a flamegraph.
///
/// The pool is warmed first, and that is not a nicety. `perf record -p`
/// attaches to the threads that exist at the moment it starts and never
/// picks up ones created later, and rayon builds its workers lazily — on a
/// run whose corpus is already packed, the first parallel region is inside
/// training, so a recorder started before it would sample the main thread
/// and nothing else, and still produce a confident-looking flamegraph of a
/// program doing one thing at a time.
fn start_profile(args: &Args, label: &str) -> Result<Option<profile::Recorder>> {
    let Some(path) = &args.flamegraph else {
        return Ok(None);
    };
    rayon::broadcast(|_| {});
    let recorder = profile::Recorder::start(profile::Options {
        svg: path.clone(),
        pid: std::process::id(),
        freq: args.flamegraph_freq,
        call_graph: args.flamegraph_call_graph.clone(),
        png: args.flamegraph_png,
        title: format!("orangu-gguf · {label}"),
        keep_data: false,
    })?;
    Ok(Some(recorder))
}

/// Renders what was recorded, and says what it saw.
///
/// Reported rather than propagated: a run that trained for hours and then
/// could not render its profile has still produced the model, and losing
/// that to a missing `perf` would be the wrong trade.
fn finish_profile(recorder: Option<profile::Recorder>) {
    let Some(recorder) = recorder else { return };
    match recorder.finish() {
        Ok(summary) => {
            println!(
                "\nprofile    {} ({} samples, {:.1} cores busy)",
                summary.svg.display(),
                summary.samples,
                summary.cores_busy
            );
            println!("           {}", summary.folded.display());
            if let Some(png) = &summary.png {
                println!("           {}", png.display());
            }
        }
        Err(why) => eprintln!("\nprofile    not written: {why:#}"),
    }
}

fn build(args: &Args, manifest_path: &Path) -> Result<()> {
    let started = Instant::now();
    let mut manifest = Manifest::load(manifest_path)?;
    // The three overrides, applied onto the manifest rather than carried
    // beside it, so everything downstream reads one description of the run.
    if let Some(size) = &args.training_size {
        manifest.training_size = size.clone();
    }
    if let Some(quantization) = &args.quantization {
        manifest.quantization = quantization.clone();
    }
    if let Some(context) = &args.context_size {
        manifest.context_size = context.clone();
    }
    if let Some(output) = &args.output {
        manifest.output = Some(output.clone());
    }

    let ftype = Ftype::parse(&manifest.quantization)?;
    // Refused here rather than at the end: the matrix is measured on the
    // trained model, so without a text to measure it over the run would
    // train for nothing.
    if ftype.needs_importance() && manifest.calibration.is_none() {
        bail!(
            "{} needs an importance matrix: give the manifest a \"calibration\" text to \
             measure it over",
            ftype.name()
        );
    }
    let shape = Shape::named(&manifest.architecture, &manifest.training_size)?;
    let context = parse_context(&manifest.context_size)?;

    let work = match &manifest.work_dir {
        Some(dir) => dir.clone(),
        None => default_work_dir(manifest_path)?,
    };
    fs::create_dir_all(&work)
        .with_context(|| format!("creating the work directory {}", work.display()))?;
    let corpus_dir = work.join("corpus");
    let vocab_path = work.join("tokenizer.json");
    let tokens_path = work.join("tokens.bin");
    let checkpoint_path = work.join("checkpoint.bin");

    println!("manifest    {}", manifest_path.display());
    println!(
        "corpus      {} repositories under {}",
        manifest.repositories.len(),
        manifest.licences().join(", ")
    );
    // Excluded rather than fatal, but never silent: a corpus that is
    // quietly smaller than the manifest says is a run nobody can
    // reproduce.
    for repo in &manifest.excluded {
        println!(
            "            excluded {} ({}) — not an OSI-approved licence",
            repo.url, repo.license
        );
    }
    println!("work        {}", work.display());
    println!(
        "model       {} {}, context {context}",
        shape.key(),
        shape.architecture()
    );
    println!("output      {}", ftype.name());
    // The devices this machine offers, in the order this tool would use
    // them. Training runs on the CPU today — see GGUF.md, T6, for the
    // measurement that says why.
    let devices = gpu::devices();
    let usable: Vec<String> = devices
        .iter()
        .filter(|d| d.class != gpu::Class::Other)
        .map(|d| format!("{} ({:?})", d.name, d.class))
        .collect();
    if usable.is_empty() {
        println!("devices     CPU only");
    } else {
        println!("devices     CPU, and {}", usable.join(", "));
    }
    println!();

    // Everything from here to the written file is inside the profile, so a
    // manifest with no steps left to run profiles corpus preparation and one
    // with steps profiles training — which is how the two are told apart
    // without a second flag that could disagree with the manifest.
    let recorder = start_profile(args, shape.key())?;
    let outcome = (|| -> Result<()> {
        // 1. The corpus.
        let mut roots: Vec<corpus::Root> = if manifest.offline {
            vec![corpus::Root::repository(
                corpus_dir.clone(),
                manifest.max_file_size,
            )]
        } else {
            corpus::fetch_all(&manifest, &corpus_dir, manifest.jobs)?
                .into_iter()
                .map(|path| corpus::Root::repository(path, manifest.max_file_size))
                .collect()
        };

        // Prose, if the manifest asked for it. It lands inside the corpus
        // directory as plain text, so every stage after this one treats it as
        // one more source and nothing downstream needs to know where it came
        // from.
        let mut wikipedia_source: Option<String> = None;
        if let Some(settings) = &manifest.wikipedia {
            // Beside the clones, not inside them: it is not a repository, and
            // keeping it separate is what lets it be walked under its own
            // rules.
            let dir = work.join("wikipedia");
            if manifest.offline {
                println!(
                    "\nwikipedia: offline — using what is already in {}",
                    dir.display()
                );
            } else {
                println!(
                    "\nwikipedia: {}wiki, up to {} of article text",
                    settings.language,
                    bytes(settings.max_bytes)
                );
                let report = wikipedia::fetch(&dir, settings, &|report| {
                    print!(
                        "\r  {} articles, {} across {} shards    ",
                        report.articles,
                        bytes(report.bytes),
                        report.shards
                    );
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                })?;
                println!(
                    "\nwikipedia: {} articles, {} across {} shards",
                    report.articles,
                    bytes(report.bytes),
                    report.shards
                );
                wikipedia_source = Some(report.source);
            }
            roots.push(corpus::Root::generated(dir));
        }
        let (files, report) = corpus::scan(&roots);
        println!(
            "\ncorpus: {} files ({} skipped by name, {} too large)",
            report.kept, report.skipped_extension, report.skipped_large
        );
        if files.is_empty() {
            bail!("the corpus has no files to train on");
        }

        // 2. The tokenizer.
        let vocabulary = if vocab_path.exists() && !manifest.rebuild {
            println!("tokenizer: reusing {}", vocab_path.display());
            vocab::Vocab::load(&vocab_path)?
        } else {
            println!(
                "tokenizer: sampling up to {} of corpus text",
                bytes(manifest.tokenizer_sample)
            );
            let sample = pack::sample(&files, manifest.tokenizer_sample);
            let sampled: u64 = sample.iter().map(|s| s.len() as u64).sum();
            println!(
                "tokenizer: training {} tokens on {} from {} documents",
                manifest.vocab_size,
                bytes(sampled),
                sample.len()
            );
            let vocabulary =
                vocab::train(sample.into_iter(), manifest.vocab_size, &|done, total| {
                    print!("\r  merges {done}/{total}");
                    let _ = std::io::Write::flush(&mut std::io::stdout());
                })?;
            println!();
            vocabulary.save(&vocab_path)?;
            vocabulary
        };
        println!("tokenizer: {} tokens", vocabulary.len());
        let encoder = vocabulary.encoder()?;
        verify_round_trip(&encoder, &files)?;

        // 3. The packed token stream.
        if !tokens_path.exists() || manifest.rebuild {
            let report = pack::pack(&files, &encoder, &tokens_path, &|report, seen| {
                print!(
                    "\r  packing {}/{} files, {} tokens",
                    seen.min(files.len()),
                    files.len(),
                    report.tokens
                );
                let _ = std::io::Write::flush(&mut std::io::stdout());
            })?;
            println!(
                "\npacked: {} documents, {} tokens ({} duplicates, {} unreadable)",
                report.documents, report.tokens, report.duplicates, report.unreadable
            );
        } else {
            println!("packed: reusing {}", tokens_path.display());
        }
        let tokens = pack::Tokens::open(&tokens_path)?;
        println!("packed: {} tokens", tokens.len());

        // 4. The model.
        let sequence = manifest.sequence_length.min(context);
        let steps = match manifest.steps {
            Some(steps) => steps,
            None => {
                let per_step = (manifest.batch * sequence) as f64;
                ((tokens.len() as f64 * manifest.epochs) / per_step).ceil() as u64
            }
        };
        let options = train::Options {
            steps,
            batch: manifest.batch,
            sequence,
            peak_lr: manifest.learning_rate.unwrap_or_else(|| shape.peak_lr()),
            warmup: (steps / 20).clamp(1, 2000),
            weight_decay: 0.1,
            grad_clip: 1.0,
            seed: manifest.seed,
            log_every: manifest.log_every,
            eval_every: manifest.eval_every,
            checkpoint_every: manifest.checkpoint_every,
        };
        let output = manifest.output.clone().unwrap_or_else(|| {
            PathBuf::from(format!(
                "{}-{}-{}.gguf",
                manifest.name,
                shape.key(),
                ftype.name()
            ))
        });
        let run = Run {
            manifest: &manifest,
            manifest_path,
            work: &work,
            checkpoint: &checkpoint_path,
            tokens: &tokens,
            vocabulary: &vocabulary,
            options: &options,
            ftype,
            size: shape.key(),
            wikipedia_source: wikipedia_source.as_deref(),
            output: &output,
        };
        let written = match shape {
            Shape::Dense(size) => {
                train_and_write::<Model>(&run, Config::from_size(size, vocabulary.len(), context))?
            }
            Shape::Hybrid(size) => train_and_write::<model4::Model>(
                &run,
                model4::Config::from_size(size, vocabulary.len(), vocabulary.eos, context),
            )?,
        };
        println!(
            "\nwrote {} ({}) in {}",
            output.display(),
            bytes(written),
            train::format_duration(started.elapsed().as_secs_f64())
        );
        Ok(())
    })();
    finish_profile(recorder);
    outcome
}

/// The architecture a manifest names, at one of its sizes.
#[derive(Clone, Copy)]
enum Shape {
    Dense(&'static model::Size),
    Hybrid(&'static model4::Size),
}

impl Shape {
    fn named(architecture: &str, size: &str) -> Result<Self> {
        let unknown = |keys: Vec<&str>| {
            anyhow::anyhow!(
                "unknown {architecture} training size {size:?} — one of: {}",
                keys.join(", ")
            )
        };
        match architecture {
            "qwen4exp" => model4::size_named(size)
                .map(Shape::Hybrid)
                .ok_or_else(|| unknown(model4::SIZES.iter().map(|s| s.key).collect())),
            "qwen3" => model::size_named(size)
                .map(Shape::Dense)
                .ok_or_else(|| unknown(model::SIZES.iter().map(|s| s.key).collect())),
            other => bail!("unknown architecture {other:?} — one of: qwen4exp, qwen3"),
        }
    }

    fn key(&self) -> &'static str {
        match self {
            Shape::Dense(s) => s.key,
            Shape::Hybrid(s) => s.key,
        }
    }

    fn peak_lr(&self) -> f32 {
        match self {
            Shape::Dense(s) => s.peak_lr(),
            Shape::Hybrid(s) => s.peak_lr(),
        }
    }

    fn architecture(&self) -> &'static str {
        match self {
            Shape::Dense(_) => <Model as Trained>::ARCHITECTURE,
            Shape::Hybrid(_) => <model4::Model as Trained>::ARCHITECTURE,
        }
    }
}

/// A trainable model as a file carries it.
trait Trained: train::Network {
    const ARCHITECTURE: &'static str;
    fn create(cfg: Self::Config, seed: u64) -> Self;
    /// Whether the backward keeps activations within
    /// [`model::activation_budget`].
    const CACHES_ACTIVATIONS: bool;
    /// Every tensor's name and GGUF shape, in file order.
    fn file_specs(&self) -> Vec<(String, Vec<u64>)>;
    fn file_values(&self, name: &str) -> Option<Vec<f32>>;
    /// The `<architecture>.*` keys.
    fn architecture_metadata(&self) -> Vec<(String, GgufValue)>;
}

impl Trained for Model {
    const ARCHITECTURE: &'static str = "qwen3";
    const CACHES_ACTIVATIONS: bool = true;

    fn create(cfg: Config, seed: u64) -> Self {
        Model::new(cfg, seed)
    }
    fn file_specs(&self) -> Vec<(String, Vec<u64>)> {
        self.layout
            .specs
            .iter()
            .map(|s| (s.name.clone(), s.dims.clone()))
            .collect()
    }
    fn file_values(&self, name: &str) -> Option<Vec<f32>> {
        let s = self.layout.specs.iter().find(|s| s.name == name)?;
        Some(self.params[s.offset..s.offset + s.len()].to_vec())
    }
    fn architecture_metadata(&self) -> Vec<(String, GgufValue)> {
        let cfg = &self.cfg;
        let arch = Self::ARCHITECTURE;
        let mut out = Vec::new();
        for (suffix, value) in [
            ("context_length", cfg.context as u32),
            ("embedding_length", cfg.hidden as u32),
            ("block_count", cfg.layers as u32),
            ("feed_forward_length", cfg.ffn as u32),
            ("attention.head_count", cfg.heads as u32),
            ("attention.head_count_kv", cfg.kv_heads as u32),
            ("attention.key_length", cfg.head_dim as u32),
            ("attention.value_length", cfg.head_dim as u32),
            ("rope.dimension_count", cfg.head_dim as u32),
            ("vocab_size", cfg.vocab as u32),
        ] {
            out.push((format!("{arch}.{suffix}"), GgufValue::U32(value)));
        }
        out.push((
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            GgufValue::F32(cfg.eps),
        ));
        out.push((
            format!("{arch}.rope.freq_base"),
            GgufValue::F32(cfg.rope_base),
        ));
        out
    }
}

impl Trained for model4::Model {
    const ARCHITECTURE: &'static str = "qwen4exp";
    const CACHES_ACTIVATIONS: bool = false;

    fn create(cfg: model4::Config, seed: u64) -> Self {
        model4::Model::new(cfg, seed)
    }
    fn file_specs(&self) -> Vec<(String, Vec<u64>)> {
        model4::Model::file_specs(self)
    }
    fn file_values(&self, name: &str) -> Option<Vec<f32>> {
        model4::Model::file_values(self, name)
    }
    fn architecture_metadata(&self) -> Vec<(String, GgufValue)> {
        self.file_metadata()
            .into_iter()
            .filter(|(k, _)| k != "general.architecture")
            .collect()
    }
}

/// What one build's training and export read.
struct Run<'a> {
    manifest: &'a Manifest,
    manifest_path: &'a Path,
    work: &'a Path,
    checkpoint: &'a Path,
    tokens: &'a pack::Tokens,
    vocabulary: &'a vocab::Vocab,
    options: &'a train::Options,
    ftype: Ftype,
    size: &'static str,
    wikipedia_source: Option<&'a str>,
    output: &'a Path,
}

/// Trains a model of `cfg` — from its checkpoint when the manifest resumes
/// — and writes it out: the bytes written.
fn train_and_write<N: Trained>(run: &Run, cfg: N::Config) -> Result<u64> {
    let (manifest, options) = (run.manifest, run.options);
    let steps = options.steps;
    if manifest.export_only && !run.checkpoint.exists() {
        bail!(
            "\"export_only\" has nothing to export: {} does not exist",
            run.checkpoint.display()
        );
    }
    let (mut network, mut optimizer, start_step) =
        if (manifest.resume || manifest.export_only) && run.checkpoint.exists() {
            let (network, optimizer, step) = train::load::<N>(run.checkpoint, &cfg)?;
            println!("resuming from step {step}");
            (network, optimizer, step)
        } else {
            // A checkpoint in the way stops the run — but only when there
            // is something in it to lose. One left by a run that reached its
            // last step has nothing unfinished in it, and refusing to start
            // over that would make a repeatable stage un-repeatable.
            if run.checkpoint.exists() && !manifest.rebuild {
                let reached = train::peek(run.checkpoint)?;
                if reached < steps {
                    bail!(
                        "{} already holds a checkpoint, stopped at step {reached} of {steps}.\n  \
                         To continue it, set \"resume\": true in {}.\n  \
                         To start over, set \"rebuild\": true there, or delete the file.",
                        run.checkpoint.display(),
                        run.manifest_path.display()
                    );
                }
                println!("replacing the finished checkpoint at step {reached}");
            }
            let network = N::create(cfg, manifest.seed);
            let optimizer = train::Optimizer::new(N::parameters(network.config()));
            (network, optimizer, 0)
        };
    let parameters = N::parameters(network.config());

    if manifest.export_only {
        println!("\nexporting {parameters} parameters as they stand\n");
    } else {
        println!(
            "\ntraining {parameters} parameters for {steps} steps of {} x {} tokens",
            options.batch, options.sequence
        );
        // What the backward will keep rather than recompute. Printed
        // because it is the one setting that trades memory for time, and a
        // run that quietly took a gigabyte should say so.
        if N::CACHES_ACTIVATIONS {
            let budget = model::activation_budget();
            if budget == 0 {
                println!("activations recomputed (ORANGU_GGUF_CACHE_MIB=0)");
            } else {
                println!("activations cached up to {}", bytes(budget as u64));
            }
        }
        println!();
        train::run(
            &mut network,
            &mut optimizer,
            run.tokens,
            options,
            run.checkpoint,
            start_step,
        )?;
        if options.checkpoint_every == 0 {
            train::save(run.checkpoint, &network, &optimizer, steps)?;
        }
    }

    // 5. The model file.
    let metadata = |ftype| {
        let mut out = model_metadata(
            manifest,
            N::ARCHITECTURE,
            parameters,
            ftype,
            run.size,
            run.wikipedia_source,
        );
        out.extend(network.architecture_metadata());
        out
    };
    match &manifest.calibration {
        None => export(
            run.output,
            &network,
            run.vocabulary,
            metadata(run.ftype),
            run.ftype,
            &manifest.chat_template,
        ),
        // Measured as BF16 in the work directory, then written at the
        // manifest's format fitted to what it measured.
        Some(text) => {
            let full = run
                .work
                .join(format!("{}-{}-BF16.gguf", manifest.name, run.size));
            export(
                &full,
                &network,
                run.vocabulary,
                metadata(Ftype::Bf16),
                Ftype::Bf16,
                &manifest.chat_template,
            )?;
            let measured = imatrix::measure(
                &full,
                text,
                Some(&run.work.join("calibration.imatrix.gguf")),
            )?;
            let matrix = imatrix::Loaded::open(&measured)?;
            convert(
                Some(run.output),
                &full,
                run.ftype,
                Some(&matrix),
                &optimize::Options::default(),
                false,
                &Layout::default(),
            )?;
            Ok(fs::metadata(run.output)
                .with_context(|| format!("reading {}", run.output.display()))?
                .len())
        }
    }
}

/// Writes a trained model out at `ftype`.
fn export<N: Trained>(
    path: &Path,
    network: &N,
    vocabulary: &vocab::Vocab,
    mut metadata: Vec<(String, GgufValue)>,
    ftype: Ftype,
    chat_template: &str,
) -> Result<u64> {
    metadata.extend(tokenizer_metadata(vocabulary, chat_template));

    let model = quant::Model::from_metadata(&metadata);
    let specs = network.file_specs();
    let inputs: Vec<quant::Tensor<'_>> = specs
        .iter()
        .map(|(name, dims)| quant::Tensor {
            name,
            dims,
            source: quant::GGML_TYPE_F32,
        })
        .collect();
    let mut plans = Vec::with_capacity(specs.len());
    let mut fallbacks = Vec::new();
    for ((name, dims), plan) in specs
        .iter()
        .zip(quant::plan_tensors(ftype, &inputs, &model))
    {
        if let Some(from) = plan.fallback_from {
            fallbacks.push(format!(
                "{name}: {} does not divide {}, wrote {}",
                quant::type_name(from),
                dims[0],
                quant::type_name(plan.ggml_type)
            ));
        }
        plans.push(write::TensorPlan {
            name: name.clone(),
            dims: dims.clone(),
            ggml_type: plan.ggml_type,
        });
    }
    for line in &fallbacks {
        println!("note: {line}");
    }
    println!(
        "\nwriting {} tensors, {} of weights",
        plans.len(),
        bytes(write::planned_bytes(&plans))
    );

    imatrix::require(ftype, &plans, None)?;
    write::write(
        path,
        &metadata,
        &plans,
        |plan| {
            network
                .file_values(&plan.name)
                .ok_or_else(|| anyhow::anyhow!("no weights for {}", plan.name))
        },
        &|n, total, plan| {
            print!(
                "\r  writing {}/{total} {} as {}    ",
                n + 1,
                plan.name,
                quant::type_name(plan.ggml_type)
            );
            let _ = std::io::Write::flush(&mut std::io::stdout());
        },
    )
}

/// The general and provenance keys of a trained model.
fn model_metadata(
    manifest: &Manifest,
    arch: &str,
    parameters: usize,
    ftype: Ftype,
    size: &str,
    wikipedia_source: Option<&str>,
) -> Vec<(String, GgufValue)> {
    let mut out: Vec<(String, GgufValue)> = vec![
        (
            "general.architecture".into(),
            GgufValue::String(arch.into()),
        ),
        ("general.type".into(), GgufValue::String("model".into())),
        (
            "general.name".into(),
            GgufValue::String(format!("{}-{size}", manifest.name)),
        ),
        (
            "general.size_label".into(),
            GgufValue::String(write::size_label(parameters)),
        ),
        (
            "general.file_type".into(),
            GgufValue::U32(ftype.file_type()),
        ),
        ("general.quantization_version".into(), GgufValue::U32(2)),
    ];
    if let Some(license) = &manifest.license {
        out.push(("general.license".into(), GgufValue::String(license.clone())));
    }
    if let Some(description) = &manifest.description {
        out.push((
            "general.description".into(),
            GgufValue::String(description.clone()),
        ));
    }

    // Provenance, in this tool's own namespace: a model trained from other
    // people's repositories should carry the list of them, and the licences
    // it was taken under, inside the file rather than in a note beside it.
    out.push((
        "orangu.training.repository_count".into(),
        GgufValue::U32(manifest.repositories.len() as u32),
    ));
    out.push((
        "orangu.training.repositories".into(),
        GgufValue::Array(
            manifest
                .repositories
                .iter()
                .map(|r| GgufValue::String(r.url.clone()))
                .collect(),
        ),
    ));
    if let Some(settings) = &manifest.wikipedia {
        out.push((
            "orangu.training.wikipedia".into(),
            GgufValue::String(
                wikipedia_source
                    .map(|source| source.to_string())
                    .unwrap_or_else(|| format!("{}wiki", settings.language)),
            ),
        ));
    }
    out.push((
        "orangu.training.licenses".into(),
        GgufValue::Array(
            manifest
                .licences()
                .into_iter()
                .map(GgufValue::String)
                .collect(),
        ),
    ));
    out
}

/// The vocabulary keys, in the shape a `"gpt2"`-model file carries them.
///
/// `chat_template` is written as `tokenizer.chat_template` unless it is
/// empty, which is how a manifest asks for a file with no template at all.
fn tokenizer_metadata(vocabulary: &vocab::Vocab, chat_template: &str) -> Vec<(String, GgufValue)> {
    let mut out = vec![
        (
            "tokenizer.ggml.model".into(),
            GgufValue::String("gpt2".into()),
        ),
        (
            "tokenizer.ggml.pre".into(),
            GgufValue::String(vocab::PRE_TYPE.into()),
        ),
        (
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(
                vocabulary
                    .tokens
                    .iter()
                    .map(|t| GgufValue::String(t.clone()))
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.token_type".into(),
            GgufValue::Array(
                vocabulary
                    .token_type
                    .iter()
                    .map(|&t| GgufValue::I32(t))
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.merges".into(),
            GgufValue::Array(
                vocabulary
                    .merges
                    .iter()
                    .map(|m| GgufValue::String(m.clone()))
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.bos_token_id".into(),
            GgufValue::U32(vocabulary.bos),
        ),
        (
            "tokenizer.ggml.eos_token_id".into(),
            GgufValue::U32(vocabulary.eos),
        ),
        (
            "tokenizer.ggml.add_bos_token".into(),
            GgufValue::Bool(false),
        ),
        (
            "tokenizer.ggml.add_eos_token".into(),
            GgufValue::Bool(false),
        ),
    ];
    // The turn ends on `<|im_end|>` and the document ends on `<|endoftext|>`;
    // a chat client needs to be told which is which, or it generates past the
    // end of the answer and into the next turn it invented.
    if let Some(eot) = vocabulary.id_of(vocab::CHATML[1]) {
        out.push(("tokenizer.ggml.eot_token_id".into(), GgufValue::U32(eot)));
    }
    if !chat_template.is_empty() {
        out.push((
            "tokenizer.chat_template".into(),
            GgufValue::String(chat_template.to_string()),
        ));
    }
    out
}

/// Rewrites an existing model at another weight format.
/// Chunks of the calibration text `-q auto` measures KL over.
const KL_CHUNKS: usize = 2;

/// How a written file is laid out and labelled, beyond its weight format.
#[derive(Debug, Default)]
struct Layout {
    /// Shards no larger than this.
    split: shards::Split,
    /// A shard set written as the source set is cut.
    keep_split: bool,
    /// One type for every quantizable tensor (`--pure`).
    pure: bool,
    /// Quantized source tensors are read back and quantized again.
    requantize: bool,
    /// Metadata changes, in order.
    edits: Vec<shards::Edit>,
}

fn convert(
    output: Option<&Path>,
    source: &Path,
    ftype: Ftype,
    matrix: Option<&imatrix::Loaded>,
    options: &optimize::Options,
    auto: bool,
    layout: &Layout,
) -> Result<()> {
    let started = Instant::now();
    let format = if auto { "auto" } else { ftype.name() };
    let set = shards::Source::open(source)?;
    let file = set.as_file();
    let architecture = file
        .metadata
        .iter()
        .find(|(k, _)| k == "general.architecture")
        .and_then(|(_, v)| match v {
            GgufValue::String(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_else(|| "unknown".to_string());
    let model = quant::Model {
        imatrix: matrix.is_some(),
        pure: layout.pure,
        ..quant::Model::from_metadata(&file.metadata)
    };
    let layers = model.layers;
    if layout.keep_split && !set.is_split() {
        bail!(
            "--keep-split needs a shard set to keep: {} is one file",
            source.display()
        );
    }
    if layout.keep_split && layout.split.is_set() {
        bail!("--keep-split keeps the source's shards; --split-max-* cuts new ones — give one");
    }

    let output = output
        .map(Path::to_path_buf)
        .unwrap_or_else(|| source.with_file_name(format!("{}-{}.gguf", base_name(source), format)));

    // Every metadata key comes across unchanged except the ones that
    // describe the encoding, which are now this file's, not the source's.
    let mut metadata: Vec<(String, GgufValue)> = file
        .metadata
        .iter()
        .filter(|(k, _)| {
            k != "general.file_type"
                && k != "general.quantization_version"
                && !k.starts_with("quantize.imatrix.")
        })
        .cloned()
        .collect();
    metadata.push((
        "general.file_type".into(),
        GgufValue::U32(ftype.file_type()),
    ));
    metadata.push(("general.quantization_version".into(), GgufValue::U32(2)));

    // Quantized tensors read back are rounded twice; done only when asked.
    let mut quantized: Vec<u32> = file
        .tensors
        .iter()
        .map(|t| t.ggml_type)
        .filter(|&t| {
            !matches!(
                t,
                quant::GGML_TYPE_F32 | quant::GGML_TYPE_F16 | quant::GGML_TYPE_BF16
            )
        })
        .collect();
    quantized.sort_unstable();
    quantized.dedup();
    let quantized: Vec<&str> = quantized.iter().map(|&t| quant::type_name(t)).collect();

    println!(
        "source      {} ({architecture}, {layers} blocks{})",
        source.display(),
        if set.is_split() {
            format!(", {} shards", set.paths.len())
        } else {
            String::new()
        }
    );
    println!("output      {} ({})\n", output.display(), format);
    if !quantized.is_empty() {
        if !layout.requantize {
            bail!(
                "{} holds {} tensors — quantizing them again rounds them twice. Quantize from \
                 the F32, F16 or BF16 file, or pass --allow-requantize",
                source.display(),
                quantized.join(", ")
            );
        }
        println!(
            "warning: requantizing from {}: each tensor is rounded twice, and the result is \
             worse than quantizing the original once\n",
            quantized.join(", ")
        );
    }

    // Listed block by block, the order the mixture's counters run in.
    let mut ordered: Vec<&orangu::gguf::TensorInfo> = file.tensors.iter().collect();
    quant::sort_for_quantizing(&mut ordered, |t| t.name.as_str());
    let inputs: Vec<quant::Tensor<'_>> = ordered
        .iter()
        .map(|t| quant::Tensor {
            name: &t.name,
            dims: &t.dims,
            source: t.ggml_type,
        })
        .collect();
    let mut plans = Vec::with_capacity(ordered.len());
    for (tensor, plan) in ordered
        .iter()
        .zip(quant::plan_tensors(ftype, &inputs, &model))
    {
        if let Some(from) = plan.fallback_from {
            println!(
                "note: {}: {} does not divide {}, writing {}",
                tensor.name,
                quant::type_name(from),
                tensor.dims[0],
                quant::type_name(plan.ggml_type)
            );
        }
        plans.push(write::TensorPlan {
            name: tensor.name.clone(),
            dims: tensor.dims.clone(),
            ggml_type: plan.ggml_type,
        });
    }

    let index: std::collections::HashMap<&str, usize> = set
        .tensors
        .iter()
        .enumerate()
        .map(|(i, t)| (t.name.as_str(), i))
        .collect();
    let read = |plan: &write::TensorPlan| -> Result<Vec<f32>> {
        let &i = index
            .get(plan.name.as_str())
            .ok_or_else(|| anyhow::anyhow!("{} vanished from the source", plan.name))?;
        set.values(i, layout.requantize)
    };

    if auto {
        let text = options
            .text
            .as_deref()
            .context("-q auto needs --calibration TEXT")?;
        let text = kl::Text::open(source, text, KL_CHUNKS)?;
        let summary = optimize::select(
            &file,
            source,
            &text,
            &mut plans,
            |name| matrix.and_then(|m| m.matrix.get(name)).map(<[f32]>::to_vec),
            read,
            options,
        )?;
        optimize::print(&summary, &plans);
        if let Some((_, value)) = metadata.iter_mut().find(|(k, _)| k == "general.file_type") {
            *value = GgufValue::U32(optimize::file_type(&plans));
        }
    }

    optimize::apply_overrides(&mut plans, &options.overrides)?;
    imatrix::require(ftype, &plans, matrix)?;
    if let Some(matrix) = matrix {
        matrix.apply(&plans, &mut metadata)?;
    }
    shards::apply(&mut metadata, &layout.edits)?;

    // The shards: the source's own, or cut to the limits given.
    let groups: Vec<Vec<usize>> = if layout.keep_split {
        let mut by_shard = vec![Vec::new(); set.paths.len()];
        for (p, plan) in plans.iter().enumerate() {
            by_shard[set.shard_of[index[plan.name.as_str()]]].push(p);
        }
        by_shard.retain(|g| !g.is_empty());
        by_shard
    } else {
        let sizes: Vec<u64> = plans
            .iter()
            .map(|p| write::aligned(p.bytes() as u64, 32))
            .collect();
        let records: Vec<(&str, &[u64], u32)> = plans
            .iter()
            .map(|p| (p.name.as_str(), p.dims.as_slice(), p.ggml_type))
            .collect();
        shards::groups(
            &sizes,
            &layout.split,
            shards::overheads(&metadata, &records, 32)?,
        )
    };
    let count = groups.len();
    for no in 0..count {
        let path = shards::shard_path(&output, no, count);
        if set.paths.contains(&path) {
            bail!("that would overwrite {} with itself", path.display());
        }
    }
    println!(
        "writing {} tensors, {} of weights{}",
        plans.len(),
        bytes(write::planned_bytes(&plans)),
        if count > 1 {
            format!(", in {count} shards")
        } else {
            String::new()
        }
    );

    // Nodes of a channel take pieces of the tensors worth dividing.
    let mut spread = if plans.iter().any(|p| quant::block_size(p.ggml_type) > 1) {
        spread::Spread::start(source, layout.requantize, jobs::claim_for("quantize"))?
    } else {
        None
    };
    let mut written = 0u64;
    for (no, group) in groups.iter().enumerate() {
        let path = shards::shard_path(&output, no, count);
        let keys = shards::shard_metadata(&metadata, no, count, plans.len());
        let part: Vec<write::TensorPlan> = group.iter().map(|&i| plans[i].clone()).collect();
        written += write::write_with(
            &path,
            &keys,
            &part,
            |plan, sink| {
                let reading = Instant::now();
                let values = read(plan)?;
                if values.len() != plan.elements() {
                    bail!(
                        "{} weights for a tensor of {}",
                        values.len(),
                        plan.elements()
                    );
                }
                let importance = matrix.and_then(|m| m.for_plan(plan));
                match spread.as_mut() {
                    Some(spread) => spread.tensor(
                        plan,
                        &values,
                        importance,
                        reading.elapsed().as_secs_f64(),
                        sink,
                    ),
                    None => sink(&match importance {
                        Some(w) => orangu::quantize::encode_weighted(
                            plan.ggml_type,
                            &values,
                            plan.ncols(),
                            w,
                        ),
                        None => orangu::quantize::encode(plan.ggml_type, &values, plan.ncols()),
                    }),
                }
            },
            &|n, total, plan| {
                print!(
                    "\r  writing {}/{total} {} as {}    ",
                    n + 1,
                    plan.name,
                    quant::type_name(plan.ggml_type)
                );
                let _ = std::io::Write::flush(&mut std::io::stdout());
            },
        )?;
        if count > 1 {
            println!("\n  wrote {}", path.display());
        }
    }

    if let Some(spread) = spread {
        spread.finish();
    }
    let was: u64 = set
        .paths
        .iter()
        .map(|p| fs::metadata(p).map(|m| m.len()))
        .sum::<std::io::Result<u64>>()?;
    println!(
        "\nwrote {} ({}, was {}) in {}",
        shards::shard_path(&output, 0, count).display(),
        bytes(written),
        bytes(was),
        train::format_duration(started.elapsed().as_secs_f64())
    );
    Ok(())
}

/// A source file's name with any weight format it already names stripped
/// off, so converting `m-BF16.gguf` produces `m-Q4_K_M.gguf` rather than
/// `m-BF16-Q4_K_M.gguf`, which claims two formats and is neither.
fn base_name(source: &Path) -> String {
    without_format(&shards::set_stem(source))
}

/// `stem` with a weight format it ends in (`-BF16`, `-Q4_K_M`, …) taken
/// off: a checkpoint published as `Model-BF16` converts to `Model-Q4_K_M`.
fn without_format(stem: &str) -> String {
    // The formats this tool writes, and the ones it does not but may well
    // be handed: a file called `model-Q8_0.gguf` is a perfectly good source
    // and its conversion should be `model-Q4_K_M.gguf`, not
    // `model-q8_0-Q4_K_M.gguf`.
    const RETIRED: [&str; 8] = [
        "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "IQ3_M", "IQ2_M",
    ];
    let written = Ftype::ALL.iter().map(|f| f.name());
    for name in written.chain(RETIRED) {
        let suffix = format!("-{name}");
        if stem.len() > suffix.len() && stem.to_ascii_uppercase().ends_with(&suffix) {
            return stem[..stem.len() - suffix.len()].to_string();
        }
    }
    stem.to_string()
}

/// `~/.orangu/gguf/<manifest name>` — beside the rest of the editor's own
/// state, and one directory per manifest so two builds do not share a
/// corpus.
fn default_work_dir(manifest: &Path) -> Result<PathBuf> {
    let home = home::home_dir().context("no home directory to put the work directory in")?;
    let name = manifest
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    Ok(home.join(".orangu").join("gguf").join(name))
}

fn bytes(count: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = count as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{count} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every flag clap parses is offered by all three hand-written
    /// completion scripts — see `orangu::shell_completions::unoffered`.
    /// The two-letter `-ts`/`-cs` are not clap's to know about, so they
    /// are checked by name.
    #[test]
    fn every_flag_is_offered_by_every_completion_script() {
        use clap::CommandFactory;
        let missing = shell_completions::unoffered(&Args::command(), &COMPLETION_SCRIPTS);
        assert!(missing.is_empty(), "not offered: {missing:?}");
        for flag in ["-ts", "-cs"] {
            assert!(
                COMPLETION_SCRIPTS.bash.contains(&format!("{flag} ")),
                "bash: {flag}"
            );
            assert!(
                COMPLETION_SCRIPTS.zsh.contains(&format!("{{{flag},")),
                "zsh: {flag}"
            );
            assert!(
                COMPLETION_SCRIPTS
                    .fish
                    .contains(&format!("-o {} ", &flag[1..])),
                "fish: {flag}"
            );
            assert!(
                COMPLETION_SCRIPTS
                    .powershell
                    .contains(&format!("@('{flag}', ")),
                "powershell: {flag}"
            );
        }
    }

    #[test]
    fn completion_script_detects_each_supported_shell() {
        for (shell, marker) in [
            ("/bin/bash", "bash completion for orangu-gguf"),
            ("/usr/bin/zsh", "#compdef orangu-gguf"),
            ("/usr/local/bin/fish", "fish completion for orangu-gguf"),
            ("pwsh", "PowerShell completion for orangu-gguf"),
            (
                r"C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe",
                "PowerShell completion for orangu-gguf",
            ),
        ] {
            let script =
                shell_completions::script("orangu-gguf", shell, false, &COMPLETION_SCRIPTS)
                    .expect(shell);
            assert!(script.contains(marker), "{shell} -> {marker}");
        }
    }

    fn normalized(args: &[&str]) -> Vec<String> {
        normalize(args.iter().map(OsString::from))
            .into_iter()
            .map(|a| a.to_string_lossy().into_owned())
            .collect()
    }

    /// `-ts` and `-cs` are two letters behind one dash, which every
    /// argument parser reads as two flags. They have to survive anyway.
    #[test]
    fn the_two_letter_short_options_become_their_long_spellings() {
        assert_eq!(
            normalized(&["orangu-gguf", "m.json", "-ts", "1b", "-cs", "8192"]),
            vec![
                "orangu-gguf",
                "m.json",
                "--training-size",
                "1b",
                "--context-size",
                "8192"
            ]
        );
        assert_eq!(
            normalized(&["orangu-gguf", "-ts=1b", "-cs=8k"]),
            vec!["orangu-gguf", "--training-size=1b", "--context-size=8k"]
        );
        // Anything else is left exactly as it was.
        assert_eq!(
            normalized(&["orangu-gguf", "-q", "q6_k", "-m", "a.gguf"]),
            vec!["orangu-gguf", "-q", "q6_k", "-m", "a.gguf"]
        );
    }

    #[test]
    fn both_spellings_of_every_option_parse_to_the_same_thing() {
        let short = Args::parse_from(normalize(
            [
                "orangu-gguf",
                "m.json",
                "-ts",
                "1b",
                "-cs",
                "8k",
                "-q",
                "q6_k",
            ]
            .iter()
            .map(OsString::from),
        ));
        let long = Args::parse_from(normalize(
            [
                "orangu-gguf",
                "m.json",
                "--training-size",
                "1b",
                "--context-size",
                "8k",
                "--quantization",
                "q6_k",
            ]
            .iter()
            .map(OsString::from),
        ));
        assert_eq!(short.training_size, long.training_size);
        assert_eq!(short.context_size, long.context_size);
        assert_eq!(short.quantization, long.quantization);
    }

    /// A bare manifest path sets no overrides at all: everything a run
    /// does comes out of the file, which is the whole point of the format.
    /// The options block in [`HELP`] is hand-written, so nothing but a
    /// test stops it drifting from the arguments it describes. An option
    /// added without a line here fails this.
    #[test]
    fn every_option_appears_in_the_help() {
        use clap::CommandFactory;
        let command = Args::command();
        let rendered = command.clone().render_help().to_string();
        for arg in command.get_arguments() {
            if let Some(long) = arg.get_long() {
                assert!(
                    rendered.contains(&format!("--{long}")),
                    "--{long} is not in the help block"
                );
            }
            if let Some(short) = arg.get_short() {
                assert!(
                    rendered.contains(&format!("-{short},")),
                    "-{short} is not in the help block"
                );
            }
        }
        // And the two spellings the argv rewrite exists for, in the column
        // an argument parser has no room for them in.
        assert!(rendered.contains("-ts, --training-size"), "{rendered}");
        assert!(rendered.contains("-cs, --context-size"), "{rendered}");
    }

    /// With nothing to go on, say what the arguments are rather than
    /// complaining that there were none.
    #[test]
    fn no_arguments_shows_the_help() {
        let error = Args::try_parse_from(["orangu-gguf"]).unwrap_err();
        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
        );
        let rendered = error.to_string();
        assert!(rendered.contains("Usage: orangu-gguf"), "{rendered}");
        assert!(rendered.contains("--model"), "{rendered}");

        // An argument that is enough on its own still works.
        assert!(Args::try_parse_from(["orangu-gguf", "--list-quantizations"]).is_ok());
    }

    #[test]
    fn a_bare_manifest_path_overrides_nothing() {
        let args = Args::parse_from(normalize(
            ["orangu-gguf", "m.json"].iter().map(OsString::from),
        ));
        assert_eq!(args.manifest.as_deref(), Some(Path::new("m.json")));
        assert!(args.training_size.is_none());
        assert!(args.quantization.is_none());
        assert!(args.context_size.is_none());
        assert!(args.output.is_none());

        // And the manifest's own defaults are the documented ones.
        let manifest =
            Manifest::parse(r#"{"repositories": [{"url": "u", "license": "MIT"}]}"#).unwrap();
        assert_eq!(manifest.training_size, "1b");
        assert_eq!(parse_context(&manifest.context_size).unwrap(), 262_144);
        assert_eq!(Ftype::parse(&manifest.quantization).unwrap(), Ftype::Bf16);
    }

    #[test]
    fn context_lengths_parse_the_way_they_are_written() {
        assert_eq!(parse_context("256k").unwrap(), 262_144);
        assert_eq!(parse_context("8192").unwrap(), 8192);
        assert_eq!(parse_context("1M").unwrap(), 1_048_576);
        assert_eq!(parse_context("0.5k").unwrap(), 512);
        assert!(parse_context("0").is_err());
        assert!(parse_context("wide").is_err());
    }

    #[test]
    fn a_conversion_replaces_the_format_in_the_name_rather_than_stacking_it() {
        assert_eq!(base_name(Path::new("/m/model-BF16.gguf")), "model");
        assert_eq!(base_name(Path::new("/m/model-q8_0.gguf")), "model");
        assert_eq!(base_name(Path::new("/m/model.gguf")), "model");
        // A name that is only a format keeps it, rather than becoming
        // nothing at all.
        assert_eq!(base_name(Path::new("/m/BF16.gguf")), "BF16");
        assert_eq!(
            without_format("Ministral-3-3B-Instruct-2512-BF16"),
            "Ministral-3-3B-Instruct-2512"
        );
    }

    /// Without a template a chat endpoint refuses the request outright, so
    /// a freshly trained model reads as broken in every chat client. The
    /// turn token has to be declared with it, or generation runs past the
    /// end of the answer.
    #[test]
    fn the_file_carries_a_chat_template_and_the_turn_token() {
        let vocabulary = vocab::Vocab {
            tokens: vec![
                "a".into(),
                vocab::END_OF_TEXT.into(),
                vocab::CHATML[0].into(),
                vocab::CHATML[1].into(),
            ],
            merges: Vec::new(),
            token_type: vec![1, 3, 3, 3],
            bos: 1,
            eos: 1,
        };

        let keys = tokenizer_metadata(&vocabulary, vocab::CHAT_TEMPLATE);
        let template = keys
            .iter()
            .find(|(k, _)| k == "tokenizer.chat_template")
            .map(|(_, v)| v.clone())
            .expect("no tokenizer.chat_template");
        assert!(matches!(template, GgufValue::String(ref t) if t.contains("<|im_start|>")));
        assert!(
            keys.iter()
                .any(|(k, v)| k == "tokenizer.ggml.eot_token_id" && matches!(v, GgufValue::U32(3)))
        );

        // A manifest asking for no template gets a file with none, rather
        // than one carrying a format its model was never trained on.
        let bare = tokenizer_metadata(&vocabulary, "");
        assert!(!bare.iter().any(|(k, _)| k == "tokenizer.chat_template"));
    }

    #[test]
    fn byte_counts_read_as_sizes() {
        assert_eq!(bytes(512), "512 B");
        assert_eq!(bytes(1536), "1.5 KiB");
        assert_eq!(bytes(4 << 30), "4.0 GiB");
    }

    /// The command line as documented has to actually reach the pipeline:
    /// the architecture keys a trained model carries are the ones the
    /// inference side reads back.
    #[test]
    fn the_metadata_names_every_key_the_reader_needs() {
        let manifest = Manifest::parse(
            r#"{"name":"n","license":"Apache-2.0","repositories":[{"url":"u","license":"MIT"}]}"#,
        )
        .unwrap();
        let network = Model::new(
            Config::from_size(model::size_named("tiny").unwrap(), 300, 262_144),
            1,
        );
        let mut metadata = model_metadata(&manifest, "qwen3", 0, Ftype::Bf16, "tiny", None);
        metadata.extend(network.architecture_metadata());
        let keys: Vec<&str> = metadata.iter().map(|(k, _)| k.as_str()).collect();
        for required in [
            "general.architecture",
            "general.file_type",
            "general.license",
            "qwen3.context_length",
            "qwen3.embedding_length",
            "qwen3.block_count",
            "qwen3.attention.head_count",
            "qwen3.attention.head_count_kv",
            "qwen3.attention.key_length",
            "qwen3.attention.layer_norm_rms_epsilon",
            "qwen3.rope.freq_base",
            "qwen3.vocab_size",
            "orangu.training.repositories",
        ] {
            assert!(
                keys.contains(&required),
                "{required} is missing from {keys:?}"
            );
        }
    }
}
