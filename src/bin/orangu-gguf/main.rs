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
use orangu::gguf::{GgufFile, GgufValue};
use orangu::profiling::profile;
use rayon::prelude::*;
use std::{
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

mod aligned;
mod corpus;
mod gpu;
mod manifest;
mod model;
mod pack;
mod quant;
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
/// A dense block with grouped-query attention, rotary positions, RMSNorm
/// and a SwiGLU feed-forward, plus the per-head query and key norms — the
/// strongest shape on the inference side's fully-supported dense path, and
/// the one whose stability a from-scratch run most benefits from.
const ARCHITECTURE: &str = "qwen3";

/// The help text, with the two-letter short options in the column they
/// belong in. `{about-with-newline}`, `{usage-heading}`, `{usage}` and
/// `{positionals}` are filled in by the parser; the options are not.
const HELP: &str = "\
{about-with-newline}
{usage-heading} {usage}

Arguments:
{positionals}

Options:
  -m,  --model <FILE>          Convert this GGUF file instead of training a new model
  -ts, --training-size <SIZE>  Training size for this run, overriding the manifest
  -q,  --quantization <QUANT>  Weight format written, overriding the manifest
  -cs, --context-size <N>      Context length the model declares, overriding the manifest
  -o,  --output <FILE>         Where the model is written, overriding the manifest
       --list-quantizations    List the weight formats a manifest's quantization accepts
       --flamegraph <PATH>     Record a CPU flamegraph of the run and render it here
       --flamegraph-freq <HZ>  Sampling frequency in Hz for --flamegraph [default: 999]
       --flamegraph-call-graph <MODE>
                               Call-graph mode for --flamegraph: fp or dwarf [default: fp]
       --flamegraph-png        Also render a PNG beside the flamegraph SVG
  -h,  --help                  Print help
  -V,  --version               Print version
";

#[derive(Parser, Debug)]
#[command(
    name = "orangu-gguf",
    version = orangu::build_info::VERSION,
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

    /// Convert this GGUF file instead of training a new model.
    #[arg(short = 'm', long, value_name = "FILE")]
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

    /// List the weight formats a manifest's `quantization` accepts and exit.
    #[arg(long = "list-quantizations")]
    list_quantizations: bool,

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
}

fn main() -> Result<()> {
    let args = Args::parse_from(normalize(std::env::args_os()));
    stages::init();

    if args.list_quantizations {
        for ftype in Ftype::ALL {
            println!("{:<8} {}", ftype.name(), ftype.description());
        }
        return Ok(());
    }

    match (&args.manifest, &args.model) {
        (Some(manifest), None) => build(&args, manifest),
        (None, Some(model)) => {
            // Nothing here has a manifest to take a default from, so the
            // command line keeps one of its own.
            let ftype = Ftype::parse(args.quantization.as_deref().unwrap_or("bf16"))?;
            convert(&args, model, ftype)
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
    let size = model::size_named(&manifest.training_size).ok_or_else(|| {
        anyhow::anyhow!(
            "unknown training size {:?} — one of: {}",
            manifest.training_size,
            model::SIZES
                .iter()
                .map(|s| s.key)
                .collect::<Vec<_>>()
                .join(", ")
        )
    })?;
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
    println!("model       {} {ARCHITECTURE}, context {context}", size.key);
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
    let recorder = start_profile(args, size.key)?;
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
        let cfg = Config::from_size(size, vocabulary.len(), context);
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
            peak_lr: manifest.learning_rate.unwrap_or_else(|| size.peak_lr()),
            warmup: (steps / 20).clamp(1, 2000),
            weight_decay: 0.1,
            grad_clip: 1.0,
            seed: manifest.seed,
            log_every: manifest.log_every,
            eval_every: manifest.eval_every,
            checkpoint_every: manifest.checkpoint_every,
        };

        if manifest.export_only && !checkpoint_path.exists() {
            bail!(
                "\"export_only\" has nothing to export: {} does not exist",
                checkpoint_path.display()
            );
        }
        let (mut network, mut optimizer, start_step) = if (manifest.resume || manifest.export_only)
            && checkpoint_path.exists()
        {
            let (network, optimizer, step) = train::load(&checkpoint_path, &cfg)?;
            println!("resuming from step {step}");
            (network, optimizer, step)
        } else {
            // A checkpoint in the way stops the run — but only when there
            // is something in it to lose. One left by a run that reached its
            // last step has nothing unfinished in it, and refusing to start
            // over that would make a repeatable stage un-repeatable.
            if checkpoint_path.exists() && !manifest.rebuild {
                let (reached, _) = train::peek(&checkpoint_path)?;
                if reached < steps {
                    bail!(
                        "{} already holds a checkpoint, stopped at step {reached} of {steps}.\n  \
                         To continue it, set \"resume\": true in {}.\n  \
                         To start over, set \"rebuild\": true there, or delete the file.",
                        checkpoint_path.display(),
                        manifest_path.display()
                    );
                }
                println!("replacing the finished checkpoint at step {reached}");
            }
            let network = Model::new(cfg.clone(), manifest.seed);
            let optimizer = train::Optimizer::new(network.layout.total);
            (network, optimizer, 0)
        };

        if manifest.export_only {
            println!(
                "\nexporting {} parameters as they stand\n",
                network.layout.total
            );
        } else {
            println!(
                "\ntraining {} parameters for {steps} steps of {} x {sequence} tokens\n",
                network.layout.total, options.batch
            );
        }
        if !manifest.export_only {
            train::run(
                &mut network,
                &mut optimizer,
                &tokens,
                &options,
                &checkpoint_path,
                start_step,
            )?;
            if options.checkpoint_every == 0 {
                train::save(&checkpoint_path, &network, &optimizer, steps)?;
            }
        }

        // 5. The model file.
        let output = manifest.output.clone().unwrap_or_else(|| {
            PathBuf::from(format!(
                "{}-{}-{}.gguf",
                manifest.name,
                size.key,
                ftype.name()
            ))
        });
        let metadata = model_metadata(
            &manifest,
            &cfg,
            ftype,
            size.key,
            wikipedia_source.as_deref(),
        );
        let written = export(
            &output,
            &network,
            &vocabulary,
            metadata,
            ftype,
            &manifest.chat_template,
        )?;
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

/// Writes a trained model out at `ftype`.
fn export(
    path: &Path,
    network: &Model,
    vocabulary: &vocab::Vocab,
    mut metadata: Vec<(String, GgufValue)>,
    ftype: Ftype,
    chat_template: &str,
) -> Result<u64> {
    metadata.extend(tokenizer_metadata(vocabulary, chat_template));

    let model = quant::Model {
        layers: network.cfg.layers,
        gqa: (network.cfg.heads / network.cfg.kv_heads.max(1)).max(1),
    };
    let mut plans = Vec::with_capacity(network.layout.specs.len());
    let mut fallbacks = Vec::new();
    for spec in &network.layout.specs {
        let plan = quant::plan_tensor(ftype, &spec.name, &spec.dims, model);
        if let Some(from) = plan.fallback_from {
            fallbacks.push(format!(
                "{}: {} does not divide {}, wrote {}",
                spec.name,
                quant::type_name(from),
                spec.dims[0],
                quant::type_name(plan.ggml_type)
            ));
        }
        plans.push(write::TensorPlan {
            name: spec.name.clone(),
            dims: spec.dims.clone(),
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

    let by_name: std::collections::HashMap<&str, &model::TensorSpec> = network
        .layout
        .specs
        .iter()
        .map(|s| (s.name.as_str(), s))
        .collect();

    write::write(
        path,
        &metadata,
        &plans,
        |plan| {
            let spec = by_name
                .get(plan.name.as_str())
                .ok_or_else(|| anyhow::anyhow!("no weights for {}", plan.name))?;
            Ok(network.params[spec.offset..spec.offset + spec.len()].to_vec())
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

/// The architecture and provenance keys of a trained model.
fn model_metadata(
    manifest: &Manifest,
    cfg: &Config,
    ftype: Ftype,
    size: &str,
    wikipedia_source: Option<&str>,
) -> Vec<(String, GgufValue)> {
    let arch = ARCHITECTURE;
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
            GgufValue::String(write::size_label(cfg.parameters())),
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
fn convert(args: &Args, source: &Path, ftype: Ftype) -> Result<()> {
    let started = Instant::now();
    let file = GgufFile::open(source).with_context(|| format!("reading {}", source.display()))?;
    let architecture = file
        .metadata
        .iter()
        .find(|(k, _)| k == "general.architecture")
        .and_then(|(_, v)| match v {
            GgufValue::String(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_else(|| ARCHITECTURE.to_string());
    let number = |key: &str| {
        file.metadata
            .iter()
            .find(|(k, _)| *k == format!("{architecture}.{key}"))
            .and_then(|(_, v)| v.as_u64())
            .unwrap_or(0) as usize
    };
    let layers = number("block_count");
    // The mixture rules need the query-per-key-value ratio: with grouped
    // query attention the value projection is small, and carrying it above
    // the file's base type costs almost nothing. A file that does not say
    // reads as 1, which is the conservative answer.
    let heads = number("attention.head_count");
    let kv_heads = number("attention.head_count_kv").max(1);
    let model = quant::Model {
        layers,
        gqa: (heads / kv_heads).max(1),
    };

    let output = args.output.clone().unwrap_or_else(|| {
        source.with_file_name(format!("{}-{}.gguf", base_name(source), ftype.name()))
    });
    if output == source {
        bail!("that would overwrite {} with itself", source.display());
    }

    println!(
        "source      {} ({architecture}, {layers} blocks)",
        source.display()
    );
    println!("output      {} ({})\n", output.display(), ftype.name());

    // Every metadata key comes across unchanged except the two that
    // describe the encoding, which are now this file's, not the source's.
    let mut metadata: Vec<(String, GgufValue)> = file
        .metadata
        .iter()
        .filter(|(k, _)| k != "general.file_type" && k != "general.quantization_version")
        .cloned()
        .collect();
    metadata.push((
        "general.file_type".into(),
        GgufValue::U32(ftype.file_type()),
    ));
    metadata.push(("general.quantization_version".into(), GgufValue::U32(2)));

    let mut plans = Vec::with_capacity(file.tensors.len());
    for tensor in &file.tensors {
        let plan = quant::plan_tensor(ftype, &tensor.name, &tensor.dims, model);
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

    println!(
        "writing {} tensors, {} of weights",
        plans.len(),
        bytes(write::planned_bytes(&plans))
    );

    let handle = fs::File::open(source)?;
    // Safety: the source file is opened read-only and this process does not
    // write to it.
    let map = unsafe { memmap2::Mmap::map(&handle) }
        .with_context(|| format!("mapping {}", source.display()))?;
    let by_name: std::collections::HashMap<&str, &orangu::gguf::TensorInfo> =
        file.tensors.iter().map(|t| (t.name.as_str(), t)).collect();

    let written = write::write(
        &output,
        &metadata,
        &plans,
        |plan| {
            let tensor = by_name
                .get(plan.name.as_str())
                .ok_or_else(|| anyhow::anyhow!("{} vanished from the source", plan.name))?;
            let elements = tensor.element_count() as usize;
            let ncols = tensor.dims[0] as usize;
            let bytes = quant::row_bytes(tensor.ggml_type, ncols) * (elements / ncols.max(1));
            let start = (file.data_offset + tensor.offset) as usize;
            let end = start + bytes;
            if bytes == 0 || end > map.len() {
                bail!(
                    "{} is {} in the source, which this tool cannot read back",
                    plan.name,
                    quant::type_name(tensor.ggml_type)
                );
            }
            quant::decode(tensor.ggml_type, &map[start..end], elements)
                .with_context(|| plan.name.clone())
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

    println!(
        "\nwrote {} ({}, was {}) in {}",
        output.display(),
        bytes(written),
        bytes(fs::metadata(source)?.len()),
        train::format_duration(started.elapsed().as_secs_f64())
    );
    Ok(())
}

/// A source file's name with any weight format it already names stripped
/// off, so converting `m-BF16.gguf` produces `m-Q4_K_M.gguf` rather than
/// `m-BF16-Q4_K_M.gguf`, which claims two formats and is neither.
fn base_name(source: &Path) -> String {
    let stem = source
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
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
        assert_eq!(manifest.training_size, "2b");
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
        let cfg = Config::from_size(model::size_named("2b").unwrap(), 32768, 262_144);
        let metadata = model_metadata(&manifest, &cfg, Ftype::Bf16, "2b", None);
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
