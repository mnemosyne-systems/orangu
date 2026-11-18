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

//! A Hugging Face checkpoint — a directory with `config.json`, the
//! tokenizer files and the `*.safetensors` weights — written as a GGUF
//! file `orangu-server` serves.
//!
//! Each supported architecture is a table rather than a code path: which
//! `config.json` keys become which hyperparameters, which checkpoint tensor
//! names become which GGUF names, and the few named transforms a tensor
//! needs on the way ([`Transform`]). Only architectures the server runs are
//! listed — a file the server cannot load is not worth writing.
//!
//! Weights pass through `f32` on their way to the planned type. From
//! `BF16` or `F16` that is exact, so a `BF16` checkpoint written as `BF16`,
//! or as `F16` where its values fit, holds the same bits a reader of the
//! checkpoint would compute.

mod bailing;
mod gemma;
mod gemma3;
mod llama;
mod phi;
mod qwen35;
mod qwen4;
pub mod tokenizer;

use anyhow::{Context, Result, anyhow, bail};
use orangu::gguf::GgufValue;
use orangu::safetensors::SafeTensorsSet;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::quant::{self, Ftype};
use crate::write::{self, TensorPlan};

/// What a tensor needs between the checkpoint and the file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transform {
    /// Written as stored.
    None,
    /// Rows regrouped from the checkpoint's rotary layout (each head's
    /// first half, then its second half) to interleaved pairs, for an
    /// architecture whose rotation pairs neighbouring elements. `heads` is
    /// how many heads the rows hold.
    PermuteRotary { heads: usize },
    /// One added to every value: a norm whose checkpoint stores its
    /// offset from one.
    AddOne,
    /// Every value `x` becomes `-exp(x)`: a decay stored as its logarithm.
    NegExp,
    /// Dimensions of size one dropped.
    Squeeze,
    /// Every value `x` becomes `exp(x)`: a rate stored as its logarithm.
    Exp,
    /// The values as one column: `[n, 1]`.
    Column,
    /// A depthwise convolution's `[channels, (1,) kernel]` weights as
    /// `[1, channels, 1, kernel]`.
    Conv4d,
    /// Value heads regrouped from the checkpoint's order (all of one key
    /// head's value heads in a row) to tiled order (one value head of
    /// every key head, then the next). The heads are `head_dim` rows each
    /// after the first `skip` rows, or the same in columns when
    /// `columns`; `groups` is the key head count, `per_group` the value
    /// heads each serves.
    TileHeads {
        skip: usize,
        groups: usize,
        per_group: usize,
        head_dim: usize,
        columns: bool,
    },
}

/// Where an output tensor's values come from.
#[derive(Debug, Clone)]
pub enum Source {
    /// One checkpoint tensor.
    Tensor(String),
    /// Checkpoint tensors of one shape stacked along a new outermost
    /// dimension — one expert's matrix per slice.
    Stack(Vec<String>),
    /// One half of a checkpoint tensor, cut along its second-innermost
    /// dimension: the gate (`second == false`) or the up projection of a
    /// fused gate-up matrix.
    Half { name: String, second: bool },
    /// Values the conversion computes from `config.json`.
    Values { values: Vec<f32>, shape: Vec<usize> },
    /// Checkpoint tensors of one row width concatenated along their rows, in
    /// order: a table the checkpoint stores in shards. Written a shard at a
    /// time when nothing transforms it.
    Rows(Vec<String>),
    /// Rows `offset..offset + count` of every one of `heads` equal blocks of
    /// a matrix's rows, as `[heads, count, columns]` — or, transposed,
    /// `[heads, columns, count]`: one projection out of a per-head fused
    /// pair.
    HeadRows {
        name: String,
        heads: usize,
        offset: usize,
        count: usize,
        transpose: bool,
    },
}

impl Source {
    pub fn tensor(name: impl Into<String>) -> Self {
        Source::Tensor(name.into())
    }

    /// The checkpoint tensors this reads.
    pub fn names(&self) -> Vec<&str> {
        match self {
            Source::Tensor(n) | Source::Half { name: n, .. } => vec![n.as_str()],
            Source::Stack(ns) => ns.iter().map(String::as_str).collect(),
            Source::HeadRows { name, .. } => vec![name.as_str()],
            Source::Rows(ns) => ns.iter().map(String::as_str).collect(),
            Source::Values { .. } => vec![],
        }
    }

    /// The row-major shape, without reading any values.
    fn shape(&self, weights: &SafeTensorsSet) -> Result<Vec<usize>> {
        let one = |n: &str| -> Result<Vec<usize>> {
            weights
                .info(n)
                .map(|(shape, _)| shape.to_vec())
                .ok_or_else(|| anyhow!("the checkpoint has no tensor '{n}'"))
        };
        match self {
            Source::Tensor(n) => one(n),
            Source::Stack(ns) => {
                let inner = one(&ns[0])?;
                for n in &ns[1..] {
                    if one(n)? != inner {
                        bail!("'{n}' does not have the shape of '{}'", ns[0]);
                    }
                }
                Ok(std::iter::once(ns.len()).chain(inner).collect())
            }
            Source::Half { name, .. } => {
                let mut shape = one(name)?;
                let at = shape
                    .len()
                    .checked_sub(2)
                    .ok_or_else(|| anyhow!("'{name}' is not a matrix"))?;
                if shape[at] % 2 != 0 {
                    bail!("'{name}' has an odd number of rows to halve");
                }
                shape[at] /= 2;
                Ok(shape)
            }
            Source::Rows(ns) => {
                let first = one(&ns[0])?;
                let mut rows = 0;
                for n in ns {
                    let shape = one(n)?;
                    if shape[1..] != first[1..] {
                        bail!("'{n}' does not have the rows of '{}'", ns[0]);
                    }
                    rows += shape[0];
                }
                Ok(std::iter::once(rows)
                    .chain(first[1..].iter().copied())
                    .collect())
            }
            Source::HeadRows {
                name,
                heads,
                count,
                transpose,
                ..
            } => {
                let shape = one(name)?;
                let [rows, cols] = shape[..] else {
                    bail!("'{name}' is not a matrix");
                };
                if *heads == 0 || !rows.is_multiple_of(*heads) {
                    bail!("'{name}' has {rows} rows, not {heads} equal heads");
                }
                Ok(if *transpose {
                    vec![*heads, cols, *count]
                } else {
                    vec![*heads, *count, cols]
                })
            }
            Source::Values { shape, .. } => Ok(shape.clone()),
        }
    }

    /// The values, in row-major order.
    fn read(&self, weights: &SafeTensorsSet) -> Result<Vec<f32>> {
        match self {
            Source::Tensor(n) => Ok(weights.tensor(n).with_context(|| n.clone())?.0),
            Source::Rows(ns) => {
                let mut out = Vec::new();
                for n in ns {
                    out.extend(weights.tensor(n).with_context(|| n.clone())?.0);
                }
                Ok(out)
            }
            Source::Stack(ns) => {
                let mut out = Vec::new();
                for n in ns {
                    out.extend(weights.tensor(n).with_context(|| n.clone())?.0);
                }
                Ok(out)
            }
            Source::Half { name, second } => {
                let (values, shape) = weights.tensor(name).with_context(|| name.clone())?;
                let cols = shape[shape.len() - 1];
                let rows = shape[shape.len() - 2];
                let block = rows * cols;
                let half = block / 2;
                let offset = if *second { half } else { 0 };
                Ok(values
                    .chunks_exact(block)
                    .flat_map(|b| b[offset..offset + half].iter().copied())
                    .collect())
            }
            Source::HeadRows {
                name,
                heads,
                offset,
                count,
                transpose,
            } => {
                let (values, shape) = weights.tensor(name).with_context(|| name.clone())?;
                let cols = shape[shape.len() - 1];
                let per_head = shape[0] / heads;
                let mut out = Vec::with_capacity(heads * count * cols);
                for h in 0..*heads {
                    let first = (h * per_head + offset) * cols;
                    let block = &values[first..first + count * cols];
                    if *transpose {
                        for c in 0..cols {
                            out.extend((0..*count).map(|r| block[r * cols + c]));
                        }
                    } else {
                        out.extend_from_slice(block);
                    }
                }
                Ok(out)
            }
            Source::Values { values, .. } => Ok(values.clone()),
        }
    }
}

/// One tensor of the output: its GGUF name, where its values come from,
/// and what is done to them.
#[derive(Debug, Clone)]
pub struct Mapped {
    pub name: String,
    pub source: Source,
    pub transforms: Vec<Transform>,
}

impl Mapped {
    pub fn new(name: impl Into<String>, source: Source) -> Self {
        Self {
            name: name.into(),
            source,
            transforms: Vec::new(),
        }
    }

    /// Adds a transform, applied after the ones before it.
    pub fn with(mut self, transform: Transform) -> Self {
        if transform != Transform::None {
            self.transforms.push(transform);
        }
        self
    }

    /// The row-major shape written, without reading any values.
    fn shape(&self, weights: &SafeTensorsSet) -> Result<Vec<usize>> {
        let mut shape = self.source.shape(weights)?;
        for &transform in &self.transforms {
            reshape(transform, &mut shape);
        }
        Ok(shape)
    }

    /// The values written, in row-major order.
    fn values(&self, weights: &SafeTensorsSet) -> Result<Vec<f32>> {
        let mut shape = self.source.shape(weights)?;
        let mut values = self.source.read(weights)?;
        for &transform in &self.transforms {
            values = apply(transform, values, &mut shape);
        }
        Ok(values)
    }
}

/// Refuses a conversion that leaves a checkpoint tensor unread.
///
/// A tensor nobody mapped is a part of the model the table does not know
/// about, and leaving it out gives a file that loads and answers wrong.
/// `skipped` names the ones left out on purpose.
pub fn check_all_read(
    weights: &SafeTensorsSet,
    tensors: &[Mapped],
    skipped: &[String],
    arch: &str,
) -> Result<()> {
    let used: std::collections::HashSet<&str> = tensors
        .iter()
        .flat_map(|t| t.source.names())
        .chain(skipped.iter().map(String::as_str))
        .collect();
    let unmapped: Vec<&str> = weights
        .names()
        .into_iter()
        .filter(|n| !used.contains(*n))
        .collect();
    if !unmapped.is_empty() {
        bail!(
            "{} checkpoint tensors have no place in a {arch} file, first {}",
            unmapped.len(),
            unmapped[0]
        );
    }
    Ok(())
}

/// An architecture's conversion: its hyperparameters and its tensors.
pub struct Conversion {
    pub architecture: &'static str,
    pub metadata: Vec<(String, GgufValue)>,
    pub tensors: Vec<Mapped>,
    pub layers: usize,
    pub vocab_size: usize,
    pub tokenizer: tokenizer::Style<'static>,
    /// Tokenizer keys the architecture's reader wants whatever the
    /// checkpoint says.
    pub tokenizer_overrides: Vec<(String, GgufValue)>,
}

/// A checkpoint to convert: where its files are, and the model's name.
pub struct Checkpoint {
    pub dir: PathBuf,
    pub name: String,
}

/// The files of a repository a conversion reads: the configuration, the
/// tokenizer, the chat template, the model card and the weights. Top-level
/// only; other formats of the same weights (ONNX, GGUF) are not fetched.
fn wanted(path: &str) -> bool {
    !path.contains('/')
        // Mistral's own copy of the same weights, under its own names.
        && !path.starts_with("consolidated")
        && (path.ends_with(".json")
            || path.ends_with(".jinja")
            || path.ends_with(".safetensors")
            || path == "README.md"
            || path == "tokenizer.model"
            || path == "merges.txt")
}

/// The models directory `orangu-server` serves from: `[orangu-server].models`
/// in `./orangu-server.conf`, else `~/.orangu/orangu-server.conf` — the
/// same files, in the same order, the server reads.
pub fn models_dir() -> Option<PathBuf> {
    let home = home::home_dir();
    let candidates = [
        Some(PathBuf::from("orangu-server.conf")),
        home.as_ref()
            .map(|h| h.join(".orangu").join("orangu-server.conf")),
    ];
    let conf = candidates.into_iter().flatten().find(|p| p.is_file())?;
    let text = std::fs::read_to_string(conf).ok()?;
    let sections = orangu::config::parse_ini_sections(&text).ok()?;
    let models = sections
        .get("orangu-server")?
        .get("models")?
        .trim()
        .to_string();
    match (models.strip_prefix("~/"), home) {
        (Some(rest), Some(home)) => Some(home.join(rest)),
        _ => Some(PathBuf::from(models)),
    }
}

/// `-m`'s argument as a checkpoint, when it is one: a directory holding a
/// `config.json`, or a Hugging Face repository (`owner/name`) — found in
/// the models directory, or downloaded into it, where `orangu-server list`
/// and a later conversion find it again. `None` for anything else, such as
/// a GGUF file.
pub fn locate(model: &Path) -> Result<Option<Checkpoint>> {
    if model.is_dir() {
        if !model.join("config.json").is_file() {
            return Ok(None);
        }
        // A snapshot directory inside the hub cache is named by its commit;
        // the model's name is the cache folder's.
        let name = model
            .ancestors()
            .filter_map(|a| a.file_name()?.to_str())
            .find_map(|n| {
                n.strip_prefix("models--")
                    .and_then(|r| r.rsplit("--").next())
            })
            .or_else(|| model.file_name()?.to_str())
            .unwrap_or("model")
            .to_string();
        return Ok(Some(Checkpoint {
            dir: model.to_path_buf(),
            name,
        }));
    }
    let Some(spec) = model.to_str() else {
        return Ok(None);
    };
    let is_repo = !model.exists()
        && spec.split('/').count() == 2
        && spec
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || "/-_.".contains(c));
    if !is_repo {
        return Ok(None);
    }
    let models = models_dir().ok_or_else(|| {
        anyhow!(
            "{spec} is not a file here, and no models directory is configured to find or \
             download it in ([orangu-server].models in orangu-server.conf)"
        )
    })?;
    let folder = models.join(orangu::model_download::repo_folder_name(spec));
    let existing = std::fs::read_to_string(folder.join("refs").join("main"))
        .ok()
        .map(|commit| folder.join("snapshots").join(commit.trim()))
        .filter(|dir| dir.join("config.json").is_file());
    let dir = match existing {
        Some(dir) => dir,
        None => {
            println!("downloading {spec} into {}", models.display());
            orangu::model_download::download_repo_files(&models, spec, &wanted, None)?
        }
    };
    Ok(Some(Checkpoint {
        dir,
        name: spec.rsplit('/').next().unwrap_or(spec).to_string(),
    }))
}

/// `config.json`, with a multimodal checkpoint's `text_config` folded over
/// the top level the way the language model reads it.
fn read_config(dir: &Path) -> Result<Value> {
    let path = dir.join("config.json");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let mut config: Value =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    if let Some(text_config) = config.get("text_config").cloned()
        && let (Some(top), Some(inner)) = (config.as_object_mut(), text_config.as_object())
    {
        for (k, v) in inner {
            top.insert(k.clone(), v.clone());
        }
    }
    Ok(config)
}

/// The conversion for the checkpoint's architecture.
fn plan(config: &Value, weights: &SafeTensorsSet) -> Result<Conversion> {
    let hf = config["architectures"][0]
        .as_str()
        .ok_or_else(|| anyhow!("config.json names no architecture"))?;
    if let Some(conversion) = llama::plan(hf, config, weights)? {
        return Ok(conversion);
    }
    if let Some(conversion) = phi::plan(hf, config, weights)? {
        return Ok(conversion);
    }
    if let Some(conversion) = gemma::plan(hf, config, weights)? {
        return Ok(conversion);
    }
    if let Some(conversion) = qwen35::plan(hf, config, weights)? {
        return Ok(conversion);
    }
    if let Some(conversion) = gemma3::plan(hf, config, weights)? {
        return Ok(conversion);
    }
    if let Some(conversion) = bailing::plan(hf, config, weights)? {
        return Ok(conversion);
    }
    if let Some(conversion) = qwen4::plan(hf, config, weights)? {
        return Ok(conversion);
    }
    bail!("{hf} is not an architecture this tool converts")
}

/// The license a checkpoint's model card declares in its front matter.
fn card_license(dir: &Path) -> Option<String> {
    let card = std::fs::read_to_string(dir.join("README.md")).ok()?;
    let front = card.strip_prefix("---")?.split("\n---").next()?;
    front.lines().find_map(|line| {
        line.strip_prefix("license:")
            .map(|v| v.trim().trim_matches('"').to_string())
            .filter(|v| !v.is_empty())
    })
}

/// The sampling defaults `generation_config.json` recommends, as the
/// `general.sampling.*` keys a server starts a request from.
fn sampling(dir: &Path) -> Vec<(String, GgufValue)> {
    let Ok(text) = std::fs::read_to_string(dir.join("generation_config.json")) else {
        return Vec::new();
    };
    let Ok(config) = serde_json::from_str::<Value>(&text) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if let Some(k) = config["top_k"].as_u64() {
        out.push(("general.sampling.top_k".into(), GgufValue::I32(k as i32)));
    }
    if let Some(p) = config["top_p"].as_f64() {
        out.push(("general.sampling.top_p".into(), GgufValue::F32(p as f32)));
    }
    if let Some(t) = config["temperature"].as_f64() {
        out.push(("general.sampling.temp".into(), GgufValue::F32(t as f32)));
    }
    out
}

/// `-m <checkpoint dir>`: writes the checkpoint as one GGUF file.
///
/// Without `-o` the file goes to `<models>/orangu-converted/`, where
/// `orangu-server list` shows it and the server serves it by name.
pub fn run(
    checkpoint: &Checkpoint,
    output: Option<PathBuf>,
    ftype: Ftype,
    matrix: Option<&crate::imatrix::Loaded>,
) -> Result<PathBuf> {
    let started = Instant::now();
    let dir = checkpoint.dir.as_path();
    let config = read_config(dir)?;
    let weights = SafeTensorsSet::open_dir(dir)?;
    let mut conversion = plan(&config, &weights)?;
    // The order every published conversion has: computed tensors first,
    // then the checkpoint's own order, a stacked tensor where its last
    // source sits. A quantized file is listed block by block instead, the
    // order the mixture's counters run in.
    conversion.tensors.sort_by_cached_key(|m| {
        m.source
            .names()
            .into_iter()
            .filter_map(|n| weights.position(n))
            .max()
            .map(|(shard, name)| (shard + 1, name.to_string()))
            .unwrap_or_default()
    });
    if ftype.is_quantized() {
        quant::sort_for_quantizing(&mut conversion.tensors, |m| m.name.as_str());
    }

    let stem = crate::without_format(&checkpoint.name);
    let file_name = format!("{stem}-{}.gguf", ftype.name());
    let output = match output {
        Some(path) => path,
        None => match models_dir() {
            Some(models) => {
                let dir = models.join("orangu-converted");
                std::fs::create_dir_all(&dir)
                    .with_context(|| format!("creating {}", dir.display()))?;
                dir.join(file_name)
            }
            None => PathBuf::from(file_name),
        },
    };

    let mut metadata: Vec<(String, GgufValue)> = vec![
        (
            "general.architecture".into(),
            GgufValue::String(conversion.architecture.into()),
        ),
        ("general.type".into(), GgufValue::String("model".into())),
        ("general.name".into(), GgufValue::String(stem.clone())),
    ];
    metadata.extend(sampling(dir));
    if let Some(license) = card_license(dir) {
        metadata.push(("general.license".into(), GgufValue::String(license)));
    }
    metadata.extend(conversion.metadata.iter().cloned());
    metadata.push((
        "general.file_type".into(),
        GgufValue::U32(ftype.file_type()),
    ));
    let mut vocab = tokenizer::metadata(dir, &config, conversion.vocab_size, conversion.tokenizer)?;
    for (key, value) in &conversion.tokenizer_overrides {
        tokenizer::set(&mut vocab, key, value.clone());
    }
    metadata.extend(vocab);
    metadata.push(("general.quantization_version".into(), GgufValue::U32(2)));

    let model = quant::Model {
        imatrix: matrix.is_some(),
        ..quant::Model::from_metadata(&metadata)
    };
    // GGUF lists dimensions fastest-first, the reverse of the checkpoint's
    // row-major shape.
    let all_dims = conversion
        .tensors
        .iter()
        .map(|m| {
            let shape = m.shape(&weights)?;
            Ok(shape.iter().rev().map(|&d| d as u64).collect::<Vec<u64>>())
        })
        .collect::<Result<Vec<_>>>()?;
    // The source of a direct conversion is the checkpoint as a 16-bit file
    // would carry it, which is what a tensor the mixture does not quantize
    // keeps.
    let inputs: Vec<quant::Tensor<'_>> = conversion
        .tensors
        .iter()
        .zip(&all_dims)
        .map(|(m, dims)| quant::Tensor {
            name: &m.name,
            dims,
            source: quant::GGML_TYPE_BF16,
        })
        .collect();
    let mut plans = Vec::with_capacity(conversion.tensors.len());
    for ((mapped, dims), plan) in conversion
        .tensors
        .iter()
        .zip(all_dims.iter().cloned())
        .zip(quant::plan_tensors(ftype, &inputs, &model))
    {
        if let Some(from) = plan.fallback_from {
            println!(
                "note: {}: {} does not divide {}, writing {}",
                mapped.name,
                quant::type_name(from),
                dims[0],
                quant::type_name(plan.ggml_type)
            );
        }
        plans.push(TensorPlan {
            name: mapped.name.clone(),
            dims,
            ggml_type: plan.ggml_type,
        });
    }

    crate::imatrix::require(ftype, &plans, matrix)?;
    if let Some(matrix) = matrix {
        matrix.apply(&plans, &mut metadata)?;
    }
    println!(
        "source      {} ({}, {} blocks)",
        dir.display(),
        conversion.architecture,
        conversion.layers
    );
    println!("output      {} ({})\n", output.display(), ftype.name());

    let by_name: std::collections::HashMap<&str, &Mapped> = conversion
        .tensors
        .iter()
        .map(|m| (m.name.as_str(), m))
        .collect();
    let written = write::write_streamed(
        &output,
        &metadata,
        &plans,
        |plan| {
            let mapped = by_name[plan.name.as_str()];
            match &mapped.source {
                // Rows concatenated as they are: a shard at a time, never
                // the whole table.
                Source::Rows(names) if mapped.transforms.is_empty() => {
                    let weights = &weights;
                    Ok(write::Weights::Rows(Box::new(names.iter().map(move |n| {
                        weights
                            .tensor(n)
                            .map(|(values, _)| values)
                            .with_context(|| n.clone())
                    }))))
                }
                _ => mapped.values(&weights).map(write::Weights::All),
            }
        },
        &|plan| matrix.and_then(|m| m.for_plan(plan)),
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
        "\nwrote {} ({} bytes) in {}",
        output.display(),
        written,
        crate::train::format_duration(started.elapsed().as_secs_f64())
    );
    Ok(output)
}

/// What `transform` does to a tensor's shape.
fn reshape(transform: Transform, shape: &mut Vec<usize>) {
    let count: usize = shape.iter().product();
    match transform {
        Transform::Squeeze => shape.retain(|&d| d != 1),
        Transform::Column => *shape = vec![count, 1],
        Transform::Conv4d => {
            let kernel = *shape.last().unwrap_or(&1);
            *shape = vec![1, count / kernel.max(1), 1, kernel];
        }
        _ => {}
    }
}

/// Applies a [`Transform`] to a tensor's values.
fn apply(transform: Transform, mut values: Vec<f32>, shape: &mut Vec<usize>) -> Vec<f32> {
    match transform {
        Transform::None => values,
        Transform::PermuteRotary { heads } => permute_rotary(&values, shape, heads),
        Transform::AddOne => {
            values.iter_mut().for_each(|v| *v += 1.0);
            values
        }
        Transform::NegExp => {
            // Rounded once, from the exact value.
            values
                .iter_mut()
                .for_each(|v| *v = -(f64::from(*v).exp() as f32));
            values
        }
        Transform::Squeeze | Transform::Column | Transform::Conv4d => {
            reshape(transform, shape);
            values
        }
        Transform::Exp => {
            // Rounded once, from the exact value.
            values
                .iter_mut()
                .for_each(|v| *v = f64::from(*v).exp() as f32);
            values
        }
        Transform::TileHeads {
            skip,
            groups,
            per_group,
            head_dim,
            columns,
        } => tile_heads(&values, shape, skip, groups, per_group, head_dim, columns),
    }
}

/// [`Transform::TileHeads`]: head `g * per_group + j` of the checkpoint
/// becomes head `j * groups + g`.
fn tile_heads(
    values: &[f32],
    shape: &[usize],
    skip: usize,
    groups: usize,
    per_group: usize,
    head_dim: usize,
    columns: bool,
) -> Vec<f32> {
    let rows = shape.first().copied().unwrap_or(1);
    let width = values.len() / rows.max(1);
    let to = |i: usize| -> usize {
        if i < skip {
            return i;
        }
        let (head, within) = ((i - skip) / head_dim, (i - skip) % head_dim);
        let (g, j) = (head / per_group, head % per_group);
        skip + (j * groups + g) * head_dim + within
    };
    let mut out = vec![0.0f32; values.len()];
    if columns {
        for (row, out_row) in values.chunks_exact(width).zip(out.chunks_exact_mut(width)) {
            for (i, &v) in row.iter().enumerate() {
                out_row[to(i)] = v;
            }
        }
    } else {
        for (i, row) in values.chunks_exact(width).enumerate() {
            let t = to(i);
            out[t * width..(t + 1) * width].copy_from_slice(row);
        }
    }
    out
}

/// Within each head's block of rows, row `j * half + i` of the checkpoint
/// becomes row `2 * i + j`: the two halves the rotation pairs are
/// interleaved. A 1-D tensor (a bias) is one value per row.
fn permute_rotary(values: &[f32], shape: &[usize], heads: usize) -> Vec<f32> {
    let rows = shape[0];
    let width = values.len() / rows.max(1);
    let per_head = rows / heads;
    let half = per_head / 2;
    let mut out = vec![0.0f32; values.len()];
    for h in 0..heads {
        for i in 0..half {
            for j in 0..2 {
                let from = h * per_head + j * half + i;
                let to = h * per_head + 2 * i + j;
                out[to * width..(to + 1) * width]
                    .copy_from_slice(&values[from * width..(from + 1) * width]);
            }
        }
    }
    out
}

/// Reads a hyperparameter under the first of several spellings.
pub fn find<'a>(config: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    keys.iter().map(|k| &config[*k]).find(|v| !v.is_null())
}

pub fn find_usize(config: &Value, keys: &[&str]) -> Option<usize> {
    find(config, keys)
        .and_then(Value::as_u64)
        .map(|v| v as usize)
}

pub fn require_usize(config: &Value, keys: &[&str]) -> Result<usize> {
    find_usize(config, keys).ok_or_else(|| anyhow!("config.json has no {}", keys[0]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rotary_permutation_interleaves_each_heads_halves() {
        // Two heads of four rows, one column: [a0 a1 b0 b1] per head
        // becomes [a0 b0 a1 b1].
        let values: Vec<f32> = (0..8).map(|v| v as f32).collect();
        let out = permute_rotary(&values, &[8, 1], 2);
        assert_eq!(out, vec![0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]);
        // Whole rows move together.
        let wide: Vec<f32> = (0..8).map(|v| v as f32).collect();
        let out = permute_rotary(&wide, &[4, 2], 1);
        assert_eq!(out, vec![0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]);
    }

    #[test]
    fn value_heads_are_tiled_in_rows_after_the_skipped_ones_or_in_columns() {
        // Two key heads serving two value heads each, one row a head, after
        // one skipped row: [s g0v0 g0v1 g1v0 g1v1] becomes
        // [s g0v0 g1v0 g0v1 g1v1].
        let tile = |skip, columns| Transform::TileHeads {
            skip,
            groups: 2,
            per_group: 2,
            head_dim: 1,
            columns,
        };
        let mut shape = vec![5];
        let rows = apply(tile(1, false), vec![9.0, 0.0, 1.0, 2.0, 3.0], &mut shape);
        assert_eq!(rows, vec![9.0, 0.0, 2.0, 1.0, 3.0]);
        // Columns move within every row.
        let mut shape = vec![2, 4];
        let values: Vec<f32> = (0..8).map(|v| v as f32).collect();
        let cols = apply(tile(0, true), values, &mut shape);
        assert_eq!(cols, vec![0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]);
    }

    #[test]
    fn the_value_transforms_squeeze_and_compute() {
        let mut shape = vec![3, 1, 2];
        let out = apply(Transform::Squeeze, vec![0.0; 6], &mut shape);
        assert_eq!((out.len(), shape), (6, vec![3, 2]));
        let mut shape = vec![2];
        let out = apply(Transform::NegExp, vec![0.0, 1.0], &mut shape);
        assert_eq!(out, vec![-1.0, -std::f32::consts::E]);
        let out = apply(Transform::AddOne, vec![0.5, -1.0], &mut shape);
        assert_eq!(out, vec![1.5, 0.0]);
    }

    #[test]
    fn half_takes_the_gate_or_the_up_rows_of_every_expert() {
        let dir = tempfile::tempdir().unwrap();
        // Two experts of four rows by one column: rows 0-1 gate, 2-3 up.
        let values: Vec<u8> = (0..8).flat_map(|v| (v as f32).to_le_bytes()).collect();
        let header = r#"{"e":{"dtype":"F32","shape":[2,4,1],"data_offsets":[0,32]}}"#.to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        bytes.extend(values);
        std::fs::write(dir.path().join("m.safetensors"), bytes).unwrap();
        let weights = SafeTensorsSet::open_dir(dir.path()).unwrap();
        let gate = Source::Half {
            name: "e".into(),
            second: false,
        };
        let up = Source::Half {
            name: "e".into(),
            second: true,
        };
        assert_eq!(gate.shape(&weights).unwrap(), vec![2, 2, 1]);
        assert_eq!(gate.read(&weights).unwrap(), vec![0.0, 1.0, 4.0, 5.0]);
        assert_eq!(up.read(&weights).unwrap(), vec![2.0, 3.0, 6.0, 7.0]);
        let stack = Source::Stack(vec!["e".into(), "e".into()]);
        assert_eq!(stack.shape(&weights).unwrap(), vec![2, 2, 4, 1]);
    }

    /// A checkpoint of F32 tensors, each `(name, shape)` filled with
    /// `0, 1, 2, …`.
    fn checkpoint(dir: &Path, tensors: &[(&str, &[usize])]) -> SafeTensorsSet {
        let mut header = serde_json::Map::new();
        let mut data: Vec<u8> = Vec::new();
        for (name, shape) in tensors {
            let count: usize = shape.iter().product();
            let begin = data.len();
            data.extend((0..count).flat_map(|v| (v as f32).to_le_bytes()));
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [begin, data.len()]}),
            );
        }
        let header = Value::Object(header).to_string();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        bytes.extend(data);
        std::fs::write(dir.join("m.safetensors"), bytes).unwrap();
        SafeTensorsSet::open_dir(dir).unwrap()
    }

    #[test]
    fn per_head_rows_split_a_fused_pair_and_rows_concatenate_shards() {
        let dir = tempfile::tempdir().unwrap();
        // Two heads of three rows (one key row, two value rows) by two
        // columns; and two shards of rows.
        let weights = checkpoint(
            dir.path(),
            &[("kv", &[6, 2]), ("s0", &[2, 3]), ("s1", &[1, 3])],
        );
        let key = Source::HeadRows {
            name: "kv".into(),
            heads: 2,
            offset: 0,
            count: 1,
            transpose: true,
        };
        assert_eq!(key.shape(&weights).unwrap(), vec![2, 2, 1]);
        assert_eq!(key.read(&weights).unwrap(), vec![0.0, 1.0, 6.0, 7.0]);
        let value = Source::HeadRows {
            name: "kv".into(),
            heads: 2,
            offset: 1,
            count: 2,
            transpose: false,
        };
        assert_eq!(value.shape(&weights).unwrap(), vec![2, 2, 2]);
        assert_eq!(
            value.read(&weights).unwrap(),
            vec![2.0, 3.0, 4.0, 5.0, 8.0, 9.0, 10.0, 11.0]
        );
        // Two rows of every head, transposed: each column's rows in turn.
        let both = Source::HeadRows {
            name: "kv".into(),
            heads: 2,
            offset: 1,
            count: 2,
            transpose: true,
        };
        assert_eq!(
            both.read(&weights).unwrap(),
            vec![2.0, 4.0, 3.0, 5.0, 8.0, 10.0, 9.0, 11.0]
        );
        let rows = Source::Rows(vec!["s0".into(), "s1".into()]);
        assert_eq!(rows.shape(&weights).unwrap(), vec![3, 3]);
        assert_eq!(
            rows.read(&weights).unwrap(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 2.0]
        );
    }

    #[test]
    fn a_rate_is_written_as_a_column_and_a_kernel_four_dimensional() {
        let mut shape = vec![3];
        let values = apply(Transform::Exp, vec![0.0, 1.0, -1.0], &mut shape);
        assert_eq!(values[0], 1.0);
        assert!((values[1] - std::f32::consts::E).abs() < 1e-6);
        apply(Transform::Column, values, &mut shape);
        assert_eq!(shape, vec![3, 1]);
        let mut shape = vec![8, 1, 4];
        apply(Transform::Conv4d, vec![0.0; 32], &mut shape);
        assert_eq!(shape, vec![1, 8, 1, 4]);
    }

    #[test]
    fn a_tensor_written_in_blocks_of_rows_is_the_tensor_written_whole() {
        let dir = tempfile::tempdir().unwrap();
        let plan = vec![TensorPlan {
            name: "t.weight".into(),
            dims: vec![256, 4],
            ggml_type: orangu::quantize::GGML_TYPE_Q4_K,
        }];
        let values: Vec<f32> = (0..1024)
            .map(|v| ((v * 7 % 29) as f32 - 14.0) * 0.01)
            .collect();
        let whole = dir.path().join("whole.gguf");
        let blocks = dir.path().join("blocks.gguf");
        write::write(&whole, &[], &plan, |_| Ok(values.clone()), &|_, _, _| {}).unwrap();
        let rows: Vec<Vec<f32>> = values.chunks(512).map(<[f32]>::to_vec).collect();
        write::write_streamed(
            &blocks,
            &[],
            &plan,
            |_| {
                Ok(write::Weights::Rows(Box::new(
                    rows.clone().into_iter().map(Ok),
                )))
            },
            &|_| None,
            &|_, _, _| {},
        )
        .unwrap();
        assert_eq!(
            std::fs::read(&whole).unwrap(),
            std::fs::read(&blocks).unwrap()
        );
        // A block that is not whole rows is refused.
        let ragged = write::write_streamed(
            &blocks,
            &[],
            &plan,
            |_| {
                Ok(write::Weights::Rows(Box::new(std::iter::once(Ok(vec![
                    0.0;
                    100
                ])))))
            },
            &|_| None,
            &|_, _, _| {},
        );
        assert!(ragged.is_err());
    }

    #[test]
    fn the_card_license_is_read_from_the_front_matter() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("README.md"),
            "---\nlibrary_name: transformers\nlicense: apache-2.0\n---\n# Model\nlicense: no\n",
        )
        .unwrap();
        assert_eq!(card_license(dir.path()).as_deref(), Some("apache-2.0"));
    }
}
