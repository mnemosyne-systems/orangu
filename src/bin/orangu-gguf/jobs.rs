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

//! Work divided over the gguf nodes of a channel, chunk by chunk.
//!
//! An importance matrix and a perplexity are both sums over chunks of a
//! text that run independently. Each chunk is measured from nothing, and the
//! chunks' results are added up in chunk order — on one machine as on
//! several — so the answer does not depend on which node ran which chunk or
//! in what order they finished. The top runs chunks itself and hands the
//! next one to whichever claimed node is free; a node that is lost has its
//! chunk run again elsewhere.
//!
//! Every node reads its own copy of the model, which must be the same file:
//! the top names it by path, size and a hash of its first bytes, and a node
//! looks for it at that path and then under its models directory.

use anyhow::{Context, Result, anyhow, bail};
use orangu::engine::calibrate::{self, Measurer};
use orangu::engine::importance::Entry;
use orangu::workers::protocol::{Message, read_frame, write_message};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::node::Claimed;

/// How much of a model file its identity hashes.
const IDENTITY_BYTES: u64 = 16 << 20;

/// A model file, as a top names it to a node.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelRef {
    pub path: String,
    pub size: u64,
    pub digest: [u8; 32],
}

impl ModelRef {
    pub fn of(path: &Path) -> Result<Self> {
        let size = std::fs::metadata(path)
            .with_context(|| format!("reading {}", path.display()))?
            .len();
        Ok(Self {
            path: path.display().to_string(),
            size,
            digest: digest(path, size)?,
        })
    }

    /// This node's copy: at the same path, or by file name under its models
    /// directory, with the same size and first bytes.
    pub fn locate(&self) -> Result<PathBuf> {
        let wanted = PathBuf::from(&self.path);
        let mut candidates = vec![wanted.clone()];
        if let (Some(name), Some(models)) = (wanted.file_name(), crate::hf::models_dir()) {
            find_named(&models, name, 6, &mut candidates);
        }
        for c in candidates {
            if std::fs::metadata(&c).is_ok_and(|m| m.len() == self.size)
                && digest(&c, self.size).is_ok_and(|d| d == self.digest)
            {
                return Ok(c);
            }
        }
        bail!(
            "this node has no copy of {} ({} bytes): put the same file at that path or under \
             its models directory",
            self.path,
            self.size
        )
    }

    pub fn encode(&self, w: &mut Bytes) {
        w.string(&self.path);
        w.u64(self.size);
        w.raw(&self.digest);
    }

    fn decode(r: &mut Reader<'_>) -> Result<Self> {
        Ok(Self {
            path: r.string()?,
            size: r.u64()?,
            digest: r.fixed()?,
        })
    }
}

fn digest(path: &Path, size: u64) -> Result<[u8; 32]> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut buf = vec![0u8; size.min(IDENTITY_BYTES) as usize];
    file.read_exact(&mut buf)?;
    let mut h = Sha256::new();
    h.update(size.to_le_bytes());
    h.update(&buf);
    Ok(h.finalize().into())
}

fn find_named(dir: &Path, name: &std::ffi::OsStr, depth: usize, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        match entry.file_type() {
            Ok(t) if t.is_dir() && depth > 0 => find_named(&path, name, depth - 1, out),
            Ok(_) if path.file_name() == Some(name) => out.push(path),
            _ => {}
        }
    }
}

/// A little-endian byte writer for job payloads.
#[derive(Default)]
pub struct Bytes(pub Vec<u8>);

impl Bytes {
    pub fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn raw(&mut self, v: &[u8]) {
        self.0.extend_from_slice(v);
    }
    pub fn string(&mut self, v: &str) {
        self.u64(v.len() as u64);
        self.raw(v.as_bytes());
    }
    /// A length and the bytes.
    pub fn blob(&mut self, v: &[u8]) {
        self.u64(v.len() as u64);
        self.raw(v);
    }
    /// A count and the floats.
    pub fn floats(&mut self, v: &[f32]) {
        self.u64(v.len() as u64);
        self.raw(bytemuck::cast_slice(v));
    }
    pub fn tokens(&mut self, v: &[u32]) {
        self.u64(v.len() as u64);
        for t in v {
            self.0.extend_from_slice(&t.to_le_bytes());
        }
    }
}

pub struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.0.len() < n {
            bail!("a job's payload is cut short");
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }
    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn fixed<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.take(N)?.try_into()?)
    }
    fn len(&mut self, each: usize) -> Result<usize> {
        let n = self.u64()? as usize;
        if n.saturating_mul(each) > self.0.len() {
            bail!("a job's payload is cut short");
        }
        Ok(n)
    }
    fn string(&mut self) -> Result<String> {
        let n = self.len(1)?;
        Ok(String::from_utf8(self.take(n)?.to_vec())?)
    }
    /// What [`Bytes::blob`] wrote.
    pub fn blob(&mut self) -> Result<&'a [u8]> {
        let n = self.len(1)?;
        self.take(n)
    }
    /// What [`Bytes::floats`] wrote.
    pub fn floats(&mut self) -> Result<Vec<f32>> {
        let n = self.len(4)?;
        Ok(self
            .take(n * 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    }
    pub fn tokens(&mut self) -> Result<Vec<u32>> {
        let n = self.len(4)?;
        Ok(self
            .take(n * 4)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect())
    }
}

/// One chunk's importance accumulators, as a node sends them.
pub fn encode_entries(stats: &BTreeMap<String, Entry>) -> Vec<u8> {
    let mut w = Bytes::default();
    w.u64(stats.len() as u64);
    for (name, e) in stats {
        w.string(name);
        w.u64(e.columns as u64);
        w.u64(e.sums.len() as u64);
        for &s in &e.sums {
            w.f64(s);
        }
        w.u64(e.counts.len() as u64);
        for &c in &e.counts {
            w.u64(c);
        }
    }
    w.0
}

pub fn decode_entries(data: &[u8]) -> Result<BTreeMap<String, Entry>> {
    let mut r = Reader(data);
    let n = r.len(1)?;
    let mut out = BTreeMap::new();
    for _ in 0..n {
        let name = r.string()?;
        let columns = r.u64()? as usize;
        let sums = (0..r.len(8)?).map(|_| r.f64()).collect::<Result<_>>()?;
        let counts = (0..r.len(8)?).map(|_| r.u64()).collect::<Result<_>>()?;
        out.insert(
            name,
            Entry {
                columns,
                sums,
                counts,
            },
        );
    }
    Ok(out)
}

/// What a window adds to a perplexity: the summed negative
/// log-probabilities of the tokens it scores, their squares, and how many.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Tally {
    pub nll: f64,
    pub nll2: f64,
    pub count: u64,
}

impl Tally {
    pub fn add(&mut self, other: Tally) {
        self.nll += other.nll;
        self.nll2 += other.nll2;
        self.count += other.count;
    }

    /// The perplexity and its standard error.
    pub fn perplexity(&self) -> (f64, f64) {
        let count = self.count.max(1) as f64;
        let mean = self.nll / count;
        let perplexity = mean.exp();
        let variance = self.nll2 / count - mean * mean;
        let error = if variance > 0.0 && self.count > 1 {
            (variance / (count - 1.0)).sqrt() * perplexity
        } else {
            0.0
        };
        (perplexity, error)
    }

    fn encode(&self) -> Vec<u8> {
        let mut w = Bytes::default();
        w.f64(self.nll);
        w.f64(self.nll2);
        w.u64(self.count);
        w.0
    }

    fn decode(data: &[u8]) -> Result<Self> {
        let mut r = Reader(data);
        Ok(Self {
            nll: r.f64()?,
            nll2: r.f64()?,
            count: r.u64()?,
        })
    }
}

/// A model loaded to score windows, on the host.
pub struct Scorer {
    model: Arc<dyn orangu::engine::arch::ModelForward>,
}

impl Scorer {
    pub fn open(path: &Path) -> Result<Self> {
        let loaded = orangu::engine::loader::LoadedModel::open(path)
            .with_context(|| format!("loading {}", path.display()))?;
        let backend: Arc<dyn orangu::engine::backend::Backend> =
            Arc::new(orangu::engine::backend::CpuBackend);
        Ok(Self {
            model: orangu::engine::build::model(&loaded, &backend)?,
        })
    }

    /// The tokens of `window` from position `first + 1` on, scored: each
    /// one's negative log-probability given everything before it.
    pub fn score(&self, window: &[u32], first: usize) -> Result<Tally> {
        let n = window.len();
        let mut cache = self.model.new_kv_cache(n);
        let rows = match self.model.forward_all_logits(&mut cache, window, 0, 0) {
            Ok(rows) => rows,
            Err(_) => {
                let mut cache = self.model.new_kv_cache(n);
                let mut rows = Vec::with_capacity(n);
                for (pos, &token) in window.iter().enumerate() {
                    rows.push(self.model.forward(&mut cache, &[token], pos, 0)?);
                }
                rows
            }
        };
        let mut tally = Tally::default();
        for (i, row) in rows.iter().enumerate().take(n - 1).skip(first) {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
            let sum: f64 = row.iter().map(|&v| (v as f64 - max).exp()).sum();
            let nll = max + sum.ln() - row[window[i + 1] as usize] as f64;
            tally.nll += nll;
            tally.nll2 += nll * nll;
            tally.count += 1;
        }
        Ok(tally)
    }
}

/// What a claimed node holds for its top between jobs.
#[derive(Default)]
pub enum Session {
    #[default]
    None,
    Imatrix(Measurer),
    Perplexity(Scorer),
    Quantize(Box<Source>),
    Train(Box<dyn crate::cotrain::Trainee>),
}

/// A source model a node encodes rows of, and the last tensor it read,
/// which the pieces of one tensor share.
pub struct Source {
    set: crate::shards::Source,
    requantize: bool,
    held: Option<(String, Vec<f32>)>,
}

/// Rows `first..first + count` of tensor `name` of a `rows`-row tensor,
/// encoded as `ggml_type`, fitted to `importance` when it is not empty.
#[derive(Clone, Copy)]
pub struct Piece<'a> {
    pub name: &'a str,
    pub ggml_type: u32,
    pub first: usize,
    pub count: usize,
    pub rows: usize,
    pub importance: &'a [f32],
}

impl Piece<'_> {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Bytes::default();
        w.string(self.name);
        w.u64(self.ggml_type as u64);
        w.u64(self.first as u64);
        w.u64(self.count as u64);
        w.u64(self.rows as u64);
        w.u64(self.importance.len() as u64);
        for v in self.importance {
            w.raw(&v.to_le_bytes());
        }
        w.0
    }

    /// The piece's bytes, from `values`: the whole tensor's rows.
    pub fn quantize(&self, values: &[f32]) -> Vec<u8> {
        let ncols = values.len() / self.rows.max(1);
        let rows = &values[self.first * ncols..(self.first + self.count) * ncols];
        if self.importance.is_empty() {
            orangu::quantize::encode(self.ggml_type, rows, ncols)
        } else {
            orangu::quantize::encode_weighted_rows(
                self.ggml_type,
                rows,
                ncols,
                self.importance,
                self.first,
                self.rows,
            )
        }
    }
}

/// Sends one job on `link` and waits for its answer.
pub fn call(link: &mut Claimed, name: &str, data: Vec<u8>) -> Result<Vec<u8>> {
    write_message(
        &mut *link.link.writer,
        0,
        &Message::Job {
            name: name.to_string(),
            data,
        },
    )?;
    match read_frame(&mut *link.link.reader)?.1 {
        Message::JobDone { data } => Ok(data),
        Message::Error(e) => Err(anyhow!("{}", e.message)),
        other => Err(anyhow!("it answered {other:?}")),
    }
}

/// A node's answer to one [`Message::Job`].
pub fn serve(session: &mut Session, name: &str, data: &[u8]) -> Result<Vec<u8>> {
    let mut r = Reader(data);
    if let Some(job) = name.strip_prefix("train-") {
        let mut held = match std::mem::take(session) {
            Session::Train(t) => Some(t),
            other => {
                *session = other;
                None
            }
        };
        let answer = crate::cotrain::serve(&mut held, job, &mut r);
        if let Some(t) = held {
            *session = Session::Train(t);
        }
        return answer;
    }
    match name {
        "open" => {
            let job = r.string()?;
            let model = ModelRef::decode(&mut r)?.locate()?;
            let chunk = r.u64()? as usize;
            *session = match job.as_str() {
                "imatrix" => Session::Imatrix(Measurer::open(&model, chunk)?),
                "perplexity" => Session::Perplexity(Scorer::open(&model)?),
                "quantize" => Session::Quantize(Box::new(Source {
                    set: crate::shards::Source::open(&model)?,
                    requantize: chunk != 0,
                    held: None,
                })),
                other => bail!("a gguf node does not take {other}"),
            };
            Ok(Vec::new())
        }
        // As many bytes back as asked for: how a top times the link.
        "echo" => Ok(vec![0; (r.u64()? as usize).min(64 << 20)]),
        "quantize" => match session {
            Session::Quantize(source) => {
                let name = r.string()?;
                let ggml_type = u32::try_from(r.u64()?)?;
                let first = r.u64()? as usize;
                let count = r.u64()? as usize;
                let rows = r.u64()? as usize;
                let n = r.len(4)?;
                let importance: Vec<f32> = r
                    .take(n * 4)?
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                if source.held.as_ref().is_none_or(|(held, _)| *held != name) {
                    source.held = None;
                    let i = source
                        .set
                        .index_of(&name)
                        .ok_or_else(|| anyhow!("{name} is not in this node's copy"))?;
                    let values = source.set.values(i, source.requantize)?;
                    source.held = Some((name.clone(), values));
                }
                let values = &source.held.as_ref().expect("held").1;
                if rows == 0 || values.len() % rows != 0 || first + count > rows {
                    bail!("rows {first}+{count} of {rows} do not fit {name}");
                }
                let ncols = values.len() / rows;
                let experts = importance.len() / ncols.max(1);
                if orangu::quantize::type_name(ggml_type) == "?"
                    || !ncols.is_multiple_of(orangu::quantize::block_size(ggml_type))
                    || (!importance.is_empty()
                        && (importance.len() != experts * ncols || !rows.is_multiple_of(experts)))
                {
                    bail!(
                        "{name} does not encode as {} with {} importances",
                        orangu::quantize::type_name(ggml_type),
                        importance.len()
                    );
                }
                Ok(Piece {
                    name: &name,
                    ggml_type,
                    first,
                    count,
                    rows,
                    importance: &importance,
                }
                .quantize(values))
            }
            _ => bail!("no model is open to quantize"),
        },
        "imatrix" => match session {
            Session::Imatrix(m) => Ok(encode_entries(&m.measure(&r.tokens()?)?)),
            _ => bail!("no model is open to measure"),
        },
        "perplexity" => match session {
            Session::Perplexity(s) => {
                let window = r.tokens()?;
                let first = r.u64()? as usize;
                Ok(s.score(&window, first)?.encode())
            }
            _ => bail!("no model is open to score"),
        },
        other => bail!("a gguf node does not take {other}"),
    }
}

/// The name [`run`] reports the top's own chunks under.
pub const HERE: &str = "this node";

/// Who ran how many chunks, for the log.
#[derive(Debug, Default)]
pub struct Report {
    pub by_node: BTreeMap<String, (usize, f64, u64)>,
    pub lost: Vec<String>,
}

impl Report {
    /// Adds another run's counts to these.
    pub fn absorb(&mut self, other: Report) {
        for (node, (chunks, seconds, bytes)) in other.by_node {
            let entry = self.by_node.entry(node).or_default();
            entry.0 += chunks;
            entry.1 += seconds;
            entry.2 += bytes;
        }
        self.lost.extend(other.lost);
    }

    pub fn print(&self, what: &str) {
        for (node, (chunks, seconds, bytes)) in &self.by_node {
            eprintln!(
                "{what}: {node} ran {chunks} chunks in {seconds:.1} s, sending {}",
                crate::bytes(*bytes)
            );
        }
        for node in &self.lost {
            eprintln!("{what}: {node} was lost; its chunk ran elsewhere");
        }
    }
}

struct Shared<T> {
    next: usize,
    retry: Vec<usize>,
    pending: BTreeMap<usize, T>,
    folded: usize,
    error: Option<anyhow::Error>,
    report: Report,
}

/// Runs chunks `0..n`: here with `local`, and on each of `links` as job
/// `job` with input `input(i)`, its answer read by `decode`. Each link is
/// first sent `open`, when given. Results are handed to `fold` in chunk
/// order, as soon as every earlier one is in. A link that is lost is
/// removed from `links`.
#[allow(clippy::too_many_arguments)]
pub fn run<T: Send>(
    n: usize,
    local: &(dyn Fn(usize) -> Result<T> + Sync),
    links: &mut Vec<Claimed>,
    job: &str,
    open: Option<&[u8]>,
    input: &(dyn Fn(usize) -> Vec<u8> + Sync),
    decode: &(dyn Fn(&[u8]) -> Result<T> + Sync),
    fold: &mut (dyn FnMut(usize, T) -> Result<()> + Send),
) -> Result<Report> {
    let shared = Mutex::new(Shared {
        next: 0,
        retry: Vec::new(),
        pending: BTreeMap::new(),
        folded: 0,
        error: None,
        report: Report::default(),
    });
    let fold = Mutex::new(fold);
    let pick = || -> Option<usize> {
        let mut s = shared.lock().unwrap();
        if s.error.is_some() {
            return None;
        }
        if let Some(i) = s.retry.pop() {
            return Some(i);
        }
        (s.next < n).then(|| {
            s.next += 1;
            s.next - 1
        })
    };
    let deliver = |who: &str, i: usize, t: T, seconds: f64, bytes: u64| {
        let mut s = shared.lock().unwrap();
        let entry = s.report.by_node.entry(who.to_string()).or_default();
        entry.0 += 1;
        entry.1 += seconds;
        entry.2 += bytes;
        s.pending.insert(i, t);
        let mut f = fold.lock().unwrap();
        while let Some(t) = {
            let at = s.folded;
            s.pending.remove(&at)
        } {
            let at = s.folded;
            if let Err(e) = f(at, t) {
                s.error.get_or_insert(e);
                return;
            }
            s.folded += 1;
        }
    };
    let lost = Mutex::new(vec![false; links.len()]);
    std::thread::scope(|scope| {
        for (l, link) in links.iter_mut().enumerate() {
            let pick = &pick;
            let deliver = &deliver;
            let shared = &shared;
            let lost = &lost;
            scope.spawn(move || {
                let who = link.found.name.clone();
                if let Some(open) = open
                    && let Err(e) = call(link, "open", open.to_vec())
                {
                    eprintln!("{job}: {who} left out: {e:#}");
                    lost.lock().unwrap()[l] = true;
                    return;
                }
                while let Some(i) = pick() {
                    let started = Instant::now();
                    let answer = call(link, job, input(i))
                        .and_then(|data| Ok((decode(&data)?, data.len() as u64)));
                    match answer {
                        Ok((t, bytes)) => {
                            deliver(&who, i, t, started.elapsed().as_secs_f64(), bytes)
                        }
                        Err(e) => {
                            let mut s = shared.lock().unwrap();
                            s.retry.push(i);
                            s.report.lost.push(format!("{who} ({e:#})"));
                            lost.lock().unwrap()[l] = true;
                            return;
                        }
                    }
                }
            });
        }
        while let Some(i) = pick() {
            let started = Instant::now();
            match local(i) {
                Ok(t) => deliver(HERE, i, t, started.elapsed().as_secs_f64(), 0),
                Err(e) => {
                    shared.lock().unwrap().error = Some(e.context(format!("chunk {}", i + 1)));
                    return;
                }
            }
        }
    });
    let lost = lost.into_inner().unwrap();
    let mut at = 0;
    links.retain(|_| {
        at += 1;
        !lost[at - 1]
    });
    let mut s = shared.into_inner().unwrap();
    if let Some(e) = s.error.take() {
        return Err(e);
    }
    // A node lost after the local loop ended leaves its chunk for here.
    while let Some(i) = s.retry.pop() {
        let t = local(i).with_context(|| format!("chunk {}", i + 1))?;
        s.pending.insert(i, t);
    }
    let f = fold.into_inner().unwrap();
    while let Some(t) = s.pending.remove(&s.folded) {
        f(s.folded, t)?;
        s.folded += 1;
    }
    if s.folded != n {
        bail!("{} of {n} chunks came back", s.folded);
    }
    Ok(s.report)
}

/// The gguf nodes of the channel this machine is configured for, claimed
/// for `job`; none without a channel.
pub fn claim_for(job: &str) -> Vec<Claimed> {
    let Ok(Some(settings)) = crate::node::Settings::load_configured() else {
        return Vec::new();
    };
    let own = crate::node::new_id();
    let found = crate::node::discover(&settings, &own);
    let claimed = crate::node::claim(&settings, &own, &found, job);
    let free = found
        .iter()
        .filter(|f| crate::node::claimable(f, job))
        .count();
    eprintln!(
        "{job}: channel '{}': {} gguf nodes free, {} claimed; {}",
        settings.name,
        free,
        claimed.len(),
        if claimed.is_empty() {
            "running alone".to_string()
        } else {
            format!("dividing the chunks over this node and {}", claimed.len())
        }
    );
    claimed
}

/// Measures an importance matrix over `text` into `out`, dividing the
/// chunks over `links` as well as this node.
pub fn imatrix(
    model: &Path,
    text: &Path,
    out: &Path,
    chunk: usize,
    mut links: Vec<Claimed>,
) -> Result<()> {
    let parts = calibrate::text_chunks(model, text, chunk, None)?;
    let measurer = Measurer::open(model, chunk)?;
    let mut open = Bytes::default();
    open.string("imatrix");
    ModelRef::of(model)?.encode(&mut open);
    open.u64(chunk as u64);
    let total = parts.len();
    let started = Instant::now();
    let mut stats = BTreeMap::new();
    let report = run(
        total,
        &|i| measurer.measure(&parts[i]),
        &mut links,
        "imatrix",
        Some(&open.0),
        &|i| {
            let mut w = Bytes::default();
            w.tokens(&parts[i]);
            w.0
        },
        &decode_entries,
        &mut |i, chunk_stats| {
            calibrate::fold(&mut stats, chunk_stats);
            eprintln!(
                "chunk {}/{total}: {:.1} s",
                i + 1,
                started.elapsed().as_secs_f64()
            );
            Ok(())
        },
    )?;
    report.print("imatrix");
    calibrate::finish(out, &stats, text, total, chunk)
}

/// The perplexity of `model` over `text`, in windows of `window` tokens
/// scored from the middle on, divided over `links` as well as this node.
pub fn perplexity(
    model: &Path,
    text: &Path,
    window: usize,
    mut links: Vec<Claimed>,
) -> Result<Tally> {
    let windows = calibrate::text_chunks(model, text, window, None)?;
    let scorer = Scorer::open(model)?;
    let first = window / 2;
    let mut open = Bytes::default();
    open.string("perplexity");
    ModelRef::of(model)?.encode(&mut open);
    open.u64(window as u64);
    let mut tally = Tally::default();
    let total = windows.len();
    let report = run(
        total,
        &|i| scorer.score(&windows[i], first),
        &mut links,
        "perplexity",
        Some(&open.0),
        &|i| {
            let mut w = Bytes::default();
            w.tokens(&windows[i]);
            w.u64(first as u64);
            w.0
        },
        &Tally::decode,
        &mut |i, t| {
            tally.add(t);
            let (p, _) = tally.perplexity();
            eprint!("\rperplexity {p:.4} after {} of {total} windows", i + 1);
            Ok(())
        },
    )?;
    eprintln!();
    report.print("perplexity");
    Ok(tally)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model4;
    use crate::node::{Settings, Worker, claim};

    fn settings(port: u16) -> Settings {
        Settings {
            name: "jobs".to_string(),
            secret: Some("s".to_string()),
            host: "127.0.0.1".to_string(),
            port,
            server_port: 0,
            networks: Vec::new(),
            peers: Vec::new(),
            scan: false,
            tls: None,
        }
    }

    /// The same perplexity, to the bit, on one, two and three nodes.
    #[test]
    fn a_perplexity_is_the_same_on_one_two_and_three_nodes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tiny.gguf");
        let model = model4::Model::new(model4::tests::tiny(), 5);
        model4::tests::write_f32(&model, &path);
        let vocab = model.cfg.vocab;
        // Enough windows that a node slow to open the model still gets some.
        let windows: Vec<Vec<u32>> = (0..28)
            .map(|w| (0..16).map(|i| ((w * 7 + i * 3) % vocab) as u32).collect())
            .collect();
        let scorer = Scorer::open(&path).unwrap();
        let mut open = Bytes::default();
        open.string("perplexity");
        ModelRef::of(&path).unwrap().encode(&mut open);
        open.u64(16);
        let top = settings(0);
        let at = |nodes: usize| -> (Tally, Report) {
            // Fresh workers each time: a released worker frees itself on its
            // own thread, after the top has moved on.
            let workers: Vec<_> = (0..nodes - 1)
                .map(|_| Worker::start(settings(0)).unwrap().1)
                .collect();
            let found: Vec<_> = workers
                .iter()
                .map(|addr| {
                    orangu::workers::channel::probe(
                        *addr,
                        &orangu::workers::channel::channel_hash("jobs", Some("s")),
                        None,
                    )
                    .unwrap()
                })
                .collect();
            let mut links = claim(&top, &format!("top{nodes}"), &found, "perplexity");
            assert_eq!(links.len(), nodes - 1);
            let mut tally = Tally::default();
            let report = run(
                windows.len(),
                &|i| {
                    // Slow enough here that every node takes chunks.
                    std::thread::sleep(std::time::Duration::from_millis(30));
                    scorer.score(&windows[i], 8)
                },
                &mut links,
                "perplexity",
                Some(&open.0),
                &|i| {
                    let mut w = Bytes::default();
                    w.tokens(&windows[i]);
                    w.u64(8);
                    w.0
                },
                &Tally::decode,
                &mut |_, t| {
                    tally.add(t);
                    Ok(())
                },
            )
            .unwrap();
            (tally, report)
        };
        let (one, _) = at(1);
        assert_eq!(one.count, 28 * 7);
        for nodes in [2, 3] {
            let (many, report) = at(nodes);
            assert_eq!(
                (many.nll.to_bits(), many.nll2.to_bits(), many.count),
                (one.nll.to_bits(), one.nll2.to_bits(), one.count),
                "{nodes} nodes: {report:?}"
            );
            assert!(report.lost.is_empty(), "{report:?}");
            assert_eq!(report.by_node.len(), nodes, "{report:?}");
        }
    }

    #[test]
    fn entries_survive_the_wire() {
        let mut stats = BTreeMap::new();
        stats.insert(
            "blk.0.ffn_down.weight".to_string(),
            Entry {
                columns: 2,
                sums: vec![1.5, f64::MIN_POSITIVE, 3.0, 0.25],
                counts: vec![7, 9],
            },
        );
        let back = decode_entries(&encode_entries(&stats)).unwrap();
        let e = &back["blk.0.ffn_down.weight"];
        assert_eq!(
            (e.columns, &e.sums, &e.counts),
            (2, &stats["blk.0.ffn_down.weight"].sums, &vec![7, 9])
        );
        assert!(decode_entries(&encode_entries(&stats)[..10]).is_err());
    }
}
