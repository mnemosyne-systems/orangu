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

//! One node's slice of a split model: the layers it runs itself, then the
//! [`Stage`]s it hands the rest to, in layer order.
//!
//! The same [`LayerPipeline`] runs at every level of a worker tree. At the
//! top it sits inside a [`DelegatingModel`], between the embedding and the
//! head; on a worker it sits inside a [`super::session::SessionStore`],
//! between the parent's `Forward` and its `ForwardResult`. A stage that is
//! itself a worker with workers of its own is what makes the tree a
//! pyramid, and nothing here needs to know how deep it goes.

use super::protocol::{ErrorCode, Rows, WorkerError};
use crate::engine::arch::{DecodeRow, ModelForward};
use crate::engine::kv_cache::{KvCache, LayerHold, RemoteLayers, RemoteSession};
use crate::engine::loader::ModelConfig;
use anyhow::{Result, ensure};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

/// A consecutive range of layers run somewhere else.
pub trait Stage: Send + Sync {
    fn layers(&self) -> Range<usize>;

    /// Who runs these layers, for traces: the worker's address.
    fn name(&self) -> String;

    /// Runs this stage's layers over `hidden` (`tokens.len()` rows entering
    /// `layers().start`, at positions from `start_pos`) for `session`, and
    /// returns the rows `rows` asks for as they leave `layers().end - 1`.
    fn forward(
        &self,
        session: u64,
        hidden: Vec<f32>,
        tokens: &[u32],
        start_pos: usize,
        rows: Rows,
    ) -> Result<Vec<f32>>;

    /// `session`'s rows past `len` are to be rolled back before its next
    /// forward.
    fn truncate(&self, session: u64, len: usize);

    /// `session` is over.
    fn release(&self, session: u64);

    /// Starts session `to` with a copy of `from`'s first `len` positions,
    /// here and below. A stage that cannot says so.
    fn fork(&self, from: u64, to: u64, len: usize) -> Result<()> {
        let _ = (from, to, len);
        anyhow::bail!("{} cannot copy a session", self.name())
    }

    /// What layer `layer` holds of `session`'s first `len` positions — its
    /// rows, or a recurrent layer's state — from whichever node at or below
    /// this stage runs it. A stage that cannot says so.
    fn layer_rows(&self, session: u64, layer: usize, len: usize) -> Result<LayerHold> {
        let _ = (session, layer, len);
        anyhow::bail!("{} cannot send rows back", self.name())
    }

    /// Several sessions' forwards at once — one decode step of every slot —
    /// each answered on its own. The default sends them one by one; a
    /// stage behind a connection sends them as one message. An `Err` is
    /// the stage itself failing, which none of the items got past.
    fn forward_batch(&self, items: Vec<BatchItem>) -> Result<Vec<Result<Vec<f32>, WorkerError>>> {
        Ok(items
            .into_iter()
            .map(|item| {
                self.forward(
                    item.session,
                    item.hidden,
                    &item.tokens,
                    item.start_pos,
                    item.rows,
                )
                .map_err(|e| worker_error(e, ErrorCode::Internal))
            })
            .collect())
    }
}

/// One session's part of a [`Stage::forward_batch`].
pub struct BatchItem {
    pub session: u64,
    pub hidden: Vec<f32>,
    pub tokens: Vec<u32>,
    pub start_pos: usize,
    pub rows: Rows,
}

/// One session's part of a [`LayerPipeline::run_batch`]: its cache, and
/// the forward to run on it.
pub struct BatchRow<'a> {
    pub cache: &'a mut KvCache,
    pub session: u64,
    pub hidden: Vec<f32>,
    pub tokens: Vec<u32>,
    pub start_pos: usize,
    pub rows: Rows,
}

/// `error` as the [`WorkerError`] it is, or as one of `code` when it is
/// some other failure.
pub fn worker_error(error: anyhow::Error, code: ErrorCode) -> WorkerError {
    match error.downcast::<WorkerError>() {
        Ok(error) => error,
        Err(error) => WorkerError::new(code, format!("{error:#}")),
    }
}

pub struct LayerPipeline {
    model: Arc<dyn ModelForward>,
    local: Range<usize>,
    stages: Vec<Arc<dyn Stage>>,
    /// This node's name in traces.
    name: String,
    /// Each sequence's queue of forwards still on their way through the
    /// stages — see [`Tail`].
    tails: Mutex<HashMap<u64, Tail>>,
}

/// `ORANGU_WORKERS_PIPELINE=0`: every forward waits for the whole tree, as
/// before prefill was pipelined, for comparison.
fn pipelining() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::engine::env::flag_on_unless_disabled("ORANGU_WORKERS_PIPELINE"))
}

/// `ORANGU_WORKERS_CHUNK`: the width, in tokens, a top-level node cuts a
/// prompt chunk into when it pipelines prefill through its tree; `0` keeps
/// the chunks the engine chose. See `DelegatingModel::run`.
fn sub_chunk() -> usize {
    static WIDTH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *WIDTH.get_or_init(|| {
        std::env::var("ORANGU_WORKERS_CHUNK")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(DEFAULT_SUB_CHUNK)
    })
}

/// [`sub_chunk`] when unset. Measured on a three-node tree of
/// Llama-3.2-3B, a 2041-token prompt: 54.1 s as the engine chunked it,
/// 48.3 s at 64, **47.8 s at 128**, 48.7 s at 256.
const DEFAULT_SUB_CHUNK: usize = 128;

type Reply = mpsc::SyncSender<Result<Vec<f32>, WorkerError>>;

enum Job {
    Run {
        /// What enters the next stage — or the error an earlier stage met,
        /// carried on so the forward is still answered in its turn.
        hidden: Result<Vec<f32>, WorkerError>,
        tokens: Vec<u32>,
        start_pos: usize,
        rows: Rows,
        /// `None` for a forward nobody waits for (`Rows::None`).
        reply: Option<Reply>,
    },
    /// Answered once every job before it is done.
    Barrier(mpsc::SyncSender<()>),
    /// The sequence is over: release it at every stage, after everything
    /// queued before.
    Release,
}

/// One sequence's forwards on their way through this node's stages, in
/// order, on a thread of their own — what lets a prompt chunk nobody waits
/// for (`Rows::None`) return as soon as this node's own layers are done,
/// so they start on the next chunk while the workers below are still on
/// this one. A failure is kept (`poison`) and answers the sequence's next
/// forward: the chunk that failed has no one waiting to be told.
///
/// A released sequence skips whatever it still had queued: its request is
/// over — finished, or its client gone mid-prompt — and the parts nobody
/// will read would otherwise hold every worker below until they were done.
struct Tail {
    jobs: mpsc::Sender<Job>,
    queued: Arc<std::sync::atomic::AtomicUsize>,
    poison: Arc<Mutex<Option<WorkerError>>>,
    released: Arc<std::sync::atomic::AtomicBool>,
}

impl Tail {
    fn start(session: u64, stages: Vec<Arc<dyn Stage>>) -> Self {
        let (jobs, inbox) = mpsc::channel::<Job>();
        let queued = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let poison: Arc<Mutex<Option<WorkerError>>> = Arc::new(Mutex::new(None));
        let released = Arc::new(std::sync::atomic::AtomicBool::new(false));
        // One thread per stage, each handing what it made to the next: the
        // first stage is on a prompt's next part while the second is still
        // on this one. A single thread running every stage in turn left the
        // workers of a node taking turns — a flat tree of a top and two
        // workers read a prompt at 27 tok/s where a pyramid of the same
        // three nodes, whose middle node overlaps with its own worker, read
        // it at 43.
        let last = stages.len().saturating_sub(1);
        let mut inbox = Some(inbox);
        for (i, stage) in stages.into_iter().enumerate() {
            let input = inbox
                .take()
                .expect("each stage takes the inbox the one before left");
            let next = (i < last).then(|| {
                let (next, after) = mpsc::channel::<Job>();
                inbox = Some(after);
                next
            });
            let (count, failed, over) = (queued.clone(), poison.clone(), released.clone());
            let _ = std::thread::Builder::new()
                .name("orangu-workers-tail".to_string())
                .spawn(move || {
                    for job in input {
                        match job {
                            Job::Run {
                                hidden,
                                tokens,
                                start_pos,
                                rows,
                                reply,
                            } => {
                                let earlier = failed.lock().unwrap().clone();
                                let result = match (hidden, earlier) {
                                    (Err(error), _) | (Ok(_), Some(error)) => Err(error),
                                    (Ok(_), None) if over.load(Ordering::Relaxed) => {
                                        Err(WorkerError::new(
                                            ErrorCode::UnknownSession,
                                            "the sequence was released",
                                        ))
                                    }
                                    (Ok(x), None) => {
                                        let wanted = if i == last { rows } else { Rows::All };
                                        let x = stage
                                            .forward(session, x, &tokens, start_pos, wanted)
                                            .map_err(|e| worker_error(e, ErrorCode::ChildLost));
                                        if let Err(error) = &x {
                                            *failed.lock().unwrap() = Some(error.clone());
                                        }
                                        x
                                    }
                                };
                                match &next {
                                    Some(next) => {
                                        let _ = next.send(Job::Run {
                                            hidden: result,
                                            tokens,
                                            start_pos,
                                            rows,
                                            reply,
                                        });
                                    }
                                    None => {
                                        count.fetch_sub(1, Ordering::Relaxed);
                                        if let Some(reply) = reply {
                                            let _ = reply.send(result);
                                        }
                                    }
                                }
                            }
                            Job::Barrier(reply) => match &next {
                                Some(next) => {
                                    let _ = next.send(Job::Barrier(reply));
                                }
                                None => {
                                    let _ = reply.send(());
                                }
                            },
                            Job::Release => {
                                // After everything this stage had queued;
                                // the stages after it release in their turn.
                                stage.release(session);
                                if let Some(next) = &next {
                                    let _ = next.send(Job::Release);
                                }
                                break;
                            }
                        }
                    }
                });
        }
        Self {
            jobs,
            queued,
            poison,
            released,
        }
    }

    fn busy(&self) -> bool {
        self.queued.load(Ordering::Relaxed) > 0
    }
}

/// Where the time of a sequence's forwards went: per node, summed —
/// this node's own layers, and each worker's round trip (network and its
/// whole subtree).
#[derive(Default, Clone, Debug)]
pub struct Trace {
    pub forwards: usize,
    pub tokens: usize,
    pub parts: Vec<(String, std::time::Duration)>,
}

impl Trace {
    fn add(&mut self, name: &str, elapsed: std::time::Duration) {
        match self.parts.iter_mut().find(|(n, _)| n == name) {
            Some((_, total)) => *total += elapsed,
            None => self.parts.push((name.to_string(), elapsed)),
        }
    }

    /// `11 tokens in 1 forward (a:8400 3 ms, b:8400 25 ms)`.
    pub fn describe(&self, unit: &str) -> String {
        let parts = self
            .parts
            .iter()
            .map(|(name, t)| format!("{name} {:.0} ms", t.as_secs_f64() * 1e3))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "{} {unit} in {} forward{} ({parts})",
            self.tokens,
            self.forwards,
            if self.forwards == 1 { "" } else { "s" }
        )
    }
}

impl LayerPipeline {
    /// `local` first, then each stage, must cover consecutive layers with
    /// no gap or overlap, and `model` must be able to run a range of them.
    pub fn new(
        model: Arc<dyn ModelForward>,
        local: Range<usize>,
        stages: Vec<Box<dyn Stage>>,
    ) -> Result<Self> {
        ensure!(
            model.supports_layer_split(),
            "the {} architecture cannot be split across workers yet",
            model.config().architecture
        );
        ensure!(
            local.start <= local.end,
            "empty-backwards layer range {local:?}"
        );
        let mut end = local.end;
        for stage in &stages {
            let range = stage.layers();
            ensure!(
                range.start == end && range.start < range.end,
                "a stage covering layers {range:?} does not continue from layer {end}"
            );
            end = range.end;
        }
        ensure!(
            end <= model.config().n_layer,
            "layers up to {end} requested of a model with {}",
            model.config().n_layer
        );
        // Every place this pipeline hands over — where it starts, where
        // its own layers end, where each stage ends — has to be a place the
        // model can be cut.
        let n_layer = model.config().n_layer;
        let cuts = std::iter::once(local.start)
            .chain(std::iter::once(local.end))
            .chain(stages.iter().map(|stage| stage.layers().end));
        for at in cuts {
            ensure!(
                at == 0 || at == n_layer || model.split_allowed(at),
                "the {} model cannot be cut before layer {at}: a layer after it reads the KV \
                 cache of a layer before it",
                model.config().architecture
            );
        }
        Ok(Self {
            model,
            local,
            stages: stages.into_iter().map(Arc::from).collect(),
            name: "local".to_string(),
            tails: Mutex::new(HashMap::new()),
        })
    }

    /// A pipeline running `local` with no stages and none of
    /// [`Self::new`]'s checks — a placeholder for a node whose model cannot
    /// be split, which never runs it.
    pub(crate) fn unchecked(model: Arc<dyn ModelForward>, local: Range<usize>) -> Self {
        Self {
            model,
            local,
            stages: Vec::new(),
            name: "local".to_string(),
            tails: Mutex::new(HashMap::new()),
        }
    }

    /// The layers this node runs itself.
    pub fn local_layers(&self) -> Range<usize> {
        self.local.clone()
    }

    /// Every layer this pipeline runs, locally or through its stages.
    pub fn layers(&self) -> Range<usize> {
        let end = self
            .stages
            .last()
            .map_or(self.local.end, |s| s.layers().end);
        self.local.start..end
    }

    pub fn model(&self) -> &Arc<dyn ModelForward> {
        &self.model
    }

    /// A cache for `session`: rows for the local layers, and — when there
    /// are stages — the handle that rolls theirs back and releases them.
    pub fn new_cache(self: &Arc<Self>, session: u64, capacity: usize) -> KvCache {
        let mut cache = self
            .model
            .new_kv_cache_for_layers(self.local.clone(), capacity);
        if !self.stages.is_empty() {
            let owner: Arc<dyn RemoteLayers> = self.clone();
            cache.remote = Some(Box::new(RemoteSession::new(session, capacity, owner)));
        }
        cache
    }

    /// [`Self::new_cache`] with a session handle whatever the stages, whose
    /// rollbacks and release go to `owner` — which passes them on to this
    /// pipeline, and keeps its own books.
    fn new_cache_owned_by(
        &self,
        session: u64,
        capacity: usize,
        owner: Arc<dyn RemoteLayers>,
    ) -> KvCache {
        let mut cache = self
            .model
            .new_kv_cache_for_layers(self.local.clone(), capacity);
        cache.remote = Some(Box::new(RemoteSession::new(session, capacity, owner)));
        cache
    }

    /// Runs the local layers and then every stage over `hidden`. Every
    /// stage but the last is sent every row, since its layers need them all
    /// for their keys and values; `rows` applies to what comes out of the
    /// end.
    pub fn run(
        &self,
        cache: &mut KvCache,
        session: u64,
        hidden: Vec<f32>,
        tokens: &[u32],
        start_pos: usize,
        rows: Rows,
    ) -> Result<Vec<f32>> {
        self.run_traced(cache, session, hidden, tokens, start_pos, rows, None)
    }

    /// Names this node in traces.
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// [`Self::run`], adding where the time went to `trace`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_traced(
        &self,
        cache: &mut KvCache,
        session: u64,
        hidden: Vec<f32>,
        tokens: &[u32],
        start_pos: usize,
        rows: Rows,
        mut trace: Option<&mut Trace>,
    ) -> Result<Vec<f32>> {
        if let Some(error) = self.poisoned(session) {
            return Err(anyhow::Error::new(error));
        }
        let started = std::time::Instant::now();
        let mut x = if self.local.is_empty() {
            hidden
        } else {
            self.model
                .forward_layers(cache, hidden, tokens, self.local.clone(), start_pos)?
        };
        if let Some(trace) = trace.as_deref_mut()
            && !self.local.is_empty()
        {
            trace.add(&self.name, started.elapsed());
        }
        if let Some(remote) = cache.remote.as_mut() {
            remote.len = start_pos + tokens.len();
        }
        if let Some(trace) = trace.as_deref_mut() {
            trace.forwards += 1;
            trace.tokens += tokens.len();
        }
        if self.stages.is_empty() {
            return self.finish(x, rows);
        }
        // Through the sequence's queue when it has forwards still in it, or
        // when nobody waits for this one; straight through otherwise — a
        // decode step has nothing to overlap with.
        let queued = self
            .tails
            .lock()
            .unwrap()
            .get(&session)
            .is_some_and(Tail::busy);
        if queued || (rows == Rows::None && pipelining()) {
            let wait = rows != Rows::None;
            let (reply, answer) = mpsc::sync_channel(1);
            self.enqueue(
                session,
                Job::Run {
                    hidden: Ok(x),
                    tokens: tokens.to_vec(),
                    start_pos,
                    rows,
                    reply: wait.then_some(reply),
                },
            );
            if !wait {
                return Ok(Vec::new());
            }
            let started = std::time::Instant::now();
            let out = answer
                .recv()
                .map_err(|_| anyhow::anyhow!("a sequence's queue stopped"))?
                .map_err(anyhow::Error::new);
            if let Some(trace) = trace {
                trace.add("workers", started.elapsed());
            }
            return out;
        }
        let last = self.stages.len() - 1;
        for (i, stage) in self.stages.iter().enumerate() {
            let wanted = if i == last { rows } else { Rows::All };
            let started = std::time::Instant::now();
            x = stage.forward(session, x, tokens, start_pos, wanted)?;
            if let Some(trace) = trace.as_deref_mut() {
                trace.add(&stage.name(), started.elapsed());
            }
        }
        Ok(x)
    }

    /// What `rows` asks for out of `x`, the stream leaving this node's last
    /// layer: every row, the last, none — or, for the logits variants, those
    /// rows through the model's output head, which only the node running the
    /// final layer may apply.
    fn finish(&self, mut x: Vec<f32>, rows: Rows) -> Result<Vec<f32>> {
        let n_embd = self.model.config().n_embd;
        match rows {
            Rows::None => x.clear(),
            Rows::Last | Rows::LastLogits if x.len() > n_embd => {
                x.drain(..x.len() - n_embd);
            }
            _ => {}
        }
        if !rows.logits() {
            return Ok(x);
        }
        ensure!(
            self.local.end == self.model.config().n_layer,
            "logits asked of layers {:?}, which do not end the model",
            self.local
        );
        Ok(self.model.head(&x, x.len() / n_embd)?.concat())
    }

    /// The failure a queued forward of `session` left behind, if any.
    fn poisoned(&self, session: u64) -> Option<WorkerError> {
        self.tails
            .lock()
            .unwrap()
            .get(&session)
            .and_then(|tail| tail.poison.lock().unwrap().clone())
    }

    /// Queues `job` for `session`, starting its queue when it has none.
    fn enqueue(&self, session: u64, job: Job) {
        let mut tails = self.tails.lock().unwrap();
        let tail = tails
            .entry(session)
            .or_insert_with(|| Tail::start(session, self.stages.clone()));
        if matches!(job, Job::Run { .. }) {
            tail.queued.fetch_add(1, Ordering::Relaxed);
        }
        let _ = tail.jobs.send(job);
    }

    /// Waits until every forward queued for `session` is through.
    fn drain(&self, session: u64) {
        let jobs = self
            .tails
            .lock()
            .unwrap()
            .get(&session)
            .filter(|tail| tail.busy())
            .map(|tail| tail.jobs.clone());
        if let Some(jobs) = jobs {
            let (done, wait) = mpsc::sync_channel(1);
            if jobs.send(Job::Barrier(done)).is_ok() {
                let _ = wait.recv();
            }
        }
    }
}

impl LayerPipeline {
    /// [`Self::run`] for several sessions at once: the local layers row by
    /// row, then each stage once for all of them. A row that fails goes no
    /// further, and the others carry on; `trace`, when given, gets the time
    /// each row waited.
    pub fn run_batch(
        &self,
        rows: &mut [BatchRow<'_>],
        mut trace: Option<&mut Trace>,
    ) -> Vec<Result<Vec<f32>, WorkerError>> {
        for row in rows.iter() {
            self.drain(row.session);
        }
        let started = std::time::Instant::now();
        let mut out: Vec<Result<Vec<f32>, WorkerError>> = rows
            .iter_mut()
            .map(|row| {
                let hidden = std::mem::take(&mut row.hidden);
                if self.local.is_empty() {
                    return Ok(hidden);
                }
                self.model
                    .forward_layers(
                        row.cache,
                        hidden,
                        &row.tokens,
                        self.local.clone(),
                        row.start_pos,
                    )
                    .map_err(|e| worker_error(e, ErrorCode::Internal))
            })
            .collect();
        if let Some(trace) = trace.as_deref_mut()
            && !self.local.is_empty()
        {
            trace.add(&self.name, started.elapsed());
        }
        let last = self.stages.len().checked_sub(1);
        for (s, stage) in self.stages.iter().enumerate() {
            let live: Vec<usize> = (0..rows.len()).filter(|i| out[*i].is_ok()).collect();
            if live.is_empty() {
                break;
            }
            let items = live
                .iter()
                .map(|&i| BatchItem {
                    session: rows[i].session,
                    hidden: std::mem::replace(&mut out[i], Ok(Vec::new())).unwrap_or_default(),
                    tokens: rows[i].tokens.clone(),
                    start_pos: rows[i].start_pos,
                    rows: if Some(s) == last {
                        rows[i].rows
                    } else {
                        Rows::All
                    },
                })
                .collect();
            let started = std::time::Instant::now();
            match stage.forward_batch(items) {
                Ok(results) if results.len() == live.len() => {
                    for (&i, result) in live.iter().zip(results) {
                        out[i] = result;
                    }
                }
                Ok(results) => {
                    let error = WorkerError::new(
                        ErrorCode::Internal,
                        format!("{} answers to {} forwards", results.len(), live.len()),
                    );
                    for &i in &live {
                        out[i] = Err(error.clone());
                    }
                }
                Err(e) => {
                    let error = worker_error(e, ErrorCode::ChildLost);
                    for &i in &live {
                        out[i] = Err(error.clone());
                    }
                }
            }
            if let Some(trace) = trace.as_deref_mut() {
                trace.add(&stage.name(), started.elapsed());
            }
        }
        for (row, result) in rows.iter_mut().zip(out.iter_mut()) {
            let Ok(x) = result else {
                continue;
            };
            if let Some(remote) = row.cache.remote.as_mut() {
                remote.len = row.start_pos + row.tokens.len();
            }
            if self.stages.is_empty() {
                let x = std::mem::take(x);
                *result = self
                    .finish(x, row.rows)
                    .map_err(|e| worker_error(e, ErrorCode::Internal));
            }
        }
        if let Some(trace) = trace {
            trace.forwards += 1;
            trace.tokens += rows.iter().map(|r| r.tokens.len()).sum::<usize>();
        }
        out
    }
}

impl LayerPipeline {
    /// Starts session `to` on every stage with a copy of `from`'s first
    /// `len` positions — after whatever `from` still has queued.
    pub fn fork(&self, from: u64, to: u64, len: usize) -> Result<()> {
        self.drain(from);
        for stage in &self.stages {
            stage.fork(from, to, len)?;
        }
        Ok(())
    }

    /// Layer `layer`'s first `len` positions of `session`, from the stage
    /// that runs it, once what the sequence has queued is through.
    pub fn stage_rows(&self, session: u64, layer: usize, len: usize) -> Result<LayerHold> {
        self.drain(session);
        let stage = self
            .stages
            .iter()
            .find(|stage| stage.layers().contains(&layer))
            .ok_or_else(|| anyhow::anyhow!("no stage runs layer {layer}"))?;
        stage.layer_rows(session, layer, len)
    }
}

impl RemoteLayers for LayerPipeline {
    fn fork(&self, from: u64, to: u64, len: usize) -> bool {
        LayerPipeline::fork(self, from, to, len).is_ok()
    }

    fn truncate(&self, session: u64, len: usize) {
        for stage in &self.stages {
            stage.truncate(session, len);
        }
    }

    fn release(&self, session: u64) {
        // After whatever the sequence still has queued, so a release never
        // overtakes a forward of its own on the way to a worker.
        match self.tails.lock().unwrap().remove(&session) {
            Some(tail) => {
                tail.released.store(true, Ordering::Relaxed);
                let _ = tail.jobs.send(Job::Release);
            }
            None => {
                for stage in &self.stages {
                    stage.release(session);
                }
            }
        }
    }
}

/// Where a [`DelegatingModel`] gets the pipeline it runs on — fixed, or a
/// node's current plan, which changes when a worker is lost or comes back.
pub trait PipelineSource: Send + Sync {
    /// The pipeline to run on, and its generation: a number that changes
    /// whenever the pipeline does.
    fn current(&self) -> (Arc<LayerPipeline>, u64);

    /// A forward on the pipeline of `generation` failed because part of the
    /// tree went away. Plans again unless that already happened; returns
    /// whether there is a newer pipeline to retry on.
    fn recover(&self, generation: u64) -> bool;

    /// Whether a sequence decodes on this node alone once its prompt is
    /// through the tree: its rows come back from the workers at the
    /// first decode step, and the model runs every later forward here.
    fn decode_alone(&self) -> bool {
        false
    }

    /// A handover failed: sequences go on decoding through the tree, until
    /// a new plan decides again.
    fn handover_failed(&self) {}

    /// A finished request's decode steps took `parts` — per node, its
    /// subtree included — over `tokens` steps: what the plan's nodes really
    /// run at.
    fn observed_decode(&self, parts: &[(String, Duration)], tokens: usize) {
        let _ = (parts, tokens);
    }

    /// A finished request's prompt and decode, each a [`Trace`] of the
    /// tree as a whole: how fast the tree really is at both.
    fn observed_request(&self, prefill: &Trace, decode: &Trace) {
        let _ = (prefill, decode);
    }

    /// Whether the node running the model's final layer applies the output
    /// head and sends logits back, rather than this node.
    fn head_on_last(&self) -> bool {
        false
    }
}

/// One pipeline, for good.
#[cfg(test)]
pub struct FixedPipeline(pub Arc<LayerPipeline>);

#[cfg(test)]
impl PipelineSource for FixedPipeline {
    fn current(&self) -> (Arc<LayerPipeline>, u64) {
        (self.0.clone(), 0)
    }

    fn recover(&self, _generation: u64) -> bool {
        false
    }
}

/// What the top-level node remembers of a sequence, to rebuild it on a new
/// pipeline: every token its cache holds. How large the cache is travels
/// with its handle ([`RemoteSession::capacity`]).
struct SessionState {
    generation: u64,
    tokens: Vec<u32>,
    /// Its cache has room for every layer's rows, to decode here alone.
    full: bool,
    prefill: Trace,
    decode: Trace,
}

type Sessions = Arc<Mutex<HashMap<u64, SessionState>>>;

/// The owner of a top-level cache's remote rows: passes rollbacks and the
/// release on to the pipeline the cache was made for, and forgets the
/// session's tokens when it ends.
struct SessionOwner {
    pipeline: Arc<LayerPipeline>,
    sessions: Sessions,
    /// Told where each finished request's decode time went.
    source: Arc<dyn PipelineSource>,
}

impl RemoteLayers for SessionOwner {
    fn truncate(&self, session: u64, len: usize) {
        self.pipeline.truncate(session, len);
    }

    /// Both sequences must be on this owner's pipeline, and `from` must
    /// hold the tokens: `to` takes them as its own, to be rebuilt from.
    fn fork(&self, from: u64, to: u64, len: usize) -> bool {
        let tokens = {
            let sessions = self.sessions.lock().unwrap();
            match (sessions.get(&from), sessions.get(&to)) {
                (Some(source), Some(target))
                    if source.generation == target.generation && source.tokens.len() >= len =>
                {
                    source.tokens[..len].to_vec()
                }
                _ => return false,
            }
        };
        if let Err(error) = self.pipeline.fork(from, to, len) {
            log::debug!("orangu-server: workers: no shared prefix: {error:#}");
            return false;
        }
        if let Some(target) = self.sessions.lock().unwrap().get_mut(&to) {
            target.tokens = tokens;
        }
        true
    }

    fn release(&self, session: u64) {
        let state = self.sessions.lock().unwrap().remove(&session);
        self.pipeline.release(session);
        if let Some(mut state) = state {
            let _ = state.log_trace();
        }
    }

    /// Where the request's time went, logged as it ends: its session may
    /// live on, carried to the slot's next request.
    fn request_finished(&self, session: u64) -> Vec<(String, Duration)> {
        let traced = self
            .sessions
            .lock()
            .unwrap()
            .get_mut(&session)
            .map(SessionState::log_trace);
        let Some((parts, prefill, decode)) = traced else {
            return Vec::new();
        };
        if decode.tokens > 0 {
            self.source.observed_decode(&decode.parts, decode.tokens);
        }
        self.source.observed_request(&prefill, &decode);
        parts
    }
}

impl SessionState {
    /// Logs the time since the last log, if any forward ran, and starts
    /// counting again. Answers that time per part, prefill and decode
    /// together.
    fn log_trace(&mut self) -> (Vec<(String, Duration)>, Trace, Trace) {
        let (prefill, decode) = (
            std::mem::take(&mut self.prefill),
            std::mem::take(&mut self.decode),
        );
        if prefill.forwards + decode.forwards == 0 {
            return (Vec::new(), Trace::default(), Trace::default());
        }
        log::info!(
            "orangu-server: workers: prefill {}; decode {}",
            prefill.describe("tokens"),
            decode.describe("tokens")
        );
        let mut total = prefill.clone();
        for (name, elapsed) in &decode.parts {
            total.add(name, *elapsed);
        }
        (total.parts, prefill, decode)
    }
}

/// How many attempts a forward gets at recovering from a lost worker.
const RECOVERIES: usize = 2;

/// The tokens a rebuilt sequence is prefilled with, per forward.
const REPLAY_CHUNK: usize = 256;

/// Whether `error` is part of the tree going away — a lost worker, or one
/// that no longer holds the session (it restarted) — which a new plan can
/// get past, as opposed to a refusal it cannot.
fn recoverable(error: &anyhow::Error) -> bool {
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<WorkerError>())
        .is_some_and(|e| {
            matches!(
                e.code,
                ErrorCode::ChildLost | ErrorCode::UnknownSession | ErrorCode::PositionMismatch
            )
        })
}

/// Whether `error` is a worker not holding the session at all.
fn unknown_session(error: &anyhow::Error) -> bool {
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<WorkerError>())
        .is_some_and(|e| e.code == ErrorCode::UnknownSession)
}

/// The top-level node's model when `[workers]` spreads it over a tree:
/// the embedding and the head run here, and everything between them goes
/// through a [`LayerPipeline`] covering every layer. To `engine::generate`
/// it is an ordinary model.
///
/// **A lost worker costs a re-prefill, not the request.** When a forward
/// fails because part of the tree went away, the source plans again over
/// the workers still there; the sequence is rebuilt on the new pipeline by
/// running every token it held through it again, and the forward is
/// retried. Tokens already streamed stay streamed: the sequence continues
/// from exactly where it was.
pub struct DelegatingModel {
    model: Arc<dyn ModelForward>,
    source: Arc<dyn PipelineSource>,
    /// The part width pipelined prefill cuts a chunk into ([`sub_chunk`]).
    sub_chunk: usize,
    next_session: AtomicU64,
    sessions: Sessions,
    /// Forwards running now, and when the last one ended: whether the tree
    /// is in use ([`Self::quiet`]).
    in_flight: Arc<AtomicUsize>,
    last_forward: Arc<Mutex<Instant>>,
}

/// Counts a forward as running while it lives ([`DelegatingModel::quiet`]).
struct InFlight {
    count: Arc<AtomicUsize>,
    last: Arc<Mutex<Instant>>,
}

impl InFlight {
    fn new(model: &DelegatingModel) -> Self {
        model.in_flight.fetch_add(1, Ordering::Relaxed);
        Self {
            count: model.in_flight.clone(),
            last: model.last_forward.clone(),
        }
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        *self.last.lock().unwrap() = Instant::now();
        self.count.fetch_sub(1, Ordering::Relaxed);
    }
}

impl DelegatingModel {
    /// A delegating model over one fixed pipeline.
    #[cfg(test)]
    pub fn new(pipeline: Arc<LayerPipeline>) -> Result<Self> {
        let n_layer = pipeline.model().config().n_layer;
        ensure!(
            pipeline.layers() == (0..n_layer),
            "a delegating model must cover every layer, 0..{n_layer}, not {:?}",
            pipeline.layers()
        );
        let model = pipeline.model().clone();
        Ok(Self::with_source(model, Arc::new(FixedPipeline(pipeline))))
    }

    /// A delegating model over whatever `source` currently plans; every
    /// pipeline it hands out must cover every layer of `model`.
    pub fn with_source(model: Arc<dyn ModelForward>, source: Arc<dyn PipelineSource>) -> Self {
        Self {
            model,
            source,
            sub_chunk: sub_chunk(),
            // Session ids are only unique per top-level node, which is all
            // they need to be: a worker serves one parent.
            next_session: AtomicU64::new(1),
            sessions: Arc::new(Mutex::new(HashMap::new())),
            in_flight: Arc::new(AtomicUsize::new(0)),
            last_forward: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Cuts pipelined prompt chunks into `width`-token parts instead of
    /// [`sub_chunk`]'s; `0` keeps them whole.
    #[cfg(test)]
    pub fn with_sub_chunk(mut self, width: usize) -> Self {
        self.sub_chunk = width;
        self
    }

    /// Sequences currently held — running, or kept for a slot's next
    /// request.
    #[cfg(test)]
    pub fn active_sessions(&self) -> usize {
        self.sessions.lock().unwrap().len()
    }

    /// Whether no forward is running and none has for `period`: the time to
    /// plan again. A sequence held for a slot's next request does not keep
    /// the tree busy; if it is taken up after a new plan, it is rebuilt.
    pub fn quiet(&self, period: Duration) -> bool {
        self.in_flight.load(Ordering::Relaxed) == 0
            && self.last_forward.lock().unwrap().elapsed() >= period
    }

    fn cache_on(
        &self,
        pipeline: &Arc<LayerPipeline>,
        generation: u64,
        capacity: usize,
        tokens: Vec<u32>,
        full: bool,
    ) -> KvCache {
        let session = self.next_session.fetch_add(1, Ordering::Relaxed);
        self.sessions.lock().unwrap().insert(
            session,
            SessionState {
                generation,
                tokens,
                full,
                prefill: Trace::default(),
                decode: Trace::default(),
            },
        );
        let owner: Arc<dyn RemoteLayers> = Arc::new(SessionOwner {
            pipeline: pipeline.clone(),
            sessions: self.sessions.clone(),
            source: self.source.clone(),
        });
        let mut cache = pipeline.new_cache_owned_by(session, capacity, owner);
        if full {
            let mut every = self.model.new_kv_cache(capacity);
            every.remote = cache.remote.take();
            cache = every;
        }
        cache
    }

    /// Whether `cache` is served by the model here alone rather than through
    /// the tree:
    /// - its rows came back from the workers already ([`Self::hand_over`]);
    /// - it is a slot's conversation that was, taken up again: every layer
    ///   holds it here and the workers hold none of it;
    /// - or it is `decode`'s first step after a prompt the tree ran, and this
    ///   node decodes alone: its rows are brought back now.
    ///
    /// A handover that fails leaves the sequence on the tree.
    fn alone(&self, cache: &mut KvCache, start_pos: usize, decode: bool) -> bool {
        let Some(remote) = cache.remote.as_ref() else {
            return cache.whole;
        };
        let (full, fresh) = {
            let generation = self.source.current().1;
            let sessions = self.sessions.lock().unwrap();
            match sessions.get(&remote.id) {
                Some(state) => (state.full, state.generation == generation),
                None => return false,
            }
        };
        if cache.whole && remote.len < start_pos && cache.holds_every_layer(start_pos) {
            cache.remote = None;
            return true;
        }
        if !full {
            return false;
        }
        if !(decode
            && fresh
            && start_pos > 0
            && remote.len == start_pos
            && self.source.decode_alone())
        {
            return false;
        }
        match self.hand_over(cache, start_pos) {
            Ok(()) => true,
            Err(error) => {
                log::warn!(
                    "orangu-server: workers: could not bring a sequence's rows back to decode \
                     here, decoding through the tree: {error:#}"
                );
                self.source.handover_failed();
                false
            }
        }
    }

    /// Brings what every layer the workers run holds of `cache`'s sequence —
    /// its rows, or a recurrent layer's state — into `cache`, which has room
    /// for them ([`SessionState::full`]), and lets the workers' session go.
    fn hand_over(&self, cache: &mut KvCache, len: usize) -> Result<()> {
        let started = Instant::now();
        let session = cache.remote.as_ref().map_or(0, |r| r.id);
        let (pipeline, _) = self.source.current();
        let local = pipeline.local_layers();
        let n_layer = self.model.config().n_layer;
        let mut moved = Vec::new();
        let result = (|| {
            let mut bytes = 0usize;
            for layer in (0..n_layer).filter(|l| !local.contains(l)) {
                let hold = pipeline.stage_rows(session, layer, len)?;
                bytes += hold.bytes();
                let slot = self.model.cache_slot(layer);
                cache.set_layer_hold(slot, hold)?;
                moved.push(slot);
            }
            ensure!(
                cache.holds_every_layer(len),
                "this node does not hold {len} positions of its own layers"
            );
            Ok(bytes)
        })();
        match result {
            Ok(bytes) => {
                cache.remote = None;
                cache.whole = true;
                log::debug!(
                    "orangu-server: workers: {len} positions of {} layers ({}) brought back in \
                     {:.0} ms to decode here",
                    moved.len(),
                    orangu::format::format_bytes(bytes as u64),
                    started.elapsed().as_secs_f64() * 1e3
                );
                Ok(())
            }
            Err(error) => {
                for slot in moved {
                    cache.clear_layer(slot);
                }
                Err(error)
            }
        }
    }

    /// Replaces `cache` with one on `pipeline` holding the same first
    /// `start_pos` tokens, by running them through it again.
    fn rebuild(
        &self,
        cache: &mut KvCache,
        pipeline: &Arc<LayerPipeline>,
        generation: u64,
        start_pos: usize,
    ) -> Result<()> {
        let (old, capacity) = cache
            .remote
            .as_ref()
            .map(|r| (r.id, r.capacity))
            .ok_or_else(|| anyhow::anyhow!("a cache this model did not make"))?;
        let (tokens, full) = {
            let sessions = self.sessions.lock().unwrap();
            let state = sessions
                .get(&old)
                .ok_or_else(|| anyhow::anyhow!("a cache this model did not make"))?;
            ensure!(
                state.tokens.len() >= start_pos,
                "the sequence holds {} tokens, and the forward starts at {start_pos}",
                state.tokens.len()
            );
            (state.tokens[..start_pos].to_vec(), state.full)
        };
        *cache = self.cache_on(pipeline, generation, capacity, tokens.clone(), full);
        let session = cache.remote.as_ref().map_or(0, |r| r.id);
        for (i, chunk) in tokens.chunks(REPLAY_CHUNK).enumerate() {
            let hidden = self.model.embed(chunk)?;
            pipeline.run(cache, session, hidden, chunk, i * REPLAY_CHUNK, Rows::Last)?;
        }
        Ok(())
    }

    /// [`Self::run_once`], with a long enough chunk cut into
    /// [`sub_chunk`]-token parts when prefill is pipelined: every part but
    /// the last goes on through the tree without waiting (`Rows::None`), so
    /// a prompt the engine hands over in a few wide chunks still fills the
    /// pipeline — four chunks over three stages left a stage idle a third
    /// of the time. A forward that wants every row back is not cut.
    fn run(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        rows: Rows,
    ) -> Result<Vec<f32>> {
        let width = self.sub_chunk;
        if matches!(rows, Rows::All | Rows::AllLogits)
            || width == 0
            || tokens.len() <= width
            || !pipelining()
        {
            return self.run_once(cache, tokens, start_pos, rows);
        }
        let parts: Vec<&[u32]> = tokens.chunks(width).collect();
        let last = parts.len() - 1;
        let mut pos = start_pos;
        for (i, part) in parts.iter().enumerate() {
            if i < last {
                self.run_once(cache, part, pos, Rows::None)?;
                pos += part.len();
            } else {
                return self.run_once(cache, part, pos, rows);
            }
        }
        unreachable!("a chunk longer than a part has a last part")
    }

    fn run_once(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        rows: Rows,
    ) -> Result<Vec<f32>> {
        ensure!(!tokens.is_empty(), "a forward needs at least one token");
        super::clock::hold();
        let _running = InFlight::new(self);
        let mut attempt = 0;
        // Set when a worker no longer held the session: rebuilt once on the
        // same pipeline before the plan is doubted.
        let mut forgotten = false;
        loop {
            let (pipeline, generation) = self.source.current();
            let result = (|| {
                let stale = forgotten || {
                    let sessions = self.sessions.lock().unwrap();
                    cache
                        .remote
                        .as_ref()
                        .and_then(|r| sessions.get(&r.id))
                        .is_none_or(|state| state.generation != generation)
                };
                if stale {
                    self.rebuild(cache, &pipeline, generation, start_pos)?;
                }
                let session = cache.remote.as_ref().map_or(0, |r| r.id);
                let hidden = self.model.embed(tokens)?;
                let mut trace = Trace::default();
                let out = pipeline.run_traced(
                    cache,
                    session,
                    hidden,
                    tokens,
                    start_pos,
                    rows,
                    Some(&mut trace),
                )?;
                Ok((out, trace))
            })();
            match result {
                Ok((out, trace)) => {
                    if let Some(remote) = cache.remote.as_ref()
                        && let Some(state) = self.sessions.lock().unwrap().get_mut(&remote.id)
                    {
                        state.tokens.truncate(start_pos);
                        state.tokens.extend_from_slice(tokens);
                        let total = if tokens.len() > 1 {
                            &mut state.prefill
                        } else {
                            &mut state.decode
                        };
                        total.forwards += trace.forwards;
                        total.tokens += trace.tokens;
                        for (name, elapsed) in trace.parts {
                            total.add(&name, elapsed);
                        }
                    }
                    return Ok(out);
                }
                // A sequence a worker let go of — an idle one it evicted,
                // such as a slot's retained conversation taken up again
                // after a pause — needs its rows back, not a new plan: the
                // tree is as it was.
                Err(error) if !forgotten && unknown_session(&error) => {
                    forgotten = true;
                    log::info!(
                        "orangu-server: {error:#}; rebuilding the sequence ({start_pos} tokens)"
                    );
                }
                Err(error) if attempt < RECOVERIES && recoverable(&error) => {
                    forgotten = false;
                    attempt += 1;
                    log::warn!(
                        "orangu-server: {error:#}; planning the workers again and rebuilding the \
                         sequence ({start_pos} tokens)"
                    );
                    if !self.source.recover(generation) {
                        return Err(error);
                    }
                }
                Err(error) => return Err(error),
            }
        }
    }
}

impl ModelForward for DelegatingModel {
    fn config(&self) -> &ModelConfig {
        self.model.config()
    }

    /// Handed over, the model's own batch samples as it would; on the tree
    /// every row comes back as logits.
    fn forward_decode_batch_sampling(
        &self,
        rows: &mut [DecodeRow<'_>],
        tokens: &[u32],
        samples: &[Option<crate::engine::arch::GreedySampleParams<'_>>],
    ) -> Result<Option<Vec<crate::engine::arch::ForwardOutcome>>> {
        if rows.iter().all(|row| row.cache.whole) {
            return self
                .model
                .forward_decode_batch_sampling(rows, tokens, samples);
        }
        Ok(self.forward_decode_batch(rows, tokens)?.map(|logits| {
            logits
                .into_iter()
                .map(crate::engine::arch::ForwardOutcome::Logits)
                .collect()
        }))
    }

    /// One decode step of every slot, as one message per worker at every
    /// level of the tree. Declines (`None`, and the batcher steps each slot
    /// on its own) when a sequence has to be rebuilt first; and when any
    /// row fails, rolls every row back to where it was and declines too, so
    /// the one-by-one path — which recovers from a lost worker — runs the
    /// step from a clean start.
    fn forward_decode_batch(
        &self,
        rows: &mut [DecodeRow<'_>],
        tokens: &[u32],
    ) -> Result<Option<Vec<Vec<f32>>>> {
        // Handed over: the model's own batch. One still on the tree when
        // this node decodes alone goes one by one, to be handed over.
        if rows.iter().all(|row| row.cache.whole) {
            return self.model.forward_decode_batch(rows, tokens);
        }
        if self.source.decode_alone() || rows.iter().any(|row| row.cache.whole) {
            return Ok(None);
        }
        super::clock::hold();
        let _running = InFlight::new(self);
        let (pipeline, generation) = self.source.current();
        {
            let sessions = self.sessions.lock().unwrap();
            let fresh = rows.iter().all(|row| {
                row.cache
                    .remote
                    .as_ref()
                    .and_then(|r| sessions.get(&r.id))
                    .is_some_and(|state| state.generation == generation)
            });
            if !fresh {
                return Ok(None);
            }
        }
        let head_on_last = self.source.head_on_last();
        let mut batch = Vec::with_capacity(rows.len());
        for (row, &token) in rows.iter_mut().zip(tokens) {
            batch.push(BatchRow {
                session: row.cache.remote.as_ref().map_or(0, |r| r.id),
                hidden: self.model.embed(&[token])?,
                cache: &mut *row.cache,
                tokens: vec![token],
                start_pos: row.pos,
                rows: if head_on_last {
                    Rows::LastLogits
                } else {
                    Rows::Last
                },
            });
        }
        let mut trace = Trace::default();
        let results = pipeline.run_batch(&mut batch, Some(&mut trace));
        drop(batch);
        if let Some(error) = results.iter().find_map(|r| r.as_ref().err()) {
            log::warn!(
                "orangu-server: a batched decode step failed ({error}); stepping one by one"
            );
            for row in rows.iter_mut() {
                row.cache.truncate(row.pos)?;
            }
            return Ok(None);
        }
        let mut sessions = self.sessions.lock().unwrap();
        let mut logits = Vec::with_capacity(rows.len());
        for ((row, &token), result) in rows.iter().zip(tokens).zip(results) {
            if let Some(state) = row
                .cache
                .remote
                .as_ref()
                .and_then(|r| sessions.get_mut(&r.id))
            {
                state.tokens.truncate(row.pos);
                state.tokens.push(token);
                state.decode.forwards += 1;
                state.decode.tokens += 1;
                for (name, elapsed) in &trace.parts {
                    state.decode.add(name, *elapsed);
                }
            }
            let hidden = result.expect("every row succeeded");
            logits.push(if head_on_last {
                hidden
            } else {
                self.model.head(&hidden, 1)?.pop().unwrap_or_default()
            });
        }
        Ok(Some(logits))
    }

    fn n_trunk_layer(&self) -> usize {
        self.model.n_trunk_layer()
    }

    fn new_kv_cache(&self, capacity: usize) -> KvCache {
        let (pipeline, generation) = self.source.current();
        let full = self.source.decode_alone();
        self.cache_on(&pipeline, generation, capacity, Vec::new(), full)
    }

    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        slot_id: usize,
    ) -> Result<Vec<f32>> {
        if self.alone(cache, start_pos, tokens.len() == 1) {
            return self.model.forward(cache, tokens, start_pos, slot_id);
        }
        if self.source.head_on_last() {
            return self.run(cache, tokens, start_pos, Rows::LastLogits);
        }
        let last = self.run(cache, tokens, start_pos, Rows::Last)?;
        let mut logits = self.model.head(&last, 1)?;
        Ok(logits.pop().unwrap_or_default())
    }

    /// Every stage still runs — its keys and values are the point — but
    /// nothing comes back, and this returns as soon as this node's own
    /// layers are done: the chunk goes on through the tree in the
    /// background while the next one starts here (`Rows::None`).
    fn forward_no_logits(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        slot_id: usize,
    ) -> Result<()> {
        if self.alone(cache, start_pos, false) {
            return self
                .model
                .forward_no_logits(cache, tokens, start_pos, slot_id);
        }
        self.run(cache, tokens, start_pos, Rows::None).map(|_| ())
    }

    fn forward_all_logits(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        slot_id: usize,
    ) -> Result<Vec<Vec<f32>>> {
        if self.alone(cache, start_pos, false) {
            return self
                .model
                .forward_all_logits(cache, tokens, start_pos, slot_id);
        }
        if self.source.head_on_last() {
            let n_vocab = self.model.config().n_vocab;
            let all = self.run(cache, tokens, start_pos, Rows::AllLogits)?;
            return Ok(all.chunks(n_vocab).map(<[f32]>::to_vec).collect());
        }
        let all = self.run(cache, tokens, start_pos, Rows::All)?;
        self.model.head(&all, tokens.len())
    }

    /// An embeddings request's hidden states: every row back from the
    /// tree, then the model's final norm — what `forward_hidden_states_at`
    /// is on the model itself. Every model a tree splits is causal, so the
    /// engine runs a long input in chunks through a cache, the way it runs
    /// a prompt.
    fn hidden_states_are_causal(&self) -> bool {
        self.model.hidden_states_are_causal()
    }

    fn forward_hidden_states_at(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
    ) -> Result<Vec<f32>> {
        if self.alone(cache, start_pos, false) {
            return self
                .model
                .forward_hidden_states_at(cache, tokens, start_pos);
        }
        let mut x = self.run(cache, tokens, start_pos, Rows::All)?;
        self.model.final_norm(&mut x, tokens.len())?;
        Ok(x)
    }

    fn forward_hidden_states(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut cache = self.new_kv_cache(tokens.len().max(1));
        self.forward_hidden_states_at(&mut cache, tokens, 0)
    }

    fn post_pool_projection(&self, pooled: Vec<f32>) -> Result<Vec<f32>> {
        self.model.post_pool_projection(pooled)
    }
}

/// Test fixtures shared by the `workers` modules: a small Llama-shaped
/// model with random weights, written as a real GGUF so the loader and the
/// architecture code run exactly as they do on a downloaded model.
#[cfg(test)]
pub(crate) mod fixture {
    use crate::engine::arch::ModelForward;
    use crate::engine::arch::llama::LlamaModel;
    use crate::engine::arch::qwen35::Qwen35Model;
    use crate::engine::backend::CpuBackend;
    use crate::engine::loader::LoadedModel;
    use std::sync::Arc;

    pub const N_LAYER: usize = 4;
    pub const N_EMBD: usize = 32;
    pub const N_VOCAB: usize = 64;
    const N_HEAD: usize = 4;
    const N_HEAD_KV: usize = 2;
    const N_FF: usize = 64;
    pub const N_CTX: usize = 128;

    fn string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn meta_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
        string(buf, key);
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&value.to_le_bytes());
    }

    fn meta_f32(buf: &mut Vec<u8>, key: &str, value: f32) {
        string(buf, key);
        buf.extend_from_slice(&6u32.to_le_bytes());
        buf.extend_from_slice(&value.to_le_bytes());
    }

    /// How a fixture file departs from the standard one — what the model
    /// identity checks are tested against.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Variant {
        /// Every `ffn_down` stored as `F16` rather than `F32`: the same
        /// model at another quantization.
        pub f16_ffn_down: bool,
        /// One weight of this layer's `attn_q` changed: the same layout
        /// with different weights, like another release of a model.
        pub nudge_layer: Option<usize>,
        /// A `qwen35` model instead: gated-DeltaNet layers, each keeping a
        /// recurrent state, and an attention layer every
        /// [`HYBRID_INTERVAL`]-th.
        pub hybrid: bool,
    }

    /// The hybrid fixture's attention layers: every second one.
    const HYBRID_INTERVAL: usize = 2;
    /// Its attention head size.
    const HYBRID_HEAD_DIM: usize = 16;
    /// Its gated-DeltaNet layers: conv window, state size per head, key
    /// heads and value heads.
    const SSM_CONV: usize = 4;
    const SSM_STATE: usize = 8;
    const SSM_GROUPS: usize = 2;
    const SSM_V_HEADS: usize = 4;

    /// The GGUF bytes: `llama` (or `qwen35`) metadata and every tensor in
    /// `F32`, filled from a fixed-seed generator so every run builds the
    /// same model.
    fn gguf(variant: Variant) -> Vec<u8> {
        let head_dim = N_EMBD / N_HEAD;
        let kv_dim = N_HEAD_KV * head_dim;
        let e = N_EMBD as u64;
        // (name, dims with the contiguous one first, is a norm weight)
        let mut tensors: Vec<(String, Vec<u64>, bool)> = vec![
            ("token_embd.weight".into(), vec![e, N_VOCAB as u64], false),
            ("output_norm.weight".into(), vec![e], true),
            ("output.weight".into(), vec![e, N_VOCAB as u64], false),
        ];
        let ffn = [
            ("ffn_gate.weight", vec![e, N_FF as u64], false),
            ("ffn_up.weight", vec![e, N_FF as u64], false),
            ("ffn_down.weight", vec![N_FF as u64, e], false),
        ];
        for i in 0..N_LAYER {
            let layer: Vec<(&str, Vec<u64>, bool)> = if !variant.hybrid {
                vec![
                    ("attn_norm.weight", vec![e], true),
                    ("attn_q.weight", vec![e, e], false),
                    ("attn_k.weight", vec![e, kv_dim as u64], false),
                    ("attn_v.weight", vec![e, kv_dim as u64], false),
                    ("attn_output.weight", vec![e, e], false),
                    ("ffn_norm.weight", vec![e], true),
                ]
            } else if (i + 1) % HYBRID_INTERVAL == 0 {
                // A gated attention layer: each head's query and gate
                // interleaved in `attn_q`.
                let hd = HYBRID_HEAD_DIM as u64;
                vec![
                    ("attn_norm.weight", vec![e], true),
                    ("post_attention_norm.weight", vec![e], true),
                    ("attn_q.weight", vec![e, 2 * N_HEAD as u64 * hd], false),
                    ("attn_q_norm.weight", vec![hd], true),
                    ("attn_k.weight", vec![e, N_HEAD_KV as u64 * hd], false),
                    ("attn_k_norm.weight", vec![hd], true),
                    ("attn_v.weight", vec![e, N_HEAD_KV as u64 * hd], false),
                    ("attn_output.weight", vec![N_HEAD as u64 * hd, e], false),
                ]
            } else {
                // A gated-DeltaNet layer.
                let key = (SSM_STATE * SSM_GROUPS) as u64;
                let value = (SSM_STATE * SSM_V_HEADS) as u64;
                let conv = 2 * key + value;
                vec![
                    ("attn_norm.weight", vec![e], true),
                    ("post_attention_norm.weight", vec![e], true),
                    ("attn_qkv.weight", vec![e, conv], false),
                    ("attn_gate.weight", vec![e, value], false),
                    ("ssm_beta.weight", vec![e, SSM_V_HEADS as u64], false),
                    ("ssm_alpha.weight", vec![e, SSM_V_HEADS as u64], false),
                    ("ssm_conv1d.weight", vec![SSM_CONV as u64, conv], false),
                    ("ssm_dt.bias", vec![SSM_V_HEADS as u64], false),
                    ("ssm_a", vec![SSM_V_HEADS as u64], false),
                    ("ssm_norm.weight", vec![SSM_STATE as u64], true),
                    ("ssm_out.weight", vec![value, e], false),
                ]
            };
            for (suffix, dims, norm) in layer.into_iter().chain(ffn.iter().cloned()) {
                tensors.push((format!("blk.{i}.{suffix}"), dims, norm));
            }
        }

        let arch = if variant.hybrid { "qwen35" } else { "llama" };
        let mut meta = Vec::new();
        let mut n_meta = 1u64;
        let mut u32_kv = |key: &str, value: u32| {
            meta_u32(&mut meta, &format!("{arch}.{key}"), value);
            n_meta += 1;
        };
        u32_kv("embedding_length", N_EMBD as u32);
        u32_kv("block_count", N_LAYER as u32);
        u32_kv("attention.head_count", N_HEAD as u32);
        u32_kv("attention.head_count_kv", N_HEAD_KV as u32);
        u32_kv("context_length", N_CTX as u32);
        u32_kv("vocab_size", N_VOCAB as u32);
        if variant.hybrid {
            u32_kv("attention.key_length", HYBRID_HEAD_DIM as u32);
            u32_kv("attention.value_length", HYBRID_HEAD_DIM as u32);
            u32_kv("rope.dimension_count", HYBRID_HEAD_DIM as u32);
            u32_kv("full_attention_interval", HYBRID_INTERVAL as u32);
            u32_kv("ssm.conv_kernel", SSM_CONV as u32);
            u32_kv("ssm.state_size", SSM_STATE as u32);
            u32_kv("ssm.group_count", SSM_GROUPS as u32);
            u32_kv("ssm.time_step_rank", SSM_V_HEADS as u32);
            u32_kv("ssm.inner_size", (SSM_STATE * SSM_V_HEADS) as u32);
        }
        meta_f32(&mut meta, &format!("{arch}.rope.freq_base"), 10000.0);
        meta_f32(
            &mut meta,
            &format!("{arch}.attention.layer_norm_rms_epsilon"),
            1e-5,
        );
        n_meta += 2;

        let mut buf = Vec::new();
        buf.extend_from_slice(b"GGUF");
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        buf.extend_from_slice(&n_meta.to_le_bytes());
        string(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes());
        string(&mut buf, arch);
        buf.extend_from_slice(&meta);

        let mut data = Vec::new();
        let mut state = 0x2545_f491_4f6c_dd1du64;
        for (name, dims, norm) in &tensors {
            string(&mut buf, name);
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            let f16 = variant.f16_ffn_down && name.ends_with("ffn_down.weight");
            let nudged = variant
                .nudge_layer
                .is_some_and(|il| *name == format!("blk.{il}.attn_q.weight"));
            // F32 or F16.
            buf.extend_from_slice(&u32::from(f16).to_le_bytes());
            buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
            let n: u64 = dims.iter().product();
            for i in 0..n {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                let mut value = if *norm { 1.0 + unit * 0.2 } else { unit * 0.4 };
                // A recurrent state decays: its `A` is negative.
                if name.ends_with("ssm_a") {
                    value = -0.5 - unit.abs();
                }
                if nudged && i == n / 2 {
                    value += 0.01;
                }
                if f16 {
                    data.extend_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
                } else {
                    data.extend_from_slice(&value.to_le_bytes());
                }
            }
            while !data.len().is_multiple_of(32) {
                data.push(0);
            }
        }
        while !buf.len().is_multiple_of(32) {
            buf.push(0);
        }
        buf.extend_from_slice(&data);
        buf
    }

    /// The fixture file for `variant`, written once per test process.
    pub fn path(variant: Variant) -> std::path::PathBuf {
        static PATHS: std::sync::Mutex<Vec<(Variant, std::path::PathBuf)>> =
            std::sync::Mutex::new(Vec::new());
        let mut paths = PATHS.lock().unwrap();
        if let Some((_, path)) = paths.iter().find(|(v, _)| *v == variant) {
            return path.clone();
        }
        // Named by its content, so every test process shares one file per
        // variant rather than leaving its own behind; written aside and
        // renamed into place, so a process never reads another's half.
        use std::hash::{Hash, Hasher};
        let bytes = gguf(variant);
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        bytes.hash(&mut hasher);
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "orangu-workers-fixture-{:016x}.gguf",
            hasher.finish()
        ));
        if std::fs::metadata(&path).map(|m| m.len()).ok() != Some(bytes.len() as u64) {
            let aside = dir.join(format!(
                "orangu-workers-fixture-{}-{}.part",
                std::process::id(),
                paths.len()
            ));
            std::fs::write(&aside, &bytes).expect("write the fixture model");
            std::fs::rename(&aside, &path).expect("put the fixture model in place");
        }
        paths.push((variant, path.clone()));
        path
    }

    /// The loaded fixture file for `variant`.
    pub fn loaded(variant: Variant) -> LoadedModel {
        LoadedModel::open(&path(variant)).expect("load the fixture model")
    }

    /// The fixture model on the CPU backend.
    pub fn model() -> Arc<dyn ModelForward> {
        model_for(Variant::default())
    }

    /// The fixture model of `variant` on the CPU backend.
    pub fn model_for(variant: Variant) -> Arc<dyn ModelForward> {
        let loaded = loaded(variant);
        if variant.hybrid {
            return Arc::new(
                Qwen35Model::load_with_backend(&loaded, Arc::new(CpuBackend))
                    .expect("build the hybrid fixture model"),
            );
        }
        Arc::new(
            LlamaModel::load_with_backend(&loaded, Arc::new(CpuBackend))
                .expect("build the fixture model"),
        )
    }

    /// A prompt and a few decode tokens, all inside the vocabulary.
    pub fn tokens() -> Vec<u32> {
        vec![1, 17, 42, 5, 63, 8, 30, 2]
    }
}

#[cfg(test)]
mod tests {
    use super::fixture::{self, N_CTX, N_EMBD, N_LAYER};
    use super::*;
    use crate::workers::protocol::ActivationFormat;
    use crate::workers::session::SessionStore;
    use crate::workers::stage::LoopbackStage;

    /// Prefill of `prompt` tokens, then one decode step per remaining token,
    /// collecting every step's logits.
    fn run(model: &dyn ModelForward, tokens: &[u32], prompt: usize) -> Vec<Vec<f32>> {
        let mut cache = model.new_kv_cache(N_CTX);
        let mut out = vec![model.forward(&mut cache, &tokens[..prompt], 0, 0).unwrap()];
        for (pos, token) in tokens.iter().enumerate().skip(prompt) {
            out.push(model.forward(&mut cache, &[*token], pos, 0).unwrap());
        }
        out
    }

    fn worker(
        model: &Arc<dyn ModelForward>,
        layers: Range<usize>,
        format: ActivationFormat,
        children: Vec<Box<dyn Stage>>,
    ) -> Box<dyn Stage> {
        let local = layers.start..children.first().map_or(layers.end, |c| c.layers().start);
        let pipeline = Arc::new(LayerPipeline::new(model.clone(), local, children).unwrap());
        assert_eq!(pipeline.layers(), layers);
        Box::new(LoopbackStage::new(
            Arc::new(SessionStore::new(pipeline, N_CTX, 8)),
            format,
        ))
    }

    /// Whether `copies` copies of the model at `path` fit the memory
    /// available now — a GPU that shares the host's memory holds its own copy
    /// of the weights beside the mapped file. Says why when they do not, so a
    /// test stops there instead of being killed for memory (gemma-4-31B on a
    /// 30 GiB board).
    fn fits_in_memory(path: &str, copies: u64) -> bool {
        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let Ok(mut system) = sysinfo::System::new() else {
            return true;
        };
        system.refresh_memory();
        let available = system.available_memory();
        if size.saturating_mul(copies) > available {
            println!(
                "{path}: {copies} copies of {:.1} GiB do not fit the {:.1} GiB available; nothing \
                 to check",
                size as f64 / (1u64 << 30) as f64,
                available as f64 / (1u64 << 30) as f64
            );
            return false;
        }
        true
    }

    fn delegating(local: Range<usize>, stages: Vec<Box<dyn Stage>>) -> DelegatingModel {
        let model = fixture::model();
        DelegatingModel::new(Arc::new(LayerPipeline::new(model, local, stages).unwrap())).unwrap()
    }

    /// Cutting the model at any layer boundary — embed, the two halves,
    /// head — computes exactly what one pass does, for a prefill and for
    /// the decode steps after it.
    #[test]
    fn the_pieces_compute_what_one_pass_does_at_every_split() {
        let model = fixture::model();
        let tokens = fixture::tokens();
        let whole = run(model.as_ref(), &tokens, 5);
        for split in 0..=N_LAYER {
            let mut first = model.new_kv_cache_for_layers(0..split, N_CTX);
            let mut second = model.new_kv_cache_for_layers(split..N_LAYER, N_CTX);
            let mut step = |chunk: &[u32], pos: usize| {
                let x = model.embed(chunk).unwrap();
                let x = model
                    .forward_layers(&mut first, x, chunk, 0..split, pos)
                    .unwrap();
                let x = model
                    .forward_layers(&mut second, x, chunk, split..N_LAYER, pos)
                    .unwrap();
                let last = &x[(chunk.len() - 1) * N_EMBD..];
                model.head(last, 1).unwrap().pop().unwrap()
            };
            let mut pieces = vec![step(&tokens[..5], 0)];
            for pos in 5..tokens.len() {
                pieces.push(step(&tokens[pos..pos + 1], pos));
            }
            assert_eq!(pieces, whole, "split at layer {split}");
        }
    }

    /// A slice's cache holds rows for its own layers and nothing else.
    #[test]
    fn a_slice_cache_holds_only_its_own_layers() {
        let model = fixture::model();
        let mut cache = model.new_kv_cache_for_layers(1..3, N_CTX);
        let x = model.embed(&[3, 4]).unwrap();
        let x = model
            .forward_layers(&mut cache, x, &[3, 4], 1..3, 0)
            .unwrap();
        assert_eq!(x.len(), 2 * N_EMBD);
        let lens: Vec<usize> = cache.layers.iter().map(|l| l.len).collect();
        assert_eq!(lens, [0, 2, 2, 0]);
        assert_eq!(cache.committed_len(), 2);
    }

    /// With `f32` on the wire, a model spread over a local slice and two
    /// workers answers bit for bit what it answers alone.
    #[test]
    fn one_level_of_workers_is_exact_in_f32() {
        let model = fixture::model();
        let tokens = fixture::tokens();
        let whole = run(model.as_ref(), &tokens, 5);
        let f32 = ActivationFormat::F32;
        let split = delegating(
            0..1,
            vec![
                worker(&model, 1..3, f32, vec![]),
                worker(&model, 3..4, f32, vec![]),
            ],
        );
        assert_eq!(run(&split, &tokens, 5), whole);
        // And with nothing kept at the top at all.
        let split = delegating(0..0, vec![worker(&model, 0..4, f32, vec![])]);
        assert_eq!(run(&split, &tokens, 5), whole);
    }

    /// A worker with workers of its own — two levels below the top, and a
    /// middle node that runs no layer itself — is still exact.
    #[test]
    fn a_pyramid_of_workers_is_exact_in_f32() {
        let model = fixture::model();
        let tokens = fixture::tokens();
        let whole = run(model.as_ref(), &tokens, 5);
        let f32 = ActivationFormat::F32;
        let a = worker(
            &model,
            1..4,
            f32,
            vec![
                worker(&model, 2..3, f32, vec![]),
                worker(&model, 3..4, f32, vec![worker(&model, 3..4, f32, vec![])]),
            ],
        );
        let split = delegating(0..1, vec![a]);
        assert_eq!(run(&split, &tokens, 5), whole);
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0
    }

    /// The narrower formats move the logits a little and the choice of
    /// token not at all, on this model.
    #[test]
    fn f16_and_q8_0_stay_close() {
        let model = fixture::model();
        let tokens = fixture::tokens();
        let whole = run(model.as_ref(), &tokens, 5);
        for (format, tolerance) in [(ActivationFormat::F16, 0.01), (ActivationFormat::Q8_0, 0.1)] {
            let split = delegating(0..2, vec![worker(&model, 2..4, format, vec![])]);
            for (got, want) in run(&split, &tokens, 5).iter().zip(&whole) {
                let worst = got
                    .iter()
                    .zip(want)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                assert!(worst < tolerance, "{format:?}: {worst}");
                assert_eq!(argmax(got), argmax(want), "{format:?}");
            }
        }
    }

    /// Rolling the cache back reaches the workers: after a truncate the
    /// sequence continues from the shorter prefix exactly as a fresh run of
    /// that prefix would.
    #[test]
    fn a_truncate_reaches_the_workers() {
        let model = fixture::model();
        let tokens = fixture::tokens();
        let split = delegating(
            0..1,
            vec![worker(
                &model,
                1..4,
                ActivationFormat::F32,
                vec![worker(&model, 2..4, ActivationFormat::F32, vec![])],
            )],
        );
        let mut cache = split.new_kv_cache(N_CTX);
        split.forward(&mut cache, &tokens[..6], 0, 0).unwrap();
        assert_eq!(cache.committed_len(), 6);
        cache.truncate(3).unwrap();
        assert_eq!(cache.committed_len(), 3);
        let after = split.forward(&mut cache, &tokens[3..5], 3, 0).unwrap();

        let mut fresh = model.new_kv_cache(N_CTX);
        let want = model.forward(&mut fresh, &tokens[..5], 0, 0).unwrap();
        assert_eq!(after, want);
    }

    /// The speculative verify pass: logits for every position of a
    /// multi-token forward.
    #[test]
    fn every_position_s_logits_come_back_for_a_verify_pass() {
        let model = fixture::model();
        let tokens = fixture::tokens();
        let split = delegating(
            0..2,
            vec![worker(&model, 2..4, ActivationFormat::F32, vec![])],
        );
        let mut cache = split.new_kv_cache(N_CTX);
        let all = split
            .forward_all_logits(&mut cache, &tokens[..4], 0, 0)
            .unwrap();
        assert_eq!(all.len(), 4);
        // Against the unsplit model over the same four rows at once: a
        // narrower pass rounds differently, split or not.
        let mut fresh = model.new_kv_cache(N_CTX);
        let x = model.embed(&tokens[..4]).unwrap();
        let x = model
            .forward_layers(&mut fresh, x, &tokens[..4], 0..N_LAYER, 0)
            .unwrap();
        assert_eq!(all, model.head(&x, 4).unwrap());
        // And the last row is the ordinary forward's answer.
        let mut fresh = model.new_kv_cache(N_CTX);
        assert_eq!(
            all[3],
            model.forward(&mut fresh, &tokens[..4], 0, 0).unwrap()
        );
    }

    /// Dropping the cache releases the session on every worker below.
    #[test]
    fn dropping_the_cache_releases_every_worker() {
        let model = fixture::model();
        let leaf = Arc::new(SessionStore::new(
            Arc::new(LayerPipeline::new(model.clone(), 3..4, vec![]).unwrap()),
            N_CTX,
            8,
        ));
        let middle = Arc::new(SessionStore::new(
            Arc::new(
                LayerPipeline::new(
                    model.clone(),
                    1..3,
                    vec![Box::new(LoopbackStage::new(
                        leaf.clone(),
                        ActivationFormat::F16,
                    ))],
                )
                .unwrap(),
            ),
            N_CTX,
            8,
        ));
        let split = delegating(
            0..1,
            vec![Box::new(LoopbackStage::new(
                middle.clone(),
                ActivationFormat::F16,
            ))],
        );
        let mut cache = split.new_kv_cache(N_CTX);
        split.forward(&mut cache, &[1, 2, 3], 0, 0).unwrap();
        assert_eq!((middle.len(), leaf.len()), (1, 1));
        drop(cache);
        assert_eq!((middle.len(), leaf.len()), (0, 0));
    }

    /// A delegating model over one worker holding layers 2..4, `f32` on the
    /// wire, and that worker's session store.
    fn one_worker_tree() -> (Arc<SessionStore>, DelegatingModel) {
        let model = fixture::model();
        let leaf = Arc::new(SessionStore::new(
            Arc::new(LayerPipeline::new(model, 2..4, vec![]).unwrap()),
            N_CTX,
            8,
        ));
        let split = delegating(
            0..2,
            vec![Box::new(LoopbackStage::new(
                leaf.clone(),
                ActivationFormat::F32,
            ))],
        );
        (leaf, split)
    }

    /// The first turn of a conversation: a five-token prompt and two
    /// decode steps, leaving seven positions in `cache`.
    fn first_turn(model: &dyn ModelForward, cache: &mut KvCache) {
        let tokens = fixture::tokens();
        model.forward(cache, &tokens[..5], 0, 0).unwrap();
        for pos in 5..7 {
            model.forward(cache, &tokens[pos..pos + 1], pos, 0).unwrap();
        }
    }

    /// A slot's retained conversation hands its rows on the workers to the
    /// slot's next request: the worker keeps one session, only the new
    /// tokens go through the tree, and the answer is the one a model alone
    /// gives.
    #[test]
    fn a_slot_hands_its_rows_on_the_workers_to_the_next_turn() {
        use crate::engine::slot_store::SlotStore;
        let (leaf, split) = one_worker_tree();
        let store = SlotStore::at(std::path::PathBuf::new(), String::new(), 1);
        let history = fixture::tokens()[..7].to_vec();
        let mut cache = split.new_kv_cache(N_CTX);
        first_turn(&split, &mut cache);
        store.retain(0, history.clone(), cache);
        assert_eq!(leaf.len(), 1);

        let mut prompt = history.clone();
        prompt.extend([11, 12, 13]);
        let mut next = split.new_kv_cache(N_CTX);
        assert_eq!(store.reuse_into(0, &prompt, &mut next), 7);
        let logits = split.forward(&mut next, &prompt[7..], 7, 0).unwrap();
        assert_eq!(leaf.len(), 1, "the retained session, carried on");

        let alone = fixture::model();
        let mut reference = alone.new_kv_cache(N_CTX);
        first_turn(alone.as_ref(), &mut reference);
        let expected = alone.forward(&mut reference, &prompt[7..], 7, 0).unwrap();
        assert_eq!(logits, expected);

        // What was left of the snapshot is gone: its rows moved on.
        let mut another = split.new_kv_cache(N_CTX);
        assert_eq!(store.reuse_into(0, &prompt, &mut another), 0);
        assert!(store.save(0, "slot.bin").is_ok_and(|saved| saved == 0));
        drop((next, another));
        assert_eq!(leaf.len(), 0);
    }

    /// A source that decodes on the top-level node alone.
    struct HandingOver(Arc<LayerPipeline>);

    impl PipelineSource for HandingOver {
        fn current(&self) -> (Arc<LayerPipeline>, u64) {
            (self.0.clone(), 0)
        }

        fn recover(&self, _generation: u64) -> bool {
            false
        }

        fn decode_alone(&self) -> bool {
            true
        }
    }

    /// A source whose last node applies the output head.
    struct HeadThere(Arc<LayerPipeline>);

    impl PipelineSource for HeadThere {
        fn current(&self) -> (Arc<LayerPipeline>, u64) {
            (self.0.clone(), 0)
        }

        fn recover(&self, _generation: u64) -> bool {
            false
        }

        fn head_on_last(&self) -> bool {
            true
        }
    }

    /// The output head on the node with the final layer, two levels down:
    /// logits come back instead of the stream, exactly the model's
    /// own — for a prompt, each decode step and a multi-position forward. A
    /// node without the final layer refuses to apply it.
    #[test]
    fn the_node_with_the_final_layer_can_apply_the_head() {
        let model = fixture::model();
        let leaf = Arc::new(SessionStore::new(
            Arc::new(LayerPipeline::new(model.clone(), 3..4, vec![]).unwrap()),
            N_CTX,
            8,
        ));
        let middle = Arc::new(SessionStore::new(
            Arc::new(
                LayerPipeline::new(
                    model.clone(),
                    1..3,
                    vec![Box::new(LoopbackStage::new(leaf, ActivationFormat::F32))],
                )
                .unwrap(),
            ),
            N_CTX,
            8,
        ));
        let pipeline = Arc::new(
            LayerPipeline::new(
                model.clone(),
                0..1,
                vec![Box::new(LoopbackStage::new(middle, ActivationFormat::F32))],
            )
            .unwrap(),
        );
        let split = DelegatingModel::with_source(model.clone(), Arc::new(HeadThere(pipeline)));
        let tokens = fixture::tokens();
        assert_eq!(run(&split, &tokens, 5), run(model.as_ref(), &tokens, 5));
        let mut here = split.new_kv_cache(N_CTX);
        let mut alone = model.new_kv_cache(N_CTX);
        assert_eq!(
            split
                .forward_all_logits(&mut here, &tokens[..4], 0, 0)
                .unwrap(),
            model
                .forward_all_logits(&mut alone, &tokens[..4], 0, 0)
                .unwrap()
        );

        let first = LayerPipeline::new(model.clone(), 0..2, vec![]).unwrap();
        let mut cache = model.new_kv_cache_for_layers(0..2, N_CTX);
        let hidden = model.embed(&tokens[..2]).unwrap();
        let err = first
            .run(&mut cache, 1, hidden, &tokens[..2], 0, Rows::LastLogits)
            .unwrap_err();
        assert!(err.to_string().contains("do not end the model"), "{err:#}");
    }

    /// Decoding alone: the prompt runs through a two-level tree, the
    /// first decode step brings every layer's rows back from the workers —
    /// the one below the worker too — and lets them go, and every logit is
    /// the model's alone. The slot's next turn goes on here, from its rows.
    #[test]
    fn a_sequence_decodes_alone_once_its_prompt_is_through_the_tree() {
        use crate::engine::slot_store::SlotStore;
        let model = fixture::model();
        let leaf = Arc::new(SessionStore::new(
            Arc::new(LayerPipeline::new(model.clone(), 3..4, vec![]).unwrap()),
            N_CTX,
            8,
        ));
        let middle = Arc::new(SessionStore::new(
            Arc::new(
                LayerPipeline::new(
                    model.clone(),
                    1..3,
                    vec![Box::new(LoopbackStage::new(
                        leaf.clone(),
                        ActivationFormat::F32,
                    ))],
                )
                .unwrap(),
            ),
            N_CTX,
            8,
        ));
        let pipeline = Arc::new(
            LayerPipeline::new(
                model.clone(),
                0..1,
                vec![Box::new(LoopbackStage::new(
                    middle.clone(),
                    ActivationFormat::F32,
                ))],
            )
            .unwrap(),
        );
        let split = DelegatingModel::with_source(model.clone(), Arc::new(HandingOver(pipeline)));
        let tokens = fixture::tokens();
        assert_eq!(run(&split, &tokens, 5), run(model.as_ref(), &tokens, 5));

        let store = SlotStore::at(std::path::PathBuf::new(), String::new(), 1);
        let mut cache = split.new_kv_cache(N_CTX);
        split.forward(&mut cache, &tokens[..5], 0, 0).unwrap();
        assert!(cache.remote.is_some());
        assert_eq!((middle.len(), leaf.len()), (1, 1));
        for pos in 5..7 {
            split
                .forward(&mut cache, &tokens[pos..pos + 1], pos, 0)
                .unwrap();
        }
        assert!(cache.remote.is_none(), "handed over");
        assert_eq!((middle.len(), leaf.len()), (0, 0), "the workers let it go");
        store.retain(0, tokens[..7].to_vec(), cache);

        let mut prompt = tokens[..7].to_vec();
        prompt.extend([11, 12, 13]);
        let mut next = split.new_kv_cache(N_CTX);
        assert_eq!(store.reuse_into(0, &prompt, &mut next), 7);
        let logits = split.forward(&mut next, &prompt[7..], 7, 0).unwrap();
        assert!(next.remote.is_none(), "went on here");
        assert_eq!((middle.len(), leaf.len()), (0, 0));
        let mut reference = model.new_kv_cache(N_CTX);
        first_turn(model.as_ref(), &mut reference);
        assert_eq!(
            logits,
            model.forward(&mut reference, &prompt[7..], 7, 0).unwrap()
        );
        store.retain(0, prompt.clone(), next);

        // Less than half of a prompt reused: the whole of it goes through the
        // tree, which reads prompts faster.
        let mut long = prompt[..7].to_vec();
        long.extend(30..40);
        let mut fresh = split.new_kv_cache(N_CTX);
        assert_eq!(store.reuse_into(0, &long, &mut fresh), 0);
        split.forward(&mut fresh, &long, 0, 0).unwrap();
        assert_eq!((middle.len(), leaf.len()), (1, 1), "through the tree");
    }

    /// A hybrid model decodes alone too: its recurrent layers' states come
    /// back with the attention layers' rows, from both levels of the tree,
    /// and every logit after is the model's alone. Its layers alternate,
    /// so each node holds both kinds — or, at the leaf, only attention.
    #[test]
    fn a_hybrid_sequence_decodes_alone_with_its_states_brought_back() {
        let model = fixture::model_for(fixture::Variant {
            hybrid: true,
            ..fixture::Variant::default()
        });
        let leaf = Arc::new(SessionStore::new(
            Arc::new(LayerPipeline::new(model.clone(), 3..4, vec![]).unwrap()),
            N_CTX,
            8,
        ));
        let middle = Arc::new(SessionStore::new(
            Arc::new(
                LayerPipeline::new(
                    model.clone(),
                    1..3,
                    vec![Box::new(LoopbackStage::new(
                        leaf.clone(),
                        ActivationFormat::F32,
                    ))],
                )
                .unwrap(),
            ),
            N_CTX,
            8,
        ));
        let pipeline = Arc::new(
            LayerPipeline::new(
                model.clone(),
                0..1,
                vec![Box::new(LoopbackStage::new(
                    middle.clone(),
                    ActivationFormat::F32,
                ))],
            )
            .unwrap(),
        );
        let split = DelegatingModel::with_source(model.clone(), Arc::new(HandingOver(pipeline)));
        let tokens = fixture::tokens();
        let mut cache = split.new_kv_cache(N_CTX);
        let mut reference = model.new_kv_cache(N_CTX);
        let mut worst = 0f32;
        let mut compare = |a: Vec<f32>, b: Vec<f32>| {
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(&b) {
                worst = worst.max((x - y).abs());
            }
        };
        compare(
            split.forward(&mut cache, &tokens[..5], 0, 0).unwrap(),
            model.forward(&mut reference, &tokens[..5], 0, 0).unwrap(),
        );
        assert_eq!((middle.len(), leaf.len()), (1, 1));
        for pos in 5..tokens.len() {
            compare(
                split
                    .forward(&mut cache, &tokens[pos..pos + 1], pos, 0)
                    .unwrap(),
                model
                    .forward(&mut reference, &tokens[pos..pos + 1], pos, 0)
                    .unwrap(),
            );
        }
        assert!(cache.remote.is_none(), "handed over");
        assert_eq!((middle.len(), leaf.len()), (0, 0), "the workers let it go");
        assert!(worst < 1e-4, "{worst}");
    }

    /// The end of a request answers where its time went, per node and in
    /// pipeline order, and starts counting afresh for the next request on
    /// the same session.
    #[test]
    fn a_finished_request_says_where_its_time_went() {
        let (_leaf, split) = one_worker_tree();
        let mut cache = split.new_kv_cache(N_CTX);
        first_turn(&split, &mut cache);
        let parts = cache.request_finished();
        let names: Vec<&str> = parts.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names.len(), 2, "{names:?}");
        assert!(parts.iter().all(|(_, t)| !t.is_zero()), "{parts:?}");
        assert!(cache.request_finished().is_empty(), "counted afresh");
        assert!(KvCache::new(1, 4, 2).request_finished().is_empty());
    }

    /// Another slot's conversation shares its prefix with a new request:
    /// every worker of a two-level tree copies its rows into the request's
    /// session, the conversation keeps its own, and the answer is the one a
    /// model alone gives. A stage that cannot copy means no sharing.
    #[test]
    fn another_slot_s_prefix_is_copied_on_every_worker() {
        use crate::engine::slot_store::SlotStore;
        let model = fixture::model();
        let f32 = ActivationFormat::F32;
        let store_of = |layers: Range<usize>, stages: Vec<Box<dyn Stage>>| {
            Arc::new(SessionStore::new(
                Arc::new(LayerPipeline::new(model.clone(), layers, stages).unwrap()),
                N_CTX,
                8,
            ))
        };
        let leaf = store_of(3..4, vec![]);
        let middle = store_of(1..3, vec![Box::new(LoopbackStage::new(leaf.clone(), f32))]);
        let split = delegating(
            0..1,
            vec![Box::new(LoopbackStage::new(middle.clone(), f32))],
        );
        let slots = SlotStore::at(std::path::PathBuf::new(), String::new(), 2);

        let history: Vec<u32> = (0..80u32).map(|t| (t * 13) % 60 + 1).collect();
        let mut conversation = split.new_kv_cache(N_CTX);
        split.forward(&mut conversation, &history, 0, 0).unwrap();
        slots.retain(1, history.clone(), conversation);

        let mut prompt = history[..75].to_vec();
        prompt.extend([7, 8, 9]);
        let mut request = split.new_kv_cache(N_CTX);
        assert_eq!(slots.reuse_into(0, &prompt, &mut request), 75);
        assert_eq!((middle.len(), leaf.len()), (2, 2), "both sessions held");
        let got = split.forward(&mut request, &prompt[75..], 75, 0).unwrap();

        let mut alone = model.new_kv_cache(N_CTX);
        model.forward(&mut alone, &history, 0, 0).unwrap();
        alone.truncate(75).unwrap();
        let want = model.forward(&mut alone, &prompt[75..], 75, 0).unwrap();
        assert_eq!(got, want);

        // The conversation still has all of its own.
        let mut next = history.clone();
        next.push(11);
        let mut again = split.new_kv_cache(N_CTX);
        assert_eq!(slots.reuse_into(1, &next, &mut again), 80);
        drop((request, again));
        assert_eq!((middle.len(), leaf.len()), (0, 0));

        // A stage that cannot copy: nothing shared.
        let slow = delegating(
            0..2,
            vec![Box::new(Slow::new(Box::new(LoopbackStage::new(
                store_of(2..4, vec![]),
                f32,
            ))))],
        );
        let mut conversation = slow.new_kv_cache(N_CTX);
        slow.forward(&mut conversation, &history, 0, 0).unwrap();
        slots.retain(1, history.clone(), conversation);
        let mut request = slow.new_kv_cache(N_CTX);
        assert_eq!(slots.reuse_into(0, &prompt, &mut request), 0);
    }

    /// A slot whose snapshot has no rows on the workers — read back from a
    /// file, say — gives a spread model nothing, and cannot be saved while
    /// it holds rows there.
    #[test]
    fn a_snapshot_without_its_workers_rows_is_not_reused() {
        use crate::engine::slot_store::SlotStore;
        let (_leaf, split) = one_worker_tree();
        let store = SlotStore::at(std::path::PathBuf::new(), String::new(), 1);
        let history = fixture::tokens()[..7].to_vec();
        let mut cache = split.new_kv_cache(N_CTX);
        first_turn(&split, &mut cache);
        store.retain(0, history.clone(), cache.duplicate());
        let mut prompt = history.clone();
        prompt.push(11);
        let mut next = split.new_kv_cache(N_CTX);
        assert_eq!(store.reuse_into(0, &prompt, &mut next), 0);

        store.retain(0, history, cache);
        assert!(store.save(0, "slot.bin").is_err());
    }

    /// A worker that let go of an idle session — as it does after
    /// `SESSION_IDLE` — has it rebuilt on the same pipeline by the next
    /// forward: no new plan (this pipeline cannot make one), the same
    /// answer.
    #[test]
    fn a_session_a_worker_forgot_is_rebuilt_in_place() {
        let (leaf, split) = one_worker_tree();
        let tokens = fixture::tokens();
        let mut cache = split.new_kv_cache(N_CTX);
        first_turn(&split, &mut cache);
        assert_eq!(leaf.evict_idle(std::time::Duration::ZERO), 1);
        let logits = split.forward(&mut cache, &tokens[7..8], 7, 0).unwrap();

        // The rebuild replays the seven tokens as one chunk.
        let alone = fixture::model();
        let mut reference = alone.new_kv_cache(N_CTX);
        alone.forward(&mut reference, &tokens[..7], 0, 0).unwrap();
        let expected = alone.forward(&mut reference, &tokens[7..8], 7, 0).unwrap();
        assert_eq!(logits, expected);
        assert_eq!(leaf.len(), 1);
    }

    /// The same exactness on a real model: a GGUF of any architecture that
    /// splits (the Llama family, Phi-3, …) split over three in-process
    /// workers, `f32` on the wire, answers a prompt and twenty greedy
    /// tokens bit for bit as it does alone, on the CPU. Run with
    /// `ORANGU_TEST_LLAMA_MODEL=/path/to/model.gguf cargo test --release
    /// --bin orangu-server a_real_model_splits_exactly -- --ignored`.
    #[test]
    #[ignore]
    fn a_real_model_splits_exactly() {
        let path = std::env::var("ORANGU_TEST_LLAMA_MODEL").expect("set ORANGU_TEST_LLAMA_MODEL");
        let loaded = crate::engine::loader::LoadedModel::open(std::path::Path::new(&path)).unwrap();
        let backend: Arc<dyn crate::engine::backend::Backend> =
            Arc::new(crate::engine::backend::CpuBackend);
        let model = crate::build_model(&loaded, &backend).unwrap();
        assert!(
            model.supports_layer_split(),
            "{} does not split",
            model.config().architecture
        );
        let n = model.config().n_layer;
        let (a, b) = (n / 3, 2 * n / 3);
        let f32 = ActivationFormat::F32;
        let split = DelegatingModel::new(Arc::new(
            LayerPipeline::new(
                model.clone(),
                0..a,
                vec![
                    worker(&model, a..b, f32, vec![]),
                    worker(&model, b..n, f32, vec![]),
                ],
            )
            .unwrap(),
        ))
        .unwrap();
        let greedy = |m: &dyn ModelForward| {
            let mut cache = m.new_kv_cache(N_CTX);
            let prompt = [1u32, 785, 6722, 315, 9625, 374];
            let mut logits = m.forward(&mut cache, &prompt, 0, 0).unwrap();
            let mut out = Vec::new();
            for pos in prompt.len()..prompt.len() + 20 {
                let next = argmax(&logits) as u32;
                out.push(next);
                logits = m.forward(&mut cache, &[next], pos, 0).unwrap();
            }
            (out, logits)
        };
        assert_eq!(greedy(&split), greedy(model.as_ref()));
    }

    /// What each activation format costs and changes: bytes per
    /// token, encode and decode time for a 128-token part, what a byte
    /// coder could still take off (order-0 entropy), and against `f32` the
    /// largest logit error and how often the greedy token agrees over a
    /// teacher-forced continuation. Prints a table; with
    /// `ORANGU_TEST_DUMP=<dir>` it also writes one encoded part per format
    /// there, for a compressor to try. Run with
    /// `ORANGU_TEST_LLAMA_MODEL=/path/to/model.gguf cargo test --release
    /// --bin orangu-server activation_formats_compared -- --ignored
    /// --nocapture`.
    #[test]
    #[ignore]
    fn activation_formats_compared() {
        use crate::workers::protocol::Activations;
        use std::time::Instant;
        let path = std::env::var("ORANGU_TEST_LLAMA_MODEL").expect("set ORANGU_TEST_LLAMA_MODEL");
        let loaded = crate::engine::loader::LoadedModel::open(std::path::Path::new(&path)).unwrap();
        let model: Arc<dyn ModelForward> = Arc::new(
            crate::engine::arch::llama::LlamaModel::load_with_backend(
                &loaded,
                Arc::new(crate::engine::backend::CpuBackend),
            )
            .unwrap(),
        );
        let n = model.config().n_layer;
        let n_embd = model.config().n_embd;
        let n_vocab = model.config().n_vocab as u32;
        let (a, b) = (n / 3, 2 * n / 3);
        // A prompt of ordinary token ids, and the continuation `f32`
        // chooses, fed to every format alike.
        let prompt: Vec<u32> = (0..128u32).map(|i| (i * 7919 + 1000) % n_vocab).collect();
        const STEPS: usize = 64;
        // Room for the prompt and the continuation; the fixture's `N_CTX`
        // is for the tiny model.
        const CTX: usize = 256;
        let stage = |layers: Range<usize>, format: ActivationFormat| -> Box<dyn Stage> {
            let pipeline = Arc::new(LayerPipeline::new(model.clone(), layers, vec![]).unwrap());
            Box::new(LoopbackStage::new(
                Arc::new(SessionStore::new(pipeline, CTX, 8)),
                format,
            ))
        };

        // A real 128-token part as it crosses the first cut.
        let mut cache = model.new_kv_cache_for_layers(0..a, CTX);
        let hidden = model.embed(&prompt).unwrap();
        let part = model
            .forward_layers(&mut cache, hidden, &prompt, 0..a, 0)
            .unwrap();

        let tree = |format: ActivationFormat| {
            DelegatingModel::new(Arc::new(
                LayerPipeline::new(
                    model.clone(),
                    0..a,
                    vec![stage(a..b, format), stage(b..n, format)],
                )
                .unwrap(),
            ))
            .unwrap()
        };
        let run = |m: &dyn ModelForward, forced: Option<&[u32]>| {
            let mut cache = m.new_kv_cache(CTX);
            let mut logits = vec![m.forward(&mut cache, &prompt, 0, 0).unwrap()];
            let mut chosen = Vec::new();
            for step in 0..STEPS {
                let next = match forced {
                    Some(tokens) => tokens[step],
                    None => argmax(logits.last().unwrap()) as u32,
                };
                chosen.push(next);
                let pos = prompt.len() + step;
                logits.push(m.forward(&mut cache, &[next], pos, 0).unwrap());
            }
            (chosen, logits)
        };
        let (reference_tokens, reference) = run(&tree(ActivationFormat::F32), None);

        println!(
            "| format | bytes/token | 128-token part | encode+decode | order-0 entropy | max logit error | greedy agreement | first free-run difference |"
        );
        println!("|---|---|---|---|---|---|---|---|");
        for format in [
            ActivationFormat::F32,
            ActivationFormat::F16,
            ActivationFormat::Q8_0,
        ] {
            let encoded = Activations::encode(&part, prompt.len(), n_embd, format);
            let started = Instant::now();
            const ROUNDS: u32 = 20;
            for _ in 0..ROUNDS {
                let e = Activations::encode(&part, prompt.len(), n_embd, format);
                std::hint::black_box(e.decode().unwrap());
            }
            let codec_ms = started.elapsed().as_secs_f64() * 1e3 / ROUNDS as f64;
            let mut counts = [0usize; 256];
            for &byte in &encoded.data {
                counts[byte as usize] += 1;
            }
            let total = encoded.data.len() as f64;
            let entropy: f64 = counts
                .iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let p = c as f64 / total;
                    -p * p.log2()
                })
                .sum();
            if let Ok(dir) = std::env::var("ORANGU_TEST_DUMP") {
                std::fs::write(format!("{dir}/part.{}", format.label()), &encoded.data).unwrap();
            }

            let split = tree(format);
            let (_, forced) = run(&split, Some(&reference_tokens));
            let mut max_error = 0f32;
            let mut agree = 0;
            for (got, want) in forced.iter().zip(&reference) {
                for (g, w) in got.iter().zip(want) {
                    max_error = max_error.max((g - w).abs());
                }
                agree += usize::from(argmax(got) == argmax(want));
            }
            let (free, _) = run(&split, None);
            let first_difference = free
                .iter()
                .zip(&reference_tokens)
                .position(|(x, y)| x != y)
                .map_or("none".to_string(), |i| format!("token {i}"));
            println!(
                "| {} | {} | {:.0} KiB | {codec_ms:.2} ms | {:.2} bits/byte ({:.0}%) | {max_error:.4} | {agree}/{} | {first_difference} |",
                format.label(),
                encoded.data.len() / prompt.len(),
                total / 1024.0,
                entropy,
                entropy / 8.0 * 100.0,
                reference.len(),
            );
        }
    }

    /// A trace names this node and each worker, and adds up across
    /// forwards.
    #[test]
    fn a_trace_says_where_the_time_went() {
        let model = fixture::model();
        let pipeline = Arc::new(
            LayerPipeline::new(
                model.clone(),
                0..2,
                vec![worker(&model, 2..4, ActivationFormat::F32, vec![])],
            )
            .unwrap()
            .named("top"),
        );
        let mut cache = pipeline.new_cache(1, N_CTX);
        let mut trace = Trace::default();
        for (pos, chunk) in [(0, &[1u32, 2, 3][..]), (3, &[4][..])] {
            let hidden = model.embed(chunk).unwrap();
            pipeline
                .run_traced(
                    &mut cache,
                    1,
                    hidden,
                    chunk,
                    pos,
                    Rows::Last,
                    Some(&mut trace),
                )
                .unwrap();
        }
        assert_eq!((trace.forwards, trace.tokens), (2, 4));
        let names: Vec<&str> = trace.parts.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["top", "loopback"]);
        let text = trace.describe("tokens");
        assert!(text.starts_with("4 tokens in 2 forwards (top "), "{text}");
    }

    /// One decode step of several sequences as a batch — through a pyramid,
    /// one message per worker at each level — gives each exactly what
    /// stepping it alone gives.
    #[test]
    fn a_batched_decode_step_is_exact() {
        use crate::engine::arch::DecodeRow;
        let model = fixture::model();
        let f32 = ActivationFormat::F32;
        let split = delegating(
            0..1,
            vec![worker(
                &model,
                1..4,
                f32,
                vec![worker(&model, 2..4, f32, vec![])],
            )],
        );
        let prompts: [&[u32]; 3] = [&[1, 2, 3], &[9, 8], &[40, 41, 42, 43]];
        let next = [5u32, 6, 7];
        let mut caches: Vec<_> = prompts
            .iter()
            .map(|p| {
                let mut cache = split.new_kv_cache(N_CTX);
                split.forward(&mut cache, p, 0, 0).unwrap();
                cache
            })
            .collect();
        let mut rows: Vec<DecodeRow<'_>> = caches
            .iter_mut()
            .zip(&prompts)
            .map(|(cache, p)| DecodeRow {
                cache,
                pos: p.len(),
                slot: 0,
            })
            .collect();
        let batched = split
            .forward_decode_batch(&mut rows, &next)
            .unwrap()
            .expect("a delegating model batches");
        drop(rows);
        for ((prompt, token), got) in prompts.iter().zip(next).zip(&batched) {
            let mut alone = model.new_kv_cache(N_CTX);
            model.forward(&mut alone, prompt, 0, 0).unwrap();
            let want = model
                .forward(&mut alone, &[token], prompt.len(), 0)
                .unwrap();
            assert_eq!(*got, want);
        }
        // And the sequences carry on from there one by one.
        for (cache, prompt) in caches.iter_mut().zip(&prompts) {
            assert_eq!(cache.committed_len(), prompt.len() + 1);
            split.forward(cache, &[11], prompt.len() + 1, 0).unwrap();
        }
    }

    /// Letting the kernel drop a layer's pages is only ever a cost, never
    /// a change: the model reads them back from the file and answers the
    /// same.
    #[test]
    fn released_weights_come_back_unchanged() {
        let loaded = fixture::loaded(fixture::Variant::default());
        let model: Arc<dyn ModelForward> = Arc::new(
            crate::engine::arch::llama::LlamaModel::load_with_backend(
                &loaded,
                Arc::new(crate::engine::backend::CpuBackend),
            )
            .unwrap(),
        );
        let tokens = fixture::tokens();
        let before = run(model.as_ref(), &tokens, 5);
        let layer_0 = loaded.release_tensors(|name| !name.starts_with("blk.0."));
        let everything = loaded.release_tensors(|_| false);
        assert!(layer_0 > 0 && everything > layer_0);
        assert_eq!(run(model.as_ref(), &tokens, 5), before);
    }

    /// A prompt in chunks, each but the last answered as soon as the top's
    /// own layers are done and carried on through the pyramid behind it,
    /// ends where the unsplit model ends — and the sequence goes on from
    /// there.
    #[test]
    fn a_pipelined_prefill_is_exact() {
        let model = fixture::model();
        let f32 = ActivationFormat::F32;
        let split = delegating(
            0..1,
            vec![worker(
                &model,
                1..4,
                f32,
                vec![worker(&model, 2..4, f32, vec![])],
            )],
        );
        let tokens: Vec<u32> = (0..40u32).map(|t| (t * 13) % 60 + 1).collect();
        let prefill = |m: &dyn ModelForward| {
            let mut cache = m.new_kv_cache(N_CTX);
            for chunk in 0..4 {
                let range = chunk * 8..chunk * 8 + 8;
                m.forward_no_logits(&mut cache, &tokens[range.clone()], range.start, 0)
                    .unwrap();
            }
            let last = m.forward(&mut cache, &tokens[32..36], 32, 0).unwrap();
            let next = m.forward(&mut cache, &tokens[36..37], 36, 0).unwrap();
            (last, next, cache.committed_len())
        };
        assert_eq!(prefill(&split), prefill(model.as_ref()));
    }

    /// A chunk cut into parts that go through the tree one behind another
    /// ends exactly where the unsplit model ends when it takes the same
    /// parts — and the last part's logits are the ones asked for.
    #[test]
    fn a_chunk_cut_into_parts_is_exact() {
        let model = fixture::model();
        let f32 = ActivationFormat::F32;
        let tree = delegating(
            0..1,
            vec![worker(
                &model,
                1..4,
                f32,
                vec![worker(&model, 2..4, f32, vec![])],
            )],
        )
        .with_sub_chunk(3);
        let tokens: Vec<u32> = (0..20u32).map(|t| (t * 7) % 60 + 1).collect();
        let mut cache = tree.new_kv_cache(N_CTX);
        tree.forward_no_logits(&mut cache, &tokens[..10], 0, 0)
            .unwrap();
        let got = tree.forward(&mut cache, &tokens[10..17], 10, 0).unwrap();
        let next = tree.forward(&mut cache, &tokens[17..18], 17, 0).unwrap();

        let mut alone = model.new_kv_cache(N_CTX);
        for (at, len) in [(0, 3), (3, 3), (6, 3), (9, 1), (10, 3), (13, 3)] {
            model
                .forward_no_logits(&mut alone, &tokens[at..at + len], at, 0)
                .unwrap();
        }
        let want = model.forward(&mut alone, &tokens[16..17], 16, 0).unwrap();
        let want_next = model.forward(&mut alone, &tokens[17..18], 17, 0).unwrap();
        assert_eq!((got, next), (want, want_next));
        assert_eq!(cache.committed_len(), 18);
    }

    /// A stage that takes its time over every forward, and counts them —
    /// and, through `busy`, how many forwards of the stages sharing it ran
    /// at once at most (`peak`).
    struct Slow {
        inner: Box<dyn Stage>,
        forwards: Arc<AtomicUsize>,
        busy: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
    }

    impl Slow {
        fn new(inner: Box<dyn Stage>) -> Self {
            Self {
                inner,
                forwards: Arc::new(AtomicUsize::new(0)),
                busy: Arc::new(AtomicUsize::new(0)),
                peak: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    impl Stage for Slow {
        fn layers(&self) -> Range<usize> {
            self.inner.layers()
        }

        fn name(&self) -> String {
            self.inner.name()
        }

        fn forward(
            &self,
            session: u64,
            hidden: Vec<f32>,
            tokens: &[u32],
            start_pos: usize,
            rows: Rows,
        ) -> Result<Vec<f32>> {
            let busy = self.busy.fetch_add(1, Ordering::SeqCst) + 1;
            self.peak.fetch_max(busy, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(40));
            self.busy.fetch_sub(1, Ordering::SeqCst);
            self.forwards.fetch_add(1, Ordering::Relaxed);
            self.inner.forward(session, hidden, tokens, start_pos, rows)
        }

        fn truncate(&self, session: u64, len: usize) {
            self.inner.truncate(session, len);
        }

        fn release(&self, session: u64) {
            self.inner.release(session);
        }
    }

    /// Two workers side by side under one node (a flat tree) work on a
    /// prompt's parts at the same time — the second on one part while the
    /// first is on the next — and the answer is the one a model alone
    /// gives.
    #[test]
    fn a_node_s_workers_overlap_on_a_prompt() {
        let model = fixture::model();
        let busy = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let slow = |layers: Range<usize>| -> Box<dyn Stage> {
            Box::new(Slow {
                busy: busy.clone(),
                peak: peak.clone(),
                ..Slow::new(Box::new(LoopbackStage::new(
                    Arc::new(SessionStore::new(
                        Arc::new(LayerPipeline::new(model.clone(), layers, vec![]).unwrap()),
                        N_CTX,
                        8,
                    )),
                    ActivationFormat::F32,
                )))
            })
        };
        let tree = delegating(0..1, vec![slow(1..3), slow(3..4)]).with_sub_chunk(2);
        let tokens: Vec<u32> = (0..21u32).map(|t| (t * 7) % 60 + 1).collect();
        let mut cache = tree.new_kv_cache(N_CTX);
        // Ten parts queued, the last forward waiting for them all.
        tree.forward_no_logits(&mut cache, &tokens[..20], 0, 0)
            .unwrap();
        let got = tree.forward(&mut cache, &tokens[20..], 20, 0).unwrap();
        assert_eq!(peak.load(Ordering::SeqCst), 2, "the workers took turns");
        let mut alone = model.new_kv_cache(N_CTX);
        for at in (0..20).step_by(2) {
            model
                .forward_no_logits(&mut alone, &tokens[at..at + 2], at, 0)
                .unwrap();
        }
        let want = model.forward(&mut alone, &tokens[20..], 20, 0).unwrap();
        assert_eq!(got, want);
    }

    /// A sequence dropped with a prompt still queued for its workers — a
    /// client gone mid-prompt — skips the parts nobody will read, and its
    /// workers let it go soon after, not once every part has run.
    #[test]
    fn a_dropped_sequence_skips_what_it_had_queued() {
        let model = fixture::model();
        let leaf = Arc::new(SessionStore::new(
            Arc::new(LayerPipeline::new(model.clone(), 2..4, vec![]).unwrap()),
            N_CTX,
            8,
        ));
        let forwards = Arc::new(AtomicUsize::new(0));
        let tree = delegating(
            0..2,
            vec![Box::new(Slow {
                forwards: forwards.clone(),
                ..Slow::new(Box::new(LoopbackStage::new(
                    leaf.clone(),
                    ActivationFormat::F32,
                )))
            })],
        )
        .with_sub_chunk(2);
        let tokens: Vec<u32> = (0..40u32).map(|t| (t * 7) % 60 + 1).collect();
        let mut cache = tree.new_kv_cache(N_CTX);
        // Twenty parts go on through the tree without waiting.
        tree.forward_no_logits(&mut cache, &tokens, 0, 0).unwrap();
        drop(cache);
        let before = forwards.load(Ordering::Relaxed);
        let deadline = Instant::now() + Duration::from_secs(5);
        while leaf.len() > 0 {
            assert!(Instant::now() < deadline, "the worker still holds it");
            std::thread::sleep(Duration::from_millis(10));
        }
        // At most the part on the worker when the drop came finishes.
        let ran = forwards.load(Ordering::Relaxed) - before;
        assert!(
            ran <= 1,
            "{ran} of {} parts ran after the sequence was dropped",
            20 - before
        );
    }

    /// A chunk that fails on its way through the tree, with nobody waiting
    /// for it, answers the sequence's next forward instead — which, the
    /// failure being a worker that forgot the sequence, rebuilds it there
    /// and goes on.
    #[test]
    fn a_failure_in_the_background_answers_the_next_forward() {
        let model = fixture::model();
        let leaf = Arc::new(
            SessionStore::new(
                Arc::new(LayerPipeline::new(model.clone(), 2..4, vec![]).unwrap()),
                N_CTX,
                4,
            )
            .named("leaf"),
        );
        let split = delegating(
            0..2,
            vec![Box::new(LoopbackStage::new(
                leaf.clone(),
                ActivationFormat::F32,
            ))],
        );
        let mut cache = split.new_kv_cache(N_CTX);
        split
            .forward_no_logits(&mut cache, &[1, 2, 3], 0, 0)
            .unwrap();
        split.forward(&mut cache, &[4], 3, 0).unwrap();
        // The leaf forgets the sequence; the next chunk, queued, fails.
        let session = cache.remote.as_ref().unwrap().id;
        leaf.release(session);
        split.forward_no_logits(&mut cache, &[5, 6], 4, 0).unwrap();
        let got = split.forward(&mut cache, &[7], 6, 0).unwrap();
        assert_ne!(cache.remote.as_ref().unwrap().id, session, "rebuilt");
        // The rebuild replays the six tokens as one chunk.
        let mut alone = model.new_kv_cache(N_CTX);
        model
            .forward(&mut alone, &[1, 2, 3, 4, 5, 6], 0, 0)
            .unwrap();
        let want = model.forward(&mut alone, &[7], 6, 0).unwrap();
        assert_eq!(got, want);
    }

    /// The largest value of the residual stream at each layer boundary of a
    /// real model — what crosses a cut, and what an activation format must
    /// hold (`f16` stops at 65504). Run with `ORANGU_TEST_LLAMA_MODEL=… cargo
    /// test --release --bin orangu-server residual_magnitudes -- --ignored
    /// --nocapture`.
    #[test]
    #[ignore]
    fn residual_magnitudes() {
        let path = std::env::var("ORANGU_TEST_LLAMA_MODEL").expect("set ORANGU_TEST_LLAMA_MODEL");
        let loaded = crate::engine::loader::LoadedModel::open(std::path::Path::new(&path)).unwrap();
        let backend: Arc<dyn crate::engine::backend::Backend> =
            Arc::new(crate::engine::backend::CpuBackend);
        let model = crate::build_model(&loaded, &backend).unwrap();
        let n = model.config().n_layer;
        let n_vocab = model.config().n_vocab as u32;
        let prompt: Vec<u32> = (0..40u32).map(|i| (i * 7919 + 1000) % n_vocab).collect();
        let mut cache = model.new_kv_cache(N_CTX);
        let mut x = model.embed(&prompt).unwrap();
        let mut worst = 0f32;
        for il in 0..n {
            x = model
                .forward_layers(&mut cache, x, &prompt, il..il + 1, 0)
                .unwrap();
            let max = x.iter().fold(0f32, |m, v| m.max(v.abs()));
            worst = worst.max(max);
            println!("after layer {il}: max |x| {max}");
        }
        println!("largest: {worst} (f16 holds up to 65504)");
    }

    /// Gemma 4 through a tree: cut wherever the file allows it — never
    /// between the layers that share a KV cache and the layers that own it
    /// — the split model answers what the whole one does. Run with
    /// `ORANGU_TEST_GEMMA4_MODEL=/path/to/gemma-4-E2B-it-Q4_K_M.gguf cargo
    /// test --release --bin orangu-server gemma4_splits -- --ignored
    /// --nocapture`.
    #[test]
    #[ignore]
    fn gemma4_splits_where_it_may() {
        let path = std::env::var("ORANGU_TEST_GEMMA4_MODEL").expect("set ORANGU_TEST_GEMMA4_MODEL");
        let loaded = crate::engine::loader::LoadedModel::open(std::path::Path::new(&path)).unwrap();
        let model: Arc<dyn ModelForward> = Arc::new(
            crate::engine::arch::gemma::GemmaModel::load_with_backend(
                &loaded,
                Arc::new(crate::engine::backend::CpuBackend),
            )
            .unwrap(),
        );
        assert!(model.supports_layer_split());
        let n = model.config().n_layer;
        let cuts: Vec<usize> = (1..n).filter(|at| model.split_allowed(*at)).collect();
        println!("{n} layers; allowed cuts: {cuts:?}");
        assert!(!cuts.is_empty(), "{cuts:?}");
        // A file with layers that share a KV cache (E2B, E4B) forbids the
        // cuts between them and their donors; one without (12B) forbids
        // none.
        if let Some(forbidden) = (1..n).find(|at| !model.split_allowed(*at)) {
            assert!(LayerPipeline::new(model.clone(), 0..forbidden, vec![]).is_err());
        }

        // Greedy: a prompt, then ten tokens, and every step's logits — for a
        // short prompt and one of forty, which takes the prompt paths a real
        // chat's does.
        let long: Vec<u32> = (0..40u32).map(|i| (i * 7919 + 1000) % 250_000).collect();
        for prompt in [vec![2u32, 818, 5279, 529, 7001, 563], long] {
            let greedy = |m: &dyn ModelForward| {
                let mut cache = m.new_kv_cache(N_CTX);
                let mut logits = m.forward(&mut cache, &prompt, 0, 0).unwrap();
                let mut steps = vec![logits.clone()];
                let mut out = Vec::new();
                for pos in prompt.len()..prompt.len() + 10 {
                    let next = argmax(&logits) as u32;
                    out.push(next);
                    logits = m.forward(&mut cache, &[next], pos, 0).unwrap();
                    steps.push(logits.clone());
                }
                (out, steps)
            };
            let (want, want_steps) = greedy(model.as_ref());
            let f32 = ActivationFormat::F32;
            let mut splits = vec![vec![cuts[0]], vec![*cuts.last().unwrap()]];
            if cuts.len() > 2 {
                splits.push(vec![cuts[0], cuts[cuts.len() / 2]]);
            }
            for split in splits {
                let mut stages: Vec<Box<dyn Stage>> = Vec::new();
                let mut bounds = split.clone();
                bounds.push(n);
                for pair in bounds.windows(2) {
                    stages.push(worker(&model, pair[0]..pair[1], f32, vec![]));
                }
                let tree = DelegatingModel::new(Arc::new(
                    LayerPipeline::new(model.clone(), 0..split[0], stages).unwrap(),
                ))
                .unwrap();
                let (got, got_steps) = greedy(&tree);
                let worst = got_steps
                    .iter()
                    .zip(&want_steps)
                    .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
                    .fold(0f32, f32::max);
                println!(
                    "{}-token prompt, cut at {split:?}: tokens {got:?}, worst logit difference \
                     {worst}",
                    prompt.len()
                );
                assert_eq!(got, want, "cut at {split:?}");
                assert_eq!(worst, 0.0, "cut at {split:?}");
            }
        }
    }

    /// Times a decode step both ways on one process — the model's own
    /// `forward`, and the pieces a tree runs (`embed`, `forward_layers` over
    /// every layer, `head`) — to tell the split path's own cost from the
    /// network's. `ORANGU_TEST_LLAMA_MODEL=… cargo test --release --bin
    /// orangu-server decode_step_costs -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn decode_step_costs_the_same_split_or_not() {
        let path = std::env::var("ORANGU_TEST_LLAMA_MODEL").expect("set ORANGU_TEST_LLAMA_MODEL");
        let loaded = crate::engine::loader::LoadedModel::open(std::path::Path::new(&path)).unwrap();
        let model: Arc<dyn ModelForward> = Arc::new(
            crate::engine::arch::llama::LlamaModel::load_with_backend(
                &loaded,
                Arc::new(crate::engine::backend::CpuBackend),
            )
            .unwrap(),
        );
        let n = model.config().n_layer;
        let steps = 32;
        let time = |label: &str, step: &dyn Fn(&mut KvCache, u32, usize) -> Vec<f32>| {
            let mut cache = model.new_kv_cache(N_CTX);
            let mut token = 1u32;
            for pos in 0..4 {
                token = argmax(&step(&mut cache, token, pos)) as u32;
            }
            let started = std::time::Instant::now();
            for pos in 4..4 + steps {
                token = argmax(&step(&mut cache, token, pos)) as u32;
            }
            let ms = started.elapsed().as_secs_f64() * 1e3 / steps as f64;
            println!("{label}: {ms:.1} ms a step");
            ms
        };
        let whole = time("forward", &|cache, token, pos| {
            model.forward(cache, &[token], pos, 0).unwrap()
        });
        let pieces = time("embed + forward_layers + head", &|cache, token, pos| {
            let x = model.embed(&[token]).unwrap();
            let x = model.forward_layers(cache, x, &[token], 0..n, pos).unwrap();
            model.head(&x, 1).unwrap().pop().unwrap()
        });
        println!("pieces / forward = {:.2}", pieces / whole);
    }

    /// On the GPU: a model built on Vulkan, split in three, answers
    /// what the same layers answer unsplit on the same layer-by-layer path
    /// — and that path answers what the fused whole-step path does, to the
    /// token. Their logits are reported: the two paths round differently.
    /// `ORANGU_TEST_VULKAN_MODEL=/path/to/model.gguf cargo test --release
    /// --bin orangu-server splits_on_vulkan -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn a_model_splits_on_vulkan() {
        let path = std::env::var("ORANGU_TEST_VULKAN_MODEL").expect("set ORANGU_TEST_VULKAN_MODEL");
        if !fits_in_memory(&path, 2) {
            return;
        }
        let Some(vulkan) = crate::engine::backend::vulkan::VulkanBackend::try_init() else {
            println!("no Vulkan device; nothing to check");
            return;
        };
        let backend: Arc<dyn crate::engine::backend::Backend> = Arc::new(vulkan);
        let loaded = crate::engine::loader::LoadedModel::open(std::path::Path::new(&path)).unwrap();
        let model = crate::build_model(&loaded, &backend).unwrap();
        assert!(
            model.supports_layer_split(),
            "{} does not split",
            model.config().architecture
        );
        let n = model.config().n_layer;
        let cuts: Vec<usize> = (1..n).filter(|at| model.split_allowed(*at)).collect();
        let (a, b) = (cuts[cuts.len() / 3], cuts[2 * cuts.len() / 3]);
        let f32 = ActivationFormat::F32;
        let split = DelegatingModel::new(Arc::new(
            LayerPipeline::new(
                model.clone(),
                0..a,
                vec![
                    worker(&model, a..b, f32, vec![]),
                    worker(&model, b..n, f32, vec![]),
                ],
            )
            .unwrap(),
        ))
        .unwrap();
        let greedy = |m: &dyn ModelForward| {
            let mut cache = m.new_kv_cache(N_CTX);
            let prompt = [2u32, 818, 5279, 529, 7001, 563];
            let mut logits = m.forward(&mut cache, &prompt, 0, 0).unwrap();
            let (mut out, mut steps) = (Vec::new(), vec![logits.clone()]);
            for pos in prompt.len()..prompt.len() + 20 {
                let next = argmax(&logits) as u32;
                out.push(next);
                logits = m.forward(&mut cache, &[next], pos, 0).unwrap();
                steps.push(logits.clone());
            }
            (out, steps)
        };
        // The same layers unsplit, on the same layer-by-layer path: what the
        // split has to match.
        let whole = DelegatingModel::new(Arc::new(
            LayerPipeline::new(model.clone(), 0..n, vec![]).unwrap(),
        ))
        .unwrap();
        let (fused, fused_steps) = greedy(model.as_ref());
        let (stepped, stepped_steps) = greedy(&whole);
        let (got, got_steps) = greedy(&split);
        let distance = |a: &[Vec<f32>], b: &[Vec<f32>]| {
            a.iter()
                .zip(b)
                .map(|(g, w)| {
                    let scale = w.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
                    g.iter().zip(w).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
                })
                .fold(0f32, f32::max)
        };
        let split_vs_stepped = distance(&got_steps, &stepped_steps);
        let stepped_vs_fused = distance(&stepped_steps, &fused_steps);
        println!(
            "{}: cut at {a} and {b} of {n}; split against the same path unsplit: tokens {}, \
             worst logit difference {split_vs_stepped:.4} of the largest; that path against the \
             fused one: tokens {}, {stepped_vs_fused:.4}",
            model.config().architecture,
            if got == stepped {
                "identical"
            } else {
                "DIFFER"
            },
            if stepped == fused {
                "identical"
            } else {
                "DIFFER"
            },
        );
        assert_eq!(got, stepped);
        assert!(split_vs_stepped < 1e-3, "{split_vs_stepped}");
        // The two whole-model paths round differently, by how much depends
        // on the model: within 0.01 for the Llama family and Gemma 4, 0.197
        // for Ternary-Bonsai-2-27B's fused device decode — with the same
        // tokens every time. Reported above, not bounded here: what this
        // test checks is the split.
        assert_eq!(stepped, fused);
    }

    /// Which of a model's two GPU paths strays from the CPU: the model's own
    /// step (fused where it has one) and the per-layer path a tree's ranges
    /// take, each fed the CPU's greedy tokens and compared with the CPU's
    /// logits, step by step — relative to each step's largest logit.
    /// `ORANGU_TEST_VULKAN_MODEL=…
    /// cargo test --release --bin orangu-server gpu_paths_against_the_cpu --
    /// --ignored --nocapture`.
    #[test]
    #[ignore]
    fn gpu_paths_against_the_cpu() {
        let path = std::env::var("ORANGU_TEST_VULKAN_MODEL").expect("set ORANGU_TEST_VULKAN_MODEL");
        if !fits_in_memory(&path, 2) {
            return;
        }
        let Some(vulkan) = crate::engine::backend::vulkan::VulkanBackend::try_init() else {
            println!("no Vulkan device; nothing to check");
            return;
        };
        let load = |backend: Arc<dyn crate::engine::backend::Backend>| {
            let loaded =
                crate::engine::loader::LoadedModel::open(std::path::Path::new(&path)).unwrap();
            crate::build_model(&loaded, &backend).unwrap()
        };
        let cpu = load(Arc::new(crate::engine::backend::CpuBackend));
        let gpu = load(Arc::new(vulkan));
        let n = gpu.config().n_layer;
        let stepped = DelegatingModel::new(Arc::new(
            LayerPipeline::new(gpu.clone(), 0..n, vec![]).unwrap(),
        ))
        .unwrap();
        let prompt = [2u32, 818, 5279, 529, 7001, 563];
        const STEPS: usize = 40;
        let run = |m: &dyn ModelForward, forced: Option<&[u32]>| {
            let mut cache = m.new_kv_cache(N_CTX);
            let mut logits = vec![m.forward(&mut cache, &prompt, 0, 0).unwrap()];
            let mut tokens = Vec::new();
            for step in 0..STEPS {
                let next = match forced {
                    Some(t) => t[step],
                    None => argmax(logits.last().unwrap()) as u32,
                };
                tokens.push(next);
                logits.push(
                    m.forward(&mut cache, &[next], prompt.len() + step, 0)
                        .unwrap(),
                );
            }
            (tokens, logits)
        };
        let (tokens, want) = run(cpu.as_ref(), None);
        let distances = |got: &[Vec<f32>]| -> Vec<f32> {
            got.iter()
                .zip(&want)
                .map(|(g, w)| {
                    let scale = w.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
                    g.iter().zip(w).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
                })
                .collect()
        };
        let (_, own) = run(gpu.as_ref(), Some(&tokens));
        let (_, per_layer) = run(&stepped, Some(&tokens));
        let show = |d: &[f32]| {
            d.iter()
                .map(|v| format!("{v:.3}"))
                .collect::<Vec<_>>()
                .join(" ")
        };
        println!(
            "{}: distance from the CPU, prompt then each step",
            gpu.config().architecture
        );
        println!("  model's own step: {}", show(&distances(&own)));
        println!("  per-layer path:   {}", show(&distances(&per_layer)));
        // Where each path is furthest off: which logit, how far, and where it
        // stands against the step's top logit — a tail logit off by a lot is
        // noise; the top one, or a near-tie, is not.
        for (label, got) in [("model's own step", &own), ("per-layer path", &per_layer)] {
            let d = distances(got);
            let (step, _) = d
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap();
            let (g, w) = (&got[step], &want[step]);
            let (at, _) = g
                .iter()
                .zip(w)
                .enumerate()
                .max_by(|a, b| (a.1.0 - a.1.1).abs().total_cmp(&(b.1.0 - b.1.1).abs()))
                .unwrap();
            let top = argmax(w);
            println!(
                "  {label}: worst at step {step}: logit {at} is {:.3} here, {:.3} on the CPU; \
                 the CPU's top is {top} at {:.3}; same top: {}",
                g[at],
                w[at],
                w[top],
                argmax(g) == top
            );
        }
    }

    /// A decode step's cost on Vulkan, three ways: the model's own fused
    /// whole step, the same layers through `forward_layers`, and split in
    /// three over in-process workers.
    /// `ORANGU_TEST_VULKAN_MODEL=… cargo test --release --bin orangu-server
    /// vulkan_decode_step_costs -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn vulkan_decode_step_costs() {
        let path = std::env::var("ORANGU_TEST_VULKAN_MODEL").expect("set ORANGU_TEST_VULKAN_MODEL");
        if !fits_in_memory(&path, 2) {
            return;
        }
        let Some(vulkan) = crate::engine::backend::vulkan::VulkanBackend::try_init() else {
            println!("no Vulkan device; nothing to measure");
            return;
        };
        let backend: Arc<dyn crate::engine::backend::Backend> = Arc::new(vulkan);
        let loaded = crate::engine::loader::LoadedModel::open(std::path::Path::new(&path)).unwrap();
        let model = crate::build_model(&loaded, &backend).unwrap();
        let n = model.config().n_layer;
        let cuts: Vec<usize> = (1..n).filter(|at| model.split_allowed(*at)).collect();
        let (a, b) = (cuts[cuts.len() / 3], cuts[2 * cuts.len() / 3]);
        let f32 = ActivationFormat::F32;
        let whole = DelegatingModel::new(Arc::new(
            LayerPipeline::new(model.clone(), 0..n, vec![]).unwrap(),
        ))
        .unwrap();
        let split = DelegatingModel::new(Arc::new(
            LayerPipeline::new(
                model.clone(),
                0..a,
                vec![
                    worker(&model, a..b, f32, vec![]),
                    worker(&model, b..n, f32, vec![]),
                ],
            )
            .unwrap(),
        ))
        .unwrap();
        let steps = 32;
        let time = |m: &dyn ModelForward| {
            let mut cache = m.new_kv_cache(N_CTX);
            let mut token = 2u32;
            let prompt = [2u32, 818, 5279, 529];
            let mut logits = m.forward(&mut cache, &prompt, 0, 0).unwrap();
            for pos in prompt.len()..prompt.len() + 4 {
                token = argmax(&logits) as u32;
                logits = m.forward(&mut cache, &[token], pos, 0).unwrap();
            }
            let started = std::time::Instant::now();
            for pos in prompt.len() + 4..prompt.len() + 4 + steps {
                token = argmax(&logits) as u32;
                logits = m.forward(&mut cache, &[token], pos, 0).unwrap();
            }
            let _ = token;
            started.elapsed().as_secs_f64() * 1e3 / steps as f64
        };
        // Twice, the second round reported: the first path timed also pays
        // for the weights' first reads and uploads, which is not its cost.
        let _ = (time(model.as_ref()), time(&whole), time(&split));
        println!(
            "{}: fused whole step {:.1} ms; forward_layers over every layer {:.1} ms; split in three {:.1} ms",
            model.config().architecture,
            time(model.as_ref()),
            time(&whole),
            time(&split)
        );
    }

    /// A prompt's cost on Vulkan: the model's own device-resident stream
    /// over every layer in 128-token chunks — wider ones outlast the
    /// driver's timeout on the Mali, as the engine's chunk sizer knows —
    /// and split in three
    /// over in-process workers (the tree's 128-token parts, each range on
    /// its own stream), with the last position's logits of the
    /// split against the whole. Run with `ORANGU_WORKERS_RANGE_CHAIN=0` for
    /// the ranges' host step path instead. `ORANGU_TEST_VULKAN_MODEL=…
    /// cargo test --release --bin orangu-server vulkan_prefill_costs --
    /// --ignored --nocapture`.
    #[test]
    #[ignore]
    fn vulkan_prefill_costs() {
        let path = std::env::var("ORANGU_TEST_VULKAN_MODEL").expect("set ORANGU_TEST_VULKAN_MODEL");
        if !fits_in_memory(&path, 2) {
            return;
        }
        let Some(vulkan) = crate::engine::backend::vulkan::VulkanBackend::try_init() else {
            println!("no Vulkan device; nothing to measure");
            return;
        };
        let backend: Arc<dyn crate::engine::backend::Backend> = Arc::new(vulkan);
        let loaded = crate::engine::loader::LoadedModel::open(std::path::Path::new(&path)).unwrap();
        let model = crate::build_model(&loaded, &backend).unwrap();
        let n = model.config().n_layer;
        let n_vocab = model.config().n_vocab as u32;
        let cuts: Vec<usize> = (1..n).filter(|at| model.split_allowed(*at)).collect();
        let (a, b) = (cuts[cuts.len() / 3], cuts[2 * cuts.len() / 3]);
        const PROMPT: usize = 512;
        const CTX: usize = 1024;
        let stage = |layers: Range<usize>| -> Box<dyn Stage> {
            let pipeline = Arc::new(LayerPipeline::new(model.clone(), layers, vec![]).unwrap());
            Box::new(LoopbackStage::new(
                Arc::new(SessionStore::new(pipeline, CTX, 8)),
                ActivationFormat::F32,
            ))
        };
        let split = DelegatingModel::new(Arc::new(
            LayerPipeline::new(model.clone(), 0..a, vec![stage(a..b), stage(b..n)]).unwrap(),
        ))
        .unwrap()
        .with_sub_chunk(128);
        let prompt: Vec<u32> = (0..PROMPT as u32)
            .map(|i| (i * 7919 + 1000) % n_vocab)
            .collect();
        let chunked = |m: &dyn ModelForward, width: usize| {
            let mut cache = m.new_kv_cache(CTX);
            let started = std::time::Instant::now();
            let mut logits = Vec::new();
            for (i, part) in prompt.chunks(width).enumerate() {
                logits = m.forward(&mut cache, part, i * width, 0).unwrap();
            }
            (started.elapsed().as_secs_f64(), logits)
        };
        // Once to warm the device's pipelines and weights, then measured.
        let _ = chunked(model.as_ref(), 128);
        let (parts, want) = chunked(model.as_ref(), 128);
        let _ = chunked(&split, 128);
        let (tree, got) = chunked(&split, 128);
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())).max(1.0);
        let worst = got
            .iter()
            .zip(&want)
            .fold(0f32, |m, (g, w)| m.max((g - w).abs()))
            / scale;
        println!(
            "{}: {PROMPT} tokens in 128-token chunks — whole model {:.0} tok/s; split in three \
             at {a} and {b} of {n} {:.0} tok/s; last logits {} ({worst:.4} of the largest)",
            model.config().architecture,
            PROMPT as f64 / parts,
            PROMPT as f64 / tree,
            if argmax(&got) == argmax(&want) {
                "agree"
            } else {
                "DIFFER"
            },
        );
    }

    /// A Llama-family model gives every position's logits for a
    /// multi-token forward (what speculation verifies with), each row the
    /// one the same rows' forward gives.
    #[test]
    fn a_llama_model_verifies_several_positions() {
        let model = fixture::model();
        let tokens = fixture::tokens();
        let mut cache = model.new_kv_cache(N_CTX);
        model.forward(&mut cache, &tokens[..3], 0, 0).unwrap();
        let all = model
            .forward_all_logits(&mut cache, &tokens[3..7], 3, 0)
            .unwrap();
        assert_eq!(all.len(), 4);
        let mut again = model.new_kv_cache(N_CTX);
        model.forward(&mut again, &tokens[..3], 0, 0).unwrap();
        let x = model.embed(&tokens[3..7]).unwrap();
        let x = model
            .forward_layers(&mut again, x, &tokens[3..7], 0..N_LAYER, 3)
            .unwrap();
        assert_eq!(all, model.head(&x, 4).unwrap());
        assert_eq!(all[3], {
            let mut once = model.new_kv_cache(N_CTX);
            model.forward(&mut once, &tokens[..3], 0, 0).unwrap();
            model.forward(&mut once, &tokens[3..7], 3, 0).unwrap()
        });
    }

    /// An embeddings request through a pyramid: the hidden states, final
    /// norm and all, are exactly the model's own, in one pass and in
    /// chunks through a cache.
    #[test]
    fn embeddings_through_a_tree_are_exact() {
        let model = fixture::model();
        let f32 = ActivationFormat::F32;
        let tree = delegating(
            0..1,
            vec![worker(
                &model,
                1..4,
                f32,
                vec![worker(&model, 2..4, f32, vec![])],
            )],
        );
        let tokens = fixture::tokens();
        assert!(tree.hidden_states_are_causal());
        assert_eq!(
            tree.forward_hidden_states(&tokens).unwrap(),
            model.forward_hidden_states(&tokens).unwrap()
        );
        let mut cache = tree.new_kv_cache(N_CTX);
        let mut alone = model.new_kv_cache(N_CTX);
        for (at, len) in [(0, 3), (3, 5)] {
            assert_eq!(
                tree.forward_hidden_states_at(&mut cache, &tokens[at..at + len], at)
                    .unwrap(),
                model
                    .forward_hidden_states_at(&mut alone, &tokens[at..at + len], at)
                    .unwrap()
            );
        }
    }

    #[test]
    fn stages_must_follow_on_from_each_other() {
        let model = fixture::model();
        let gap = LayerPipeline::new(
            model.clone(),
            0..1,
            vec![worker(&model, 2..4, ActivationFormat::F32, vec![])],
        );
        assert!(gap.err().unwrap().to_string().contains("does not continue"));
        let short = LayerPipeline::new(model.clone(), 0..2, vec![]).unwrap();
        let err = DelegatingModel::new(Arc::new(short)).err().unwrap();
        assert!(
            err.to_string().contains("must cover every layer"),
            "{err:#}"
        );
        let past = LayerPipeline::new(model, 0..N_LAYER + 1, vec![]);
        assert!(past.is_err());
    }
}
