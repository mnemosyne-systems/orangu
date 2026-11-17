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
use super::room::{self, KvRoom};
use crate::engine::arch::{DecodeRow, ModelForward};
use crate::engine::generate::NoRoomForPrompt;
use crate::engine::kv_cache::{KvCache, RemoteLayers, RemoteSession};
use crate::engine::loader::ModelConfig;
use crate::engine::prefix_cache::PrefixCache;
use anyhow::{Result, ensure};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, Weak, mpsc};
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

    /// Promises `session` memory for `positions` here and below (`0` takes a
    /// promise back). A stage that cannot ask promises nothing, and refuses
    /// nothing.
    fn reserve(&self, session: u64, positions: usize) -> Result<()> {
        let _ = (session, positions);
        Ok(())
    }

    /// Starts session `to` with a copy of `from`'s first `len` positions,
    /// here and below. A stage that cannot says so.
    fn fork(&self, from: u64, to: u64, len: usize) -> Result<()> {
        let _ = (from, to, len);
        anyhow::bail!("{} cannot copy a session", self.name())
    }

    /// Keeps `session` here and below until it is released (`held`), or
    /// returns it to ordinary eviction. A stage that cannot does nothing:
    /// the session may then be evicted, and is rebuilt if it is used again.
    fn retain(&self, session: u64, held: bool) {
        let _ = (session, held);
    }

    /// The sessions this stage, and every node below it, keeps for this
    /// node's prefix pool, as their handshakes said. A stage that cannot say
    /// keeps none.
    fn pool_reserve(&self) -> usize {
        0
    }

    /// Whether this stage's node writes a session's rows to its disk and
    /// opens one from them again ([`super::protocol::FEATURE_PREFIXES`]).
    fn saves_prefixes(&self) -> bool {
        false
    }

    /// The prefixes this stage's node holds on its disk, as its handshake
    /// listed them.
    fn saved_prefixes(&self) -> Vec<super::protocol::SavedPrefix> {
        Vec::new()
    }

    /// Writes `session`'s rows to disk under `key`, here and below. A stage
    /// that cannot says so.
    fn save(&self, session: u64, key: &str) -> Result<()> {
        let _ = (session, key);
        anyhow::bail!("{} cannot save a session", self.name())
    }

    /// Opens `session`, held, from the first `len` positions saved under
    /// `key`, here and below. A stage that cannot, or whose node did not list
    /// `key`, says so and sends nothing.
    fn load(&self, session: u64, key: &str, len: usize) -> Result<()> {
        let _ = (session, key, len);
        anyhow::bail!("{} cannot open a saved session", self.name())
    }

    /// Deletes what is saved under `key`, here and below. A stage that
    /// cannot does nothing.
    fn forget(&self, key: &str) {
        let _ = key;
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
    /// What this node has promised of its memory for its own layers.
    room: KvRoom,
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

/// `ORANGU_WORKERS_WINDOW`: how many prompt parts nobody waits for a node
/// keeps per stage below it before it holds the next one back; `0` is no
/// bound, for comparison.
fn window() -> usize {
    static WIDTH: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *WIDTH.get_or_init(|| {
        std::env::var("ORANGU_WORKERS_WINDOW")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(DEFAULT_WINDOW)
    })
}

/// [`window`] when unset: one part being worked on and one ready to follow,
/// so a stage never waits for this node. More only lengthens what the
/// prompt's last part waits behind, and holds more rows in memory; the
/// slowest stage sets the pace whatever the queue.
const DEFAULT_WINDOW: usize = 2;

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
    /// One slot per forward nobody waits for, freed once it is through the
    /// last stage. `None` when the queue is not bounded.
    room: Option<mpsc::SyncSender<()>>,
}

impl Tail {
    fn start(session: u64, stages: Vec<Arc<dyn Stage>>) -> Self {
        let (jobs, inbox) = mpsc::channel::<Job>();
        let queued = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let poison: Arc<Mutex<Option<WorkerError>>> = Arc::new(Mutex::new(None));
        let released = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let (room, mut freed) = match window() * stages.len() {
            0 => (None, None),
            slots => {
                let (room, freed) = mpsc::sync_channel::<()>(slots);
                (Some(room), Some(freed))
            }
        };
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
            let freed = if i == last { freed.take() } else { None };
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
                                        } else if let Some(freed) = &freed {
                                            let _ = freed.recv();
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
            room,
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
        let room = KvRoom::of(model.as_ref(), &local);
        Ok(Self {
            model,
            local,
            stages: stages.into_iter().map(Arc::from).collect(),
            name: "local".to_string(),
            tails: Mutex::new(HashMap::new()),
            room,
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
            room: KvRoom::with_free_memory(0, Box::new(|| u64::MAX)),
        }
    }

    /// The same, with its room as a test has it.
    #[cfg(test)]
    pub fn with_room(mut self, room: KvRoom) -> Self {
        self.room = room;
        self
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
        self.room.wrote(session, start_pos + tokens.len());
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
            if !wait {
                self.wait_for_room(session);
            }
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

    /// Holds a forward nobody waits for until `session`'s queue has room,
    /// starting the queue when it has none. Only this sequence waits: the
    /// lock is let go first. A queue whose last stage is gone frees nothing
    /// more and sends nothing back, which ends the wait.
    fn wait_for_room(&self, session: u64) {
        let room = self
            .tails
            .lock()
            .unwrap()
            .entry(session)
            .or_insert_with(|| Tail::start(session, self.stages.clone()))
            .room
            .clone();
        if let Some(room) = room {
            let _ = room.send(());
        }
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
    pub fn room(&self) -> &KvRoom {
        &self.room
    }

    /// Promises `session` memory for `positions` on every stage below this
    /// node. One refusal takes back what the others promised.
    pub fn reserve_below(&self, session: u64, positions: usize) -> Result<()> {
        for (i, stage) in self.stages.iter().enumerate() {
            if let Err(error) = stage.reserve(session, positions) {
                for asked in &self.stages[..i] {
                    let _ = asked.reserve(session, 0);
                }
                return Err(error);
            }
        }
        Ok(())
    }

    /// Starts session `to` on every stage with a copy of `from`'s first
    /// `len` positions — after whatever `from` still has queued.
    pub fn fork(&self, from: u64, to: u64, len: usize) -> Result<()> {
        self.drain(from);
        for stage in &self.stages {
            stage.fork(from, to, len)?;
        }
        Ok(())
    }

    /// Keeps `session` on every stage until it is released, or returns it to
    /// their ordinary eviction ([`Stage::retain`]).
    pub fn retain(&self, session: u64, held: bool) {
        for stage in &self.stages {
            stage.retain(session, held);
        }
    }

    /// The fewest sessions any stage keeps for this node's prefix pool
    /// ([`Stage::pool_reserve`]): a pool entry holds one on each. Unbounded
    /// with no stage below.
    pub fn pool_reserve(&self) -> usize {
        self.stages
            .iter()
            .map(|stage| stage.pool_reserve())
            .min()
            .unwrap_or(usize::MAX)
    }

    /// The stages, in the order the layers pass through them.
    pub fn stages(&self) -> &[Arc<dyn Stage>] {
        &self.stages
    }

    /// Whether every stage saves prefixes ([`Stage::saves_prefixes`]).
    pub fn saves_prefixes(&self) -> bool {
        self.stages.iter().all(|stage| stage.saves_prefixes())
    }

    /// Writes `session`'s rows under `key` on every stage — after whatever
    /// `session` still has queued. One that fails has the others forget it.
    pub fn save(&self, session: u64, key: &str) -> Result<()> {
        self.drain(session);
        for (i, stage) in self.stages.iter().enumerate() {
            if let Err(error) = stage.save(session, key) {
                for saved in &self.stages[..=i] {
                    saved.forget(key);
                }
                return Err(error);
            }
        }
        Ok(())
    }

    /// Opens `session` from `key`'s first `len` positions on every stage. One
    /// that fails has the others release it.
    pub fn load(&self, session: u64, key: &str, len: usize) -> Result<()> {
        for (i, stage) in self.stages.iter().enumerate() {
            if let Err(error) = stage.load(session, key, len) {
                for loaded in &self.stages[..i] {
                    loaded.release(session);
                }
                return Err(error);
            }
        }
        Ok(())
    }

    /// Deletes what every stage saved under `key`.
    pub fn forget(&self, key: &str) {
        for stage in &self.stages {
            stage.forget(key);
        }
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
        self.room.forget(session);
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

    /// A finished request's decode steps took `parts` — per node, its
    /// subtree included — over `tokens` steps: what the plan's nodes really
    /// run at.
    fn observed_decode(&self, parts: &[(String, Duration)], tokens: usize) {
        let _ = (parts, tokens);
    }

    /// Whether the node running the model's final layer applies the output
    /// head and sends logits back, rather than this node.
    fn head_on_last(&self) -> bool {
        false
    }

    /// A request ran on the current pipeline from start to finish, with no
    /// replay: whatever failed before is behind the workers it used. One that
    /// finished only by a replay says nothing, or a worker that dies once a
    /// request and comes back would be forgiven every time.
    fn served_whole(&self) {}
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
    /// Replayed onto a new pipeline since the last request finished: the
    /// request running now did not go through in one piece.
    rebuilt: bool,
    /// The positions its prompt was promised, for a rebuild to ask for again.
    promised: usize,
    /// Its first positions that every node holds on the host: written by
    /// prompt forwards from position 0 without a gap. A decode on a worker's
    /// device can keep its rows there alone (a sliding window's ring), so
    /// decoded positions do not count.
    host: usize,
    prefill: Trace,
    decode: Trace,
}

type Sessions = Arc<Mutex<HashMap<u64, SessionState>>>;

/// Sessions held on the workers for the prefix pool.
type Held = Arc<Mutex<std::collections::HashSet<u64>>>;

/// The prefix pool's entries saved on every node's disk
/// ([`super::prefixes`]): where this node keeps its own files, and the key
/// each saved entry's session was saved under.
#[derive(Default)]
struct Saved {
    files: OnceLock<Arc<super::prefixes::PrefixFiles>>,
    keys: Mutex<HashMap<u64, String>>,
    /// Entries put back in the pool at the last claim.
    loaded: AtomicUsize,
}

impl Saved {
    /// `session`'s entry left the pool: what it was saved under is deleted
    /// on every node, this one included.
    fn forget(&self, session: u64, pipeline: &LayerPipeline) {
        let key = self.keys.lock().unwrap().remove(&session);
        if let Some(key) = key {
            pipeline.forget(&key);
            if let Some(files) = self.files.get() {
                files.forget_lead(&key);
            }
        }
    }
}

/// What a save or a load of the prefix pool did: entries, and the positions
/// they hold.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefixesMoved {
    pub entries: usize,
    pub tokens: usize,
}

/// The owner of a top-level cache's remote rows: passes rollbacks and the
/// release on to the pipeline the cache was made for, and forgets the
/// session's tokens when it ends.
struct SessionOwner {
    pipeline: Arc<LayerPipeline>,
    sessions: Sessions,
    /// Where a session made by [`RemoteLayers::fork_new`] takes its id.
    next_session: Arc<AtomicU64>,
    /// The sessions held for the prefix pool ([`RemoteLayers::hold`]).
    held: Held,
    /// The pool entries saved on disk.
    saved: Arc<Saved>,
    /// Told where each finished request's decode time went.
    source: Arc<dyn PipelineSource>,
}

impl RemoteLayers for SessionOwner {
    fn truncate(&self, session: u64, len: usize) {
        self.pipeline.truncate(session, len);
    }

    fn host_len(&self, session: u64) -> usize {
        self.sessions
            .lock()
            .unwrap()
            .get(&session)
            .map_or(0, |state| state.host)
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

    /// A new session, registered with `from`'s generation and first `len`
    /// tokens, holding a copy of `from`'s rows on every worker. Not asked
    /// of the workers at all when the pool already holds as many sessions as
    /// one of them said it keeps: it would be refused.
    fn fork_new(&self, from: u64, len: usize) -> Option<u64> {
        let (held, reserve) = (
            self.held.lock().unwrap().len(),
            self.pipeline.pool_reserve(),
        );
        if held >= reserve {
            log::info!(
                "orangu-server: workers: no prefix pool copy of {len} positions: the workers keep {reserve} \
                 sessions for it, and {held} are held"
            );
            return None;
        }
        let to = self.next_session.fetch_add(1, Ordering::Relaxed);
        {
            let mut sessions = self.sessions.lock().unwrap();
            let Some(source) = sessions.get(&from).filter(|s| s.tokens.len() >= len) else {
                log::info!(
                    "orangu-server: workers: no prefix pool copy of {len} positions: session {from} holds {} \
                     of the {len} positions",
                    sessions.get(&from).map_or(0, |s| s.tokens.len())
                );
                return None;
            };
            let state = SessionState {
                generation: source.generation,
                tokens: source.tokens[..len].to_vec(),
                rebuilt: false,
                promised: 0,
                host: len,
                prefill: Trace::default(),
                decode: Trace::default(),
            };
            sessions.insert(to, state);
        }
        if let Err(error) = self.pipeline.fork(from, to, len) {
            // A node that decoded on a device holds a reply's rows there alone:
            // the copy is asked again at what every node holds on the host
            // (`KvCache::share`), so that refusal is no news.
            let decoded_on_a_device = error
                .chain()
                .find_map(|cause| cause.downcast_ref::<WorkerError>())
                .is_some_and(|e| e.code == ErrorCode::PositionMismatch);
            let state_host = self.host_len(from);
            if decoded_on_a_device && state_host < len {
                log::debug!(
                    "orangu-server: workers: no prefix pool copy of {len} positions: {error:#}"
                );
            } else {
                log::info!(
                    "orangu-server: workers: no prefix pool copy of {len} positions: {error:#}"
                );
            }
            self.sessions.lock().unwrap().remove(&to);
            self.pipeline.release(to);
            return None;
        }
        Some(to)
    }

    /// Out of the pool — taken by a request, which goes on from it — an
    /// entry saved on disk is deleted there: the pool saves what it holds
    /// next time.
    fn hold(&self, session: u64, held: bool) {
        match held {
            true => self.held.lock().unwrap().insert(session),
            false => self.held.lock().unwrap().remove(&session),
        };
        if !held {
            self.saved.forget(session, &self.pipeline);
        }
        self.pipeline.retain(session, held);
    }

    fn pool_reserve(&self) -> usize {
        self.pipeline.pool_reserve()
    }

    fn release(&self, session: u64) {
        self.held.lock().unwrap().remove(&session);
        self.saved.forget(session, &self.pipeline);
        let state = self.sessions.lock().unwrap().remove(&session);
        self.pipeline.release(session);
        if let Some(mut state) = state {
            let _ = state.log_trace();
        }
    }

    /// Where the request's time went, logged as it ends: its session may
    /// live on, carried to the slot's next request.
    fn request_finished(&self, session: u64) -> Vec<(String, Duration)> {
        let generation = self.source.current().1;
        let traced = self
            .sessions
            .lock()
            .unwrap()
            .get_mut(&session)
            .map(|state| {
                let whole = !std::mem::take(&mut state.rebuilt) && state.generation == generation;
                (whole, state.log_trace())
            });
        let Some((whole, (parts, prefill, decode))) = traced else {
            return Vec::new();
        };
        if whole && prefill.forwards + decode.forwards > 0 {
            self.source.served_whole();
        }
        if decode.tokens > 0 {
            self.source.observed_decode(&decode.parts, decode.tokens);
        }
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
/// that no longer holds the session (it restarted) — or changing, a worker
/// already on a plan this node is still making: a new plan gets past those, as
/// opposed to a refusal it cannot.
fn recoverable(error: &anyhow::Error) -> bool {
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<WorkerError>())
        .is_some_and(|e| {
            matches!(
                e.code,
                ErrorCode::ChildLost
                    | ErrorCode::UnknownSession
                    | ErrorCode::PositionMismatch
                    | ErrorCode::LayersChanged
            )
        })
}

/// Whether `error` is a node saying it has no room for a prompt.
fn out_of_memory(error: &anyhow::Error) -> bool {
    error
        .chain()
        .find_map(|cause| cause.downcast_ref::<WorkerError>())
        .is_some_and(|e| e.code == ErrorCode::OutOfMemory)
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
    next_session: Arc<AtomicU64>,
    sessions: Sessions,
    held: Held,
    saved: Arc<Saved>,
    /// The engine's prefix pool, whose entries each keep a session on the
    /// workers: dropped oldest first when a node has no room for a prompt
    /// ([`Self::attach_prefix_cache`]).
    prefix_cache: OnceLock<Weak<PrefixCache>>,
    /// Sequences replayed onto a pipeline to rebuild them.
    rebuilds: AtomicUsize,
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
            next_session: Arc::new(AtomicU64::new(1)),
            sessions: Arc::new(Mutex::new(HashMap::new())),
            held: Arc::default(),
            saved: Arc::default(),
            prefix_cache: OnceLock::new(),
            rebuilds: AtomicUsize::new(0),
            in_flight: Arc::new(AtomicUsize::new(0)),
            last_forward: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Gives this model the engine's prefix pool, to drop entries from when
    /// a node has no room for a prompt. Set once; later calls change nothing.
    pub fn attach_prefix_cache(&self, pool: &Arc<PrefixCache>) {
        let _ = self.prefix_cache.set(Arc::downgrade(pool));
    }

    /// Sessions each worker holds for the prefix pool.
    pub fn pool_sessions(&self) -> usize {
        self.held.lock().unwrap().len()
    }

    /// The engine's prefix pool, if one was attached and still exists.
    pub fn prefix_cache(&self) -> Option<Arc<PrefixCache>> {
        self.prefix_cache.get().and_then(Weak::upgrade)
    }

    /// Where this node keeps its own files of the pool's saved entries
    /// ([`Self::save_prefixes`]). Set once.
    pub fn attach_prefix_files(&self, files: Arc<super::prefixes::PrefixFiles>) {
        let _ = self.saved.files.set(files);
    }

    /// Entries saved on disk, as this node's manifest lists them, and
    /// entries put back in the pool at the last claim
    /// ([`Self::load_prefixes`]).
    pub fn prefixes_saved(&self) -> (usize, usize) {
        let saved = self
            .saved
            .files
            .get()
            .and_then(|files| files.manifest())
            .map_or(0, |m| m.entries.len());
        (saved, self.saved.loaded.load(Ordering::Relaxed))
    }

    /// Sequences replayed onto a pipeline to rebuild them, since the start.
    #[cfg(test)]
    pub fn rebuilds(&self) -> usize {
        self.rebuilds.load(Ordering::Relaxed)
    }

    /// Saves every entry of the prefix pool on every node of the current
    /// pipeline — each worker its own layers' rows ([`Stage::save`]), this
    /// node its own and the tokens — and empties the pool: what a node does
    /// as it lets its workers go, or stops. `base` is the split the entries
    /// were made on; it is written with them as the manifest a later claim
    /// plans and loads by.
    ///
    /// An entry is saved only when every stage advertised the bit and its
    /// session is on the current pipeline. One a node refuses is deleted wherever it was written. An
    /// entry put back from disk and not used since is on disk already.
    /// Whatever the last manifest named and this one does not is deleted on
    /// every node.
    pub fn save_prefixes(&self, base: super::prefixes::Manifest) -> PrefixesMoved {
        let mut moved = PrefixesMoved::default();
        let Some(files) = self.saved.files.get().cloned() else {
            return moved;
        };
        let Some(pool) = self.prefix_cache() else {
            return moved;
        };
        let (pipeline, generation) = self.source.current();
        let entries = pool.take_all();
        let saves = pipeline.saves_prefixes();
        let mut manifest = base;
        for entry in &entries {
            let Some(remote) = entry.cache.remote.as_ref() else {
                continue;
            };
            let session = remote.id;
            let current = self
                .sessions
                .lock()
                .unwrap()
                .get(&session)
                .is_some_and(|state| state.generation == generation);
            let len = entry.reusable_prefix_len(&entry.tokens);
            let capacity = remote.capacity;
            if !saves || !current || len == 0 {
                continue;
            }
            let known = self.saved.keys.lock().unwrap().get(&session).cloned();
            let key = match known {
                Some(key) => key,
                None => {
                    let key = super::prefixes::new_key();
                    if let Err(error) = pipeline.save(session, &key) {
                        log::warn!("orangu-server: workers: a prefix not saved: {error:#}");
                        continue;
                    }
                    if let Err(error) = files.write(
                        &key,
                        true,
                        pipeline.local_layers(),
                        capacity,
                        len,
                        &entry.tokens[..len],
                        &entry.cache,
                    ) {
                        log::warn!("orangu-server: workers: a prefix not saved: {error:#}");
                        pipeline.forget(&key);
                        continue;
                    }
                    key
                }
            };
            manifest
                .entries
                .push(super::prefixes::SavedEntry { key, len, capacity });
            moved.entries += 1;
            moved.tokens += len;
        }
        // In the last manifest and not this one: an entry let go of while no node
        // listened, or saved under another split.
        if let Some(old) = files.manifest() {
            for stale in old
                .entries
                .iter()
                .filter(|e| !manifest.entries.iter().any(|kept| kept.key == e.key))
            {
                pipeline.forget(&stale.key);
                files.remove(&stale.key, true);
            }
        }
        if let Err(error) = files.write_manifest(&manifest) {
            log::warn!("orangu-server: workers: the saved prefixes' manifest: {error:#}");
        }
        // On disk: dropping the entries deletes nothing there.
        self.saved.keys.lock().unwrap().clear();
        drop(entries);
        if moved.entries > 0 {
            log::info!(
                "orangu-server: workers: {} prefix{} of {} tokens saved on every node",
                moved.entries,
                if moved.entries == 1 { "" } else { "es" },
                moved.tokens
            );
        }
        moved
    }

    /// Puts the entries of `manifest` named in `keys` back in the prefix
    /// pool, on the current pipeline: a session opened from its file on every
    /// worker ([`Stage::load`]), this node's own rows from its file, held as
    /// any pool entry. An entry any node refuses is deleted on every node.
    pub fn load_prefixes(
        &self,
        manifest: &super::prefixes::Manifest,
        keys: &[String],
    ) -> PrefixesMoved {
        let mut moved = PrefixesMoved::default();
        self.saved.loaded.store(0, Ordering::Relaxed);
        let (Some(files), Some(pool)) = (self.saved.files.get().cloned(), self.prefix_cache())
        else {
            return moved;
        };
        let (pipeline, generation) = self.source.current();
        for entry in manifest.entries.iter().filter(|e| keys.contains(&e.key)) {
            let key = &entry.key;
            let capacity = match entry.capacity {
                0 => manifest.n_ctx,
                capacity => capacity,
            };
            let mut cache = self.cache_on(&pipeline, generation, capacity, Vec::new());
            let session = cache.remote.as_ref().map_or(0, |r| r.id);
            let shape = cache.structure_tag();
            let expect = super::prefixes::Expect {
                layers: pipeline.local_layers(),
                shape: &shape,
                n_ctx: capacity,
                len: entry.len,
            };
            let opened = files.read(key, true, &expect).and_then(|(tokens, saved)| {
                ensure!(
                    tokens.len() == entry.len,
                    "prefix {key} holds {} tokens, not {}",
                    tokens.len(),
                    entry.len
                );
                cache.copy_prefix_from(&saved, entry.len);
                pipeline.load(session, key, entry.len)?;
                Ok(tokens)
            });
            let tokens = match opened {
                Ok(tokens) => tokens,
                Err(error) => {
                    log::warn!(
                        "orangu-server: workers: a saved prefix is dropped on every node: {error:#}"
                    );
                    drop(cache);
                    pipeline.forget(key);
                    files.forget_lead(key);
                    continue;
                }
            };
            if let Some(remote) = cache.remote.as_mut() {
                remote.len = entry.len;
            }
            if let Some(state) = self.sessions.lock().unwrap().get_mut(&session) {
                state.tokens = tokens.clone();
                // Opened from files: on the host on every node.
                state.host = tokens.len();
            }
            pool.store(tokens, cache);
            self.saved.keys.lock().unwrap().insert(session, key.clone());
            moved.entries += 1;
            moved.tokens += entry.len;
        }
        self.saved.loaded.store(moved.entries, Ordering::Relaxed);
        if moved.entries > 0 {
            log::info!(
                "orangu-server: workers: {} saved prefix{} of {} tokens put back in the pool",
                moved.entries,
                if moved.entries == 1 { "" } else { "es" },
                moved.tokens
            );
        }
        moved
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
    ) -> KvCache {
        let session = self.next_session.fetch_add(1, Ordering::Relaxed);
        self.sessions.lock().unwrap().insert(
            session,
            SessionState {
                generation,
                // Replayed or loaded from files: written as a prompt is.
                host: tokens.len(),
                tokens,
                rebuilt: false,
                promised: 0,
                prefill: Trace::default(),
                decode: Trace::default(),
            },
        );
        let owner: Arc<dyn RemoteLayers> = Arc::new(SessionOwner {
            pipeline: pipeline.clone(),
            sessions: self.sessions.clone(),
            next_session: self.next_session.clone(),
            held: self.held.clone(),
            saved: self.saved.clone(),
            source: self.source.clone(),
        });
        pipeline.new_cache_owned_by(session, capacity, owner)
    }

    /// Asks this node and every node below for `positions` of `session`. Only
    /// a node saying it has no room is an answer: one that fails to reply
    /// fails the forward that follows, which recovers as usual.
    ///
    /// A node with no room is given some by this node: the prefix pool's
    /// oldest entry is dropped, releasing its session on every node, and the
    /// question is asked again, until the pool is empty. Workers do not
    /// evict what the pool holds; this node does.
    fn ask_for_room(
        &self,
        pipeline: &LayerPipeline,
        cache: &KvCache,
        session: u64,
        positions: usize,
    ) -> Result<()> {
        loop {
            let Err(refusal) = Self::ask_once(pipeline, cache, session, positions) else {
                return Ok(());
            };
            let evicted = self.prefix_cache().is_some_and(|pool| pool.evict_oldest());
            if !evicted {
                room::note_refused();
                return Err(anyhow::Error::new(NoRoomForPrompt(refusal)));
            }
            log::info!(
                "orangu-server: workers: a prefix-pool entry let go to make room for a prompt"
            );
        }
    }

    /// One round of [`Self::ask_for_room`]: the refusal, if any node has no
    /// room.
    fn ask_once(
        pipeline: &LayerPipeline,
        cache: &KvCache,
        session: u64,
        positions: usize,
    ) -> Result<(), String> {
        // A paged cache was admitted by the pool its rows live in.
        if !cache.is_paged()
            && let Err(short) = pipeline.room().reserve(session, positions)
        {
            let name = &pipeline.name;
            return Err(format!("OutOfMemory: {short} (at {name})"));
        }
        match pipeline.reserve_below(session, positions) {
            Err(error) if out_of_memory(&error) => {
                pipeline.room().forget(session);
                Err(format!("{error:#}"))
            }
            Err(error) => {
                log::debug!("orangu-server: workers: no answer on memory: {error:#}");
                Ok(())
            }
            Ok(()) => Ok(()),
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
        let (tokens, promised) = {
            let sessions = self.sessions.lock().unwrap();
            let state = sessions
                .get(&old)
                .ok_or_else(|| anyhow::anyhow!("a cache this model did not make"))?;
            ensure!(
                state.tokens.len() >= start_pos,
                "the sequence holds {} tokens, and the forward starts at {start_pos}",
                state.tokens.len()
            );
            (state.tokens[..start_pos].to_vec(), state.promised)
        };
        *cache = self.cache_on(pipeline, generation, capacity, tokens.clone());
        self.rebuilds.fetch_add(1, Ordering::Relaxed);
        let session = cache.remote.as_ref().map_or(0, |r| r.id);
        if let Some(state) = self.sessions.lock().unwrap().get_mut(&session) {
            state.rebuilt = true;
            state.promised = promised;
        }
        if promised > 0 {
            self.ask_for_room(pipeline, cache, session, promised)?;
        }
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
                        state.host = if tokens.len() > 1 && start_pos <= state.host {
                            start_pos + tokens.len()
                        } else {
                            state.host.min(start_pos)
                        };
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

    fn admit_prompt(&self, cache: &KvCache, positions: usize) -> Result<()> {
        let Some(session) = cache.remote.as_ref().map(|r| r.id) else {
            return Ok(());
        };
        let (pipeline, generation) = self.source.current();
        // A sequence left on an older plan is rebuilt by its first forward,
        // which asks then.
        let fresh = {
            let mut sessions = self.sessions.lock().unwrap();
            match sessions.get_mut(&session) {
                Some(state) => {
                    state.promised = positions;
                    state.generation == generation
                }
                None => false,
            }
        };
        if !fresh {
            return Ok(());
        }
        self.ask_for_room(&pipeline, cache, session, positions)
    }

    /// Every row comes back from the tree as logits.
    fn forward_decode_batch_sampling(
        &self,
        rows: &mut [DecodeRow<'_>],
        tokens: &[u32],
        _samples: &[Option<crate::engine::arch::GreedySampleParams<'_>>],
    ) -> Result<Option<Vec<crate::engine::arch::ForwardOutcome>>> {
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
                state.host = state.host.min(row.pos);
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
        self.cache_on(&pipeline, generation, capacity, Vec::new())
    }

    fn forward(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<f32>> {
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
        _slot_id: usize,
    ) -> Result<()> {
        self.run(cache, tokens, start_pos, Rows::None).map(|_| ())
    }

    fn forward_all_logits(
        &self,
        cache: &mut KvCache,
        tokens: &[u32],
        start_pos: usize,
        _slot_id: usize,
    ) -> Result<Vec<Vec<f32>>> {
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
    }

    /// The GGUF bytes: `llama` metadata and every tensor in
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
            let layer: Vec<(&str, Vec<u64>, bool)> = vec![
                ("attn_norm.weight", vec![e], true),
                ("attn_q.weight", vec![e, e], false),
                ("attn_k.weight", vec![e, kv_dim as u64], false),
                ("attn_v.weight", vec![e, kv_dim as u64], false),
                ("attn_output.weight", vec![e, e], false),
                ("ffn_norm.weight", vec![e], true),
            ];
            for (suffix, dims, norm) in layer.into_iter().chain(ffn.iter().cloned()) {
                tensors.push((format!("blk.{i}.{suffix}"), dims, norm));
            }
        }

        let arch = "llama";
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

    const MIB: u64 = 1 << 20;

    fn room_of(free: u64) -> KvRoom {
        KvRoom::with_free_memory(MIB, Box::new(move || free))
    }

    fn store_over(layers: Range<usize>, free: u64, name: &str) -> Arc<SessionStore> {
        let pipeline = LayerPipeline::new(fixture::model(), layers, vec![])
            .unwrap()
            .with_room(room_of(free));
        Arc::new(SessionStore::new(Arc::new(pipeline), N_CTX, 8).named(name))
    }

    fn over(store: &Arc<SessionStore>) -> Box<dyn Stage> {
        Box::new(LoopbackStage::new(store.clone(), ActivationFormat::F32))
    }

    /// A worker that cannot hold the prompt says so before any of it runs,
    /// and the error names it: what the client is told.
    #[test]
    fn a_prompt_a_worker_cannot_hold_is_refused_before_any_of_it_runs() {
        let worker = store_over(1..N_LAYER, 10 * MIB, "w1:8400");
        let split = DelegatingModel::new(Arc::new(
            LayerPipeline::new(fixture::model(), 0..1, vec![over(&worker)])
                .unwrap()
                .with_room(room_of(u64::MAX)),
        ))
        .unwrap();
        let cache = split.new_kv_cache(N_CTX);
        let refused_before = room::refused();
        split.admit_prompt(&cache, 10).unwrap();
        let cache = split.new_kv_cache(N_CTX);
        let error = split.admit_prompt(&cache, 11).unwrap_err();
        let error = error.downcast::<NoRoomForPrompt>().unwrap();
        assert!(error.0.contains("w1:8400"), "{}", error.0);
        assert!(error.0.contains("OutOfMemory"), "{}", error.0);
        assert!(room::refused() > refused_before);
        assert_eq!(worker.len(), 0, "nothing ran");
    }

    /// The node asking has rows of its own to find room for, and refuses
    /// before troubling the workers.
    #[test]
    fn this_node_can_refuse_for_its_own_layers() {
        let worker = store_over(1..N_LAYER, 100 * MIB, "w1:8400");
        let split = DelegatingModel::new(Arc::new(
            LayerPipeline::new(fixture::model(), 0..1, vec![over(&worker)])
                .unwrap()
                .named("top")
                .with_room(room_of(MIB)),
        ))
        .unwrap();
        let cache = split.new_kv_cache(N_CTX);
        let error = split.admit_prompt(&cache, 2).unwrap_err();
        assert!(error.to_string().contains("(at top)"), "{error}");
    }

    /// The second worker refuses: the first gives back what it promised.
    #[test]
    fn one_refusal_takes_back_what_the_other_workers_promised() {
        let first = store_over(1..2, 100 * MIB, "w1:8400");
        let second = store_over(2..N_LAYER, MIB, "w2:8400");
        let pipeline =
            LayerPipeline::new(fixture::model(), 0..1, vec![over(&first), over(&second)])
                .unwrap()
                .with_room(room_of(u64::MAX));
        let split = DelegatingModel::new(Arc::new(pipeline)).unwrap();
        let cache = split.new_kv_cache(N_CTX);
        let error = split.admit_prompt(&cache, 5).unwrap_err();
        assert!(error.to_string().contains("w2:8400"), "{error}");
        assert_eq!(first.promised_bytes(), 0);
    }

    /// A stage that fails to answer has refused nothing: the forward after it
    /// finds the fault and recovers as it always has.
    #[test]
    fn a_stage_that_does_not_answer_on_memory_has_not_refused() {
        struct Gone(Range<usize>);
        impl Stage for Gone {
            fn layers(&self) -> Range<usize> {
                self.0.clone()
            }
            fn name(&self) -> String {
                "gone".to_string()
            }
            fn forward(
                &self,
                _session: u64,
                _hidden: Vec<f32>,
                _tokens: &[u32],
                _start_pos: usize,
                _rows: Rows,
            ) -> Result<Vec<f32>> {
                anyhow::bail!("not used here")
            }
            fn truncate(&self, _session: u64, _len: usize) {}
            fn release(&self, _session: u64) {}
            fn reserve(&self, _session: u64, _positions: usize) -> Result<()> {
                Err(anyhow::Error::new(WorkerError::new(
                    ErrorCode::ChildLost,
                    "the connection closed",
                )))
            }
        }
        let split = DelegatingModel::new(Arc::new(
            LayerPipeline::new(fixture::model(), 0..1, vec![Box::new(Gone(1..N_LAYER))]).unwrap(),
        ))
        .unwrap();
        let cache = split.new_kv_cache(N_CTX);
        assert!(split.admit_prompt(&cache, 5).is_ok());
    }

    /// A worker of a release before this one is not asked at all, which would
    /// end the connection, and the prompt goes down as it used to.
    #[test]
    fn a_worker_that_cannot_answer_is_not_asked() {
        struct Older(Arc<AtomicUsize>);
        impl crate::workers::stage::Exchange for Older {
            fn exchange(
                &self,
                _message: &crate::workers::protocol::Message,
            ) -> Result<crate::workers::protocol::Message> {
                self.0.fetch_add(1, Ordering::Relaxed);
                anyhow::bail!("unexpected message")
            }
            fn name(&self) -> String {
                "older".to_string()
            }
            fn features(&self) -> u64 {
                crate::workers::protocol::FEATURES & !crate::workers::protocol::FEATURE_ROOM
            }
        }
        let asked = Arc::new(AtomicUsize::new(0));
        let stage = crate::workers::stage::MessageStage::over(
            Older(asked.clone()),
            1..N_LAYER,
            ActivationFormat::F32,
        );
        assert!(Stage::reserve(&stage, 1, 5).is_ok());
        assert_eq!(asked.load(Ordering::Relaxed), 0);
    }

    /// A sequence with no remote rows has no tree to ask.
    #[test]
    fn a_cache_with_nothing_on_the_tree_needs_no_asking() {
        let split = delegating(0..N_LAYER, vec![]);
        let cache = KvCache::new(1, 4, 2);
        assert!(split.admit_prompt(&cache, 1 << 30).is_ok());
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

    /// A finished conversation kept the way a request's end keeps it, with
    /// both the slot store and the prefix pool on.
    fn finish_first_turn(
        split: &DelegatingModel,
        pool: &PrefixCache,
        slots: &crate::engine::slot_store::SlotStore,
    ) -> Vec<u32> {
        let history = fixture::tokens()[..7].to_vec();
        let mut cache = split.new_kv_cache(N_CTX);
        first_turn(split, &mut cache);
        crate::engine::generate::keep_finished(
            Some(pool),
            Some(slots),
            0,
            false,
            history.clone(),
            cache,
        );
        history
    }

    /// What a model alone answers for `prompt[7..]` after the first turn.
    fn alone_after_first_turn(prompt: &[u32]) -> Vec<f32> {
        let alone = fixture::model();
        let mut reference = alone.new_kv_cache(N_CTX);
        first_turn(alone.as_ref(), &mut reference);
        alone.forward(&mut reference, &prompt[7..], 7, 0).unwrap()
    }

    /// A finished conversation's pool entry keeps a session of its own on
    /// the worker, held there; a new prompt sharing its prefix takes it over
    /// and gets the answer a model alone gives.
    #[test]
    fn a_pool_entry_serves_a_shared_prefix_from_the_workers() {
        use crate::engine::slot_store::SlotStore;
        let (leaf, split) = one_worker_tree();
        let pool = PrefixCache::new(crate::engine::prefix_cache::ENTRIES);
        let slots = SlotStore::at(std::path::PathBuf::new(), String::new(), 2);
        let history = finish_first_turn(&split, &pool, &slots);
        assert_eq!(leaf.len(), 2, "the slot's session and the pool's copy");
        assert_eq!(leaf.retained().0, 1, "the pool's copy is held");
        assert_eq!(pool.stats().entries, 1);
        assert_eq!(pool.stats().tokens_held, 7);

        let mut prompt = history.clone();
        prompt.extend([11, 12, 13]);
        let mut next = split.new_kv_cache(N_CTX);
        let reused = crate::engine::generate::reuse_from_pool(&pool, &prompt, &mut next);
        assert_eq!(reused, 7);
        assert_eq!(leaf.retained().0, 0, "out of the pool, no longer held");
        assert_eq!(pool.stats().hits, 1);
        let logits = split.forward(&mut next, &prompt[7..], 7, 0).unwrap();
        assert_eq!(logits, alone_after_first_turn(&prompt));
        assert_eq!(leaf.len(), 2, "taken over, not copied again");

        drop(next);
        slots.clear();
        assert_eq!(leaf.len(), 0);
    }

    /// A worker that, as one decoding on a device, holds a session's decoded
    /// rows there alone: it copies no more than the prompt's positions.
    struct DeviceDecode {
        inner: Box<dyn Stage>,
        host: usize,
    }

    impl Stage for DeviceDecode {
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
            self.inner.forward(session, hidden, tokens, start_pos, rows)
        }

        fn truncate(&self, session: u64, len: usize) {
            self.inner.truncate(session, len);
        }

        fn release(&self, session: u64) {
            self.inner.release(session);
        }

        fn fork(&self, from: u64, to: u64, len: usize) -> Result<()> {
            anyhow::ensure!(
                len <= self.host,
                "holds {} positions on the host",
                self.host
            );
            self.inner.fork(from, to, len)
        }

        fn retain(&self, session: u64, held: bool) {
            self.inner.retain(session, held);
        }

        fn pool_reserve(&self) -> usize {
            self.inner.pool_reserve()
        }
    }

    /// A worker that refuses a conversation's decoded rows still gives the
    /// pool a copy of the prompt's, which every node holds on the host: the
    /// entry keeps those tokens, and a prompt sharing them is served from it
    /// with the answer a model alone gives.
    #[test]
    fn a_worker_decoding_on_a_device_gives_the_pool_its_prompt_rows() {
        use crate::engine::slot_store::SlotStore;
        let model = fixture::model();
        let leaf = Arc::new(SessionStore::new(
            Arc::new(LayerPipeline::new(model, 2..4, vec![]).unwrap()),
            N_CTX,
            8,
        ));
        let split = delegating(
            0..2,
            vec![Box::new(DeviceDecode {
                inner: Box::new(LoopbackStage::new(leaf.clone(), ActivationFormat::F32)),
                host: 5,
            })],
        );
        let pool = PrefixCache::new(crate::engine::prefix_cache::ENTRIES);
        let slots = SlotStore::at(std::path::PathBuf::new(), String::new(), 2);
        let history = finish_first_turn(&split, &pool, &slots);
        assert_eq!(pool.stats().entries, 1);
        assert_eq!(pool.stats().tokens_held, 5, "the prompt's positions");
        assert_eq!(leaf.retained().0, 1);

        let mut prompt = history[..5].to_vec();
        prompt.extend([11, 12, 13]);
        let mut next = split.new_kv_cache(N_CTX);
        let reused = crate::engine::generate::reuse_from_pool(&pool, &prompt, &mut next);
        assert_eq!(reused, 5);
        let logits = split.forward(&mut next, &prompt[5..], 5, 0).unwrap();
        let alone = fixture::model();
        let mut reference = alone.new_kv_cache(N_CTX);
        let want = alone.forward(&mut reference, &prompt, 0, 0).unwrap();
        let worst = logits
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(worst < 1e-3, "{worst}");
    }

    /// With the pool on, the slot store keeps the conversation itself — rows
    /// on the workers included — and still serves the slot's next turn.
    #[test]
    fn the_slot_keeps_its_own_conversation_beside_the_pool() {
        use crate::engine::slot_store::SlotStore;
        let (leaf, split) = one_worker_tree();
        let pool = PrefixCache::new(crate::engine::prefix_cache::ENTRIES);
        let slots = SlotStore::at(std::path::PathBuf::new(), String::new(), 1);
        let history = finish_first_turn(&split, &pool, &slots);

        let mut prompt = history.clone();
        prompt.extend([11, 12, 13]);
        let mut next = split.new_kv_cache(N_CTX);
        assert_eq!(slots.reuse_into(0, &prompt, &mut next), 7);
        let logits = split.forward(&mut next, &prompt[7..], 7, 0).unwrap();
        assert_eq!(logits, alone_after_first_turn(&prompt));
        assert_eq!(pool.stats().entries, 1, "the pool keeps its copy");
        assert_eq!(leaf.len(), 2);

        // The slot's next end replaces its conversation, and the pool's
        // oldest entry gives way rather than a session past the worker's.
        crate::engine::generate::keep_finished(
            Some(&pool),
            Some(&slots),
            0,
            false,
            prompt.clone(),
            next,
        );
        assert_eq!(pool.stats().entries, 2);
        assert_eq!(leaf.len(), 3, "the slot's one and the pool's two");
    }

    /// The pool keeps no more sessions on a worker than it said it keeps,
    /// and sends a worker that keeps none no copy at all.
    #[test]
    fn the_pool_keeps_to_what_each_worker_reserves() {
        use crate::engine::slot_store::SlotStore;
        for reserve in [0, 1, 2] {
            let leaf = Arc::new(
                SessionStore::new(
                    Arc::new(LayerPipeline::new(fixture::model(), 2..4, vec![]).unwrap()),
                    N_CTX,
                    16,
                )
                .with_pool_reserve(reserve),
            );
            let split = delegating(0..2, vec![over(&leaf)]);
            let pool = PrefixCache::new(crate::engine::prefix_cache::ENTRIES);
            let slots = SlotStore::at(std::path::PathBuf::new(), String::new(), 4);
            for (slot, start) in [(0usize, 1u32), (1, 20), (2, 40), (3, 50)] {
                let history: Vec<u32> = (start..start + 6).collect();
                let mut cache = split.new_kv_cache(N_CTX);
                split.forward(&mut cache, &history, 0, 0).unwrap();
                crate::engine::generate::keep_finished(
                    Some(&pool),
                    Some(&slots),
                    slot,
                    false,
                    history,
                    cache,
                );
                assert!(leaf.retained().0 <= reserve, "reserve {reserve}");
            }
            assert_eq!(pool.stats().entries, reserve, "reserve {reserve}");
            assert_eq!(leaf.retained().0, reserve);
            assert_eq!(leaf.len(), 4 + reserve, "four slots and the pool's copies");
        }
    }

    /// A worker without room for a prompt makes this node drop its pool's
    /// oldest entry, which releases the entry's session there; the prompt is
    /// then admitted.
    #[test]
    fn a_worker_short_of_room_costs_the_pool_an_entry() {
        use crate::engine::slot_store::SlotStore;
        let handle: Arc<OnceLock<Weak<SessionStore>>> = Arc::new(OnceLock::new());
        let reading = handle.clone();
        // 20 MiB, less 7 MiB for every session held: 6 MiB free with the
        // slot's conversation and the pool's copy, 13 MiB with one.
        let free = move || {
            let held = reading
                .get()
                .and_then(Weak::upgrade)
                .map_or(0, |store| store.len() as u64);
            (20 * MIB).saturating_sub(held * 7 * MIB)
        };
        let pipeline = LayerPipeline::new(fixture::model(), 2..4, vec![])
            .unwrap()
            .with_room(KvRoom::with_free_memory(MIB, Box::new(free)));
        let leaf = Arc::new(SessionStore::new(Arc::new(pipeline), N_CTX, 8).named("w1:8400"));
        handle.set(Arc::downgrade(&leaf)).ok().unwrap();
        let split = DelegatingModel::new(Arc::new(
            LayerPipeline::new(fixture::model(), 0..2, vec![over(&leaf)])
                .unwrap()
                .with_room(room_of(u64::MAX)),
        ))
        .unwrap();
        let pool = Arc::new(PrefixCache::new(crate::engine::prefix_cache::ENTRIES));
        split.attach_prefix_cache(&pool);
        let slots = SlotStore::at(std::path::PathBuf::new(), String::new(), 1);
        finish_first_turn(&split, &pool, &slots);
        assert_eq!((leaf.len(), pool.stats().entries), (2, 1));

        let cache = split.new_kv_cache(N_CTX);
        split.admit_prompt(&cache, 10).unwrap();
        assert_eq!(pool.stats().entries, 0, "the entry gave way");
        assert_eq!(leaf.len(), 1, "the slot's conversation stays");

        // Nothing left to drop: refused as without a pool.
        let cache = split.new_kv_cache(N_CTX);
        let error = split.admit_prompt(&cache, 15).unwrap_err();
        assert!(error.downcast::<NoRoomForPrompt>().is_ok());
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

    /// Saved-prefix files under a fresh directory of their own, named
    /// `node`.
    fn prefix_files(tag: &str, node: &str) -> Arc<super::super::prefixes::PrefixFiles> {
        let root = std::env::temp_dir().join(format!(
            "orangu-saved-pool-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let files = super::super::prefixes::PrefixFiles::under(&root, [3; 32], node);
        let _ = std::fs::remove_dir_all(files.dir());
        Arc::new(files)
    }

    /// A worker over `layers` saving into `files`.
    fn saving_store(
        layers: Range<usize>,
        files: &Arc<super::super::prefixes::PrefixFiles>,
    ) -> Arc<SessionStore> {
        Arc::new(
            SessionStore::new(
                Arc::new(LayerPipeline::new(fixture::model(), layers, vec![]).unwrap()),
                N_CTX,
                8,
            )
            .with_prefixes(Some(files.clone())),
        )
    }

    /// A top-level node over `stages`, its own layers `local`, with a prefix
    /// pool and its own saved-prefix files.
    fn saving_top(
        local: Range<usize>,
        stages: Vec<Box<dyn Stage>>,
        files: &Arc<super::super::prefixes::PrefixFiles>,
    ) -> (DelegatingModel, Arc<PrefixCache>) {
        let split = delegating(local, stages);
        let pool = Arc::new(PrefixCache::new(crate::engine::prefix_cache::ENTRIES));
        split.attach_prefix_cache(&pool);
        split.attach_prefix_files(files.clone());
        (split, pool)
    }

    /// The manifest a top-level node over `local` and `workers` saves under.
    fn base(
        local: Range<usize>,
        workers: &[(&str, Range<usize>)],
    ) -> super::super::prefixes::Manifest {
        use super::super::prefixes::SavedNode;
        let node = |name: &str, layers: &Range<usize>| SavedNode {
            name: name.to_string(),
            layers: (layers.start, layers.end),
            budget: 0,
        };
        super::super::prefixes::Manifest {
            n_ctx: N_CTX,
            local: (local.start, local.end),
            local_budget: 0,
            workers: workers.iter().map(|(n, l)| node(n, l)).collect(),
            plan: std::iter::once(node("top", &local))
                .chain(workers.iter().map(|(n, l)| node(n, l)))
                .collect(),
            entries: Vec::new(),
        }
    }

    /// A conversation of `history`, finished and kept as a request's end
    /// keeps it.
    fn keep_conversation(
        split: &DelegatingModel,
        pool: &PrefixCache,
        slots: &crate::engine::slot_store::SlotStore,
        slot: usize,
        history: &[u32],
    ) {
        let mut cache = split.new_kv_cache(N_CTX);
        split.forward(&mut cache, history, 0, 0).unwrap();
        crate::engine::generate::keep_finished(
            Some(pool),
            Some(slots),
            slot,
            false,
            history.to_vec(),
            cache,
        );
    }

    /// The pool saved on every node as the tree is let go, and put back on a
    /// tree formed again of the same nodes, serves a prompt sharing its
    /// prefix with no replay of the prefix: the answer a model alone gives.
    /// The entry taken out of the pool is deleted from every disk.
    #[test]
    fn a_saved_pool_is_put_back_and_serves_its_prefix_without_a_replay() {
        use crate::engine::slot_store::SlotStore;
        let worker_files = prefix_files("roundtrip", "leaf");
        let top_files = prefix_files("roundtrip", "top");
        let manifest = {
            let leaf = saving_store(2..4, &worker_files);
            let (split, pool) = saving_top(0..2, vec![over(&leaf)], &top_files);
            let slots = SlotStore::at(std::path::PathBuf::new(), String::new(), 2);
            finish_first_turn(&split, &pool, &slots);
            let moved = split.save_prefixes(base(0..2, &[("leaf", 2..4)]));
            assert_eq!((moved.entries, moved.tokens), (1, 7));
            assert_eq!(pool.stats().entries, 0, "the pool is emptied");
            assert_eq!(split.prefixes_saved().0, 1);
            slots.clear();
            assert_eq!(leaf.len(), 0, "the saved session is let go of");
            top_files.manifest().expect("a manifest")
        };
        let listed = worker_files.list();
        assert_eq!(listed.len(), 1);
        assert_eq!(
            (listed[0].layer_start, listed[0].layer_end, listed[0].len),
            (2, 4, 7)
        );
        assert_eq!(manifest.entries[0].key, listed[0].key);

        // The same nodes, started again.
        let leaf = saving_store(2..4, &worker_files);
        let (split, pool) = saving_top(0..2, vec![over(&leaf)], &top_files);
        let keys: Vec<String> = manifest.entries.iter().map(|e| e.key.clone()).collect();
        let moved = split.load_prefixes(&manifest, &keys);
        assert_eq!((moved.entries, moved.tokens), (1, 7));
        assert_eq!(split.prefixes_saved(), (1, 1));
        assert_eq!(pool.stats().entries, 1);
        assert_eq!(leaf.retained().0, 1, "opened held");

        let mut prompt = fixture::tokens()[..7].to_vec();
        prompt.extend([11, 12, 13]);
        let mut next = split.new_kv_cache(N_CTX);
        let reused = crate::engine::generate::reuse_from_pool(&pool, &prompt, &mut next);
        assert_eq!(reused, 7);
        let logits = split.forward(&mut next, &prompt[7..], 7, 0).unwrap();
        assert_eq!(logits, alone_after_first_turn(&prompt));
        assert_eq!(split.rebuilds(), 0, "the prefix was not replayed");
        assert!(worker_files.list().is_empty(), "taken out of the pool");
        assert_eq!(top_files.manifest(), None);
    }

    /// An entry whose file one node finds damaged, or made for other layers,
    /// is dropped on every node, the others' copies deleted; an entry every
    /// node opens is put back.
    #[test]
    fn a_damaged_or_mismatched_file_drops_the_entry_everywhere() {
        use crate::engine::slot_store::SlotStore;
        let first_files = prefix_files("damaged", "w1");
        let second_files = prefix_files("damaged", "w2");
        let top_files = prefix_files("damaged", "top");
        let conversations: [Vec<u32>; 2] = [(1..9).collect(), (20..30).collect()];
        let manifest = {
            let w1 = saving_store(1..2, &first_files);
            let w2 = saving_store(2..4, &second_files);
            let (split, pool) = saving_top(0..1, vec![over(&w1), over(&w2)], &top_files);
            let slots = SlotStore::at(std::path::PathBuf::new(), String::new(), 2);
            for (slot, history) in conversations.iter().enumerate() {
                keep_conversation(&split, &pool, &slots, slot, history);
            }
            let moved = split.save_prefixes(base(0..1, &[("w1", 1..2), ("w2", 2..4)]));
            assert_eq!(moved.entries, 2);
            top_files.manifest().unwrap()
        };
        let (damaged, kept) = (&manifest.entries[0].key, &manifest.entries[1].key);
        let path = second_files.dir().join(format!("{damaged}.bin"));
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() / 2]).unwrap();

        let w1 = saving_store(1..2, &first_files);
        let w2 = saving_store(2..4, &second_files);
        let (split, pool) = saving_top(0..1, vec![over(&w1), over(&w2)], &top_files);
        let keys = vec![damaged.clone(), kept.clone()];
        let moved = split.load_prefixes(&manifest, &keys);
        assert_eq!(moved.entries, 1);
        assert_eq!(pool.stats().entries, 1);
        let on = |files: &super::super::prefixes::PrefixFiles| -> Vec<String> {
            files.list().into_iter().map(|s| s.key).collect()
        };
        assert_eq!(
            on(&first_files),
            vec![kept.clone()],
            "the damaged entry's copy"
        );
        assert_eq!(on(&second_files), vec![kept.clone()]);
        assert_eq!(top_files.manifest().unwrap().entries.len(), 1);
        assert_eq!((w1.len(), w2.len()), (1, 1), "only the kept entry is open");
        let mut prompt = conversations[1].clone();
        prompt.push(5);
        let mut next = split.new_kv_cache(N_CTX);
        assert_eq!(
            crate::engine::generate::reuse_from_pool(&pool, &prompt, &mut next),
            conversations[1].len()
        );
        let logits = split
            .forward(
                &mut next,
                &prompt[conversations[1].len()..],
                conversations[1].len(),
                0,
            )
            .unwrap();
        let alone = fixture::model();
        let mut reference = alone.new_kv_cache(N_CTX);
        let held = conversations[1].len();
        alone
            .forward(&mut reference, &prompt[..held], 0, 0)
            .unwrap();
        let expected = alone
            .forward(&mut reference, &prompt[held..], held, 0)
            .unwrap();
        assert_eq!(logits, expected);
        drop((next, split, pool));

        // The same nodes cut differently: their files are of other layers.
        let manifest = {
            let w1 = saving_store(1..2, &first_files);
            let w2 = saving_store(2..4, &second_files);
            let (split, pool) = saving_top(0..1, vec![over(&w1), over(&w2)], &top_files);
            let slots = SlotStore::at(std::path::PathBuf::new(), String::new(), 1);
            keep_conversation(&split, &pool, &slots, 0, &conversations[0]);
            split.save_prefixes(base(0..1, &[("w1", 1..2), ("w2", 2..4)]));
            top_files.manifest().unwrap()
        };
        let w1 = saving_store(1..3, &first_files);
        let w2 = saving_store(3..4, &second_files);
        let (split, pool) = saving_top(0..1, vec![over(&w1), over(&w2)], &top_files);
        let keys: Vec<String> = manifest.entries.iter().map(|e| e.key.clone()).collect();
        assert_eq!(split.load_prefixes(&manifest, &keys).entries, 0);
        assert_eq!(pool.stats().entries, 0);
        assert!(on(&first_files).is_empty() && on(&second_files).is_empty());
        assert_eq!(top_files.manifest(), None);
        assert_eq!((w1.len(), w2.len()), (0, 0));
    }

    /// A worker as the tests' loopback, minus [`FEATURE_PREFIXES`], noting
    /// every message it is sent.
    use super::super::protocol::Message;

    struct WithoutPrefixes {
        inner: super::super::stage::Loopback,
        sent: Mutex<Vec<String>>,
    }

    impl super::super::stage::Exchange for WithoutPrefixes {
        fn exchange(&self, message: &Message) -> Result<Message> {
            let kind = format!("{message:?}");
            self.sent.lock().unwrap().push(
                kind.split([' ', '(', '{'])
                    .next()
                    .unwrap_or_default()
                    .to_string(),
            );
            self.inner.exchange(message)
        }

        fn name(&self) -> String {
            "old".to_string()
        }

        fn features(&self) -> u64 {
            super::super::protocol::FEATURES & !super::super::protocol::FEATURE_PREFIXES
        }

        fn pool_reserve(&self) -> usize {
            self.inner.pool_reserve()
        }

        fn saved(&self) -> Vec<super::super::protocol::SavedPrefix> {
            self.inner.saved()
        }
    }

    impl super::super::stage::Exchange for Arc<WithoutPrefixes> {
        fn exchange(&self, message: &Message) -> Result<Message> {
            self.as_ref().exchange(message)
        }

        fn name(&self) -> String {
            self.as_ref().name()
        }

        fn features(&self) -> u64 {
            self.as_ref().features()
        }

        fn pool_reserve(&self) -> usize {
            self.as_ref().pool_reserve()
        }

        fn saved(&self) -> Vec<super::super::protocol::SavedPrefix> {
            self.as_ref().saved()
        }
    }

    /// A worker that did not advertise the bit is sent no Save, Load or
    /// Forget, and an entry it would hold a part of is not saved anywhere.
    #[test]
    fn a_node_without_the_bit_is_never_sent_save_or_load() {
        use super::super::stage::{Loopback, MessageStage};
        use crate::engine::slot_store::SlotStore;
        let worker_files = prefix_files("nobit", "old");
        let top_files = prefix_files("nobit", "top");
        let leaf = saving_store(2..4, &worker_files);
        let old = Arc::new(WithoutPrefixes {
            inner: Loopback(leaf.clone()),
            sent: Mutex::new(Vec::new()),
        });
        let stage = MessageStage::over(old.clone(), 2..4, ActivationFormat::F32);
        let (split, pool) = saving_top(0..2, vec![Box::new(stage)], &top_files);
        let slots = SlotStore::at(std::path::PathBuf::new(), String::new(), 1);
        finish_first_turn(&split, &pool, &slots);
        assert_eq!(pool.stats().entries, 1);
        assert_eq!(split.save_prefixes(base(0..2, &[("old", 2..4)])).entries, 0);
        assert_eq!(top_files.manifest(), None);
        assert!(worker_files.list().is_empty());

        // A manifest naming it, from elsewhere, and this node's own file of
        // the entry: nothing is asked of the worker.
        let key = super::super::prefixes::new_key();
        let mut own = fixture::model().new_kv_cache_for_layers(0..2, N_CTX);
        for layer in &mut own.layers[..2] {
            let width = layer.kv_dim();
            for _ in 0..7 {
                layer.push(&vec![0.5; width], &vec![0.5; width]);
            }
        }
        top_files
            .write(&key, true, 0..2, N_CTX, 7, &fixture::tokens()[..7], &own)
            .unwrap();
        let manifest = super::super::prefixes::Manifest {
            entries: vec![super::super::prefixes::SavedEntry {
                key: key.clone(),
                len: 7,
                capacity: N_CTX,
            }],
            ..base(0..2, &[("old", 2..4)])
        };
        let keys = vec![key];
        assert_eq!(split.load_prefixes(&manifest, &keys).entries, 0);
        let sent = old.sent.lock().unwrap().clone();
        assert!(
            !sent
                .iter()
                .any(|m| m == "Save" || m == "Load" || m == "Forget"),
            "{sent:?}"
        );
        assert!(sent.iter().any(|m| m == "Forward"), "{sent:?}");
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

    /// A stage slower than this node holds it back: parts wait here rather
    /// than pile up behind the stage. A worker's own queue is what the last
    /// part waits behind on its parent's clock, so the queue's length has to
    /// be bounded for that clock to mean anything. The answer does not change.
    #[test]
    fn a_slow_stage_holds_the_parts_back() {
        let model = fixture::model();
        let slow = Slow::new(Box::new(LoopbackStage::new(
            Arc::new(SessionStore::new(
                Arc::new(LayerPipeline::new(model.clone(), 1..4, vec![]).unwrap()),
                N_CTX,
                8,
            )),
            ActivationFormat::F32,
        )));
        let done = slow.forwards.clone();
        let tree = delegating(0..1, vec![Box::new(slow) as Box<dyn Stage>]).with_sub_chunk(2);
        let tokens: Vec<u32> = (0..41u32).map(|t| (t * 7) % 60 + 1).collect();
        let mut cache = tree.new_kv_cache(N_CTX);
        tree.forward_no_logits(&mut cache, &tokens[..40], 0, 0)
            .unwrap();
        let waiting = 20 - done.load(Ordering::Relaxed);
        assert!(waiting <= 2, "{waiting} parts were left behind the stage");
        let got = tree.forward(&mut cache, &tokens[40..], 40, 0).unwrap();
        let mut alone = model.new_kv_cache(N_CTX);
        for at in (0..40).step_by(2) {
            model
                .forward_no_logits(&mut alone, &tokens[at..at + 2], at, 0)
                .unwrap();
        }
        assert_eq!(
            got,
            model.forward(&mut alone, &tokens[40..], 40, 0).unwrap()
        );
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
