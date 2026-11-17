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

//! The worker side of a forward: a [`SessionStore`] keeps one KV cache per
//! session for the layers this node is responsible for, checks that every
//! forward continues exactly where the last one left off, and runs it
//! through the node's [`LayerPipeline`].

use super::pipeline::{BatchRow, LayerPipeline, Trace};
use super::prefixes::{Expect, PrefixFiles};
use super::protocol::{
    ActivationFormat, Activations, ErrorCode, Forward, ForwardResult, Message, WorkerError,
};
use crate::engine::kv_cache::KvCache;
use std::collections::HashMap;
#[cfg(test)]
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

struct Session {
    cache: KvCache,
    last_used: Instant,
    /// Held for the parent's prefix pool ([`Message::Retain`]): kept until
    /// the parent releases it, whatever its idleness or a prompt's
    /// shortfall.
    held: bool,
    /// Where this node's time for the session went, kept only under
    /// `ORANGU_WORKERS_TRACE` and logged when the session ends.
    trace: Option<Trace>,
}

/// `ORANGU_WORKERS_TRACE=1`: every worker logs, as each session ends, the
/// time its forwards took here — its own layers, and each of its workers'
/// round trips — beside the forwards' own count. What tells a slow tree's
/// compute from its hand-overs; off by default, since a worker would
/// otherwise log a line for every request its parent serves.
fn tracing() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::engine::env::flag_on("ORANGU_WORKERS_TRACE"))
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Some(trace) = &self.trace
            && trace.forwards > 0
        {
            log::info!(
                "orangu-server: workers: this node served {}",
                trace.describe("tokens")
            );
        }
    }
}

pub struct SessionStore {
    pipeline: Arc<LayerPipeline>,
    /// Positions one session may hold: the `n_ctx` the parent assigned.
    capacity: usize,
    /// Sessions held at once: two a parent's slot, and the pool reserve.
    max_sessions: usize,
    /// This node's address, prefixed to the path of every error that passes
    /// through it.
    node: String,
    sessions: Mutex<HashMap<u64, Arc<Mutex<Session>>>>,
    /// How long a session has gone unused before a prompt with no room may
    /// take its place. Shorter would catch a session between two forwards.
    evictable_after: Duration,
    /// Sessions this node keeps for its parent's prefix pool, as its own
    /// configuration sets it.
    pool_reserve: usize,
    /// Where this node saves sessions its parent asks it to
    /// ([`Message::Save`]); `None` saves nothing.
    files: Option<Arc<PrefixFiles>>,
}

/// [`SessionStore::evictable_after`].
const EVICTABLE_AFTER: Duration = Duration::from_secs(30);

impl SessionStore {
    pub fn new(pipeline: Arc<LayerPipeline>, capacity: usize, max_sessions: usize) -> Self {
        Self {
            pipeline,
            capacity,
            max_sessions,
            node: "loopback".to_string(),
            sessions: Mutex::new(HashMap::new()),
            evictable_after: EVICTABLE_AFTER,
            pool_reserve: crate::engine::prefix_cache::ENTRIES,
            files: None,
        }
    }

    /// Saves and opens sessions in `files` when the parent asks.
    pub fn with_prefixes(mut self, files: Option<Arc<PrefixFiles>>) -> Self {
        self.files = files;
        self
    }

    /// The prefixes this node holds on its disk.
    #[cfg(test)]
    pub fn saved(&self) -> Vec<super::protocol::SavedPrefix> {
        self.files.as_ref().map_or_else(Vec::new, |f| f.list())
    }

    /// Keeps `reserve` sessions for the parent's prefix pool rather than the
    /// pool's whole size.
    pub fn with_pool_reserve(mut self, reserve: usize) -> Self {
        self.pool_reserve = reserve;
        self
    }

    /// The sessions this node and every node below it keep for the parent's
    /// prefix pool: the fewest of them.
    #[cfg(test)]
    pub fn pool_reserve(&self) -> usize {
        self.pool_reserve.min(self.pipeline.pool_reserve())
    }

    #[cfg(test)]
    pub fn evictable_after(mut self, after: Duration) -> Self {
        self.evictable_after = after;
        self
    }

    pub fn named(mut self, node: impl Into<String>) -> Self {
        self.node = node.into();
        self
    }

    #[cfg(test)]
    pub fn name(&self) -> &str {
        &self.node
    }

    #[cfg(test)]
    pub fn layers(&self) -> Range<usize> {
        self.pipeline.layers()
    }

    /// What this node has promised and not yet written.
    #[cfg(test)]
    pub fn promised_bytes(&self) -> u64 {
        self.pipeline.room().promised_bytes()
    }

    /// Sessions currently held.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.sessions.lock().unwrap().len()
    }

    /// Whether no session is running a forward, and none has for `period`.
    /// A session kept for its parent's next request (a slot's retained
    /// conversation) does not keep this node busy.
    pub fn quiet(&self, period: Duration) -> bool {
        let now = Instant::now();
        self.sessions.lock().unwrap().values().all(|s| {
            s.try_lock()
                .is_ok_and(|s| now.duration_since(s.last_used) >= period)
        })
    }

    fn error(&self, code: ErrorCode, message: impl Into<String>) -> WorkerError {
        let mut error = WorkerError::new(code, message);
        error.path = vec![self.node.clone()];
        error
    }

    /// Answers one message from the parent.
    pub fn handle(&self, message: Message) -> Message {
        match message {
            Message::Forward(forward) => match self.forward(forward) {
                Ok(result) => Message::ForwardResult(result),
                Err(error) => Message::Error(error),
            },
            Message::ForwardBatch(items) => Message::ForwardBatchResult(self.forward_batch(items)),
            Message::Truncate { session, len } => {
                let entry = self.sessions.lock().unwrap().get(&session).cloned();
                match entry {
                    Some(entry) => match entry.lock().unwrap().cache.truncate(len as usize) {
                        Ok(()) => Message::Truncate { session, len },
                        Err(err) => {
                            Message::Error(self.error(ErrorCode::Unsupported, err.to_string()))
                        }
                    },
                    None => Message::Error(self.error(
                        ErrorCode::UnknownSession,
                        format!("no session {session} to truncate"),
                    )),
                }
            }
            Message::Release { session } => {
                self.release(session);
                Message::Release { session }
            }
            Message::Reserve { session, positions } => {
                match self.reserve(session, positions as usize) {
                    Ok(()) => Message::Reserve { session, positions },
                    Err(error) => Message::Error(error),
                }
            }
            Message::Fork { from, to, len } => match self.fork(from, to, len as usize) {
                Ok(()) => Message::Fork { from, to, len },
                Err(error) => Message::Error(error),
            },
            // A forward runs to completion once started; there is nothing
            // in flight here to stop. Acknowledged so a parent need not
            // special-case it.
            Message::Retain { session, held } => match self.retain(session, held) {
                Ok(()) => Message::Retain { session, held },
                Err(error) => Message::Error(error),
            },
            Message::Save { session, key } => match self.save(session, &key) {
                Ok(()) => Message::Save { session, key },
                Err(error) => Message::Error(error),
            },
            Message::Load { session, key, len } => match self.load(session, &key, len as usize) {
                Ok(()) => Message::Load { session, key, len },
                Err(error) => Message::Error(error),
            },
            Message::Forget { key } => {
                self.forget(&key);
                Message::Forget { key }
            }
            Message::Cancel { session, step } => Message::Cancel { session, step },
            Message::Ping { nonce } => Message::Pong { nonce },
            other => Message::Error(self.error(
                ErrorCode::Unsupported,
                format!("unexpected message {:?}", std::mem::discriminant(&other)),
            )),
        }
    }

    /// Promises `session` memory for `positions` here and on the workers
    /// below, letting go of conversations nobody is using if that makes room:
    /// one taken up again is rebuilt by its parent. `0` takes a promise back.
    pub fn reserve(&self, session: u64, positions: usize) -> Result<(), WorkerError> {
        if positions == 0 {
            self.pipeline.room().forget(session);
            let _ = self.pipeline.reserve_below(session, 0);
            return Ok(());
        }
        while let Err(short) = self.pipeline.room().reserve(session, positions) {
            if !self.evict_oldest_idle(session) {
                return Err(self.error(ErrorCode::OutOfMemory, short.to_string()));
            }
        }
        self.pipeline
            .reserve_below(session, positions)
            .map_err(|error| {
                self.pipeline.room().forget(session);
                match error
                    .chain()
                    .find_map(|cause| cause.downcast_ref::<WorkerError>())
                {
                    Some(below) if below.code == ErrorCode::OutOfMemory => {
                        let mut below = below.clone();
                        below.path.insert(0, self.node.clone());
                        below
                    }
                    _ => self.error(ErrorCode::ChildLost, format!("{error:#}")),
                }
            })
    }

    /// Lets go of the session unused for longest, other than `keep`; whether
    /// there was one to let go of. Not one running, or still owed memory for
    /// a prompt on its way.
    fn evict_oldest_idle(&self, keep: u64) -> bool {
        let now = Instant::now();
        let mut sessions = self.sessions.lock().unwrap();
        let oldest = sessions
            .iter()
            .filter(|(id, _)| **id != keep && !self.pipeline.room().owes(**id))
            .filter_map(|(id, session)| {
                let session = session.try_lock().ok()?;
                let used = session.last_used;
                (!session.held && now.duration_since(used) >= self.evictable_after)
                    .then_some((*id, used))
            })
            .min_by_key(|(_, used)| *used)
            .map(|(id, _)| id);
        let Some(id) = oldest else {
            return false;
        };
        let removed = sessions.remove(&id);
        drop(sessions);
        self.pipeline.room().forget(id);
        log::info!("orangu-server: workers: session {id} let go to make room for a prompt");
        drop(removed);
        true
    }

    /// Starts session `to` with a copy of `from`'s first `len` positions:
    /// this node's rows copied, and the workers below told to do the same.
    /// Refused when `from` is unknown, `to` exists, there is no room for
    /// another session, or `from` does not hold `len` positions on the host.
    pub fn fork(&self, from: u64, to: u64, len: usize) -> Result<(), WorkerError> {
        let source = {
            let sessions = self.sessions.lock().unwrap();
            if sessions.contains_key(&to) {
                return Err(self.error(ErrorCode::Internal, format!("session {to} already exists")));
            }
            if sessions.len() >= self.max_sessions {
                return Err(self.error(
                    ErrorCode::Busy,
                    format!("already holding {} sessions", self.max_sessions),
                ));
            }
            sessions.get(&from).cloned().ok_or_else(|| {
                self.error(
                    ErrorCode::UnknownSession,
                    format!("no session {from} to copy"),
                )
            })?
        };
        let mut source = source.lock().unwrap();
        let held = self.host_rows(&mut source.cache);
        let local = &source.cache;
        if len > held || len > self.capacity {
            return Err(self.error(
                ErrorCode::PositionMismatch,
                format!("session {from} holds {held} positions here, not {len}"),
            ));
        }
        let mut cache = self.pipeline.new_cache(to, self.capacity);
        cache.copy_prefix_from(local, len);
        if let Some(remote) = cache.remote.as_mut() {
            self.pipeline
                .fork(from, to, len)
                .map_err(|e| self.error(ErrorCode::Internal, format!("{e:#}")))?;
            remote.len = len;
        }
        drop(source);
        let entry = Arc::new(Mutex::new(Session {
            cache,
            last_used: Instant::now(),
            held: false,
            trace: tracing().then(Trace::default),
        }));
        self.sessions.lock().unwrap().insert(to, entry);
        Ok(())
    }

    /// The positions `cache` holds on the host, where a copy or a file is
    /// made from: a decode on the device keeps its rows there, so they are
    /// read back first.
    fn host_rows(&self, cache: &mut KvCache) -> usize {
        if cache.layers.iter().all(|l| l.len == 0) {
            return cache.committed_len();
        }
        if let Some(vulkan) = self.pipeline.model().vulkan_backend() {
            cache.pull_decoded_rows(vulkan);
        }
        cache.host_committed_len()
    }

    /// Frees `session` here and, through its cache, on every worker below.
    /// Marks `session` held for the parent's prefix pool, or returns it to
    /// ordinary eviction, here and on every node below. Unknown here: an
    /// error when it was to be held, nothing to do when it was not.
    pub fn retain(&self, session: u64, held: bool) -> Result<(), WorkerError> {
        let entry = self.sessions.lock().unwrap().get(&session).cloned();
        match entry {
            Some(entry) => entry.lock().unwrap().held = held,
            None if held => {
                return Err(self.error(
                    ErrorCode::UnknownSession,
                    format!("no session {session} to keep"),
                ));
            }
            None => return Ok(()),
        }
        self.pipeline.retain(session, held);
        Ok(())
    }

    fn files(&self) -> Result<&PrefixFiles, WorkerError> {
        self.files
            .as_deref()
            .ok_or_else(|| self.error(ErrorCode::Unsupported, "this node saves no sessions"))
    }

    /// Writes `session`'s rows on this node's own layers under `key`, then
    /// has the workers below do the same; keeps no more files than the
    /// parent's pool holds entries, the oldest deleted first.
    pub fn save(&self, session: u64, key: &str) -> Result<(), WorkerError> {
        let files = self.files()?;
        if !super::prefixes::valid_key(key) {
            return Err(self.error(ErrorCode::Internal, format!("{key:?} is not a key")));
        }
        let entry = self.sessions.lock().unwrap().get(&session).cloned();
        let Some(entry) = entry else {
            return Err(self.error(
                ErrorCode::UnknownSession,
                format!("no session {session} to save"),
            ));
        };
        let has_below = {
            let mut state = entry.lock().unwrap();
            let held = self.host_rows(&mut state.cache);
            let cache = &state.cache;
            files
                .write(
                    key,
                    false,
                    self.pipeline.local_layers(),
                    self.capacity,
                    held,
                    &[],
                    cache,
                )
                .map_err(|e| self.error(ErrorCode::Internal, format!("{e:#}")))?;
            cache.remote.is_some()
        };
        if has_below && let Err(error) = self.pipeline.save(session, key) {
            files.remove(key, false);
            return Err(
                self.with_this_node(super::pipeline::worker_error(error, ErrorCode::Internal))
            );
        }
        files.keep_newest(crate::engine::prefix_cache::ENTRIES);
        Ok(())
    }

    /// Opens `session`, held for the parent's pool, from the first `len`
    /// positions saved under `key` on this node's own layers, then has the
    /// workers below do the same. Refused, with no session opened, when the
    /// file is missing, damaged, or not of this model, these layers, this
    /// cache's shape or this context.
    pub fn load(&self, session: u64, key: &str, len: usize) -> Result<(), WorkerError> {
        let files = self.files()?;
        {
            let sessions = self.sessions.lock().unwrap();
            if sessions.contains_key(&session) {
                return Err(self.error(
                    ErrorCode::Internal,
                    format!("session {session} already exists"),
                ));
            }
            if sessions.len() >= self.max_sessions {
                return Err(self.error(
                    ErrorCode::Busy,
                    format!("already holding {} sessions", self.max_sessions),
                ));
            }
        }
        let mut cache = self.pipeline.new_cache(session, self.capacity);
        let shape = cache.structure_tag();
        let expect = Expect {
            layers: self.pipeline.local_layers(),
            shape: &shape,
            n_ctx: self.capacity,
            len,
        };
        let (_, saved) = files
            .read(key, false, &expect)
            .map_err(|e| self.error(ErrorCode::ModelMismatch, format!("{e:#}")))?;
        cache.copy_prefix_from(&saved, len);
        drop(saved);
        if cache.remote.is_some() {
            if let Err(error) = self.pipeline.load(session, key, len) {
                // Nothing opened below: the cache's release has nothing to
                // let go of there.
                return Err(
                    self.with_this_node(super::pipeline::worker_error(error, ErrorCode::Internal))
                );
            }
            if let Some(remote) = cache.remote.as_mut() {
                remote.len = len;
            }
        }
        let entry = Arc::new(Mutex::new(Session {
            cache,
            last_used: Instant::now(),
            held: true,
            trace: tracing().then(Trace::default),
        }));
        self.sessions.lock().unwrap().insert(session, entry);
        Ok(())
    }

    /// Deletes what is saved under `key` here and below.
    pub fn forget(&self, key: &str) {
        if let Some(files) = &self.files {
            files.remove(key, false);
        }
        self.pipeline.forget(key);
    }

    /// Sessions held for the parent's prefix pool, and the bytes their
    /// positions take on this node's layers.
    pub fn retained(&self) -> (usize, u64) {
        let sessions = self.sessions.lock().unwrap();
        let room = self.pipeline.room();
        sessions
            .values()
            .filter_map(|s| {
                let s = s.try_lock().ok()?;
                s.held.then(|| room.bytes_for(s.cache.committed_len()))
            })
            .fold((0, 0), |(n, bytes), b| (n + 1, bytes + b))
    }

    pub fn release(&self, session: u64) {
        self.pipeline.room().forget(session);
        let removed = self.sessions.lock().unwrap().remove(&session);
        // Dropped after the map's lock is let go: dropping the cache
        // releases the session on this node's own workers.
        drop(removed);
    }

    /// Frees every session nobody has used for `idle`, returning how many.
    /// A parent that went away without releasing its sessions leaves them
    /// behind; this is what reclaims them.
    pub fn evict_idle(&self, idle: Duration) -> usize {
        let now = Instant::now();
        let mut sessions = self.sessions.lock().unwrap();
        let stale: Vec<u64> = sessions
            .iter()
            .filter(|(_, s)| {
                s.try_lock()
                    .is_ok_and(|s| !s.held && now.duration_since(s.last_used) >= idle)
            })
            .map(|(id, _)| *id)
            .collect();
        let removed: Vec<_> = stale.iter().filter_map(|id| sessions.remove(id)).collect();
        drop(sessions);
        for id in &stale {
            self.pipeline.room().forget(*id);
        }
        removed.len()
    }

    /// The checks a forward passes before it may touch a session: its shape,
    /// its room in the context, and the session itself — found, or started
    /// when the forward is at position 0.
    fn admit(&self, forward: &Forward) -> Result<Arc<Mutex<Session>>, WorkerError> {
        let session = forward.session;
        let start_pos = forward.start_pos as usize;
        let n_tokens = forward.tokens.len();
        let n_embd = self.pipeline.model().config().n_embd;
        if n_tokens == 0
            || forward.hidden.rows as usize != n_tokens
            || forward.hidden.width as usize != n_embd
        {
            return Err(self.error(
                ErrorCode::Internal,
                format!(
                    "{n_tokens} tokens with {} rows of {} hidden values; this model is {n_embd} \
                     wide",
                    forward.hidden.rows, forward.hidden.width
                ),
            ));
        }
        if start_pos + n_tokens > self.capacity {
            return Err(self.error(
                ErrorCode::ContextFull,
                format!(
                    "positions {start_pos}..{} do not fit in {}",
                    start_pos + n_tokens,
                    self.capacity
                ),
            ));
        }
        let mut sessions = self.sessions.lock().unwrap();
        match sessions.get(&session) {
            Some(entry) => Ok(entry.clone()),
            // A session starts at position 0 and nowhere else: a forward
            // further on for a session this node does not hold means it
            // lost the rows before it, and computing on without them would
            // be silently wrong.
            None if start_pos == 0 => {
                if sessions.len() >= self.max_sessions {
                    return Err(self.error(
                        ErrorCode::Busy,
                        format!("already holding {} sessions", self.max_sessions),
                    ));
                }
                let entry = Arc::new(Mutex::new(Session {
                    cache: self.pipeline.new_cache(session, self.capacity),
                    last_used: Instant::now(),
                    held: false,
                    trace: tracing().then(Trace::default),
                }));
                sessions.insert(session, entry.clone());
                Ok(entry)
            }
            None => Err(self.error(
                ErrorCode::UnknownSession,
                format!("no session {session} to continue at position {start_pos}"),
            )),
        }
    }

    /// With the session locked: applies the forward's rollback, checks that
    /// it starts where the session ends, and decodes its hidden state.
    fn ready(&self, state: &mut Session, forward: &Forward) -> Result<Vec<f32>, WorkerError> {
        state.last_used = Instant::now();
        if let Some(len) = forward.kv_truncate_to {
            state
                .cache
                .truncate(len as usize)
                .map_err(|err| self.error(ErrorCode::Unsupported, err.to_string()))?;
        }
        let held = state.cache.committed_len();
        if held != forward.start_pos as usize {
            return Err(self.error(
                ErrorCode::PositionMismatch,
                format!(
                    "session {} holds {held} positions, forward starts at {}",
                    forward.session, forward.start_pos
                ),
            ));
        }
        forward
            .hidden
            .decode()
            .map_err(|e| self.error(ErrorCode::Internal, e.to_string()))
    }

    /// A failure below this node, with this node put in front of its path.
    fn with_this_node(&self, mut error: WorkerError) -> WorkerError {
        if error.path.first() != Some(&self.node) {
            error.path.insert(0, self.node.clone());
        }
        error
    }

    fn result(&self, forward: &Forward, out: Vec<f32>) -> ForwardResult {
        let n_embd = self.pipeline.model().config().n_embd;
        let layers = self.pipeline.layers();
        ForwardResult {
            session: forward.session,
            step: forward.step,
            layer_start: layers.start as u32,
            layer_end: layers.end as u32,
            hidden: if forward.rows.logits() {
                // Logits go back exact: what they are sampled from.
                let n_vocab = self.pipeline.model().config().n_vocab;
                Activations::encode(&out, out.len() / n_vocab, n_vocab, ActivationFormat::F32)
            } else {
                Activations::encode(&out, out.len() / n_embd, n_embd, forward.hidden.format)
            },
        }
    }

    pub fn forward(&self, forward: Forward) -> Result<ForwardResult, WorkerError> {
        let entry = self.admit(&forward)?;
        let mut guard = entry.lock().unwrap();
        let values = self.ready(&mut guard, &forward)?;
        let state = &mut *guard;
        let out = self
            .pipeline
            .run_traced(
                &mut state.cache,
                forward.session,
                values,
                &forward.tokens,
                forward.start_pos as usize,
                forward.rows,
                state.trace.as_mut(),
            )
            .map_err(|e| match e.downcast::<WorkerError>() {
                Ok(child) => self.with_this_node(child),
                Err(e) => self.error(ErrorCode::Internal, format!("{e:#}")),
            })?;
        Ok(self.result(&forward, out))
    }

    /// Several sessions' forwards as one batch through this node's layers
    /// and on to its workers as one message each, every forward answered
    /// on its own. A session named twice is refused the second time.
    pub fn forward_batch(&self, forwards: Vec<Forward>) -> Vec<Result<ForwardResult, WorkerError>> {
        let mut answers: Vec<Option<Result<ForwardResult, WorkerError>>> =
            (0..forwards.len()).map(|_| None).collect();
        let mut admitted: Vec<(usize, Arc<Mutex<Session>>)> = Vec::new();
        for (i, forward) in forwards.iter().enumerate() {
            if forwards[..i].iter().any(|f| f.session == forward.session) {
                answers[i] = Some(Err(self.error(
                    ErrorCode::Internal,
                    format!("session {} named twice in one batch", forward.session),
                )));
                continue;
            }
            match self.admit(forward) {
                Ok(entry) => admitted.push((i, entry)),
                Err(error) => answers[i] = Some(Err(error)),
            }
        }
        // Locked in one order, whoever asks: two batches naming the same
        // sessions cannot each hold one the other waits for.
        admitted.sort_by_key(|(i, _)| forwards[*i].session);
        let mut guards: Vec<(usize, std::sync::MutexGuard<'_, Session>)> = admitted
            .iter()
            .map(|(i, entry)| (*i, entry.lock().unwrap()))
            .collect();
        let mut rows = Vec::new();
        let mut order = Vec::new();
        for (i, guard) in guards.iter_mut() {
            let forward = &forwards[*i];
            match self.ready(guard, forward) {
                Ok(values) => {
                    order.push(*i);
                    rows.push(BatchRow {
                        cache: &mut guard.cache,
                        session: forward.session,
                        hidden: values,
                        tokens: forward.tokens.clone(),
                        start_pos: forward.start_pos as usize,
                        rows: forward.rows,
                    });
                }
                Err(error) => answers[*i] = Some(Err(error)),
            }
        }
        let results = self.pipeline.run_batch(&mut rows, None);
        drop(rows);
        drop(guards);
        for (i, result) in order.into_iter().zip(results) {
            answers[i] = Some(
                result
                    .map(|out| self.result(&forwards[i], out))
                    .map_err(|e| self.with_this_node(e)),
            );
        }
        answers
            .into_iter()
            .map(|a| a.expect("every forward is answered"))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workers::pipeline::fixture::{self, N_CTX, N_EMBD};
    use crate::workers::protocol::{ActivationFormat, Rows};
    use crate::workers::room::KvRoom;
    use crate::workers::stage::LoopbackStage;

    fn store(max_sessions: usize) -> SessionStore {
        let model = fixture::model();
        SessionStore::new(
            Arc::new(LayerPipeline::new(model, 1..3, vec![]).unwrap()),
            N_CTX,
            max_sessions,
        )
        .named("w1:8400")
    }

    fn forward(session: u64, start_pos: u32, tokens: &[u32]) -> Forward {
        let hidden = vec![0.1; tokens.len() * N_EMBD];
        Forward {
            session,
            step: 1,
            start_pos,
            rows: Rows::All,
            kv_truncate_to: None,
            tokens: tokens.to_vec(),
            hidden: Activations::encode(&hidden, tokens.len(), N_EMBD, ActivationFormat::F32),
        }
    }

    #[test]
    fn a_forward_must_continue_where_the_last_one_ended() {
        let store = store(4);
        let result = store.forward(forward(1, 0, &[1, 2, 3])).unwrap();
        assert_eq!((result.layer_start, result.layer_end), (1, 3));
        assert_eq!(result.hidden.rows, 3);
        let err = store.forward(forward(1, 2, &[4])).unwrap_err();
        assert_eq!(err.code, ErrorCode::PositionMismatch);
        assert_eq!(err.path, ["w1:8400"]);
        store.forward(forward(1, 3, &[4])).unwrap();
        // A rollback carried on the forward makes an earlier start valid.
        let mut again = forward(1, 2, &[9]);
        again.kv_truncate_to = Some(2);
        store.forward(again).unwrap();
    }

    /// A worker that lost a session — restarted, or evicted it — refuses to
    /// continue it rather than computing without its rows.
    #[test]
    fn a_session_starts_at_position_zero_or_not_at_all() {
        let store = store(4);
        let err = store.forward(forward(5, 7, &[1])).unwrap_err();
        assert_eq!(err.code, ErrorCode::UnknownSession);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn sessions_are_bounded_by_the_slot_count() {
        let store = store(2);
        store.forward(forward(1, 0, &[1])).unwrap();
        store.forward(forward(2, 0, &[1])).unwrap();
        assert_eq!(
            store.forward(forward(3, 0, &[1])).unwrap_err().code,
            ErrorCode::Busy
        );
        store.release(1);
        store.forward(forward(3, 0, &[1])).unwrap();
    }

    #[test]
    fn the_context_is_bounded_by_the_assigned_n_ctx() {
        let store = store(2);
        let tokens: Vec<u32> = (0..N_CTX as u32 + 1).map(|t| t % 60).collect();
        let err = store.forward(forward(1, 0, &tokens)).unwrap_err();
        assert_eq!(err.code, ErrorCode::ContextFull);
    }

    #[test]
    fn a_hidden_state_of_the_wrong_shape_is_refused() {
        let store = store(2);
        let mut bad = forward(1, 0, &[1, 2]);
        bad.hidden = Activations::encode(&[0.0; 10], 2, 5, ActivationFormat::F32);
        assert_eq!(store.forward(bad).unwrap_err().code, ErrorCode::Internal);
    }

    #[test]
    fn idle_sessions_are_evicted() {
        let store = store(4);
        store.forward(forward(1, 0, &[1])).unwrap();
        assert_eq!(store.evict_idle(Duration::from_secs(3600)), 0);
        assert_eq!(store.evict_idle(Duration::ZERO), 1);
        assert_eq!(store.len(), 0);
    }

    /// A session held but not used does not keep the node busy; one used
    /// within the period, or running a forward, does.
    #[test]
    fn a_held_session_is_quiet_once_unused() {
        let store = store(4);
        assert!(store.quiet(Duration::from_secs(3600)), "nothing held");
        store.forward(forward(1, 0, &[1])).unwrap();
        assert!(!store.quiet(Duration::from_secs(3600)));
        assert!(store.quiet(Duration::ZERO));
        let session = store.sessions.lock().unwrap()[&1].clone();
        let _running = session.lock().unwrap();
        assert!(!store.quiet(Duration::ZERO), "a forward is running");
    }

    #[test]
    fn the_control_messages_are_answered() {
        let store = store(4);
        assert_eq!(
            store.handle(Message::Ping { nonce: 3 }),
            Message::Pong { nonce: 3 }
        );
        store.forward(forward(1, 0, &[1, 2])).unwrap();
        assert_eq!(
            store.handle(Message::Truncate { session: 1, len: 1 }),
            Message::Truncate { session: 1, len: 1 }
        );
        store.forward(forward(1, 1, &[2])).unwrap();
        match store.handle(Message::Truncate { session: 9, len: 0 }) {
            Message::Error(e) => assert_eq!(e.code, ErrorCode::UnknownSession),
            other => panic!("{other:?}"),
        }
        match store.handle(Message::Auth { proof: [0; 32] }) {
            Message::Error(e) => assert_eq!(e.code, ErrorCode::Unsupported),
            other => panic!("{other:?}"),
        }
        match store.handle(Message::ForwardBatch(vec![
            forward(2, 0, &[1]),
            forward(3, 4, &[1]),
        ])) {
            Message::ForwardBatchResult(items) => {
                assert!(items[0].is_ok());
                assert_eq!(
                    items[1].as_ref().unwrap_err().code,
                    ErrorCode::UnknownSession
                );
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_batch_answers_each_forward_on_its_own() {
        let store = store(4);
        store.forward(forward(1, 0, &[1, 2])).unwrap();
        let answers = store.forward_batch(vec![
            forward(1, 2, &[3]),
            forward(2, 0, &[1]),
            forward(3, 5, &[1]),
            forward(1, 3, &[4]),
        ]);
        assert!(answers[0].is_ok() && answers[1].is_ok());
        assert_eq!(
            answers[2].as_ref().unwrap_err().code,
            ErrorCode::UnknownSession
        );
        let twice = answers[3].as_ref().unwrap_err();
        assert!(twice.message.contains("twice"), "{twice}");
        assert_eq!(answers[0].as_ref().unwrap().hidden.rows, 1);
    }

    /// A failure deep in the tree arrives at the top naming every node on
    /// the way down.
    #[test]
    fn a_child_s_error_carries_the_path_to_it() {
        let model = fixture::model();
        let leaf = Arc::new(
            SessionStore::new(
                Arc::new(LayerPipeline::new(model.clone(), 3..4, vec![]).unwrap()),
                N_CTX,
                1,
            )
            .named("leaf"),
        );
        let middle = SessionStore::new(
            Arc::new(
                LayerPipeline::new(
                    model,
                    1..3,
                    vec![Box::new(LoopbackStage::new(
                        leaf.clone(),
                        ActivationFormat::F32,
                    ))],
                )
                .unwrap(),
            ),
            N_CTX,
            4,
        )
        .named("middle");
        middle.forward(forward(1, 0, &[1])).unwrap();
        // The leaf holds one session, so a second is refused down there.
        let err = middle.forward(forward(2, 0, &[1])).unwrap_err();
        assert_eq!(err.code, ErrorCode::Busy);
        assert_eq!(err.path, ["middle", "leaf"]);
    }

    const MIB: u64 = 1 << 20;

    /// A store with `free` as its machine's memory, a position costing 1 MiB.
    fn with_room(free: impl Fn() -> u64 + Send + Sync + 'static) -> SessionStore {
        let pipeline = LayerPipeline::new(fixture::model(), 1..3, vec![])
            .unwrap()
            .with_room(KvRoom::with_free_memory(MIB, Box::new(free)));
        SessionStore::new(Arc::new(pipeline), N_CTX, 8).named("w1:8400")
    }

    fn reserve(store: &SessionStore, session: u64, positions: u32) -> Message {
        store.handle(Message::Reserve { session, positions })
    }

    #[test]
    fn a_prompt_with_no_room_is_refused_saying_where_and_by_how_much() {
        let store = with_room(|| 10 * MIB);
        assert_eq!(
            reserve(&store, 1, 10),
            Message::Reserve {
                session: 1,
                positions: 10
            }
        );
        let Message::Error(error) = reserve(&store, 2, 11) else {
            panic!("not refused");
        };
        assert_eq!(error.code, ErrorCode::OutOfMemory);
        assert_eq!(error.path, vec!["w1:8400".to_string()]);
        assert!(error.message.contains("11.00 MiB"), "{error}");
    }

    #[test]
    fn a_forward_settles_what_was_promised_and_a_release_gives_it_back() {
        let store = with_room(|| 10 * MIB);
        assert!(matches!(reserve(&store, 1, 3), Message::Reserve { .. }));
        assert_eq!(store.pipeline.room().promised_bytes(), 3 * MIB);
        store.forward(forward(1, 0, &[1, 2])).unwrap();
        assert_eq!(store.pipeline.room().promised_bytes(), MIB);
        store.release(1);
        assert_eq!(store.pipeline.room().promised_bytes(), 0);
    }

    #[test]
    fn asking_for_nothing_takes_a_promise_back() {
        let store = with_room(|| 10 * MIB);
        assert!(matches!(reserve(&store, 1, 10), Message::Reserve { .. }));
        assert!(matches!(reserve(&store, 1, 0), Message::Reserve { .. }));
        assert_eq!(store.pipeline.room().promised_bytes(), 0);
        assert!(matches!(reserve(&store, 2, 10), Message::Reserve { .. }));
    }

    /// Memory that frees up as sessions go, as it does when their rows are
    /// dropped: `room` holds `free`, plus `each` for every session let go of
    /// of the `sessions` it started with.
    fn room_as_sessions_go(
        each: u64,
        count: Arc<std::sync::atomic::AtomicUsize>,
    ) -> impl Fn() -> u64 + Send + Sync + 'static {
        use std::sync::atomic::Ordering;
        move || each * count.fetch_add(1, Ordering::Relaxed) as u64
    }

    #[test]
    fn a_conversation_nobody_is_using_makes_room_oldest_first() {
        let store = with_room(room_as_sessions_go(
            10 * MIB,
            Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        ))
        .evictable_after(Duration::ZERO);
        store.forward(forward(1, 0, &[1])).unwrap();
        std::thread::sleep(Duration::from_millis(5));
        store.forward(forward(2, 0, &[1])).unwrap();
        assert!(matches!(reserve(&store, 3, 10), Message::Reserve { .. }));
        let held = store.sessions.lock().unwrap();
        assert!(!held.contains_key(&1), "the oldest goes first");
        assert!(held.contains_key(&2), "and no more than it takes");
    }

    /// A session held for the parent's prefix pool outlives both of the
    /// worker's own evictions; one beside it that is not held does not, and
    /// a session let go of the hold is evicted as any other.
    #[test]
    fn a_retained_session_outlives_idle_and_shortfall_eviction() {
        let store = with_room(|| 0).evictable_after(Duration::ZERO);
        store.forward(forward(1, 0, &[1, 2])).unwrap();
        store.forward(forward(2, 0, &[1])).unwrap();
        assert_eq!(
            store.handle(Message::Retain {
                session: 1,
                held: true
            }),
            Message::Retain {
                session: 1,
                held: true
            }
        );
        assert_eq!(store.retained(), (1, 2 * MIB));

        // The shortfall evicts what it may — session 2 — and refuses then.
        assert!(matches!(reserve(&store, 3, 1), Message::Error(_)));
        let held: Vec<u64> = store.sessions.lock().unwrap().keys().copied().collect();
        assert_eq!(held, vec![1]);

        store.forward(forward(2, 0, &[1])).unwrap();
        assert_eq!(store.evict_idle(Duration::ZERO), 1, "only the unheld one");
        assert_eq!(store.len(), 1);

        // Let go of the hold, it goes as any session does.
        store.retain(1, false).unwrap();
        assert_eq!(store.retained(), (0, 0));
        assert_eq!(store.evict_idle(Duration::ZERO), 1);
        assert_eq!(store.len(), 0);

        // Held only once it exists; letting go of an unknown one is nothing.
        assert!(matches!(
            store.handle(Message::Retain {
                session: 9,
                held: true
            }),
            Message::Error(WorkerError {
                code: ErrorCode::UnknownSession,
                ..
            })
        ));
        assert!(store.retain(9, false).is_ok());
    }

    /// The hold reaches every node below, as a fork does.
    #[test]
    fn a_retain_reaches_the_nodes_below() {
        let leaf = Arc::new(
            SessionStore::new(
                Arc::new(
                    LayerPipeline::new(fixture::model(), 3..fixture::N_LAYER, vec![]).unwrap(),
                ),
                N_CTX,
                4,
            )
            .named("leaf"),
        );
        let stage = LoopbackStage::new(leaf.clone(), ActivationFormat::F32);
        let middle = SessionStore::new(
            Arc::new(LayerPipeline::new(fixture::model(), 1..3, vec![Box::new(stage)]).unwrap()),
            N_CTX,
            4,
        )
        .named("middle");
        middle.forward(forward(1, 0, &[1])).unwrap();
        middle.retain(1, true).unwrap();
        assert_eq!(middle.retained().0, 1);
        assert_eq!(leaf.retained().0, 1);
        assert_eq!(leaf.evict_idle(Duration::ZERO), 0);
        middle.release(1);
        assert_eq!((middle.len(), leaf.len()), (0, 0));
    }

    fn prefix_files(tag: &str, node: &str) -> Arc<PrefixFiles> {
        let root = std::env::temp_dir().join(format!(
            "orangu-session-prefixes-{tag}-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let files = PrefixFiles::under(&root, [5; 32], node);
        let _ = std::fs::remove_dir_all(files.dir());
        Arc::new(files)
    }

    fn saving(
        layers: Range<usize>,
        files: &Arc<PrefixFiles>,
        children: Vec<Box<dyn super::super::pipeline::Stage>>,
    ) -> Arc<SessionStore> {
        Arc::new(
            SessionStore::new(
                Arc::new(LayerPipeline::new(fixture::model(), layers, children).unwrap()),
                N_CTX,
                8,
            )
            .with_prefixes(Some(files.clone())),
        )
    }

    /// Save, Load and Forget each reach the node below, as a fork does; a
    /// session opened from disk is held, and continues where it was saved.
    #[test]
    fn a_saved_session_opens_again_here_and_below() {
        let leaf_files = prefix_files("below", "leaf");
        let middle_files = prefix_files("below", "middle");
        let leaf = saving(3..fixture::N_LAYER, &leaf_files, vec![]);
        let stage = LoopbackStage::new(leaf.clone(), ActivationFormat::F32);
        let middle = saving(1..3, &middle_files, vec![Box::new(stage)]);
        middle.forward(forward(1, 0, &[1, 2, 3])).unwrap();
        let key = super::super::prefixes::new_key();
        let save = Message::Save {
            session: 1,
            key: key.clone(),
        };
        assert_eq!(middle.handle(save.clone()), save);
        assert_eq!(middle.saved().len(), 1);
        assert_eq!(leaf.saved()[0].layer_start, 3);
        assert_eq!(leaf.saved()[0].len, 3);

        let load = Message::Load {
            session: 2,
            key: key.clone(),
            len: 3,
        };
        assert_eq!(middle.handle(load.clone()), load);
        assert_eq!((middle.retained().0, leaf.retained().0), (1, 1));
        let resumed = middle.forward(forward(2, 3, &[4])).unwrap();
        let along = middle.forward(forward(1, 3, &[4])).unwrap();
        assert_eq!(resumed.hidden, along.hidden);

        // Opened once: a second session of the same id is refused.
        assert!(matches!(middle.handle(load), Message::Error(_)));
        assert_eq!(
            middle.handle(Message::Forget { key: key.clone() }),
            Message::Forget { key: key.clone() }
        );
        assert!(middle.saved().is_empty() && leaf.saved().is_empty());
        let Message::Error(error) = middle.handle(Message::Load {
            session: 3,
            key,
            len: 3,
        }) else {
            panic!("a forgotten prefix opened");
        };
        assert_eq!(error.code, ErrorCode::ModelMismatch);
        assert_eq!(middle.len(), 2, "nothing opened for the refusal");
    }

    /// Rows of more positions than saved, a key that is not one, or a node
    /// that saves nothing: refused, and nothing written.
    #[test]
    fn what_cannot_be_saved_or_opened_is_refused() {
        let files = prefix_files("refused", "w1");
        let store = saving(1..3, &files, vec![]);
        store.forward(forward(1, 0, &[1, 2])).unwrap();
        let key = super::super::prefixes::new_key();
        assert!(store.save(1, "../../escape").is_err());
        assert!(store.save(9, &key).is_err());
        store.save(1, &key).unwrap();
        assert_eq!(
            store.load(2, &key, 3).unwrap_err().code,
            ErrorCode::ModelMismatch
        );
        assert_eq!(store.len(), 1);
        let plain = self::store(4);
        plain.forward(forward(1, 0, &[1])).unwrap();
        assert_eq!(
            plain.save(1, &key).unwrap_err().code,
            ErrorCode::Unsupported
        );
    }

    /// A node keeps no more saved sessions than a pool holds entries: the
    /// oldest go first.
    #[test]
    fn a_node_keeps_no_more_saved_sessions_than_a_pool_holds() {
        use crate::engine::prefix_cache::ENTRIES;
        let files = prefix_files("cap", "w1");
        let store = saving(1..3, &files, vec![]);
        store.forward(forward(1, 0, &[1, 2])).unwrap();
        let keys: Vec<String> = (0..ENTRIES + 2)
            .map(|_| super::super::prefixes::new_key())
            .collect();
        for key in &keys {
            store.save(1, key).unwrap();
            std::thread::sleep(Duration::from_millis(5));
        }
        let held: Vec<String> = store.saved().into_iter().map(|s| s.key).collect();
        assert_eq!(held, keys[2..].to_vec());
    }

    #[test]
    fn a_conversation_in_recent_use_is_not_let_go() {
        let store = with_room(|| 0);
        store.forward(forward(1, 0, &[1])).unwrap();
        assert!(matches!(reserve(&store, 2, 1), Message::Error(_)));
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn a_session_with_a_prompt_on_its_way_is_not_let_go() {
        let free = Arc::new(std::sync::atomic::AtomicU64::new(100 * MIB));
        let reading = free.clone();
        let store = with_room(move || reading.load(std::sync::atomic::Ordering::Relaxed))
            .evictable_after(Duration::ZERO);
        store.forward(forward(1, 0, &[1])).unwrap();
        assert!(matches!(reserve(&store, 1, 50), Message::Reserve { .. }));
        free.store(MIB, std::sync::atomic::Ordering::Relaxed);
        assert!(matches!(reserve(&store, 2, 60), Message::Error(_)));
        assert_eq!(store.len(), 1, "it is still owed");
    }

    /// The last stage refuses: the node's own promise goes too, the refusal
    /// keeps the path it came by, and the answer is the worker's own.
    #[test]
    fn a_refusal_from_below_gives_back_this_node_s_promise() {
        let below = LayerPipeline::new(fixture::model(), 3..fixture::N_LAYER, vec![])
            .unwrap()
            .with_room(KvRoom::with_free_memory(MIB, Box::new(|| 0)));
        let below = Arc::new(SessionStore::new(Arc::new(below), N_CTX, 8).named("w1:8400"));
        let stage = LoopbackStage::new(below, ActivationFormat::F32);
        let pipeline = LayerPipeline::new(fixture::model(), 1..3, vec![Box::new(stage)])
            .unwrap()
            .with_room(KvRoom::with_free_memory(MIB, Box::new(|| 100 * MIB)));
        let store = SessionStore::new(Arc::new(pipeline), N_CTX, 8).named("w0:8400");
        let Message::Error(error) = reserve(&store, 1, 10) else {
            panic!("not refused");
        };
        assert_eq!(error.code, ErrorCode::OutOfMemory);
        assert_eq!(
            error.path,
            vec!["w0:8400".to_string(), "w1:8400".to_string()]
        );
        assert_eq!(store.pipeline.room().promised_bytes(), 0);
    }
}
