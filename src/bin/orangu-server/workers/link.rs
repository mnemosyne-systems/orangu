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

//! A parent's connection to one worker: dialing it, the handshake, and then
//! any number of requests in flight at once over the one connection.
//!
//! Each request carries an id and waits for the answer with that id, which
//! a reader thread hands over as it arrives — so one sequence's forward at
//! this worker does not wait for another's to come back, and while the
//! worker computes one, it can already be sent the next.
//!
//! Any failure to reach the worker — a refused dial, a closed connection, a
//! worker silent while a request is out, a reply that does not come within
//! its limit — is reported as [`ErrorCode::ChildLost`] naming its address,
//! and drops the connection with every request still waiting on it: after a
//! timeout nobody knows where in the stream the other side is, so the only
//! safe next step is a fresh handshake, which planning does.
//!
//! A dead worker and a slow one are told apart. A machine that loses power
//! closes nothing, so while a request is out the link pings, and any byte
//! from the worker counts as a sign of life ([`SILENCE`]). `[workers].timeout`
//! is left for a worker that answers pings and never finishes ([`forward_limit`]).

use super::auth;
use super::metrics::LinkMetrics;
use super::protocol::{
    Capacity, ErrorCode, FEATURES, Hello, Message, PROTOCOL_VERSION, WorkerError, check_path,
    read_frame, read_message, write_message,
};
use super::stage::Exchange;
use super::transport::{self, Tls};
use anyhow::Result;
use std::collections::HashMap;
use std::io::{BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// What a parent echoes off a new worker to time the link: 4 MiB, 13 ms
/// each way at 2.5 Gbit/s.
const ECHO_BYTES: usize = 4 << 20;

/// How long a worker may say nothing while a request is out. It answers
/// pings from its read loop, whatever its forwards are doing, so this is
/// independent of how much work a request is.
const SILENCE: Duration = Duration::from_secs(15);

/// How many other parts a forward carrying a prompt may wait behind on the
/// worker, which holds that many for the stages below it.
const PARTS_AHEAD: usize = 3;

/// The longest a request of `tokens` may take on a worker that answers
/// pings. A decode step gets `timeout`. A prompt's forward gets it for each
/// 128 tokens and for each part it may wait behind, since its time grows with
/// both and a slow worker is not a lost one.
fn forward_limit(timeout: Duration, tokens: usize) -> Duration {
    if tokens <= 1 {
        return timeout;
    }
    timeout.saturating_mul((PARTS_AHEAD + tokens.div_ceil(128)) as u32)
}

/// When a connection last heard anything from its worker.
struct LastHeard {
    start: Instant,
    millis: AtomicU64,
}

impl LastHeard {
    fn new() -> Self {
        Self {
            start: Instant::now(),
            millis: AtomicU64::new(0),
        }
    }

    fn now(&self) {
        self.millis
            .store(self.start.elapsed().as_millis() as u64, Ordering::Relaxed);
    }

    fn since(&self) -> Duration {
        self.start
            .elapsed()
            .saturating_sub(Duration::from_millis(self.millis.load(Ordering::Relaxed)))
    }
}

/// A connection's reads, each noted in [`LastHeard`] as it happens, so that a
/// frame slow to arrive is a sign of life before it has ended.
struct HeardReader {
    inner: Box<dyn Read + Send>,
    heard: Arc<LastHeard>,
}

impl Read for HeardReader {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        if n > 0 {
            self.heard.now();
        }
        Ok(n)
    }
}

/// Failures in a row that leave a worker out of the plans.
const FAILURES_TO_LEAVE_OUT: u32 = 2;
/// How long the first time, doubling with every failure after that.
const FIRST_OUT: Duration = Duration::from_secs(2 * 60);
const LONGEST_OUT: Duration = Duration::from_secs(30 * 60);
/// A failure this long after the last one starts the count over. A node below
/// the top sees no request finish, so nothing else forgives its workers.
const FORGOTTEN: Duration = Duration::from_secs(60 * 60);

/// A worker's failures in a row, and until when they leave it out.
///
/// What counts: a connection lost with a request out, and a worker that takes
/// the connection and fails the handshake. Each costs a request a replay, or
/// planning a wait. A worker that cannot be dialled costs neither, and one
/// that refuses is working: both come back as soon as they can.
#[derive(Default)]
struct Backoff {
    failures: u32,
    last: Option<Instant>,
    out_until: Option<Instant>,
}

impl Backoff {
    /// Counts a failure at `now`; how long it leaves the worker out, if it does.
    fn failed(&mut self, now: Instant) -> Option<Duration> {
        if self
            .last
            .is_some_and(|last| now.duration_since(last) >= FORGOTTEN)
        {
            self.failures = 0;
        }
        self.failures = self.failures.saturating_add(1);
        self.last = Some(now);
        let doublings = self.failures.checked_sub(FAILURES_TO_LEAVE_OUT)?;
        let out = FIRST_OUT
            .checked_mul(1u32 << doublings.min(16))
            .map_or(LONGEST_OUT, |out| out.min(LONGEST_OUT));
        self.out_until = Some(now + out);
        Some(out)
    }

    /// How much longer the worker is left out, at `now`.
    fn out_for(&self, now: Instant) -> Option<Duration> {
        self.out_until
            .and_then(|until| until.checked_duration_since(now))
            .filter(|left| !left.is_zero())
    }
}

/// A worker's [`Backoff`], shared with its connections.
struct Failures {
    addr: String,
    backoff: Mutex<Backoff>,
}

impl Failures {
    fn note(&self) {
        let mut backoff = self.backoff.lock().unwrap();
        if let Some(out) = backoff.failed(Instant::now()) {
            log::warn!(
                "orangu-server: worker {} failed {} times in a row; left out for {} s",
                self.addr,
                backoff.failures,
                out.as_secs()
            );
        }
    }
}

/// What the worker said about itself in the handshake.
#[derive(Clone, Debug, Default)]
pub struct ChildInfo {
    pub node: String,
    pub subtree: Vec<String>,
    pub capacity: Capacity,
    /// The extensions it understands (`protocol::FEATURES`).
    pub features: u64,
    /// The link's round trip and bandwidth (bytes a second each way), as
    /// timed after the handshake; `None` from a worker that cannot echo.
    pub link: Option<(Duration, f64)>,
    /// The model it serves, as its handshake said.
    pub model: Option<super::protocol::ModelIdentity>,
}

type Answer = Result<(Message, usize), WorkerError>;

/// One live connection: requests are written under `writer`'s lock and
/// wait in `pending` for the reader thread to hand them their answer.
struct Connection {
    writer: Mutex<Box<dyn Write + Send>>,
    tcp: TcpStream,
    pending: Mutex<HashMap<u64, SyncSender<Answer>>>,
    next_id: AtomicU64,
    dead: AtomicBool,
    heard: Arc<LastHeard>,
    /// Requests out that the worker answers from its read loop, where it
    /// answers no ping until it is done.
    unwatched: AtomicUsize,
    failures: Arc<Failures>,
}

impl Connection {
    /// Ends the connection, failing every request still waiting on it. Counts
    /// against the worker once, however many were waiting, and not at all if
    /// none was: a worker going away idle has cost nothing.
    fn fail(&self, error: &WorkerError) {
        let waiting = !self.pending.lock().unwrap().is_empty();
        if !self.dead.swap(true, Ordering::Relaxed) && waiting {
            self.failures.note();
        }
        self.end(error);
    }

    /// Ends the connection because this node is done with it.
    fn let_go(&self, error: &WorkerError) {
        self.dead.store(true, Ordering::Relaxed);
        self.end(error);
    }

    fn end(&self, error: &WorkerError) {
        let _ = self.tcp.shutdown(std::net::Shutdown::Both);
        for (_, waiter) in self.pending.lock().unwrap().drain() {
            let _ = waiter.try_send(Err(error.clone()));
        }
    }

    fn is_live(&self) -> bool {
        !self.dead.load(Ordering::Relaxed)
    }
}

pub struct ChildLink {
    /// The worker's address as `[workers].workers` lists it.
    pub addr: String,
    timeout: Duration,
    /// How long a worker may say nothing while a request is out.
    silence: Duration,
    connect_timeout: Duration,
    tls: Option<Tls>,
    pub metrics: LinkMetrics,
    connection: Mutex<Option<Arc<Connection>>>,
    info: Mutex<Option<ChildInfo>>,
    /// The path the current connection was made with.
    path: Mutex<Vec<String>>,
    /// Whether the current outage has been reported.
    unreachable: AtomicBool,
    /// Why the last plan left it out, said once until the reason changes.
    left_out: Mutex<Option<String>>,
    failures: Arc<Failures>,
}

impl ChildLink {
    #[cfg(test)]
    pub fn new(addr: impl Into<String>, timeout: Duration, connect_timeout: Duration) -> Self {
        Self::with_tls(addr, timeout, connect_timeout, None)
    }

    /// A link that dials with TLS when `tls` has a client side.
    pub fn with_tls(
        addr: impl Into<String>,
        timeout: Duration,
        connect_timeout: Duration,
        tls: Option<Tls>,
    ) -> Self {
        let addr = addr.into();
        Self {
            failures: Arc::new(Failures {
                addr: addr.clone(),
                backoff: Mutex::default(),
            }),
            addr,
            timeout,
            silence: SILENCE.min(timeout),
            connect_timeout,
            tls,
            metrics: LinkMetrics::default(),
            connection: Mutex::new(None),
            info: Mutex::new(None),
            path: Mutex::new(Vec::new()),
            unreachable: AtomicBool::new(false),
            left_out: Mutex::new(None),
        }
    }

    /// Records why a plan left the worker out; whether that is news — a
    /// reason other than the last one — so a worker left out for the same
    /// reason at every plan is said once.
    pub fn note_left_out(&self, reason: &str) -> bool {
        let mut last = self.left_out.lock().unwrap();
        if last.as_deref() == Some(reason) {
            return false;
        }
        *last = Some(reason.to_string());
        true
    }

    /// A plan used the worker: the next reason to leave it out is news.
    pub fn clear_left_out(&self) {
        *self.left_out.lock().unwrap() = None;
    }

    /// Marks the worker unreachable; whether that is news — the first
    /// failure since it was last reached.
    pub fn note_unreachable(&self) -> bool {
        !self.unreachable.swap(true, Ordering::Relaxed)
    }

    /// How much longer plans leave the worker out for its failures.
    pub fn out_for(&self) -> Option<Duration> {
        self.failures
            .backoff
            .lock()
            .unwrap()
            .out_for(Instant::now())
    }

    /// The failures counted against the worker since it was last forgiven.
    pub fn failures_in_a_row(&self) -> u32 {
        self.failures.backoff.lock().unwrap().failures
    }

    /// A request ran on the worker from start to finish.
    pub fn forgive(&self) {
        *self.failures.backoff.lock().unwrap() = Backoff::default();
    }

    /// A failure with no connection to fail.
    #[cfg(test)]
    pub fn count_failure(&self) {
        self.failures.note();
    }

    /// The wait is over, its failures still counted.
    #[cfg(test)]
    pub fn end_wait(&self) {
        self.failures.backoff.lock().unwrap().out_until = None;
    }

    /// The live connection when there is one made with the same `path`,
    /// otherwise a fresh [`Self::connect`].
    pub fn ensure_connected(
        &self,
        secret: Option<&str>,
        path: &[String],
    ) -> Result<ChildInfo, WorkerError> {
        if *self.path.lock().unwrap() == path
            && let Some(info) = self.info()
        {
            return Ok(info);
        }
        self.connect(secret, path)
    }

    fn lost(&self, why: impl std::fmt::Display) -> WorkerError {
        let mut error = WorkerError::new(ErrorCode::ChildLost, format!("{}: {why}", self.addr));
        error.path = vec![self.addr.clone()];
        error
    }

    fn live(&self) -> Option<Arc<Connection>> {
        self.connection
            .lock()
            .unwrap()
            .as_ref()
            .filter(|c| c.is_live())
            .cloned()
    }

    pub fn is_connected(&self) -> bool {
        self.live().is_some()
    }

    /// The worker's handshake answer, while connected.
    pub fn info(&self) -> Option<ChildInfo> {
        if !self.is_connected() {
            return None;
        }
        self.info.lock().unwrap().clone()
    }

    pub fn disconnect(&self) {
        if let Some(connection) = self.connection.lock().unwrap().take() {
            connection.let_go(&self.lost("disconnected"));
        }
        *self.info.lock().unwrap() = None;
    }

    /// Dials the worker and runs the handshake, replacing any connection
    /// there was. `path` holds the node ids from the root down to this
    /// node.
    pub fn connect(&self, secret: Option<&str>, path: &[String]) -> Result<ChildInfo, WorkerError> {
        self.disconnect();
        let addrs = self
            .addr
            .to_socket_addrs()
            .map_err(|e| self.lost(format!("cannot resolve: {e}")))?;
        let mut last = None;
        let mut stream = None;
        for addr in addrs {
            match TcpStream::connect_timeout(&addr, self.connect_timeout) {
                Ok(s) => {
                    stream = Some(s);
                    break;
                }
                Err(e) => last = Some(e),
            }
        }
        let stream = stream.ok_or_else(|| {
            self.lost(match last {
                Some(e) => format!("cannot connect: {e}"),
                None => "no address to connect to".to_string(),
            })
        })?;
        let shook = self.handshake(stream, secret, path);
        if shook
            .as_ref()
            .is_err_and(|error| error.code == ErrorCode::ChildLost)
        {
            self.failures.note();
        }
        shook
    }

    /// The handshake over a connection just dialled.
    fn handshake(
        &self,
        stream: TcpStream,
        secret: Option<&str>,
        path: &[String],
    ) -> Result<ChildInfo, WorkerError> {
        let _ = stream.set_nodelay(true);
        // A worker answers the handshake straight away, whatever it is doing,
        // and a hung process's kernel still accepts a dial. Planning waits on
        // this, and a request on planning.
        stream
            .set_read_timeout(Some(self.silence))
            .map_err(|e| self.lost(e))?;
        // A worker that stopped reading fills the socket's buffers, and a
        // write would then wait for as long as TCP keeps retrying. This holds
        // per write, so a large frame on a slow link is not cut short.
        stream
            .set_write_timeout(Some(self.silence))
            .map_err(|e| self.lost(e))?;
        let split = transport::client(self.tls.as_ref(), stream, &self.addr)
            .map_err(|e| self.lost(format!("TLS: {e:#}")))?;
        let heard = Arc::new(LastHeard::new());
        let mut reader = BufReader::new(Box::new(HeardReader {
            inner: split.reader,
            heard: heard.clone(),
        }) as Box<dyn Read + Send>);
        let mut writer = split.writer;
        let mut round_trip = |message: &Message| -> Result<Message, WorkerError> {
            write_message(&mut *writer, 0, message).map_err(|e| self.lost(e))?;
            read_message(&mut reader).map_err(|e| self.lost(e))
        };

        let nonce = auth::nonce();
        let reply = round_trip(&Message::Hello(Hello {
            version: PROTOCOL_VERSION,
            features: FEATURES,
            nonce,
            path: path.to_vec(),
        }))?;
        let ack = match reply {
            Message::HelloAck(ack) => ack,
            Message::Error(error) => return Err(self.refused(error)),
            other => return Err(self.lost(format!("unexpected handshake reply {other:?}"))),
        };
        if ack.version != PROTOCOL_VERSION {
            return Err(self.refused(WorkerError::new(
                ErrorCode::Unsupported,
                format!(
                    "speaks worker protocol {} and this node {PROTOCOL_VERSION}: run the same \
                     orangu-server release on every node",
                    ack.version
                ),
            )));
        }
        if !auth::parent_accepts(secret, &nonce, &ack.nonce, &ack.proof) {
            return Err(self.refused(WorkerError::new(
                ErrorCode::Unauthorized,
                "could not prove [workers].secret: the secrets differ, or it has none",
            )));
        }
        check_path(path, &ack.subtree).map_err(|e| self.refused(e))?;
        let proof = auth::parent_proof(secret, &nonce, &ack.nonce);
        match round_trip(&Message::Auth { proof })? {
            Message::Ready => {}
            Message::Error(error) => return Err(self.refused(error)),
            other => return Err(self.lost(format!("unexpected handshake reply {other:?}"))),
        }
        // From here the reader thread waits for answers as long as it
        // takes; each request keeps its own clock.
        let _ = split.tcp.set_read_timeout(None);
        let connection = Arc::new(Connection {
            writer: Mutex::new(writer),
            tcp: split.tcp,
            pending: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(1),
            dead: AtomicBool::new(false),
            heard,
            unwatched: AtomicUsize::new(0),
            failures: self.failures.clone(),
        });
        {
            let connection = connection.clone();
            let lost = self.lost("the connection closed");
            std::thread::Builder::new()
                .name("orangu-workers-link".to_string())
                .spawn(move || read_answers(connection, reader, lost))
                .map_err(|e| self.lost(e))?;
        }
        {
            let connection = connection.clone();
            let silence = self.silence;
            let lost = self.lost(format!(
                "nothing heard for {} s with a request out",
                silence.as_secs()
            ));
            std::thread::Builder::new()
                .name("orangu-workers-watch".to_string())
                .spawn(move || keep_watch(connection, silence, lost))
                .map_err(|e| self.lost(e))?;
        }
        let mut info = ChildInfo {
            node: ack.node,
            subtree: ack.subtree,
            capacity: ack.capacity,
            features: ack.features,
            link: None,
            model: ack.model,
        };
        *self.connection.lock().unwrap() = Some(connection);
        if info.features & super::protocol::FEATURE_ECHO != 0 {
            info.link = self.time_link();
        }
        *self.info.lock().unwrap() = Some(info.clone());
        *self.path.lock().unwrap() = path.to_vec();
        if self.unreachable.swap(false, Ordering::Relaxed) {
            log::info!("orangu-server: worker {} is back", self.addr);
        }
        Ok(info)
    }

    /// The link's round trip — the fastest of three pings — and its
    /// bandwidth, from echoing [`ECHO_BYTES`] (the better of two) less a
    /// round trip — each way, the echo going there and coming back in turn.
    /// `None` when the worker does not answer.
    fn time_link(&self) -> Option<(Duration, f64)> {
        let mut rtt = Duration::MAX;
        for nonce in 0..3u64 {
            let at = Instant::now();
            match self.request(&Message::Ping { nonce }) {
                Ok(Message::Pong { nonce: n }) if n == nonce => rtt = rtt.min(at.elapsed()),
                _ => return None,
            }
        }
        // The better of two: the first also pays for the buffers growing.
        let mut echo = Duration::MAX;
        for _ in 0..2 {
            let at = Instant::now();
            match self.request(&Message::Echo {
                data: vec![0x5a; ECHO_BYTES],
            }) {
                Ok(Message::Echo { data }) if data.len() == ECHO_BYTES => {
                    echo = echo.min(at.elapsed())
                }
                _ => return None,
            }
        }
        let moving = echo.saturating_sub(rtt).as_secs_f64().max(1e-6);
        Some((rtt, 2.0 * ECHO_BYTES as f64 / moving))
    }

    /// A refusal from the worker, named after it.
    fn refused(&self, mut error: WorkerError) -> WorkerError {
        if error.path.first() != Some(&self.addr) {
            error.path.insert(0, self.addr.clone());
        }
        error
    }

    /// One request and its answer, alongside whatever else is in flight.
    /// A failure — the connection's, or no answer in time — ends the
    /// connection.
    pub fn request(&self, message: &Message) -> Result<Message, WorkerError> {
        let tokens = match message {
            Message::Forward(forward) => forward.tokens.len(),
            Message::ForwardBatch(forwards) => {
                forwards.iter().map(|f| f.tokens.len()).max().unwrap_or(0)
            }
            _ => 0,
        };
        self.send_and_wait(message, forward_limit(self.timeout, tokens))
    }

    /// A request the worker answers from its read loop, which answers no
    /// ping until it is done: an assignment, which may include a download.
    /// Silence means nothing here, and only `timeout` bounds it.
    pub fn request_within(
        &self,
        message: &Message,
        timeout: Duration,
    ) -> Result<Message, WorkerError> {
        let Some(connection) = self.live() else {
            return Err(self.lost("not connected"));
        };
        connection.unwatched.fetch_add(1, Ordering::Relaxed);
        let reply = self.send_and_wait(message, timeout);
        connection.unwatched.fetch_sub(1, Ordering::Relaxed);
        reply
    }

    fn send_and_wait(&self, message: &Message, limit: Duration) -> Result<Message, WorkerError> {
        let Some(connection) = self.live() else {
            return Err(self.lost("not connected"));
        };
        let id = connection.next_id.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = mpsc::sync_channel(1);
        connection.pending.lock().unwrap().insert(id, tx);
        let started = Instant::now();
        let sent = {
            let mut writer = connection.writer.lock().unwrap();
            write_message(&mut **writer, id, message)
        };
        let failed = |error: WorkerError| {
            self.metrics.failed();
            connection.fail(&error);
            *self.info.lock().unwrap() = None;
            Err(error)
        };
        match sent {
            Ok(bytes) => self.metrics.sent(bytes),
            Err(e) => return failed(self.lost(e)),
        }
        match rx.recv_timeout(limit) {
            Ok(Ok((reply, bytes))) => {
                self.metrics.received(bytes);
                if matches!(message, Message::Forward(_) | Message::ForwardBatch(_)) {
                    self.metrics.observe_forward(started.elapsed());
                }
                Ok(reply)
            }
            Ok(Err(error)) => failed(error),
            Err(RecvTimeoutError::Timeout) => failed(self.lost(format!(
                "no answer within {} s ([workers].timeout)",
                limit.as_secs()
            ))),
            Err(RecvTimeoutError::Disconnected) => failed(self.lost("the connection closed")),
        }
    }

    /// A ping when nothing is in flight: `None` when something is (which
    /// proves it alive anyway) or when not connected, otherwise whether the
    /// worker answered.
    pub fn ping_if_idle(&self) -> Option<bool> {
        let connection = self.live()?;
        if !connection.pending.lock().unwrap().is_empty() {
            return None;
        }
        let nonce: u64 = rand::random();
        Some(matches!(
            self.request(&Message::Ping { nonce }),
            Ok(Message::Pong { nonce: n }) if n == nonce
        ))
    }
}

/// The watch thread: while a request is out, pings the worker, and ends the
/// connection when `silence` goes by with nothing heard. An idle connection
/// is not watched: [`ChildLink::ping_if_idle`] makes a request of it, and
/// that request is.
fn keep_watch(connection: Arc<Connection>, silence: Duration, lost: WorkerError) {
    while connection.is_live() {
        std::thread::sleep(silence / 6);
        let waiting = !connection.pending.lock().unwrap().is_empty();
        if !waiting || connection.unwatched.load(Ordering::Relaxed) > 0 {
            // The clock starts when there is something to hear.
            connection.heard.now();
            continue;
        }
        let Ok(mut writer) = connection.writer.try_lock() else {
            // A frame is going out, which the socket's write timeout watches,
            // and the worker reads frames in turn before it answers a ping.
            connection.heard.now();
            continue;
        };
        if connection.heard.since() >= silence {
            drop(writer);
            connection.fail(&lost);
            return;
        }
        // Nobody waits for this one: the reader drops the answer, having
        // noted that it came.
        let nonce = connection.next_id.fetch_add(1, Ordering::Relaxed);
        if write_message(&mut **writer, nonce, &Message::Ping { nonce }).is_err() {
            drop(writer);
            connection.fail(&lost);
            return;
        }
    }
}

/// The reader thread: hands each answer to the request with its id, until
/// the connection ends — and then fails whatever is still waiting.
fn read_answers(
    connection: Arc<Connection>,
    mut reader: BufReader<Box<dyn Read + Send>>,
    lost: WorkerError,
) {
    while connection.is_live() {
        match read_frame(&mut reader) {
            Ok((id, message, bytes)) => {
                if let Some(waiter) = connection.pending.lock().unwrap().remove(&id) {
                    let _ = waiter.try_send(Ok((message, bytes)));
                }
            }
            Err(_) => break,
        }
    }
    connection.fail(&lost);
}

impl Exchange for std::sync::Arc<ChildLink> {
    fn exchange(&self, message: &Message) -> Result<Message> {
        self.request(message).map_err(anyhow::Error::new)
    }

    fn name(&self) -> String {
        self.addr.clone()
    }

    fn features(&self) -> u64 {
        self.info().map_or(0, |info| info.features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    const PATIENT: Duration = Duration::from_secs(30);
    const SHORT: Duration = Duration::from_millis(300);

    /// A worker that does the handshake and then whatever `then` says, over
    /// the socket it was given. Returns its address.
    fn worker(then: impl FnOnce(BufReader<TcpStream>, TcpStream) + Send + 'static) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut writer = stream.try_clone().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            read_message(&mut reader).unwrap();
            let ack = Message::HelloAck(super::super::protocol::HelloAck {
                version: PROTOCOL_VERSION,
                features: 0,
                nonce: [0; 32],
                proof: [0; 32],
                node: "worker".to_string(),
                subtree: vec!["worker".to_string()],
                capacity: Capacity::default(),
                model: None,
            });
            write_message(&mut writer, 0, &ack).unwrap();
            read_message(&mut reader).unwrap();
            write_message(&mut writer, 0, &Message::Ready).unwrap();
            then(reader, writer);
        });
        addr
    }

    fn link_to(addr: String) -> ChildLink {
        let mut link = ChildLink::new(addr, PATIENT, Duration::from_secs(2));
        link.silence = SHORT;
        link.connect(None, &["top".to_string()]).unwrap();
        link
    }

    fn truncate() -> Message {
        Message::Truncate { session: 1, len: 0 }
    }

    #[test]
    fn a_forward_is_allowed_for_its_size_and_what_it_waits_behind() {
        let timeout = Duration::from_secs(60);
        assert_eq!(forward_limit(timeout, 0), timeout);
        assert_eq!(forward_limit(timeout, 1), timeout);
        // The short last part of a prompt waits behind the parts before it.
        assert_eq!(forward_limit(timeout, 2), 4 * timeout);
        assert_eq!(forward_limit(timeout, 128), 4 * timeout);
        assert_eq!(forward_limit(timeout, 256), 5 * timeout);
    }

    /// A machine that lost power closes nothing: reads go on, answers never
    /// come. That is told within the silence, not the forward's timeout.
    #[test]
    fn a_silent_worker_is_lost_long_before_the_timeout() {
        let addr = worker(|mut reader, _writer| while read_frame(&mut reader).is_ok() {});
        let link = link_to(addr);
        let started = Instant::now();
        let error = link.request(&truncate()).unwrap_err();
        assert_eq!(error.code, ErrorCode::ChildLost);
        assert_eq!(error.path, vec![link.addr.clone()]);
        assert!(started.elapsed() < Duration::from_secs(5), "{error}");
        assert!(!link.is_connected());
    }

    /// Slow is not dead: a worker that answers pings keeps its link for as
    /// long as its forward takes.
    #[test]
    fn a_worker_answering_pings_may_take_its_time() {
        let addr = worker(|mut reader, mut writer| {
            let started = Instant::now();
            let mut pending = None;
            while let Ok((id, message, _)) = read_frame(&mut reader) {
                match message {
                    Message::Ping { nonce } => {
                        write_message(&mut writer, id, &Message::Pong { nonce }).unwrap();
                    }
                    _ => pending = Some(id),
                }
                if started.elapsed() > 4 * SHORT
                    && let Some(id) = pending.take()
                {
                    write_message(&mut writer, id, &Message::Ready).unwrap();
                }
            }
        });
        let link = link_to(addr);
        let started = Instant::now();
        assert_eq!(link.request(&truncate()).unwrap(), Message::Ready);
        assert!(started.elapsed() >= 2 * SHORT, "it answered too soon");
        assert!(link.is_connected());
    }

    /// Bytes are a sign of life: a frame longer in the arriving than the
    /// silence does not lose the worker, though no frame has ended.
    #[test]
    fn a_frame_on_its_way_counts_as_a_sign_of_life() {
        let addr = worker(|mut reader, mut writer| {
            let (id, _, _) = read_frame(&mut reader).unwrap();
            let frame = Message::Echo {
                data: vec![1; 1 << 10],
            }
            .to_frame(id);
            for piece in frame.chunks(frame.len() / 10) {
                writer.write_all(piece).unwrap();
                std::thread::sleep(SHORT / 2);
            }
        });
        let link = link_to(addr);
        assert!(matches!(
            link.request(&truncate()),
            Ok(Message::Echo { .. })
        ));
    }

    /// A worker does an assignment in its read loop, downloads and all, and
    /// answers no ping meanwhile; it is given all of the time asked for.
    #[test]
    fn an_assignment_is_not_held_to_the_silence() {
        let addr = worker(|mut reader, mut writer| {
            let (id, _, _) = read_frame(&mut reader).unwrap();
            std::thread::sleep(4 * SHORT);
            write_message(&mut writer, id, &Message::Ready).unwrap();
            while read_frame(&mut reader).is_ok() {}
        });
        let link = link_to(addr);
        assert_eq!(
            link.request_within(&truncate(), PATIENT).unwrap(),
            Message::Ready
        );
    }

    /// A hung process's kernel still accepts a dial. Planning waits on the
    /// handshake while a request waits on planning, so it is held to the
    /// silence as well.
    #[test]
    fn a_worker_silent_in_the_handshake_is_not_waited_for() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let mut link = ChildLink::new(addr, PATIENT, Duration::from_secs(2));
        link.silence = SHORT;
        let started = Instant::now();
        let error = link.connect(None, &["top".to_string()]).unwrap_err();
        assert_eq!(error.code, ErrorCode::ChildLost);
        assert!(started.elapsed() < Duration::from_secs(5), "{error}");
    }

    /// A worker that stopped reading fills the socket's buffers. A frame
    /// larger than they hold ends in an error, not a write that never
    /// returns.
    #[test]
    fn a_stalled_socket_does_not_hold_a_request() {
        let addr = worker(|_reader, _writer| std::thread::sleep(Duration::from_secs(15)));
        let link = link_to(addr);
        let started = Instant::now();
        let error = link
            .request(&Message::Echo {
                data: vec![0; 256 << 20],
            })
            .unwrap_err();
        assert_eq!(error.code, ErrorCode::ChildLost);
        assert!(started.elapsed() < Duration::from_secs(5), "{error}");
    }

    #[test]
    fn failures_in_a_row_leave_a_worker_out_for_longer_each_time() {
        let now = Instant::now();
        let mut backoff = Backoff::default();
        assert_eq!(backoff.failed(now), None, "once is not left out");
        assert_eq!(backoff.out_for(now), None);
        let minutes = [2, 4, 8, 16, 30, 30, 30];
        for m in minutes {
            assert_eq!(backoff.failed(now), Some(Duration::from_secs(m * 60)));
        }
        assert_eq!(backoff.out_for(now), Some(LONGEST_OUT));
        assert_eq!(backoff.out_for(now + LONGEST_OUT), None, "the wait ends");
        backoff.failures = u32::MAX;
        assert_eq!(backoff.failed(now), Some(LONGEST_OUT), "never overflows");
    }

    #[test]
    fn a_failure_long_after_the_last_starts_the_count_over() {
        let now = Instant::now();
        let mut backoff = Backoff::default();
        backoff.failed(now);
        assert_eq!(
            backoff.failed(now + FORGOTTEN - Duration::from_secs(1)),
            Some(FIRST_OUT)
        );
        let later = now + FORGOTTEN * 3;
        assert_eq!(backoff.failed(later), None);
        assert_eq!(backoff.failures, 1);
    }

    #[test]
    fn a_worker_that_cannot_be_dialled_is_not_counted() {
        let addr = {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.local_addr().unwrap().to_string()
        };
        let link = ChildLink::new(addr, PATIENT, SHORT);
        for _ in 0..3 {
            assert!(link.connect(None, &["top".to_string()]).is_err());
        }
        assert_eq!(link.failures_in_a_row(), 0);
        assert_eq!(link.out_for(), None);
    }

    #[test]
    fn a_worker_failing_the_handshake_is_counted() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        std::thread::spawn(move || {
            // Takes every dial and closes it.
            for stream in listener.incoming() {
                drop(stream);
            }
        });
        let link = ChildLink::new(addr, PATIENT, SHORT);
        let path = ["top".to_string()];
        assert!(link.connect(None, &path).is_err());
        assert_eq!(link.failures_in_a_row(), 1);
        assert_eq!(link.out_for(), None);
        assert!(link.connect(None, &path).is_err());
        assert!(link.out_for().is_some_and(|left| left > FIRST_OUT / 2));
        link.forgive();
        assert_eq!(link.failures_in_a_row(), 0);
        assert_eq!(link.out_for(), None);
    }

    #[test]
    fn a_connection_lost_counts_once_however_many_requests_were_out() {
        let addr = worker(|_reader, stream| {
            std::thread::sleep(Duration::from_millis(300));
            let _ = stream.shutdown(std::net::Shutdown::Both);
        });
        let link = Arc::new(link_to(addr));
        let waiting: Vec<_> = (0..4)
            .map(|_| {
                let link = link.clone();
                std::thread::spawn(move || link.request(&truncate()))
            })
            .collect();
        for request in waiting {
            assert!(request.join().unwrap().is_err());
        }
        assert_eq!(link.failures_in_a_row(), 1);
    }

    #[test]
    fn a_connection_ended_with_nothing_out_is_not_counted() {
        let addr = worker(|_reader, stream| {
            let _ = stream.shutdown(std::net::Shutdown::Both);
        });
        let link = link_to(addr);
        let deadline = Instant::now() + PATIENT;
        while link.is_connected() {
            assert!(Instant::now() < deadline, "the close was not seen");
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(link.failures_in_a_row(), 0, "closed by the worker");

        let addr = worker(|_reader, _stream| std::thread::sleep(SHORT));
        let link = link_to(addr);
        link.disconnect();
        assert_eq!(link.failures_in_a_row(), 0, "let go by this node");
    }
}
