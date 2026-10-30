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
//! reply that does not come within `[workers].timeout` — is reported as
//! [`ErrorCode::ChildLost`] naming its address, and drops the connection
//! with every request still waiting on it: after a timeout nobody knows
//! where in the stream the other side is, so the only safe next step is a
//! fresh handshake, which planning does.

use super::auth;
use super::metrics::LinkMetrics;
use super::protocol::{
    Capacity, DiscoveryRecord, ErrorCode, FEATURES, Hello, Message, PROTOCOL_VERSION, WorkerError,
    check_path, read_frame, read_message, write_message,
};
use super::stage::Exchange;
use super::transport::{self, Tls};
use anyhow::Result;
use std::collections::HashMap;
use std::io::{BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// What a parent echoes off a new worker to time the link: 4 MiB, 13 ms
/// each way at 2.5 Gbit/s.
const ECHO_BYTES: usize = 4 << 20;

/// What the worker said about itself in the handshake.
#[derive(Clone, Debug, Default)]
pub struct ChildInfo {
    pub node: String,
    pub subtree: Vec<String>,
    pub capacity: Capacity,
    /// The cards this worker handed over in the handshake: its own
    /// metrics address first, then every node below it. Passed upward
    /// with this parent's own handshake, so the cards climb one hop at
    /// a time.
    pub discovery: Vec<DiscoveryRecord>,
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
}

impl Connection {
    /// Ends the connection, failing every request still waiting on it.
    fn fail(&self, error: &WorkerError) {
        self.dead.store(true, Ordering::Relaxed);
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
        Self {
            addr: addr.into(),
            timeout,
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
            connection.fail(&self.lost("disconnected"));
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
        let _ = stream.set_nodelay(true);
        // The handshake may take no longer than a request.
        stream
            .set_read_timeout(Some(self.timeout))
            .map_err(|e| self.lost(e))?;
        let split = transport::client(self.tls.as_ref(), stream, &self.addr)
            .map_err(|e| self.lost(format!("TLS: {e:#}")))?;
        let mut reader = BufReader::new(split.reader);
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
        });
        {
            let connection = connection.clone();
            let lost = self.lost("the connection closed");
            std::thread::Builder::new()
                .name("orangu-workers-link".to_string())
                .spawn(move || read_answers(connection, reader, lost))
                .map_err(|e| self.lost(e))?;
        }
        let mut info = ChildInfo {
            node: ack.node,
            subtree: ack.subtree,
            capacity: ack.capacity,
            discovery: ack.discovery,
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
        self.request_within(message, self.timeout)
    }

    /// [`Self::request`], waiting up to `timeout` for the answer rather
    /// than the link's own.
    pub fn request_within(
        &self,
        message: &Message,
        timeout: Duration,
    ) -> Result<Message, WorkerError> {
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
        match rx.recv_timeout(timeout) {
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
                self.timeout.as_secs()
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
