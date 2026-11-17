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

//! Byte ranges of a node's model file, served to the nodes of its channel
//! that fetch a model in part (`[workers].download = range`), so such a
//! node needs no Hugging Face at all when another node holds the file.
//!
//! The serving side is a node that holds its whole model, in one file:
//! it advertises [`FEATURE_FETCH`] in its probe answers, and answers a
//! connection that opens with [`Message::FetchHello`] — authenticated with
//! `[workers].secret` the way a parent is, over TLS when the channel uses
//! it — with byte ranges of that file and nothing else. Any node that proves
//! the secret may fetch, not only the one that claimed this node: a node
//! fetching in part needs the model's header before anyone can claim it.
//!
//! While the serving node runs requests — its own, or forwards for a
//! parent — its ranges go out at [`BUSY_BYTES_PER_SECOND`] in all, so a
//! download does not take the link and the disk from the requests; idle,
//! or waiting for a worker to take the layers it assigned (its requests wait
//! for that worker), at whatever the link carries. The wait is on the fetch
//! connection's own thread, never on one that runs the model.
//!
//! The fetching side is a [`PeerSource`], one of the sources a
//! [`PartialModel`](orangu::model_download::PartialModel) takes its bytes
//! from.

use super::auth;
use super::protocol::{
    ErrorCode, FEATURE_FETCH, MAX_FETCH_BYTES, Message, WorkerError, read_frame, read_message,
    write_message,
};
use super::transport::{self, Tls};
use anyhow::{Context, Result, anyhow, bail, ensure};
use orangu::model_download::RangeSource;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// How fast a node running requests serves ranges, across every fetching
/// node: enough to bring a worker in over minutes, little enough to leave
/// the link and the disk to the requests.
pub const BUSY_BYTES_PER_SECOND: u64 = 8 << 20;

/// How long a fetching node waits for a connection, and for each answer.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
const ANSWER_TIMEOUT: Duration = Duration::from_secs(60);

/// The model file a node serves.
#[derive(Clone, Debug)]
pub struct ServedFile {
    path: PathBuf,
    size: u64,
    /// Its name, and the snapshot directory it is in: where a fetching
    /// node puts its partial copy, so the Hub's copy would go to the same
    /// place.
    name: String,
    commit: String,
}

impl ServedFile {
    /// `path`, when it is a whole model in a single file.
    pub fn new(path: &Path) -> Option<Self> {
        let name = path.file_name()?.to_str()?.to_string();
        if !orangu::model_download::single_file_model(&name) {
            return None;
        }
        let size = std::fs::metadata(path).ok().filter(|m| m.is_file())?.len();
        let parent = path.parent();
        let commit = parent
            .filter(|dir| {
                dir.parent()
                    .and_then(Path::file_name)
                    .is_some_and(|n| n == "snapshots")
            })
            .and_then(Path::file_name)
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        Some(Self {
            path: path.to_path_buf(),
            size,
            name,
            commit,
        })
    }
}

/// Whether `offset..offset + len` is a range one [`Message::FetchRange`]
/// may ask of a `size`-byte file.
pub fn range_allowed(size: u64, offset: u64, len: u64) -> Result<(), String> {
    if len == 0 || len > MAX_FETCH_BYTES {
        return Err(format!(
            "{len} bytes asked for at once; 1 to {MAX_FETCH_BYTES} may be"
        ));
    }
    match offset.checked_add(len) {
        Some(end) if end <= size => Ok(()),
        _ => Err(format!(
            "bytes {offset}..{} are outside the {size}-byte model file",
            offset.saturating_add(len)
        )),
    }
}

/// The shared allowance of a node's fetch connections while it is busy.
#[derive(Default)]
pub struct Pace {
    /// When the bytes already let out have had their time.
    next: Mutex<Option<Instant>>,
}

impl Pace {
    /// Waits until `bytes` more may go out: at once when the node is not
    /// `busy`, else once every byte let out before has had its time at
    /// [`BUSY_BYTES_PER_SECOND`].
    pub fn admit(&self, bytes: u64, busy: bool) {
        if !busy {
            *self.next.lock().unwrap() = None;
            return;
        }
        let cost = Duration::from_secs_f64(bytes as f64 / BUSY_BYTES_PER_SECOND as f64);
        let wait = {
            let mut next = self.next.lock().unwrap();
            let now = Instant::now();
            let start = next.filter(|at| *at > now).unwrap_or(now);
            *next = Some(start + cost);
            start - now
        };
        std::thread::sleep(wait);
    }
}

/// What a node serving a file tells the node fetching it.
pub struct Offer<'a> {
    pub file: &'a ServedFile,
    /// The served model's [`super::protocol::ModelIdentity::header_hash`].
    pub model: [u8; 32],
    pub secret: Option<&'a str>,
}

/// The serving side of a connection that opened with
/// [`Message::FetchHello`] carrying `nonce`: the rest of the handshake, then
/// ranges until the fetching node goes away. `busy` says whether the node
/// runs requests, for [`Pace`].
pub fn serve(
    reader: &mut dyn Read,
    writer: &mut dyn Write,
    nonce: [u8; 32],
    offer: Option<Offer<'_>>,
    pace: &Pace,
    busy: &dyn Fn() -> bool,
) {
    let refuse = |writer: &mut dyn Write, code: ErrorCode, message: &str| {
        let _ = write_message(writer, 0, &Message::Error(WorkerError::new(code, message)));
    };
    let Some(offer) = offer else {
        refuse(
            writer,
            ErrorCode::Unsupported,
            "this node holds no whole model file to serve",
        );
        return;
    };
    let own = auth::nonce();
    let ack = Message::FetchAck {
        features: FEATURE_FETCH,
        nonce: own,
        proof: auth::worker_proof(offer.secret, &nonce, &own),
        model: offer.model,
        size: offer.file.size,
        file: offer.file.name.clone(),
        commit: offer.file.commit.clone(),
    };
    if write_message(writer, 0, &ack).is_err() {
        return;
    }
    let proof = match read_message(reader) {
        Ok(Message::Auth { proof }) => proof,
        _ => return,
    };
    if !auth::worker_accepts(offer.secret, &nonce, &own, &proof) {
        log::warn!("orangu-server: refused a fetching node: it could not prove [workers].secret");
        refuse(
            writer,
            ErrorCode::Unauthorized,
            "the fetching node could not prove [workers].secret",
        );
        return;
    }
    let mut file = match std::fs::File::open(&offer.file.path) {
        Ok(file) => file,
        Err(error) => {
            refuse(
                writer,
                ErrorCode::Internal,
                &format!("the model file cannot be read: {error}"),
            );
            return;
        }
    };
    if write_message(writer, 0, &Message::Ready).is_err() {
        return;
    }
    // A range is a few bytes: what the handshake's frame limit allows is
    // all a fetching node is ever read with.
    while let Ok(message) = read_message(reader) {
        let reply = match message {
            Message::FetchRange { offset, len } => {
                match range_allowed(offer.file.size, offset, len) {
                    Err(why) => Message::Error(WorkerError::new(ErrorCode::Unsupported, why)),
                    Ok(()) => {
                        let mut data = vec![0u8; len as usize];
                        match file
                            .seek(SeekFrom::Start(offset))
                            .and_then(|_| file.read_exact(&mut data))
                        {
                            Ok(()) => {
                                pace.admit(len, busy());
                                Message::FetchData { offset, data }
                            }
                            Err(error) => Message::Error(WorkerError::new(
                                ErrorCode::Internal,
                                format!("reading the model file: {error}"),
                            )),
                        }
                    }
                }
            }
            Message::Ping { nonce } => Message::Pong { nonce },
            _ => Message::Error(WorkerError::new(
                ErrorCode::Unsupported,
                "a fetch connection carries ranges only",
            )),
        };
        if write_message(writer, 0, &reply).is_err() {
            return;
        }
    }
}

/// What a serving node said of its file in its [`Message::FetchAck`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FetchInfo {
    pub model: [u8; 32],
    pub size: u64,
    pub file: String,
    pub commit: String,
}

struct Connection {
    reader: BufReader<Box<dyn Read + Send>>,
    writer: Box<dyn Write + Send>,
    tcp: std::sync::Arc<TcpStream>,
}

impl Drop for Connection {
    fn drop(&mut self) {
        let _ = self.tcp.shutdown(std::net::Shutdown::Both);
    }
}

/// Another node's copy of the model, by [`Message::FetchRange`]: one
/// connection, made again when it breaks.
pub struct PeerSource {
    key: String,
    addr: String,
    secret: Option<String>,
    tls: Option<Tls>,
    /// What the node must go on serving: the file of its first handshake.
    info: Mutex<Option<FetchInfo>>,
    connection: Mutex<Option<Connection>>,
}

impl PeerSource {
    /// The node at `addr`, named `key` in messages; nothing is dialled
    /// until bytes are asked for, or [`Self::info`].
    pub fn new(key: String, addr: String, secret: Option<String>, tls: Option<Tls>) -> Self {
        Self {
            key,
            addr,
            secret,
            tls,
            info: Mutex::new(None),
            connection: Mutex::new(None),
        }
    }

    /// What the node serves, from its handshake.
    pub fn info(&self) -> Result<FetchInfo> {
        let mut connection = self.connection.lock().unwrap();
        if connection.is_none() {
            *connection = Some(self.connect()?);
        }
        Ok(self
            .info
            .lock()
            .unwrap()
            .clone()
            .expect("set by a handshake"))
    }

    fn connect(&self) -> Result<Connection> {
        let addr = self
            .addr
            .to_socket_addrs()
            .with_context(|| format!("{} does not resolve", self.addr))?
            .next()
            .ok_or_else(|| anyhow!("{} names no address", self.addr))?;
        let tcp = TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT)
            .with_context(|| format!("cannot reach {}", self.addr))?;
        tcp.set_read_timeout(Some(ANSWER_TIMEOUT))?;
        tcp.set_write_timeout(Some(ANSWER_TIMEOUT))?;
        let _ = tcp.set_nodelay(true);
        let split = transport::client(self.tls.as_ref(), tcp, &self.addr)
            .with_context(|| format!("TLS with {}", self.addr))?;
        let mut connection = Connection {
            reader: BufReader::new(split.reader),
            writer: split.writer,
            tcp: split.tcp,
        };
        let nonce = auth::nonce();
        write_message(&mut *connection.writer, 0, &Message::FetchHello { nonce })?;
        let (features, info, theirs, proof) = match read_message(&mut connection.reader)
            .with_context(|| format!("{} does not serve ranges", self.addr))?
        {
            Message::FetchAck {
                features,
                nonce: theirs,
                proof,
                model,
                size,
                file,
                commit,
            } => (
                features,
                FetchInfo {
                    model,
                    size,
                    file,
                    commit,
                },
                theirs,
                proof,
            ),
            Message::Error(error) => bail!("{} refused: {error}", self.addr),
            other => bail!("{} answered {other:?}", self.addr),
        };
        ensure!(
            features & FEATURE_FETCH != 0,
            "{} does not serve ranges",
            self.addr
        );
        let secret = self.secret.as_deref();
        ensure!(
            auth::parent_accepts(secret, &nonce, &theirs, &proof),
            "{} could not prove [workers].secret",
            self.addr
        );
        write_message(
            &mut *connection.writer,
            0,
            &Message::Auth {
                proof: auth::parent_proof(secret, &nonce, &theirs),
            },
        )?;
        match read_message(&mut connection.reader)? {
            Message::Ready => {}
            Message::Error(error) => bail!("{} refused: {error}", self.addr),
            other => bail!("{} answered {other:?}", self.addr),
        }
        let mut known = self.info.lock().unwrap();
        match known.as_ref() {
            Some(before) if *before != info => {
                bail!("{} serves another file than it did", self.addr)
            }
            _ => *known = Some(info),
        }
        Ok(connection)
    }

    fn ask(connection: &mut Connection, offset: u64, len: u64) -> Result<Vec<u8>> {
        write_message(
            &mut *connection.writer,
            0,
            &Message::FetchRange { offset, len },
        )?;
        match read_frame(&mut connection.reader)?.1 {
            Message::FetchData { offset: at, data } if at == offset => {
                ensure!(
                    data.len() as u64 == len,
                    "{} bytes sent for {len}",
                    data.len()
                );
                Ok(data)
            }
            Message::Error(error) => Err(anyhow!("refused: {error}")),
            other => Err(anyhow!("answered {other:?}")),
        }
    }
}

impl RangeSource for PeerSource {
    fn key(&self) -> String {
        self.key.clone()
    }

    /// Asked again once on a fresh connection when the first breaks.
    fn read(&self, offset: u64, len: u64) -> Result<Vec<u8>> {
        let mut connection = self.connection.lock().unwrap();
        let mut failure = None;
        for _ in 0..2 {
            if connection.is_none() {
                *connection = Some(self.connect()?);
            }
            match Self::ask(connection.as_mut().expect("connected"), offset, len) {
                Ok(data) => return Ok(data),
                Err(error) => {
                    *connection = None;
                    failure = Some(error);
                }
            }
        }
        Err(failure
            .expect("asked")
            .context(format!("{} at {}", self.key, self.addr)))
    }

    fn chunk(&self) -> u64 {
        MAX_FETCH_BYTES
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    /// A listener serving `body` as a model file, with `secret`, until the
    /// test ends; `busy` is what the pace is told.
    fn serving(
        body: &[u8],
        secret: Option<&'static str>,
        busy: Arc<AtomicBool>,
    ) -> (String, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let snapshot = dir.path().join("models--u--m/snapshots/c0ffee");
        std::fs::create_dir_all(&snapshot).unwrap();
        let path = snapshot.join("m-Q8_0.gguf");
        std::fs::write(&path, body).unwrap();
        let file = ServedFile::new(&path).expect("a single model file");
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let pace = Arc::new(Pace::default());
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { break };
                let (file, pace, busy) = (file.clone(), pace.clone(), busy.clone());
                std::thread::spawn(move || {
                    let split = transport::server(None, stream).unwrap();
                    let mut reader = BufReader::new(split.reader);
                    let mut writer = split.writer;
                    if let Ok(Message::FetchHello { nonce }) = read_message(&mut reader) {
                        serve(
                            &mut reader,
                            &mut *writer,
                            nonce,
                            Some(Offer {
                                file: &file,
                                model: [7; 32],
                                secret,
                            }),
                            &pace,
                            &|| busy.load(Ordering::Relaxed),
                        );
                    }
                });
            }
        });
        (addr, dir)
    }

    fn body(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 7 % 253) as u8).collect()
    }

    #[test]
    fn a_peer_serves_its_file_by_range() {
        let body = body(20_000);
        let (addr, _dir) = serving(&body, Some("s"), Arc::new(AtomicBool::new(false)));
        let peer = PeerSource::new("peer".into(), addr, Some("s".into()), None);
        assert_eq!(
            peer.info().unwrap(),
            FetchInfo {
                model: [7; 32],
                size: body.len() as u64,
                file: "m-Q8_0.gguf".to_string(),
                commit: "c0ffee".to_string(),
            }
        );
        assert_eq!(peer.read(100, 5000).unwrap(), body[100..5100]);
        assert_eq!(peer.read(19_999, 1).unwrap(), body[19_999..]);
    }

    /// A range past the end, an empty one or one too large is refused, and
    /// the connection serves on.
    #[test]
    fn a_range_outside_the_file_is_refused() {
        let body = body(1000);
        let (addr, _dir) = serving(&body, None, Arc::new(AtomicBool::new(false)));
        let peer = PeerSource::new("peer".into(), addr, None, None);
        assert!(peer.read(990, 11).is_err());
        assert!(peer.read(u64::MAX, 2).is_err());
        assert!(peer.read(0, 0).is_err());
        assert!(peer.read(0, MAX_FETCH_BYTES + 1).is_err());
        assert_eq!(peer.read(990, 10).unwrap(), body[990..]);
        assert!(range_allowed(1000, 0, 1000).is_ok());
        assert!(range_allowed(1000, 1, 1000).is_err());
    }

    /// A fetching node with the wrong secret, or none, gets nothing; a
    /// serving node that cannot prove the fetching node's secret is not
    /// believed.
    #[test]
    fn a_fetch_needs_the_secret() {
        let body = body(1000);
        let (addr, _dir) = serving(&body, Some("right"), Arc::new(AtomicBool::new(false)));
        for secret in [Some("wrong"), None] {
            let peer = PeerSource::new("peer".into(), addr.clone(), secret.map(Into::into), None);
            let error = peer.read(0, 10).unwrap_err();
            assert!(format!("{error:#}").contains("secret"), "{error:#}");
        }
        let (open, _dir) = serving(&body, None, Arc::new(AtomicBool::new(false)));
        let peer = PeerSource::new("peer".into(), open, Some("mine".into()), None);
        let error = peer.read(0, 10).unwrap_err();
        assert!(format!("{error:#}").contains("prove"), "{error:#}");
    }

    /// A node with no whole file to serve says so.
    #[test]
    fn a_node_without_a_file_refuses() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let split = transport::server(None, stream).unwrap();
            let mut reader = BufReader::new(split.reader);
            let mut writer = split.writer;
            if let Ok(Message::FetchHello { nonce }) = read_message(&mut reader) {
                serve(
                    &mut reader,
                    &mut *writer,
                    nonce,
                    None,
                    &Pace::default(),
                    &|| false,
                );
            }
        });
        let peer = PeerSource::new("peer".into(), addr, None, None);
        let error = peer.info().unwrap_err();
        assert!(format!("{error:#}").contains("no whole model"), "{error:#}");
    }

    /// Busy, a node lets ranges out at its pace; idle, at once.
    #[test]
    fn a_busy_node_paces_its_ranges() {
        let pace = Pace::default();
        let started = Instant::now();
        for _ in 0..4 {
            pace.admit(BUSY_BYTES_PER_SECOND, false);
        }
        assert!(started.elapsed() < Duration::from_millis(100));
        let started = Instant::now();
        // The first range goes at once, the next waits out the first.
        pace.admit(BUSY_BYTES_PER_SECOND / 10, true);
        pace.admit(BUSY_BYTES_PER_SECOND / 10, true);
        pace.admit(BUSY_BYTES_PER_SECOND / 10, true);
        let took = started.elapsed();
        assert!(took >= Duration::from_millis(190), "{took:?}");
        assert!(took < Duration::from_millis(1000), "{took:?}");
    }

    /// Only a plain single-file `.gguf` is served.
    #[test]
    fn only_a_single_model_file_is_served() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["m.gguf.partial", "m-00001-of-00002.gguf", "m.bin"] {
            let path = dir.path().join(name);
            std::fs::write(&path, b"GGUF").unwrap();
            assert!(ServedFile::new(&path).is_none(), "{name}");
        }
        let path = dir.path().join("m.gguf");
        std::fs::write(&path, b"GGUF").unwrap();
        let file = ServedFile::new(&path).unwrap();
        assert_eq!((file.size, file.commit.as_str()), (4, ""));
    }
}
