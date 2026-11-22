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

//! `orangu-gguf` on a `[workers]` channel.
//!
//! A channel holds two kinds of node. `orangu-server` nodes serve models;
//! `orangu-gguf --worker` nodes ([`NodeKind::Gguf`]) take a share of
//! building a model file. Both answer the same probe, so either kind finds
//! the other, and both follow the same claiming rules: a node about to
//! claim asks each free node for a lease first, a free node promises itself
//! for [`LEASE_FOR`] to the lowest id that asked, and only then is it
//! claimed, proving `[workers].secret` both ways. A gguf node works for the
//! top that claimed it until that top closes the connection.
//!
//! The settings come from `orangu-gguf.conf`, with the server's
//! `[workers]` spelling. Which machines do the work is not a property of
//! the model, so none of it is in the manifest.

use anyhow::{Context, Result, anyhow, bail};
use orangu::workers::auth;
use orangu::workers::channel::{self, Found, LEASE_FOR, LeaseAnswer};
use orangu::workers::protocol::{
    ErrorCode, Message, NodeKind, Pace, ProbeState, WorkerError, read_message, write_message,
};
use orangu::workers::transport::{self, Split, Tls};
#[cfg(test)]
use std::collections::HashMap;
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
#[cfg(test)]
use std::time::Duration;
use std::time::Instant;

/// The port a gguf node listens on when `[workers].port` names none: next
/// to a server's 8400, so both run on one machine.
pub const DEFAULT_PORT: u16 = 8401;

/// The port the channel's server nodes listen on when
/// `[workers].server_port` names none.
pub const DEFAULT_SERVER_PORT: u16 = 8400;

/// The jobs a gguf node takes.
pub const CAPABILITIES: [&str; 4] = ["imatrix", "perplexity", "quantize", "train"];

/// The busiest a node may be and still be claimed.
pub const MAX_LOAD: f32 = 0.8;

/// `orangu-gguf.conf`'s `[workers]` section.
#[derive(Clone)]
pub struct Settings {
    /// `[workers].name`: the channel.
    pub name: String,
    pub secret: Option<String>,
    /// The address `--worker` listens on.
    pub host: String,
    pub port: u16,
    /// Where the channel's server nodes listen.
    pub server_port: u16,
    /// The networks a scan looks at; empty scans this machine's own.
    pub networks: Vec<(Ipv4Addr, u8)>,
    /// `host:port` entries asked as well as, or instead of, a scan
    /// (`ORANGU_WORKERS_PEERS`).
    pub peers: Vec<String>,
    /// Whether to scan at all: off when only peers are to be asked.
    pub scan: bool,
    pub tls: Option<Tls>,
}

impl Settings {
    /// `path`, or the first of `./orangu-gguf.conf` and
    /// `~/.orangu/orangu-gguf.conf` that exists. `None` when there is no
    /// file, or it names no channel.
    pub fn load(path: Option<&Path>) -> Result<Option<Self>> {
        let path = match path {
            Some(p) => Some(p.to_path_buf()),
            None => default_paths().into_iter().find(|p| p.is_file()),
        };
        let Some(path) = path else {
            return Ok(None);
        };
        let text = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("in {}", path.display()))
    }

    /// [`Self::load`] of the `--config` file this run was given.
    pub fn load_configured() -> Result<Option<Self>> {
        Self::load(CONFIG.get().and_then(|p| p.as_deref()))
    }

    pub fn parse(text: &str) -> Result<Option<Self>> {
        let sections = orangu::config::parse_ini_sections(text)?;
        let Some(section) = sections.get("workers") else {
            return Ok(None);
        };
        let get = |key: &str| {
            section
                .get(key)
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        };
        let Some(name) = get("name") else {
            return Ok(None);
        };
        let port = |key: &str, default: u16| -> Result<u16> {
            get(key).map_or(Ok(default), |v| {
                v.parse()
                    .map_err(|_| anyhow!("[workers].{key} = {v:?} is not a port"))
            })
        };
        let networks = match get("network") {
            Some(v) => channel::parse_networks(&v)?,
            None => Vec::new(),
        };
        let mut peers = match std::env::var("ORANGU_WORKERS_PEERS") {
            Ok(v) => channel::parse_peers(&v, port("server_port", DEFAULT_SERVER_PORT)?)?,
            Err(_) => Vec::new(),
        };
        if let Some(v) = get("peers") {
            peers.extend(channel::parse_peers(
                &v,
                port("server_port", DEFAULT_SERVER_PORT)?,
            )?);
        }
        let scan = match get("scan").as_deref() {
            None | Some("on") | Some("auto") => peers.is_empty() || get("network").is_some(),
            Some("off") => false,
            Some(other) => bail!("[workers].scan = {other:?}: expected on or off"),
        };
        let pair = match (get("tls_cert"), get("tls_key")) {
            (Some(c), Some(k)) => Some((PathBuf::from(c), PathBuf::from(k))),
            (None, None) => None,
            _ => bail!("[workers].tls_cert and tls_key go together"),
        };
        let ca = get("tls_ca").map(PathBuf::from);
        let tls = Tls::from_paths(
            pair.as_ref().map(|(c, k)| (c.as_path(), k.as_path())),
            ca.as_deref(),
        )?;
        Ok(Some(Settings {
            name,
            secret: get("secret"),
            host: get("host").unwrap_or_else(|| "0.0.0.0".to_string()),
            port: port("port", DEFAULT_PORT)?,
            server_port: port("server_port", DEFAULT_SERVER_PORT)?,
            networks,
            peers,
            scan,
            tls,
        }))
    }

    fn hash(&self) -> [u8; 32] {
        channel::channel_hash(&self.name, self.secret.as_deref())
    }
}

/// The `--config` file of this run, when one was given.
static CONFIG: std::sync::OnceLock<Option<PathBuf>> = std::sync::OnceLock::new();

/// Remembers `--config` for [`Settings::load_configured`].
pub fn configure(path: Option<PathBuf>) {
    let _ = CONFIG.set(path);
}

fn default_paths() -> Vec<PathBuf> {
    let mut out = vec![PathBuf::from("orangu-gguf.conf")];
    if let Some(home) = std::env::var_os("HOME") {
        out.push(Path::new(&home).join(".orangu").join("orangu-gguf.conf"));
    }
    out
}

/// A node id for this run of the process, as a server makes its own.
pub fn new_id() -> String {
    format!("{:016x}", rand::random::<u64>())
}

/// Every node of the channel: servers and gguf nodes, one entry each,
/// ordered by name.
pub fn discover(settings: &Settings, own: &str) -> Vec<Found> {
    let hash = settings.hash();
    let tls = settings.tls.as_ref();
    let mut addrs: Vec<SocketAddr> = channel::resolve_peers(&settings.peers);
    if settings.scan {
        let hosts = channel::scan_hosts(&settings.networks);
        for port in [settings.server_port, settings.port] {
            addrs.extend(
                hosts
                    .iter()
                    .map(|&ip| SocketAddr::V4(SocketAddrV4::new(ip, port))),
            );
        }
    }
    addrs.sort();
    addrs.dedup();
    let mut found = channel::probe_all(&addrs, &hash, tls);
    found.retain(|f| f.node != own);
    channel::one_per_node(found, &[])
}

/// `--channel`: the nodes the channel has, of both kinds.
pub fn list(settings: &Settings) -> Result<()> {
    let found = discover(settings, "");
    println!("channel '{}': {} nodes", settings.name, found.len());
    if found.is_empty() {
        return Ok(());
    }
    println!();
    println!(
        "{:<7} {:<28} {:<22} {:<11} {:>5}  does",
        "kind", "name", "address", "state", "load"
    );
    for f in &found {
        let does = match f.kind {
            NodeKind::Server if !f.label.is_empty() => format!("serves {} {}", f.label, f.quant),
            NodeKind::Server => "serves".to_string(),
            NodeKind::Gguf => f.capabilities.join(", "),
        };
        println!(
            "{:<7} {:<28} {:<22} {:<11} {:>4.0}%  {does}",
            f.kind.name(),
            f.name,
            f.addr,
            state_name(f.state),
            f.load * 100.0
        );
    }
    Ok(())
}

fn state_name(state: ProbeState) -> &'static str {
    match state {
        ProbeState::Idle => "free",
        ProbeState::Leading => "leading",
        ProbeState::Serving => "working",
        ProbeState::WorkerOnly => "worker only",
    }
}

/// A gguf node this top claimed: the connection jobs go over.
pub struct Claimed {
    pub found: Found,
    pub link: Split,
}

/// Whether `f` is a gguf node free to take `job`.
pub fn claimable(f: &Found, job: &str) -> bool {
    f.kind == NodeKind::Gguf
        && f.state == ProbeState::Idle
        && f.load <= MAX_LOAD
        && f.capabilities.iter().any(|c| c == job)
}

/// Claims every node of `found` free to take `job`: a lease first, then
/// the handshake. A node promised to a lower id, or that fails the
/// handshake, is left alone.
pub fn claim(settings: &Settings, own: &str, found: &[Found], job: &str) -> Vec<Claimed> {
    let hash = settings.hash();
    let tls = settings.tls.as_ref();
    let mut out = Vec::new();
    for f in found.iter().filter(|f| claimable(f, job)) {
        let Ok(addr) = f.addr.parse::<SocketAddr>() else {
            continue;
        };
        match channel::lease(addr, &hash, own, tls) {
            Some(LeaseAnswer::Granted) => {}
            _ => continue,
        }
        match handshake(addr, settings, own) {
            Ok(link) => out.push(Claimed {
                found: f.clone(),
                link,
            }),
            Err(e) => log::info!("orangu-gguf: workers: {} not claimed: {e:#}", f.addr),
        }
    }
    out
}

/// The top's side of [`Message::Claim`].
fn handshake(addr: SocketAddr, settings: &Settings, own: &str) -> Result<Split> {
    let tcp = TcpStream::connect_timeout(&addr, channel::CONNECT_TIMEOUT)?;
    tcp.set_read_timeout(Some(channel::ANSWER_TIMEOUT))?;
    tcp.set_write_timeout(Some(channel::ANSWER_TIMEOUT))?;
    let _ = tcp.set_nodelay(true);
    let mut split = transport::client(settings.tls.as_ref(), tcp, &addr.to_string())?;
    let nonce = auth::nonce();
    write_message(
        &mut *split.writer,
        0,
        &Message::Claim {
            nonce,
            node: own.to_string(),
        },
    )?;
    let secret = settings.secret.as_deref();
    let theirs = match read_message(&mut *split.reader)? {
        Message::ClaimAck {
            nonce: theirs,
            proof,
            ..
        } => {
            if !auth::parent_accepts(secret, &nonce, &theirs, &proof) {
                bail!("it does not know the secret");
            }
            theirs
        }
        Message::Error(e) => bail!("{}", e.message),
        other => bail!("it answered {other:?}"),
    };
    write_message(
        &mut *split.writer,
        0,
        &Message::Auth {
            proof: auth::parent_proof(secret, &nonce, &theirs),
        },
    )?;
    match read_message(&mut *split.reader)? {
        Message::Ready => {
            split.tcp.set_read_timeout(None)?;
            Ok(split)
        }
        Message::Error(e) => bail!("{}", e.message),
        other => bail!("it answered {other:?}"),
    }
}

/// `--worker`: this process as a gguf node of the channel.
pub struct Worker {
    pub id: String,
    settings: Settings,
    /// Where it answers, `host:port`, as probes report it.
    name: String,
    lease: Mutex<Option<(String, Instant)>>,
    /// The top it works for.
    top: Mutex<Option<String>>,
}

impl Worker {
    /// Listens on `[workers].host:port` and answers on threads of its own.
    pub fn start(settings: Settings) -> Result<(Arc<Self>, SocketAddr)> {
        let listener =
            TcpListener::bind((settings.host.as_str(), settings.port)).with_context(|| {
                format!(
                    "listening on {}:{} for the channel",
                    settings.host, settings.port
                )
            })?;
        let local = listener.local_addr()?;
        let worker = Arc::new(Worker {
            id: new_id(),
            name: format!("{}:{}", shown_host(&settings.host), local.port()),
            settings,
            lease: Mutex::new(None),
            top: Mutex::new(None),
        });
        let serving = Arc::clone(&worker);
        std::thread::spawn(move || {
            for tcp in listener.incoming().flatten() {
                let w = Arc::clone(&serving);
                std::thread::spawn(move || w.answer(tcp));
            }
        });
        Ok((worker, local))
    }

    /// Runs until the process is stopped.
    pub fn serve(settings: Settings) -> Result<()> {
        let (worker, local) = Self::start(settings)?;
        println!(
            "orangu-gguf: worker {} on channel '{}', listening on {local}",
            worker.id, worker.settings.name
        );
        loop {
            std::thread::park();
        }
    }

    #[cfg(test)]
    pub fn working_for(&self) -> Option<String> {
        self.top.lock().unwrap().clone()
    }

    fn state(&self) -> ProbeState {
        if self.top.lock().unwrap().is_some() {
            ProbeState::Serving
        } else {
            ProbeState::Idle
        }
    }

    fn answer(&self, tcp: TcpStream) {
        let _ = tcp.set_read_timeout(Some(channel::ANSWER_TIMEOUT));
        let _ = tcp.set_write_timeout(Some(channel::ANSWER_TIMEOUT));
        let _ = tcp.set_nodelay(true);
        let Ok(mut split) = transport::server(self.settings.tls.as_ref(), tcp) else {
            return;
        };
        let hash = self.settings.hash();
        let reply = |split: &mut Split, m: &Message| write_message(&mut *split.writer, 0, m);
        match read_message(&mut *split.reader) {
            Ok(Message::Probe { channel }) if channel == hash => {
                let _ = reply(&mut split, &self.probe_ack());
            }
            Ok(Message::Lease { channel, node }) if channel == hash => {
                let _ = reply(&mut split, &self.answer_lease(&node));
            }
            Ok(Message::Claim { nonce, node }) => self.claimed(split, nonce, node),
            _ => {}
        }
    }

    fn probe_ack(&self) -> Message {
        let state = self.state();
        Message::ProbeAck {
            node: self.id.clone(),
            name: self.name.clone(),
            state,
            load: if state == ProbeState::Idle { 0.0 } else { 1.0 },
            whole: false,
            api_port: 0,
            api_tls: false,
            model: [0; 32],
            content: [0; 32],
            workers: 0,
            label: String::new(),
            quant: String::new(),
            features: 0,
            pace: Pace::default(),
            demand: Vec::new(),
            kind: NodeKind::Gguf,
            capabilities: CAPABILITIES.iter().map(|c| c.to_string()).collect(),
        }
    }

    /// The server's rule: a node working for a top is committed to it; a
    /// free one promises itself to the lowest id asking within
    /// [`LEASE_FOR`].
    fn answer_lease(&self, candidate: &str) -> Message {
        if let Some(top) = self.top.lock().unwrap().as_ref() {
            return Message::LeaseAck {
                granted: false,
                holder: top.clone(),
                committed: true,
            };
        }
        let mut lease = self.lease.lock().unwrap();
        let now = Instant::now();
        match channel::promised_against(lease.as_ref(), candidate, now) {
            Some(holder) => Message::LeaseAck {
                granted: false,
                holder,
                committed: false,
            },
            None => {
                *lease = Some((candidate.to_string(), now + LEASE_FOR));
                Message::LeaseAck {
                    granted: true,
                    holder: candidate.to_string(),
                    committed: false,
                }
            }
        }
    }

    /// The node's side of [`Message::Claim`], then the jobs until the top
    /// closes the connection.
    fn claimed(&self, mut split: Split, theirs: [u8; 32], top: String) {
        let refuse = |split: &mut Split, code, text: String| {
            let _ = write_message(
                &mut *split.writer,
                0,
                &Message::Error(WorkerError::new(code, text)),
            );
        };
        if let Some(holder) = self.busy_for(&top) {
            return refuse(
                &mut split,
                ErrorCode::Busy,
                format!("working for or promised to {holder}"),
            );
        }
        let secret = self.settings.secret.as_deref();
        let nonce = auth::nonce();
        let ack = Message::ClaimAck {
            nonce,
            proof: auth::worker_proof(secret, &theirs, &nonce),
            node: self.id.clone(),
        };
        if write_message(&mut *split.writer, 0, &ack).is_err() {
            return;
        }
        match read_message(&mut *split.reader) {
            Ok(Message::Auth { proof })
                if auth::worker_accepts(secret, &theirs, &nonce, &proof) => {}
            Ok(_) => {
                return refuse(
                    &mut split,
                    ErrorCode::Unauthorized,
                    "the secret does not match".to_string(),
                );
            }
            Err(_) => return,
        }
        {
            let mut held = self.top.lock().unwrap();
            if let Some(other) = held.as_ref().filter(|t| **t != top) {
                let other = other.clone();
                drop(held);
                return refuse(&mut split, ErrorCode::Busy, format!("working for {other}"));
            }
            *held = Some(top.clone());
        }
        *self.lease.lock().unwrap() = None;
        log::info!("orangu-gguf: worker: claimed by {top}");
        if write_message(&mut *split.writer, 0, &Message::Ready).is_ok() {
            let _ = split.tcp.set_read_timeout(None);
            self.work(&mut split);
        }
        *self.top.lock().unwrap() = None;
        log::info!("orangu-gguf: worker: released by {top}");
    }

    /// Another top this node is committed or promised to, against `top`.
    fn busy_for(&self, top: &str) -> Option<String> {
        if let Some(held) = self.top.lock().unwrap().as_ref()
            && held != top
        {
            return Some(held.clone());
        }
        let lease = self.lease.lock().unwrap();
        match lease.as_ref() {
            Some((holder, until)) if Instant::now() < *until && holder != top => {
                Some(holder.clone())
            }
            _ => None,
        }
    }

    /// The jobs of a claimed connection, answered in order until it closes.
    fn work(&self, split: &mut Split) {
        let mut session = crate::jobs::Session::default();
        loop {
            let message = match orangu::workers::protocol::read_frame(&mut *split.reader)
                .map(|(_, m, _)| m)
            {
                Ok(m) => m,
                Err(_) => return,
            };
            let answer = match message {
                Message::Ping { nonce } => Message::Pong { nonce },
                Message::Job { name, data } => match crate::jobs::serve(&mut session, &name, &data)
                {
                    Ok(data) => Message::JobDone { data },
                    Err(e) => {
                        Message::Error(WorkerError::new(ErrorCode::Internal, format!("{e:#}")))
                    }
                },
                other => Message::Error(WorkerError::new(
                    ErrorCode::Unsupported,
                    format!("a gguf node does not take {}", message_name(&other)),
                )),
            };
            if write_message(&mut *split.writer, 0, &answer).is_err() {
                return;
            }
        }
    }
}

fn message_name(m: &Message) -> String {
    let text = format!("{m:?}");
    text.split([' ', '(', '{'])
        .next()
        .unwrap_or("that message")
        .to_string()
}

/// The host a node shows: the listen address, or this machine's first
/// network address when it listens on all of them.
fn shown_host(host: &str) -> String {
    if host != "0.0.0.0" && host != "::" {
        return host.to_string();
    }
    if_addrs::get_if_addrs()
        .unwrap_or_default()
        .into_iter()
        .find_map(|i| match i.addr {
            if_addrs::IfAddr::V4(v4) if !v4.ip.is_loopback() => Some(v4.ip.to_string()),
            _ => None,
        })
        .unwrap_or_else(|| "127.0.0.1".to_string())
}

/// Pings every claimed node: each one's round trip, or why it failed.
#[cfg(test)]
pub fn ping(claimed: &mut [Claimed]) -> HashMap<String, Result<Duration>> {
    let mut out = HashMap::new();
    for c in claimed.iter_mut() {
        let started = Instant::now();
        let nonce = rand::random::<u64>();
        let result = (|| -> Result<Duration> {
            write_message(&mut *c.link.writer, 0, &Message::Ping { nonce })?;
            match read_message(&mut *c.link.reader)? {
                Message::Pong { nonce: n } if n == nonce => Ok(started.elapsed()),
                other => bail!("it answered {other:?}"),
            }
        })();
        out.insert(c.found.node.clone(), result);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(name: &str, secret: Option<&str>) -> Settings {
        Settings {
            name: name.to_string(),
            secret: secret.map(str::to_string),
            host: "127.0.0.1".to_string(),
            port: 0,
            server_port: DEFAULT_SERVER_PORT,
            networks: Vec::new(),
            peers: Vec::new(),
            scan: false,
            tls: None,
        }
    }

    fn found_at(top: &Settings, addr: SocketAddr) -> Found {
        channel::probe(addr, &top.hash(), None).expect("the worker answers its channel")
    }

    #[test]
    fn the_config_reads_the_servers_spelling() {
        let s = Settings::parse(
            "[workers]\nname = lab\nsecret = s3\nport = 9001\nserver_port = 9000\n\
             network = 10.0.0.0/24\n",
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            (s.name.as_str(), s.secret.as_deref(), s.port, s.server_port),
            ("lab", Some("s3"), 9001, 9000)
        );
        assert_eq!(s.networks, vec![(Ipv4Addr::new(10, 0, 0, 0), 24)]);
        assert!(s.scan);
        assert!(Settings::parse("[workers]\nport = 1\n").unwrap().is_none());
        assert!(Settings::parse("[other]\nname = x\n").unwrap().is_none());
    }

    /// A gguf worker answers the channel's probe as a gguf node that can
    /// quantize, is claimed once its lease and the secret check out, and
    /// is free again when its top lets go.
    #[test]
    fn a_worker_is_found_claimed_and_released() {
        let (worker, addr) = Worker::start(settings("lab", Some("s"))).unwrap();
        let top = settings("lab", Some("s"));
        let found = found_at(&top, addr);
        assert_eq!(found.kind, NodeKind::Gguf);
        assert_eq!(found.state, ProbeState::Idle);
        assert!(claimable(&found, "imatrix"));

        let mut claimed = claim(&top, "a-top", std::slice::from_ref(&found), "imatrix");
        assert_eq!(claimed.len(), 1);
        assert_eq!(worker.working_for().as_deref(), Some("a-top"));
        let rtt = ping(&mut claimed);
        assert!(rtt[&found.node].is_ok());
        // Busy for anyone else.
        assert_eq!(found_at(&top, addr).state, ProbeState::Serving);
        assert!(claim(&top, "b-top", std::slice::from_ref(&found), "imatrix").is_empty());

        drop(claimed);
        let deadline = Instant::now() + Duration::from_secs(5);
        while worker.working_for().is_some() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(worker.working_for(), None);
        assert_eq!(found_at(&top, addr).state, ProbeState::Idle);
    }

    /// The wrong secret finds nothing, and a claim with it fails.
    #[test]
    fn the_secret_decides() {
        let (worker, addr) = Worker::start(settings("lab", Some("right"))).unwrap();
        let wrong = settings("lab", Some("wrong"));
        assert!(channel::probe(addr, &wrong.hash(), None).is_none());
        // Asked directly, its proof does not check out against the wrong
        // secret, and it is not claimed.
        let err = handshake(addr, &wrong, "x").err().unwrap().to_string();
        assert!(err.contains("secret"), "{err}");
        assert_eq!(worker.working_for(), None);
    }

    /// Two tops asking at once: the worker stays promised to the lower id.
    #[test]
    fn the_lower_id_keeps_the_lease() {
        let (worker, addr) = Worker::start(settings("lab", None)).unwrap();
        let s = settings("lab", None);
        let hash = s.hash();
        assert_eq!(
            channel::lease(addr, &hash, "b", None),
            Some(LeaseAnswer::Granted)
        );
        assert_eq!(
            channel::lease(addr, &hash, "a", None),
            Some(LeaseAnswer::Granted)
        );
        assert_eq!(
            channel::lease(addr, &hash, "b", None),
            Some(LeaseAnswer::Promised("a".to_string()))
        );
        let found = found_at(&s, addr);
        assert!(claim(&s, "b", std::slice::from_ref(&found), "imatrix").is_empty());
        // Held until the check: dropping the claim releases the worker.
        let claimed = claim(&s, "a", &[found], "imatrix");
        assert_eq!(claimed.len(), 1);
        assert_eq!(worker.working_for().as_deref(), Some("a"));
    }
}
