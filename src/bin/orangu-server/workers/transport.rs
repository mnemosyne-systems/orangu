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

//! The byte stream between a parent and a worker: plain TCP, or TLS over
//! it.
//!
//! TLS is `[workers].tls_cert`/`tls_key` on the listening side, and on the
//! dialing side trust in `[workers].tls_ca` — or in the node's own
//! `tls_cert`, so a tree whose nodes share one certificate needs nothing
//! else. The worker is checked against the address its parent dials, so the
//! certificate has to name it (a DNS name, or an IP address as a
//! `subjectAltName`).
//!
//! TLS keeps what travels between nodes — prompts, as hidden states, and
//! everything generated — private. Who may take part is still
//! `[workers].secret`'s job: TLS here authenticates the worker to its
//! parent, not the parent to its worker.

use anyhow::{Context, Result};
use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio_rustls::rustls;

/// One direction of a connection, readable or writable from its own
/// thread: a parent reads answers on one thread while any number of others
/// send requests, and a worker answers from as many threads as it runs
/// forwards on.
pub struct Split {
    pub reader: Box<dyn Read + Send>,
    pub writer: Box<dyn Write + Send>,
    /// The socket under both, to shut down and to set timeouts on.
    pub tcp: Arc<TcpStream>,
}

/// One handle on a socket, shared by the threads using it. Never a
/// [`TcpStream::try_clone`] for reading or writing: on Windows a timeout set on
/// one handle does not reach a read or write on its duplicate.
struct Socket(Arc<TcpStream>);

impl Read for Socket {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        (&*self.0).read(buf)
    }
}

impl Write for Socket {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        (&*self.0).write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        (&*self.0).flush()
    }
}

#[derive(Clone, Default)]
pub struct Tls {
    /// Serves the worker listener, when set.
    pub server: Option<Arc<rustls::ServerConfig>>,
    /// Dials workers, when set.
    pub client: Option<Arc<rustls::ClientConfig>>,
}

impl Tls {
    /// From `[workers].tls_cert`/`tls_key` and `tls_ca`: `None` when none is
    /// set.
    pub fn from_paths(pair: Option<(&Path, &Path)>, ca: Option<&Path>) -> Result<Option<Self>> {
        if pair.is_none() && ca.is_none() {
            return Ok(None);
        }
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let server = match pair {
            Some((cert, key)) => Some(crate::tls::server_config(&crate::tls::TlsPaths {
                cert: cert.to_path_buf(),
                key: key.to_path_buf(),
            })?),
            None => None,
        };
        let trusted = ca.or(pair.map(|(cert, _)| cert));
        let client = match trusted {
            Some(path) => {
                let mut roots = rustls::RootCertStore::empty();
                for cert in crate::tls::read_certs(path)? {
                    roots.add(cert).with_context(|| {
                        format!("trusting the certificate in {}", path.display())
                    })?;
                }
                Some(Arc::new(
                    rustls::ClientConfig::builder()
                        .with_root_certificates(roots)
                        .with_no_client_auth(),
                ))
            }
            None => None,
        };
        Ok(Some(Self { server, client }))
    }
}

/// The name a worker's certificate has to carry: the host of the address
/// its parent dials, without an IPv6 address's brackets.
fn server_name(addr: &str) -> Result<rustls::pki_types::ServerName<'static>> {
    let host = addr.rsplit_once(':').map_or(addr, |(host, _)| host);
    let host = host.trim_start_matches('[').trim_end_matches(']');
    rustls::pki_types::ServerName::try_from(host.to_string())
        .map_err(|e| anyhow::anyhow!("'{host}' cannot be checked against a certificate: {e}"))
}

/// The dialing side's connection to the worker at `addr`: plain, or with
/// the TLS handshake done, within the socket's read timeout.
pub fn client(tls: Option<&Tls>, tcp: TcpStream, addr: &str) -> Result<Split> {
    match tls.and_then(|t| t.client.clone()) {
        Some(config) => {
            let mut connection = rustls::ClientConnection::new(config, server_name(addr)?)?;
            let mut io = &tcp;
            while connection.is_handshaking() {
                connection.complete_io(&mut io)?;
            }
            Ok(split_tls(rustls::Connection::Client(connection), tcp))
        }
        None => Ok(split_plain(tcp)),
    }
}

/// The listening side's connection from a parent.
pub fn server(tls: Option<&Tls>, tcp: TcpStream) -> Result<Split> {
    match tls.and_then(|t| t.server.clone()) {
        Some(config) => {
            let mut connection = rustls::ServerConnection::new(config)?;
            let mut io = &tcp;
            while connection.is_handshaking() {
                connection.complete_io(&mut io)?;
            }
            Ok(split_tls(rustls::Connection::Server(connection), tcp))
        }
        None => Ok(split_plain(tcp)),
    }
}

fn split_plain(tcp: TcpStream) -> Split {
    let tcp = Arc::new(tcp);
    Split {
        reader: Box::new(Socket(tcp.clone())),
        writer: Box::new(Socket(tcp.clone())),
        tcp,
    }
}

/// TLS split in two around one shared session: the reader takes the lock
/// only to decrypt what it has already read off the socket, never while it
/// waits for more, so a writer is never held up by a read with nothing to
/// read.
fn split_tls(connection: rustls::Connection, tcp: TcpStream) -> Split {
    let tcp = Arc::new(tcp);
    let session = Arc::new(Mutex::new(connection));
    let out = Arc::new(Mutex::new(Socket(tcp.clone())));
    Split {
        reader: Box::new(TlsReader {
            session: session.clone(),
            socket: Socket(tcp.clone()),
            out: out.clone(),
        }),
        writer: Box::new(TlsWriter { session, out }),
        tcp,
    }
}

/// Sends whatever the session has to send: application data after a
/// write, and after a read any alert or key update the protocol answers
/// with.
fn flush_tls(session: &mut rustls::Connection, out: &Mutex<Socket>) -> io::Result<()> {
    let mut out = out.lock().unwrap();
    while session.wants_write() {
        session.write_tls(&mut *out)?;
    }
    out.flush()
}

struct TlsReader {
    session: Arc<Mutex<rustls::Connection>>,
    socket: Socket,
    out: Arc<Mutex<Socket>>,
}

impl Read for TlsReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let mut raw = vec![0u8; 16 * 1024];
        loop {
            {
                let mut session = self.session.lock().unwrap();
                match session.reader().read(buf) {
                    Ok(n) => return Ok(n),
                    Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
                    Err(e) => return Err(e),
                }
            }
            let n = self.socket.read(&mut raw)?;
            if n == 0 {
                return Ok(0);
            }
            let mut session = self.session.lock().unwrap();
            let mut pending = &raw[..n];
            while !pending.is_empty() {
                session.read_tls(&mut pending)?;
                session
                    .process_new_packets()
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            }
            flush_tls(&mut session, &self.out)?;
        }
    }
}

struct TlsWriter {
    session: Arc<Mutex<rustls::Connection>>,
    out: Arc<Mutex<Socket>>,
}

/// Plaintext handed to the session at a time: one TLS record's worth. The
/// session buffers only so much before it has been sent (64 KiB by
/// default), and a frame larger than that — a prompt chunk's activations,
/// a layer's rows — was refused whole: "failed to write whole buffer".
const TLS_PIECE: usize = 16 << 10;

impl Write for TlsWriter {
    /// In pieces, each sent before the next is taken, the session's lock let
    /// go in between so the reader can go on with what arrives meanwhile.
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        for piece in buf.chunks(TLS_PIECE) {
            let mut session = self.session.lock().unwrap();
            session.writer().write_all(piece)?;
            flush_tls(&mut session, &self.out)?;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        let mut session = self.session.lock().unwrap();
        flush_tls(&mut session, &self.out)
    }
}

#[cfg(test)]
pub(crate) mod fixture {
    use std::path::{Path, PathBuf};

    /// A self-signed certificate for `127.0.0.1` and `localhost`, and its
    /// key, written under `dir` with `openssl` as `tls.rs`'s own tests do.
    pub fn certificate(dir: &Path, name: &str) -> (PathBuf, PathBuf) {
        let cert = dir.join(format!("{name}.pem"));
        let key = dir.join(format!("{name}.key"));
        let out = std::process::Command::new("openssl")
            .args([
                "req",
                "-x509",
                "-newkey",
                "ec",
                "-pkeyopt",
                "ec_paramgen_curve:prime256v1",
                "-nodes",
                "-days",
                "1",
                "-subj",
                "/CN=localhost",
                "-addext",
                "subjectAltName=IP:127.0.0.1,DNS:localhost",
                // A server's own certificate, not an authority: rustls
                // refuses a CA certificate presented as a server's.
                "-addext",
                "basicConstraints=critical,CA:FALSE",
                "-keyout",
                key.to_str().unwrap(),
                "-out",
                cert.to_str().unwrap(),
            ])
            .output()
            .expect("openssl is needed for this test");
        assert!(
            out.status.success(),
            "openssl: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        (cert, key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::time::Duration;

    fn connected() -> (TcpStream, TcpStream) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let ours = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (theirs, _) = listener.accept().unwrap();
        (ours, theirs)
    }

    /// A worker shortens its read timeout once the handshake is done. Should
    /// that not reach the reader, this read lasts until the other side closes,
    /// and ends without an error.
    #[test]
    fn a_read_timeout_set_after_the_split_is_shorter_at_the_reader() {
        let (ours, theirs) = connected();
        ours.set_read_timeout(Some(Duration::from_secs(60)))
            .unwrap();
        let mut split = split_plain(ours);
        split
            .tcp
            .set_read_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_secs(5));
            drop(theirs);
        });
        let error = split.reader.read(&mut [0; 1]).unwrap_err();
        assert!(
            matches!(
                error.kind(),
                io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
            ),
            "{error}"
        );
    }

    /// A parent lifts its read timeout once the handshake is done, and a
    /// reply may then take as long as it takes.
    #[test]
    fn a_read_timeout_lifted_after_the_split_is_gone_at_the_reader() {
        let (ours, mut theirs) = connected();
        ours.set_read_timeout(Some(Duration::from_millis(50)))
            .unwrap();
        let mut split = split_plain(ours);
        split.tcp.set_read_timeout(None).unwrap();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(500));
            theirs.write_all(&[7]).unwrap();
        });
        let mut byte = [0; 1];
        assert_eq!(split.reader.read(&mut byte).unwrap(), 1);
        assert_eq!(byte, [7]);
    }

    #[test]
    fn the_server_name_is_the_dialed_host() {
        assert!(server_name("127.0.0.1:8400").is_ok());
        assert!(server_name("[::1]:8400").is_ok());
        assert!(server_name("node1.lan:8400").is_ok());
    }

    #[test]
    fn nothing_set_means_no_tls() {
        assert!(Tls::from_paths(None, None).unwrap().is_none());
    }

    /// Trust defaults to the node's own certificate; `tls_ca` alone dials
    /// with TLS and listens without.
    #[test]
    fn a_shared_certificate_is_trusted_by_default() {
        let dir = tempfile::tempdir().unwrap();
        let (cert, key) = fixture::certificate(dir.path(), "node");
        let tls = Tls::from_paths(Some((&cert, &key)), None).unwrap().unwrap();
        assert!(tls.server.is_some() && tls.client.is_some());
        let tls = Tls::from_paths(None, Some(&cert)).unwrap().unwrap();
        assert!(tls.server.is_none() && tls.client.is_some());
    }
}
