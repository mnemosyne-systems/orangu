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

//! Serving over TLS.
//!
//! Built in rather than left to a reverse proxy, because "one static binary,
//! nothing to install" is the property this project trades other things for —
//! and an answer of "put nginx in front of it" spends exactly that property in
//! the deployments the binary exists for: air-gapped, sovereign, one machine,
//! no package manager. Terminating in front stays perfectly valid and is what
//! most fleets will do; it just should not be the *only* way to reach the
//! server over a network safely.
//!
//! It costs a small amount of glue and no new cryptography: `rustls` and
//! `tokio-rustls` are already in the dependency tree for `reqwest`'s HTTPS, so
//! what is added here is a PEM reader and an acceptor around the listener.
//!
//! # Why a `Listener` rather than a second serving path
//!
//! [`axum::serve::Listener`] is a two-method trait — accept a connection,
//! report the local address — so a TLS listener drops into the existing
//! `axum::serve` call unchanged, keeping one serving path, one shutdown
//! `select!`, and the `ConnectInfo<SocketAddr>` extractor that the routes
//! already rely on. The alternative, a hand-rolled accept loop feeding
//! `hyper`, would have duplicated all of that in order to add one wrapper.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::{TlsAcceptor, server::TlsStream};

/// The certificate and key a TLS listener needs.
pub struct TlsPaths {
    pub cert: PathBuf,
    pub key: PathBuf,
}

/// Builds a `rustls` server config from a PEM certificate chain and key.
///
/// Errors name the file and what was wrong with it. A server that fails to
/// start because a key is unreadable has said something useful; one that
/// starts *without* TLS because the key was unreadable has quietly published
/// an inference engine in the clear, which is the failure worth being loud
/// about.
pub fn server_config(paths: &TlsPaths) -> Result<Arc<tokio_rustls::rustls::ServerConfig>> {
    // `rustls` needs a crypto provider chosen before any config is built.
    // Installing it is idempotent-by-ignoring: a second call returns `Err`
    // because one is already installed, which is exactly what we want.
    let _ = tokio_rustls::rustls::crypto::aws_lc_rs::default_provider().install_default();

    let certs = read_certs(&paths.cert)?;
    let key = read_key(&paths.key)?;
    let config = tokio_rustls::rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .with_context(|| {
            format!(
                "building a TLS configuration from {} and {}",
                paths.cert.display(),
                paths.key.display()
            )
        })?;
    Ok(Arc::new(config))
}

fn read_certs(
    path: &Path,
) -> Result<Vec<tokio_rustls::rustls::pki_types::CertificateDer<'static>>> {
    let pem = std::fs::read(path)
        .with_context(|| format!("reading the TLS certificate {}", path.display()))?;
    // `rustls_pki_types`' own PEM support, not `rustls-pemfile`: that crate is
    // the historical spelling and is now flagged unmaintained
    // (RUSTSEC-2025-0134), while this one is already in the tree as a
    // `rustls` dependency. One fewer crate and one fewer advisory.
    use tokio_rustls::rustls::pki_types::pem::PemObject;
    let certs: Vec<_> =
        tokio_rustls::rustls::pki_types::CertificateDer::pem_slice_iter(pem.as_slice())
            .collect::<Result<_, _>>()
            .with_context(|| format!("parsing the TLS certificate {}", path.display()))?;
    if certs.is_empty() {
        bail!(
            "{} contains no CERTIFICATE block — a PEM chain was expected",
            path.display()
        );
    }
    Ok(certs)
}

/// Reads the private key, accepting any of the three PEM spellings.
///
/// All three are in the wild and which one a tool emits is not something an
/// operator chooses: `openssl req -newkey` writes PKCS#8, older tooling writes
/// PKCS#1 (`BEGIN RSA PRIVATE KEY`), and EC keys come out as SEC1. Accepting
/// only one of them would reject a perfectly good key with an error about
/// nothing the reader did.
fn read_key(path: &Path) -> Result<tokio_rustls::rustls::pki_types::PrivateKeyDer<'static>> {
    let pem = std::fs::read(path)
        .with_context(|| format!("reading the TLS private key {}", path.display()))?;
    use tokio_rustls::rustls::pki_types::pem::PemObject;
    tokio_rustls::rustls::pki_types::PrivateKeyDer::from_pem_slice(pem.as_slice()).map_err(|_| {
        anyhow::anyhow!(
            "{} contains no PRIVATE KEY block (PKCS#8, PKCS#1 or SEC1 all work)",
            path.display()
        )
    })
}

/// A [`TcpListener`] that completes a TLS handshake before handing the
/// connection on.
pub struct TlsListener {
    tcp: TcpListener,
    acceptor: TlsAcceptor,
}

impl TlsListener {
    pub fn new(tcp: TcpListener, config: Arc<tokio_rustls::rustls::ServerConfig>) -> Self {
        Self {
            tcp,
            acceptor: TlsAcceptor::from(config),
        }
    }
}

impl axum::serve::Listener for TlsListener {
    type Io = TlsStream<TcpStream>;
    type Addr = std::net::SocketAddr;

    /// Accepts until a connection completes its handshake.
    ///
    /// The trait's contract is that `accept` cannot fail, so both failure
    /// modes are handled here rather than propagated. A refused handshake is
    /// **not** a server error and must not be treated as one: on any port
    /// reachable from a network it happens constantly — scanners, plain-HTTP
    /// requests to an HTTPS port, clients with no shared cipher — and a loop
    /// that gave up, or logged each one, would turn ordinary background noise
    /// into an outage or a flooded log.
    async fn accept(&mut self) -> (Self::Io, Self::Addr) {
        loop {
            let Ok((stream, addr)) = self.tcp.accept().await else {
                // Out of descriptors, or the listener is wedged. Yield rather
                // than spin; `axum`'s own TCP listener does the same.
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                continue;
            };
            match self.acceptor.accept(stream).await {
                Ok(tls) => return (tls, addr),
                Err(_) => continue,
            }
        }
    }

    fn local_addr(&self) -> io::Result<Self::Addr> {
        self.tcp.local_addr()
    }
}

impl TlsListener {
    /// This listener in the form `axum::serve` will hand a
    /// `ConnectInfo<SocketAddr>` out of.
    ///
    /// `axum` implements `Connected<IncomingStream<'_, L>>` for `SocketAddr`
    /// only for its own `TcpListener`, plus a blanket impl for anything
    /// wrapped in `TapIo`. Writing the impl here is not allowed — both
    /// `IncomingStream` and `SocketAddr` are foreign, and a local type
    /// appearing only as a type parameter does not satisfy the orphan rule —
    /// so the blanket impl is what earns it, via a tap that does nothing.
    ///
    /// Worth the indirection because the alternative was dropping
    /// `ConnectInfo` to make it compile, and the loopback-only routes
    /// (`/model-cache/drop`) decide what they allow from the peer address.
    /// "It compiles without it" would have quietly widened those.
    pub fn with_connect_info(
        self,
    ) -> axum::serve::TapIo<Self, impl FnMut(&mut TlsStream<TcpStream>) + Send + 'static> {
        use axum::serve::ListenerExt;
        self.tap_io(|_| {})
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A self-signed pair, so the loader is exercised on real PEM rather than
    /// on a fixture that only resembles one.
    fn write_pair(dir: &Path) -> TlsPaths {
        let cert = dir.join("cert.pem");
        let key = dir.join("key.pem");
        let out = std::process::Command::new("openssl")
            .args([
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-days",
                "1",
                "-subj",
                "/CN=localhost",
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
        TlsPaths { cert, key }
    }

    /// A directory of this test's own.
    ///
    /// Named per test, not just per process: the two tests here ran
    /// concurrently in one directory and deleted each other's files, which
    /// showed up as one of them failing only when the other was also run.
    fn tempdir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("orangu-tls-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_real_certificate_and_key_load() {
        let dir = tempdir("load");
        let paths = write_pair(&dir);
        server_config(&paths).expect("a freshly generated pair should load");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Every failure names the file and what was wrong with it. A server that
    /// refuses to start has said something useful; one that starts without TLS
    /// because a key was unreadable has published itself in the clear.
    #[test]
    fn a_broken_key_pair_fails_with_the_file_in_the_message() {
        let dir = tempdir("broken");
        let paths = write_pair(&dir);

        let missing = TlsPaths {
            cert: dir.join("nope.pem"),
            key: paths.key.clone(),
        };
        let err = server_config(&missing).unwrap_err().to_string();
        assert!(err.contains("nope.pem"), "{err}");

        let empty = dir.join("empty.pem");
        std::fs::write(&empty, b"not a pem file\n").unwrap();
        let err = server_config(&TlsPaths {
            cert: empty.clone(),
            key: paths.key.clone(),
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("empty.pem"), "{err}");
        assert!(err.contains("CERTIFICATE"), "{err}");

        let err = server_config(&TlsPaths {
            cert: paths.cert.clone(),
            key: empty,
        })
        .unwrap_err()
        .to_string();
        assert!(err.contains("PRIVATE KEY"), "{err}");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
