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

//! The egress proxy of a session with `environment_variable` credentials.
//! The sandbox's variables hold placeholders, and its `HTTPS_PROXY` and
//! `HTTP_PROXY` name this proxy, on loopback. A request to a host one of
//! the secrets is allowed for is read here — over TLS, under a certificate
//! for that host from this server's own authority, which the sandbox's
//! certificate bundle trusts — has its placeholders replaced with the
//! secrets in the headers and body the credential allows, and goes on to
//! the host over TLS of its own. A request to any other host passes through
//! untouched, so a placeholder sent there stays a placeholder.
//!
//! One request per connection: each is sent on with `Connection: close`,
//! and the host's answer is relayed until it closes.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, Mutex};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio_rustls::rustls;

/// The largest request head and body the proxy reads.
const MAX_HEAD: usize = 64 * 1024;
const MAX_BODY: usize = 64 * 1024 * 1024;

/// Where the sandbox finds the certificate bundle that trusts the proxy.
pub(super) const BUNDLE_IN_SANDBOX: &str = "/etc/orangu-egress/ca-bundle.pem";

/// One secret: the variable it is, the placeholder the sandbox sees, the
/// hosts it may go to (`None` for any), and where in a request it goes.
pub(super) struct Secret {
    pub name: String,
    value: String,
    pub placeholder: String,
    hosts: Option<Vec<String>>,
    header: bool,
    body: bool,
}

impl Secret {
    pub(super) fn new(
        name: String,
        value: String,
        hosts: Option<Vec<String>>,
        header: bool,
        body: bool,
    ) -> Self {
        Self {
            name,
            value,
            placeholder: format!("orangu-vault-{}", uuid::Uuid::new_v4().simple()),
            hosts,
            header,
            body,
        }
    }

    /// Whether this secret may go to `host`: `*.example.com` covers the
    /// subdomains of `example.com`.
    fn allowed_for(&self, host: &str) -> bool {
        let host = host.to_ascii_lowercase();
        self.hosts.as_ref().is_none_or(|hosts| {
            hosts
                .iter()
                .any(|pattern| match pattern.strip_prefix("*.") {
                    Some(domain) => host.ends_with(&format!(".{domain}")),
                    None => host == *pattern,
                })
        })
    }
}

/// This server's certificate authority, made when first needed, and the
/// certificates it has issued, by host.
struct Authority {
    issuer: rcgen::Issuer<'static, rcgen::KeyPair>,
    pem: String,
    issued: Mutex<HashMap<String, Arc<rustls::ServerConfig>>>,
}

static AUTHORITY: LazyLock<Result<Authority, String>> = LazyLock::new(|| {
    let mut params =
        rcgen::CertificateParams::new(Vec::<String>::new()).map_err(|e| e.to_string())?;
    params.is_ca = rcgen::IsCa::Ca(rcgen::BasicConstraints::Constrained(0));
    params
        .distinguished_name
        .push(rcgen::DnType::CommonName, "orangu-server egress authority");
    params.key_usages = vec![
        rcgen::KeyUsagePurpose::KeyCertSign,
        rcgen::KeyUsagePurpose::CrlSign,
        rcgen::KeyUsagePurpose::DigitalSignature,
    ];
    let key = rcgen::KeyPair::generate().map_err(|e| e.to_string())?;
    let pem = params.self_signed(&key).map_err(|e| e.to_string())?.pem();
    Ok(Authority {
        issuer: rcgen::Issuer::new(params, key),
        pem,
        issued: Mutex::new(HashMap::new()),
    })
});

fn authority() -> Result<&'static Authority, String> {
    AUTHORITY
        .as_ref()
        .map_err(|e| format!("the egress authority: {e}"))
}

impl Authority {
    /// The TLS server settings that present a certificate for `host`.
    fn for_host(&self, host: &str) -> Result<Arc<rustls::ServerConfig>, String> {
        let mut issued = self.issued.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(config) = issued.get(host) {
            return Ok(config.clone());
        }
        let params =
            rcgen::CertificateParams::new(vec![host.to_string()]).map_err(|e| e.to_string())?;
        let key = rcgen::KeyPair::generate().map_err(|e| e.to_string())?;
        let cert = params
            .signed_by(&key, &self.issuer)
            .map_err(|e| e.to_string())?;
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let key_der = rustls::pki_types::PrivateKeyDer::try_from(key.serialize_der())
            .map_err(|e| e.to_string())?;
        let mut config = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![cert.der().clone()], key_der)
            .map_err(|e| e.to_string())?;
        config.alpn_protocols = vec![b"http/1.1".to_vec()];
        let config = Arc::new(config);
        issued.insert(host.to_string(), config.clone());
        Ok(config)
    }
}

/// The certificate bundle a sandbox trusts: the host's own, with this
/// server's authority added — written once per run of the server, under
/// `api_data`.
pub(super) fn bundle(api_data: &Path) -> Result<PathBuf, String> {
    static WRITTEN: Mutex<Option<PathBuf>> = Mutex::new(None);
    let mut written = WRITTEN.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(path) = written.as_ref() {
        return Ok(path.clone());
    }
    let mut text = String::new();
    for system in [
        "/etc/pki/tls/certs/ca-bundle.crt",
        "/etc/ssl/certs/ca-certificates.crt",
        "/etc/ssl/cert.pem",
    ] {
        if let Ok(found) = std::fs::read_to_string(system) {
            text = found;
            break;
        }
    }
    if !text.ends_with('\n') && !text.is_empty() {
        text.push('\n');
    }
    text.push_str(&authority()?.pem);
    let path = api_data.join("egress").join("ca-bundle.pem");
    let write = || -> std::io::Result<()> {
        std::fs::create_dir_all(path.parent().expect("a parent"))?;
        std::fs::write(&path, &text)
    };
    write().map_err(|e| format!("writing the egress certificate bundle: {e}"))?;
    *written = Some(path.clone());
    Ok(path)
}

/// The TLS client settings the proxy reaches hosts with: the host's own
/// trusted roots.
fn client_config() -> Result<Arc<rustls::ClientConfig>, String> {
    static CONFIG: LazyLock<Result<Arc<rustls::ClientConfig>, String>> = LazyLock::new(|| {
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let mut roots = rustls::RootCertStore::empty();
        for cert in rustls_native_certs::load_native_certs().certs {
            let _ = roots.add(cert);
        }
        if roots.is_empty() {
            return Err("the host has no trusted root certificates".into());
        }
        let mut config = rustls::ClientConfig::builder()
            .with_root_certificates(roots)
            .with_no_client_auth();
        config.alpn_protocols = vec![b"http/1.1".to_vec()];
        Ok(Arc::new(config))
    });
    CONFIG.clone()
}

/// A running proxy, on loopback at `port`; it stops when dropped.
pub(super) struct Proxy {
    pub port: u16,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for Proxy {
    fn drop(&mut self) {
        self.task.abort();
    }
}

/// Starts a proxy for `secrets`.
pub(super) async fn start(secrets: Vec<Secret>) -> Result<Proxy, String> {
    authority()?;
    let listener = TcpListener::bind(("127.0.0.1", 0))
        .await
        .map_err(|e| format!("starting the egress proxy: {e}"))?;
    let port = listener.local_addr().map_err(|e| e.to_string())?.port();
    let secrets = Arc::new(secrets);
    let task = tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            let secrets = secrets.clone();
            tokio::spawn(async move {
                if let Err(err) = serve(stream, &secrets).await {
                    log::debug!("orangu-server: [egress] {err}");
                }
            });
        }
    });
    Ok(Proxy { port, task })
}

/// A request head, read up to its blank line, and what was read past it.
async fn read_head<S: AsyncRead + Unpin>(stream: &mut S) -> std::io::Result<(Vec<u8>, Vec<u8>)> {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        if let Some(at) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let rest = buf.split_off(at + 4);
            return Ok((buf, rest));
        }
        if buf.len() > MAX_HEAD {
            return Err(std::io::Error::other("a request head too large"));
        }
        let n = stream.read(&mut chunk).await?;
        if n == 0 {
            return Err(std::io::Error::other(
                "the client closed before a whole request",
            ));
        }
        buf.extend_from_slice(&chunk[..n]);
    }
}

/// A request's headers, as name and raw value.
type Headers = Vec<(String, Vec<u8>)>;

/// The parts of a request head: method, target, and headers.
fn parse_head(head: &[u8]) -> std::io::Result<(String, String, Headers)> {
    let mut headers = [httparse::EMPTY_HEADER; 128];
    let mut request = httparse::Request::new(&mut headers);
    match request.parse(head) {
        Ok(httparse::Status::Complete(_)) => {}
        _ => return Err(std::io::Error::other("a malformed request head")),
    }
    Ok((
        request.method.unwrap_or_default().to_string(),
        request.path.unwrap_or_default().to_string(),
        request
            .headers
            .iter()
            .map(|h| (h.name.to_string(), h.value.to_vec()))
            .collect(),
    ))
}

/// Reads exactly `n` bytes more than `have` holds.
async fn read_exact_more<S: AsyncRead + Unpin>(
    stream: &mut S,
    have: &mut Vec<u8>,
    n: usize,
) -> std::io::Result<()> {
    let mut chunk = [0u8; 8192];
    while have.len() < n {
        let read = stream.read(&mut chunk).await?;
        if read == 0 {
            return Err(std::io::Error::other("the client closed mid-body"));
        }
        have.extend_from_slice(&chunk[..read]);
    }
    Ok(())
}

/// A request's body: by `Content-Length`, decoded from chunks, or none.
async fn read_body<S: AsyncRead + Unpin>(
    stream: &mut S,
    headers: &[(String, Vec<u8>)],
    mut have: Vec<u8>,
) -> std::io::Result<Vec<u8>> {
    let header = |name: &str| {
        headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| String::from_utf8_lossy(v).trim().to_ascii_lowercase())
    };
    if header("transfer-encoding").is_some_and(|v| v.contains("chunked")) {
        let mut body = Vec::new();
        loop {
            let line_end = loop {
                if let Some(at) = have.windows(2).position(|w| w == b"\r\n") {
                    break at;
                }
                let len = have.len();
                read_exact_more(stream, &mut have, len + 1).await?;
            };
            let size_text = String::from_utf8_lossy(&have[..line_end]).to_string();
            let size = usize::from_str_radix(size_text.split(';').next().unwrap_or("").trim(), 16)
                .map_err(|_| std::io::Error::other("a malformed chunk size"))?;
            have.drain(..line_end + 2);
            if size == 0 {
                // Trailers, to the blank line.
                while !have.windows(2).any(|w| w == b"\r\n") {
                    let len = have.len();
                    read_exact_more(stream, &mut have, len + 1).await?;
                }
                return Ok(body);
            }
            if size > MAX_BODY.saturating_sub(body.len()) {
                return Err(std::io::Error::other("a request body too large"));
            }
            read_exact_more(stream, &mut have, size + 2).await?;
            body.extend_from_slice(&have[..size]);
            have.drain(..size + 2);
        }
    }
    let length: usize = header("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    if length > MAX_BODY {
        return Err(std::io::Error::other("a request body too large"));
    }
    read_exact_more(stream, &mut have, length).await?;
    have.truncate(length);
    Ok(have)
}

fn replace_all(data: &[u8], from: &[u8], to: &[u8]) -> Vec<u8> {
    if from.is_empty() {
        return data.to_vec();
    }
    let mut out = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        if data[i..].starts_with(from) {
            out.extend_from_slice(to);
            i += from.len();
        } else {
            out.push(data[i]);
            i += 1;
        }
    }
    out
}

/// The request as it goes on to `host`: origin-form target, the secrets
/// allowed there put in place of their placeholders, its length restated,
/// and `Connection: close`.
fn rewrite(
    method: &str,
    target: &str,
    headers: &[(String, Vec<u8>)],
    body: Vec<u8>,
    host: &str,
    secrets: &[Secret],
) -> Vec<u8> {
    let allowed: Vec<&Secret> = secrets.iter().filter(|s| s.allowed_for(host)).collect();
    let mut body = body;
    for secret in &allowed {
        if secret.body {
            body = replace_all(
                &body,
                secret.placeholder.as_bytes(),
                secret.value.as_bytes(),
            );
        }
    }
    let mut out = format!("{method} {target} HTTP/1.1\r\n").into_bytes();
    for (name, value) in headers {
        let lower = name.to_ascii_lowercase();
        if matches!(
            lower.as_str(),
            "proxy-connection"
                | "connection"
                | "keep-alive"
                | "transfer-encoding"
                | "content-length"
                | "proxy-authorization"
        ) {
            continue;
        }
        let mut value = value.clone();
        for secret in &allowed {
            if secret.header {
                value = replace_all(
                    &value,
                    secret.placeholder.as_bytes(),
                    secret.value.as_bytes(),
                );
            }
        }
        out.extend_from_slice(name.as_bytes());
        out.extend_from_slice(b": ");
        out.extend_from_slice(&value);
        out.extend_from_slice(b"\r\n");
    }
    if !body.is_empty() || !matches!(method, "GET" | "HEAD" | "DELETE" | "OPTIONS") {
        out.extend_from_slice(format!("Content-Length: {}\r\n", body.len()).as_bytes());
    }
    out.extend_from_slice(b"Connection: close\r\n\r\n");
    out.extend_from_slice(&body);
    out
}

fn host_and_port(authority: &str, default_port: u16) -> (String, u16) {
    let authority = authority.trim();
    if let Some(rest) = authority.strip_prefix('[')
        && let Some((host, after)) = rest.split_once(']')
    {
        let port = after.strip_prefix(':').and_then(|p| p.parse().ok());
        return (host.to_string(), port.unwrap_or(default_port));
    }
    match authority.rsplit_once(':') {
        Some((host, port)) if port.parse::<u16>().is_ok() => {
            (host.to_string(), port.parse().unwrap_or(default_port))
        }
        _ => (authority.to_string(), default_port),
    }
}

/// Sends `request` to `host` and relays the answer to `client` until the
/// host closes.
async fn forward<C, U>(client: &mut C, mut upstream: U, request: &[u8]) -> std::io::Result<()>
where
    C: AsyncWrite + Unpin,
    U: AsyncRead + AsyncWrite + Unpin,
{
    upstream.write_all(request).await?;
    upstream.flush().await?;
    tokio::io::copy(&mut upstream, client).await?;
    client.flush().await?;
    let _ = client.shutdown().await;
    Ok(())
}

async fn serve(mut stream: TcpStream, secrets: &[Secret]) -> std::io::Result<()> {
    let (head, rest) = read_head(&mut stream).await?;
    let (method, target, headers) = parse_head(&head)?;
    if method.eq_ignore_ascii_case("CONNECT") {
        let (host, port) = host_and_port(&target, 443);
        stream
            .write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            .await?;
        if !secrets.iter().any(|s| s.allowed_for(&host)) {
            // Nothing to put in: the bytes pass through as they are.
            let mut upstream = TcpStream::connect((host.as_str(), port)).await?;
            upstream.write_all(&rest).await?;
            tokio::io::copy_bidirectional(&mut stream, &mut upstream).await?;
            return Ok(());
        }
        let config = authority()
            .and_then(|a| a.for_host(&host))
            .map_err(std::io::Error::other)?;
        let mut tls = tokio_rustls::TlsAcceptor::from(config)
            .accept(stream)
            .await?;
        let (head, rest) = read_head(&mut tls).await?;
        let (method, target, headers) = parse_head(&head)?;
        let body = read_body(&mut tls, &headers, rest).await?;
        let request = rewrite(&method, &target, &headers, body, &host, secrets);
        let connector =
            tokio_rustls::TlsConnector::from(client_config().map_err(std::io::Error::other)?);
        let name = rustls::pki_types::ServerName::try_from(host.clone())
            .map_err(|_| std::io::Error::other("not a host name"))?;
        let tcp = TcpStream::connect((host.as_str(), port)).await?;
        let upstream = connector.connect(name, tcp).await?;
        return forward(&mut tls, upstream, &request).await;
    }
    // Plain HTTP, as a proxy is asked for it: an absolute URL.
    let url = reqwest::Url::parse(&target).map_err(|_| std::io::Error::other("not a URL"))?;
    let host = url.host_str().unwrap_or_default().to_string();
    let port = url.port_or_known_default().unwrap_or(80);
    let path = match url.query() {
        Some(query) => format!("{}?{query}", url.path()),
        None => url.path().to_string(),
    };
    let body = read_body(&mut stream, &headers, rest).await?;
    let request = rewrite(&method, &path, &headers, body, &host, secrets);
    let upstream = TcpStream::connect((host.as_str(), port)).await?;
    forward(&mut stream, upstream, &request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    fn secret(hosts: Option<Vec<&str>>, header: bool, body: bool) -> Secret {
        let mut s = Secret::new(
            "API_KEY".into(),
            "real-key".into(),
            hosts.map(|h| h.into_iter().map(str::to_string).collect()),
            header,
            body,
        );
        s.placeholder = "PLACEHOLDER".into();
        s
    }

    #[test]
    fn a_secret_goes_only_where_it_is_allowed() {
        let s = secret(Some(vec!["api.example.com", "*.twilio.com"]), true, true);
        assert!(s.allowed_for("api.example.com"));
        assert!(s.allowed_for("API.example.com"));
        assert!(s.allowed_for("eu.twilio.com"));
        assert!(!s.allowed_for("twilio.com.evil.example"));
        assert!(!s.allowed_for("example.com"));
        assert!(secret(None, true, true).allowed_for("anything.example"));
    }

    /// Placeholders become the secret where the credential allows, for a
    /// host it allows, and the length is restated.
    #[test]
    fn a_request_has_its_placeholders_replaced_where_allowed() {
        let headers = vec![
            ("Host".to_string(), b"api.example.com".to_vec()),
            ("Authorization".to_string(), b"Bearer PLACEHOLDER".to_vec()),
            ("Proxy-Connection".to_string(), b"keep-alive".to_vec()),
        ];
        let body = b"key=PLACEHOLDER".to_vec();
        let header_only = [secret(Some(vec!["api.example.com"]), true, false)];
        let out = String::from_utf8(rewrite(
            "POST",
            "/v1",
            &headers,
            body.clone(),
            "api.example.com",
            &header_only,
        ))
        .unwrap();
        assert!(out.starts_with("POST /v1 HTTP/1.1\r\n"), "{out}");
        assert!(out.contains("Authorization: Bearer real-key\r\n"), "{out}");
        assert!(
            out.ends_with("Content-Length: 15\r\nConnection: close\r\n\r\nkey=PLACEHOLDER"),
            "{out}"
        );
        assert!(!out.contains("Proxy-Connection"));
        let both = [secret(Some(vec!["api.example.com"]), true, true)];
        let out = String::from_utf8(rewrite(
            "POST",
            "/v1",
            &headers,
            body.clone(),
            "api.example.com",
            &both,
        ))
        .unwrap();
        assert!(
            out.ends_with("Content-Length: 12\r\nConnection: close\r\n\r\nkey=real-key"),
            "{out}"
        );
        let out = String::from_utf8(rewrite(
            "POST",
            "/v1",
            &headers,
            body,
            "other.example",
            &both,
        ))
        .unwrap();
        assert!(!out.contains("real-key"), "{out}");
    }

    /// A chunked body is read whole.
    #[tokio::test]
    async fn a_chunked_body_is_decoded() {
        let headers = vec![("Transfer-Encoding".to_string(), b"chunked".to_vec())];
        let mut wire: &[u8] = b"4\r\nWiki\r\n5;x=1\r\npedia\r\n0\r\n\r\n";
        let body = read_body(&mut wire, &headers, Vec::new()).await.unwrap();
        assert_eq!(body, b"Wikipedia");
    }

    /// An oversized chunk that would overflow `body.len() + size` is rejected without panicking.
    #[tokio::test]
    async fn a_chunk_size_overflow_is_rejected() {
        let headers = vec![("Transfer-Encoding".to_string(), b"chunked".to_vec())];
        let payload = format!("1\r\na\r\n{:x}\r\n", usize::MAX);
        let mut wire = payload.as_bytes();
        let err = read_body(&mut wire, &headers, Vec::new())
            .await
            .unwrap_err();
        assert_eq!(err.to_string(), "a request body too large");
    }

    /// The authority issues a certificate a TLS server can present.
    #[test]
    fn the_authority_issues_certificates() {
        assert!(authority().unwrap().pem.contains("BEGIN CERTIFICATE"));
        assert!(authority().unwrap().for_host("api.example.com").is_ok());
    }
}
