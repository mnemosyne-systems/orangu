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

//! SSH sessions for the `use_ssh` subagent: log in, copy files, log out.
//!
//! The model names a host and nothing else. Credentials come from the user's
//! agent, their key files, or a prompt answered in the TUI, so a passphrase or
//! password is never a tool argument and never reaches the transcript.
//! Remote commands are not offered.

mod auth;
mod host;
mod known_hosts;
mod transfer;

use anyhow::{Context, Result, anyhow};
use russh::client;
use russh::keys::{self, HashAlg, PublicKeyBase64};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

pub use host::SshPaths;
pub use transfer::Direction;

/// How long a session stays open with nothing copied through it.
const IDLE_TIMEOUT: Duration = Duration::from_secs(15 * 60);
/// How long a TCP connection may take to open.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
/// How long the user has to answer one prompt.
const PROMPT_TIMEOUT: Duration = Duration::from_secs(120);
/// How long a server may take to answer one SFTP request, in seconds.
const SFTP_TIMEOUT_SECONDS: u64 = 60;

/// Answers the questions a login needs a person for. The reply is the typed
/// text, or `None` when the prompt was cancelled.
pub trait Prompter: Send + Sync {
    fn ask(&self, prompt: String) -> tokio::sync::oneshot::Receiver<Option<String>>;
}

/// [`Prompter`], if there is anyone to ask.
#[derive(Clone)]
struct Prompts {
    prompter: Option<Arc<dyn Prompter>>,
    timeout: Duration,
}

impl Prompts {
    fn available(&self) -> bool {
        self.prompter.is_some()
    }

    /// The answer to `prompt`; `None` when nobody is there to ask, the prompt
    /// is cancelled, or no answer comes in time.
    async fn ask(&self, prompt: String) -> Option<String> {
        let reply = self.prompter.as_ref()?.ask(prompt);
        tokio::time::timeout(self.timeout, reply)
            .await
            .ok()?
            .ok()
            .flatten()
    }

    async fn confirm(&self, prompt: String) -> bool {
        self.ask(prompt)
            .await
            .is_some_and(|answer| matches!(answer.trim().to_lowercase().as_str(), "y" | "yes"))
    }
}

/// Checks the server's host key against `known_hosts`, during key exchange.
struct HostKeyCheck {
    host: String,
    port: u16,
    known_hosts: PathBuf,
    prompts: Prompts,
    /// Why the key was refused, for the login error.
    refusal: Arc<Mutex<Option<String>>>,
    /// The accepted key, for the login result.
    fingerprint: Arc<Mutex<Option<String>>>,
}

impl HostKeyCheck {
    async fn accepts(&self, key: &keys::PublicKey) -> std::result::Result<(), String> {
        let name = format!("{}:{}", self.host, self.port);
        let fingerprint = key.fingerprint(HashAlg::Sha256).to_string();
        if is_revoked(&self.known_hosts, key) {
            return Err(format!(
                "the host key of {name} ({fingerprint}) is marked @revoked in {}",
                self.known_hosts.display()
            ));
        }
        let changed = |line: usize| {
            format!(
                "the host key of {name} CHANGED ({fingerprint} is not the key recorded at \
                 {}:{line}); refusing to connect, this can be an attack",
                self.known_hosts.display()
            )
        };
        match keys::check_known_hosts_path(&self.host, self.port, key, &self.known_hosts) {
            Ok(true) => return Ok(()),
            Ok(false) => {}
            Err(keys::Error::KeyChanged { line }) => return Err(changed(line)),
            Err(error) => {
                return Err(format!(
                    "cannot read {} for {name}: {error}",
                    self.known_hosts.display()
                ));
            }
        }
        // russh matches only exact and hashed names; `*.example.com` and the
        // like are read here.
        let wildcard = known_hosts::wildcard_host_keys(&self.host, self.port, &self.known_hosts);
        match known_hosts::compare(key, &wildcard) {
            known_hosts::Recorded::Matches => return Ok(()),
            known_hosts::Recorded::Changed(line) => return Err(changed(line)),
            known_hosts::Recorded::Absent => {}
        }
        if !self.prompts.available() {
            return Err(format!(
                "{name} is not in {} and nobody is here to confirm its key {fingerprint}; \
                 connect to it once from the TUI or with ssh",
                self.known_hosts.display()
            ));
        }
        let known_with_other_type = !wildcard.is_empty()
            || keys::known_hosts::known_host_keys_path(&self.host, self.port, &self.known_hosts)
                .is_ok_and(|keys| !keys.is_empty());
        let intro = if known_with_other_type {
            format!("{name} is known with a different key type; it now offers")
        } else {
            format!("Unknown host {name} offers")
        };
        if !self
            .prompts
            .confirm(format!(
                "{intro} key {fingerprint}. Trust it and add it to known_hosts (yes/no)? "
            ))
            .await
        {
            return Err(format!(
                "the host key {fingerprint} of {name} was not trusted"
            ));
        }
        keys::known_hosts::learn_known_hosts_path(&self.host, self.port, key, &self.known_hosts)
            .map_err(|error| {
                format!(
                    "cannot record the host key in {}: {error}",
                    self.known_hosts.display()
                )
            })
    }
}

impl client::Handler for HostKeyCheck {
    type Error = russh::Error;

    async fn check_server_key(
        &mut self,
        server_public_key: &keys::PublicKeyOrCertificate,
    ) -> std::result::Result<bool, Self::Error> {
        let keys::PublicKeyOrCertificate::PublicKey { key, .. } = server_public_key else {
            if let Ok(mut refusal) = self.refusal.lock() {
                *refusal = Some(format!(
                    "{}:{} presented a host certificate, which is not supported",
                    self.host, self.port
                ));
            }
            return Ok(false);
        };
        match self.accepts(key).await {
            Ok(()) => {
                if let Ok(mut fingerprint) = self.fingerprint.lock() {
                    *fingerprint = Some(key.fingerprint(HashAlg::Sha256).to_string());
                }
                Ok(true)
            }
            Err(reason) => {
                if let Ok(mut refusal) = self.refusal.lock() {
                    *refusal = Some(reason);
                }
                Ok(false)
            }
        }
    }
}

/// Whether `known_hosts` marks `key` `@revoked`, whichever host it names.
fn is_revoked(known_hosts: &Path, key: &keys::PublicKey) -> bool {
    let Ok(contents) = std::fs::read_to_string(known_hosts) else {
        return false;
    };
    let encoded = key.public_key_base64();
    contents.lines().any(|line| {
        let mut fields = line.split_whitespace();
        fields.next() == Some("@revoked") && fields.nth(2) == Some(encoded.as_str())
    })
}

/// One logged-in host.
struct Session {
    handle: client::Handle<HostKeyCheck>,
    sftp: Arc<russh_sftp::client::SftpSession>,
    label: String,
    last_used: Mutex<Instant>,
}

impl Session {
    fn touch(&self) {
        if let Ok(mut last_used) = self.last_used.lock() {
            *last_used = Instant::now();
        }
    }

    fn idle_for(&self) -> Duration {
        self.last_used
            .lock()
            .map(|last_used| last_used.elapsed())
            .unwrap_or_default()
    }

    async fn close(&self) {
        let _ = self.sftp.close().await;
        let _ = self
            .handle
            .disconnect(russh::Disconnect::ByApplication, "", "en")
            .await;
    }
}

/// The SSH sessions of one conversation. Dropping it closes them.
pub struct SshSessions {
    mode: crate::config::SshMode,
    prompts: Prompts,
    paths: SshPaths,
    idle_timeout: Duration,
    sessions: Mutex<HashMap<String, Arc<Session>>>,
    /// The `user@host:port`s the user has approved in this conversation.
    approved: Mutex<HashSet<String>>,
    /// Holds logins to one at a time, so their prompts cannot interleave.
    login_gate: tokio::sync::Mutex<()>,
    reaper_started: std::sync::atomic::AtomicBool,
}

impl SshSessions {
    /// Sessions under `[orangu].ssh = mode`; `None` when SSH is off.
    /// `prompter` is whoever can answer a login's questions, if anyone.
    pub fn for_mode(
        mode: crate::config::SshMode,
        prompter: Option<Arc<dyn Prompter>>,
    ) -> Option<Arc<Self>> {
        (mode != crate::config::SshMode::Deny)
            .then(|| Self::new(mode, prompter, SshPaths::of_user()))
    }

    fn new(
        mode: crate::config::SshMode,
        prompter: Option<Arc<dyn Prompter>>,
        paths: SshPaths,
    ) -> Arc<Self> {
        Arc::new(Self {
            mode,
            prompts: Prompts {
                prompter,
                timeout: PROMPT_TIMEOUT,
            },
            paths,
            idle_timeout: IDLE_TIMEOUT,
            sessions: Mutex::default(),
            approved: Mutex::default(),
            login_gate: tokio::sync::Mutex::new(()),
            reaper_started: false.into(),
        })
    }

    fn session(&self, name: &str) -> Result<Arc<Session>> {
        let sessions = self
            .sessions
            .lock()
            .map_err(|_| anyhow!("SSH sessions mutex poisoned"))?;
        if let Some(session) = sessions.get(name) {
            return Ok(session.clone());
        }
        let mut open: Vec<&str> = sessions.keys().map(String::as_str).collect();
        open.sort_unstable();
        Err(anyhow!(
            "no SSH session '{name}'; open sessions: {}",
            if open.is_empty() {
                "none".to_string()
            } else {
                open.join(", ")
            }
        ))
    }

    /// The session `name` if it is still connected; a dead one is forgotten.
    fn live_session(&self, name: &str) -> Option<Arc<Session>> {
        let session = self.session(name).ok()?;
        if !session.handle.is_closed() {
            return Some(session);
        }
        if let Ok(mut sessions) = self.sessions.lock() {
            sessions.remove(name);
        }
        None
    }

    /// Asks the user, once per conversation, whether this host may be used.
    async fn approve(&self, target: &host::Target) -> Result<()> {
        if self.mode == crate::config::SshMode::Auto {
            return Ok(());
        }
        let key = target.label();
        if self.approved.lock().is_ok_and(|set| set.contains(&key)) {
            return Ok(());
        }
        if !self.prompts.available() {
            return Err(anyhow!(
                "logging in to {key} needs approval, which only the TUI can give; \
                 [orangu].ssh = auto skips it"
            ));
        }
        if !self
            .prompts
            .confirm(format!(
                "Allow this conversation to log in to {key} and copy files (yes/no)? "
            ))
            .await
        {
            return Err(anyhow!("the user did not allow logging in to {key}"));
        }
        if let Ok(mut approved) = self.approved.lock() {
            approved.insert(key);
        }
        Ok(())
    }

    /// `ssh_login`: opens a session to `spec`, or finds the one already open.
    pub async fn login(self: &Arc<Self>, spec: &str) -> Result<String> {
        let spec = spec.trim();
        let _gate = self.login_gate.lock().await;
        if let Some(session) = self.live_session(spec) {
            session.touch();
            return Ok(format!(
                "Already logged in to {} as session \"{spec}\".",
                session.label
            ));
        }
        let target = host::resolve(spec, &self.paths)?;
        self.approve(&target).await?;

        let refusal = Arc::new(Mutex::new(None));
        let fingerprint = Arc::new(Mutex::new(None));
        let check = HostKeyCheck {
            host: target.host.clone(),
            port: target.port,
            known_hosts: target.known_hosts.clone(),
            prompts: self.prompts.clone(),
            refusal: refusal.clone(),
            fingerprint: fingerprint.clone(),
        };
        let address = (target.host.as_str(), target.port);
        let stream = tokio::time::timeout(CONNECT_TIMEOUT, tokio::net::TcpStream::connect(address))
            .await
            .map_err(|_| anyhow!("connecting to {} timed out", target.label()))?
            .with_context(|| format!("connecting to {}", target.label()))?;
        let config = Arc::new(client::Config {
            keepalive_interval: Some(Duration::from_secs(30)),
            ..client::Config::default()
        });
        let mut handle = match tokio::time::timeout(
            CONNECT_TIMEOUT + 2 * self.prompts.timeout,
            client::connect_stream(config, stream, check),
        )
        .await
        {
            Err(_) => {
                return Err(anyhow!(
                    "the SSH handshake with {} timed out",
                    target.label()
                ));
            }
            Ok(Ok(handle)) => handle,
            Ok(Err(error)) => {
                let refused = refusal.lock().ok().and_then(|mut reason| reason.take());
                return Err(match refused {
                    Some(reason) => anyhow!("{reason}"),
                    None => anyhow!("the SSH handshake with {} failed: {error}", target.label()),
                });
            }
        };

        let method = auth::authenticate(&mut handle, &target, &self.paths, &self.prompts).await?;
        let channel = handle
            .channel_open_session()
            .await
            .context("opening an SSH channel")?;
        channel
            .request_subsystem(true, "sftp")
            .await
            .with_context(|| format!("{} refused the SFTP subsystem", target.label()))?;
        let sftp = russh_sftp::client::SftpSession::new(channel.into_stream())
            .await
            .with_context(|| format!("starting SFTP on {}", target.label()))?;
        sftp.set_timeout(SFTP_TIMEOUT_SECONDS);

        let label = target.label();
        let session = Arc::new(Session {
            handle,
            sftp: Arc::new(sftp),
            label: label.clone(),
            last_used: Mutex::new(Instant::now()),
        });
        self.sessions
            .lock()
            .map_err(|_| anyhow!("SSH sessions mutex poisoned"))?
            .insert(spec.to_string(), session);
        self.start_reaper();

        let fingerprint = fingerprint
            .lock()
            .ok()
            .and_then(|fingerprint| fingerprint.clone())
            .unwrap_or_default();
        let mut report = format!(
            "Logged in to {label} as session \"{spec}\" with {method}; host key {fingerprint}."
        );
        for note in target.notes {
            report.push_str("\nNote: ");
            report.push_str(&note);
        }
        Ok(report)
    }

    /// `ssh_copy`: copies one file between the workspace and session `name`.
    pub async fn copy(
        &self,
        name: &str,
        direction: Direction,
        workspace: &Path,
        local: &str,
        remote: &str,
        overwrite: bool,
    ) -> Result<String> {
        let session = self.live_session(name).ok_or_else(|| {
            anyhow!("SSH session '{name}' is not open; log in with ssh_login first")
        })?;
        session.touch();
        let result = transfer::copy(
            &session.sftp,
            direction,
            workspace,
            local,
            remote,
            overwrite,
            &session.label,
        )
        .await;
        session.touch();
        result
    }

    /// `ssh_logout`: closes session `name`.
    pub async fn logout(&self, name: &str) -> Result<String> {
        let session = self.session(name)?;
        if let Ok(mut sessions) = self.sessions.lock() {
            sessions.remove(name);
        }
        session.close().await;
        Ok(format!("Logged out of {}.", session.label))
    }

    /// Closes every session with nothing running through it for the idle
    /// timeout.
    async fn close_idle(&self) {
        let idle: Vec<Arc<Session>> = match self.sessions.lock() {
            Ok(mut sessions) => {
                let names: Vec<String> = sessions
                    .iter()
                    // A copy holds its own reference while it runs.
                    .filter(|(_, session)| {
                        Arc::strong_count(session) == 1
                            && (session.idle_for() >= self.idle_timeout
                                || session.handle.is_closed())
                    })
                    .map(|(name, _)| name.clone())
                    .collect();
                names
                    .iter()
                    .filter_map(|name| sessions.remove(name))
                    .collect()
            }
            Err(_) => Vec::new(),
        };
        for session in idle {
            session.close().await;
        }
    }

    fn start_reaper(self: &Arc<Self>) {
        if self
            .reaper_started
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            return;
        }
        let sessions: Weak<Self> = Arc::downgrade(self);
        let period = self
            .idle_timeout
            .div_f32(4.0)
            .max(Duration::from_millis(50));
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(period).await;
                match sessions.upgrade() {
                    Some(sessions) => sessions.close_idle().await,
                    None => return,
                }
            }
        });
    }
}

#[cfg(all(test, unix))]
mod tests;
