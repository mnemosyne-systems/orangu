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

//! Logging in: the agent's keys, then key files, then a password, stopping at
//! the first that works. Passphrases and passwords come from the prompt alone.

use super::host::{SshPaths, Target};
use super::{HostKeyCheck, Prompts};
use anyhow::{Result, anyhow};
use russh::MethodKind;
use russh::client::{AuthResult, Handle};
use russh::keys::{self, PrivateKeyWithHashAlg};
use std::path::Path;
use std::sync::Arc;

/// How many times one passphrase or password may be asked for.
const PROMPT_ATTEMPTS: usize = 3;
/// How many of an agent's keys are offered: a server cuts a client off after
/// a few failures (6 by default), and key files and passwords come after.
const MAX_AGENT_KEYS: usize = 4;

/// What the next method needs to know from the last failure.
enum Attempt {
    Success,
    Failure,
}

fn outcome(result: AuthResult, offered: &mut Vec<MethodKind>) -> Result<Attempt> {
    match result {
        AuthResult::Success => Ok(Attempt::Success),
        AuthResult::Failure {
            partial_success: true,
            ..
        } => Err(anyhow!(
            "the server wants a second authentication factor, which is not supported"
        )),
        AuthResult::Failure {
            remaining_methods, ..
        } => {
            *offered = remaining_methods.iter().copied().collect();
            Ok(Attempt::Failure)
        }
    }
}

/// Logs `handle` in and names the method that worked.
pub async fn authenticate(
    handle: &mut Handle<HostKeyCheck>,
    target: &Target,
    paths: &SshPaths,
    prompts: &Prompts,
) -> Result<String> {
    let mut offered = Vec::new();
    let none = handle.authenticate_none(&target.user).await?;
    if let Attempt::Success = outcome(none, &mut offered)? {
        return Ok("no authentication".to_string());
    }
    let mut tried: Vec<String> = Vec::new();

    if offered.contains(&MethodKind::PublicKey) {
        #[cfg(unix)]
        if let Some(socket) = &paths.agent_socket
            && let Some(method) =
                with_agent(handle, target, socket, &mut offered, &mut tried).await?
        {
            return Ok(method);
        }
        #[cfg(not(unix))]
        let _ = paths;
        if !handle.is_closed()
            && let Some(method) =
                with_key_files(handle, target, prompts, &mut offered, &mut tried).await?
        {
            return Ok(method);
        }
    }

    if offered.contains(&MethodKind::Password) && !handle.is_closed() {
        if !prompts.available() {
            tried.push("no password, as nobody is here to type one".to_string());
        }
        for _ in 0..PROMPT_ATTEMPTS {
            let Some(password) = prompts
                .ask(format!("Password for {}: ", target.label()))
                .await
            else {
                break;
            };
            let result = handle.authenticate_password(&target.user, password).await?;
            if let Attempt::Success = outcome(result, &mut offered)? {
                return Ok("a password".to_string());
            }
            tried.push("a password".to_string());
            if handle.is_closed() || !offered.contains(&MethodKind::Password) {
                break;
            }
        }
    }

    if handle.is_closed() {
        return Err(anyhow!(
            "could not log in to {}: the server closed the connection, possibly after too many \
             attempts (MaxAuthTries); tried {}",
            target.label(),
            tried.join(", ")
        ));
    }
    Err(anyhow!(
        "could not log in to {}: tried {}; the server accepts: {}",
        target.label(),
        if tried.is_empty() {
            "nothing".to_string()
        } else {
            tried.join(", ")
        },
        offered
            .iter()
            .map(<&'static str>::from)
            .collect::<Vec<_>>()
            .join(", ")
    ))
}

/// The hash an RSA key signs with here, which the server chooses among.
async fn rsa_hash(handle: &Handle<HostKeyCheck>) -> Option<keys::HashAlg> {
    handle
        .best_supported_rsa_hash()
        .await
        .ok()
        .flatten()
        .flatten()
}

/// Offers the agent's keys, those of the host's identity files first.
#[cfg(unix)]
async fn with_agent(
    handle: &mut Handle<HostKeyCheck>,
    target: &Target,
    socket: &Path,
    offered: &mut Vec<MethodKind>,
    tried: &mut Vec<String>,
) -> Result<Option<String>> {
    use keys::agent::client::AgentClient;

    let Ok(mut agent) = AgentClient::connect_uds(socket).await else {
        return Ok(None);
    };
    if let Some(binding) = handle.agent_session_binding().cloned() {
        // Older agents do not know session binding, and still sign.
        let _ = agent.bind_session(&binding, false).await;
    }
    let Ok(identities) = agent.request_identities().await else {
        return Ok(None);
    };
    let preferred: Vec<keys::PublicKey> = target
        .identity_files
        .iter()
        .filter_map(|file| {
            let mut public = file.clone().into_os_string();
            public.push(".pub");
            keys::load_public_key(public).ok()
        })
        .collect();
    let mut keys: Vec<keys::PublicKey> = identities
        .iter()
        .map(|identity| identity.public_key().into_owned())
        .collect();
    keys.sort_by_key(|key| !preferred.contains(key));
    keys.truncate(MAX_AGENT_KEYS);

    let hash = rsa_hash(handle).await;
    for key in keys {
        let label = format!("agent key {}", key.fingerprint(keys::HashAlg::Sha256));
        let hash_alg = key.algorithm().is_rsa().then_some(hash).flatten();
        // A key the agent will not sign with fails this attempt only.
        let Ok(result) = handle
            .authenticate_publickey_with(&target.user, key, hash_alg, &mut agent)
            .await
        else {
            tried.push(format!("{label} (the agent did not sign)"));
            continue;
        };
        if let Attempt::Success = outcome(result, offered)? {
            return Ok(Some(label));
        }
        tried.push(label);
        if handle.is_closed() || !offered.contains(&MethodKind::PublicKey) {
            break;
        }
    }
    Ok(None)
}

/// A key file's key, asking for its passphrase when it has one. `None` when
/// the file cannot be used, with why in `tried`.
async fn load_key(
    file: &Path,
    prompts: &Prompts,
    tried: &mut Vec<String>,
) -> Option<keys::PrivateKey> {
    let shown = file.display();
    match keys::load_secret_key(file, None) {
        Ok(key) => return Some(key),
        Err(keys::Error::KeyIsEncrypted) => {}
        Err(error) => {
            tried.push(format!("{shown} (unusable: {error})"));
            return None;
        }
    }
    if !prompts.available() {
        tried.push(format!(
            "{shown} (encrypted, and nobody is here to unlock it)"
        ));
        return None;
    }
    for _ in 0..PROMPT_ATTEMPTS {
        let Some(passphrase) = prompts.ask(format!("Passphrase for {shown}: ")).await else {
            tried.push(format!("{shown} (passphrase not given)"));
            return None;
        };
        if let Ok(key) = keys::load_secret_key(file, Some(&passphrase)) {
            return Some(key);
        }
    }
    tried.push(format!("{shown} (wrong passphrase)"));
    None
}

async fn with_key_files(
    handle: &mut Handle<HostKeyCheck>,
    target: &Target,
    prompts: &Prompts,
    offered: &mut Vec<MethodKind>,
    tried: &mut Vec<String>,
) -> Result<Option<String>> {
    let hash = rsa_hash(handle).await;
    for file in &target.identity_files {
        if !file.exists() {
            tried.push(format!("{} (no such file)", file.display()));
            continue;
        }
        let Some(key) = load_key(file, prompts, tried).await else {
            continue;
        };
        let hash_alg = key.algorithm().is_rsa().then_some(hash).flatten();
        let key = PrivateKeyWithHashAlg::new(Arc::new(key), hash_alg);
        let result = handle.authenticate_publickey(&target.user, key).await?;
        if let Attempt::Success = outcome(result, offered)? {
            return Ok(Some(format!("key {}", file.display())));
        }
        tried.push(file.display().to_string());
        if handle.is_closed() || !offered.contains(&MethodKind::PublicKey) {
            break;
        }
    }
    Ok(None)
}
