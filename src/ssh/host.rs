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

//! What `ssh_login`'s `host` argument means: an alias from `~/.ssh/config`, or
//! `[user@]host[:port]`.

use anyhow::{Context, Result, anyhow};
use std::path::{Path, PathBuf};

/// The files and agent a login uses.
#[derive(Clone, Debug)]
pub struct SshPaths {
    pub config: PathBuf,
    pub known_hosts: PathBuf,
    /// Where the default identity files live.
    pub directory: PathBuf,
    pub agent_socket: Option<PathBuf>,
}

impl SshPaths {
    /// The current user's `~/.ssh`, and their `SSH_AUTH_SOCK`.
    pub fn of_user() -> Self {
        let directory = home::home_dir().unwrap_or_default().join(".ssh");
        Self {
            config: directory.join("config"),
            known_hosts: directory.join("known_hosts"),
            directory,
            agent_socket: std::env::var_os("SSH_AUTH_SOCK").map(PathBuf::from),
        }
    }
}

/// Where a login goes and with what.
#[derive(Debug)]
pub struct Target {
    pub user: String,
    pub host: String,
    pub port: u16,
    /// The keys named for this host in the SSH config, or the default ones.
    pub identity_files: Vec<PathBuf>,
    pub known_hosts: PathBuf,
    /// What the login result should warn about.
    pub notes: Vec<String>,
}

impl Target {
    pub fn label(&self) -> String {
        format!("{}@{}:{}", self.user, self.host, self.port)
    }
}

/// `[user@]host[:port]`, with `[host]:port` for an IPv6 address.
fn parse_spec(spec: &str) -> Result<(Option<&str>, &str, Option<u16>)> {
    let refuse = || anyhow!("invalid SSH host '{}'", spec.escape_default());
    if spec.is_empty()
        || spec.starts_with('-')
        || spec.chars().any(|c| c.is_whitespace() || c.is_control())
    {
        return Err(refuse());
    }
    let (user, rest) = match spec.rsplit_once('@') {
        Some(("", _)) => return Err(refuse()),
        Some((user, rest)) => (Some(user), rest),
        None => (None, spec),
    };
    let port = |text: &str| text.parse::<u16>().map_err(|_| refuse());
    let (host, port) = if let Some(rest) = rest.strip_prefix('[') {
        let (host, after) = rest.split_once(']').ok_or_else(refuse)?;
        match after.strip_prefix(':') {
            Some(number) => (host, Some(port(number)?)),
            None if after.is_empty() => (host, None),
            None => return Err(refuse()),
        }
    } else {
        match rest.split_once(':') {
            // More than one colon: a bare IPv6 address.
            Some((_, after)) if after.contains(':') => (rest, None),
            Some((host, number)) => (host, Some(port(number)?)),
            None => (rest, None),
        }
    };
    if host.is_empty() || host.starts_with('-') {
        return Err(refuse());
    }
    Ok((user, host, port))
}

/// `contents` in the one shape the SSH config parser reads: `Keyword value`
/// with a single space, inside a `Host` block. Directives before the first
/// `Host` apply to every host, and it refuses a file that has any; a `Match`
/// block it would read as part of the block before, so it becomes a block that
/// matches nothing.
fn normalized_config(contents: &str) -> String {
    let mut normalized = String::new();
    let mut in_block = false;
    for line in contents.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (keyword, value) = line
            .split_once(|c: char| c.is_whitespace() || c == '=')
            .unwrap_or((line, ""));
        let value = value.trim_start_matches(|c: char| c.is_whitespace() || c == '=');
        if keyword.eq_ignore_ascii_case("match") {
            normalized.push_str("Host !*\n");
            in_block = true;
            continue;
        }
        if !keyword.eq_ignore_ascii_case("host") && !in_block {
            normalized.push_str("Host *\n");
        }
        in_block = true;
        normalized.push_str(&format!("{keyword} {value}\n"));
    }
    normalized
}

/// The directives the SSH config parser skips, which the file uses.
fn unparsed_directives(contents: &str) -> Vec<&'static str> {
    let mut found = Vec::new();
    for (keyword, name) in [("include", "Include"), ("match", "Match")] {
        let used = contents.lines().any(|line| {
            line.split_whitespace()
                .next()
                .is_some_and(|word| word.eq_ignore_ascii_case(keyword))
        });
        if used {
            found.push(name);
        }
    }
    found
}

/// The login `spec` asks for, under the SSH config in `paths`.
pub fn resolve(spec: &str, paths: &SshPaths) -> Result<Target> {
    let (user, host, port) = parse_spec(spec)?;
    let contents = match std::fs::read_to_string(&paths.config) {
        Ok(contents) => contents,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => {
            return Err(error).with_context(|| format!("reading {}", paths.config.display()));
        }
    };
    let config = russh_config::parse(&normalized_config(&contents), host)
        .with_context(|| format!("parsing {}", paths.config.display()))?;
    if config.host_config.proxy_jump.is_some() || config.host_config.proxy_command.is_some() {
        return Err(anyhow!(
            "{host} is reached through ProxyJump or ProxyCommand in {}, which ssh_login does not \
             support; it never connects to such a host directly",
            paths.config.display()
        ));
    }
    let mut notes: Vec<String> = unparsed_directives(&contents)
        .into_iter()
        .map(|directive| {
            format!(
                "{} uses {directive}, which is not read here, so this host may resolve \
                 differently than with ssh.",
                paths.config.display()
            )
        })
        .collect();
    if config.host_config.strict_host_key_checking == Some(false) {
        notes.push(
            "StrictHostKeyChecking no is not honoured here: host keys are always checked."
                .to_string(),
        );
    }
    let identity_files = config
        .host_config
        .identity_file
        .clone()
        .unwrap_or_else(|| default_identity_files(&paths.directory));
    Ok(Target {
        user: user.map_or_else(|| config.user(), str::to_string),
        host: config.host().to_string(),
        port: port.unwrap_or_else(|| config.port()),
        identity_files,
        known_hosts: config
            .host_config
            .user_known_hosts_file
            .clone()
            .unwrap_or_else(|| paths.known_hosts.clone()),
        notes,
    })
}

fn default_identity_files(directory: &Path) -> Vec<PathBuf> {
    ["id_ed25519", "id_ecdsa", "id_rsa"]
        .iter()
        .map(|name| directory.join(name))
        .filter(|path| path.exists())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paths(config: &str) -> (tempfile::TempDir, SshPaths) {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("config"), config).unwrap();
        let paths = SshPaths {
            config: directory.path().join("config"),
            known_hosts: directory.path().join("known_hosts"),
            directory: directory.path().to_path_buf(),
            agent_socket: None,
        };
        (directory, paths)
    }

    #[test]
    fn specs_parse_with_and_without_user_and_port() {
        assert_eq!(parse_spec("box").unwrap(), (None, "box", None));
        assert_eq!(parse_spec("me@box").unwrap(), (Some("me"), "box", None));
        assert_eq!(parse_spec("box:2222").unwrap(), (None, "box", Some(2222)));
        assert_eq!(
            parse_spec("me@[::1]:2222").unwrap(),
            (Some("me"), "::1", Some(2222))
        );
        assert_eq!(parse_spec("fe80::1").unwrap(), (None, "fe80::1", None));
    }

    /// The host reaches a prompt shown in the terminal and a command-line
    /// style parser: nothing that could act on either gets through.
    #[test]
    fn hostile_specs_are_refused() {
        for spec in [
            "",
            "-oProxyCommand=x",
            "me@-box",
            "a b",
            "box\u{1b}[2J",
            "@box",
            "box:notaport",
            "[::1",
            "box\n",
        ] {
            assert!(parse_spec(spec).is_err(), "{spec:?}");
        }
    }

    #[test]
    fn an_alias_takes_its_settings_from_the_config() {
        let (_directory, paths) = paths(
            "Host build\n  HostName build.example.com\n  User deploy\n  Port 2200\n  \
             IdentityFile /keys/build\n",
        );
        let target = resolve("build", &paths).unwrap();
        assert_eq!(target.label(), "deploy@build.example.com:2200");
        assert_eq!(target.identity_files, vec![PathBuf::from("/keys/build")]);
    }

    #[test]
    fn what_the_spec_says_wins_over_the_config() {
        let (_directory, paths) = paths("Host build\n  User deploy\n  Port 2200\n");
        let target = resolve("root@build:22", &paths).unwrap();
        assert_eq!((target.user.as_str(), target.port), ("root", 22));
    }

    /// Connecting directly would reach a different machine, or none.
    #[test]
    fn a_proxied_host_is_refused_rather_than_connected_to_directly() {
        for directive in ["ProxyJump bastion", "ProxyCommand nc %h %p"] {
            let (_directory, paths) = paths(&format!("Host inner\n  {directive}\n"));
            let error = resolve("inner", &paths).unwrap_err().to_string();
            assert!(error.contains("ProxyJump or ProxyCommand"), "{error}");
        }
    }

    #[test]
    fn config_the_parser_cannot_see_is_reported() {
        let (_directory, paths) = paths("Include conf.d/*\nHost *\n  StrictHostKeyChecking no\n");
        let notes = resolve("box", &paths).unwrap().notes.join("\n");
        assert!(notes.contains("Include"), "{notes}");
        assert!(notes.contains("always checked"), "{notes}");
    }

    /// As common as a config gets, and the parser alone refuses it.
    #[test]
    fn settings_before_the_first_host_apply_to_every_host() {
        let (_directory, paths) =
            paths("# mine\nUser everyone\nPort 2200\n\nHost build\n  HostName b.example.com\n");
        assert_eq!(
            resolve("build", &paths).unwrap().label(),
            "everyone@b.example.com:2200"
        );
        assert_eq!(
            resolve("other", &paths).unwrap().label(),
            "everyone@other:2200"
        );
    }

    #[test]
    fn tabs_and_equals_signs_separate_like_spaces() {
        let (_directory, paths) =
            paths("Host\tbuild\n\tUser\tdeploy\n  Port=2200\n  HostName = b\n");
        assert_eq!(resolve("build", &paths).unwrap().label(), "deploy@b:2200");
    }

    /// Its condition is not evaluated, so its settings must not reach a host
    /// through the block above it.
    #[test]
    fn a_match_block_does_not_leak_into_the_block_before_it() {
        let (_directory, paths) = paths(
            "Host build\n  User deploy\nMatch host elsewhere\n  Port 2200\n  \
             ProxyJump bastion\nHost other\n  User second\n",
        );
        assert_eq!(resolve("build", &paths).unwrap().label(), "deploy@build:22");
        assert_eq!(resolve("other", &paths).unwrap().label(), "second@other:22");
    }

    #[test]
    fn a_missing_config_is_not_an_error() {
        let (directory, paths) = paths("");
        std::fs::remove_file(directory.path().join("config")).unwrap();
        assert_eq!(resolve("me@box", &paths).unwrap().label(), "me@box:22");
    }
}
