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
        let (keyword, value) = keyword_and_value(line);
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

/// A trimmed config line split into its keyword and value, which spaces, tabs
/// or an `=` separate.
fn keyword_and_value(line: &str) -> (&str, &str) {
    let (keyword, value) = line
        .split_once(|c: char| c.is_whitespace() || c == '=')
        .unwrap_or((line, ""));
    (
        keyword,
        value.trim_start_matches(|c: char| c.is_whitespace() || c == '='),
    )
}

/// Whether the config uses `Match`, which the SSH config parser skips.
fn uses_match(contents: &str) -> bool {
    contents.lines().any(|line| {
        keyword_and_value(line.trim())
            .0
            .eq_ignore_ascii_case("match")
    })
}

/// How deep `Include` may nest, as in `ssh`.
const MAX_INCLUDE_DEPTH: usize = 16;

/// The block a config line sits in.
#[derive(Clone, Copy)]
enum Block {
    /// Before the first `Host` or `Match`: every host.
    None,
    Host,
    Match,
}

/// `contents` with each `Include` replaced by the files it names, read the way
/// `ssh` reads them. What cannot be read that way goes to `notes`.
fn with_includes(contents: &str, directory: &Path, notes: &mut Vec<String>) -> String {
    let mut expanded = String::new();
    expand(contents, directory, Block::None, 0, &mut expanded, notes);
    expanded
}

/// Appends `contents` to `out`, its includes expanded, starting in `outer`.
/// Whether it has `Host` or `Match` lines of its own.
fn expand(
    contents: &str,
    directory: &Path,
    outer: Block,
    depth: usize,
    out: &mut String,
    notes: &mut Vec<String>,
) -> bool {
    let mut block = outer;
    let mut has_blocks = false;
    for line in contents.lines() {
        let (keyword, value) = keyword_and_value(line.trim());
        if keyword.eq_ignore_ascii_case("include") {
            include(value, directory, block, depth, out, notes);
            continue;
        }
        if keyword.eq_ignore_ascii_case("host") {
            block = Block::Host;
            has_blocks = true;
        } else if keyword.eq_ignore_ascii_case("match") {
            block = Block::Match;
            has_blocks = true;
        }
        out.push_str(line);
        out.push('\n');
    }
    has_blocks
}

/// Appends the files an `Include` in `block` names to `out`.
fn include(
    value: &str,
    directory: &Path,
    block: Block,
    depth: usize,
    out: &mut String,
    notes: &mut Vec<String>,
) {
    // Its condition is not evaluated, so neither its settings nor its
    // includes reach any host.
    if matches!(block, Block::Match) {
        return;
    }
    if depth >= MAX_INCLUDE_DEPTH {
        notes.push(format!(
            "Include nests more than {MAX_INCLUDE_DEPTH} deep at '{value}', which is not read."
        ));
        return;
    }
    for path in include_paths(value, directory, notes) {
        let contents = match std::fs::read_to_string(&path) {
            Ok(contents) => contents,
            Err(error) => {
                notes.push(format!("{} could not be read: {error}.", path.display()));
                continue;
            }
        };
        // Inside a `Host` block, `ssh` reads the file only for hosts the
        // block matches, its own blocks included; that is not evaluated here.
        if matches!(block, Block::Host)
            && contents.lines().any(|line| {
                let keyword = keyword_and_value(line.trim()).0;
                keyword.eq_ignore_ascii_case("host") || keyword.eq_ignore_ascii_case("match")
            })
        {
            notes.push(format!(
                "{} is included inside a Host block and has blocks of its own, which is not \
                 read here, so this host may resolve differently than with ssh.",
                path.display()
            ));
            continue;
        }
        // Only a file included before the first block can have blocks of its
        // own; the lines after the Include apply to every host again.
        if expand(&contents, directory, block, depth + 1, out, notes) {
            out.push_str("Host *\n");
        }
    }
}

/// The files an `Include` names: a relative path is under `directory`, `~/`
/// is the home directory, and a file name may hold `*`, `?` or `[...]`, whose
/// matches are read in order. A missing file is skipped, as `ssh` does.
fn include_paths(value: &str, directory: &Path, notes: &mut Vec<String>) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for word in value.split_whitespace().map(|word| word.trim_matches('"')) {
        let path = match word.strip_prefix("~/") {
            Some(rest) => home::home_dir().unwrap_or_default().join(rest),
            None => directory.join(word),
        };
        let is_pattern = |text: &str| text.contains(['*', '?', '[']);
        let (Some(parent), Some(name)) = (path.parent(), path.file_name().and_then(|n| n.to_str()))
        else {
            continue;
        };
        if !is_pattern(name) {
            if path.is_file() {
                paths.push(path);
            }
            continue;
        }
        if is_pattern(&parent.to_string_lossy()) {
            notes.push(format!(
                "Include {word} has a wildcard before the file name, which is not read."
            ));
            continue;
        }
        let Ok(glob) = globset::Glob::new(name) else {
            continue;
        };
        let matcher = glob.compile_matcher();
        let Ok(entries) = std::fs::read_dir(parent) else {
            continue;
        };
        let mut found: Vec<PathBuf> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|found| {
                found
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|found_name| {
                        // As with a shell glob, `*` does not match a hidden file.
                        matcher.is_match(found_name)
                            && (!found_name.starts_with('.') || name.starts_with('.'))
                    })
            })
            .filter(|found| found.is_file())
            .collect();
        found.sort();
        paths.extend(found);
    }
    paths
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
    let mut notes = Vec::new();
    let contents = with_includes(&contents, &paths.directory, &mut notes);
    let config = russh_config::parse(&normalized_config(&contents), host)
        .with_context(|| format!("parsing {}", paths.config.display()))?;
    if config.host_config.proxy_jump.is_some() || config.host_config.proxy_command.is_some() {
        return Err(anyhow!(
            "{host} is reached through ProxyJump or ProxyCommand in {}, which ssh_login does not \
             support; it never connects to such a host directly",
            paths.config.display()
        ));
    }
    if uses_match(&contents) {
        notes.push(format!(
            "{} or a file it includes uses Match, which is not read here, so this host may \
             resolve differently than with ssh.",
            paths.config.display()
        ));
    }
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
        let (_directory, paths) =
            paths("Match host x\n  User y\nHost *\n  StrictHostKeyChecking no\n");
        let notes = resolve("box", &paths).unwrap().notes.join("\n");
        assert!(notes.contains("Match"), "{notes}");
        assert!(notes.contains("always checked"), "{notes}");
    }

    /// A file next to the config, under `~/.ssh` in real use.
    fn write(directory: &tempfile::TempDir, name: &str, contents: &str) {
        let path = directory.path().join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, contents).unwrap();
    }

    #[test]
    fn included_files_are_read_in_order_and_the_first_value_wins() {
        let (directory, paths) = paths("Include conf.d/*\nHost build\n  User late\n  Port 2200\n");
        write(
            &directory,
            "conf.d/b",
            "Host build\n  User second\n  HostName b.example.com\n",
        );
        write(&directory, "conf.d/a", "Host build\n  User first\n");
        write(&directory, "conf.d/.hidden", "Host build\n  User hidden\n");
        let target = resolve("build", &paths).unwrap();
        assert_eq!(target.label(), "first@b.example.com:2200");
        assert!(target.notes.is_empty(), "{:?}", target.notes);
    }

    #[test]
    fn lines_after_an_include_apply_to_every_host_again() {
        let (directory, paths) = paths("Include extra\nUser everyone\n");
        write(&directory, "extra", "Host build\n  User deploy\n");
        assert_eq!(resolve("build", &paths).unwrap().label(), "deploy@build:22");
        assert_eq!(
            resolve("other", &paths).unwrap().label(),
            "everyone@other:22"
        );
    }

    #[test]
    fn an_include_in_a_host_block_applies_to_that_host() {
        let (directory, paths) = paths("Host build\n  Include build.conf\nHost *\n  User rest\n");
        write(&directory, "build.conf", "User deploy\nPort 2200\n");
        assert_eq!(
            resolve("build", &paths).unwrap().label(),
            "deploy@build:2200"
        );
        assert_eq!(resolve("other", &paths).unwrap().label(), "rest@other:22");
    }

    /// `ssh` reads it only for the hosts the block matches, blocks and all.
    #[test]
    fn a_file_with_blocks_included_in_a_host_block_is_reported_not_read() {
        let (directory, paths) = paths("Host build\n  Include nested\n");
        write(&directory, "nested", "Host *\n  User leaked\n");
        let target = resolve("other", &paths).unwrap();
        assert_ne!(target.user, "leaked");
        assert!(target.notes.join("\n").contains("inside a Host block"));
    }

    #[test]
    fn an_include_in_a_match_block_is_not_read() {
        let (directory, paths) = paths("Match host elsewhere\n  Include extra\n");
        write(&directory, "extra", "Host *\n  User leaked\n");
        assert_ne!(resolve("box", &paths).unwrap().user, "leaked");
    }

    #[test]
    fn a_missing_include_is_skipped_quietly() {
        let (_directory, paths) = paths("Include missing other/*\nUser deploy\n");
        let target = resolve("box", &paths).unwrap();
        assert_eq!(target.label(), "deploy@box:22");
        assert!(target.notes.is_empty(), "{:?}", target.notes);
    }

    #[test]
    fn an_include_that_includes_itself_stops() {
        let (directory, paths) = paths("Include loop\n");
        write(&directory, "loop", "Include loop\nUser deploy\n");
        let target = resolve("box", &paths).unwrap();
        assert_eq!(target.user, "deploy");
        assert!(target.notes.join("\n").contains("nests more than"));
    }

    #[test]
    fn an_absolute_include_and_equals_signs_are_read() {
        let (directory, paths) = paths("");
        write(&directory, "elsewhere/conf", "Host build\n  User deploy\n");
        let absolute = directory.path().join("elsewhere/conf");
        std::fs::write(&paths.config, format!("Include={}\n", absolute.display())).unwrap();
        assert_eq!(resolve("build", &paths).unwrap().label(), "deploy@build:22");
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
