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

//! Logins and copies against a real OpenSSH server, on a port of its own with
//! keys and files in a temporary directory: a server from the client's own
//! library would share what is under test. Skipped where OpenSSH is not
//! installed. Nothing here reads or writes the user's `~/.ssh`.

use super::*;
use crate::config::SshMode;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

const SSHD: &str = "/usr/sbin/sshd";

/// Answers prompts from a script and remembers what was asked.
#[derive(Default)]
struct Script {
    answers: Mutex<VecDeque<Option<String>>>,
    asked: Mutex<Vec<String>>,
}

impl Script {
    fn with(answers: &[&str]) -> Arc<Self> {
        Arc::new(Self {
            answers: Mutex::new(answers.iter().map(|a| Some(a.to_string())).collect()),
            asked: Mutex::default(),
        })
    }

    fn asked(&self) -> Vec<String> {
        self.asked.lock().unwrap().clone()
    }
}

impl Prompter for Script {
    fn ask(&self, prompt: String) -> tokio::sync::oneshot::Receiver<Option<String>> {
        self.asked.lock().unwrap().push(prompt);
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let _ = sender.send(self.answers.lock().unwrap().pop_front().flatten());
        receiver
    }
}

fn run(program: &str, args: &[&str]) {
    let output = Command::new(program)
        .args(args)
        .env_remove("SSH_AUTH_SOCK")
        .stdin(Stdio::null())
        .output()
        .unwrap_or_else(|error| panic!("{program}: {error}"));
    assert!(
        output.status.success(),
        "{program} {args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// A running `sshd` and everything around it.
struct Server {
    child: Child,
    directory: tempfile::TempDir,
    port: u16,
    user: String,
    workspace: tempfile::TempDir,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Server {
    /// `None` where OpenSSH is not installed.
    fn start() -> Option<Self> {
        if !Path::new(SSHD).exists() || Command::new("ssh-keygen").arg("-?").output().is_err() {
            eprintln!("skipped: OpenSSH is not installed");
            return None;
        }
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        std::fs::create_dir(root.join("remote")).unwrap();
        std::fs::create_dir(root.join("ssh")).unwrap();
        let host_key = root.join("host_key");
        Self::keygen(&host_key, "ed25519", "");
        Self::keygen(&root.join("ssh/id_ed25519"), "ed25519", "");
        std::fs::copy(
            root.join("ssh/id_ed25519.pub"),
            root.join("authorized_keys"),
        )
        .unwrap();

        let (child, port) = (0..5)
            .find_map(|_| Self::listen(root, &host_key))
            .expect("sshd did not start");
        let user = String::from_utf8(Command::new("id").arg("-un").output().unwrap().stdout)
            .unwrap()
            .trim()
            .to_string();
        let server = Self {
            child,
            directory,
            port,
            user,
            workspace: tempfile::tempdir().unwrap(),
        };
        server.trust_host_key();
        Some(server)
    }

    /// An `sshd` on a free port, heard to answer as one. `None` when it could
    /// not have the port: nothing holds it between finding it and `sshd`
    /// binding it, and another test may take it, which would answer a dial.
    fn listen(root: &Path, host_key: &Path) -> Option<(Child, u16)> {
        use std::io::Read;
        let port = std::net::TcpListener::bind(("127.0.0.1", 0))
            .unwrap()
            .local_addr()
            .unwrap()
            .port();
        let config = root.join("sshd_config");
        std::fs::write(
            &config,
            format!(
                "ListenAddress 127.0.0.1:{port}\nHostKey {host_key}\nPidFile none\nUsePAM no\n\
                 StrictModes no\nPasswordAuthentication no\nKbdInteractiveAuthentication no\n\
                 AuthorizedKeysFile {authorized}\n\
                 Subsystem sftp internal-sftp -d {remote}\n",
                host_key = host_key.display(),
                authorized = root.join("authorized_keys").display(),
                remote = root.join("remote").display(),
            ),
        )
        .unwrap();
        let mut child = Command::new(SSHD)
            .arg("-D")
            .arg("-e")
            .arg("-f")
            .arg(&config)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if child.try_wait().unwrap().is_some() {
                return None;
            }
            if let Ok(mut stream) = std::net::TcpStream::connect(("127.0.0.1", port)) {
                stream
                    .set_read_timeout(Some(Duration::from_secs(1)))
                    .unwrap();
                let mut banner = [0; 4];
                if stream.read_exact(&mut banner).is_ok() && &banner == b"SSH-" {
                    return Some((child, port));
                }
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        let _ = child.kill();
        let _ = child.wait();
        None
    }

    fn keygen(path: &Path, kind: &str, passphrase: &str) {
        let mut args = vec!["-q", "-t", kind, "-N", passphrase, "-f"];
        let path = path.to_str().unwrap();
        args.push(path);
        run("ssh-keygen", &args);
    }

    fn root(&self) -> &Path {
        self.directory.path()
    }

    fn remote(&self, name: &str) -> PathBuf {
        self.root().join("remote").join(name)
    }

    fn local(&self, name: &str) -> PathBuf {
        self.workspace.path().join(name)
    }

    fn spec(&self) -> String {
        format!("{}@127.0.0.1:{}", self.user, self.port)
    }

    fn known_hosts(&self) -> PathBuf {
        self.root().join("ssh/known_hosts")
    }

    /// `known_hosts` naming this server with `public_key_file`'s key.
    fn known_hosts_line(&self, public_key_file: &Path) -> String {
        let key = std::fs::read_to_string(public_key_file).unwrap();
        let mut fields = key.split_whitespace();
        let (kind, encoded) = (fields.next().unwrap(), fields.next().unwrap());
        format!("[127.0.0.1]:{} {kind} {encoded}\n", self.port)
    }

    fn trust_host_key(&self) {
        std::fs::write(
            self.known_hosts(),
            self.known_hosts_line(&self.root().join("host_key.pub")),
        )
        .unwrap();
    }

    fn forget_host_key(&self) {
        let _ = std::fs::remove_file(self.known_hosts());
    }

    fn paths(&self) -> SshPaths {
        SshPaths {
            config: self.root().join("ssh/config"),
            known_hosts: self.known_hosts(),
            directory: self.root().join("ssh"),
            agent_socket: None,
        }
    }

    /// Sessions that need no approval and have nobody to ask.
    fn unattended(&self) -> Arc<SshSessions> {
        SshSessions::new(SshMode::Auto, None, self.paths())
    }

    fn attended(&self, mode: SshMode, script: &Arc<Script>) -> Arc<SshSessions> {
        SshSessions::new(mode, Some(script.clone()), self.paths())
    }

    /// Replaces the default key with one of `kind` protected by `passphrase`.
    fn use_key(&self, kind: &str, passphrase: &str) {
        let ssh = self.root().join("ssh");
        std::fs::remove_file(ssh.join("id_ed25519")).unwrap();
        std::fs::remove_file(ssh.join("id_ed25519.pub")).unwrap();
        let key = ssh.join(format!("id_{kind}"));
        Self::keygen(&key, kind, passphrase);
        std::fs::copy(
            key.with_extension("pub"),
            self.root().join("authorized_keys"),
        )
        .unwrap();
    }

    async fn upload(
        &self,
        ssh: &SshSessions,
        local: &str,
        remote: &str,
        overwrite: bool,
    ) -> Result<String> {
        ssh.copy(
            &self.spec(),
            Direction::Upload,
            self.workspace.path(),
            local,
            remote,
            overwrite,
        )
        .await
    }

    async fn download(
        &self,
        ssh: &SshSessions,
        remote: &str,
        local: &str,
        overwrite: bool,
    ) -> Result<String> {
        ssh.copy(
            &self.spec(),
            Direction::Download,
            self.workspace.path(),
            local,
            remote,
            overwrite,
        )
        .await
    }
}

/// Files in `directory` a copy should not have left behind.
fn leftovers(directory: &Path) -> Vec<String> {
    std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().to_string())
        .filter(|name| name.contains("orangu-part"))
        .collect()
}

#[tokio::test]
async fn login_copy_both_ways_and_logout() {
    let Some(server) = Server::start() else {
        return;
    };
    let ssh = server.unattended();
    let report = ssh.login(&server.spec()).await.unwrap();
    assert!(report.contains("id_ed25519"), "{report}");
    assert!(report.contains("SHA256:"), "{report}");

    std::fs::write(server.local("a.txt"), "hello\n").unwrap();
    let report = server.upload(&ssh, "a.txt", "a.txt", false).await.unwrap();
    assert!(report.contains("6 bytes"), "{report}");
    assert_eq!(std::fs::read(server.remote("a.txt")).unwrap(), b"hello\n");

    std::fs::write(server.remote("b.txt"), "from afar\n").unwrap();
    server
        .download(&ssh, "b.txt", "sub/dir/b.txt", false)
        .await
        .unwrap();
    assert_eq!(
        std::fs::read(server.local("sub/dir/b.txt")).unwrap(),
        b"from afar\n"
    );

    let again = ssh.login(&server.spec()).await.unwrap();
    assert!(again.contains("Already logged in"), "{again}");

    let report = ssh.logout(&server.spec()).await.unwrap();
    assert!(report.contains("Logged out"), "{report}");
    let error = server
        .upload(&ssh, "a.txt", "c.txt", false)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("not open"), "{error}");
    assert!(!server.remote("c.txt").exists());
    assert!(ssh.logout(&server.spec()).await.is_err());
}

#[tokio::test]
async fn a_copy_lands_in_a_directory_under_its_own_name() {
    let Some(server) = Server::start() else {
        return;
    };
    let ssh = server.unattended();
    ssh.login(&server.spec()).await.unwrap();
    std::fs::create_dir(server.remote("in")).unwrap();
    std::fs::write(server.local("a.txt"), "up").unwrap();
    server.upload(&ssh, "a.txt", "in", false).await.unwrap();
    assert_eq!(std::fs::read(server.remote("in/a.txt")).unwrap(), b"up");

    std::fs::create_dir(server.local("here")).unwrap();
    server
        .download(&ssh, "in/a.txt", "here", false)
        .await
        .unwrap();
    assert_eq!(std::fs::read(server.local("here/a.txt")).unwrap(), b"up");
}

#[tokio::test]
async fn an_existing_file_is_replaced_only_when_asked() {
    let Some(server) = Server::start() else {
        return;
    };
    let ssh = server.unattended();
    ssh.login(&server.spec()).await.unwrap();
    std::fs::write(server.local("a.txt"), "new").unwrap();
    std::fs::write(server.remote("a.txt"), "old").unwrap();

    let error = server
        .upload(&ssh, "a.txt", "a.txt", false)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("overwrite"), "{error}");
    assert_eq!(std::fs::read(server.remote("a.txt")).unwrap(), b"old");
    server.upload(&ssh, "a.txt", "a.txt", true).await.unwrap();
    assert_eq!(std::fs::read(server.remote("a.txt")).unwrap(), b"new");

    std::fs::write(server.remote("a.txt"), "newer").unwrap();
    let error = server
        .download(&ssh, "a.txt", "a.txt", false)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("overwrite"), "{error}");
    assert_eq!(std::fs::read(server.local("a.txt")).unwrap(), b"new");
    server.download(&ssh, "a.txt", "a.txt", true).await.unwrap();
    assert_eq!(std::fs::read(server.local("a.txt")).unwrap(), b"newer");
}

/// A private file stays private on the far side, in both directions.
#[tokio::test]
async fn a_copy_keeps_the_file_mode() {
    use std::os::unix::fs::PermissionsExt;
    let Some(server) = Server::start() else {
        return;
    };
    let ssh = server.unattended();
    ssh.login(&server.spec()).await.unwrap();
    let mode = |path: &Path| std::fs::metadata(path).unwrap().permissions().mode() & 0o777;

    std::fs::write(server.local("secret"), "x").unwrap();
    std::fs::set_permissions(
        server.local("secret"),
        std::fs::Permissions::from_mode(0o600),
    )
    .unwrap();
    server
        .upload(&ssh, "secret", "secret", false)
        .await
        .unwrap();
    assert_eq!(mode(&server.remote("secret")), 0o600);

    std::fs::write(server.remote("tool"), "x").unwrap();
    std::fs::set_permissions(
        server.remote("tool"),
        std::fs::Permissions::from_mode(0o750),
    )
    .unwrap();
    server.download(&ssh, "tool", "tool", false).await.unwrap();
    assert_eq!(mode(&server.local("tool")), 0o750);
}

/// Larger than the copy buffer, the channel window and any SFTP packet, and
/// not a repeating pattern: a reordered or dropped chunk would show.
#[tokio::test]
async fn a_large_file_arrives_intact() {
    let Some(server) = Server::start() else {
        return;
    };
    let ssh = server.unattended();
    ssh.login(&server.spec()).await.unwrap();
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let data: Vec<u8> = (0..(40 * 1024 * 1024 / 8))
        .flat_map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state.to_le_bytes()
        })
        .collect();
    std::fs::write(server.local("big"), &data).unwrap();

    server.upload(&ssh, "big", "big", false).await.unwrap();
    assert!(std::fs::read(server.remote("big")).unwrap() == data);
    assert!(leftovers(&server.root().join("remote")).is_empty());

    server.download(&ssh, "big", "back", false).await.unwrap();
    assert!(std::fs::read(server.local("back")).unwrap() == data);
    assert!(leftovers(server.workspace.path()).is_empty());
}

/// Esc drops the turn, and the copy with it, wherever it was.
#[tokio::test]
async fn a_cancelled_copy_leaves_nothing_behind() {
    let Some(server) = Server::start() else {
        return;
    };
    let ssh = server.unattended();
    ssh.login(&server.spec()).await.unwrap();
    let data = vec![7u8; 128 * 1024 * 1024];
    std::fs::write(server.local("big"), &data).unwrap();
    std::fs::write(server.remote("big"), &data).unwrap();

    let upload = server.upload(&ssh, "big", "copy", false);
    assert!(
        tokio::time::timeout(Duration::from_millis(30), upload)
            .await
            .is_err()
    );
    let download = server.download(&ssh, "big", "copy", false);
    assert!(
        tokio::time::timeout(Duration::from_millis(30), download)
            .await
            .is_err()
    );

    let remote = server.root().join("remote");
    let deadline = Instant::now() + Duration::from_secs(5);
    while !leftovers(&remote).is_empty() {
        assert!(Instant::now() < deadline, "{:?}", leftovers(&remote));
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(leftovers(server.workspace.path()).is_empty());
    assert!(!server.remote("copy").exists());
    assert!(!server.local("copy").exists());
    // The session survives.
    std::fs::write(server.local("small"), "x").unwrap();
    server.upload(&ssh, "small", "small", false).await.unwrap();
}

#[tokio::test]
async fn local_paths_stay_in_the_workspace() {
    use std::os::unix::fs::symlink;
    let Some(server) = Server::start() else {
        return;
    };
    let ssh = server.unattended();
    ssh.login(&server.spec()).await.unwrap();
    let outside = server.root().join("outside");
    std::fs::write(&outside, "private").unwrap();
    std::fs::write(server.remote("a.txt"), "x").unwrap();
    symlink(&outside, server.local("link")).unwrap();

    for path in ["../outside", outside.to_str().unwrap(), "link"] {
        let error = server
            .upload(&ssh, path, "stolen", false)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("escapes"), "{path}: {error}");
    }
    assert!(!server.remote("stolen").exists());
    let error = server
        .download(&ssh, "a.txt", "../dropped", false)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("escapes"), "{error}");
    assert!(!server.root().join("dropped").exists());
    // Through a symlink as well, whether overwriting or not.
    let error = server
        .download(&ssh, "a.txt", "link", true)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("escapes"), "{error}");
    assert_eq!(std::fs::read(&outside).unwrap(), b"private");
}

#[tokio::test]
async fn directories_are_refused_both_ways() {
    let Some(server) = Server::start() else {
        return;
    };
    let ssh = server.unattended();
    ssh.login(&server.spec()).await.unwrap();
    std::fs::create_dir(server.local("dir")).unwrap();
    std::fs::create_dir(server.remote("dir")).unwrap();
    let error = server.upload(&ssh, "dir", "dir2", false).await.unwrap_err();
    assert!(error.to_string().contains("not a regular file"), "{error}");
    let error = server
        .download(&ssh, "dir", "dir3", false)
        .await
        .unwrap_err();
    assert!(error.to_string().contains("directory"), "{error}");
}

#[tokio::test]
async fn an_unknown_host_needs_a_person() {
    let Some(server) = Server::start() else {
        return;
    };
    server.forget_host_key();
    let error = server.unattended().login(&server.spec()).await.unwrap_err();
    assert!(error.to_string().contains("not in"), "{error}");
    assert!(!server.known_hosts().exists());

    let script = Script::with(&["no"]);
    let error = server
        .attended(SshMode::Auto, &script)
        .login(&server.spec())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("not trusted"), "{error}");
    assert!(!server.known_hosts().exists());
    let asked = script.asked().join("\n");
    assert!(
        asked.contains("SHA256:") && asked.contains("(yes/no)"),
        "{asked}"
    );

    let script = Script::with(&["yes"]);
    server
        .attended(SshMode::Auto, &script)
        .login(&server.spec())
        .await
        .unwrap();
    assert!(
        std::fs::read_to_string(server.known_hosts())
            .unwrap()
            .lines()
            .any(|line| line.starts_with(&format!("[127.0.0.1]:{} ssh-ed25519 ", server.port)))
    );

    // Now known, and asked about no more.
    let script = Script::with(&[]);
    server
        .attended(SshMode::Auto, &script)
        .login(&server.spec())
        .await
        .unwrap();
    assert!(script.asked().is_empty(), "{:?}", script.asked());
}

/// A changed key is never a question, or a yes could be talked out of anyone.
#[tokio::test]
async fn a_changed_host_key_is_refused_without_asking() {
    let Some(server) = Server::start() else {
        return;
    };
    let other = server.root().join("other");
    Server::keygen(&other, "ed25519", "");
    std::fs::write(
        server.known_hosts(),
        server.known_hosts_line(&other.with_extension("pub")),
    )
    .unwrap();
    let script = Script::with(&["yes", "yes", "yes"]);
    let ssh = server.attended(SshMode::Auto, &script);
    let error = ssh.login(&server.spec()).await.unwrap_err();
    assert!(error.to_string().contains("CHANGED"), "{error}");
    assert!(script.asked().is_empty(), "{:?}", script.asked());
}

#[tokio::test]
async fn a_revoked_host_key_is_refused_without_asking() {
    let Some(server) = Server::start() else {
        return;
    };
    let line = server.known_hosts_line(&server.root().join("host_key.pub"));
    std::fs::write(server.known_hosts(), format!("@revoked {line}")).unwrap();
    let script = Script::with(&["yes"]);
    let error = server
        .attended(SshMode::Auto, &script)
        .login(&server.spec())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("@revoked"), "{error}");
    assert!(script.asked().is_empty(), "{:?}", script.asked());
}

#[tokio::test]
async fn hashed_known_hosts_entries_are_recognised() {
    let Some(server) = Server::start() else {
        return;
    };
    run(
        "ssh-keygen",
        &["-q", "-H", "-f", server.known_hosts().to_str().unwrap()],
    );
    assert!(
        std::fs::read_to_string(server.known_hosts())
            .unwrap()
            .starts_with("|1|")
    );
    server.unattended().login(&server.spec()).await.unwrap();
}

#[tokio::test]
async fn approval_is_asked_once_and_before_connecting() {
    let Some(server) = Server::start() else {
        return;
    };
    // Nobody to ask.
    let ssh = SshSessions::new(SshMode::Prompt, None, server.paths());
    let error = ssh.login(&server.spec()).await.unwrap_err();
    assert!(error.to_string().contains("needs approval"), "{error}");

    // A refusal comes before any connection: this port has no server.
    let closed = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
    let port = closed.local_addr().unwrap().port();
    drop(closed);
    let script = Script::with(&["no"]);
    let error = server
        .attended(SshMode::Prompt, &script)
        .login(&format!("{}@127.0.0.1:{port}", server.user))
        .await
        .unwrap_err();
    assert!(error.to_string().contains("did not allow"), "{error}");

    let script = Script::with(&["yes"]);
    let ssh = server.attended(SshMode::Prompt, &script);
    ssh.login(&server.spec()).await.unwrap();
    ssh.logout(&server.spec()).await.unwrap();
    ssh.login(&server.spec()).await.unwrap();
    assert_eq!(script.asked().len(), 1, "{:?}", script.asked());
    assert!(
        script.asked()[0].contains(&server.spec()),
        "{:?}",
        script.asked()
    );
}

/// The passphrase goes to the key file and nowhere the model can see.
#[tokio::test]
async fn an_encrypted_key_asks_for_its_passphrase() {
    let Some(server) = Server::start() else {
        return;
    };
    server.use_key("ed25519", "correct horse");
    let script = Script::with(&["wrong", "correct horse"]);
    let ssh = server.attended(SshMode::Auto, &script);
    let report = ssh.login(&server.spec()).await.unwrap();
    assert!(!report.contains("correct horse"), "{report}");
    assert_eq!(script.asked().len(), 2, "{:?}", script.asked());
    assert!(
        script.asked()[0].starts_with("Passphrase for"),
        "{:?}",
        script.asked()
    );

    let script = Script::with(&["a", "b", "c"]);
    let error = server
        .attended(SshMode::Auto, &script)
        .login(&server.spec())
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("wrong passphrase"), "{error}");
    for secret in ["a\"", "b\"", "c\""] {
        assert!(!error.contains(secret), "{error}");
    }

    let error = server.unattended().login(&server.spec()).await.unwrap_err();
    assert!(error.to_string().contains("nobody is here"), "{error}");
}

/// The key type most existing setups have.
#[tokio::test]
async fn an_rsa_key_logs_in() {
    let Some(server) = Server::start() else {
        return;
    };
    server.use_key("rsa", "");
    let report = server.unattended().login(&server.spec()).await.unwrap();
    assert!(report.contains("id_rsa"), "{report}");
}

#[tokio::test]
async fn an_rsa_key_in_the_agent_logs_in() {
    agent_login("rsa").await;
}

#[tokio::test]
async fn an_ed25519_key_in_the_agent_logs_in() {
    agent_login("ed25519").await;
}

async fn agent_login(kind: &str) {
    let Some(server) = Server::start() else {
        return;
    };
    let socket = server.root().join("agent.sock");
    let mut agent = Command::new("ssh-agent")
        .arg("-D")
        .arg("-a")
        .arg(&socket)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while !socket.exists() {
        assert!(Instant::now() < deadline, "ssh-agent did not start");
        std::thread::sleep(Duration::from_millis(10));
    }
    server.use_key(kind, "");
    let key = server.root().join(format!("ssh/id_{kind}"));
    let added = Command::new("ssh-add")
        .arg("-q")
        .arg(&key)
        .env("SSH_AUTH_SOCK", &socket)
        .stdin(Stdio::null())
        .output()
        .unwrap();
    assert!(
        added.status.success(),
        "{}",
        String::from_utf8_lossy(&added.stderr)
    );
    // Only the agent can log in now.
    std::fs::remove_file(&key).unwrap();

    let mut paths = server.paths();
    paths.agent_socket = Some(socket);
    let result = SshSessions::new(SshMode::Auto, None, paths)
        .login(&server.spec())
        .await;
    let _ = agent.kill();
    let _ = agent.wait();
    let report = result.unwrap();
    assert!(report.contains("agent key"), "{report}");
}

#[tokio::test]
async fn a_failed_login_says_what_was_tried() {
    let Some(server) = Server::start() else {
        return;
    };
    std::fs::write(server.root().join("authorized_keys"), "").unwrap();
    let error = server
        .unattended()
        .login(&server.spec())
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("id_ed25519"), "{error}");
    assert!(error.contains("publickey"), "{error}");
}

/// OpenSSH hangs up on too many failed keys: that has to read as what it is.
#[tokio::test]
async fn a_server_hanging_up_mid_login_is_reported() {
    let Some(mut server) = Server::start() else {
        return;
    };
    let config = server.root().join("sshd_config");
    let text = std::fs::read_to_string(&config).unwrap();
    std::fs::write(&config, format!("MaxAuthTries 2\n{text}")).unwrap();
    let _ = server.child.kill();
    let _ = server.child.wait();
    server.child = Command::new(SSHD)
        .arg("-D")
        .arg("-e")
        .arg("-f")
        .arg(&config)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while std::net::TcpStream::connect(("127.0.0.1", server.port)).is_err() {
        assert!(Instant::now() < deadline, "sshd did not start");
        std::thread::sleep(Duration::from_millis(20));
    }
    // Keys the server does not know, more than it allows.
    std::fs::write(server.root().join("authorized_keys"), "").unwrap();
    for kind in ["ecdsa", "rsa"] {
        Server::keygen(&server.root().join(format!("ssh/id_{kind}")), kind, "");
    }
    let error = server
        .unattended()
        .login(&server.spec())
        .await
        .unwrap_err()
        .to_string();
    assert!(error.contains("could not log in"), "{error}");
    assert!(!error.contains("Channel send"), "{error}");
}

#[tokio::test]
async fn an_idle_session_closes_by_itself() {
    let Some(server) = Server::start() else {
        return;
    };
    let mut sessions = SshSessions::new(SshMode::Auto, None, server.paths());
    Arc::get_mut(&mut sessions).unwrap().idle_timeout = Duration::from_millis(200);
    sessions.login(&server.spec()).await.unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    while sessions.session(&server.spec()).is_ok() {
        assert!(Instant::now() < deadline, "still open");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    // Logging in again is all it takes.
    sessions.login(&server.spec()).await.unwrap();
}

/// A session with a copy running is not idle, however long the copy takes.
#[tokio::test]
async fn a_session_in_use_is_not_closed_as_idle() {
    let Some(server) = Server::start() else {
        return;
    };
    let mut sessions = SshSessions::new(SshMode::Auto, None, server.paths());
    Arc::get_mut(&mut sessions).unwrap().idle_timeout = Duration::from_secs(3600);
    sessions.login(&server.spec()).await.unwrap();
    let session = sessions.session(&server.spec()).unwrap();
    *session.last_used.lock().unwrap() = Instant::now() - Duration::from_secs(7200);
    sessions.close_idle().await;
    assert!(sessions.session(&server.spec()).is_ok());
    drop(session);
    sessions.close_idle().await;
    assert!(sessions.session(&server.spec()).is_err());
}

/// The whole path from the parent's tool call: a scripted model logs in and
/// copies through the subagent's tools.
#[tokio::test]
async fn the_subagent_logs_in_and_copies() {
    use crate::test_support::{serve_chat_responses, text_stream};
    use serde_json::json;
    let Some(server) = Server::start() else {
        return;
    };
    std::fs::write(server.local("a.txt"), "payload").unwrap();

    let call = |name: &str, arguments: serde_json::Value| {
        let delta = json!({"tool_calls": [{
            "index": 0, "id": "call0", "type": "function",
            "function": {"name": name, "arguments": arguments.to_string()}
        }]});
        format!(
            "data: {}\n\ndata: [DONE]\n\n",
            json!({"choices": [{"delta": delta, "finish_reason": "tool_calls"}]})
        )
    };
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}/v1", listener.local_addr().unwrap());
    let served = serve_chat_responses(
        listener,
        vec![
            call("ssh_login", json!({"host": server.spec()})),
            call(
                "ssh_copy",
                json!({"session": server.spec(), "direction": "upload",
                       "local": "a.txt", "remote": "a.txt"}),
            ),
            // Not among its tools, whatever the model asks for.
            call(
                "run_shell_command",
                json!({"command": "echo hacked > hacked"}),
            ),
            text_stream("<final_answer>copied a.txt</final_answer>"),
        ],
    );

    let mut conf = tempfile::NamedTempFile::new().unwrap();
    std::io::Write::write_all(
        &mut conf,
        format!("[orangu]\nserver = main\n\n[main]\nendpoint = {endpoint}\nmodel = m\n").as_bytes(),
    )
    .unwrap();
    let config = crate::config::load_client_configuration(conf.path()).unwrap();
    let parent = crate::tools::ToolExecutor::new(server.workspace.path())
        .with_ssh(Some(server.unattended()));
    parent.set_active_profile(&config.llms["main"]);

    let mut arguments = serde_json::Map::new();
    arguments.insert("task".into(), json!("copy a.txt"));
    let report = parent.execute("use_ssh", &arguments).await.unwrap();
    assert!(report.contains("copied a.txt"), "{report}");
    let requests = served.join().unwrap();
    assert_eq!(
        std::fs::read(server.remote("a.txt")).ok().as_deref(),
        Some(&b"payload"[..]),
        "what ssh_copy told the model: {}",
        requests[2]
    );
    assert!(!server.local("hacked").exists());

    let offered: serde_json::Value = serde_json::from_str(&requests[0]).unwrap();
    let mut names: Vec<&str> = offered["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["function"]["name"].as_str().unwrap())
        .collect();
    names.sort_unstable();
    assert_eq!(
        names,
        ["list_directory", "ssh_copy", "ssh_login", "ssh_logout"]
    );
    assert!(
        requests[3].contains("not available to this agent"),
        "{}",
        requests[3]
    );

    // The session outlives the subagent, for the next task.
    assert!(parent.ssh().unwrap().session(&server.spec()).is_ok());
}

/// `use_ssh` for the conversation; the tools themselves only for its subagent.
#[test]
fn the_ssh_tools_are_offered_only_where_they_belong() {
    use crate::tools::ToolExecutor;
    let workspace = tempfile::tempdir().unwrap();
    let names = |executor: &ToolExecutor| -> Vec<String> {
        executor
            .definitions()
            .into_iter()
            .map(|def| def.function.name)
            .collect()
    };
    let ssh = || SshSessions::for_mode(SshMode::Prompt, None);

    let without = names(&ToolExecutor::new(workspace.path()));
    assert!(
        !without.iter().any(|name| name.contains("ssh")),
        "{without:?}"
    );

    let off = SshSessions::for_mode(SshMode::Deny, None);
    assert!(off.is_none());

    let main = names(&ToolExecutor::new(workspace.path()).with_ssh(ssh()));
    assert!(main.contains(&"use_ssh".to_string()));
    assert!(
        !main.iter().any(|name| name.starts_with("ssh_")),
        "{main:?}"
    );

    let read_only = names(&ToolExecutor::new_read_only(workspace.path()).with_ssh_tools(ssh()));
    assert!(
        !read_only.iter().any(|name| name.contains("ssh")),
        "{read_only:?}"
    );
}

#[tokio::test]
async fn the_main_agent_cannot_call_the_ssh_tools_itself() {
    use crate::tools::ToolExecutor;
    let workspace = tempfile::tempdir().unwrap();
    let main =
        ToolExecutor::new(workspace.path()).with_ssh(SshSessions::for_mode(SshMode::Auto, None));
    let mut arguments = serde_json::Map::new();
    arguments.insert("host".into(), "box".into());
    let error = main.execute("ssh_login", &arguments).await.unwrap_err();
    assert!(error.to_string().contains("only available"), "{error}");
}

/// No schema has room for a secret, so none can pass through the model.
#[test]
fn no_ssh_tool_takes_a_credential() {
    let workspace = tempfile::tempdir().unwrap();
    let tools = crate::tools::ToolExecutor::new_for_subagent(workspace.path())
        .with_ssh_tools(SshSessions::for_mode(SshMode::Prompt, None));
    let definitions: Vec<String> = tools
        .definitions()
        .into_iter()
        .filter(|def| def.function.name.starts_with("ssh_"))
        .map(|def| def.function.parameters.to_string().to_lowercase())
        .collect();
    assert_eq!(definitions.len(), 3);
    for schema in definitions {
        for word in ["password", "passphrase", "secret", "identity", "key"] {
            assert!(!schema.contains(word), "{schema}");
        }
    }
}

/// A server of the client's own library, since an unprivileged `sshd` cannot
/// check a password: this tests the prompt and the order of methods, not
/// compatibility with one.
mod password {
    use super::super::auth;
    use super::*;
    use russh::server::{self, Auth};

    struct PasswordOnly;

    impl server::Handler for PasswordOnly {
        type Error = russh::Error;

        async fn auth_password(
            &mut self,
            _user: &str,
            password: &str,
        ) -> std::result::Result<Auth, Self::Error> {
            Ok(if password == "open sesame" {
                Auth::Accept
            } else if password == "last try" {
                Auth::reject()
            } else {
                // What OpenSSH does: password stays on offer.
                Auth::Reject {
                    proceed_with_methods: Some((&[russh::MethodKind::Password][..]).into()),
                    partial_success: false,
                }
            })
        }
    }

    async fn attempt(answers: &[&str]) -> (Result<String>, Vec<String>) {
        let directory = tempfile::tempdir().unwrap();
        let listener = tokio::net::TcpListener::bind(("127.0.0.1", 0))
            .await
            .unwrap();
        let port = listener.local_addr().unwrap().port();
        let config = Arc::new(server::Config {
            methods: (&[russh::MethodKind::Password][..]).into(),
            auth_rejection_time: Duration::ZERO,
            auth_rejection_time_initial: Some(Duration::ZERO),
            keys: vec![
                keys::PrivateKey::random(&mut rand::rng(), keys::Algorithm::Ed25519).unwrap(),
            ],
            ..server::Config::default()
        });
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            if let Ok(session) = server::run_stream(config, stream, PasswordOnly).await {
                let _ = session.await;
            }
        });

        let paths = SshPaths {
            config: directory.path().join("config"),
            known_hosts: directory.path().join("known_hosts"),
            directory: directory.path().to_path_buf(),
            agent_socket: None,
        };
        let target = host::resolve(&format!("me@127.0.0.1:{port}"), &paths).unwrap();
        let script = Script::with(answers);
        let prompts = Prompts {
            prompter: Some(script.clone()),
            timeout: PROMPT_TIMEOUT,
        };
        let check = HostKeyCheck {
            host: target.host.clone(),
            port,
            known_hosts: target.known_hosts.clone(),
            prompts: Prompts {
                prompter: Some(Script::with(&["yes"])),
                timeout: PROMPT_TIMEOUT,
            },
            refusal: Arc::default(),
            fingerprint: Arc::default(),
        };
        let stream = tokio::net::TcpStream::connect(("127.0.0.1", port))
            .await
            .unwrap();
        let mut handle = client::connect_stream(Arc::new(client::Config::default()), stream, check)
            .await
            .unwrap();
        let result = auth::authenticate(&mut handle, &target, &paths, &prompts).await;
        (result, script.asked())
    }

    #[tokio::test]
    async fn a_password_is_asked_for_and_retried() {
        let (result, asked) = attempt(&["wrong", "open sesame"]).await;
        assert_eq!(result.unwrap(), "a password");
        assert_eq!(asked.len(), 2, "{asked:?}");
        assert!(
            asked[0].starts_with("Password for me@127.0.0.1"),
            "{asked:?}"
        );
    }

    #[tokio::test]
    async fn wrong_passwords_end_the_login_and_stay_out_of_the_error() {
        let (result, asked) = attempt(&["hunter1", "hunter2", "hunter3"]).await;
        let error = result.unwrap_err().to_string();
        assert_eq!(asked.len(), 3, "{asked:?}");
        assert!(!error.contains("hunter"), "{error}");
    }

    /// No further prompt once the server stops offering passwords.
    #[tokio::test]
    async fn a_password_is_not_asked_for_again_once_it_is_off_the_table() {
        let (result, asked) = attempt(&["last try", "unused"]).await;
        assert!(result.is_err());
        assert_eq!(asked.len(), 1, "{asked:?}");
    }

    #[tokio::test]
    async fn a_cancelled_password_prompt_ends_the_login() {
        let (result, asked) = attempt(&[]).await;
        assert!(result.is_err());
        assert_eq!(asked.len(), 1, "{asked:?}");
    }
}

#[test]
fn a_prompt_nobody_answers_counts_as_cancelled() {
    struct Silent;
    impl Prompter for Silent {
        fn ask(&self, _prompt: String) -> tokio::sync::oneshot::Receiver<Option<String>> {
            let (sender, receiver) = tokio::sync::oneshot::channel();
            std::mem::forget(sender);
            receiver
        }
    }
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let prompts = Prompts {
        prompter: Some(Arc::new(Silent)),
        timeout: Duration::from_millis(20),
    };
    assert_eq!(runtime.block_on(prompts.ask("x".to_string())), None);
    assert!(!runtime.block_on(prompts.confirm("x".to_string())));
}
