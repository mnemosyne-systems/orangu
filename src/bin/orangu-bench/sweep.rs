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

//! Restarting the server once per value of one tuning variable, and measuring
//! each — the mechanical half of tuning a knob that cannot be changed without
//! a restart.
//!
//! # Why a restart, and why that is the whole difficulty
//!
//! Nearly every `ORANGU_*` knob is read once, in `VulkanBackend::try_init`,
//! and most of them are then *baked into generated WGSL* — a workgroup width,
//! a tile geometry, a register-block shape. There is no runtime setter that
//! could exist for those; a different value is a different compiled kernel. So
//! a sweep is: start a server, measure it, stop it, repeat. That is a
//! twenty-line shell script, and the twenty-line shell script is how this has
//! gone wrong every time.
//!
//! The failure is specific and it does not look like a failure. Stop the old
//! server by process name and a build copied under a different filename
//! survives; the next server then cannot bind the port and exits; the
//! benchmark measures the *survivor* and reports every configuration as
//! identical — which reads as a credible "this knob does nothing" result. A
//! sweep is exactly the shape that hides it, because the same server serving
//! every point produces beautifully consistent numbers.
//!
//! So this module refuses to measure anything it has not proved it started:
//!
//! - the port must be **free before** a child is launched, or the run stops;
//! - the pid the server reports through `/props` must be the pid of **this
//!   process's own child**, or the run stops;
//! - the child is killed and reaped through a [`Server`] guard whose `Drop`
//!   runs on every path, including the error one, and the port is waited back
//!   to free before the next point starts.
//!
//! None of that is defensive programming. Each check corresponds to a way one
//! of these sweeps has previously produced a confident wrong answer.
//!
//! # One axis
//!
//! `--sweep VAR=a,b,c` sweeps one variable. Crossing two would be a cartesian
//! product whose result is not readable as a table and whose cost is the
//! product of two model loads, and two axes that genuinely interact (the five
//! fields of `ORANGU_COOP_GEOM`, say) are already one variable as far as this
//! is concerned — they travel in one string. `--sweep-env` holds anything else
//! constant across every point, which is what makes a sweep of one knob a
//! sweep of one knob.

use std::process::Child;
// Only the POSIX launch path (and the test that pins it) spawns anything —
// see `start`'s own `cfg`.
#[cfg(unix)]
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

/// A `VAR=v1,v2,v3` sweep specification.
pub struct Spec {
    pub var: String,
    pub values: Vec<String>,
}

impl Spec {
    /// `ORANGU_COOP_MIN_TOKENS=8,16,24` → the variable and its three values.
    ///
    /// An empty value is kept deliberately: `VAR=,1` sweeps "unset" against
    /// "set to 1", which is the shape of every opt-in flag in the engine and
    /// would otherwise need a second mechanism.
    pub fn parse(spec: &str) -> anyhow::Result<Self> {
        let (var, values) = spec
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("--sweep wants VAR=v1,v2,..., got {spec:?}"))?;
        let var = var.trim();
        if var.is_empty() {
            anyhow::bail!("--sweep {spec:?} has no variable name");
        }
        let values: Vec<String> = values.split(',').map(|v| v.trim().to_string()).collect();
        if values.is_empty() {
            anyhow::bail!("--sweep {spec:?} has no values");
        }
        Ok(Spec {
            var: var.to_string(),
            values,
        })
    }

    /// The series name for one point, and the stem of its bundle. `VAR=value`,
    /// with an empty value spelled out rather than left blank — a legend entry
    /// reading `ORANGU_SUBGROUP=` is indistinguishable from a truncated one.
    pub fn label(&self, value: &str) -> String {
        if value.is_empty() {
            format!("{}=<unset>", self.var)
        } else {
            format!("{}={value}", self.var)
        }
    }
}

/// A server this process started, killed on drop.
///
/// `Drop` rather than an explicit `stop()` because the interesting path is the
/// one where the measurement fails: a sweep that leaves an orphaned server
/// holding the port makes every *subsequent* point of the same sweep measure
/// the orphan, converting one failed point into a whole run of quietly wrong
/// ones.
#[derive(Debug)]
pub struct Server {
    child: Child,
    port: u16,
    log: std::path::PathBuf,
}

impl Server {
    /// The pid of the process this struct owns — what
    /// [`Server::wait_until_serving`] checks `/props` against.
    pub fn pid(&self) -> u32 {
        self.child.id()
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        // Killing the child is not the same as the port being free: the
        // socket outlives the process briefly, and a next server that fails
        // to bind exits, leaving the *previous* one measured. Wait for the
        // release rather than assuming it.
        if !wait_for(Duration::from_secs(30), || !port_is_busy(self.port)) {
            eprintln!(
                "orangu-bench: port {} still busy after stopping the server (log {})",
                self.port,
                self.log.display()
            );
        }
    }
}

/// Start a server with `var=value` in its environment and wait until it
/// answers on `port`.
///
/// `cmd` runs through the shell so the caller can pass a full command line
/// with its own arguments — the same latitude `--flamegraph`'s `perf` handling
/// takes, and necessary because the way to launch a server is
/// installation-specific in a way this tool has no business modelling.
///
/// It is run as `sh -c "exec <cmd>"`, and the `exec` is load-bearing twice
/// over: it makes the shell *become* the server rather than fork it, so the
/// pid this struct holds is the server's — which is both the pid `kill` has to
/// reach on teardown and the one [`wait_until_serving`] checks `/props`
/// against. Without it a shell that decided to fork would leave the real
/// server unkilled and the identity check comparing two unrelated pids. A
/// `cmd` exotic enough that `exec` cannot apply (a pipeline, say) fails that
/// check and stops the run, which is the right outcome: this cannot supervise
/// what it cannot identify.
pub fn start(
    cmd: &str,
    env: &[(String, String)],
    port: u16,
    log: &std::path::Path,
    timeout: Duration,
) -> anyhow::Result<Server> {
    // Before anything is spawned. A port already in use means either a server
    // left over from a previous run or one somebody else is using; measuring
    // through it would attribute its numbers to this sweep's configuration.
    if port_is_busy(port) {
        anyhow::bail!(
            "port {port} is already in use before this sweep started a server — stop whatever \
             owns it, or the sweep would measure that process and attribute its numbers to \
             every configuration in turn"
        );
    }
    // `sh -c "exec …"` is the whole supervision mechanism: it is what makes
    // the spawned pid the *server's* pid, which is what the teardown kills
    // and what `wait_until_serving` checks `/props` against. Windows has no
    // `exec`, so `cmd /C` would leave this holding a shell's pid — the pid
    // check would then fail on every point of every sweep, or worse, pass
    // against a shell whose child outlives it. Refuse clearly instead of
    // failing obscurely; nobody is running GPU tuning sweeps there, and a
    // half-working supervisor is worse than an absent one.
    // `Err(..)` as the block's tail expression, not `bail!`: on a non-unix
    // build this block is the function's *only* body after cfg-stripping, so
    // it has to evaluate to the return type rather than diverge out of a
    // statement position.
    #[cfg(not(unix))]
    {
        let _ = (cmd, env, log, timeout);
        Err(anyhow::anyhow!(
            "--sweep needs a POSIX shell: it launches each server through `sh -c \"exec …\"` so \
             that the process it supervises is the server itself and not a shell wrapping it"
        ))
    }
    #[cfg(unix)]
    {
        let out =
            std::fs::File::create(log).map_err(|e| anyhow::anyhow!("{}: {e}", log.display()))?;
        let err = out.try_clone()?;
        let mut command = Command::new("sh");
        command
            .arg("-c")
            .arg(format!("exec {cmd}"))
            .stdin(Stdio::null())
            .stdout(Stdio::from(out))
            .stderr(Stdio::from(err));
        for (k, v) in env {
            // An empty value means *unset*, not "set to the empty string": that is
            // how every opt-in flag in the engine is spelled (`is_some()`), so
            // `VAR=,1` has to actually turn the flag off for the first point.
            if v.is_empty() {
                command.env_remove(k);
            } else {
                command.env(k, v);
            }
        }
        let child = command
            .spawn()
            .map_err(|e| anyhow::anyhow!("could not start the server ({cmd:?}): {e}"))?;
        let server = Server {
            child,
            port,
            log: log.to_path_buf(),
        };
        // From here on the guard owns the child, so every error path below stops
        // it rather than leaking it into the next point.
        if !wait_for(timeout, || port_is_busy(port)) {
            anyhow::bail!(
                "the server did not start listening on port {port} within {}s — see {}",
                timeout.as_secs(),
                log.display()
            );
        }
        Ok(server)
    }
}

/// Wait until the server answers `/health`, then prove it is **this** server.
///
/// The identity check is the point. `/props` reporting a pid that is not the
/// child this process spawned means the request was answered by something
/// else that owns the port — the exact accident that makes a sweep report
/// every configuration as identical, and one that produces no error of its own.
pub fn wait_until_serving(
    client: &reqwest::blocking::Client,
    url: &str,
    server: &Server,
    timeout: Duration,
) -> anyhow::Result<()> {
    let healthy = wait_for(timeout, || {
        client
            .get(format!("{url}/health"))
            .timeout(Duration::from_secs(2))
            .send()
            .is_ok_and(|r| r.status().is_success())
    });
    if !healthy {
        anyhow::bail!(
            "the server bound the port but never answered {url}/health within {}s — see {}",
            timeout.as_secs(),
            server.log.display()
        );
    }
    let reported = client
        .get(format!("{url}/props"))
        .send()
        .ok()
        .and_then(|r| r.json::<serde_json::Value>().ok())
        .and_then(|p| p.get("pid").and_then(serde_json::Value::as_u64))
        .map(|p| p as u32);
    match reported {
        Some(pid) if pid == server.pid() => Ok(()),
        Some(pid) => anyhow::bail!(
            "{url} is answered by pid {pid}, but this sweep started pid {} — something else \
             owns the port, and every point of this sweep would have measured it",
            server.pid()
        ),
        // A third-party server may report no pid. Sweeping `ORANGU_*` against it
        // is meaningless anyway, so this is a mistake worth naming rather
        // than a case to support.
        None => anyhow::bail!(
            "{url} did not report a pid, so this sweep cannot prove it is measuring the server \
             it started — --sweep needs orangu-server"
        ),
    }
}

/// Poll `cond` until it holds or `timeout` elapses.
fn wait_for(timeout: Duration, mut cond: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + timeout;
    loop {
        if cond() {
            return true;
        }
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// Whether anything is listening on `port`.
///
/// By connecting, not by reading `/proc`: this has to work on macOS, which is
/// where the sweeps this exists for are going to be run, and a connect is the
/// one probe that means the same thing on both. It is also the *right*
/// question — "can a client reach a server here" is what the next point cares
/// about, and a process holding the socket without accepting is still a
/// process that will make the next bind fail.
fn port_is_busy(port: u16) -> bool {
    std::net::TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([127, 0, 0, 1], port)),
        Duration::from_millis(200),
    )
    .is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_spec_splits_into_a_variable_and_its_values() {
        let s = Spec::parse("ORANGU_COOP_MIN_TOKENS=8,16,24").unwrap();
        assert_eq!(s.var, "ORANGU_COOP_MIN_TOKENS");
        assert_eq!(s.values, ["8", "16", "24"]);
        assert_eq!(s.label("16"), "ORANGU_COOP_MIN_TOKENS=16");
    }

    /// An empty value is a real point — "unset" against "set" is how every
    /// opt-in flag in the engine is A/B'd — so it must survive parsing and be
    /// legible in a legend rather than rendering as a blank.
    #[test]
    fn an_empty_value_is_a_point_meaning_unset() {
        let s = Spec::parse("ORANGU_SUBGROUP=,1").unwrap();
        assert_eq!(s.values, ["", "1"]);
        assert_eq!(s.label(""), "ORANGU_SUBGROUP=<unset>");
        assert_eq!(s.label("1"), "ORANGU_SUBGROUP=1");
    }

    #[test]
    fn a_spec_without_a_variable_or_an_equals_is_refused() {
        for bad in ["ORANGU_COOP_MIN_TOKENS", "=8,16", "  =1"] {
            assert!(Spec::parse(bad).is_err(), "{bad:?} should be refused");
        }
    }

    /// The pre-flight check has to actually detect a listener, or it is
    /// decoration — and it is the check that stops a sweep from measuring a
    /// server it did not start.
    #[test]
    fn a_busy_port_is_detected_and_a_free_one_is_not() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(port_is_busy(port), "a bound port must read as busy");
        drop(listener);
        assert!(
            wait_for(Duration::from_secs(5), || !port_is_busy(port)),
            "a released port must read as free"
        );
    }

    /// `start` must refuse before spawning anything when the port is taken —
    /// the alternative is a child that immediately fails to bind and a sweep
    /// that measures whatever was already there.
    #[test]
    fn starting_against_an_occupied_port_is_refused_without_spawning() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let log =
            std::env::temp_dir().join(format!("orangu-sweep-test-{}.log", std::process::id()));
        let canary =
            std::env::temp_dir().join(format!("orangu-sweep-must-not-run-{}", std::process::id()));
        let err = start(
            // Would create the file if it ever ran; it must not run.
            &format!("touch {}", canary.display()),
            &[],
            port,
            &log,
            Duration::from_secs(1),
        )
        .expect_err("an occupied port must be refused");
        assert!(err.to_string().contains("already in use"), "{err}");
        assert!(!canary.exists(), "the command must not have been spawned");
        let _ = std::fs::remove_file(&log);
    }

    /// Whether a pid is still running, via POSIX `kill -0`.
    #[cfg(unix)]
    fn alive(pid: u32) -> bool {
        Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    }

    /// **The guard has to actually kill what it owns.** A leaked server holds
    /// the port, every subsequent point of the sweep then measures *it*, and
    /// the run reports the swept variable as having no effect — which reads as
    /// a real result. Nothing else in this module catches that; the pre-flight
    /// port check only fires on the point after the leak, by which time the
    /// numbers are already wrong.
    ///
    /// Also pins the `exec`: `child.id()` must be the pid of the process the
    /// command names, not of a shell that forked it, or both the kill and the
    /// `/props` identity check are aimed at the wrong process.
    /// Unix-only, like the mechanism it pins: `start` refuses to run at all
    /// without a POSIX shell (see its `cfg`), and this drives `sh` and
    /// `kill` directly to check the `exec` and the teardown.
    #[cfg(unix)]
    #[test]
    fn dropping_the_guard_kills_the_process_it_owns() {
        let child = Command::new("sh")
            .arg("-c")
            .arg("exec sleep 60")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawning a sleep must work");
        let pid = child.id();
        // A port nothing is listening on, so `Drop`'s wait-for-release
        // returns on its first poll.
        let free = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let server = Server {
            child,
            port: free,
            log: std::path::PathBuf::from("/dev/null"),
        };
        assert_eq!(server.pid(), pid);
        assert!(alive(pid), "the child should be running before the drop");
        drop(server);
        assert!(
            !alive(pid),
            "the guard must kill and reap the process it owns — a leaked server \
             would be measured by every later point of the sweep"
        );
    }
}
