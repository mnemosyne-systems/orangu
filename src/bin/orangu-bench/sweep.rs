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
//! - no *accelerator* may be held by a process this sweep did not start, or
//!   the run stops — see [`Baseline`];
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
    ///
    /// # Values that contain commas
    ///
    /// `ORANGU_DEVICE_SPLIT` takes a ratio — `3,1` puts three quarters of the
    /// layers on the first device — so the separator this list has always used
    /// is also a character inside a legal value. Comma-splitting
    /// `ORANGU_DEVICE_SPLIT=off,all,3,1` yields **four** points, and because
    /// `3` and `1` are themselves legal one-entry ratios, nothing errors: the
    /// sweep runs, the table has four columns, and two of them are
    /// configurations nobody asked for. That is the failure this module's
    /// documentation is otherwise entirely about — a sweep that answers
    /// confidently and wrongly.
    ///
    /// So: **a spec containing `;` is split on `;` and never on `,`**, which
    /// makes `ORANGU_DEVICE_SPLIT=off;all;3,1` say what it looks like it says.
    /// A spec with no `;` splits on `,` exactly as before, so every existing
    /// invocation means what it meant. `;` rather than `/` because a swept
    /// value here is as often a *path* as a list — `MODEL=a.gguf;b.gguf`
    /// through `--sweep-cmd` is how models get swept — and a separator that
    /// cannot appear in a path is the only one that can carry them.
    ///
    /// The remaining ambiguity is a comma list that *wanted* to be a ratio,
    /// which no parser can see. For `ORANGU_DEVICE_SPLIT` specifically it can:
    /// a bare integer is never a useful split point on its own, so it is
    /// refused with the fix rather than measured.
    pub fn parse(spec: &str) -> anyhow::Result<Self> {
        let (var, values) = spec
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("--sweep wants VAR=v1,v2,..., got {spec:?}"))?;
        let var = var.trim();
        if var.is_empty() {
            anyhow::bail!("--sweep {spec:?} has no variable name");
        }
        let separator = if values.contains(';') { ';' } else { ',' };
        let values: Vec<String> = values
            .split(separator)
            .map(|v| v.trim().to_string())
            .collect();
        if values.is_empty() {
            anyhow::bail!("--sweep {spec:?} has no values");
        }
        // A ratio that lost its comma. `3` alone means "one device, all of
        // it", which is `off` with extra steps — nobody sweeps it, so its
        // presence is evidence of the split rather than of intent.
        if var == "ORANGU_DEVICE_SPLIT"
            && separator == ','
            && let Some(bare) = values.iter().find(|v| v.parse::<u32>().is_ok())
        {
            anyhow::bail!(
                "--sweep {spec:?}: {bare:?} on its own is not a split to sweep — a ratio \
                 like `3,1` contains the comma this list is separated by, so it was read \
                 as two points. Separate the points with `;` instead: \
                 --sweep '{var}=off;all;3,1'"
            );
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
    baseline: &Baseline,
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
    // A free port is not a free machine. The previous point's server may have
    // orphaned an `npu-compile` child, which holds the NPU for minutes and
    // never had a port at all — so it passes the check above and then shares
    // this machine with the server about to start, and the sharing is
    // reported as the swept variable being slower. Compare against the
    // baseline rather than demanding an idle GPU, so a machine with a
    // compositor on it is still usable; see [`Baseline`].
    let busy = wait_for_idle_accelerators(baseline, timeout);
    if let Some(holder) = busy.first() {
        anyhow::bail!(
            "pid {} still holds an accelerator after {}s and this sweep did not start it — \
             most likely an `npu-compile` child orphaned by an earlier server, which keeps \
             compiling for minutes after its parent is gone. Measuring alongside it would \
             attribute its contention to the configuration under test. Wait for it or stop \
             it: {}",
            holder.pid,
            timeout.as_secs(),
            holder.cmd
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

/// The accelerator holders that were already on this machine before the sweep.
///
/// # Why a free port is not a free machine
///
/// The server's own teardown is *not* the problem, and it is worth writing
/// down that it was measured rather than assumed: killing a server with a
/// 10.8 GB resident set released its port and reaped the process in the same
/// 0.57 s, and a smaller one in 0.35 s. `Drop`'s `wait()` already covers that
/// entirely, and there is no window there to close.
///
/// The problem is the **child**. `npu-compile` runs as a separate process —
/// it has to, because the NPU runtime and the GPU runtime cannot share an
/// address space — and the server blocks in `status()` while it works. Kill
/// the server during a compile and the compiler is reparented to init and
/// keeps going. Measured on this project's own hardware: an orphaned
/// `npu-compile` was still running **eleven minutes** after its parent was
/// killed — uninterruptible, holding `/dev/aipu` and every NPU core, and
/// still going when it was finally killed by hand.
///
/// Nothing that looks for a leftover *server* can see that process. It never
/// bound a port, so the pre-flight port check passes; it answers no `/health`
/// and reports no pid, so the `/props` identity check never gets a chance.
/// What it does is make the next point of the sweep share the NPU with the
/// last point's compiler, which reads as the swept variable being slower. No
/// error is produced and the number is entirely plausible — the failure this
/// whole module exists to refuse.
///
/// The server side of this is fixed too (`compile_child` now arms
/// `PR_SET_PDEATHSIG`), so a current server orphans nothing. This check is
/// what makes that verifiable rather than assumed, and it still catches the
/// case that fix cannot reach: a server from an *older* build, or one started
/// by hand outside the sweep.
///
/// So the check cannot simply be "no process holds an accelerator": on any
/// machine with a display, the compositor holds one permanently and always
/// will, and a tuning tool that refuses to run on a desktop is a tool nobody
/// runs. What matters is *change* — a holder that appeared during the sweep is
/// one of the sweep's own leftovers, and a holder that was there at the start
/// is part of the furniture. This records the furniture, once, before the
/// first point.
///
/// Pids rather than pid-and-start-time: a recycled pid would have to be
/// recycled onto a process that also holds an accelerator, within one sweep,
/// to matter, and the cost of that miss is one point measured the way every
/// point was measured before this existed.
#[derive(Debug, Default, Clone)]
pub struct Baseline {
    pids: Vec<u32>,
}

/// The accelerators in use right now, as the furniture this sweep runs
/// against. Take it **once**, before the first server is started.
pub fn accelerator_baseline() -> Baseline {
    Baseline {
        pids: accelerator_holders().into_iter().map(|h| h.pid).collect(),
    }
}

/// A process holding an accelerator device open.
#[derive(Debug, Clone)]
struct Holder {
    pid: u32,
    /// The command line, for the error message. A pid alone tells the
    /// operator nothing they can act on, and by the time they read it the
    /// process may be gone.
    cmd: String,
}

/// The device-node prefixes that mean "an accelerator is in use".
///
/// Prefixes, so `/dev/mali0`, `/dev/dri/renderD128` and `/dev/nvidia0` all
/// match without this having to enumerate minor numbers. Not exhaustive and
/// not trying to be: an accelerator this misses costs the check, not
/// correctness, because the port check and the `/props` pid check are both
/// still in front of it.
// Only the `/proc` walk consults this, and that is Linux-only; `cfg(test)`
// keeps it present for the test that pins the matching on every platform.
#[cfg(any(target_os = "linux", test))]
const ACCELERATOR_NODES: &[&str] = &[
    // Arm Mali, through the vendor kbase driver rather than DRM.
    "/dev/mali",
    // Arm Zhouyi NPU, which is what `npu-compile` opens.
    "/dev/aipu",
    // Any DRM render node — Mesa, and the GPU half of most other stacks.
    "/dev/dri/render",
    "/dev/nvidia",
];

/// Whether `target` is one of the device nodes [`ACCELERATOR_NODES`] names.
#[cfg(any(target_os = "linux", test))]
fn is_accelerator_node(target: &str) -> bool {
    ACCELERATOR_NODES.iter().any(|p| target.starts_with(p))
}

/// Every process other than this one that holds an accelerator device open.
#[cfg(target_os = "linux")]
fn accelerator_holders() -> Vec<Holder> {
    holders_of(is_accelerator_node)
}

/// Every process other than this one holding open a file `matches` accepts.
///
/// By walking `/proc/<pid>/fd`, which sees only processes this user owns.
/// Another user's server is therefore invisible here — but it is not
/// invisible to the port check, which is a connect and does not care who owns
/// what, so the case that actually corrupts a sweep is still caught.
///
/// The predicate is a parameter so the walk can be tested against an ordinary
/// file. The alternative — a test that holds a real accelerator open — would
/// mean the suite opening `/dev/aipu` on every run, which is the NPU this
/// project is trying to keep free.
#[cfg(target_os = "linux")]
fn holders_of(matches: impl Fn(&str) -> bool) -> Vec<Holder> {
    let me = std::process::id();
    let mut held = Vec::new();
    let Ok(entries) = std::fs::read_dir("/proc") else {
        return held;
    };
    for entry in entries.flatten() {
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|n| n.parse::<u32>().ok())
        else {
            continue;
        };
        if pid == me {
            continue;
        }
        let Ok(fds) = std::fs::read_dir(entry.path().join("fd")) else {
            // Gone between the listing and the read, or another user's.
            continue;
        };
        let holds = fds.flatten().any(|fd| {
            std::fs::read_link(fd.path())
                .ok()
                .and_then(|t| t.to_str().map(&matches))
                .unwrap_or(false)
        });
        if holds {
            held.push(Holder {
                pid,
                cmd: process_command(pid),
            });
        }
    }
    held
}

/// Nothing to report where there is no `/proc` to read it from, which leaves
/// the sweep behaving exactly as it did before this check existed.
#[cfg(not(target_os = "linux"))]
fn accelerator_holders() -> Vec<Holder> {
    Vec::new()
}

/// `pid`'s command line, space-separated, or a placeholder.
#[cfg(target_os = "linux")]
fn process_command(pid: u32) -> String {
    match std::fs::read(format!("/proc/{pid}/cmdline")) {
        Ok(raw) if !raw.is_empty() => String::from_utf8_lossy(&raw)
            .replace('\0', " ")
            .trim()
            .to_string(),
        // A kernel thread, or one that exited underneath the read.
        _ => "<unknown>".to_string(),
    }
}

/// Wait for every accelerator holder outside `baseline` to go away.
///
/// Returns the ones still there when `timeout` runs out, so the caller can
/// name them. An empty result is the machine being as free as it was when the
/// sweep started.
fn wait_for_idle_accelerators(baseline: &Baseline, timeout: Duration) -> Vec<Holder> {
    let mut remaining = Vec::new();
    let idle = wait_for(timeout, || {
        remaining = holders_outside(baseline, accelerator_holders());
        remaining.is_empty()
    });
    if idle { Vec::new() } else { remaining }
}

/// The holders in `holders` that are not part of `baseline`.
///
/// Separated from the polling above so the policy — *which* holders stop a
/// sweep — can be checked without a GPU to hold.
fn holders_outside(baseline: &Baseline, holders: Vec<Holder>) -> Vec<Holder> {
    holders
        .into_iter()
        .filter(|h| !baseline.pids.contains(&h.pid))
        .collect()
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

    /// A split ratio contains the separator the value list uses. Semicolons
    /// take precedence so the ratio survives as one point, and the comma form
    /// keeps meaning exactly what it meant for every sweep written before this
    /// existed.
    #[test]
    fn a_semicolon_list_keeps_commas_inside_a_value() {
        let s = Spec::parse("ORANGU_DEVICE_SPLIT=off;all;3,1").unwrap();
        assert_eq!(s.values, ["off", "all", "3,1"]);
        assert_eq!(s.label("3,1"), "ORANGU_DEVICE_SPLIT=3,1");
        // A path list is the other value that cannot use `/`, and is why the
        // separator is `;`.
        let m = Spec::parse("MODEL=/models/a.gguf;/models/b.gguf").unwrap();
        assert_eq!(m.values, ["/models/a.gguf", "/models/b.gguf"]);
        // Unchanged: no semicolon, no new behaviour.
        assert_eq!(
            Spec::parse("ORANGU_COOP_MIN_TOKENS=8,16").unwrap().values,
            ["8", "16"]
        );
    }

    /// The trap this replaced: `off,all,3,1` parsed into four points, two of
    /// them halves of a ratio, and ran a whole sweep without a complaint. It
    /// has to be refused, and the refusal has to say how to spell it.
    #[test]
    fn a_split_ratio_flattened_by_commas_is_refused_with_the_fix() {
        let text = match Spec::parse("ORANGU_DEVICE_SPLIT=off,all,3,1") {
            Ok(spec) => panic!("a flattened ratio must not be measured: {:?}", spec.values),
            Err(err) => err.to_string(),
        };
        assert!(text.contains(';'), "{text}");
        assert!(text.contains("3,1"), "{text}");
        // Modes on their own are still a legal comma sweep — the refusal is
        // for the numbers, not for the variable.
        assert_eq!(
            Spec::parse("ORANGU_DEVICE_SPLIT=off,all").unwrap().values,
            ["off", "all"]
        );
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
            &accelerator_baseline(),
        )
        .expect_err("an occupied port must be refused");
        assert!(err.to_string().contains("already in use"), "{err}");
        assert!(!canary.exists(), "the command must not have been spawned");
        let _ = std::fs::remove_file(&log);
    }

    /// The node list matches by prefix so it covers a whole driver's minor
    /// numbers, and it must not match its way onto the rest of `/dev` — a
    /// check that fires on `/dev/null` would stop every sweep on every
    /// machine.
    #[test]
    fn accelerator_nodes_match_by_prefix_and_nothing_else() {
        for held in [
            "/dev/mali0",
            "/dev/aipu",
            "/dev/dri/renderD128",
            "/dev/dri/renderD129",
            "/dev/nvidia0",
        ] {
            assert!(is_accelerator_node(held), "{held} should count as held");
        }
        for other in [
            "/dev/null",
            "/dev/zero",
            "/dev/dri/card0",
            "/home/someone/model.gguf",
            "socket:[12345]",
        ] {
            assert!(!is_accelerator_node(other), "{other} must not count");
        }
    }

    /// **The baseline is what makes this check usable on a real machine.**
    /// Anything already holding an accelerator when the sweep started is
    /// furniture — a compositor, another user's long-running job — and must
    /// not stop a single point. Only a holder that *appeared* is a leftover
    /// of the previous point, and that one has to stop the run.
    #[test]
    fn only_holders_that_appeared_after_the_baseline_stop_a_sweep() {
        let holder = |pid: u32| Holder {
            pid,
            cmd: format!("proc-{pid}"),
        };
        let baseline = Baseline { pids: vec![10, 20] };
        // Exactly the furniture: nothing to wait for.
        assert!(
            holders_outside(&baseline, vec![holder(10), holder(20)]).is_empty(),
            "a holder present before the sweep must not stop it"
        );
        // The previous point's server, still unwinding its GPU memory.
        let new = holders_outside(&baseline, vec![holder(10), holder(30), holder(20)]);
        assert_eq!(new.len(), 1);
        assert_eq!(new[0].pid, 30);
        // The command line travels with it — a bare pid is not something an
        // operator can act on, least of all after the process has gone.
        assert_eq!(new[0].cmd, "proc-30");
    }

    /// A machine that has not changed since the baseline was taken must clear
    /// the wait immediately, however many accelerators are in use on it. This
    /// is the case every well-behaved sweep is in at every point, so a
    /// mistake here would cost the whole timeout on each one.
    #[test]
    fn an_unchanged_machine_clears_the_wait_at_once() {
        let baseline = accelerator_baseline();
        let start = Instant::now();
        let busy = wait_for_idle_accelerators(&baseline, Duration::from_secs(30));
        assert!(
            busy.is_empty(),
            "nothing new can be holding an accelerator between two adjacent \
             calls, but {busy:?} was reported"
        );
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "an unchanged machine must not spend the timeout"
        );
    }

    /// **The walk has to actually find a holder.** Everything above this is
    /// policy — which holders matter — and none of it is worth anything if
    /// the `/proc` scan that feeds it comes back empty on a machine that has
    /// a leftover on it. That is the failure mode with no symptom: the check
    /// passes every time and protects nothing.
    ///
    /// Driven against an ordinary file rather than a device, through the same
    /// code path, so it neither needs an accelerator nor opens the NPU this
    /// project is trying to keep free.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_proc_walk_finds_a_process_holding_a_file_open() {
        let path = std::env::temp_dir().join(format!(
            "orangu-sweep-held-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, b"held").unwrap();
        let target = path.to_str().unwrap().to_string();

        // Holds the file open on fd 3 and does nothing else.
        let mut child = Command::new("sh")
            .arg("-c")
            .arg(format!("exec 3<{target}; sleep 30"))
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .expect("spawning the holder must work");

        // The `exec` replaces the shell, so the open happens in `child`'s own
        // pid — but the scan can still run before the redirect has. Poll.
        let wanted = target.clone();
        let mut found = Vec::new();
        let seen = wait_for(Duration::from_secs(10), || {
            found = holders_of(|t| t == wanted);
            found.iter().any(|h| h.pid == child.id())
        });

        let holder = found.iter().find(|h| h.pid == child.id()).cloned();
        let _ = child.kill();
        let _ = child.wait();
        let _ = std::fs::remove_file(&path);

        assert!(
            seen,
            "the /proc walk must find the process holding {target} open; it \
             reported {found:?}"
        );
        // The command line travels with the pid, or the error a sweep prints
        // names a number and nothing an operator can act on.
        let holder = holder.expect("checked by the assertion above");
        assert!(
            holder.cmd.contains("sleep 30"),
            "the holder's command line must come back with it, got {:?}",
            holder.cmd
        );
    }

    /// A file nobody has open must not read as held — a walk that matched
    /// too eagerly would refuse to start every point of every sweep.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_file_no_one_holds_open_has_no_holders() {
        let path = std::env::temp_dir().join(format!("orangu-sweep-unheld-{}", std::process::id()));
        let target = path.to_str().unwrap().to_string();
        assert!(
            holders_of(|t| t == target).is_empty(),
            "nothing should hold {target} open"
        );
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
