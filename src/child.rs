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

//! Child processes that do not outlive the process that wanted them.
//!
//! # Why this exists
//!
//! Two places here spawn a long-running child and then block waiting for it:
//! `npu-compile`, which the server runs out-of-process because the NPU and GPU
//! vendor runtimes cannot share an address space, and `perf record`, which the
//! benchmark runs while it measures. Both parents get killed in normal use — a
//! sweep moving to its next point, a Ctrl-C, an operator restarting a server —
//! and a killed parent runs no cleanup code at all. The child is reparented and
//! keeps going.
//!
//! That is not a tidiness problem. Measured on this project's hardware before
//! it was fixed: an orphaned `npu-compile` was still running **eleven minutes**
//! after its server was killed, holding `/dev/aipu` and every NPU core, and had
//! to be killed by hand. Nothing looking for a leftover *server* can see such a
//! process — it binds no port and answers no health check — so the next
//! measurement quietly shares the machine with it and reports the contention as
//! whatever it was varying.
//!
//! [`die_with_parent`] is the only thing that closes that window, because it is
//! the kernel rather than the parent that does the work.

/// Ask the kernel to send `signal` to this command's child when **this**
/// process dies.
///
/// Call it on the [`Command`](std::process::Command) before spawning; it
/// arms `PR_SET_PDEATHSIG` between `fork` and `exec`, where the setting
/// survives the `exec` that follows (the kernel clears it only for
/// set-user-ID binaries).
///
/// # Choosing the signal
///
/// `SIGKILL` when the child has nothing worth finishing — a compile whose
/// requester is gone produces an artifact nobody asked for. `SIGINT`, or
/// another signal the child handles, when stopping cleanly matters: `perf
/// record` writes its data file while shutting down and a killed one leaves an
/// unreadable stub behind.
///
/// # The race, and why the parent check is not against pid 1
///
/// `PR_SET_PDEATHSIG` fires on the parent's death, so a parent that died in
/// the window between `fork` and this code running has *already* missed it —
/// the signal would never arrive and the child would be the very orphan this
/// prevents. Re-reading the parent afterwards closes that: if it changed, the
/// child declines to exec at all.
///
/// The comparison is against the caller's own pid, captured before the fork,
/// and deliberately **not** against 1. Reparenting does not always go to init:
/// a `systemd --user` session sets `PR_SET_CHILD_SUBREAPER` and adopts the
/// orphans of its own processes, so `getppid() == 1` would never become true on
/// exactly the desktop and server machines most likely to run this.
///
/// # Platforms
///
/// Linux-only, because `prctl` is. Everywhere else this does nothing and the
/// child behaves as it did before — which is also why it is not a substitute
/// for a parent that stops its children on the paths where it still can.
#[cfg(target_os = "linux")]
pub fn die_with_parent(command: &mut std::process::Command, signal: libc::c_int) {
    use std::os::unix::process::CommandExt;
    let parent = std::process::id() as libc::pid_t;
    // Safety: the closure runs between `fork` and `exec` in the child, where
    // only async-signal-safe calls are legal. `prctl` and `getppid` are plain
    // syscalls with no allocation, no locking and no libc state behind them,
    // and `parent` is an integer copied in before the fork.
    unsafe {
        command.pre_exec(move || {
            if libc::prctl(libc::PR_SET_PDEATHSIG, signal) == -1 {
                return Err(std::io::Error::last_os_error());
            }
            if libc::getppid() != parent {
                return Err(std::io::Error::from_raw_os_error(libc::ESRCH));
            }
            Ok(())
        });
    }
}

/// Nothing to arm without `prctl`; the child is spawned exactly as before.
#[cfg(not(target_os = "linux"))]
pub fn die_with_parent(command: &mut std::process::Command, signal: i32) {
    let _ = (command, signal);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::{Command, Stdio};

    /// **An armed command must still run.** Everything this module protects
    /// depends on the `pre_exec` succeeding: the closure runs in the child
    /// between `fork` and `exec`, and an error returned from it does not warn
    /// — it fails the spawn outright. A wrong constant, or a `prctl` the
    /// kernel refuses, would therefore not leak a child but stop every compile
    /// and every profile from starting at all.
    #[test]
    fn an_armed_command_still_spawns_and_exits_normally() {
        let mut command = Command::new("sh");
        command
            .arg("-c")
            .arg("exit 0")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        #[cfg(target_os = "linux")]
        die_with_parent(&mut command, libc::SIGKILL);
        #[cfg(not(target_os = "linux"))]
        die_with_parent(&mut command, 9);

        let status = command
            .status()
            .expect("arming the death signal must not stop the child from starting");
        assert!(
            status.success(),
            "the child must run normally once armed, got {status}"
        );
    }

    /// The child's own exit status has to survive intact — the `pre_exec` must
    /// not swallow or rewrite it, or a failed compile would read as a
    /// successful one.
    #[test]
    fn an_armed_command_reports_its_own_failure() {
        let mut command = Command::new("sh");
        command
            .arg("-c")
            .arg("exit 3")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        #[cfg(target_os = "linux")]
        die_with_parent(&mut command, libc::SIGINT);
        #[cfg(not(target_os = "linux"))]
        die_with_parent(&mut command, 2);

        let status = command.status().expect("the child must start");
        assert_eq!(status.code(), Some(3), "the child's exit code must survive");
    }
}
