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

//! `--daemon`, in two steps around the start-up.
//!
//! A fork keeps only the thread that called it. The server used to fork
//! once the model was loaded and the listeners bound — and by then loading
//! had built thread pools (rayon's, the CPU backend's) and a `[workers]`
//! node had started its listener: the daemon woke up without any of them,
//! answered `/health` and waited forever on the first request that needed a
//! pool.
//!
//! So [`begin`] forks first, before any thread exists, and the original
//! process only waits: on a pipe, for the child to say it is ready. The
//! child loads and binds still attached to the terminal, so a bad config, an
//! unresolvable model or an address in use is printed there as before, and
//! then [`ready`] detaches it — a new session, `/`, `/dev/null` for stdio —
//! and tells the waiting process, which exits 0. A child that fails first
//! closes the pipe without a word, and the waiting process exits with the
//! child's own status.
//!
//! The pipe is inherited across `exec`, on purpose: a start that fails over
//! to a fallback model re-executes the child in the foreground, as an
//! attached start would, and the waiting process then waits on that.
//!
//! The child [`begin`] forks is a watchdog, and the server its own child in
//! turn. The watchdog starts no thread, so it can fork again: a server that
//! got as far as [`ready`] and then ends any way but a clean exit or a
//! `SIGTERM`/`SIGINT` — a panic, the OOM killer's `SIGKILL`, a lost GPU's
//! exit 75 — is started again after a backoff of 1, 2, 4 … seconds, at most
//! [`MAX_BACKOFF`], back to 1 once a run lasted [`STEADY`]. The new server
//! finds why the last one ended in `ORANGU_LAST_CRASH`, and when in
//! `ORANGU_LAST_CRASH_AT` (seconds since the epoch). A server that fails
//! before it is ready is not started again: its status reaches the
//! terminal as before. `SIGTERM` or `SIGINT` sent to the watchdog goes on
//! to the server.

use anyhow::Result;

#[cfg(unix)]
mod imp {
    use anyhow::{Context, Result, anyhow, bail};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
    use std::time::{Duration, Instant};

    /// The write end of the pipe, in the child between [`begin`] and
    /// [`ready`].
    static WRITER: Mutex<Option<i32>> = Mutex::new(None);

    /// The write end of the watchdog's pipe, in the first server: [`ready`]
    /// tells the watchdog the start went through.
    static WATCH: Mutex<Option<i32>> = Mutex::new(None);

    /// Set in a server the watchdog started again: it is detached already.
    static RESTARTED: AtomicBool = AtomicBool::new(false);

    /// The server the watchdog runs, for its signal handler.
    static SERVER: AtomicI32 = AtomicI32::new(0);

    const READY: &[u8] = b"ok";

    pub fn begin() -> Result<()> {
        let mut fds = [0i32; 2];
        // Safety: `pipe` fills the two descriptors of a two-element array.
        if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
            bail!("pipe: {}", std::io::Error::last_os_error());
        }
        let [reader, writer] = fds;
        // Safety: called before any thread exists (see `main`), so the
        // child is a whole copy of this process.
        match unsafe { libc::fork() } {
            -1 => bail!("fork: {}", std::io::Error::last_os_error()),
            0 => {
                // Safety: closing this process's own copy of the read end.
                unsafe { libc::close(reader) };
                let mut watch = [0i32; 2];
                // Safety: as above.
                if unsafe { libc::pipe(watch.as_mut_ptr()) } != 0 {
                    bail!("pipe: {}", std::io::Error::last_os_error());
                }
                let [watch_reader, watch_writer] = watch;
                // Safety: still before any thread exists.
                match unsafe { libc::fork() } {
                    -1 => bail!("fork: {}", std::io::Error::last_os_error()),
                    0 => {
                        // Safety: this process's own copy of the read end.
                        unsafe { libc::close(watch_reader) };
                        *WRITER.lock().unwrap() = Some(writer);
                        *WATCH.lock().unwrap() = Some(watch_writer);
                        Ok(())
                    }
                    server => {
                        // Safety: the watchdog keeps neither write end: the
                        // waiting process hears from the server alone.
                        unsafe {
                            libc::close(writer);
                            libc::close(watch_writer);
                        }
                        supervise(server, watch_reader)
                    }
                }
            }
            child => {
                // Safety: closing the write end, so the read below ends when
                // the child's copy closes.
                unsafe { libc::close(writer) };
                std::process::exit(wait_for(reader, child));
            }
        }
    }

    /// The waiting process: 0 when the child said it is ready, else the
    /// child's exit status.
    fn wait_for(reader: i32, child: libc::pid_t) -> i32 {
        let mut said = Vec::new();
        let mut buf = [0u8; 16];
        loop {
            // Safety: reads into a buffer of the length given.
            let n = unsafe { libc::read(reader, buf.as_mut_ptr().cast(), buf.len()) };
            if n > 0 {
                said.extend_from_slice(&buf[..n as usize]);
            } else if n == 0
                || std::io::Error::last_os_error().kind() != std::io::ErrorKind::Interrupted
            {
                break;
            }
        }
        if said == READY {
            return 0;
        }
        let mut status = 0;
        // Safety: waits for this process's own child.
        if unsafe { libc::waitpid(child, &mut status, 0) } == child && libc::WIFEXITED(status) {
            libc::WEXITSTATUS(status).max(1)
        } else {
            1
        }
    }

    /// A run that lasts this long is steady: the next backoff starts over.
    pub const STEADY: Duration = Duration::from_secs(600);

    /// The longest wait before a server is started again.
    pub const MAX_BACKOFF: Duration = Duration::from_secs(60);

    extern "C" fn pass_on(signal: libc::c_int) {
        let server = SERVER.load(Ordering::Relaxed);
        if server > 0 {
            // Safety: `kill` is async-signal-safe.
            unsafe { libc::kill(server, signal) };
        }
    }

    /// Why a server ended, as its wait status says, or `None` for an end
    /// that was asked for: a clean exit, `SIGTERM` or `SIGINT`.
    pub fn crash_reason(status: i32) -> Option<String> {
        if libc::WIFEXITED(status) {
            match libc::WEXITSTATUS(status) {
                0 => None,
                75 => Some("exited 75: the GPU was lost".to_string()),
                code => Some(format!("exited {code}")),
            }
        } else if libc::WIFSIGNALED(status) {
            match libc::WTERMSIG(status) {
                libc::SIGTERM | libc::SIGINT => None,
                libc::SIGKILL => Some("killed (SIGKILL; the OOM killer?)".to_string()),
                libc::SIGSEGV => Some("crashed (SIGSEGV)".to_string()),
                libc::SIGABRT => Some("aborted (SIGABRT)".to_string()),
                libc::SIGBUS => Some("crashed (SIGBUS)".to_string()),
                signal => Some(format!("ended by signal {signal}")),
            }
        } else {
            Some(format!("ended with status {status}"))
        }
    }

    /// The wait before start number `restarts` + 1 of a run of crashes.
    pub fn backoff(restarts: u32) -> Duration {
        Duration::from_secs(1u64 << restarts.min(6)).min(MAX_BACKOFF)
    }

    /// The watchdog: waits for `server`, and starts it again after a crash.
    /// Returns only in a server it started again.
    fn supervise(mut server: libc::pid_t, watch_reader: i32) -> Result<()> {
        // Safety: handlers that only call `kill`; the hang-up of the
        // terminal the first server detached from is the watchdog's to
        // ignore.
        unsafe {
            libc::signal(libc::SIGTERM, pass_on as *const () as libc::sighandler_t);
            libc::signal(libc::SIGINT, pass_on as *const () as libc::sighandler_t);
            libc::signal(libc::SIGHUP, libc::SIG_IGN);
        }
        SERVER.store(server, Ordering::Relaxed);
        let mut started = Instant::now();
        let mut restarts = 0u32;
        let mut detached = false;
        loop {
            let mut status = 0;
            // Safety: waits for this process's own child.
            let waited = unsafe { libc::waitpid(server, &mut status, 0) };
            if waited == -1 {
                if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted {
                    continue;
                }
                std::process::exit(1);
            }
            if !detached {
                let mut said = [0u8; 2];
                // Safety: reads into a buffer of the length given; the write
                // end is closed by now, so this does not block.
                let n = unsafe { libc::read(watch_reader, said.as_mut_ptr().cast(), said.len()) };
                if n != READY.len() as isize || said != READY {
                    // Never ready: the waiting process reports it.
                    std::process::exit(if libc::WIFEXITED(status) {
                        libc::WEXITSTATUS(status)
                    } else {
                        1
                    });
                }
                // Safety: as `ready` does for the server.
                unsafe {
                    libc::close(watch_reader);
                    libc::setsid();
                    libc::umask(0o027);
                    let null = libc::open(c"/dev/null".as_ptr(), libc::O_RDWR);
                    if null >= 0 {
                        for fd in 0..3 {
                            libc::dup2(null, fd);
                        }
                        if null > 2 {
                            libc::close(null);
                        }
                    }
                }
                detached = true;
            }
            let Some(reason) = crash_reason(status) else {
                std::process::exit(if libc::WIFEXITED(status) {
                    libc::WEXITSTATUS(status)
                } else {
                    0
                });
            };
            if started.elapsed() >= STEADY {
                restarts = 0;
            }
            std::thread::sleep(backoff(restarts));
            restarts += 1;
            let at = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());
            // Safety: the watchdog is single-threaded.
            unsafe {
                std::env::set_var("ORANGU_LAST_CRASH", &reason);
                std::env::set_var("ORANGU_LAST_CRASH_AT", at.to_string());
            }
            // Safety: the watchdog is single-threaded.
            match unsafe { libc::fork() } {
                -1 => std::process::exit(1),
                0 => {
                    // Safety: the server takes its own signals again.
                    unsafe {
                        libc::signal(libc::SIGTERM, libc::SIG_DFL);
                        libc::signal(libc::SIGINT, libc::SIG_DFL);
                        libc::signal(libc::SIGHUP, libc::SIG_DFL);
                    }
                    RESTARTED.store(true, Ordering::Relaxed);
                    return Ok(());
                }
                pid => {
                    server = pid;
                    SERVER.store(server, Ordering::Relaxed);
                    started = Instant::now();
                }
            }
        }
    }

    pub fn ready() -> Result<()> {
        if RESTARTED.load(Ordering::Relaxed) {
            // Detached with the watchdog; only the directory is left.
            std::env::set_current_dir("/").context("chdir /")?;
            return Ok(());
        }
        let Some(writer) = WRITER.lock().unwrap().take() else {
            return Err(anyhow!("--daemon: not started through daemon::begin"));
        };
        // Safety: each call changes only this process's own session,
        // directory, mask and descriptors; `/dev/null` is opened read-write
        // and duplicated onto 0, 1 and 2.
        unsafe {
            if libc::setsid() == -1 {
                bail!("setsid: {}", std::io::Error::last_os_error());
            }
            libc::umask(0o027);
            std::env::set_current_dir("/").context("chdir /")?;
            let null = libc::open(c"/dev/null".as_ptr(), libc::O_RDWR);
            if null < 0 {
                bail!("/dev/null: {}", std::io::Error::last_os_error());
            }
            for fd in 0..3 {
                libc::dup2(null, fd);
            }
            if null > 2 {
                libc::close(null);
            }
            libc::write(writer, READY.as_ptr().cast(), READY.len());
            libc::close(writer);
            if let Some(watch) = WATCH.lock().unwrap().take() {
                libc::write(watch, READY.as_ptr().cast(), READY.len());
                libc::close(watch);
            }
        }
        Ok(())
    }
}

/// Forks; only the child returns. The parent waits for [`ready`], or for
/// the child to fail, and exits. Must be called before any thread exists.
#[cfg(unix)]
pub fn begin() -> Result<()> {
    imp::begin()
}

/// Detaches the child from the terminal and releases the waiting parent.
#[cfg(unix)]
pub fn ready() -> Result<()> {
    imp::ready()
}

#[cfg(not(unix))]
pub fn begin() -> Result<()> {
    Err(anyhow::anyhow!(
        "--daemon is only supported on Unix-like platforms"
    ))
}

#[cfg(not(unix))]
pub fn ready() -> Result<()> {
    Ok(())
}

#[cfg(all(test, unix))]
mod tests {
    use super::imp::{backoff, crash_reason};
    use std::time::Duration;

    #[test]
    fn an_asked_for_end_is_no_crash() {
        assert_eq!(crash_reason(0), None);
        assert_eq!(crash_reason(libc::SIGTERM), None);
        assert_eq!(crash_reason(libc::SIGINT), None);
        // Exit statuses sit in the second byte of a wait status.
        assert_eq!(
            crash_reason(75 << 8).as_deref(),
            Some("exited 75: the GPU was lost")
        );
        assert_eq!(crash_reason(1 << 8).as_deref(), Some("exited 1"));
        assert!(crash_reason(libc::SIGKILL).unwrap().contains("SIGKILL"));
        assert!(crash_reason(libc::SIGSEGV).unwrap().contains("SIGSEGV"));
    }

    #[test]
    fn the_backoff_doubles_up_to_a_minute() {
        let waits: Vec<u64> = (0..9).map(|n| backoff(n).as_secs()).collect();
        assert_eq!(waits, [1, 2, 4, 8, 16, 32, 60, 60, 60]);
        assert_eq!(backoff(40), Duration::from_secs(60));
    }
}
