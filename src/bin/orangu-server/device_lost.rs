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

//! What this process does when the GPU goes away underneath it.
//!
//! A driver reset ("the CS has been cancelled because the context is lost",
//! in RADV's wording) destroys the `wgpu` device permanently: every buffer
//! map, poll, and submission on it fails from then on, and there is no API
//! to re-create it in place — the weights that were uploaded to it are gone
//! with it. Nothing the engine can do makes the *current* request finish
//! correctly.
//!
//! So the response is not to retry, and not to soldier on: it's to fail the
//! in-flight request with one clear sentence, write the real detail to this
//! server's own log, and exit with [`EXIT_CODE`] a moment later, so whatever
//! supervises this process — `orangu-coordinator` in the usual stack, which
//! restarts a dead child on the very next request (see its
//! `Coordinator::ensure_active`) — brings it back on a *fresh* device at
//! full speed.
//!
//! Before this, a lost device surfaced as `.expect()` panics from inside the
//! `wgpu` readback paths: the client got a Rust backtrace as its "error
//! message", and the process stayed up with a dead GPU, so every following
//! request panicked the same way. [`fail`] is the one funnel that replaced
//! them.

use std::sync::atomic::{AtomicBool, Ordering};

/// Exit status this process uses when it gives up on a lost GPU device.
/// `75`/`EX_TEMPFAIL` from `sysexits.h` — "temporary failure, the caller is
/// invited to retry" — which is exactly the situation: the next process gets
/// a working device. `orangu-coordinator` names the same number (see its
/// `process::SERVER_EXIT_DEVICE_LOST`) to report the restart in those terms
/// rather than as an unexplained crash.
pub const EXIT_CODE: i32 = 75;

/// What a client is told. One sentence, no backtrace, and specific about
/// what to do next — the full detail goes to this server's own log (see
/// [`fail`]), which is where a diagnosis is actually made.
pub const CLIENT_MESSAGE: &str = "the server lost its GPU device (the graphics driver reset it) and is restarting; \
     retry in a moment";

/// How long the process keeps running after a loss is detected, so the
/// in-flight request's error event is actually written to its socket before
/// the listener goes away. The error is sent as soon as the panic unwinds
/// (microseconds), so this is slack, not a wait for anything in particular.
const EXIT_GRACE: std::time::Duration = std::time::Duration::from_secs(2);

/// Process-wide in a real run: one device, one answer, whichever thread
/// asks. Under `cargo test` it is per-thread instead — see [`set_lost`].
#[cfg(not(test))]
static LOST: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
thread_local! {
    static LOST: AtomicBool = const { AtomicBool::new(false) };
}

/// Whether the GPU device has been lost. Once true, never false again:
/// there is no recovery within this process.
pub fn is_lost() -> bool {
    #[cfg(not(test))]
    {
        LOST.load(Ordering::Acquire)
    }
    #[cfg(test)]
    {
        LOST.with(|lost| lost.load(Ordering::Acquire))
    }
}

/// Marks the device lost, answering whether this is the first time.
///
/// The flag is thread-local under `cfg(test)` and global otherwise. Tests
/// run concurrently in one process, and this one is a latch that real code
/// can never clear — so a test that exercises [`fail`] would otherwise flip
/// a switch every *other* test in the binary reads, including
/// `engine::generate`'s own "a panic reaches the client with its backtrace"
/// test, whose expected output is the exact opposite of a lost device's.
/// Per-thread keeps that test honest without weakening what the production
/// path actually does.
fn set_lost() -> bool {
    #[cfg(not(test))]
    {
        !LOST.swap(true, Ordering::AcqRel)
    }
    #[cfg(test)]
    {
        LOST.with(|lost| !lost.swap(true, Ordering::AcqRel))
    }
}

/// Whether a panic message is `wgpu` reporting that the device is gone.
///
/// Necessary because `wgpu` does not always hand a lost device back as an
/// `Err` for [`fail`] to funnel: `Device::poll` routes it through
/// `handle_error_fatal`, which **panics inside `wgpu` itself** —
/// `"Error in Device::poll: Validation Error / Caused by: Parent device is
/// lost"` — so the `Result` this code checks never arrives. Every `wgpu`
/// call made after the device dies ends the same way, which is what makes
/// the panic message the most complete detector available, not a fallback.
///
/// Matched case-insensitively on the wording `wgpu`/`wgpu-core` use for the
/// condition itself (`Parent device is lost`, `Device is lost`,
/// `DeviceLost`), not on any one call's name, so a loss surfacing through
/// `submit`, `create_buffer`, or a map behaves like one surfacing through
/// `poll`.
pub fn is_device_lost_message(message: &str) -> bool {
    let message = message.to_lowercase();
    message.contains("device is lost")
        || message.contains("device lost")
        || message.contains("devicelost")
}

/// Records a device loss seen only as a panic — [`is_device_lost_message`]
/// deciding — logging it and arming the exit exactly as [`fail`] does, but
/// **without panicking**: the caller is the panic hook, already inside one.
///
/// A no-op for every other panic, and for a loss already recorded (`fail`'s
/// own panic message trips this too, and must not report twice).
pub fn note_panic(message: &str) {
    if !is_device_lost_message(message) {
        return;
    }
    if set_lost() {
        report(
            "a wgpu call",
            "the device was reported lost by wgpu itself",
        );
    }
}

/// Records a lost device, logs `context`/`detail` for diagnosis, arms the
/// exit, and panics to unwind whatever request was in flight.
///
/// The panic is deliberate and is not the error report: it's how the
/// half-finished forward pass is abandoned. `engine::generate` catches it
/// (its `catch_unwind` around each request's blocking closure), sees
/// [`is_lost`], and sends the client [`CLIENT_MESSAGE`] instead of the
/// captured panic detail.
///
/// Only ever called from *our own* frames — never from inside a
/// `wgpu` `map_async` callback, which would unwind through `wgpu-core` while
/// it holds its internal locks. A callback records the failure instead
/// (`VulkanBackend`'s `MapWait`) and the waiter calls this.
///
/// The first caller wins: a submission whose buffers all fail their maps
/// calls this once per buffer, and one log line and one exit timer is what
/// that should produce.
pub fn fail(context: &str, detail: impl std::fmt::Display) -> ! {
    if set_lost() {
        report(context, &detail);
    }
    panic!("GPU device lost while {context}: {detail}");
}

/// Writes the one paragraph a lost device gets in this server's log, and
/// arms the exit. Shared by [`fail`] and [`note_panic`] so a loss reads the
/// same whether we caught it as an `Err` or as `wgpu`'s own panic.
fn report(context: &str, detail: impl std::fmt::Display) {
    eprintln!(
        "orangu-server: GPU device lost while {context}: {detail}\n\
         orangu-server: the graphics driver reset the device (check `dmesg` for a GPU \
         hang or reset); a lost device cannot be re-created in this process, so \
         orangu-server is exiting with status {EXIT_CODE} — orangu-coordinator (or any \
         supervisor) restarts it on the next request, with a fresh device."
    );
    arm_exit();
}

/// Exits the process after [`EXIT_GRACE`], from a thread of its own so the
/// caller can go on and unwind its request first.
///
/// Skipped entirely under `cargo test`: the GPU cross-check tests drive
/// these same readback paths, and a test that trips this should fail with a
/// panic like any other, not take the whole test binary — and every other
/// test in that binary — down with it.
fn arm_exit() {
    if cfg!(test) {
        return;
    }
    std::thread::spawn(|| {
        std::thread::sleep(EXIT_GRACE);
        std::process::exit(EXIT_CODE);
    });
}

/// Clears this thread's flag, so one test's deliberate loss cannot be seen
/// by the next test if `libtest` happens to run them on the same thread
/// (`--test-threads=1`). No production counterpart exists, and must not: a
/// real lost device is a latch.
#[cfg(test)]
pub(crate) fn reset_for_test() {
    LOST.with(|lost| lost.store(false, Ordering::Release));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact wording `wgpu` 30 panics with when the device is gone —
    /// taken from a real driver reset on this machine, not invented — must
    /// be recognized, whichever call surfaced it. `wgpu` does not hand this
    /// one back as an `Err`: `Device::poll` routes it through
    /// `handle_error_fatal`, which panics from inside `wgpu` itself, so the
    /// message is the only thing there is to match on.
    #[test]
    fn recognizes_wgpus_own_device_lost_panic() {
        assert!(is_device_lost_message(
            "Error in Device::poll: Validation Error\n\nCaused by:\n  Parent device is lost"
        ));
        assert!(is_device_lost_message("Device is lost"));
        assert!(is_device_lost_message("surface error: DeviceLost"));
        // Our own `fail` message, so the hook that sees it does not report a
        // second time.
        assert!(is_device_lost_message(
            "GPU device lost while reading back a fused layer's output: ..."
        ));
    }

    /// An ordinary panic is not a device loss, and must not take the
    /// process down or replace a caller's real error with the GPU sentence.
    #[test]
    fn ordinary_panics_are_not_device_losses() {
        for message in [
            "index out of bounds: the len is 3 but the index is 5",
            "called `Option::unwrap()` on a `None` value",
            "assertion failed: prompt fits in the context window",
            "attempt to divide by zero",
        ] {
            assert!(!is_device_lost_message(message), "{message}");
        }
    }

    /// `note_panic` marks the device lost for a `wgpu` fatal and does
    /// nothing for anything else — it is called from the panic hook, on
    /// *every* panic in the process.
    #[test]
    fn note_panic_marks_only_a_device_loss() {
        assert!(!is_lost());
        note_panic("called `Option::unwrap()` on a `None` value");
        assert!(
            !is_lost(),
            "an ordinary panic must not mark the device lost"
        );

        note_panic("Error in Device::poll: Validation Error: Parent device is lost");
        assert!(is_lost());
        reset_for_test();
    }

    /// `fail` panics (so the request unwinds) and leaves [`is_lost`] set (so
    /// `engine::generate` knows to send the clean message rather than the
    /// panic's own detail). The flag it sets is this thread's own — see
    /// [`set_lost`] for why that is what `cfg(test)` does.
    #[test]
    fn fail_panics_and_marks_the_device_lost() {
        assert!(!is_lost(), "this thread starts with a working device");
        let panicked = std::panic::catch_unwind(|| {
            fail("a unit test", "DEVICE_LOST_UNIT_TEST");
        });
        assert!(panicked.is_err());
        assert!(is_lost(), "the flag must survive the unwind");
        // The panic's own payload names the context and the detail, so a
        // log or a debug report says which readback died and why.
        let payload = panicked.unwrap_err();
        let message = payload
            .downcast_ref::<String>()
            .expect("fail panics with a formatted message");
        assert!(message.contains("a unit test"), "got: {message}");
        assert!(message.contains("DEVICE_LOST_UNIT_TEST"), "got: {message}");
        reset_for_test();
    }

    /// Only the first loss reports: a submission whose whole set of buffers
    /// fails calls `fail` once per buffer, and that must not print the same
    /// paragraph (or arm the exit) once per buffer.
    #[test]
    fn only_the_first_loss_reports() {
        assert!(set_lost(), "the first call claims the report");
        assert!(!set_lost(), "every later call is a no-op");
        assert!(!set_lost());
        reset_for_test();
    }

    /// The message a client sees says what happened and what to do, and
    /// carries no panic/backtrace wording at all.
    #[test]
    fn the_client_message_is_one_plain_sentence() {
        assert!(!CLIENT_MESSAGE.contains("panic"));
        assert!(!CLIENT_MESSAGE.contains("backtrace"));
        assert!(CLIENT_MESSAGE.contains("retry"));
    }
}
