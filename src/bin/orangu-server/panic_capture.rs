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

//! Stashes a panic's own message/location/backtrace where `engine::
//! generate::Engine::generate`'s `catch_unwind` (around a request's
//! `spawn_blocking` closure) can retrieve it on the *same* thread right
//! after catching the unwind — the only way to recover this detail at all,
//! since `tokio::task::JoinError`'s own `Display` for a panic is just an
//! opaque "task panicked" note, never the payload or a backtrace. Without
//! this, a panicking request degraded to that one-line note both in the
//! server's own log and (via `StreamEvent::Error`) the client — the web
//! UI's "download a debug report" feature (`web::mod::system_report`,
//! `app.js`'s Save button on an error bubble) needs the real detail to be
//! worth anything.
//!
//! A thread-local, not a global `Mutex`/channel: `std::panic::set_hook`'s
//! closure runs synchronously, on the panicking thread itself, before
//! unwinding starts — by the time `catch_unwind` on that *same* thread
//! returns, the stash is already populated and nothing else could have
//! raced to overwrite it (each `spawn_blocking` closure runs to completion
//! on whichever blocking-pool thread picked it up before that thread is
//! handed another task).

use std::cell::RefCell;

thread_local! {
    static LAST_PANIC: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// Installs the capturing hook, chained after whatever hook was already
/// registered (the default one, unless something else installed its own
/// first) so existing behavior — printing the panic to stderr — is
/// unchanged; this only adds the stash. Call once, as early as possible in
/// `main`, before any thread that might panic (in particular, any
/// `spawn_blocking` closure) starts.
pub fn install() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let location = info
            .location()
            .map(|l| l.to_string())
            .unwrap_or_else(|| "<unknown location>".to_string());
        let message = panic_payload_string(info.payload());
        // Before anything else: a panic that says the GPU device is gone is
        // this process's death notice, whichever thread it happened on.
        // `wgpu` does not always report a lost device as an `Err` the engine
        // can funnel through `device_lost::fail` — `Device::poll` panics from
        // inside `wgpu` itself — and a panic on a thread nothing catches
        // (a rayon worker, say) would otherwise leave the process up with a
        // dead GPU and no explanation. Recording it here means the request's
        // `catch_unwind` finds `device_lost::is_lost()` already set and
        // sends the caller one sentence, and the exit that lets the
        // supervisor restart on a fresh device is armed either way.
        crate::device_lost::note_panic(&message);
        let backtrace = std::backtrace::Backtrace::force_capture();
        let detail = format!("panicked at {location}:\n{message}\n\nbacktrace:\n{backtrace}");
        LAST_PANIC.with(|cell| *cell.borrow_mut() = Some(detail));
        previous(info);
    }));
}

fn panic_payload_string(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

/// Takes (clearing) whatever this thread's most recent panic stashed —
/// `Some` only when called on the same thread, shortly after a
/// `catch_unwind` around the code that panicked, before that thread runs
/// anything else that could panic and overwrite it. `None` for a normal
/// (non-panic) `Err`, or if [`install`] was never called.
pub fn take_last_panic_detail() -> Option<String> {
    LAST_PANIC.with(|cell| cell.borrow_mut().take())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hook's own message/backtrace stash, exercised directly (no
    /// `Engine::generate`/`spawn_blocking` involved — `engine::generate::
    /// tests::a_panic_during_generation_reaches_the_client_with_a_captured_
    /// backtrace` already covers the full pipeline end to end). Lets the
    /// deliberate panic print to stderr rather than trying to silence it —
    /// see that other test's own comment for why swapping `std::panic`'s
    /// process-global hook out for a silencing one is worse than the noise
    /// it would save.
    #[test]
    fn install_captures_the_panic_message_and_a_backtrace() {
        install();
        let result = std::panic::catch_unwind(|| {
            panic!("PANIC_CAPTURE_UNIT_TEST_PANIC");
        });
        assert!(result.is_err());

        let detail = take_last_panic_detail().expect("a panic just occurred on this thread");
        assert!(
            detail.contains("PANIC_CAPTURE_UNIT_TEST_PANIC"),
            "got: {detail}"
        );
        assert!(detail.contains("backtrace:"), "got: {detail}");
        assert!(detail.contains("panic_capture.rs"), "got: {detail}");
    }

    /// The hook recognizes `wgpu`'s own device-lost panic and marks the
    /// device lost, *without* the engine having caught anything.
    ///
    /// This wiring is the whole point: `wgpu`'s `Device::poll` panics from
    /// inside `wgpu` (`handle_error_fatal`) rather than returning the `Err`
    /// `device_lost::fail` funnels, so a real driver reset reached the
    /// client as a raw Rust backtrace and left the process up with a dead
    /// GPU — until this hook started reading the message. The panic text
    /// here is verbatim from a real reset.
    #[test]
    fn the_hook_recognizes_a_wgpu_device_lost_panic() {
        install();
        assert!(!crate::device_lost::is_lost());

        let result = std::panic::catch_unwind(|| {
            panic!(
                "Error in Device::poll: Validation Error\n\nCaused by:\n  Parent device is lost"
            );
        });
        assert!(result.is_err());
        assert!(
            crate::device_lost::is_lost(),
            "the hook must mark the device lost for the request's catch_unwind to find"
        );
        crate::device_lost::reset_for_test();
    }

    /// ...and leaves every other panic alone, which is most of them.
    #[test]
    fn the_hook_leaves_an_ordinary_panic_alone() {
        install();
        let _ = std::panic::catch_unwind(|| {
            panic!("PANIC_CAPTURE_ORDINARY_TEST_PANIC");
        });
        assert!(!crate::device_lost::is_lost());
    }

    /// `take_last_panic_detail` clears the stash — a second call right
    /// after (no new panic in between) must find nothing, not the same
    /// detail again (which would silently attribute a *later* unrelated
    /// panic on this thread to an *earlier* one it didn't actually cause).
    #[test]
    fn take_last_panic_detail_clears_the_stash() {
        install();
        let _ = std::panic::catch_unwind(|| {
            panic!("PANIC_CAPTURE_UNIT_TEST_CLEAR_CHECK");
        });
        assert!(take_last_panic_detail().is_some());
        assert!(take_last_panic_detail().is_none());
    }
}
