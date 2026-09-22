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

//! Which build this is: the release version, and the commit it was built
//! from (`build.rs`).
//!
//! Both, not either. The version dates the release; the commit identifies the
//! build — and during performance work every build worth telling apart shares
//! one version number. `orangu-server` reports both on `GET /props`, which is
//! how a benchmark result records what produced it without anyone having to
//! remember to say so.

/// The package version — `1.2.0`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The short commit this was built from, `-dirty` when tracked files differed
/// from it, or `unknown` when it could not be resolved. See `build.rs`.
///
/// `option_env!`, not `env!`, and that is the whole point: `env!` makes the
/// *build script* a hard dependency of the library compiling at all. A tree
/// without it — a vendored copy, a re-packaged source drop, or a checkout
/// where `build.rs` was left untracked — then fails with "environment variable
/// not defined" on every target at once, which is a build failure for a
/// provenance string. It is not worth that: an unresolved commit is a fact to
/// report, not a reason to refuse to compile.
pub const COMMIT: &str = match option_env!("ORANGU_BUILD_COMMIT") {
    Some(commit) => commit,
    None => "unknown",
};

/// The package name — `orangu`.
pub const NAME: &str = env!("CARGO_PKG_NAME");

/// The `rustc` version this was built with, or `unknown`. See `build.rs`.
pub const RUSTC: &str = match option_env!("ORANGU_BUILD_RUSTC") {
    Some(rustc) => rustc,
    None => "unknown",
};

/// The Cargo profile — `release`, `debug`, or `unknown`.
pub const PROFILE: &str = match option_env!("ORANGU_BUILD_PROFILE") {
    Some(profile) => profile,
    None => "unknown",
};

/// The target triple this was built for, or `unknown`.
pub const TARGET: &str = match option_env!("ORANGU_BUILD_TARGET") {
    Some(target) => target,
    None => "unknown",
};

/// `1.2.0 (52c0443ab)` — the two together, as one string for a banner, a
/// report header or a series label.
///
/// The commit is omitted rather than printed as `unknown`: a line that ends
/// "(unknown)" reads as a failure, when in fact it is an ordinary release
/// build from a source tarball, and the version alone is the whole truth
/// available about it.
pub fn id() -> String {
    if is_known() {
        format!("{VERSION} ({COMMIT})")
    } else {
        VERSION.to_string()
    }
}

/// Whether this build keeps frame pointers.
///
/// What it decides is how a profile of this process can be unwound: `perf
/// --call-graph fp` walks the frame-pointer chain and needs them, and a
/// build without them loses the call chain for most samples in the hot leaf,
/// which renders as a flamegraph of a process doing nothing. `dwarf` works
/// on either kind, from the unwind tables every build carries.
///
/// Reported rather than assumed because the flag is not part of the build
/// profile: stable cargo has no per-profile `rustflags`, so whether a
/// `release-with-debug` binary has them depends on how it was invoked. A
/// profiler that asks the process it is about to sample needs no convention
/// about which directory holds which kind of build — `orangu-server` answers
/// this on `GET /props` and `orangu-bench --flamegraph-call-graph auto` asks.
///
/// They are not simply always on because they are not free: measured on this
/// project's own hardware, the same source built with them is 9% slower at a
/// small model's decode and 5% at a prefill.
pub fn frame_pointers() -> bool {
    frame_pointers_in(option_env!("ORANGU_BUILD_RUSTFLAGS").unwrap_or_default())
}

/// [`frame_pointers`] over the flags cargo passed to `rustc`, as
/// `CARGO_ENCODED_RUSTFLAGS` holds them: the arguments separated by a unit
/// separator, so `-C force-frame-pointers=yes` may arrive either as one
/// argument or as `-C` followed by the rest.
///
/// Split out to be tested. The obvious reading — "an argument beginning
/// `-C`, then look at the next one" — reports every build as having frame
/// pointers, because `[target.'cfg(...)'] rustflags = ["-C",
/// "target-feature=..."]` in `.cargo/config.toml` is exactly that shape.
fn frame_pointers_in(flags: &str) -> bool {
    let args: Vec<&str> = flags.split('\u{1f}').filter(|a| !a.is_empty()).collect();
    let mut i = 0;
    while i < args.len() {
        // `-C name=value` and `-Cname=value` are the same flag to `rustc`.
        let flag = match args[i] {
            "-C" => {
                i += 1;
                args.get(i).copied().unwrap_or_default()
            }
            arg => arg.strip_prefix("-C").unwrap_or_default(),
        };
        i += 1;
        let Some(rest) = flag.strip_prefix("force-frame-pointers") else {
            continue;
        };
        // No value means yes, as it does to `rustc` itself.
        return match rest.strip_prefix('=') {
            None | Some("") => true,
            Some(value) => !matches!(value, "no" | "n" | "off" | "false"),
        };
    }
    false
}

#[cfg(test)]
mod frame_pointer_tests {
    use super::frame_pointers_in;

    /// The flag in every spelling cargo can deliver it in, and — the case
    /// that made this a function with a test — the flags this project
    /// actually builds with, which have a bare `-C` in them and no frame
    /// pointers at all.
    #[test]
    fn frame_pointers_are_read_from_the_flags_cargo_passed() {
        let sep = '\u{1f}';
        let joined = |args: &[&str]| args.join(&sep.to_string());

        assert!(frame_pointers_in(&joined(&["-Cforce-frame-pointers=yes"])));
        assert!(frame_pointers_in(&joined(&[
            "-C",
            "force-frame-pointers=yes"
        ])));
        assert!(frame_pointers_in(&joined(&["-Cforce-frame-pointers"])));
        assert!(frame_pointers_in(&joined(&[
            "-C",
            "target-feature=+avx2,+fma",
            "-C",
            "force-frame-pointers=yes",
        ])));

        assert!(!frame_pointers_in(""));
        assert!(!frame_pointers_in(&joined(&["-C", "target-feature=+avx2"])));
        assert!(!frame_pointers_in(&joined(&["-Ctarget-cpu=native"])));
        assert!(!frame_pointers_in(&joined(&[
            "-C",
            "force-frame-pointers=no"
        ])));
    }
}

/// Whether the commit is a real one. Kept as a function rather than left to
/// each caller to compare against the magic string.
pub fn is_known() -> bool {
    COMMIT != "unknown" && !COMMIT.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_build_identifies_itself() {
        assert!(!VERSION.is_empty());
        // Whatever `build.rs` resolved, it must be *something*: an empty
        // string would silently become an empty field in every report.
        assert!(!COMMIT.is_empty());
    }

    #[test]
    fn an_unknown_commit_is_left_out_rather_than_printed() {
        // The composition is what is being pinned here — a tarball build must
        // read "1.2.0", never "1.2.0 (unknown)".
        let id = id();
        assert!(id.starts_with(VERSION), "{id}");
        assert_eq!(id.contains('('), is_known(), "{id}");
        assert!(!id.contains("unknown"), "{id}");
    }
}
