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
/// from it, or `unknown` when built without git and without an override. See
/// `build.rs`.
pub const COMMIT: &str = env!("ORANGU_BUILD_COMMIT");

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
