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

//! Bakes the commit this binary was built from into it, as
//! `ORANGU_BUILD_COMMIT` (read back through `orangu::build_info`).
//!
//! Named by `build = "src/build.rs"` in `Cargo.toml` rather than left at
//! Cargo's default `./build.rs`: no source lives in this repository's root.
//! Cargo still runs it from the package root, so the `.git` paths below are
//! relative to that and not to this file.
//!
//! It exists for one reason: a benchmark result has to say which build
//! produced it. A version number cannot — `1.2.0` is every build between two
//! releases, which during performance work is every build that matters. The
//! alternative is a human remembering to pass `--label`, and
//! `LESSONS-LEARNED.md` records what happens when they don't.

use std::process::Command;

fn main() {
    // Re-run when HEAD moves, and when the branch HEAD points at moves. Without
    // these, a rebuild after `git checkout` would keep the old commit baked in
    // — a stale provenance string is worse than none, because it is believed.
    println!("cargo:rerun-if-changed=.git/HEAD");
    if let Ok(head) = std::fs::read_to_string(".git/HEAD")
        && let Some(reference) = head.strip_prefix("ref: ")
    {
        println!("cargo:rerun-if-changed=.git/{}", reference.trim());
    }
    // A packager building from a tarball has no `.git` but does know the
    // commit; this is how they say so.
    println!("cargo:rerun-if-env-changed=ORANGU_BUILD_COMMIT");

    println!("cargo:rustc-env=ORANGU_BUILD_COMMIT={}", commit());
}

/// The short commit, with `-dirty` appended when tracked files differ from it.
///
/// `unknown` when there is no git and no override — a source tarball is a
/// legitimate way to build this, and it must not fail the build. It must also
/// not *claim* a commit: "unknown" is a usable answer, a wrong hash is not.
fn commit() -> String {
    if let Ok(injected) = std::env::var("ORANGU_BUILD_COMMIT")
        && !injected.trim().is_empty()
    {
        return injected.trim().to_string();
    }

    let Some(short) = git(&["rev-parse", "--short=9", "HEAD"]) else {
        return "unknown".to_string();
    };

    // Tracked changes only. An untracked file cannot reach the build without a
    // `mod` declaration somewhere, which is itself a tracked change — so
    // `diff` answers the question `status` would, without walking the tree.
    let dirty = Command::new("git")
        .args(["diff", "--quiet", "HEAD", "--"])
        .status()
        .map(|status| !status.success())
        .unwrap_or(false);

    if dirty {
        format!("{short}-dirty")
    } else {
        short
    }
}

fn git(args: &[&str]) -> Option<String> {
    let output = Command::new("git").args(args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?.trim().to_string();
    (!text.is_empty()).then_some(text)
}
