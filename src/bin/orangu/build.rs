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

use anyhow::{Context, Result, anyhow};
use std::{
    io::{BufRead, BufReader, Read},
    path::Path,
    process::{Command, Stdio},
    thread,
};
use tokio::sync::mpsc::UnboundedSender;

/// Sink for streaming build output. Each sent string is one line that the
/// caller appends to the output window as soon as it arrives.
pub type BuildSink = UnboundedSender<String>;

/// Which optimization profile `/build` should invoke. Each backend maps this
/// to its own toolchain's native concept of a profile (a cargo flag, a CMake
/// cache variable, a Maven profile, ...); it is never inferred, only ever
/// read off this enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BuildProfile {
    Debug,
    #[default]
    Release,
}

impl BuildProfile {
    /// Parse the trimmed argument of `/build [debug|release]`. Empty defaults
    /// to `Release`; anything else unrecognized is rejected so a typo falls
    /// through to the "unknown command" error rather than silently building.
    pub fn parse(arg: &str) -> Option<Self> {
        match arg.trim().to_ascii_lowercase().as_str() {
            "" => Some(Self::default()),
            "debug" => Some(Self::Debug),
            "release" => Some(Self::Release),
            _ => None,
        }
    }
}

pub fn build_output(workspace: &Path, profile: BuildProfile, sink: &BuildSink) -> Result<()> {
    if workspace.join("Cargo.toml").exists() {
        rust_build(workspace, profile, sink)
    } else if workspace.join("CMakeLists.txt").exists() {
        cmake_build(workspace, profile, sink)
    } else if workspace.join("configure").exists() {
        autotools_build(workspace, profile, sink)
    } else if workspace.join("meson.build").exists() {
        meson_build(workspace, profile, sink)
    } else if workspace.join("pom.xml").exists() {
        java_build(workspace, profile, sink)
    } else if workspace.join("pyproject.toml").exists()
        || workspace.join("setup.py").exists()
        || workspace.join("setup.cfg").exists()
    {
        python_build(workspace, profile, sink)
    } else {
        Err(anyhow!(
            "no supported project found (expected Cargo.toml, CMakeLists.txt, configure, meson.build, pom.xml, pyproject.toml, setup.py, or setup.cfg)"
        ))
    }
}

fn make_cmd(program: &str, args: &[&str], cwd: &Path) -> Command {
    let mut cmd = Command::new(program);
    cmd.args(args);
    cmd.current_dir(cwd);
    cmd
}

/// Forward every line from a child pipe to the sink as it is produced. Also
/// used by `/shell` (see `shell_command.rs`) to stream a plain command's
/// output through the same sink type.
pub(crate) fn stream_pipe<R: Read>(pipe: R, sink: &BuildSink) {
    let reader = BufReader::new(pipe);
    for line in reader.lines() {
        match line {
            Ok(line) => {
                if sink.send(line).is_err() {
                    break;
                }
            }
            Err(_) => break,
        }
    }
}

struct BuildSteps<'a> {
    sink: &'a BuildSink,
    first: bool,
}

impl<'a> BuildSteps<'a> {
    fn new(sink: &'a BuildSink) -> Self {
        Self { sink, first: true }
    }

    fn emit(&self, line: impl Into<String>) {
        let _ = self.sink.send(line.into());
    }

    fn run(&mut self, label: &str, mut command: Command) -> Result<()> {
        if !self.first {
            self.emit("");
        }
        self.first = false;
        self.emit(format!("{label}:"));

        command.stdout(Stdio::piped());
        command.stderr(Stdio::piped());
        let mut child = command
            .spawn()
            .with_context(|| format!("failed to run {label}"))?;

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let out_handle = stdout.map(|pipe| {
            let sink = self.sink.clone();
            thread::spawn(move || stream_pipe(pipe, &sink))
        });
        let err_handle = stderr.map(|pipe| {
            let sink = self.sink.clone();
            thread::spawn(move || stream_pipe(pipe, &sink))
        });
        if let Some(handle) = out_handle {
            let _ = handle.join();
        }
        if let Some(handle) = err_handle {
            let _ = handle.join();
        }

        let status = child
            .wait()
            .with_context(|| format!("failed to wait for {label}"))?;
        if status.success() {
            Ok(())
        } else {
            Err(anyhow!("{label} failed"))
        }
    }
}

fn rust_build(workspace: &Path, profile: BuildProfile, sink: &BuildSink) -> Result<()> {
    let mut steps = BuildSteps::new(sink);
    steps.run("cargo fmt", make_cmd("cargo", &["fmt"], workspace))?;
    steps.run("cargo clippy", make_cmd("cargo", &["clippy"], workspace))?;

    let release_flag: &[&str] = match profile {
        BuildProfile::Debug => &[],
        BuildProfile::Release => &["--release"],
    };
    let mut build_args = vec!["build"];
    build_args.extend_from_slice(release_flag);
    steps.run("cargo build", make_cmd("cargo", &build_args, workspace))?;

    let mut test_args = vec!["test"];
    test_args.extend_from_slice(release_flag);
    steps.run("cargo test", make_cmd("cargo", &test_args, workspace))?;
    Ok(())
}

fn c_format(workspace: &Path, steps: &mut BuildSteps) -> Result<()> {
    if workspace.join("clang-format.sh").exists() {
        steps.run(
            "clang-format.sh",
            make_cmd("bash", &["clang-format.sh"], workspace),
        )?;
    }
    Ok(())
}

/// If the workspace root has an in-place `./configure`-style build, wipe it
/// with `make distclean`. Both a VPATH Autotools build and a Meson build of
/// the same source tree conflict with a pre-existing in-tree Autotools
/// configuration — PostgreSQL's own `meson.build` checks for exactly this and
/// refuses to proceed otherwise — so it is cleaned up first rather than left
/// to fail confusingly (or not fail at all, and silently mix stale state in).
///
/// Detected via `config.status` or a generated `GNUmakefile`, never a bare
/// `Makefile`: some projects (PostgreSQL included) check in a portable
/// `Makefile` stub that just re-execs GNU make, so its mere presence doesn't
/// mean `configure` has ever been run — and running `make distclean` against
/// that stub before configuring fails outright ("You need to run the
/// 'configure' program first").
fn clean_stale_autotools_build(workspace: &Path, steps: &mut BuildSteps) -> Result<()> {
    if workspace.join("config.status").exists() || workspace.join("GNUmakefile").exists() {
        steps.run(
            "make distclean",
            make_cmd("make", &["distclean"], workspace),
        )?;
    }
    Ok(())
}

/// The `CMAKE_BUILD_TYPE` an existing `build/` directory was last configured
/// with, read straight out of its `CMakeCache.txt` (`CMAKE_BUILD_TYPE:STRING=
/// Debug`, one `key:type=value` entry per line). `None` when the directory
/// has not been configured yet.
fn cmake_cached_build_type(build_dir: &Path) -> Option<String> {
    let cache = std::fs::read_to_string(build_dir.join("CMakeCache.txt")).ok()?;
    cache.lines().find_map(|line| {
        let (key, value) = line.split_once('=')?;
        if key.starts_with("CMAKE_BUILD_TYPE:") {
            Some(value.to_string())
        } else {
            None
        }
    })
}

/// A single `build/` directory is reused across both profiles (unlike
/// Autotools' in-place build, CMake keeps generated files out of the source
/// tree by convention, and there is nothing that stops one build directory
/// from serving both). Switching profiles reconfigures that same directory:
/// `cmake` happily updates `CMAKE_BUILD_TYPE` in an existing cache in place,
/// so — unlike Meson, which refuses to change a configured directory's build
/// type without its own `configure` subcommand — a plain `cmake ..
/// -DCMAKE_BUILD_TYPE=...` handles both the first configure and any later
/// profile switch. It only reruns when the cached build type differs from
/// the requested one, so repeat `/build`s in the same profile skip straight
/// to `make`.
fn cmake_build(workspace: &Path, profile: BuildProfile, sink: &BuildSink) -> Result<()> {
    let mut steps = BuildSteps::new(sink);
    c_format(workspace, &mut steps)?;

    let build_type = match profile {
        BuildProfile::Debug => "Debug",
        BuildProfile::Release => "Release",
    };
    let build_dir = workspace.join("build");
    if !build_dir.exists() {
        std::fs::create_dir(&build_dir)
            .with_context(|| format!("failed to create {}", build_dir.display()))?;
    }

    if cmake_cached_build_type(&build_dir).as_deref() != Some(build_type) {
        let build_type_arg = format!("-DCMAKE_BUILD_TYPE={build_type}");
        steps.run(
            "cmake",
            make_cmd("cmake", &["..", build_type_arg.as_str()], &build_dir),
        )?;
    }

    steps.run("make", make_cmd("make", &[], &build_dir))?;

    Ok(())
}

/// Meson projects (a `meson.build` at the workspace root, no `CMakeLists.txt`
/// or `configure` — Autotools takes priority when a project has both, e.g.
/// PostgreSQL mid-migration). Meson cannot build in place — it refuses to let
/// the build directory equal the source directory — so it gets a single
/// reused `build/` directory (Meson's own convention) rather than one per
/// profile; switching profiles reconfigures that same directory with `meson
/// configure`, which is a cheap no-op when the profile is unchanged and
/// triggers the right incremental rebuild when it isn't.
fn meson_build(workspace: &Path, profile: BuildProfile, sink: &BuildSink) -> Result<()> {
    let mut steps = BuildSteps::new(sink);
    c_format(workspace, &mut steps)?;
    clean_stale_autotools_build(workspace, &mut steps)?;

    let buildtype = match profile {
        BuildProfile::Debug => "debug",
        BuildProfile::Release => "release",
    };
    let buildtype_arg = format!("--buildtype={buildtype}");

    if !workspace.join("build").join("build.ninja").exists() {
        steps.run(
            "meson setup",
            make_cmd("meson", &["setup", "build", &buildtype_arg], workspace),
        )?;
    } else {
        steps.run(
            "meson configure",
            make_cmd("meson", &["configure", "build", &buildtype_arg], workspace),
        )?;
    }

    steps.run(
        "meson compile",
        make_cmd("meson", &["compile", "-C", "build"], workspace),
    )?;

    Ok(())
}

/// Autotools projects (a `configure` script at the workspace root, no
/// `CMakeLists.txt`). Built in place, like a plain `./configure && make`:
/// autotools has no separate build-type flag, and an out-of-tree VPATH build
/// does not mix safely with an in-tree one (GNU Make's VPATH search can pick
/// up stale headers/objects from whichever configuration is lying around,
/// producing confusing failures unrelated to the actual code). So instead of
/// building alongside any existing configuration, wipe it with `make
/// distclean` first, then reconfigure from scratch for the requested profile.
fn autotools_build(workspace: &Path, profile: BuildProfile, sink: &BuildSink) -> Result<()> {
    let mut steps = BuildSteps::new(sink);
    c_format(workspace, &mut steps)?;
    clean_stale_autotools_build(workspace, &mut steps)?;

    let opt_flags = match profile {
        BuildProfile::Debug => "-g -O0",
        BuildProfile::Release => "-O2",
    };
    let cflags_arg = format!("CFLAGS={opt_flags}");
    let cxxflags_arg = format!("CXXFLAGS={opt_flags}");
    // Invoked via `sh` rather than executed directly so a missing executable
    // bit on the checked-in script doesn't matter.
    steps.run(
        "configure",
        make_cmd(
            "sh",
            &["./configure", &cflags_arg, &cxxflags_arg],
            workspace,
        ),
    )?;

    steps.run("make", make_cmd("make", &[], workspace))?;

    Ok(())
}

fn java_build(workspace: &Path, profile: BuildProfile, sink: &BuildSink) -> Result<()> {
    let mut steps = BuildSteps::new(sink);

    let frontend_dir = workspace.join("src").join("frontend");
    if frontend_dir.exists() {
        let needs_install = !frontend_dir
            .join("node_modules")
            .join(".package-lock.json")
            .exists()
            || is_newer(
                &frontend_dir.join("package.json"),
                &frontend_dir.join("node_modules").join(".package-lock.json"),
            )
            || is_newer(
                &frontend_dir.join("package-lock.json"),
                &frontend_dir.join("node_modules").join(".package-lock.json"),
            );

        if needs_install {
            steps.run(
                "npm ci",
                make_cmd("npm", &["--prefix", "src/frontend", "ci"], workspace),
            )?;
        }

        steps.run(
            "npm run fix",
            make_cmd(
                "npm",
                &["--prefix", "src/frontend", "run", "fix"],
                workspace,
            ),
        )?;

        steps.run(
            "npm run check",
            make_cmd(
                "npm",
                &["--prefix", "src/frontend", "run", "check"],
                workspace,
            ),
        )?;
    }

    // Maven has no built-in debug/release axis, so this maps onto its own
    // profile activation: release packaging is expected to be defined as a
    // Maven profile named "release" in the project's pom.xml.
    let mvn_args: &[&str] = match profile {
        BuildProfile::Debug => &["package"],
        BuildProfile::Release => &["-P", "release", "package"],
    };
    steps.run("mvn package", make_cmd("mvn", mvn_args, workspace))?;

    Ok(())
}

/// Python projects (`pyproject.toml`, `setup.py`, or `setup.cfg` at the
/// workspace root, checked in that order). `pip install -e .` covers both the
/// initial install and picking up subsequent source changes, so it is the
/// whole build step; unlike the compiled backends there is no separate
/// debug/release artifact, so `profile` is accepted for signature parity with
/// `build_output`'s other branches but otherwise unused.
fn python_build(workspace: &Path, _profile: BuildProfile, sink: &BuildSink) -> Result<()> {
    let mut steps = BuildSteps::new(sink);
    steps.run(
        "pip install -e .",
        make_cmd("pip", &["install", "-e", "."], workspace),
    )?;
    Ok(())
}

fn is_newer(a: &Path, b: &Path) -> bool {
    let Ok(a_meta) = a.metadata() else {
        return false;
    };
    let Ok(b_meta) = b.metadata() else {
        return true;
    };
    let Ok(a_time) = a_meta.modified() else {
        return false;
    };
    let Ok(b_time) = b_meta.modified() else {
        return true;
    };
    a_time > b_time
}

#[cfg(test)]
mod tests {
    use super::{BuildProfile, cmake_cached_build_type};

    #[test]
    fn cmake_cached_build_type_reads_the_cache_entry() {
        let build_dir = tempfile::tempdir().expect("build dir");
        std::fs::write(
            build_dir.path().join("CMakeCache.txt"),
            "// comment\nCMAKE_BUILD_TYPE:STRING=Debug\nOTHER:BOOL=ON\n",
        )
        .expect("cache file");
        assert_eq!(
            cmake_cached_build_type(build_dir.path()),
            Some("Debug".to_string())
        );
    }

    #[test]
    fn cmake_cached_build_type_is_none_without_a_cache() {
        let build_dir = tempfile::tempdir().expect("build dir");
        assert_eq!(cmake_cached_build_type(build_dir.path()), None);
    }

    #[test]
    fn build_profile_parse_defaults_to_release() {
        assert_eq!(BuildProfile::parse(""), Some(BuildProfile::Release));
        assert_eq!(BuildProfile::parse("   "), Some(BuildProfile::Release));
        assert_eq!(BuildProfile::default(), BuildProfile::Release);
    }

    #[test]
    fn build_profile_parse_is_case_insensitive() {
        assert_eq!(BuildProfile::parse("debug"), Some(BuildProfile::Debug));
        assert_eq!(BuildProfile::parse("DEBUG"), Some(BuildProfile::Debug));
        assert_eq!(
            BuildProfile::parse(" Release "),
            Some(BuildProfile::Release)
        );
    }

    #[test]
    fn build_profile_parse_rejects_unknown_input() {
        assert_eq!(BuildProfile::parse("nightly"), None);
    }
}
