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

/// The full `/build` request: the optimization profile plus an optional
/// build target, each backend mapping the target to its own native concept —
/// a Makefile rule for the make-driven backends, `--bin` for cargo, a goal
/// for Maven, a package path for Go (see [`build_output`]).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct BuildRequest {
    pub profile: BuildProfile,
    pub target: Option<String>,
}

impl BuildRequest {
    /// Parse the argument of `/build [debug|release] [<target>]`. The profile
    /// keyword and the target may come in either order (`/build docs release`
    /// works too); the profile defaults to `Release` when absent. Rejected
    /// (`None`) when two profiles or two targets are given, so `/build debug
    /// release` or `/build docs install` doesn't silently drop one.
    pub fn parse(arg: &str) -> Option<Self> {
        let mut profile: Option<BuildProfile> = None;
        let mut target: Option<String> = None;
        for token in arg.split_whitespace() {
            // `split_whitespace` never yields an empty token, so this only
            // matches the literal profile keywords.
            if let Some(parsed) = BuildProfile::parse(token) {
                if profile.replace(parsed).is_some() {
                    return None;
                }
            } else if target.replace(token.to_string()).is_some() {
                return None;
            }
        }
        Some(Self {
            profile: profile.unwrap_or_default(),
            target,
        })
    }
}

/// Run the workspace's build, streaming each output line through `sink`. An
/// optional `target` narrows the build to one backend-native target:
///
/// - cargo: a binary name (`cargo build --bin <target>`, tests scoped the
///   same way)
/// - CMake, Autotools, plain Makefile: a Makefile rule (`make <target>`)
/// - Meson: a compile target (`meson compile <target>`)
/// - Maven: a lifecycle phase or goal, replacing the default `package`
/// - Go: a package path, replacing the default `./...`
/// - Python: rejected — `pip install -e .` has no target concept
pub fn build_output(
    workspace: &Path,
    profile: BuildProfile,
    target: Option<&str>,
    compile_workers: usize,
    sink: &BuildSink,
) -> Result<()> {
    if workspace.join("Cargo.toml").exists() {
        rust_build(workspace, profile, target, compile_workers, sink)
    } else if workspace.join("CMakeLists.txt").exists() {
        cmake_build(workspace, profile, target, compile_workers, sink)
    } else if workspace.join("configure").exists() {
        autotools_build(workspace, profile, target, compile_workers, sink)
    } else if workspace.join("meson.build").exists() {
        meson_build(workspace, profile, target, compile_workers, sink)
    } else if workspace.join("pom.xml").exists() {
        java_build(workspace, profile, target, sink)
    } else if workspace.join("pyproject.toml").exists()
        || workspace.join("setup.py").exists()
        || workspace.join("setup.cfg").exists()
    {
        python_build(workspace, profile, target, sink)
    } else if workspace.join("go.mod").exists() {
        go_build(workspace, profile, target, compile_workers, sink)
    } else if makefile_path(workspace).is_some() {
        makefile_build(workspace, target, compile_workers, sink)
    } else {
        Err(anyhow!(
            "no supported project found (expected Cargo.toml, CMakeLists.txt, configure, meson.build, pom.xml, pyproject.toml, setup.py, setup.cfg, go.mod, or a Makefile)"
        ))
    }
}

/// The workspace's plain Makefile, if any — checked in GNU make's own lookup
/// order (`GNUmakefile`, `makefile`, `Makefile`). Only consulted as the last
/// resort in [`build_output`]: every managed build system above (CMake,
/// Autotools, Meson) generates or ships its own Makefile, so a bare one only
/// drives the build when nothing else claims the workspace.
pub fn makefile_path(workspace: &Path) -> Option<std::path::PathBuf> {
    ["GNUmakefile", "makefile", "Makefile"]
        .iter()
        .map(|name| workspace.join(name))
        .find(|path| path.is_file())
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

fn rust_build(
    workspace: &Path,
    profile: BuildProfile,
    target: Option<&str>,
    compile_workers: usize,
    sink: &BuildSink,
) -> Result<()> {
    let mut steps = BuildSteps::new(sink);
    steps.run("cargo fmt", make_cmd("cargo", &["fmt"], workspace))?;
    steps.run("cargo clippy", make_cmd("cargo", &["clippy"], workspace))?;

    let release_flag: &[&str] = match profile {
        BuildProfile::Debug => &[],
        BuildProfile::Release => &["--release"],
    };
    // A target names one binary: both the build and its tests are scoped to
    // it (`--bin <target>`), leaving the rest of the workspace untouched.
    let bin_flags: Vec<&str> = match target {
        Some(target) => vec!["--bin", target],
        None => Vec::new(),
    };
    // `0` means unused: omit the flag entirely and let Cargo pick its own
    // (already-parallel) default rather than forcing a job count on it.
    let jobs_arg = (compile_workers > 0).then(|| format!("--jobs={compile_workers}"));
    let mut build_args = vec!["build"];
    build_args.extend_from_slice(release_flag);
    build_args.extend_from_slice(&bin_flags);
    if let Some(arg) = jobs_arg.as_deref() {
        build_args.push(arg);
    }
    steps.run("cargo build", make_cmd("cargo", &build_args, workspace))?;

    let mut test_args = vec!["test"];
    test_args.extend_from_slice(release_flag);
    test_args.extend_from_slice(&bin_flags);
    if let Some(arg) = jobs_arg.as_deref() {
        test_args.push(arg);
    }
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
fn cmake_build(
    workspace: &Path,
    profile: BuildProfile,
    target: Option<&str>,
    compile_workers: usize,
    sink: &BuildSink,
) -> Result<()> {
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

    // `0` means unused: run a bare `make`, its own (serial) default.
    let jobs_arg = (compile_workers > 0).then(|| format!("-j{compile_workers}"));
    let mut make_args: Vec<&str> = Vec::new();
    if let Some(arg) = jobs_arg.as_deref() {
        make_args.push(arg);
    }
    // A target is one of the generated Makefile's rules (CMake emits one per
    // add_executable/add_library target, plus install and friends).
    if let Some(target) = target {
        make_args.push(target);
    }
    steps.run("make", make_cmd("make", &make_args, &build_dir))?;

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
fn meson_build(
    workspace: &Path,
    profile: BuildProfile,
    target: Option<&str>,
    compile_workers: usize,
    sink: &BuildSink,
) -> Result<()> {
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

    // `0` means unused: omit `-j` and let `meson compile` fall back to its
    // own (ninja) default.
    let jobs_arg = compile_workers.to_string();
    let mut compile_args = vec!["compile", "-C", "build"];
    if compile_workers > 0 {
        compile_args.push("-j");
        compile_args.push(&jobs_arg);
    }
    // A target is one of Meson's own compile targets (`meson compile
    // <target>`, e.g. an executable or library name from meson.build).
    if let Some(target) = target {
        compile_args.push(target);
    }
    steps.run("meson compile", make_cmd("meson", &compile_args, workspace))?;

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
fn autotools_build(
    workspace: &Path,
    profile: BuildProfile,
    target: Option<&str>,
    compile_workers: usize,
    sink: &BuildSink,
) -> Result<()> {
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

    // `0` means unused: run a bare `make`, its own (serial) default.
    let jobs_arg = (compile_workers > 0).then(|| format!("-j{compile_workers}"));
    let mut make_args: Vec<&str> = Vec::new();
    if let Some(arg) = jobs_arg.as_deref() {
        make_args.push(arg);
    }
    // A target is a rule in the generated Makefile (`make <target>`).
    if let Some(target) = target {
        make_args.push(target);
    }
    steps.run("make", make_cmd("make", &make_args, workspace))?;

    Ok(())
}

/// Plain-Makefile projects — the last-resort backend, only reached when no
/// managed build system (Cargo, CMake, Autotools, Meson, Maven, Python, Go)
/// claims the workspace (see [`build_output`]). Runs `make`, optionally with
/// one rule (`/build <target>` — the `make docs`/`make install` case). Plain
/// make has no universal debug/release convention, so the profile is
/// accepted but not mapped to anything.
fn makefile_build(
    workspace: &Path,
    target: Option<&str>,
    compile_workers: usize,
    sink: &BuildSink,
) -> Result<()> {
    let mut steps = BuildSteps::new(sink);
    c_format(workspace, &mut steps)?;

    // `0` means unused: run a bare `make`, its own (serial) default.
    let jobs_arg = (compile_workers > 0).then(|| format!("-j{compile_workers}"));
    let mut make_args: Vec<&str> = Vec::new();
    if let Some(arg) = jobs_arg.as_deref() {
        make_args.push(arg);
    }
    if let Some(target) = target {
        make_args.push(target);
    }
    steps.run("make", make_cmd("make", &make_args, workspace))?;

    Ok(())
}

fn java_build(
    workspace: &Path,
    profile: BuildProfile,
    target: Option<&str>,
    sink: &BuildSink,
) -> Result<()> {
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
    // Maven profile named "release" in the project's pom.xml. A target is a
    // Maven lifecycle phase or goal, replacing the default `package` (e.g.
    // `/build verify` runs `mvn verify`).
    let goal = target.unwrap_or("package");
    let mvn_args: &[&str] = match profile {
        BuildProfile::Debug => &[goal],
        BuildProfile::Release => &["-P", "release", goal],
    };
    steps.run(&format!("mvn {goal}"), make_cmd("mvn", mvn_args, workspace))?;

    Ok(())
}

/// Python projects (`pyproject.toml`, `setup.py`, or `setup.cfg` at the
/// workspace root, checked in that order). `pip install -e .` covers both the
/// initial install and picking up subsequent source changes, so it is the
/// whole build step; unlike the compiled backends there is no separate
/// debug/release artifact, so `profile` is accepted for signature parity with
/// `build_output`'s other branches but otherwise unused. There's no target
/// concept either, so a requested one is rejected rather than ignored.
fn python_build(
    workspace: &Path,
    _profile: BuildProfile,
    target: Option<&str>,
    sink: &BuildSink,
) -> Result<()> {
    if let Some(target) = target {
        return Err(anyhow!(
            "build targets are not supported for Python projects (got '{target}'); pip install -e . is the whole build"
        ));
    }
    let mut steps = BuildSteps::new(sink);
    steps.run(
        "pip install -e .",
        make_cmd("pip", &["install", "-e", "."], workspace),
    )?;
    Ok(())
}

/// Go projects (`go.mod` at the workspace root). Go has no separate
/// debug/release artifact; debug instead disables optimizations and inlining
/// (`-gcflags=all=-N -l`), mirroring the C backends' `-O0` so a debugger (e.g.
/// delve) can step through unoptimized code. Release passes no extra flags.
/// A target is a package path (e.g. `./cmd/server`), replacing the default
/// whole-module `./...`.
fn go_build(
    workspace: &Path,
    profile: BuildProfile,
    target: Option<&str>,
    compile_workers: usize,
    sink: &BuildSink,
) -> Result<()> {
    let mut steps = BuildSteps::new(sink);

    let gcflags_arg = match profile {
        BuildProfile::Debug => Some("-gcflags=all=-N -l".to_string()),
        BuildProfile::Release => None,
    };
    // `0` means unused: omit `-p` and let `go build` pick its own default
    // parallelism.
    let jobs_arg = (compile_workers > 0).then(|| format!("-p={compile_workers}"));

    let mut args = vec!["build"];
    if let Some(arg) = gcflags_arg.as_deref() {
        args.push(arg);
    }
    if let Some(arg) = jobs_arg.as_deref() {
        args.push(arg);
    }
    args.push(target.unwrap_or("./..."));
    steps.run("go build", make_cmd("go", &args, workspace))?;

    Ok(())
}

/// The build targets Tab completion offers for `/build` in this workspace:
/// cargo binary names for a Cargo project, Makefile rule names for anything
/// driven by a checked-in Makefile (plain make, and Autotools projects whose
/// generated Makefile is present). Best-effort and possibly empty — an
/// unlisted target can still be typed by hand.
pub fn completion_targets(workspace: &Path) -> Vec<String> {
    let mut targets = Vec::new();
    if workspace.join("Cargo.toml").exists() {
        targets.extend(cargo_bin_names(workspace));
    } else if let Some(makefile) = makefile_path(workspace)
        && let Ok(content) = std::fs::read_to_string(&makefile)
    {
        targets.extend(makefile_rule_names(&content));
    }
    targets
}

/// The rule names defined in a Makefile's `content`, in order of appearance:
/// lines starting a rule (`name:` at column 0), skipping recipes (tab
/// indented), variable assignments (`:=`), pattern rules (`%`), and the
/// special dot rules (`.PHONY` and friends).
pub fn makefile_rule_names(content: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in content.lines() {
        if line.starts_with(['\t', ' ', '#', '.']) {
            continue;
        }
        let Some(colon) = line.find(':') else {
            continue;
        };
        // `:=`, `::=` and `?=`-style lines are assignments, not rules.
        if line[colon..].starts_with(":=") || line[colon..].starts_with("::=") {
            continue;
        }
        // One line can declare several targets (`a b: deps`).
        for name in line[..colon].split_whitespace() {
            if !name.is_empty()
                && !name.contains(['%', '$', '='])
                && !names.iter().any(|existing| existing == name)
            {
                names.push(name.to_string());
            }
        }
    }
    names
}

/// The workspace's cargo binary names: `[[bin]] name = "..."` entries in
/// Cargo.toml, plus the auto-discovered `src/bin/<name>.rs` files and
/// `src/bin/<name>/main.rs` directories.
fn cargo_bin_names(workspace: &Path) -> Vec<String> {
    let mut names = Vec::new();
    if let Ok(manifest) = std::fs::read_to_string(workspace.join("Cargo.toml")) {
        let mut in_bin_section = false;
        for line in manifest.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('[') {
                in_bin_section = trimmed == "[[bin]]";
                continue;
            }
            if in_bin_section
                && let Some(rest) = trimmed.strip_prefix("name")
                && let Some(value) = rest.trim_start().strip_prefix('=')
            {
                let name = value.trim().trim_matches('"');
                if !name.is_empty() && !names.iter().any(|existing| existing == name) {
                    names.push(name.to_string());
                }
            }
        }
    }
    if let Ok(entries) = std::fs::read_dir(workspace.join("src").join("bin")) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = if path.is_dir() && path.join("main.rs").is_file() {
                path.file_name().map(|n| n.to_string_lossy().into_owned())
            } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
                path.file_stem().map(|n| n.to_string_lossy().into_owned())
            } else {
                None
            };
            if let Some(name) = name
                && !names.iter().any(|existing| existing == &name)
            {
                names.push(name);
            }
        }
    }
    names.sort();
    names
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

    #[test]
    fn build_request_parse_takes_profile_and_target_in_either_order() {
        use super::BuildRequest;

        assert_eq!(
            BuildRequest::parse(""),
            Some(BuildRequest {
                profile: BuildProfile::Release,
                target: None
            })
        );
        assert_eq!(
            BuildRequest::parse("debug"),
            Some(BuildRequest {
                profile: BuildProfile::Debug,
                target: None
            })
        );
        assert_eq!(
            BuildRequest::parse("docs"),
            Some(BuildRequest {
                profile: BuildProfile::Release,
                target: Some("docs".to_string())
            })
        );
        assert_eq!(
            BuildRequest::parse("debug docs"),
            BuildRequest::parse("docs debug"),
        );
        // Two profiles or two targets are rejected.
        assert_eq!(BuildRequest::parse("debug release"), None);
        assert_eq!(BuildRequest::parse("docs install"), None);
    }

    #[test]
    fn makefile_rule_names_skips_recipes_assignments_and_pattern_rules() {
        let content = "\
CC := gcc
VERSION ?= 1.0
.PHONY: all clean
all: build docs
\tmake -C src
%.o: %.c
\t$(CC) -c $<
clean install:
\trm -rf build
$(GENERATED): deps
";
        assert_eq!(
            super::makefile_rule_names(content),
            vec!["all", "clean", "install"]
        );
    }

    #[test]
    fn makefile_path_prefers_gnu_make_lookup_order() {
        // Compared case-insensitively: on a case-insensitive filesystem
        // (Windows, default macOS) the `makefile` probe matches a checked-in
        // `Makefile`, so the reported casing follows the probe, not the file.
        let name_of = |path: Option<std::path::PathBuf>| {
            path.and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_lowercase()))
        };
        let dir = tempfile::tempdir().expect("dir");
        assert_eq!(super::makefile_path(dir.path()), None);
        std::fs::write(dir.path().join("Makefile"), "all:\n").expect("write");
        assert_eq!(
            name_of(super::makefile_path(dir.path())),
            Some("makefile".to_string())
        );
        std::fs::write(dir.path().join("GNUmakefile"), "all:\n").expect("write");
        assert_eq!(
            name_of(super::makefile_path(dir.path())),
            Some("gnumakefile".to_string())
        );
    }

    #[test]
    fn completion_targets_reads_cargo_bins_and_makefile_rules() {
        // A cargo workspace: [[bin]] names plus src/bin entries.
        let dir = tempfile::tempdir().expect("dir");
        std::fs::write(
            dir.path().join("Cargo.toml"),
            "[package]\nname = \"demo\"\n\n[[bin]]\nname = \"demo-cli\"\npath = \"src/main.rs\"\n",
        )
        .expect("manifest");
        let bin_dir = dir.path().join("src").join("bin");
        std::fs::create_dir_all(bin_dir.join("helper")).expect("bin dir");
        std::fs::write(bin_dir.join("tool.rs"), "fn main() {}\n").expect("bin file");
        std::fs::write(bin_dir.join("helper").join("main.rs"), "fn main() {}\n")
            .expect("bin dir main");
        assert_eq!(
            super::completion_targets(dir.path()),
            vec!["demo-cli", "helper", "tool"]
        );

        // A plain-Makefile workspace: the rule names.
        let dir = tempfile::tempdir().expect("dir");
        std::fs::write(
            dir.path().join("Makefile"),
            "all: docs\n\ttrue\ndocs:\n\ttrue\n",
        )
        .expect("makefile");
        assert_eq!(super::completion_targets(dir.path()), vec!["all", "docs"]);
    }
}
