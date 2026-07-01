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
        c_build(workspace, profile, sink)
    } else if workspace.join("pom.xml").exists() {
        java_build(workspace, profile, sink)
    } else {
        Err(anyhow!(
            "no supported project found (expected Cargo.toml, CMakeLists.txt, or pom.xml)"
        ))
    }
}

fn make_cmd(program: &str, args: &[&str], cwd: &Path) -> Command {
    let mut cmd = Command::new(program);
    cmd.args(args);
    cmd.current_dir(cwd);
    cmd
}

/// Forward every line from a child pipe to the sink as it is produced.
fn stream_pipe<R: Read>(pipe: R, sink: &BuildSink) {
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

fn c_build(workspace: &Path, profile: BuildProfile, sink: &BuildSink) -> Result<()> {
    let mut steps = BuildSteps::new(sink);

    if workspace.join("clang-format.sh").exists() {
        steps.run(
            "clang-format.sh",
            make_cmd("bash", &["clang-format.sh"], workspace),
        )?;
    }

    // Each profile gets its own build directory so switching between debug
    // and release never reconfigures an existing CMakeCache.txt with a
    // mismatched CMAKE_BUILD_TYPE.
    let (dir_name, build_type) = match profile {
        BuildProfile::Debug => ("build-debug", "Debug"),
        BuildProfile::Release => ("build-release", "Release"),
    };
    let build_dir = workspace.join(dir_name);
    if !build_dir.exists() {
        std::fs::create_dir(&build_dir)
            .with_context(|| format!("failed to create {}", build_dir.display()))?;
    }

    if !build_dir.join("CMakeCache.txt").exists() {
        let build_type_arg = format!("-DCMAKE_BUILD_TYPE={build_type}");
        steps.run(
            "cmake",
            make_cmd("cmake", &["..", build_type_arg.as_str()], &build_dir),
        )?;
    }

    steps.run("make", make_cmd("make", &[], &build_dir))?;

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
    use super::BuildProfile;

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
