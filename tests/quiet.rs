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

//! `orangu -q`: what the flag promises is about the process's own streams, so
//! it is checked by running the process. Each case is paired with the same run
//! without `-q`, since "printed nothing" only means something next to a run
//! that printed something.

use std::path::Path;
use std::process::{Command, Output};

use tempfile::tempdir;

/// A configuration with one server, enough for the commands that never reach
/// it. The endpoint is a closed port on purpose: nothing here may talk to a
/// model.
const CONFIG: &str = "[orangu]\nserver = test-server\nmodel = test-model\n\n\
     [test-server]\nprovider = llama.cpp\nendpoint = http://127.0.0.1:1\n";

fn orangu(config: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_orangu"))
        .arg("--config")
        .arg(config)
        .args(args)
        .env("SHELL", "/bin/bash")
        .output()
        .expect("run orangu")
}

fn config_file() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempdir().expect("config dir");
    let path = dir.path().join("orangu.conf");
    std::fs::write(&path, CONFIG).expect("write config");
    (dir, path)
}

#[test]
fn quiet_silences_a_command_that_succeeds() {
    let (_dir, config) = config_file();

    let loud = orangu(&config, &["-p", "/help"]);
    assert!(loud.status.success(), "{loud:?}");
    assert!(!loud.stdout.is_empty(), "/help should print without -q");

    let quiet = orangu(&config, &["-q", "-p", "/help"]);
    assert!(quiet.status.success(), "{quiet:?}");
    assert!(
        quiet.stdout.is_empty() && quiet.stderr.is_empty(),
        "-q should print nothing: {quiet:?}"
    );
}

#[test]
fn quiet_never_hides_a_failure() {
    let (_dir, config) = config_file();

    let quiet = orangu(&config, &["-q", "-p", "/bogus"]);
    assert!(
        !quiet.status.success(),
        "an unknown command must exit non-zero"
    );
    assert!(quiet.stdout.is_empty(), "{quiet:?}");
    assert!(
        String::from_utf8_lossy(&quiet.stderr).contains("/bogus"),
        "the error must still name the command: {quiet:?}"
    );
}

#[test]
fn quiet_silences_the_completion_script_but_not_its_detection() {
    let (_dir, config) = config_file();

    let loud = orangu(&config, &["-s"]);
    assert!(loud.status.success(), "{loud:?}");
    assert!(!loud.stdout.is_empty(), "-s should print a script");

    let quiet = orangu(&config, &["-q", "-s"]);
    assert!(quiet.status.success(), "{quiet:?}");
    assert!(quiet.stdout.is_empty(), "{quiet:?}");

    // An undetectable shell is still an error, silenced or not.
    let unknown = Command::new(env!("CARGO_BIN_EXE_orangu"))
        .args(["-q", "-s"])
        .env("SHELL", "/usr/bin/nonesuch")
        .output()
        .expect("run orangu");
    assert!(!unknown.status.success(), "{unknown:?}");
}

#[test]
fn quiet_is_refused_where_there_is_nothing_to_silence() {
    let (_dir, config) = config_file();

    let interactive = orangu(&config, &["-q"]);
    assert!(!interactive.status.success(), "{interactive:?}");
    assert!(
        String::from_utf8_lossy(&interactive.stderr).contains("terminal interface"),
        "{interactive:?}"
    );

    let init = orangu(&config, &["-q", "-i"]);
    assert!(!init.status.success(), "{init:?}");
    assert!(
        String::from_utf8_lossy(&init.stderr).contains("--init"),
        "{init:?}"
    );
}
