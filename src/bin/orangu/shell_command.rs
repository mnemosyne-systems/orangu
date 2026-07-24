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
use std::{path::Path, process::Stdio, thread};

use crate::build::{BuildSink, stream_pipe};

/// Runs `command_line` through the platform's shell in `workspace`, streaming
/// stdout and stderr lines to `sink` as they arrive (see `/build`'s
/// `build_output`, which shares the same streaming plumbing). Which shell, and
/// why, is [`orangu::shell::command_parts`] — `bash -lc` on Unix, PowerShell
/// on Windows, both resolving the command the way the user's own terminal
/// would rather than against a fixed allow-list.
pub fn shell_output(workspace: &Path, command_line: &str, sink: &BuildSink) -> Result<()> {
    let (program, args) = orangu::shell::command_parts();
    let mut command = std::process::Command::new(program);
    command
        .args(args)
        .arg(command_line)
        .current_dir(workspace)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child = command
        .spawn()
        .with_context(|| format!("failed to run: {command_line}"))?;

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let out_handle = stdout.map(|pipe| {
        let sink = sink.clone();
        thread::spawn(move || stream_pipe(pipe, &sink))
    });
    let err_handle = stderr.map(|pipe| {
        let sink = sink.clone();
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
        .with_context(|| format!("failed to wait for: {command_line}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(anyhow!("command exited with {status}"))
    }
}

#[cfg(test)]
mod tests {
    use super::shell_output;
    use tokio::sync::mpsc::unbounded_channel;

    // The two shells share no syntax beyond the simplest commands, so the
    // *command* that provokes a behaviour is per-platform even though what is
    // asserted about it is not. `echo hello` and `exit 3` happen to mean the
    // same thing in both and are written inline below; these two do not.
    //
    // Writing through `[Console]::Error` rather than `Write-Error` keeps the
    // bytes raw: PowerShell decorates its own error stream with the offending
    // command and position, which is not what a caller streaming a command's
    // stderr wants to receive.
    #[cfg(unix)]
    const ECHO_TO_STDERR: &str = "echo oops 1>&2";
    #[cfg(windows)]
    const ECHO_TO_STDERR: &str = "[Console]::Error.WriteLine('oops')";

    #[cfg(unix)]
    const PRINT_CWD: &str = "pwd";
    #[cfg(windows)]
    const PRINT_CWD: &str = "(Get-Location).Path";

    #[test]
    fn shell_output_streams_stdout_lines() {
        let (tx, mut rx) = unbounded_channel::<String>();
        let workspace = std::env::temp_dir();
        shell_output(&workspace, "echo hello", &tx).expect("echo succeeds");
        drop(tx);

        let mut lines = Vec::new();
        while let Ok(line) = rx.try_recv() {
            lines.push(line);
        }
        assert_eq!(lines, vec!["hello".to_string()]);
    }

    #[test]
    fn shell_output_streams_stderr_lines_too() {
        let (tx, mut rx) = unbounded_channel::<String>();
        let workspace = std::env::temp_dir();
        shell_output(&workspace, ECHO_TO_STDERR, &tx).expect("echo succeeds");
        drop(tx);

        let mut lines = Vec::new();
        while let Ok(line) = rx.try_recv() {
            lines.push(line);
        }
        assert_eq!(lines, vec!["oops".to_string()]);
    }

    #[test]
    fn shell_output_runs_in_the_given_workspace() {
        let (tx, mut rx) = unbounded_channel::<String>();
        let workspace = std::env::temp_dir();
        shell_output(&workspace, PRINT_CWD, &tx).expect("pwd succeeds");
        drop(tx);

        let line = rx.try_recv().expect("pwd printed a line");
        // Canonicalize both sides: on macOS `/tmp` is itself a symlink to
        // `/private/tmp`, which `pwd` resolves but `temp_dir()` may not, and
        // on Windows `temp_dir()` may hand back an 8.3 short name
        // (`RUNNER~1`) where the shell prints the long one.
        let printed = std::fs::canonicalize(line.trim()).unwrap_or_else(|_| line.trim().into());
        let expected = std::fs::canonicalize(&workspace).unwrap_or(workspace);
        assert_eq!(printed, expected);
    }

    #[test]
    fn shell_output_errors_on_nonzero_exit() {
        let (tx, _rx) = unbounded_channel::<String>();
        let workspace = std::env::temp_dir();
        let err = shell_output(&workspace, "exit 3", &tx).expect_err("nonzero exit is an error");
        assert!(err.to_string().contains("exit"), "{err}");
    }
}
