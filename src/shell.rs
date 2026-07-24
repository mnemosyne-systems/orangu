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

//! Which shell runs a user-supplied command line, per platform.
//!
//! Shared by the two places that run one: `/shell` in the terminal client
//! (`bin/orangu/shell_command`) and the `run_shell_command` tool the model
//! calls (`tools`). They ran `bash` unconditionally, which simply is not
//! there on a stock Windows machine.

/// The shell to spawn and the arguments that precede the command line, so a
/// caller ends up running `<program> <args...> <command_line>`.
///
/// On Unix this is `bash -lc`. A *login* shell is deliberate: it resolves the
/// command against the user's full `$PATH` as their own terminal would, so
/// what runs here is whatever they could have typed themselves, rather than a
/// fixed allow-list.
///
/// On Windows it is `powershell -NoLogo -NonInteractive -Command`:
///
/// - `powershell`, not `pwsh` — Windows PowerShell ships with the operating
///   system, while PowerShell 7 is an optional install that many machines do
///   not have. Choosing the one that is always present matters more here than
///   the newer language version, since this only ever forwards a command line
///   the user wrote.
/// - The profile is *not* suppressed, mirroring `-l` above: a user's aliases
///   and functions are part of "what they could have typed".
/// - `-NonInteractive` so a command that would prompt fails outright instead
///   of blocking forever on input that nothing is attached to answer — this
///   runs under an agent or a pipe, never a live console.
///
/// Note that the two shells share no syntax beyond the simplest commands, so
/// a command line written for one will not generally run on the other. This
/// picks the shell; it does not translate anything.
pub fn command_parts() -> (&'static str, &'static [&'static str]) {
    if cfg!(windows) {
        ("powershell", &["-NoLogo", "-NonInteractive", "-Command"])
    } else {
        ("bash", &["-lc"])
    }
}

#[cfg(test)]
mod tests {
    use super::command_parts;

    /// Whichever platform this is built for, the parts have to name a real
    /// program and end with the flag that takes the command line — getting
    /// that wrong breaks every `/shell` invocation at once.
    #[test]
    fn command_parts_end_with_the_flag_that_takes_a_command() {
        let (program, args) = command_parts();
        assert!(!program.is_empty());
        let last = args.last().expect("at least one argument");
        if cfg!(windows) {
            assert_eq!(program, "powershell");
            assert_eq!(*last, "-Command");
        } else {
            assert_eq!(program, "bash");
            assert_eq!(*last, "-lc");
        }
    }

    /// The command line is appended after these, so none of them may be a
    /// placeholder the caller is expected to substitute.
    #[test]
    fn command_parts_are_all_flags() {
        let (_, args) = command_parts();
        assert!(args.iter().all(|arg| arg.starts_with('-')));
    }
}
