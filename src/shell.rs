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
/// - Named by its **absolute path** where one can be found, rather than left
///   to a `PATH` lookup — see [`windows_powershell`] for the failure that
///   forces this.
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
    #[cfg(windows)]
    {
        (
            windows_powershell(),
            &["-NoLogo", "-NonInteractive", "-Command"],
        )
    }
    #[cfg(not(windows))]
    {
        ("bash", &["-lc"])
    }
}

/// Where Windows PowerShell lives under a given `%SystemRoot%`.
///
/// Split out from the lookup below so the path itself is testable on any
/// platform — the part that can silently go wrong is this layout, not the
/// environment read around it.
#[cfg_attr(not(windows), allow(dead_code))]
fn windows_powershell_path(system_root: &std::path::Path) -> std::path::PathBuf {
    system_root
        .join("System32")
        .join("WindowsPowerShell")
        .join("v1.0")
        .join("powershell.exe")
}

/// The absolute path to Windows PowerShell, falling back to the bare name.
///
/// Spawning it as plain `powershell` makes the shell a `PATH` lookup, and
/// that is not as safe as it looks. Rust resolves a bare program name against
/// the parent executable's directory, `System32`, the Windows directory, and
/// `PATH` — deliberately *not* the working directory. PowerShell is not in
/// `System32` itself but in `System32\WindowsPowerShell\v1.0`, so `PATH` is
/// the only thing that finds it, and a `PATH` missing that one entry turns
/// every `/shell` invocation into "program not found" with nothing naming the
/// shell as the culprit.
///
/// That is not hypothetical: it is an intermittent failure on GitHub's
/// Windows runners, whose `PATH` is long enough to be at risk of truncation
/// and is appended to by each setup step. It surfaced twice on unrelated
/// commits, landing on a different test in `shell_command` each time, because
/// those tests race for the same lookup.
///
/// Resolved once — this is on the path of every shell command.
#[cfg(windows)]
fn windows_powershell() -> &'static str {
    use std::sync::OnceLock;
    static RESOLVED: OnceLock<String> = OnceLock::new();
    RESOLVED
        .get_or_init(|| {
            // `windir` as well as `SystemRoot`: both are standard, and a
            // stripped-down environment may carry only one.
            resolve_powershell(
                ["SystemRoot", "windir"]
                    .into_iter()
                    .filter_map(std::env::var_os),
            )
        })
        .as_str()
}

/// Picks the first `%SystemRoot%` in `roots` that actually holds PowerShell,
/// falling back to the bare name.
///
/// Separated from the environment read so both outcomes are testable on any
/// platform — the fallback is what a Windows machine would hit if this went
/// wrong, and it would look like nothing had changed until `PATH` betrayed
/// it again. Falling back rather than inventing a path keeps a machine with
/// PowerShell somewhere unusual working through `PATH`.
#[cfg_attr(not(windows), allow(dead_code))]
fn resolve_powershell(roots: impl IntoIterator<Item = std::ffi::OsString>) -> String {
    for root in roots {
        let candidate = windows_powershell_path(std::path::Path::new(&root));
        if candidate.is_file() {
            return candidate.to_string_lossy().into_owned();
        }
    }
    "powershell".to_string()
}

#[cfg(test)]
mod tests {
    use super::{command_parts, windows_powershell_path};
    use std::path::Path;

    /// Whichever platform this is built for, the parts have to name a real
    /// program and end with the flag that takes the command line — getting
    /// that wrong breaks every `/shell` invocation at once.
    #[test]
    fn command_parts_end_with_the_flag_that_takes_a_command() {
        let (program, args) = command_parts();
        assert!(!program.is_empty());
        let last = args.last().expect("at least one argument");
        if cfg!(windows) {
            // Either the resolved absolute path or the bare fallback — what
            // must not change is *which* shell, so match the file name
            // rather than pinning one of the two forms.
            let file = Path::new(program)
                .file_stem()
                .and_then(|s| s.to_str())
                .expect("a program name");
            assert_eq!(file, "powershell");
            assert_eq!(*last, "-Command");
        } else {
            assert_eq!(program, "bash");
            assert_eq!(*last, "-lc");
        }
    }

    /// The layout under `%SystemRoot%` is the part that can silently go
    /// wrong — a typo here would send every Windows shell command back to
    /// the `PATH` lookup this exists to avoid, and nothing would fail
    /// loudly enough to notice.
    #[test]
    fn windows_powershell_sits_under_system32() {
        let path = windows_powershell_path(Path::new("C:\\Windows"));
        let parts: Vec<_> = path
            .components()
            .map(|c| c.as_os_str().to_string_lossy().into_owned())
            .collect();
        let tail = &parts[parts.len() - 4..];
        assert_eq!(
            tail,
            ["System32", "WindowsPowerShell", "v1.0", "powershell.exe"]
        );
        // PowerShell is *not* in System32 itself, which is why a bare name
        // can only ever be found through PATH.
        assert_ne!(
            path.parent().and_then(|p| p.file_name()),
            Some("System32".as_ref())
        );
    }

    /// The whole point of the change: when PowerShell is where it should be,
    /// it is named absolutely so no `PATH` lookup is involved.
    #[test]
    fn a_present_powershell_resolves_to_its_absolute_path() {
        let root = tempfile::tempdir().expect("tempdir");
        let dir = root.path().join("System32/WindowsPowerShell/v1.0");
        std::fs::create_dir_all(&dir).expect("layout");
        std::fs::write(dir.join("powershell.exe"), b"").expect("stub");

        let resolved = super::resolve_powershell([root.path().as_os_str().to_os_string()]);
        assert_eq!(resolved, dir.join("powershell.exe").to_string_lossy());
        assert!(Path::new(&resolved).is_absolute());
    }

    /// A root that doesn't hold PowerShell must not be returned as if it
    /// did — the next candidate, then `PATH`, is what should happen.
    #[test]
    fn a_missing_powershell_falls_back_to_the_bare_name() {
        let empty = tempfile::tempdir().expect("tempdir");
        assert_eq!(
            super::resolve_powershell([empty.path().as_os_str().to_os_string()]),
            "powershell"
        );
        assert_eq!(super::resolve_powershell([]), "powershell");
    }

    /// `SystemRoot` first, `windir` second — a stripped-down environment may
    /// carry only one, and the earlier candidate wins when both exist.
    #[test]
    fn the_first_root_that_has_it_wins() {
        let empty = tempfile::tempdir().expect("tempdir");
        let real = tempfile::tempdir().expect("tempdir");
        let dir = real.path().join("System32/WindowsPowerShell/v1.0");
        std::fs::create_dir_all(&dir).expect("layout");
        std::fs::write(dir.join("powershell.exe"), b"").expect("stub");

        let resolved = super::resolve_powershell([
            empty.path().as_os_str().to_os_string(),
            real.path().as_os_str().to_os_string(),
        ]);
        assert_eq!(resolved, dir.join("powershell.exe").to_string_lossy());
    }

    /// A resolved path must still be spawnable as a program — an argument
    /// or a quoted string here would break every invocation.
    #[test]
    fn the_program_is_a_program_not_a_flag() {
        let (program, _) = command_parts();
        assert!(!program.starts_with('-'));
        assert!(!program.contains('"'));
    }

    /// The command line is appended after these, so none of them may be a
    /// placeholder the caller is expected to substitute.
    #[test]
    fn command_parts_are_all_flags() {
        let (_, args) = command_parts();
        assert!(args.iter().all(|arg| arg.starts_with('-')));
    }
}
