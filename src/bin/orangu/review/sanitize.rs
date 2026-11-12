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

//! The `/auto_review` sanitizer pass: build the reviewed C/C++ project with
//! AddressSanitizer, LeakSanitizer, and UndefinedBehaviorSanitizer, run its
//! tests, and turn every sanitizer report into a Memory finding.
//!
//! The project is copied into a private mirror in the repository's git
//! directory (`.git/orangu-sanitize/`), so the build never touches the
//! working tree. Every step runs with an allow-listed environment and a
//! deadline, and — when bubblewrap is available — inside a sandbox with no
//! network, a read-only view of the system, and only the mirror writable.
//! Nothing is downloaded, so the pass works the same in an air-gapped
//! environment.

use super::secrets::Severity;
use anyhow::{Context, Result, anyhow};
use std::collections::HashSet;
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Extensions of C and C++ sources and headers.
const C_EXTENSIONS: [&str; 10] = [
    "c", "h", "cc", "cpp", "cxx", "c++", "hh", "hpp", "hxx", "h++",
];

/// The sanitizers every sanitized translation unit is built with, one
/// option each (no commas, so every build system passes them through
/// unsplit). Beyond AddressSanitizer (with LeakSanitizer) and the
/// UndefinedBehaviorSanitizer default set: comparisons and subtractions of
/// pointers into different objects, floating-point division by zero, and
/// floating-point to integer conversions that overflow.
const SANITIZERS: [&str; 6] = [
    "-fsanitize=address",
    "-fsanitize=undefined",
    "-fsanitize=pointer-compare",
    "-fsanitize=pointer-subtract",
    "-fsanitize=float-divide-by-zero",
    "-fsanitize=float-cast-overflow",
];

/// Further compiler options: undefined behaviour stops the process (so a
/// test that hits it fails and its output is shown), and frames stay
/// readable.
const COMPILE_OPTIONS: [&str; 5] = [
    "-fno-sanitize-recover=all",
    "-fno-omit-frame-pointer",
    "-fno-common",
    "-g",
    "-O1",
];

/// C++ only: the standard library's own bounds and precondition checks, and
/// `std::vector` annotations so AddressSanitizer sees past-the-end accesses
/// into its spare capacity.
const CXX_OPTIONS: [&str; 2] = ["-D_GLIBCXX_ASSERTIONS", "-D_GLIBCXX_SANITIZE_VECTOR"];

/// The allocation functions the failure-injection shim wraps.
const WRAPPED_ALLOCATORS: [&str; 7] = [
    "malloc",
    "calloc",
    "realloc",
    "strdup",
    "strndup",
    "aligned_alloc",
    "posix_memalign",
];

/// The failure-injection shim, linked into every sanitized program through
/// the linker's `--wrap`: with `ORANGU_FAIL_ALLOC=<n>` set, the `n`th call
/// to a wrapped allocation function in the process fails as if memory ran
/// out; without it, every call goes straight through. It is built without
/// sanitizers, and the sanitizer runtime still serves the real allocations.
const ALLOC_SHIM: &str = r#"#include <errno.h>
#include <stddef.h>
#include <stdlib.h>
void *__real_malloc(size_t);
void *__real_calloc(size_t, size_t);
void *__real_realloc(void *, size_t);
char *__real_strdup(const char *);
char *__real_strndup(const char *, size_t);
void *__real_aligned_alloc(size_t, size_t);
int __real_posix_memalign(void **, size_t, size_t);
static long orangu_calls;
static long orangu_fail_at = -2;
static int orangu_fail(void) {
    if (orangu_fail_at == -2) {
        const char *at = getenv("ORANGU_FAIL_ALLOC");
        orangu_fail_at = at ? atol(at) : -1;
    }
    if (orangu_fail_at <= 0) return 0;
    if (__atomic_add_fetch(&orangu_calls, 1, __ATOMIC_RELAXED) != orangu_fail_at) return 0;
    errno = ENOMEM;
    return 1;
}
void *__wrap_malloc(size_t n) { return orangu_fail() ? NULL : __real_malloc(n); }
void *__wrap_calloc(size_t c, size_t n) { return orangu_fail() ? NULL : __real_calloc(c, n); }
void *__wrap_realloc(void *p, size_t n) { return orangu_fail() ? NULL : __real_realloc(p, n); }
char *__wrap_strdup(const char *s) { return orangu_fail() ? NULL : __real_strdup(s); }
char *__wrap_strndup(const char *s, size_t n) { return orangu_fail() ? NULL : __real_strndup(s, n); }
void *__wrap_aligned_alloc(size_t a, size_t n) { return orangu_fail() ? NULL : __real_aligned_alloc(a, n); }
int __wrap_posix_memalign(void **p, size_t a, size_t n) { return orangu_fail() ? ENOMEM : __real_posix_memalign(p, a, n); }
"#;

/// The compiler and linker flags of the sanitized build, each a
/// space-separated list.
#[derive(Clone, Debug, Default)]
struct BuildFlags {
    c: String,
    cxx: String,
    link: String,
}

impl BuildFlags {
    /// `shim`: the compiled failure-injection object, when there is one.
    fn new(shim: Option<&Path>) -> Self {
        let c: Vec<&str> = SANITIZERS
            .iter()
            .chain(COMPILE_OPTIONS.iter())
            .copied()
            .collect();
        let cxx: Vec<&str> = c.iter().chain(CXX_OPTIONS.iter()).copied().collect();
        let mut link: Vec<String> = SANITIZERS.iter().map(|flag| flag.to_string()).collect();
        if let Some(shim) = shim {
            link.extend(
                WRAPPED_ALLOCATORS
                    .iter()
                    .map(|name| format!("-Wl,--wrap={name}")),
            );
            // As a linker argument it is linked wherever the flags reach a
            // link, and ignored by `-c` compiles.
            link.push(format!("-Wl,{}", shim.display()));
        }
        Self {
            c: c.join(" "),
            cxx: cxx.join(" "),
            link: link.join(" "),
        }
    }
}

/// Sanitizer log directories kept from earlier runs.
const KEPT_LOG_RUNS: usize = 5;

/// How often a running step is checked for completion, cancellation, or its
/// deadline.
const POLL: Duration = Duration::from_millis(200);

/// Whether `path` is a C or C++ source or header.
pub(crate) fn is_c_source(path: &str) -> bool {
    path.rsplit_once('.')
        .is_some_and(|(_, ext)| C_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
}

/// The build systems the pass can drive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CBuild {
    CMake,
    Meson,
    Autotools,
    Make,
}

impl CBuild {
    fn name(self) -> &'static str {
        match self {
            CBuild::CMake => "CMake",
            CBuild::Meson => "Meson",
            CBuild::Autotools => "Autotools",
            CBuild::Make => "Make",
        }
    }
}

/// The build system of the project rooted at `dir`, in order of preference.
pub(crate) fn detect_c_build(dir: &Path) -> Option<CBuild> {
    if dir.join("CMakeLists.txt").is_file() {
        Some(CBuild::CMake)
    } else if dir.join("meson.build").is_file() {
        Some(CBuild::Meson)
    } else if dir.join("configure.ac").is_file() || dir.join("configure").is_file() {
        Some(CBuild::Autotools)
    } else {
        makefile(dir).map(|_| CBuild::Make)
    }
}

/// The makefile GNU make would read in `dir`.
fn makefile(dir: &Path) -> Option<PathBuf> {
    ["GNUmakefile", "makefile", "Makefile"]
        .iter()
        .map(|name| dir.join(name))
        .find(|path| path.is_file())
}

/// The test target a makefile defines: `check`, else `test`.
fn make_test_target(makefile: &str) -> Option<&'static str> {
    ["check", "test"].into_iter().find(|target| {
        makefile.lines().any(|line| {
            line.strip_prefix(target)
                .is_some_and(|rest| rest.trim_start().starts_with(':') && !rest.contains(":="))
        })
    })
}

/// One stack frame of a sanitizer report that has a source location.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SanitizerFrame {
    pub(crate) function: String,
    pub(crate) file: String,
    pub(crate) line: u32,
}

/// One error a sanitizer reported.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SanitizerReport {
    /// `AddressSanitizer`, `LeakSanitizer`, or `UndefinedBehaviorSanitizer`.
    pub(crate) tool: &'static str,
    /// The error class: `heap-buffer-overflow`, `direct leak`, or the
    /// undefined-behaviour message.
    pub(crate) kind: String,
    /// The access or size detail: `WRITE of size 4`, `40 byte(s) in 1
    /// object(s)`.
    pub(crate) detail: Option<String>,
    /// The frames with a source location, innermost first.
    pub(crate) frames: Vec<SanitizerFrame>,
    /// The log file the report was read from.
    pub(crate) log: PathBuf,
    /// The error appeared only when this allocation (counted per process)
    /// was made to fail: the code does not handle allocation failure there.
    pub(crate) alloc_failure: Option<u32>,
}

impl SanitizerReport {
    fn new(tool: &'static str, kind: String, log: &Path) -> Self {
        Self {
            tool,
            kind,
            detail: None,
            frames: Vec::new(),
            log: log.to_path_buf(),
            alloc_failure: None,
        }
    }

    /// The CWE the error class corresponds to, when there is a clear one.
    pub(crate) fn cwe(&self) -> Option<u32> {
        if self.alloc_failure.is_some() {
            return Some(690);
        }
        let kind = self.kind.as_str();
        let write = self
            .detail
            .as_deref()
            .is_some_and(|d| d.starts_with("WRITE"));
        Some(match kind {
            "heap-use-after-free" => 416,
            "double-free" => 415,
            "bad-free" | "alloc-dealloc-mismatch" | "new-delete-type-mismatch" => 762,
            "stack-use-after-return" | "stack-use-after-scope" => 562,
            "direct leak" | "indirect leak" => 401,
            "SEGV" => 476,
            "allocation-size-too-big" | "calloc-overflow" | "out-of-memory" => 789,
            "invalid-pointer-pair" => 469,
            _ if kind.ends_with("-overflow") || kind.ends_with("-underflow") => {
                if write {
                    787
                } else {
                    125
                }
            }
            _ if kind.contains("signed integer overflow") => 190,
            _ if kind.contains("null pointer") => 476,
            _ if kind.contains("shift") => 1335,
            _ if kind.contains("division by zero") => 369,
            _ if kind.contains("misaligned") => 704,
            _ if kind.contains("insufficient space") => {
                if kind.starts_with("store") {
                    787
                } else {
                    125
                }
            }
            _ if kind.contains("outside the range of representable values") => 681,
            _ => return None,
        })
    }

    /// Leaks found in passing are reported for confirmation; every other
    /// sanitizer error is undefined behaviour and rejects the file.
    pub(crate) fn severity(&self) -> Severity {
        if self.kind == "indirect leak" {
            Severity::Medium
        } else {
            Severity::High
        }
    }
}

/// A frame line: `    #3 0x55d0c1 in copy_name /src/name.c:12:5`. The
/// function may contain spaces (a C++ signature); the location is the last
/// token. Frames without a source location (`(/lib64/libc.so.6+0x2a1)`)
/// return `None`.
fn parse_frame(line: &str) -> Option<Option<SanitizerFrame>> {
    let rest = line.trim_start().strip_prefix('#')?;
    let (number, rest) = rest.split_once(' ')?;
    if number.is_empty() || !number.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let rest = rest.trim_start();
    if !rest.starts_with("0x") {
        return None;
    }
    let Some((_, after)) = rest.split_once(" in ") else {
        return Some(None);
    };
    let Some((function, location)) = after.trim_end().rsplit_once(' ') else {
        return Some(None);
    };
    let mut parts = location.split(':');
    let file = parts.next().unwrap_or_default();
    let line_number = parts.next().and_then(|line| line.parse::<u32>().ok());
    match line_number {
        Some(line_number) if !file.is_empty() && !file.starts_with('(') => {
            Some(Some(SanitizerFrame {
                function: function.to_string(),
                file: file.to_string(),
                line: line_number,
            }))
        }
        _ => Some(None),
    }
}

/// `text` with every hexadecimal address (`0x7be91f5e0020`) replaced by
/// `<address>`, so the same error from two runs reads the same.
fn without_addresses(text: &str) -> String {
    text.split(' ')
        .map(|word| {
            let digits = word.strip_prefix("0x").unwrap_or("");
            if !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_hexdigit()) {
                "<address>"
            } else {
                word
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// An UndefinedBehaviorSanitizer line: `src/a.c:7:9: runtime error: <message>`.
fn parse_runtime_error(line: &str) -> Option<(SanitizerFrame, String)> {
    let (location, message) = line.split_once(": runtime error: ")?;
    let mut parts = location.rsplitn(3, ':');
    let _column = parts.next()?;
    let line_number = parts.next()?.parse::<u32>().ok()?;
    let file = parts.next()?.trim();
    if file.is_empty() {
        return None;
    }
    let mut message = without_addresses(message.trim());
    if message.chars().count() > 120 {
        message = message.chars().take(117).collect::<String>() + "...";
    }
    Some((
        SanitizerFrame {
            function: String::new(),
            file: file.to_string(),
            line: line_number,
        },
        message,
    ))
}

/// The reports in one sanitizer log file. Only the first stack of each
/// report is kept — the stack of the faulting access, or of a leak's
/// allocation — not the `allocated by` / `freed by` histories after it.
pub(crate) fn parse_sanitizer_log(text: &str, log: &Path) -> Vec<SanitizerReport> {
    let mut reports = Vec::new();
    let mut current: Option<SanitizerReport> = None;
    // Whether the current report's first stack has started, and ended.
    let mut in_stack = false;
    let mut stack_done = false;
    fn start(
        report: SanitizerReport,
        current: &mut Option<SanitizerReport>,
        reports: &mut Vec<SanitizerReport>,
    ) {
        if let Some(done) = current.replace(report) {
            reports.push(done);
        }
    }
    for raw in text.lines() {
        let line = raw.trim_end();
        if let Some((_, rest)) = line.split_once("ERROR: AddressSanitizer: ") {
            let kind = rest.split(" on ").next().unwrap_or(rest).trim();
            let kind = kind.strip_prefix("attempting ").unwrap_or(kind);
            let kind = kind.split_whitespace().next().unwrap_or(kind);
            start(
                SanitizerReport::new("AddressSanitizer", kind.to_string(), log),
                &mut current,
                &mut reports,
            );
            (in_stack, stack_done) = (false, false);
            continue;
        }
        if line.contains("ERROR: LeakSanitizer:") {
            // The leaks themselves follow, one block each.
            if let Some(done) = current.take() {
                reports.push(done);
            }
            continue;
        }
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed
            .strip_prefix("Direct leak of ")
            .map(|rest| ("direct leak", rest))
            .or_else(|| {
                trimmed
                    .strip_prefix("Indirect leak of ")
                    .map(|rest| ("indirect leak", rest))
            })
        {
            let (kind, detail) = rest;
            let mut report = SanitizerReport::new("LeakSanitizer", kind.to_string(), log);
            report.detail = Some(
                detail
                    .trim_end_matches(" allocated from:")
                    .trim()
                    .to_string(),
            );
            start(report, &mut current, &mut reports);
            (in_stack, stack_done) = (false, false);
            continue;
        }
        if let Some((frame, message)) = parse_runtime_error(line) {
            let mut report = SanitizerReport::new("UndefinedBehaviorSanitizer", message, log);
            report.frames.push(frame);
            start(report, &mut current, &mut reports);
            // Its stack starts at its first frame, after any `note:` lines.
            (in_stack, stack_done) = (false, false);
            continue;
        }
        let Some(report) = current.as_mut() else {
            continue;
        };
        if trimmed.starts_with("SUMMARY: ") {
            stack_done = true;
            continue;
        }
        if report.detail.is_none()
            && (trimmed.starts_with("READ of size") || trimmed.starts_with("WRITE of size"))
        {
            let detail = trimmed.split(" at ").next().unwrap_or(trimmed);
            report.detail = Some(detail.to_string());
            continue;
        }
        if report.detail.is_none()
            && let Some(access) = trimmed
                .strip_prefix("The signal is caused by a ")
                .and_then(|rest| rest.strip_suffix(" memory access."))
        {
            report.detail = Some(access.to_string());
            continue;
        }
        match parse_frame(line) {
            Some(frame) if !stack_done => {
                in_stack = true;
                if let Some(frame) = frame {
                    // An UndefinedBehaviorSanitizer report's location line
                    // has no function; its stack's first frame does.
                    if let [only] = report.frames.as_mut_slice()
                        && only.function.is_empty()
                        && only.line == frame.line
                    {
                        only.function = frame.function;
                        continue;
                    }
                    report.frames.push(frame);
                }
            }
            Some(_) => {}
            None => {
                if in_stack {
                    stack_done = true;
                }
            }
        }
    }
    if let Some(done) = current {
        reports.push(done);
    }
    reports
}

/// `path` with `.` and `..` components resolved lexically.
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            other => out.push(other),
        }
    }
    out
}

/// A frame's file as a repository-relative path, when it is inside the
/// mirrored sources. A relative frame path is resolved against the build
/// directory, where the compiler ran.
pub(crate) fn project_path(file: &str, sources: &Path, build: &Path) -> Option<String> {
    let path = Path::new(file);
    let absolute = if path.is_absolute() {
        normalize(path)
    } else {
        normalize(&build.join(path))
    };
    let relative = absolute.strip_prefix(sources).ok()?;
    let relative = relative.to_str()?.replace('\\', "/");
    (!relative.is_empty()).then_some(relative)
}

/// What one sanitizer run produced.
#[derive(Debug, Default)]
pub(crate) struct SanitizeOutcome {
    /// One-line notes on what ran and how it ended, for the report's
    /// `Overall` section.
    pub(crate) notes: Vec<String>,
    /// Every report, deduplicated by error class and location, with how many
    /// times each was seen.
    pub(crate) reports: Vec<(SanitizerReport, usize)>,
    /// The mirrored sources, to resolve frame paths against.
    pub(crate) sources: PathBuf,
    /// The build directory, to resolve relative frame paths against.
    pub(crate) build: PathBuf,
    /// The repository root, to show log paths relative to it.
    pub(crate) repo: PathBuf,
}

/// The settings of one run.
pub(crate) struct SanitizeRequest {
    pub(crate) workspace: PathBuf,
    pub(crate) timeout: Duration,
    pub(crate) cancel: Arc<AtomicBool>,
    pub(crate) progress: Arc<Mutex<String>>,
    /// How many failure-injection reruns of the tests: rerun `n` fails the
    /// `n`th allocation of every test process. `0` turns injection off.
    pub(crate) alloc_failures: u32,
}

fn set_progress(progress: &Mutex<String>, text: impl Into<String>) {
    if let Ok(mut progress) = progress.lock() {
        *progress = text.into();
    }
}

fn git(dir: &Path, args: &[&str]) -> Result<Vec<u8>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .stdin(Stdio::null())
        .output()
        .context("failed to run git")?;
    if !output.status.success() {
        return Err(anyhow!(
            "git {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(output.stdout)
}

fn git_line(dir: &Path, args: &[&str]) -> Result<String> {
    Ok(String::from_utf8_lossy(&git(dir, args)?).trim().to_string())
}

/// Bring `mirror` in line with the repository's tracked and untracked (not
/// ignored) files. A file whose size and modification time already match is
/// left alone, and copies keep the source's modification time, so the build
/// in the mirror stays incremental. Files listed in the previous run's
/// manifest but gone from the repository are removed; anything else in the
/// mirror (build output) is kept.
fn sync_mirror(repo: &Path, mirror: &Path, manifest: &Path) -> Result<usize> {
    let listing = git(repo, &["ls-files", "-z", "-co", "--exclude-standard"])?;
    let files: Vec<String> = listing
        .split(|byte| *byte == 0)
        .filter(|name| !name.is_empty())
        .map(|name| String::from_utf8_lossy(name).into_owned())
        .collect();
    let current: HashSet<&str> = files.iter().map(String::as_str).collect();
    if let Ok(previous) = std::fs::read_to_string(manifest) {
        for old in previous.lines().filter(|old| !current.contains(old)) {
            let _ = std::fs::remove_file(mirror.join(old));
        }
    }
    let mut copied = 0usize;
    for name in &files {
        let source = repo.join(name);
        let target = mirror.join(name);
        let Ok(meta) = std::fs::symlink_metadata(&source) else {
            continue;
        };
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        if meta.file_type().is_symlink() {
            let link = std::fs::read_link(&source)?;
            if std::fs::read_link(&target).ok().as_deref() != Some(link.as_path()) {
                let _ = std::fs::remove_file(&target);
                std::os::unix::fs::symlink(&link, &target)?;
                copied += 1;
            }
            continue;
        }
        if !meta.is_file() {
            continue;
        }
        let modified = meta.modified().ok();
        if let Ok(existing) = std::fs::symlink_metadata(&target)
            && existing.is_file()
            && existing.len() == meta.len()
            && existing.modified().ok() == modified
        {
            continue;
        }
        let _ = std::fs::remove_file(&target);
        std::fs::copy(&source, &target)
            .with_context(|| format!("failed to copy {name} into the sanitizer mirror"))?;
        if let Some(modified) = modified {
            std::fs::File::options()
                .write(true)
                .open(&target)
                .and_then(|file| file.set_modified(modified))
                .ok();
        }
        copied += 1;
    }
    std::fs::write(manifest, files.join("\n"))?;
    Ok(copied)
}

/// Whether bubblewrap can create a sandbox with no network here.
fn bubblewrap_works() -> bool {
    Command::new("bwrap")
        .args([
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--unshare-net",
            "--die-with-parent",
            "true",
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success())
}

/// Why a step stopped short.
#[derive(Debug, PartialEq, Eq)]
enum StepError {
    Cancelled,
    TimedOut,
    Failed(String),
}

/// Runs the build and test steps: each in the sandbox (when available), with
/// the allow-listed environment, its output appended to a log file, and the
/// whole run bounded by one deadline.
struct Runner {
    /// The only directory the steps may write to.
    root: PathBuf,
    sandbox: bool,
    deadline: Instant,
    cancel: Arc<AtomicBool>,
    env: Vec<(String, String)>,
}

impl Runner {
    fn command(&self, dir: &Path, program: &str, args: &[String]) -> Command {
        let mut command = if self.sandbox {
            let mut command = Command::new("bwrap");
            let root = self.root.to_string_lossy().into_owned();
            command.args([
                "--ro-bind",
                "/",
                "/",
                "--dev",
                "/dev",
                "--proc",
                "/proc",
                "--tmpfs",
                "/tmp",
                "--bind",
                &root,
                &root,
                "--unshare-net",
                "--unshare-pid",
                "--unshare-ipc",
                "--die-with-parent",
                "--new-session",
                "--chdir",
            ]);
            command.arg(dir).arg("--").arg(program).args(args);
            command
        } else {
            let mut command = Command::new(program);
            command.args(args).current_dir(dir);
            command
        };
        command.env_clear();
        command.envs(self.env.iter().map(|(k, v)| (k.as_str(), v.as_str())));
        command
    }

    /// Run one step; `Ok(false)` when it ran to completion but failed.
    fn run(
        &self,
        label: &str,
        dir: &Path,
        program: &str,
        args: &[String],
        extra_env: &[(String, String)],
        log: &Path,
    ) -> std::result::Result<bool, StepError> {
        use std::os::unix::process::CommandExt;
        let mut log = std::fs::File::options()
            .create(true)
            .append(true)
            .open(log)
            .map_err(|err| StepError::Failed(format!("cannot open the build log: {err}")))?;
        let _ = writeln!(log, "\n### {label}: {program} {}", args.join(" "));
        let stdout = log
            .try_clone()
            .map_err(|err| StepError::Failed(err.to_string()))?;
        let mut command = self.command(dir, program, args);
        command
            .envs(extra_env.iter().map(|(k, v)| (k.as_str(), v.as_str())))
            .stdin(Stdio::null())
            .stdout(stdout)
            .stderr(log)
            .process_group(0);
        let mut child = command
            .spawn()
            .map_err(|err| StepError::Failed(format!("cannot start {program}: {err}")))?;
        loop {
            match child.try_wait() {
                Ok(Some(status)) => return Ok(status.success()),
                Ok(None) => {}
                Err(err) => return Err(StepError::Failed(err.to_string())),
            }
            let cancelled = self.cancel.load(Ordering::Relaxed);
            if cancelled || Instant::now() >= self.deadline {
                // The step leads its own process group; take all of it.
                let pid = child.id() as libc::pid_t;
                // SAFETY: signalling a process group we created; no memory
                // is shared with it.
                unsafe {
                    libc::kill(-pid, libc::SIGKILL);
                }
                let _ = child.kill();
                let _ = child.wait();
                return Err(if cancelled {
                    StepError::Cancelled
                } else {
                    StepError::TimedOut
                });
            }
            std::thread::sleep(POLL);
        }
    }
}

/// Words for an argument list.
fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|arg| arg.to_string()).collect()
}

/// One build or test step: its label, directory, program, and arguments.
struct Step {
    label: &'static str,
    dir: PathBuf,
    program: String,
    args: Vec<String>,
}

fn step(label: &'static str, dir: &Path, program: impl Into<String>, args: Vec<String>) -> Step {
    Step {
        label,
        dir: dir.to_path_buf(),
        program: program.into(),
        args,
    }
}

/// The build steps for `build`, and its test step when it has one.
fn c_steps(
    build: CBuild,
    sources: &Path,
    build_dir: &Path,
    jobs: usize,
    flags: &BuildFlags,
) -> (Vec<Step>, Option<Step>) {
    let jobs = jobs.to_string();
    let src = sources.to_string_lossy().into_owned();
    let out = build_dir.to_string_lossy().into_owned();
    match build {
        CBuild::CMake => (
            vec![
                step(
                    "configure",
                    sources,
                    "cmake",
                    vec![
                        "-S".into(),
                        src,
                        "-B".into(),
                        out.clone(),
                        "-DCMAKE_BUILD_TYPE=Debug".into(),
                        format!("-DCMAKE_C_FLAGS={}", flags.c),
                        format!("-DCMAKE_CXX_FLAGS={}", flags.cxx),
                        format!("-DCMAKE_EXE_LINKER_FLAGS={}", flags.link),
                        format!("-DCMAKE_SHARED_LINKER_FLAGS={}", flags.link),
                        format!("-DCMAKE_MODULE_LINKER_FLAGS={}", flags.link),
                    ],
                ),
                step(
                    "build",
                    sources,
                    "cmake",
                    vec!["--build".into(), out.clone(), "-j".into(), jobs.clone()],
                ),
            ],
            Some(step(
                "test",
                sources,
                "ctest",
                vec![
                    "--test-dir".into(),
                    out,
                    "--output-on-failure".into(),
                    "-j".into(),
                    jobs,
                ],
            )),
        ),
        CBuild::Meson => {
            let mut setup = vec!["setup".to_string()];
            if build_dir.join("meson-private").is_dir() {
                setup.push("--reconfigure".into());
            }
            setup.extend([
                out.clone(),
                src,
                "-Dbuildtype=debug".into(),
                "-Db_lundef=false".into(),
                format!("-Dc_args={}", flags.c),
                format!("-Dcpp_args={}", flags.cxx),
                format!("-Dc_link_args={}", flags.link),
                format!("-Dcpp_link_args={}", flags.link),
            ]);
            (
                vec![
                    step("configure", sources, "meson", setup),
                    step(
                        "build",
                        sources,
                        "meson",
                        vec![
                            "compile".into(),
                            "-C".into(),
                            out.clone(),
                            "-j".into(),
                            jobs.clone(),
                        ],
                    ),
                ],
                Some(step(
                    "test",
                    sources,
                    "meson",
                    vec![
                        "test".into(),
                        "-C".into(),
                        out,
                        "--print-errorlogs".into(),
                        "--num-processes".into(),
                        jobs,
                    ],
                )),
            )
        }
        CBuild::Autotools => {
            let mut steps = Vec::new();
            if !sources.join("configure").is_file() {
                steps.push(step("bootstrap", sources, "autoreconf", args(&["-fi"])));
            }
            if !sources.join("config.status").is_file() {
                steps.push(step(
                    "configure",
                    sources,
                    sources.join("configure").to_string_lossy(),
                    vec![
                        format!("CFLAGS={}", flags.c),
                        format!("CXXFLAGS={}", flags.cxx),
                        format!("LDFLAGS={}", flags.link),
                    ],
                ));
            }
            steps.push(step(
                "build",
                sources,
                "make",
                vec!["-j".into(), jobs.clone()],
            ));
            (
                steps,
                Some(step(
                    "test",
                    sources,
                    "make",
                    vec!["-j".into(), jobs, "check".into()],
                )),
            )
        }
        CBuild::Make => {
            // The compiler carries the flags, so the makefile's own CFLAGS
            // (include paths, defines) still apply.
            let cc = format!("CC=cc {} {}", flags.c, flags.link);
            let cxx = format!("CXX=c++ {} {}", flags.cxx, flags.link);
            let build = step(
                "build",
                sources,
                "make",
                vec!["-j".into(), jobs.clone(), cc.clone(), cxx.clone()],
            );
            let target = makefile(sources)
                .and_then(|path| std::fs::read_to_string(path).ok())
                .and_then(|text| make_test_target(&text));
            let test = target.map(|target| {
                step(
                    "test",
                    sources,
                    "make",
                    vec!["-j".into(), jobs, cc, cxx, target.to_string()],
                )
            });
            (vec![build], test)
        }
    }
}

/// Remove all but the newest `KEPT_LOG_RUNS` log directories.
fn prune_logs(logs: &Path) {
    let Ok(entries) = std::fs::read_dir(logs) else {
        return;
    };
    let mut runs: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.is_dir())
        .collect();
    runs.sort();
    let excess = runs.len().saturating_sub(KEPT_LOG_RUNS);
    for old in runs.into_iter().take(excess) {
        let _ = std::fs::remove_dir_all(old);
    }
}

/// Run the sanitizer pass on the C/C++ project of `request.workspace`.
/// Never fails: anything that stops it is reported as a note instead.
pub(crate) fn run_sanitizers(request: SanitizeRequest) -> SanitizeOutcome {
    let mut outcome = SanitizeOutcome::default();
    if let Err(err) = run_c(&request, &mut outcome) {
        outcome
            .notes
            .push(format!("C/C++ sanitizers did not run: {err:#}"));
    }
    set_progress(&request.progress, "");
    outcome
}

fn run_c(request: &SanitizeRequest, outcome: &mut SanitizeOutcome) -> Result<()> {
    let workspace = &request.workspace;
    let repo = PathBuf::from(
        git_line(workspace, &["rev-parse", "--show-toplevel"])
            .context("the sanitizer pass needs a git repository")?,
    );
    let common = PathBuf::from(git_line(
        workspace,
        &["rev-parse", "--path-format=absolute", "--git-common-dir"],
    )?);
    let prefix = git_line(workspace, &["rev-parse", "--show-prefix"])?;
    let root = common.join("orangu-sanitize");
    let mirror = root.join("src");
    std::fs::create_dir_all(&mirror)?;
    std::fs::create_dir_all(root.join("home"))?;
    outcome.repo = repo.clone();

    set_progress(&request.progress, "copying the sources");
    sync_mirror(&repo, &mirror, &root.join("manifest"))?;

    // The project lives at the workspace if it has a build there, else at
    // the repository root.
    let (sources, build) = [mirror.join(prefix.trim_end_matches('/')), mirror.clone()]
        .into_iter()
        .find_map(|dir| detect_c_build(&dir).map(|build| (dir, build)))
        .ok_or_else(|| anyhow!("no CMake, Meson, Autotools, or Make build was found"))?;
    let build_dir = match build {
        CBuild::CMake | CBuild::Meson => root.join("build"),
        CBuild::Autotools | CBuild::Make => sources.clone(),
    };
    outcome.sources = mirror.clone();
    outcome.build = build_dir.clone();

    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_millis());
    let logs = root.join("logs");
    let run_logs = logs.join(format!("{stamp}"));
    let test_logs = run_logs.join("test");
    let build_logs = run_logs.join("build");
    std::fs::create_dir_all(&test_logs)?;
    std::fs::create_dir_all(&build_logs)?;
    prune_logs(&logs);
    let build_log = run_logs.join("build.log");

    let sandbox = bubblewrap_works();
    let jobs = std::thread::available_parallelism()
        .map_or(2, |n| n.get())
        .div_ceil(2)
        .max(1);
    let home = root.join("home").to_string_lossy().into_owned();
    let mut env: Vec<(String, String)> = vec![
        ("HOME".into(), home.clone()),
        ("XDG_CACHE_HOME".into(), format!("{home}/.cache")),
        ("TMPDIR".into(), "/tmp".into()),
        ("LANG".into(), "C.UTF-8".into()),
        ("CCACHE_DISABLE".into(), "1".into()),
    ];
    for key in ["PATH", "PKG_CONFIG_PATH", "USER"] {
        if let Ok(value) = std::env::var(key) {
            env.push((key.into(), value));
        }
    }
    let runner = Runner {
        root: root.clone(),
        sandbox,
        deadline: Instant::now() + request.timeout,
        cancel: Arc::clone(&request.cancel),
        env,
    };
    let where_ = if sandbox {
        "sandboxed, no network"
    } else {
        "not sandboxed: bubblewrap is unavailable"
    };
    let name = build.name();

    // The failure-injection shim. Without it the build still runs, just
    // without the injection reruns.
    let shim_dir = root.join("shim");
    std::fs::create_dir_all(&shim_dir)?;
    let shim_source = shim_dir.join("alloc_fail.c");
    let shim = shim_dir.join("alloc_fail.o");
    let mut shim_ready = false;
    if request.alloc_failures > 0 {
        std::fs::write(&shim_source, ALLOC_SHIM)?;
        set_progress(
            &request.progress,
            format!("C/C++ ({name}): allocation shim"),
        );
        shim_ready = matches!(
            runner.run(
                "allocation shim",
                &shim_dir,
                "cc",
                &args(&["-O2", "-fPIC", "-c", "alloc_fail.c", "-o", "alloc_fail.o"]),
                &[],
                &build_log,
            ),
            Ok(true)
        );
        if !shim_ready {
            outcome.notes.push(format!(
                "C/C++ sanitizers ({name}): the allocation-failure shim did not build, so \
                 allocation failures were not injected; see {}",
                display_path(&repo, &build_log)
            ));
        }
    }
    let flags = BuildFlags::new(shim_ready.then_some(shim.as_path()));
    let (steps, test) = c_steps(build, &sources, &build_dir, jobs, &flags);

    // Programs a configure step compiles are sanitized too; their reports go
    // to a directory nobody reads.
    for step in &steps {
        set_progress(&request.progress, format!("C/C++ ({name}): {}", step.label));
        let env = sanitizer_env(&build_logs, false);
        match runner.run(
            step.label,
            &step.dir,
            &step.program,
            &step.args,
            &env,
            &build_log,
        ) {
            Ok(true) => {}
            Ok(false) => {
                outcome.notes.push(format!(
                    "C/C++ sanitizers ({name}, {where_}): the {} step failed; see {}",
                    step.label,
                    display_path(&repo, &build_log)
                ));
                return Ok(());
            }
            Err(err) => {
                outcome
                    .notes
                    .push(step_error_note(name, step.label, &err, request));
                return Ok(());
            }
        }
    }
    let Some(test) = test else {
        outcome.notes.push(format!(
            "C/C++ sanitizers ({name}, {where_}): the project built, but it has no test \
             step to run, so nothing was executed"
        ));
        return Ok(());
    };

    // The baseline: the tests as they are. A test a sanitizer stops fails;
    // the reports say why, so a failing test step is not an error here.
    set_progress(&request.progress, format!("C/C++ ({name}): tests"));
    let test_log = run_logs.join("test.log");
    let env = sanitizer_env(&test_logs, true);
    let baseline = runner.run(
        test.label,
        &test.dir,
        &test.program,
        &test.args,
        &env,
        &test_log,
    );
    let baseline_reports = read_reports(&test_logs, &test_log);
    let known: HashSet<ReportKey> = baseline_reports
        .iter()
        .map(|report| report_key(report, outcome))
        .collect();
    merge_reports(baseline_reports, outcome);
    if let Err(err) = baseline {
        outcome
            .notes
            .push(step_error_note(name, "test", &err, request));
        return Ok(());
    }

    // Allocation-failure injection: rerun the tests with the `n`th
    // allocation of every process failing. An error that only appears then
    // is code that does not handle a failed allocation.
    let mut injected = 0u32;
    let mut stopped: Option<StepError> = None;
    if shim_ready {
        for n in 1..=request.alloc_failures {
            set_progress(
                &request.progress,
                format!(
                    "C/C++ ({name}): tests with allocation {n} of {} failing",
                    request.alloc_failures
                ),
            );
            let dir = run_logs.join(format!("alloc-fail-{n}"));
            std::fs::create_dir_all(&dir)?;
            let console = dir.join("test.log");
            let mut env = sanitizer_env(&dir, false);
            env.push(("ORANGU_FAIL_ALLOC".into(), n.to_string()));
            let result = runner.run(
                test.label,
                &test.dir,
                &test.program,
                &test.args,
                &env,
                &console,
            );
            let mut fresh: Vec<SanitizerReport> = read_reports(&dir, &console)
                .into_iter()
                .filter(|report| !known.contains(&report_key(report, outcome)))
                .collect();
            for report in &mut fresh {
                report.alloc_failure = Some(n);
            }
            merge_reports(fresh, outcome);
            if let Err(err) = result {
                stopped = Some(err);
                break;
            }
            injected = n;
        }
    }

    let found = outcome.reports.len();
    let mut note = format!(
        "C/C++ sanitizers ({name}, {where_}): the tests ran under AddressSanitizer, \
         LeakSanitizer, and UndefinedBehaviorSanitizer"
    );
    if injected > 0 {
        note.push_str(&format!(
            ", and again with each of the first {injected} allocations of every test process \
             failing"
        ));
    }
    note.push_str(&format!(
        "; {found} distinct error{} reported; logs in {}",
        if found == 1 { "" } else { "s" },
        display_path(&repo, &run_logs)
    ));
    outcome.notes.push(note);
    if let Some(err) = stopped {
        outcome.notes.push(step_error_note(
            name,
            "allocation-failure test",
            &err,
            request,
        ));
    }
    Ok(())
}

/// The sanitizer runtime options for a step whose reports go to `dir`.
/// Leak detection is off for build steps (a compiler probe that leaks is
/// not the project's problem).
fn sanitizer_env(dir: &Path, leaks: bool) -> Vec<(String, String)> {
    let dir = dir.to_string_lossy();
    vec![
        (
            "ASAN_OPTIONS".into(),
            format!(
                "log_path={dir}/asan:detect_leaks={}:abort_on_error=0:symbolize=1:\
                 detect_stack_use_after_return=1:strict_string_checks=1:\
                 check_initialization_order=1:strict_init_order=1:\
                 detect_invalid_pointer_pairs=1:alloc_dealloc_mismatch=1:\
                 new_delete_type_mismatch=1:detect_container_overflow=1:\
                 detect_odr_violation=2",
                u8::from(leaks)
            ),
        ),
        (
            "UBSAN_OPTIONS".into(),
            format!("log_path={dir}/ubsan:print_stacktrace=1"),
        ),
    ]
}

/// The note for a step that was cancelled, timed out, or could not start.
fn step_error_note(name: &str, label: &str, err: &StepError, request: &SanitizeRequest) -> String {
    match err {
        StepError::Cancelled => "C/C++ sanitizers were cancelled".to_string(),
        StepError::TimedOut => format!(
            "C/C++ sanitizers ({name}): stopped at the {label} step after the {} s limit \
             (review_sanitize_timeout); reports so far are included",
            request.timeout.as_secs()
        ),
        StepError::Failed(err) => {
            format!("C/C++ sanitizers ({name}): the {label} step could not run: {err}")
        }
    }
}

/// `path` relative to the repository root when it is inside it.
fn display_path(repo: &Path, path: &Path) -> String {
    path.strip_prefix(repo)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

/// The reports in every sanitizer log file in `dir`, and in the console
/// output `console` (where UndefinedBehaviorSanitizer writes when it runs
/// beside AddressSanitizer).
fn read_reports(dir: &Path, console: &Path) -> Vec<SanitizerReport> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|entries| {
            entries
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| path.is_file() && path.as_path() != console)
                .collect()
        })
        .unwrap_or_default();
    paths.sort();
    paths.push(console.to_path_buf());
    let mut reports = Vec::new();
    for path in paths {
        if let Ok(bytes) = std::fs::read(&path) {
            reports.extend(parse_sanitizer_log(&String::from_utf8_lossy(&bytes), &path));
        }
    }
    reports
}

/// What makes two reports the same error: the sanitizer, the error class,
/// and the innermost project frame.
type ReportKey = (String, String, String, u32);

fn report_key(report: &SanitizerReport, outcome: &SanitizeOutcome) -> ReportKey {
    let location = report
        .frames
        .iter()
        .find(|frame| project_path(&frame.file, &outcome.sources, &outcome.build).is_some())
        .or(report.frames.first())
        .map(|frame| (frame.file.clone(), frame.line))
        .unwrap_or_default();
    (
        report.tool.to_string(),
        report.kind.clone(),
        location.0,
        location.1,
    )
}

/// Add `reports` to `outcome.reports`, counting repeats of the same error
/// (the first one seen is kept).
fn merge_reports(reports: Vec<SanitizerReport>, outcome: &mut SanitizeOutcome) {
    let mut keys: Vec<ReportKey> = outcome
        .reports
        .iter()
        .map(|(existing, _)| report_key(existing, outcome))
        .collect();
    for report in reports {
        let key = report_key(&report, outcome);
        if let Some(index) = keys.iter().position(|existing| *existing == key) {
            outcome.reports[index].1 += 1;
        } else {
            keys.push(key);
            outcome.reports.push((report, 1));
        }
    }
}

/// One sanitizer report as a review finding.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SanitizerFinding {
    /// The repository-relative file and line the finding is recorded at:
    /// the innermost frame in a reviewed file, else the innermost frame in
    /// the project, else none.
    pub(crate) path: Option<String>,
    pub(crate) line: Option<u32>,
    /// Whether `path` is one of the reviewed files.
    pub(crate) in_review: bool,
    pub(crate) severity: Severity,
    /// `(<severity>) <tool> <kind> (CWE-n): <detail> in `<function>`; ...`
    pub(crate) text: String,
}

/// Turn `outcome`'s reports into findings. An error whose stack passes
/// through a reviewed file is recorded there at its own severity; one that
/// never does is recorded at its innermost project frame as `medium`, since
/// the change may only have exposed it.
pub(crate) fn sanitizer_findings(
    outcome: &SanitizeOutcome,
    reviewed: &HashSet<String>,
) -> Vec<SanitizerFinding> {
    outcome
        .reports
        .iter()
        .map(|(report, count)| {
            let located: Vec<(String, &SanitizerFrame)> = report
                .frames
                .iter()
                .filter_map(|frame| {
                    project_path(&frame.file, &outcome.sources, &outcome.build)
                        .map(|path| (path, frame))
                })
                .collect();
            let in_review = located.iter().find(|(path, _)| reviewed.contains(path));
            let (place, in_review_file) = match in_review {
                Some(hit) => (Some(hit), true),
                None => (located.first(), false),
            };
            let severity = if in_review_file {
                report.severity()
            } else {
                Severity::Medium
            };
            let mut text = match report.alloc_failure {
                Some(n) => format!(
                    "({}) Allocation failure not handled: with allocation {n} of the process \
                     failing, {} reports {}",
                    severity.label(),
                    report.tool,
                    report.kind
                ),
                None => format!("({}) {} {}", severity.label(), report.tool, report.kind),
            };
            if let Some(cwe) = report.cwe() {
                text.push_str(&format!(" (CWE-{cwe})"));
            }
            if let Some(detail) = &report.detail {
                text.push_str(&format!(": {detail}"));
            }
            if let Some((_, frame)) = place
                && !frame.function.is_empty()
            {
                text.push_str(&format!(" in `{}`", frame.function));
            }
            if !in_review_file {
                text.push_str(" (outside the reviewed files)");
            }
            if *count > 1 {
                text.push_str(&format!(", seen {count} times"));
            }
            text.push_str(&format!(
                "; full report in {}",
                display_path(&outcome.repo, &report.log)
            ));
            SanitizerFinding {
                path: place.map(|(path, _)| path.clone()),
                line: place.map(|(_, frame)| frame.line),
                in_review: in_review_file,
                severity,
                text,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ASAN: &str = "\
=================================================================
==4242==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000014 at pc 0x55 bp 0x7f sp 0x7f
WRITE of size 4 at 0x602000000014 thread T0
    #0 0x55d0c1 in copy_name /m/src/lib/name.c:12:5
    #1 0x55d0d2 in main /m/src/tests/name_test.c:7:3
    #2 0x7f12 in __libc_start_main (/lib64/libc.so.6+0x2a1)

0x602000000014 is located 0 bytes after 4-byte region
allocated by thread T0 here:
    #0 0x7f in malloc (/lib64/libasan.so.8+0xf1)
    #1 0x55d0e3 in main /m/src/tests/name_test.c:6:14

SUMMARY: AddressSanitizer: heap-buffer-overflow /m/src/lib/name.c:12:5 in copy_name
";

    #[test]
    fn an_asan_report_keeps_its_faulting_stack_only() {
        let reports = parse_sanitizer_log(ASAN, Path::new("asan.4242"));
        assert_eq!(reports.len(), 1);
        let report = &reports[0];
        assert_eq!(report.tool, "AddressSanitizer");
        assert_eq!(report.kind, "heap-buffer-overflow");
        assert_eq!(report.detail.as_deref(), Some("WRITE of size 4"));
        assert_eq!(report.frames.len(), 2);
        assert_eq!(report.frames[0].function, "copy_name");
        assert_eq!(report.frames[0].file, "/m/src/lib/name.c");
        assert_eq!(report.frames[0].line, 12);
        assert_eq!(report.cwe(), Some(787));
    }

    #[test]
    fn leaks_and_undefined_behaviour_are_separate_reports() {
        let log = "\
==9==ERROR: LeakSanitizer: detected memory leaks

Direct leak of 40 byte(s) in 1 object(s) allocated from:
    #0 0x7f in malloc (/lib64/libasan.so.8+0xf1)
    #1 0x55 in make_list /m/src/list.c:7:10

Indirect leak of 8 byte(s) in 1 object(s) allocated from:
    #0 0x7f in malloc (/lib64/libasan.so.8+0xf1)
    #1 0x55 in push /m/src/list.c:20:3

SUMMARY: AddressSanitizer: 48 byte(s) leaked in 2 allocation(s).
";
        let reports = parse_sanitizer_log(log, Path::new("asan.9"));
        assert_eq!(reports.len(), 2);
        assert_eq!(reports[0].kind, "direct leak");
        assert_eq!(
            reports[0].detail.as_deref(),
            Some("40 byte(s) in 1 object(s)")
        );
        assert_eq!(reports[0].frames[0].function, "make_list");
        assert_eq!(reports[0].severity(), Severity::High);
        assert_eq!(reports[1].severity(), Severity::Medium);

        let ub = "/m/src/calc.c:9:14: runtime error: signed integer overflow: 2147483647 + 1 cannot be represented in type 'int'\n    #0 0x55 in add /m/src/calc.c:9:14\n";
        let reports = parse_sanitizer_log(ub, Path::new("ubsan.3"));
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].tool, "UndefinedBehaviorSanitizer");
        assert_eq!(reports[0].frames[0].file, "/m/src/calc.c");
        assert_eq!(reports[0].frames[0].line, 9);
        assert_eq!(reports[0].cwe(), Some(190));
    }

    #[test]
    fn a_double_free_and_a_cpp_frame_parse() {
        let log = "==1==ERROR: AddressSanitizer: attempting double-free on 0x6 in thread T0:\n    #0 0x7f in operator delete(void*) (/lib64/libasan.so.8+0x1)\n    #1 0x55 in Store::drop(int, char*) /m/src/store.cpp:44:5\n";
        let reports = parse_sanitizer_log(log, Path::new("asan.1"));
        assert_eq!(reports[0].kind, "double-free");
        assert_eq!(reports[0].frames.len(), 1);
        assert_eq!(reports[0].frames[0].function, "Store::drop(int, char*)");
        assert_eq!(reports[0].cwe(), Some(415));
    }

    #[test]
    fn frames_map_to_repository_paths() {
        let sources = Path::new("/m/src");
        let build = Path::new("/m/build");
        assert_eq!(
            project_path("/m/src/lib/name.c", sources, build).as_deref(),
            Some("lib/name.c")
        );
        assert_eq!(
            project_path("../src/lib/name.c", sources, build).as_deref(),
            Some("lib/name.c")
        );
        assert_eq!(project_path("/usr/include/stdio.h", sources, build), None);
    }

    #[test]
    fn findings_land_on_the_first_reviewed_frame() {
        let report = parse_sanitizer_log(
            ASAN,
            Path::new("/repo/.git/orangu-sanitize/logs/1/test/asan.4242"),
        )
        .remove(0);
        let outcome = SanitizeOutcome {
            reports: vec![(report, 2)],
            sources: PathBuf::from("/m/src"),
            build: PathBuf::from("/m/build"),
            repo: PathBuf::from("/repo"),
            ..Default::default()
        };
        let reviewed: HashSet<String> = ["tests/name_test.c".to_string()].into();
        let findings = sanitizer_findings(&outcome, &reviewed);
        assert_eq!(findings[0].path.as_deref(), Some("tests/name_test.c"));
        assert_eq!(findings[0].line, Some(7));
        assert!(findings[0].in_review);
        assert_eq!(findings[0].severity, Severity::High);
        assert_eq!(
            findings[0].text,
            "(high) AddressSanitizer heap-buffer-overflow (CWE-787): WRITE of size 4 in `main`, \
             seen 2 times; full report in .git/orangu-sanitize/logs/1/test/asan.4242"
        );
        // Outside the reviewed files it is recorded at the innermost
        // project frame, as medium.
        let findings = sanitizer_findings(&outcome, &HashSet::new());
        assert_eq!(findings[0].path.as_deref(), Some("lib/name.c"));
        assert_eq!(findings[0].severity, Severity::Medium);
        assert!(findings[0].text.contains("outside the reviewed files"));
    }

    #[test]
    fn make_test_targets_are_found() {
        assert_eq!(
            make_test_target("all:\n\tcc x.c\ncheck: all\n\t./t\n"),
            Some("check")
        );
        assert_eq!(make_test_target("test:\n\t./t\n"), Some("test"));
        assert_eq!(make_test_target("check := yes\nall:\n"), None);
        assert!(is_c_source("src/a.c") && is_c_source("inc/b.HPP") && !is_c_source("x.rs"));
    }

    /// A C project in a fresh git repository, with `files` committed.
    fn c_project(files: &[(&str, &str)]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("tempdir");
        for (name, content) in files {
            let path = dir.path().join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, content).unwrap();
        }
        for args in [
            &["init", "-q"][..],
            &["add", "."],
            &[
                "-c",
                "user.name=t",
                "-c",
                "user.email=t@t",
                "commit",
                "-qm",
                "c",
            ],
        ] {
            git(dir.path(), args).expect("git");
        }
        dir
    }

    fn tool_available(program: &str) -> bool {
        Command::new(program)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok_and(|status| status.success())
    }

    fn run(dir: &Path) -> SanitizeOutcome {
        run_sanitizers(SanitizeRequest {
            workspace: dir.to_path_buf(),
            timeout: Duration::from_secs(120),
            cancel: Arc::new(AtomicBool::new(false)),
            progress: Arc::new(Mutex::new(String::new())),
            alloc_failures: 4,
        })
    }

    const OVERFLOW: &str = "#include <stdlib.h>\n#include <string.h>\n\
        int fill(int n) {\n    int *v = malloc(4 * sizeof *v);\n\
            for (int i = 0; i <= n; i++) v[i] = i;\n    int r = v[0];\n\
            free(v);\n    return r;\n}\n";

    /// The whole pass on a Make project whose test writes past a heap
    /// buffer: built, run, reported, and placed on the reviewed file.
    #[test]
    fn a_make_project_overflow_is_found_end_to_end() {
        if !tool_available("cc") || !tool_available("make") {
            return;
        }
        let dir = c_project(&[
            ("fill.c", OVERFLOW),
            (
                "t.c",
                "int fill(int);\nint main(void) { return fill(4); }\n",
            ),
            (
                "Makefile",
                "t: t.c fill.c\n\t$(CC) -o t t.c fill.c\ncheck: t\n\t./t\n",
            ),
        ]);
        let outcome = run(dir.path());
        let reviewed: HashSet<String> = ["fill.c".to_string()].into();
        let findings = sanitizer_findings(&outcome, &reviewed);
        // The overflow, seen by the object-size check before the write lands
        // in AddressSanitizer's redzone — and, from the injection reruns, the
        // unchecked `malloc` the same line writes through.
        assert_eq!(findings.len(), 2, "{findings:?} {:?}", outcome.notes);
        assert!(findings.iter().all(|f| f.path.as_deref() == Some("fill.c")));
        assert!(findings.iter().all(|f| f.line == Some(5)));
        assert!(findings.iter().all(|f| f.severity == Severity::High));
        assert_eq!(
            findings[0].text.split("; full report").next().unwrap(),
            "(high) UndefinedBehaviorSanitizer store to address <address> with insufficient \
             space for an object of type 'int' (CWE-787) in `fill`"
        );
        assert!(
            findings[1].text.starts_with(
                "(high) Allocation failure not handled: with allocation 1 of the process \
                 failing, UndefinedBehaviorSanitizer reports store to null pointer"
            ),
            "{}",
            findings[1].text
        );
        // The working tree is untouched: the build ran in the mirror.
        assert!(!dir.path().join("t").exists());
        // A second run reuses the mirror and finds the same.
        assert_eq!(run(dir.path()).reports.len(), 2);
    }

    /// CMake with ctest: a leak and a signed overflow, both found, and the
    /// notes say the tests ran sandboxed when bubblewrap works.
    #[test]
    fn a_cmake_project_leak_and_overflow_are_found() {
        if !tool_available("cc") || !tool_available("cmake") || !tool_available("ctest") {
            return;
        }
        let dir = c_project(&[
            (
                "CMakeLists.txt",
                "cmake_minimum_required(VERSION 3.10)\nproject(p C)\nenable_testing()\n\
                 add_executable(t t.c lib.c)\nadd_test(NAME leak COMMAND t leak)\n\
                 add_test(NAME ub COMMAND t ub)\n",
            ),
            (
                "lib.c",
                "#include <stdlib.h>\nint *keep(void) { return malloc(16); }\n\
                 int add(int a, int b) { return a + b; }\n",
            ),
            (
                "t.c",
                "#include <limits.h>\n#include <string.h>\nint *keep(void);\n\
                 int add(int, int);\nint main(int argc, char **argv) {\n\
                     if (argc > 1 && strcmp(argv[1], \"leak\") == 0) return keep() == 0;\n\
                     return add(INT_MAX, 1) == 0;\n}\n",
            ),
        ]);
        let outcome = run(dir.path());
        let kinds: Vec<&str> = outcome
            .reports
            .iter()
            .map(|(report, _)| report.kind.as_str())
            .collect();
        assert!(
            kinds.contains(&"direct leak"),
            "{kinds:?} {:?}",
            outcome.notes
        );
        assert!(
            kinds
                .iter()
                .any(|kind| kind.starts_with("signed integer overflow")),
            "{kinds:?}"
        );
        let note = outcome.notes.last().expect("note");
        assert!(note.contains("(CMake, "), "{note}");
        assert_eq!(
            note.contains("sandboxed, no network"),
            bubblewrap_works(),
            "{note}"
        );
    }

    #[test]
    fn a_meson_project_use_after_free_is_found() {
        if !tool_available("cc") || !tool_available("meson") || !tool_available("ninja") {
            return;
        }
        let dir = c_project(&[
            (
                "meson.build",
                "project('p', 'c')\nt = executable('t', 't.c')\ntest('t', t)\n",
            ),
            (
                "t.c",
                "#include <stdlib.h>\nint main(void) {\n    int *p = malloc(4);\n\
                     free(p);\n    return *p;\n}\n",
            ),
        ]);
        let outcome = run(dir.path());
        let baseline: Vec<&SanitizerReport> = outcome
            .reports
            .iter()
            .map(|(report, _)| report)
            .filter(|report| report.alloc_failure.is_none())
            .collect();
        assert_eq!(baseline.len(), 1, "{:?}", outcome.notes);
        assert_eq!(baseline[0].kind, "heap-use-after-free");
        assert_eq!(baseline[0].cwe(), Some(416));
    }

    #[test]
    fn a_project_without_a_build_says_so() {
        let dir = c_project(&[("a.c", "int x;\n")]);
        let outcome = run(dir.path());
        assert!(outcome.reports.is_empty());
        assert!(outcome.notes[0].contains("no CMake, Meson, Autotools, or Make build"));
    }

    /// An unchecked `malloc` passes the plain test run and is found by the
    /// rerun that makes that allocation fail — and a checked one is not.
    #[test]
    fn an_unchecked_allocation_is_found_by_injection() {
        if !tool_available("cc") || !tool_available("make") {
            return;
        }
        let dir = c_project(&[
            (
                "name.c",
                "#include <stdlib.h>\n#include <string.h>\n\
                 char *copy(const char *s) {\n    char *p = malloc(strlen(s) + 1);\n\
                     strcpy(p, s);\n    return p;\n}\n\
                 char *safe(const char *s) {\n    char *p = malloc(strlen(s) + 1);\n\
                     if (!p) return NULL;\n    strcpy(p, s);\n    return p;\n}\n",
            ),
            (
                "t.c",
                "#include <stdlib.h>\nchar *copy(const char *);\nchar *safe(const char *);\n\
                 int main(void) {\n    char *a = safe(\"a\");\n    char *b = copy(\"b\");\n\
                     int ok = (a == NULL || a[0] == 'a') && b[0] == 'b';\n\
                     free(a);\n    free(b);\n    return !ok;\n}\n",
            ),
            (
                "Makefile",
                "t: t.c name.c\n\t$(CC) -o t t.c name.c\ncheck: t\n\t./t\n",
            ),
        ]);
        let outcome = run(dir.path());
        let reviewed: HashSet<String> = ["name.c".to_string()].into();
        let findings = sanitizer_findings(&outcome, &reviewed);
        assert_eq!(findings.len(), 1, "{findings:?} {:?}", outcome.notes);
        assert_eq!(findings[0].path.as_deref(), Some("name.c"));
        assert_eq!(findings[0].line, Some(5));
        assert!(
            findings[0].text.starts_with(
                "(high) Allocation failure not handled: with allocation 2 of the process failing"
            ),
            "{}",
            findings[0].text
        );
        assert!(
            findings[0].text.contains("(CWE-690)"),
            "{}",
            findings[0].text
        );
        assert!(
            outcome.notes[0].contains("each of the first 4 allocations"),
            "{:?}",
            outcome.notes
        );
    }
}
