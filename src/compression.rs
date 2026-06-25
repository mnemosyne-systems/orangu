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

#[derive(Debug)]
pub struct CompressionStats {
    pub original_lines: usize,
    pub compressed_lines: usize,
    pub pattern_matched: Option<String>,
}

#[derive(Debug, Default, Clone)]
pub struct CompressionMetrics {
    pub total_original_lines: usize,
    pub total_compressed_lines: usize,
    pub pattern_hits: std::collections::HashMap<String, usize>,
}

impl CompressionMetrics {
    pub fn record(&mut self, stats: &CompressionStats) {
        self.total_original_lines += stats.original_lines;
        self.total_compressed_lines += stats.compressed_lines;
        if let Some(pattern) = &stats.pattern_matched {
            *self.pattern_hits.entry(pattern.clone()).or_insert(0) += 1;
        }
    }
}

pub struct LlmContext {
    pub content: String,
    pub note: Option<String>,
}

pub fn compress_shell_output(command: &str, output: &str, diff_file_cap: usize) -> String {
    compress_shell_output_with_stats(command, output, diff_file_cap).0
}

pub fn compress_shell_output_with_stats(
    command: &str,
    output: &str,
    diff_file_cap: usize,
) -> (String, CompressionStats) {
    let original_lines = output.lines().count();

    // Normalise the command so that wrapped variants like
    //   `CARGO_TERM_COLOR=always cargo build`, `time cargo test`, etc.
    // all reach the same pattern checks as their bare form.
    let normalised = strip_command_prefix(command);

    let (compressed, pattern) =
        if normalised.contains("cargo build") || normalised.contains("cargo check") {
            (
                compress_cargo_build(output),
                Some("cargo_build".to_string()),
            )
        } else if normalised.starts_with("make ")
            || normalised == "make"
            || normalised.starts_with("cmake ")
            || normalised.starts_with("ninja ")
            || normalised == "ninja"
            || normalised.starts_with("gcc ")
            || normalised.starts_with("g++ ")
            || normalised.starts_with("clang ")
            || normalised.starts_with("clang++ ")
        {
            (
                compress_c_cpp_build(output),
                Some("c_cpp_build".to_string()),
            )
        } else if normalised.contains("cargo test") {
            (compress_cargo_test(output), Some("cargo_test".to_string()))
        } else if normalised.contains("pytest") || normalised.contains("python -m unittest") {
            (compress_pytest(output), Some("pytest".to_string()))
        } else if normalised.contains("npm test")
            || normalised.contains("jest")
            || normalised.contains("yarn test")
        {
            (compress_node_test(output), Some("node_test".to_string()))
        } else if normalised.contains("mvn test")
            || normalised.contains("mvnw test")
            || normalised.contains("gradle test")
            || normalised.contains("gradlew test")
        {
            (compress_java_test(output), Some("java_test".to_string()))
        } else if normalised.starts_with("git log") {
            (compress_git_log(output), Some("git_log".to_string()))
        } else if normalised.starts_with("git diff") || normalised.starts_with("git show") {
            (
                compress_git_diff(output, diff_file_cap),
                Some("git_diff".to_string()),
            )
        } else if normalised.starts_with("ls")
            || normalised.starts_with("find .")
            || normalised.starts_with("find /")
        {
            (compress_ls_output(output), Some("ls_find".to_string()))
        } else if normalised.contains("rg ")
            || normalised.contains("grep ")
            || normalised.contains("git grep ")
        {
            (compress_search_output(output), Some("search".to_string()))
        } else if normalised.contains("npm install")
            || normalised.contains("npm i")
            || normalised.contains("yarn install")
            || normalised.contains("yarn add")
            || normalised.contains("pip install")
            || normalised.contains("pip3 install")
        {
            (
                compress_package_install(output),
                Some("package_install".to_string()),
            )
        } else if original_lines > 300 {
            (compress_generic(output), Some("generic".to_string()))
        } else {
            (output.to_string(), None)
        };

    let compressed_lines = compressed.lines().count();

    let pattern_matched = if original_lines == 0 { None } else { pattern };

    (
        compressed,
        CompressionStats {
            original_lines,
            compressed_lines,
            pattern_matched,
        },
    )
}

/// Strip common command-line prefixes so that the core tool name is what
/// reaches the pattern-matching logic in `compress_shell_output_with_stats`.
///
/// Strips (repeatedly, left-to-right):
/// - Shell environment variable assignments of the form `KEY=value`
/// - Leading runner words: `time`, `sudo`, `watch` (with optional flags)
/// - Toolchain selectors for cargo: `+nightly`, `+stable`, `+beta`
///
/// Returns a sub-slice of the original `command` string (zero-copy where
/// possible) starting at the first character of the effective command.
pub fn strip_command_prefix(command: &str) -> &str {
    let mut s = command.trim_start();

    loop {
        // Strip `KEY=value` environment variable assignments.
        //
        // A key must start with a letter or underscore and contain only
        // alphanumeric characters or underscores; the value continues up to
        // the next ASCII space.
        if let Some(rest) = strip_env_assignment(s) {
            s = rest.trim_start();
            continue;
        }

        // Strip known runner prefixes (with optional simple flags).
        let stripped = strip_runner_prefix(s);
        if stripped.len() < s.len() {
            s = stripped;
            continue;
        }

        // Strip cargo toolchain selectors like `+nightly`.
        if s.starts_with('+')
            && let Some(idx) = s.find(' ')
        {
            s = s[idx..].trim_start();
            continue;
        }

        break;
    }

    s
}

/// If `s` starts with a shell `KEY=value` assignment, return the remainder
/// after the assignment token (including the space between tokens); otherwise
/// return `None`.
fn strip_env_assignment(s: &str) -> Option<&str> {
    // Key: starts with [A-Za-z_], followed by [A-Za-z0-9_]*
    let key_len = s
        .bytes()
        .take_while(|&b| b.is_ascii_alphanumeric() || b == b'_')
        .count();

    if key_len == 0 {
        return None;
    }

    let after_key = &s[key_len..];
    if !after_key.starts_with('=') {
        return None;
    }

    // Value extends until the next whitespace.
    let value_start = key_len + 1; // skip the '='
    let value_len = s[value_start..]
        .bytes()
        .take_while(|b| !b.is_ascii_whitespace())
        .count();

    Some(&s[value_start + value_len..])
}

/// Strip a single well-known runner prefix from `s`, returning the remainder.
/// Returns `s` unchanged if no runner was recognised.
fn strip_runner_prefix(s: &str) -> &str {
    // `watch` may be followed by flags such as `-n1` or `-n 1` before the
    // real command.
    if let Some(rest) = s.strip_prefix("watch") {
        let rest = rest.trim_start();
        // Consume optional flag groups like `-n1` or `-n 1`.
        let rest = if let Some(r) = rest.strip_prefix("-n") {
            // skip the numeric value (e.g. "1" or " 1")
            let r = r.trim_start();
            let num_len = r.bytes().take_while(|b| b.is_ascii_digit()).count();
            r[num_len..].trim_start()
        } else {
            rest
        };
        return rest;
    }

    for prefix in &["time ", "sudo ", "env "] {
        if let Some(rest) = s.strip_prefix(prefix) {
            return rest.trim_start();
        }
    }

    s
}

pub fn prepare_llm_diff_context(
    diff: &str,
    compression_enabled: bool,
    diff_file_cap: usize,
) -> LlmContext {
    prepare_llm_diff_context_with_stats(diff, compression_enabled, diff_file_cap).0
}

pub fn prepare_llm_diff_context_with_stats(
    diff: &str,
    compression_enabled: bool,
    diff_file_cap: usize,
) -> (LlmContext, CompressionStats) {
    let original_lines = diff.lines().count();

    if !compression_enabled {
        return (
            LlmContext {
                content: diff.to_string(),
                note: None,
            },
            CompressionStats {
                original_lines,
                compressed_lines: original_lines,
                pattern_matched: None,
            },
        );
    }

    let compressed = compress_git_diff(diff, diff_file_cap);
    let compressed_lines = compressed.lines().count();
    let note = (compressed_lines < original_lines).then(|| {
        format!(
            "Context note: orangu shortened this diff before sending it to the model ({} -> {} lines). Omitted sections are marked inline.",
            original_lines, compressed_lines
        )
    });

    (
        LlmContext {
            content: compressed,
            note,
        },
        CompressionStats {
            original_lines,
            compressed_lines,
            pattern_matched: Some("git_diff".to_string()),
        },
    )
}

pub fn prepare_llm_file_context(
    path: &str,
    content: &str,
    compression_enabled: bool,
) -> LlmContext {
    prepare_llm_file_context_with_stats(path, content, compression_enabled).0
}

pub fn prepare_llm_file_context_with_stats(
    _path: &str,
    content: &str,
    compression_enabled: bool,
) -> (LlmContext, CompressionStats) {
    let original_lines = content.lines().count();

    if !compression_enabled {
        return (
            LlmContext {
                content: content.to_string(),
                note: None,
            },
            CompressionStats {
                original_lines,
                compressed_lines: original_lines,
                pattern_matched: None,
            },
        );
    }

    let compressed = compress_generic(content);
    let compressed_lines = compressed.lines().count();
    let note = (compressed_lines < original_lines).then(|| {
        format!(
            "Context note: orangu shortened this file output before sending it to the model ({} -> {} lines). Omitted sections are marked inline.",
            original_lines, compressed_lines
        )
    });

    (
        LlmContext {
            content: compressed,
            note,
        },
        CompressionStats {
            original_lines,
            compressed_lines,
            pattern_matched: Some("file".to_string()),
        },
    )
}

/// Compress grep output into compact `file:line: snippet` citations for the
/// LLM. The user still sees the full output; the model receives a condensed
/// version capped at `max_matches` entries, grouped by file.
pub fn prepare_llm_grep_context(
    pattern: &str,
    output: &str,
    compression_enabled: bool,
) -> LlmContext {
    prepare_llm_grep_context_with_stats(pattern, output, compression_enabled).0
}

pub fn prepare_llm_grep_context_with_stats(
    pattern: &str,
    output: &str,
    compression_enabled: bool,
) -> (LlmContext, CompressionStats) {
    let original_lines = output.lines().count();

    if !compression_enabled {
        return (
            LlmContext {
                content: output.to_string(),
                note: None,
            },
            CompressionStats {
                original_lines,
                compressed_lines: original_lines,
                pattern_matched: None,
            },
        );
    }

    const MAX_MATCHES: usize = 40;

    // Count total matches before we truncate.
    let total_matches = output.lines().filter(|l| !l.is_empty()).count();

    if total_matches == 0 {
        return (
            LlmContext {
                content: output.to_string(),
                note: None,
            },
            CompressionStats {
                original_lines: 0,
                compressed_lines: 0,
                pattern_matched: None,
            },
        );
    }

    // Build a compact citation block: keep up to MAX_MATCHES lines, each as
    // a terse `file:line: snippet` reference stripped of colour codes.
    let citations: Vec<&str> = output
        .lines()
        .filter(|l| !l.is_empty())
        .take(MAX_MATCHES)
        .collect();

    let mut content = format!(
        "grep results for `{}`:\n\n{}\n",
        pattern,
        citations.join("\n")
    );

    let note = if total_matches > MAX_MATCHES {
        let omitted = total_matches - MAX_MATCHES;
        content.push_str(&format!(
            "\n... {} more matches omitted (use /grep with a narrower pattern to see all)\n",
            omitted
        ));
        Some(format!(
            "Context note: orangu truncated these grep results ({} matches found, sending first {} to the model).",
            total_matches, MAX_MATCHES
        ))
    } else {
        None
    };

    let compressed_lines = content.lines().count();

    (
        LlmContext { content, note },
        CompressionStats {
            original_lines,
            compressed_lines,
            pattern_matched: Some("grep".to_string()),
        },
    )
}

// ---------------------------------------------------------------------------
// Directory-listing compressor (`ls`, `find`)
// ---------------------------------------------------------------------------

/// Compress `ls` / `find` output when there are many entries.
///
/// - Fewer than 30 entries → returned unchanged.
/// - 30 or more entries → show the first 20 followed by a summary line that
///   includes the total number of remaining entries as well as a breakdown
///   of how many are regular files and how many are directories (as reported
///   by `ls -la`-style leading characters).
pub fn compress_ls_output(output: &str) -> String {
    let lines: Vec<&str> = output.lines().collect();

    // Detect `ls -la` style: lines starting with `-`, `d`, or `l` (symlinks).
    let is_long_format = lines
        .iter()
        .any(|l| matches!(l.chars().next(), Some('-' | 'd' | 'l')));

    // Collect only the "entry" lines (skip totals / blank lines).
    let entry_lines: Vec<&str> = if is_long_format {
        lines
            .iter()
            .filter(|l| matches!(l.chars().next(), Some('-' | 'd' | 'l')))
            .copied()
            .collect()
    } else {
        lines
            .iter()
            .filter(|l| !l.trim().is_empty())
            .copied()
            .collect()
    };

    const THRESHOLD: usize = 30;
    const SHOW: usize = 20;

    if entry_lines.len() < THRESHOLD {
        return output.to_string();
    }

    let remaining = entry_lines.len() - SHOW;

    let (file_count, dir_count) = if is_long_format {
        let files = entry_lines.iter().filter(|l| l.starts_with('-')).count();
        let dirs = entry_lines.iter().filter(|l| l.starts_with('d')).count();
        (files, dirs)
    } else {
        // Without long-format metadata we can't tell; report zero for both.
        (0, 0)
    };

    let result: Vec<&str> = lines
        .iter()
        .filter(|l| {
            !matches!(l.chars().next(), Some('-' | 'd' | 'l')) || {
                // In long-format mode, only keep the first SHOW entry lines.
                true
            }
        })
        .copied()
        .collect();

    // Re-build: emit non-entry lines verbatim (header, "total" line, blanks),
    // then entry lines up to SHOW, then the summary.
    let mut out: Vec<String> = Vec::new();
    let mut shown_entries = 0;

    for line in &lines {
        let is_entry = if is_long_format {
            matches!(line.chars().next(), Some('-' | 'd' | 'l'))
        } else {
            !line.trim().is_empty()
        };

        if is_entry {
            if shown_entries < SHOW {
                out.push(line.to_string());
                shown_entries += 1;
            }
            // Skip entries beyond SHOW; we'll append the summary below.
        } else {
            out.push(line.to_string());
        }
    }

    // Suppress the unused-variable warning for `result`.
    let _ = result;

    let summary = if is_long_format && (file_count > 0 || dir_count > 0) {
        format!(
            "... {} more entries ({} files, {} dirs)",
            remaining, file_count, dir_count
        )
    } else {
        format!("... {} more entries", remaining)
    };
    out.push(summary);

    out.join("\n")
}

// ---------------------------------------------------------------------------
// Search-output compressor (`rg`, `grep`, `git grep`)
// ---------------------------------------------------------------------------

/// Compress ripgrep / grep search output that contains many matches.
///
/// Ripgrep (and grep `-n`) typically outputs lines of the form
/// `file:line_number:content`. This function:
///
/// - Returns output unchanged when there are fewer than 50 match lines.
/// - Otherwise groups match lines by file prefix, keeps up to 5 per file,
///   appends `... N more matches in this file` for truncated files, and
///   prepends a header `[search: N total matches across M files]`.
pub fn compress_search_output(output: &str) -> String {
    // A "match line" is any non-empty line; separator / header lines emitted
    // by ripgrep (e.g. `--` between match groups) are preserved as-is but
    // not counted as matches for the threshold test.
    let all_lines: Vec<&str> = output.lines().collect();

    // Count lines that look like actual matches: contain at least one `:`.
    let match_lines: Vec<&str> = all_lines
        .iter()
        .filter(|l| !l.is_empty() && l.contains(':'))
        .copied()
        .collect();

    const THRESHOLD: usize = 50;
    const MAX_PER_FILE: usize = 5;

    if match_lines.len() < THRESHOLD {
        return output.to_string();
    }

    // Group by file (the portion before the first `:`).
    // We use an insertion-ordered structure to preserve file order.
    let mut file_order: Vec<String> = Vec::new();
    let mut file_matches: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

    for line in &match_lines {
        let file = line.split(':').next().unwrap_or("").to_string();
        if !file_matches.contains_key(&file) {
            file_order.push(file.clone());
            file_matches.insert(file.clone(), Vec::new());
        }
        file_matches.get_mut(&file).unwrap().push(line.to_string());
    }

    let total_matches = match_lines.len();
    let total_files = file_order.len();

    let mut out: Vec<String> = Vec::new();
    out.push(format!(
        "[search: {} total matches across {} files]",
        total_matches, total_files
    ));

    for file in &file_order {
        let matches = &file_matches[file];
        let shown = matches.len().min(MAX_PER_FILE);
        for m in &matches[..shown] {
            out.push(m.clone());
        }
        if matches.len() > MAX_PER_FILE {
            out.push(format!(
                "... {} more matches in this file",
                matches.len() - MAX_PER_FILE
            ));
        }
    }

    out.join("\n")
}

// ---------------------------------------------------------------------------
// Package-install compressor (`npm install`, `yarn`, `pip install`)
// ---------------------------------------------------------------------------

/// Compress noisy package-manager install output.
///
/// Strips:
/// - `npm warn` / `npm notice` lines
/// - Dependency-tree indentation lines (whitespace-leading lines containing
///   `->`)
/// - Progress-bar / ANSI-carriage-return lines (contain `\r` or ESC `\x1b`)
///
/// Always keeps:
/// - Lines containing `error` (case-insensitive)
/// - Lines containing `warn` (case-insensitive) that are *not* npm noise
/// - Summary lines: "added N packages", "Successfully installed", etc.
///
/// Returns unchanged when there are fewer than 20 lines.
pub fn compress_package_install(output: &str) -> String {
    let lines: Vec<&str> = output.lines().collect();

    const THRESHOLD: usize = 20;
    if lines.len() < THRESHOLD {
        return output.to_string();
    }

    let mut result: Vec<&str> = Vec::new();

    for line in &lines {
        // Always drop progress-bar / ANSI-CR lines.
        if line.contains('\r') || line.contains('\x1b') {
            continue;
        }

        // Drop npm noise.
        let trimmed = line.trim_start();
        if trimmed.starts_with("npm warn") || trimmed.starts_with("npm notice") {
            continue;
        }

        // Drop dependency-tree decoration lines (indented lines with `->`)
        if line.starts_with(|c: char| c.is_ascii_whitespace()) && line.contains("->") {
            continue;
        }

        result.push(line);
    }

    result.join("\n")
}

// ---------------------------------------------------------------------------
// Internal compressors (cargo, git, generic)
// ---------------------------------------------------------------------------

fn compress_cargo_build(output: &str) -> String {
    let mut compressed = Vec::new();
    let mut compiled_count = 0;
    let mut downloading_count = 0;

    for line in output.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("Compiling ") {
            compiled_count += 1;
        } else if trimmed.starts_with("Downloading ") || trimmed.starts_with("Downloaded ") {
            downloading_count += 1;
        } else {
            compressed.push(line);
        }
    }

    let mut result = Vec::new();
    if downloading_count > 0 {
        result.push(format!("... (downloaded {} packages)", downloading_count));
    }
    if compiled_count > 0 {
        result.push(format!("... (compiled {} packages)", compiled_count));
    }
    for line in compressed {
        result.push(line.to_string());
    }

    result.join("\n")
}

fn compress_c_cpp_build(output: &str) -> String {
    let mut compressed = Vec::new();
    let mut compiled_count = 0;

    for line in output.lines() {
        let trimmed = line.trim_start();

        let has_c_flag = trimmed.split_whitespace().any(|token| token == "-c");

        let is_compilation_step =
            // CMake / Ninja progress
            (trimmed.starts_with('[') && trimmed.contains("%]") && trimmed.contains("Building"))
            || (trimmed.starts_with('[') && trimmed.contains('/') && (trimmed.contains("CXX object") || trimmed.contains("C object")))
            // Make command echoes
            || ((trimmed.starts_with("gcc ") 
                || trimmed.starts_with("g++ ") 
                || trimmed.starts_with("clang ") 
                || trimmed.starts_with("clang++ ")
                || trimmed.starts_with("cc ")
                || trimmed.starts_with("c++ ")) && has_c_flag);

        if is_compilation_step {
            compiled_count += 1;
        } else {
            if compiled_count > 0 {
                compressed.push(format!("... (compiled {} files)", compiled_count));
                compiled_count = 0;
            }
            compressed.push(line.to_string());
        }
    }

    if compiled_count > 0 {
        compressed.push(format!("... (compiled {} files)", compiled_count));
    }

    compressed.join("\n")
}

fn compress_cargo_test(output: &str) -> String {
    let mut compressed = Vec::new();
    let mut in_failures_section = false;
    let mut ok_count = 0;

    for line in output.lines() {
        if line.starts_with("failures:") {
            in_failures_section = true;
            if ok_count > 0 {
                compressed.push(format!("... ({} tests passed)", ok_count));
                ok_count = 0;
            }
            compressed.push(line.to_string());
            continue;
        }

        if in_failures_section {
            compressed.push(line.to_string());
            continue;
        }

        if line.contains("... ok") {
            ok_count += 1;
        } else {
            if ok_count > 0 {
                compressed.push(format!("... ({} tests passed)", ok_count));
                ok_count = 0;
            }
            compressed.push(line.to_string());
        }
    }

    if ok_count > 0 && !in_failures_section {
        compressed.push(format!("... ({} tests passed)", ok_count));
    }

    compressed.join("\n")
}

fn compress_pytest(output: &str) -> String {
    let mut compressed = Vec::new();
    let mut dropped_passed = 0;

    for line in output.lines() {
        // Strip lines that are explicitly marking passing tests
        if line.contains(" PASSED ") || line.trim_end().ends_with(" PASSED") {
            dropped_passed += 1;
            continue;
        }
        compressed.push(line.to_string());
    }

    if dropped_passed > 0 {
        compressed.push(format!(
            "... ({} passed test lines omitted)",
            dropped_passed
        ));
    }

    compressed.join("\n")
}

fn compress_node_test(output: &str) -> String {
    let mut compressed = Vec::new();
    let mut dropped_passed = 0;

    for line in output.lines() {
        // Jest/Mocha often prefix passing suites with "PASS " or passing tests with " ✓ "
        if line.starts_with("PASS ") || line.contains(" ✓ ") {
            dropped_passed += 1;
            continue;
        }
        compressed.push(line.to_string());
    }

    if dropped_passed > 0 {
        compressed.push(format!(
            "... ({} passed test lines omitted)",
            dropped_passed
        ));
    }

    compressed.join("\n")
}

fn compress_java_test(output: &str) -> String {
    let mut compressed = Vec::new();
    let mut dropped_passed = 0;

    for line in output.lines() {
        // Gradle uses " PASSED", Maven uses "[INFO] Running..." and "[INFO] Tests run: X, Failures: 0"
        if line.ends_with(" PASSED")
            || (line.starts_with("[INFO] Tests run:") && line.contains("Failures: 0, Errors: 0"))
            || line.starts_with("[INFO] Running ")
        {
            dropped_passed += 1;
            continue;
        }
        compressed.push(line.to_string());
    }

    if dropped_passed > 0 {
        compressed.push(format!(
            "... ({} passed test lines omitted)",
            dropped_passed
        ));
    }

    compressed.join("\n")
}

fn compress_git_log(output: &str) -> String {
    let lines: Vec<&str> = output.lines().collect();
    let commit_indices: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| line.starts_with("commit "))
        .map(|(i, _)| i)
        .collect();

    if commit_indices.len() <= 20 {
        return output.to_string();
    }

    let cutoff_idx = commit_indices[15];
    let remaining_commits = commit_indices.len() - 15;

    let mut result = lines[..cutoff_idx].join("\n");
    result.push_str(&format!("\n\n... and {} more commits", remaining_commits));
    result
}

fn compress_git_diff(output: &str, diff_file_cap: usize) -> String {
    crate::diff::compress_git_diff(output, diff_file_cap)
}

const HOT_KEYWORDS: &[&str] = &[
    "error:",
    "Error:",
    "fatal:",
    "panic:",
    "exception",
    "Exception:",
    "Traceback",
    "stack trace",
    "warning:",
    "aborting",
    "Segmentation fault",
    "segfault",
    "SIGSEGV",
    "NullPointerException",
    "SyntaxError",
    "IndexOutOfBoundsException",
    "IndexError",
    "ArithmeticException",
    "ZeroDivisionError",
];
const GENERIC_HEAD_LINES: usize = 50;
const GENERIC_TAIL_LINES: usize = 50;
const GENERIC_CONTEXT_LINES: usize = 3;

fn compress_array_logs(output: &str) -> Option<String> {
    let lines: Vec<&str> = output.lines().collect();
    if lines.len() <= 50 {
        return None;
    }

    let trimmed = output.trim();
    let is_json_array = trimmed.starts_with('[') && trimmed.ends_with(']');

    let is_repetitive = {
        let mut prefix_counts = std::collections::HashMap::new();
        for line in &lines {
            if line.len() >= 4 {
                let prefix = &line[0..4];
                *prefix_counts.entry(prefix).or_insert(0) += 1;
            }
        }
        let threshold = (lines.len() as f64 * 0.8) as usize;
        prefix_counts.values().any(|&count| count >= threshold)
    };

    if is_json_array || is_repetitive {
        let mut result = Vec::new();
        for line in lines.iter().take(3) {
            result.push(line.to_string());
        }
        let omitted = lines.len() - 6;
        result.push(format!("... [{} items omitted] ...", omitted));
        for line in lines.iter().skip(lines.len() - 3) {
            result.push(line.to_string());
        }
        Some(result.join("\n"))
    } else {
        None
    }
}

fn compress_generic(output: &str) -> String {
    if let Some(compressed) = compress_array_logs(output) {
        return compressed;
    }

    let lines: Vec<&str> = output.lines().collect();
    if lines.len() <= 300 {
        return output.to_string();
    }

    let mut keep_indices = std::collections::BTreeSet::new();

    // 1. Keep the first 50 lines (head)
    for i in 0..GENERIC_HEAD_LINES.min(lines.len()) {
        keep_indices.insert(i);
    }

    // 2. Keep the last 50 lines (tail)
    let tail_start = lines.len().saturating_sub(GENERIC_TAIL_LINES);
    for i in tail_start..lines.len() {
        keep_indices.insert(i);
    }

    // 3. Keep any line containing a hot keyword, plus +/- 3 lines context
    for (i, line) in lines.iter().enumerate() {
        if HOT_KEYWORDS.iter().any(|&k| line.contains(k)) {
            let start = i.saturating_sub(GENERIC_CONTEXT_LINES);
            let end = (i + GENERIC_CONTEXT_LINES).min(lines.len() - 1);
            for j in start..=end {
                keep_indices.insert(j);
            }
        }
    }

    // 4. Reconstruct the output with omitted markers for gaps
    let mut result = String::new();
    let mut last_idx: Option<usize> = None;

    for &idx in &keep_indices {
        if let Some(last) = last_idx {
            if idx > last + 1 {
                let omitted = idx - last - 1;
                result.push_str(&format!("\n\n... [{} lines omitted] ...\n\n", omitted));
            } else {
                result.push('\n');
            }
        }
        result.push_str(lines[idx]);
        last_idx = Some(idx);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Existing tests (unchanged)
    // -----------------------------------------------------------------------

    #[test]
    fn test_cargo_build_compression() {
        let output = "\
Compiling foo v0.1.0\n\
Compiling bar v0.2.0\n\
warning: unused variable\n\
error: could not compile `foo`\n\
Finished dev [unoptimized + debuginfo] target(s) in 2.0s\n";

        let compressed = compress_cargo_build(output);
        assert!(!compressed.contains("Compiling foo"));
        assert!(compressed.contains("compiled 2 packages"));
        assert!(compressed.contains("warning: unused variable"));
        assert!(compressed.contains("error: could not compile `foo`"));
    }

    #[test]
    fn test_cargo_test_compression() {
        let output = "\
running 3 tests\n\
test test_foo ... ok\n\
test test_bar ... ok\n\
test test_baz ... FAILED\n\
failures:\n\
---- test_baz stdout ----\n\
panicked at 'explicit panic'\n\
failures:\n\
    test_baz\n\
test result: FAILED. 2 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out\n";

        let compressed = compress_cargo_test(output);
        assert!(!compressed.contains("test test_foo ... ok"));
        assert!(compressed.contains("2 tests passed"));
        assert!(compressed.contains("test test_baz ... FAILED"));
        assert!(compressed.contains("panicked at"));
    }

    #[test]
    fn llm_diff_context_keeps_small_diffs_raw() {
        let diff = "diff --git a/a b/a\n+one\n";
        let context = prepare_llm_diff_context(diff, true, 20);
        assert_eq!(context.content, diff);
        assert!(context.note.is_none());
    }

    #[test]
    fn llm_diff_context_marks_large_compressed_diffs() {
        let mut diff =
            "diff --git a/a b/a\nindex 0..1 100644\n--- a/a\n+++ b/a\n@@ -1,600 +1,600 @@\n"
                .to_string();
        for i in 0..600 {
            diff.push_str(&format!(" line {i}\n"));
        }
        diff.push_str("+new line\n");
        let context = prepare_llm_diff_context(&diff, true, 20);
        assert!(!context.content.contains(" line 10\n"));
        assert!(context.note.expect("note").contains(" -> "));
    }

    #[test]
    fn llm_diff_context_can_be_disabled() {
        let diff = (0..600)
            .map(|line| format!("+line {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        let context = prepare_llm_diff_context(&diff, false, 20);
        assert_eq!(context.content, diff);
        assert!(context.note.is_none());
    }

    // -----------------------------------------------------------------------
    // strip_command_prefix tests
    // -----------------------------------------------------------------------

    #[test]
    fn strip_prefix_env_var() {
        assert_eq!(
            strip_command_prefix("CARGO_TERM_COLOR=always cargo build"),
            "cargo build"
        );
    }

    #[test]
    fn strip_prefix_multiple_env_vars() {
        assert_eq!(strip_command_prefix("FOO=1 BAR=2 cargo test"), "cargo test");
    }

    #[test]
    fn strip_prefix_time() {
        assert_eq!(strip_command_prefix("time cargo test"), "cargo test");
    }

    #[test]
    fn strip_prefix_sudo() {
        assert_eq!(strip_command_prefix("sudo cargo build"), "cargo build");
    }

    #[test]
    fn strip_prefix_watch() {
        assert_eq!(strip_command_prefix("watch -n1 cargo check"), "cargo check");
    }

    #[test]
    fn strip_prefix_toolchain_selector() {
        assert_eq!(
            strip_command_prefix("cargo +nightly check"),
            // strip_command_prefix strips env/runner prefixes; `+nightly` is
            // inside the cargo invocation, so only `cargo` is left as-is.
            "cargo +nightly check"
        );
    }

    #[test]
    fn strip_prefix_combined() {
        assert_eq!(
            strip_command_prefix("CARGO_TERM_COLOR=always time cargo build --release"),
            "cargo build --release"
        );
    }

    #[test]
    fn strip_prefix_bare_command_unchanged() {
        assert_eq!(strip_command_prefix("cargo test"), "cargo test");
        assert_eq!(strip_command_prefix("git log"), "git log");
    }

    // -----------------------------------------------------------------------
    // compress_shell_output_with_stats: prefix-aware routing tests
    // -----------------------------------------------------------------------

    #[test]
    fn dispatch_env_wrapped_cargo_build() {
        let output = "Compiling foo v0.1.0\nFinished dev target(s) in 1.0s\n";
        let (_, stats) =
            compress_shell_output_with_stats("CARGO_TERM_COLOR=always cargo build", output, 20);
        assert_eq!(stats.pattern_matched.as_deref(), Some("cargo_build"));
    }

    #[test]
    fn dispatch_time_wrapped_cargo_test() {
        let output = "running 1 test\ntest t ... ok\ntest result: ok.\n";
        let (_, stats) = compress_shell_output_with_stats("time, cargo test --release", output, 20);
        assert_eq!(stats.pattern_matched.as_deref(), Some("cargo_test"));
    }

    // -----------------------------------------------------------------------
    // compress_ls_output tests
    // -----------------------------------------------------------------------

    #[test]
    fn ls_output_small_unchanged() {
        // 5 entries — well below the 30-entry threshold.
        let output = "file1.rs\nfile2.rs\nfile3.rs\nfile4.rs\nfile5.rs";
        assert_eq!(compress_ls_output(output), output);
    }

    #[test]
    fn ls_output_large_truncated() {
        // 35 plain-filename entries.
        let output = (0..35)
            .map(|i| format!("file{}.rs", i))
            .collect::<Vec<_>>()
            .join("\n");
        let compressed = compress_ls_output(&output);
        assert!(
            compressed.contains("... 15 more entries"),
            "got: {compressed}"
        );
        // Must show exactly 20 entries before the summary.
        assert!(compressed.contains("file0.rs"));
        assert!(compressed.contains("file19.rs"));
        assert!(!compressed.contains("file20.rs"));
    }

    #[test]
    fn ls_la_output_large_truncated_with_counts() {
        // Build a fake `ls -la` listing: 2 dirs + 33 regular files = 35 entries.
        let mut lines = vec![
            "total 140".to_string(),
            "drwxr-xr-x 2 user group 4096 Jun 23 00:00 .".to_string(),
            "drwxr-xr-x 5 user group 4096 Jun 23 00:00 ..".to_string(),
        ];
        for i in 0..33 {
            lines.push(format!(
                "-rw-r--r-- 1 user group 128 Jun 23 00:00 file{}.rs",
                i
            ));
        }
        let output = lines.join("\n");
        let compressed = compress_ls_output(&output);
        assert!(compressed.contains("more entries"), "got: {compressed}");
        assert!(
            compressed.contains("files") || compressed.contains("dirs"),
            "expected file/dir breakdown in: {compressed}"
        );
    }

    // -----------------------------------------------------------------------
    // compress_search_output tests
    // -----------------------------------------------------------------------

    #[test]
    fn search_output_small_unchanged() {
        // 10 match lines — below the 50-line threshold.
        let output = (0..10)
            .map(|i| format!("src/foo.rs:{}:let x = {};", i, i))
            .collect::<Vec<_>>()
            .join("\n");
        assert_eq!(compress_search_output(&output), output);
    }

    #[test]
    fn search_output_large_grouped() {
        // 60 matches: 30 in foo.rs, 30 in bar.rs.
        let mut lines = Vec::new();
        for i in 0..30 {
            lines.push(format!("src/foo.rs:{}:match {}", i, i));
        }
        for i in 0..30 {
            lines.push(format!("src/bar.rs:{}:match {}", i, i));
        }
        let output = lines.join("\n");
        let compressed = compress_search_output(&output);

        assert!(
            compressed.starts_with("[search: 60 total matches across 2 files]"),
            "header missing: {compressed}"
        );
        // Each file should have at most 5 matches shown.
        let foo_shown = compressed
            .lines()
            .filter(|l| l.starts_with("src/foo.rs:"))
            .count();
        let bar_shown = compressed
            .lines()
            .filter(|l| l.starts_with("src/bar.rs:"))
            .count();
        assert_eq!(foo_shown, 5);
        assert_eq!(bar_shown, 5);
        assert!(compressed.contains("25 more matches in this file"));
    }

    // -----------------------------------------------------------------------
    // compress_package_install tests
    // -----------------------------------------------------------------------

    #[test]
    fn package_install_small_unchanged() {
        let output =
            "Collecting requests\nInstalling requests\nSuccessfully installed requests-2.31.0";
        assert_eq!(compress_package_install(output), output);
    }

    #[test]
    fn package_install_strips_npm_noise() {
        let mut lines: Vec<String> = (0..25)
            .map(|i| format!("npm warn deprecated pkg{}@1.0.0: please update", i))
            .collect();
        lines.push("npm notice created a lockfile as package-lock.json".to_string());
        lines.push("added 42 packages from 30 contributors in 3.14s".to_string());
        let output = lines.join("\n");
        let compressed = compress_package_install(&output);

        assert!(
            !compressed.contains("npm warn"),
            "npm warn should be stripped"
        );
        assert!(
            !compressed.contains("npm notice"),
            "npm notice should be stripped"
        );
        assert!(
            compressed.contains("added 42 packages"),
            "summary line must be kept"
        );
    }

    #[test]
    fn package_install_strips_dep_tree_lines() {
        let mut lines: Vec<String> = vec!["Installing dependencies...".to_string()];
        for i in 0..25 {
            lines.push(format!("  pkg{} -> dep{}", i, i));
        }
        lines.push("Successfully installed 25 packages.".to_string());
        let output = lines.join("\n");
        let compressed = compress_package_install(&output);

        assert!(
            !compressed.contains("  pkg0 -> dep0"),
            "dep-tree lines should be stripped"
        );
        assert!(compressed.contains("Successfully installed"));
    }

    #[test]
    fn package_install_strips_progress_bar_lines() {
        let mut lines: Vec<String> = vec!["Starting install...".to_string()];
        for _ in 0..25 {
            lines.push("\rDownloading... [=====>    ] 50%".to_string());
        }
        lines.push("Done.".to_string());
        let output = lines.join("\n");
        let compressed = compress_package_install(&output);

        assert!(
            !compressed.contains('\r'),
            "progress-bar lines should be stripped"
        );
        assert!(compressed.contains("Done."));
    }

    #[test]
    fn test_generic_hotline_compression() {
        let mut output = String::new();
        for i in 0..100 {
            output.push_str(&format!("setup line {}\n", i));
        }
        for i in 0..100 {
            output.push_str(&format!("noise line {}\n", i));
        }
        output.push_str("Traceback (most recent call last):\n");
        output.push_str("  File \"foo.py\", line 10\n");
        output.push_str("    x = 1 / 0\n");
        output.push_str("ZeroDivisionError: division by zero\n");
        for i in 0..100 {
            output.push_str(&format!("more noise {}\n", i));
        }
        for i in 0..50 {
            output.push_str(&format!("tail line {}\n", i));
        }

        let compressed = compress_generic(&output);

        // Should keep first 50 setup lines
        assert!(compressed.contains("setup line 0"));
        assert!(compressed.contains("setup line 49"));
        assert!(!compressed.contains("setup line 50")); // Omitted

        // Should keep the traceback
        assert!(compressed.contains("Traceback (most recent call last):"));
        assert!(compressed.contains("ZeroDivisionError: division by zero"));

        // Should keep context noise around traceback (+/- 3 lines)
        assert!(compressed.contains("noise line 97")); // -3 from traceback
        assert!(compressed.contains("more noise 2")); // +3 from error

        // Should drop far noise
        assert!(!compressed.contains("noise line 50"));
        assert!(!compressed.contains("more noise 50"));

        // Should keep tail lines
        assert!(compressed.contains("tail line 0"));
        assert!(compressed.contains("tail line 49"));

        // Should have omission markers
        assert!(compressed.contains("lines omitted"));
    }
}
