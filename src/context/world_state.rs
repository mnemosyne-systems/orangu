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

use crate::diff::compress_git_diff;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::process::Command;

#[derive(Debug, Clone, Default)]
pub struct WorldState {
    pub open_files: Vec<PathBuf>,
    pub env_vars: HashMap<String, String>,
}

impl WorldState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn diff(&self, previous: &WorldState) -> WorldStateDiff {
        let mut added_files = Vec::new();
        let mut removed_files = Vec::new();

        for file in &self.open_files {
            if !previous.open_files.contains(file) {
                added_files.push(file.clone());
            }
        }

        for file in &previous.open_files {
            if !self.open_files.contains(file) {
                removed_files.push(file.clone());
            }
        }

        let mut changed_env = HashMap::new();
        for (k, v) in &self.env_vars {
            if previous.env_vars.get(k) != Some(v) {
                changed_env.insert(k.clone(), v.clone());
            }
        }

        WorldStateDiff {
            added_files,
            removed_files,
            changed_env,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WorldStateDiff {
    pub added_files: Vec<PathBuf>,
    pub removed_files: Vec<PathBuf>,
    pub changed_env: HashMap<String, String>,
}

/// How many lines of a *new* (untracked) file are worth inlining before the
/// rest is summarised away. An untracked file is entirely additions, so
/// without a cap a single one contributes its whole length to the prompt —
/// and unlike a tracked hunk there is no surrounding context for the
/// compressor to trim back down. This fragment is a *change notification*,
/// not a document delivery: a screenful says what the new file is, and
/// anything more is what the file-reading tools are for.
const UNTRACKED_FILE_LINE_CAP: usize = 80;

/// Past this size an untracked file is *announced* rather than inlined. A
/// large new file is a document, not a change: `git diff` does not show
/// untracked content at all, and the workspace's file-reading tools can
/// fetch it on demand if the answer needs it. Inlining it instead put every
/// stray note, log and export in the tree into the prompt of every turn.
const UNTRACKED_FILE_INLINE_MAX_BYTES: usize = 4 * 1024;

/// Ceiling on the rendered fragment, applied after compression as the last
/// line of defence. Whatever the diff turns out to be, the per-turn context
/// this adds is bounded — see [`truncate_to_budget`]. The two caps above
/// normally bind first; this one exists so that no working tree, however
/// unusual, can make a turn's prompt unbounded.
pub const DEFAULT_WORLD_STATE_MAX_BYTES: usize = 8 * 1024;

/// The `diff --git` header every untracked entry gets, whether or not its
/// content is inlined. Split out so the announce-only path can be produced
/// from [`std::fs::Metadata`] alone, without reading the file.
///
/// That header line is not cosmetic: it is the *only* thing
/// [`compress_git_diff`] treats as a file boundary. A synthetic diff that
/// omits it does not merely look odd — its content is either attributed to
/// whichever tracked file happened to come last (escaping the file cap
/// entirely) or, when there are no tracked changes at all, dropped on the
/// floor with the hunk that has no file to attach to. Both were live, and
/// which one you got turned on whether the tree happened to have a tracked
/// change in it: with one, every untracked text file in the workspace went
/// into the prompt whole; without one, none of them did.
fn untracked_header(file: &str, line_count: usize) -> String {
    format!(
        "diff --git a/{file} b/{file}\nnew file mode 100644\n--- /dev/null\n+++ b/{file}\n@@ -0,0 +1,{line_count} @@\n"
    )
}

/// The announce-only form, for a file too large to inline. Takes the size
/// from the directory entry so an oversized file — a 42 MB profiler capture
/// sitting untracked in the tree, say — is never read into memory at all,
/// let alone once per turn.
fn announce_untracked_file(file: &str, size: u64) -> String {
    let mut out = untracked_header(file, 0);
    out.push_str(&format!(
        " ... [new file, {size} bytes — content not inlined] ...\n"
    ));
    out
}

/// Render one untracked file as a `diff --git` new-file diff, inlining up to
/// [`UNTRACKED_FILE_LINE_CAP`] lines of it.
fn render_untracked_file(file: &str, content: &str) -> String {
    let lines: Vec<&str> = content.lines().collect();
    let mut out = untracked_header(file, lines.len());
    if content.len() > UNTRACKED_FILE_INLINE_MAX_BYTES {
        out.push_str(&format!(
            " ... [new file, {} lines / {} bytes — content not inlined] ...\n",
            lines.len(),
            content.len()
        ));
        return out;
    }
    let shown = lines.len().min(UNTRACKED_FILE_LINE_CAP);
    for line in &lines[..shown] {
        out.push('+');
        out.push_str(line);
        out.push('\n');
    }
    if lines.len() > shown {
        out.push_str(&format!(
            " ... [{} more lines of this new file omitted] ...\n",
            lines.len() - shown
        ));
    }
    out
}

/// Cut `diff` to at most `max_bytes`, on a line boundary, with a marker saying
/// what was left out. `0` disables the cap.
///
/// The cut is found by scanning *bytes* for the last `\n` within budget, never
/// by slicing the `str` at `max_bytes` directly: a diff carries whatever the
/// source files carry, and an arbitrary byte index lands mid-character often
/// enough that slicing there is a panic waiting for the first non-ASCII
/// working tree. `\n` is ASCII and UTF-8 is self-synchronizing, so a byte
/// equal to `\n` is always a real newline and the index after it is always a
/// character boundary.
fn truncate_to_budget(diff: &str, max_bytes: usize) -> String {
    if max_bytes == 0 || diff.len() <= max_bytes {
        return diff.to_string();
    }
    let cut = diff.as_bytes()[..max_bytes]
        .iter()
        .rposition(|&b| b == b'\n')
        .map(|i| i + 1)
        .unwrap_or(0);
    let mut out = diff[..cut].to_string();
    out.push_str(&format!(
        "... [workspace diff truncated: {} of {} bytes shown] ...\n",
        cut,
        diff.len()
    ));
    out
}

pub async fn get_current_workspace_diff(
    workspace_path: &Path,
    diff_file_cap: usize,
    max_bytes: usize,
) -> Option<(u64, String)> {
    // 1. Get tracked changes
    let tracked_output = Command::new("git")
        .args(["diff", "HEAD", "-M"])
        .current_dir(workspace_path)
        .output()
        .await
        .ok()?;
    let tracked_diff = String::from_utf8_lossy(&tracked_output.stdout).to_string();

    // 2. Get untracked files
    let untracked_output = Command::new("git")
        .args(["ls-files", "--others", "--exclude-standard"])
        .current_dir(workspace_path)
        .output()
        .await
        .ok()?;
    let untracked_files = String::from_utf8_lossy(&untracked_output.stdout);

    let mut untracked_diff = String::new();
    for file in untracked_files.lines() {
        let file = file.trim();
        if file.is_empty() {
            continue;
        }
        let file_path = workspace_path.join(file);
        // Size first, content second: anything past the inline threshold is
        // announced from its metadata, so a large untracked file costs one
        // `stat` per turn rather than a full read plus UTF-8 validation.
        match std::fs::metadata(&file_path) {
            Ok(meta) if meta.len() as usize > UNTRACKED_FILE_INLINE_MAX_BYTES => {
                untracked_diff.push_str(&announce_untracked_file(file, meta.len()));
            }
            Ok(_) => {
                if let Ok(content) = std::fs::read_to_string(&file_path) {
                    untracked_diff.push_str(&render_untracked_file(file, &content));
                }
            }
            Err(_) => {}
        }
    }

    let combined_diff = format!("{}{}", tracked_diff, untracked_diff);
    if combined_diff.trim().is_empty() {
        return None;
    }

    let mut hasher = Sha256::new();
    hasher.update(combined_diff.as_bytes());
    let hash_array = hasher.finalize();
    // We only need a fast equality check, so folding into a u64 is fine
    let hash = u64::from_le_bytes(hash_array[0..8].try_into().unwrap());

    let final_diff = if combined_diff.len() > 500 * 1024 {
        // Massive diff, fallback to summary
        let summary_output = Command::new("git")
            .args(["diff", "--compact-summary", "HEAD"])
            .current_dir(workspace_path)
            .output()
            .await
            .ok()?;
        let mut summary = String::from_utf8_lossy(&summary_output.stdout).to_string();
        if !untracked_diff.is_empty() {
            summary.push_str("\nUntracked files:\n");
            summary.push_str(&untracked_files);
        }
        summary
    } else {
        compress_git_diff(&combined_diff, diff_file_cap)
    };

    Some((hash, truncate_to_budget(&final_diff, max_bytes)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diff::compress_git_diff;

    /// The regression that made a whole workspace's untracked files land in
    /// every prompt: without a `diff --git` header the compressor has no file
    /// to cap, so a new file's content survives the file cap intact.
    #[test]
    fn an_untracked_file_is_capped_by_the_file_cap_like_any_other() {
        let body: String = (0..50).map(|i| format!("line {i}\n")).collect();
        let mut diff = String::new();
        for n in 0..30 {
            diff.push_str(&render_untracked_file(&format!("new{n}.md"), &body));
        }
        let compressed = compress_git_diff(&diff, 5);
        assert_eq!(compressed.matches("diff --git").count(), 5);
        assert!(compressed.contains("files omitted due to size limits"));
    }

    /// The other half of the same bug: with no tracked changes at all there
    /// was no open file for the synthetic hunk to attach to, and every line
    /// of it was silently discarded. A capped, attributed diff must survive.
    #[test]
    fn an_untracked_file_survives_compression_with_no_tracked_changes() {
        let diff = render_untracked_file("notes.md", "alpha\nbeta\n");
        let compressed = compress_git_diff(&diff, 20);
        assert!(compressed.contains("+alpha"), "{compressed}");
        assert!(compressed.contains("+beta"), "{compressed}");
    }

    #[test]
    fn a_long_new_file_is_cut_to_the_line_cap() {
        // Many lines, but still under the inline byte threshold.
        let body: String = (0..UNTRACKED_FILE_LINE_CAP * 3).map(|_| "x\n").collect();
        assert!(body.len() < UNTRACKED_FILE_INLINE_MAX_BYTES);
        let rendered = render_untracked_file("big.md", &body);
        assert_eq!(
            rendered.lines().filter(|l| l.starts_with('+')).count(),
            // `+++ b/big.md` is a header line, not an addition.
            UNTRACKED_FILE_LINE_CAP + 1
        );
        assert!(rendered.contains("more lines of this new file omitted"));
    }

    #[test]
    fn an_oversized_new_file_is_announced_from_its_size_alone() {
        // The announce-only path never sees the content, so a huge untracked
        // file must still produce a well-formed, cappable entry.
        let rendered = announce_untracked_file("capture.rgp", 42_082_423);
        assert!(rendered.starts_with("diff --git a/capture.rgp b/capture.rgp\n"));
        assert!(rendered.contains("42082423 bytes"), "{rendered}");
        assert!(rendered.contains("content not inlined"), "{rendered}");
        assert_eq!(
            compress_git_diff(&rendered, 20)
                .matches("diff --git")
                .count(),
            1
        );
    }

    #[test]
    fn a_large_new_file_is_announced_not_inlined() {
        let body = "some prose\n".repeat(UNTRACKED_FILE_INLINE_MAX_BYTES);
        let rendered = render_untracked_file("notes.md", &body);
        assert!(rendered.contains("content not inlined"), "{rendered}");
        assert!(!rendered.contains("+some prose"), "{rendered}");
        // Still a well-formed, cappable file entry.
        assert!(rendered.starts_with("diff --git a/notes.md b/notes.md\n"));
        assert!(rendered.len() < 200);
    }

    #[test]
    fn the_budget_cuts_on_a_line_boundary_and_says_so() {
        let diff: String = (0..1000).map(|i| format!("+line {i}\n")).collect();
        let cut = truncate_to_budget(&diff, 200);
        assert!(cut.len() < diff.len());
        assert!(cut.contains("workspace diff truncated"));
        assert!(
            cut.lines()
                .all(|l| l.starts_with('+') || l.starts_with("..."))
        );
    }

    /// Slicing a `str` at the raw budget index panics the moment a diff
    /// carries a multi-byte character across it — and these diffs are of this
    /// project's own prose, which is full of em-dashes.
    #[test]
    fn the_budget_cut_survives_multibyte_characters() {
        let diff: String = (0..200).map(|i| format!("+line — {i}\n")).collect();
        for budget in 1..diff.len() {
            let cut = truncate_to_budget(&diff, budget);
            assert!(cut.contains("workspace diff truncated") || cut == diff);
        }
    }

    #[test]
    fn a_diff_under_budget_is_untouched() {
        let diff = "+one\n+two\n";
        assert_eq!(truncate_to_budget(diff, 32 * 1024), diff);
        assert_eq!(truncate_to_budget(diff, 0), diff);
    }
}
