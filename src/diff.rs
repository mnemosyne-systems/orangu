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

#[derive(Debug, Clone)]
struct DiffHunk {
    header: String,
    lines: Vec<String>,
    additions: usize,
    deletions: usize,
    score: f64,
    original_index: usize,
}

#[derive(Debug, Clone)]
struct DiffFile {
    header_lines: Vec<String>,
    hunks: Vec<DiffHunk>,
    additions: usize,
    deletions: usize,
    original_index: usize,
}

pub fn compress_git_diff(output: &str, file_cap: usize) -> String {
    let mut files: Vec<DiffFile> = Vec::new();
    let mut current_file: Option<DiffFile> = None;
    let mut current_hunk: Option<DiffHunk> = None;

    let mut file_idx = 0;
    let mut hunk_idx = 0;

    // Parse diff
    let mut preamble = Vec::new();
    for line in output.lines() {
        if line.starts_with("diff --git") {
            if let Some(h) = current_hunk.take() {
                if let Some(mut f) = current_file.take() {
                    f.additions += h.additions;
                    f.deletions += h.deletions;
                    f.hunks.push(h);
                    files.push(f);
                }
            } else if let Some(f) = current_file.take() {
                files.push(f);
            }

            current_file = Some(DiffFile {
                header_lines: vec![line.to_string()],
                hunks: Vec::new(),
                additions: 0,
                deletions: 0,
                original_index: file_idx,
            });
            file_idx += 1;
            hunk_idx = 0;
        } else if line.starts_with("@@ ") {
            // A hunk with no `diff --git` above it still belongs to *some*
            // file. Left unattributed it used to be dropped outright — the
            // `current_hunk.take()` below runs whether or not there is a file
            // to attach it to — so a hand-built diff missing that header lost
            // its content silently. Open a synthetic file instead, so the
            // hunk is capped and counted like every other one.
            if current_file.is_none() {
                current_file = Some(DiffFile {
                    header_lines: vec!["diff --git a/(unnamed) b/(unnamed)".to_string()],
                    hunks: Vec::new(),
                    additions: 0,
                    deletions: 0,
                    original_index: file_idx,
                });
                file_idx += 1;
                hunk_idx = 0;
            }
            if let (Some(h), Some(f)) = (current_hunk.take(), current_file.as_mut()) {
                f.additions += h.additions;
                f.deletions += h.deletions;
                f.hunks.push(h);
                hunk_idx += 1;
            }
            current_hunk = Some(DiffHunk {
                header: line.to_string(),
                lines: Vec::new(),
                additions: 0,
                deletions: 0,
                score: 0.0,
                original_index: hunk_idx,
            });
        } else if let Some(ref mut h) = current_hunk {
            h.lines.push(line.to_string());
            if line.starts_with('+') {
                h.additions += 1;
            } else if line.starts_with('-') {
                h.deletions += 1;
            }
        } else if let Some(ref mut f) = current_file {
            f.header_lines.push(line.to_string());
        } else {
            preamble.push(line.to_string());
        }
    }

    if let Some(h) = current_hunk.take() {
        if let Some(mut f) = current_file.take() {
            f.additions += h.additions;
            f.deletions += h.deletions;
            f.hunks.push(h);
            files.push(f);
        }
    } else if let Some(f) = current_file.take() {
        files.push(f);
    }

    // 2. File Capping
    // Calculate totals first if missing, actually we just updated them above.
    let mut file_refs: Vec<&DiffFile> = files.iter().collect();
    // sort descending by additions + deletions
    file_refs.sort_by_key(|f| std::cmp::Reverse(f.additions + f.deletions));
    let mut top_files: Vec<DiffFile> = file_refs.into_iter().take(file_cap).cloned().collect();
    // restore original file order
    top_files.sort_by_key(|f| f.original_index);

    // Process files
    let mut result_files = Vec::new();
    for mut file in top_files {
        // 3. Context Trimming
        for hunk in &mut file.hunks {
            let mut keep = vec![false; hunk.lines.len()];
            let mut last_change: Option<isize> = None;
            for (i, l) in hunk.lines.iter().enumerate() {
                if l.starts_with('+') || l.starts_with('-') || l.starts_with('\\') {
                    last_change = Some(i as isize);
                    keep[i] = true;
                } else if l.starts_with(' ') {
                    if last_change.is_some_and(|lc| (i as isize - lc) <= 2) {
                        keep[i] = true;
                    }
                } else {
                    keep[i] = true;
                }
            }
            last_change = None;
            for (i, l) in hunk.lines.iter().enumerate().rev() {
                if l.starts_with('+') || l.starts_with('-') || l.starts_with('\\') {
                    last_change = Some(i as isize);
                } else if l.starts_with(' ') && last_change.is_some_and(|lc| (lc - i as isize) <= 2)
                {
                    keep[i] = true;
                }
            }

            let mut trimmed = Vec::new();
            let mut omitted_count = 0;
            for (i, line) in hunk.lines.iter().enumerate() {
                if keep[i] {
                    if omitted_count > 0 {
                        trimmed.push(format!(
                            " ... [{} context lines omitted] ...",
                            omitted_count
                        ));
                        omitted_count = 0;
                    }
                    trimmed.push(line.clone());
                } else {
                    omitted_count += 1;
                }
            }
            if omitted_count > 0 {
                trimmed.push(format!(
                    " ... [{} context lines omitted] ...",
                    omitted_count
                ));
            }
            hunk.lines = trimmed;
        }

        // 4. Hunk Scoring
        if file.hunks.len() > 10 {
            let first = file.hunks.remove(0);
            let last = file.hunks.pop().unwrap();

            let keywords = [
                "error",
                "panic",
                "exception",
                "todo",
                "fixme",
                "bug",
                "auth",
                "secret",
                "password",
            ];
            for hunk in &mut file.hunks {
                let total_changes = hunk.additions + hunk.deletions;
                let density_score = (0.03 * total_changes as f64).min(0.3);
                let mut kw_score = 0.0;
                let text = hunk.lines.join("\n").to_lowercase();
                for kw in keywords {
                    if text.contains(kw) {
                        kw_score += 0.3;
                    }
                }
                hunk.score = density_score + kw_score;
            }

            file.hunks.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            file.hunks.truncate(8); // keep up to 8 middle hunks + 2 (first and last) = 10

            file.hunks.push(first);
            file.hunks.push(last);
            file.hunks.sort_by_key(|h| h.original_index);
        }

        result_files.push(file);
    }

    // 5. Reassemble
    let mut out = Vec::new();
    if !preamble.is_empty() {
        out.push(preamble.join("\n"));
    }
    for file in result_files {
        out.push(file.header_lines.join("\n"));
        for hunk in file.hunks {
            out.push(hunk.header);
            if !hunk.lines.is_empty() {
                out.push(hunk.lines.join("\n"));
            }
        }
    }

    let mut final_out = out.join("\n");
    let dropped_files = files.len().saturating_sub(file_cap);
    if dropped_files > 0 {
        final_out.push_str(&format!(
            "\n... [{} files omitted due to size limits] ...",
            dropped_files
        ));
    }

    if output.ends_with('\n') && !final_out.ends_with('\n') {
        final_out.push('\n');
    }
    final_out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_capping() {
        let mut diff = String::new();
        for i in 0..25 {
            diff.push_str(&format!("diff --git a/file{} b/file{}\n", i, i));
            diff.push_str(&format!(
                "index 0000000..1111111 100644\n--- a/file{}\n+++ b/file{}\n",
                i, i
            ));
            diff.push_str("@@ -1,1 +1,2 @@\n");
            diff.push_str(&format!("+{}\n", i));
        }
        let compressed = compress_git_diff(&diff, 20);
        let count = compressed.matches("diff --git").count();
        assert_eq!(count, 20);
    }

    #[test]
    fn a_hunk_with_no_file_header_is_kept_not_dropped() {
        // A diff whose first line is a hunk header has no `diff --git` to
        // attach to. That used to discard every line of it — the compressor
        // returned an empty string and the caller had no way to tell that
        // from "nothing changed".
        let diff = "@@ -0,0 +1,2 @@\n+alpha\n+beta\n";
        let compressed = compress_git_diff(diff, 20);
        assert!(compressed.contains("+alpha"), "{compressed}");
        assert!(compressed.contains("+beta"), "{compressed}");
    }

    #[test]
    fn test_context_trimming() {
        let diff = "\
diff --git a/a b/b
index 0000..1111 100644
--- a/a
+++ b/b
@@ -1,9 +1,9 @@
  1
  2
  3
+4
  5
  6
  7
  8
-9
  10
";
        let compressed = compress_git_diff(diff, 20);
        // We should keep 2 and 3 (around 4), 5 and 6 (around 4)
        // 7 and 8 (around 9), 10 (around 9). Wait! 10 is kept.
        // Wait, ' 1' is distance 3 from '+4', so it's dropped.
        assert!(!compressed.contains(" 1\n"));
        assert!(compressed.contains(" 2\n"));
        assert!(compressed.contains(" 3\n"));
        assert!(compressed.contains("+4\n"));
        assert!(compressed.contains(" 5\n"));
        assert!(compressed.contains(" 6\n"));
        assert!(compressed.contains(" 7\n"));
        assert!(compressed.contains(" 8\n"));
        assert!(compressed.contains("-9\n"));
        assert!(compressed.contains(" 10\n"));
    }

    #[test]
    fn test_hunk_scoring() {
        let mut diff = String::new();
        diff.push_str("diff --git a/file b/file\nindex 0..1 100644\n--- a/file\n+++ b/file\n");
        for i in 0..15 {
            diff.push_str(&format!("@@ -{},1 +{},1 @@\n", i, i));
            if i == 5 {
                diff.push_str("+TODO: fix this\n"); // keyword
            } else if i == 7 {
                for _ in 0..20 {
                    diff.push_str("+\n"); // dense
                }
            } else {
                diff.push_str("+\n");
            }
        }
        let compressed = compress_git_diff(&diff, 20);
        let count = compressed.matches("@@ ").count();
        assert_eq!(count, 10);
        // first (0) and last (14) must be present
        assert!(compressed.contains("@@ -0,1 +0,1 @@"));
        assert!(compressed.contains("@@ -14,1 +14,1 @@"));
        // 5 and 7 should be present due to high scores
        assert!(compressed.contains("@@ -5,1 +5,1 @@"));
        assert!(compressed.contains("@@ -7,1 +7,1 @@"));
    }
}
