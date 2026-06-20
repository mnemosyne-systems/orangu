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

use crate::*;

/// Categories of the `/auto_review` report, in display order: index 0 is
/// `Overall`, filled by the final whole-change pass; indices 1..=6 are the
/// per-file categories that `AUTO_REVIEW_FILE_CATEGORIES` maps into (its
/// entries carry these indices as their report section). The same indices
/// order `AutoReviewState::sections`.
pub(crate) const AUTO_REVIEW_CATEGORIES: [&str; 7] = [
    "Overall",
    "Code",
    "Security",
    "Memory",
    "Performance",
    "Test Suite",
    "Documentation",
];

/// The per-file categories as (report section index, prompt focus), reviewed
/// in this order — one focused LLM request per enabled category.
pub(crate) const AUTO_REVIEW_FILE_CATEGORIES: [(usize, &str); 6] = [
    (1, "correctness, error handling, and style"),
    (2, "vulnerabilities and unsafe input handling"),
    (3, "leaks, unbounded growth, and unsafe memory use"),
    (4, "inefficiencies and unnecessary work"),
    (5, "missing or broken test coverage"),
    (6, "missing or outdated documentation and comments"),
];

/// The synthesized final category of the report: the verdict for the whole
/// patch, derived from the file statuses rather than collected from the model.
pub(crate) const AUTO_REVIEW_CONCLUSION: &str = "Conclusion";

/// File extensions detected as documentation. Such files skip the
/// code-related checks and are reviewed only for the `Documentation` category.
pub(crate) const AUTO_REVIEW_DOCUMENTATION_EXTENSIONS: [&str; 16] = [
    "md", "markdown", "mkd", "mdown", "mdx", "rst", "adoc", "asciidoc", "txt", "text", "org",
    "tex", "texi", "texinfo", "pod", "rdoc",
];

/// File extensions whose changes a code review cannot act on, so a file
/// carrying one is approved at once with no category requests — the "skip
/// list". These are deterministic, machine-generated dependency lock files and
/// binary assets (images, fonts, …) whose diffs are noise or unreadable.
/// Matched alongside `AUTO_REVIEW_SKIP_FILENAMES`, which catches the lock and
/// checksum files whose extension is shared with reviewable files.
pub(crate) const AUTO_REVIEW_SKIP_EXTENSIONS: [&str; 20] = [
    // Dependency lock files: Cargo.lock, poetry.lock, Pipfile.lock,
    // Gemfile.lock, composer.lock, flake.lock, deno.lock, mix.lock, …
    "lock", //
    // Images.
    "png", "jpg", "jpeg", "gif", "bmp", "ico", "svg", "webp", "tiff", "tif", //
    // Fonts.
    "otf", "ttf", "woff", "woff2", "eot", //
    // Other binary artifacts.
    "pdf", "p12", "jks", "keystore",
];

/// Exact file names approved at once like the skip extensions: generated lock
/// and checksum files whose extension is shared with files a review must still
/// read (`package-lock.json` is JSON, `pnpm-lock.yaml` is YAML, `go.sum` would
/// match nothing), so they are matched by name rather than extension.
pub(crate) const AUTO_REVIEW_SKIP_FILENAMES: [&str; 5] = [
    "package-lock.json",
    "npm-shrinkwrap.json",
    "pnpm-lock.yaml",
    "go.sum",
    "go.work.sum",
];

/// Exact file names that must go through the full per-file review regardless of
/// their extension — build-system and metadata files whose extension would
/// otherwise misclassify them. `CMakeLists.txt` and `requirements.txt` carry a
/// `.txt` extension but are not documentation, so they are pulled back out of
/// the documentation bucket here. Extensionless metadata (`Makefile`,
/// `Dockerfile`) needs no entry: with no documentation or skip extension it
/// already falls through to the full review.
pub(crate) const AUTO_REVIEW_SOURCE_FILENAMES: [&str; 2] = ["CMakeLists.txt", "requirements.txt"];

/// File extensions forced to a full code review because their guessed MIME
/// misclassifies them: `mime_guess` maps TypeScript's `.ts`/`.mts` to a
/// `video/*` MPEG transport-stream type, which would otherwise skip the file as
/// a video asset. Listed here, they are recognized as code before the skip
/// check runs — the deliberate "fall back to a code review unless we are sure"
/// guard against a confident-looking but wrong MIME.
const AUTO_REVIEW_SOURCE_EXTENSIONS: [&str; 3] = ["ts", "mts", "cts"];

/// MIME essences (from `mime_guess`) recognized as documentation on top of the
/// curated extension list, so a markup or typesetting source whose extension is
/// not listed but whose MIME is known still skips the code-related checks. Only
/// unambiguous documentation types belong here: `text/plain` is deliberately
/// absent, since metadata files like `CMakeLists.txt` and `requirements.txt`
/// also map to it.
const AUTO_REVIEW_DOCUMENTATION_MIMES: [&str; 4] = [
    "text/markdown",
    "text/x-markdown",
    "application/x-tex",
    "application/x-texinfo",
];

/// Binary `application/*` MIME essences (from `mime_guess`) whose changes a
/// review cannot act on: archives, compiled artifacts, and packaged documents.
/// Top-level `image`, `audio`, `video`, and `font` types — and any essence
/// naming a font (`application/font-woff`, `application/vnd.ms-fontobject`) —
/// are recognized directly in `auto_review_skipped_file` and need no entry. The
/// generic `application/octet-stream` is deliberately absent: too many code
/// extensions (`.java`, …) map to it, so a real binary carrying it is caught by
/// the content-based detection instead.
const AUTO_REVIEW_SKIP_MIMES: [&str; 11] = [
    "application/pdf",
    "application/zip",
    "application/gzip",
    "application/x-tar",
    "application/x-7z-compressed",
    "application/x-bzip2",
    "application/x-rar-compressed",
    "application/java-archive",
    "application/wasm",
    "application/x-msdownload",
    "application/vnd.android.package-archive",
];

/// The base file name of `path` (the final path component), or `""` when there
/// is none.
fn auto_review_file_name(path: &str) -> &str {
    Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("")
}

/// The extension of `path` (without the dot), or `""` when there is none.
fn auto_review_extension(path: &str) -> &str {
    Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("")
}

/// The MIME type guessed for `path` from its extension (`mime_guess`), or `None`
/// when the extension is unknown. Used to widen the documentation and skip
/// detection beyond the curated extension lists.
fn auto_review_mime(path: &str) -> Option<mime_guess::Mime> {
    mime_guess::from_path(path).first()
}

/// Whether `path` is detected as documentation: its extension is on the curated
/// list, or its guessed MIME is an unambiguous documentation type
/// (`AUTO_REVIEW_DOCUMENTATION_MIMES`). A metadata file forced to a full review
/// (`auto_review_source_file`) is not pulled out here — that override is applied
/// by `auto_review_kind` — so this can still report `true` for `CMakeLists.txt`.
pub(crate) fn auto_review_documentation_file(path: &str) -> bool {
    let extension = auto_review_extension(path);
    if AUTO_REVIEW_DOCUMENTATION_EXTENSIONS
        .iter()
        .any(|known| extension.eq_ignore_ascii_case(known))
    {
        return true;
    }
    auto_review_mime(path).is_some_and(|mime| {
        AUTO_REVIEW_DOCUMENTATION_MIMES
            .iter()
            .any(|known| mime.essence_str().eq_ignore_ascii_case(known))
    })
}

/// Whether `path` must take the full per-file review regardless of how its
/// extension would otherwise classify it: a metadata/build file matched by name
/// (`AUTO_REVIEW_SOURCE_FILENAMES`), or a source file whose extension a guessed
/// MIME misreads (`AUTO_REVIEW_SOURCE_EXTENSIONS`). Takes precedence over the
/// documentation and skip checks.
pub(crate) fn auto_review_source_file(path: &str) -> bool {
    let name = auto_review_file_name(path);
    if AUTO_REVIEW_SOURCE_FILENAMES
        .iter()
        .any(|known| name.eq_ignore_ascii_case(known))
    {
        return true;
    }
    let extension = auto_review_extension(path);
    AUTO_REVIEW_SOURCE_EXTENSIONS
        .iter()
        .any(|known| extension.eq_ignore_ascii_case(known))
}

/// Whether `path` is on the skip list: a generated lock file or binary asset
/// whose diff a review cannot act on, matched by file name
/// (`AUTO_REVIEW_SKIP_FILENAMES`) or by extension (`AUTO_REVIEW_SKIP_EXTENSIONS`).
/// Such a file is approved at once, with no category requests.
pub(crate) fn auto_review_skipped_file(path: &str) -> bool {
    let name = auto_review_file_name(path);
    if AUTO_REVIEW_SKIP_FILENAMES
        .iter()
        .any(|known| name.eq_ignore_ascii_case(known))
    {
        return true;
    }
    let extension = auto_review_extension(path);
    if AUTO_REVIEW_SKIP_EXTENSIONS
        .iter()
        .any(|known| extension.eq_ignore_ascii_case(known))
    {
        return true;
    }
    // Widen the skip detection with the guessed MIME: audio, video, image, and
    // font assets (by top-level type or a font-naming essence) and the binary
    // `application/*` types are all unreviewable.
    auto_review_mime(path).is_some_and(|mime| {
        let essence = mime.essence_str();
        matches!(mime.type_().as_str(), "image" | "audio" | "video" | "font")
            || essence.contains("font")
            || AUTO_REVIEW_SKIP_MIMES
                .iter()
                .any(|known| essence.eq_ignore_ascii_case(known))
    })
}

/// Which of the three review groups a changed file falls into, decided from its
/// path. `Code` is the default — the group a file lands in unless it is
/// confidently documentation or skippable — so an unrecognized change is always
/// given a full review rather than dropped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AutoReviewKind {
    /// Approved at once with no category requests (lock files, binary assets).
    Skip,
    /// Reviewed only for the `Documentation` category.
    Documentation,
    /// Reviewed for every per-file category — the fallback.
    Code,
}

/// The review group of `path`, by name and extension alone. A forced-full
/// metadata file (`CMakeLists.txt`, …) is `Code` whatever its extension; a
/// skip-list or binary-asset file is `Skip`; a documentation file is
/// `Documentation`; everything else falls back to `Code`. Content-based binary
/// detection (`auto_review_entry_categories`) can still pull a `Code` file down
/// to a skip when its change turns out to be binary.
pub(crate) fn auto_review_kind(path: &str) -> AutoReviewKind {
    if auto_review_source_file(path) {
        return AutoReviewKind::Code;
    }
    if auto_review_skipped_file(path) {
        return AutoReviewKind::Skip;
    }
    if auto_review_documentation_file(path) {
        return AutoReviewKind::Documentation;
    }
    AutoReviewKind::Code
}

/// The per-file categories enabled for a review group: none for `Skip`, only
/// `Documentation` (the last per-file category) for `Documentation`, and every
/// category for `Code`.
fn auto_review_kind_categories(kind: AutoReviewKind) -> &'static [(usize, &'static str)] {
    match kind {
        // Approved at once: no category requests.
        AutoReviewKind::Skip => &[],
        // `Documentation` is the last per-file category.
        AutoReviewKind::Documentation => {
            &AUTO_REVIEW_FILE_CATEGORIES[AUTO_REVIEW_FILE_CATEGORIES.len() - 1..]
        }
        AutoReviewKind::Code => &AUTO_REVIEW_FILE_CATEGORIES[..],
    }
}

/// Whether `patch` is the diff git produces for a binary file — the `Binary
/// files a/… and b/… differ` marker it writes in place of a textual hunk. Such
/// a change carries no reviewable lines, so the file is skipped even when its
/// extension is unknown or misleading.
pub(crate) fn auto_review_binary_patch(patch: &str) -> bool {
    patch.lines().any(|line| {
        let line = line.trim();
        line.starts_with("Binary files ") && line.ends_with(" differ")
    })
}

/// Whether an `infer` magic-number match is a binary file type a review cannot
/// act on: an image, audio, video, archive, document, font, e-book, or
/// executable. Plain text and unrecognized content are not binary.
fn auto_review_binary_matcher(kind: infer::Type) -> bool {
    use infer::MatcherType::{App, Archive, Audio, Book, Doc, Font, Image, Video};
    matches!(
        kind.matcher_type(),
        App | Archive | Audio | Book | Doc | Font | Image | Video
    )
}

/// Whether `bytes` are detected as binary by their magic number (`infer`). An
/// undetected buffer is treated as text, so the file falls back to a code
/// review.
pub(crate) fn auto_review_binary_content(bytes: &[u8]) -> bool {
    infer::get(bytes).is_some_and(auto_review_binary_matcher)
}

/// Whether the file at `path` is detected as binary by its on-disk magic number
/// (`infer`). Only the file's header is read — magic numbers live at the start —
/// so a large asset is not slurped whole. Best-effort: an unreadable or missing
/// file (e.g. a deletion that is no longer on disk) is treated as not binary,
/// leaving the content-based skip to the patch marker.
fn auto_review_path_is_binary(path: &Path) -> bool {
    use std::io::Read;
    let Ok(file) = std::fs::File::open(path) else {
        return false;
    };
    // 8 KiB comfortably covers every magic number `infer` looks for.
    let mut header = Vec::new();
    if file.take(8192).read_to_end(&mut header).is_err() {
        return false;
    }
    auto_review_binary_content(&header)
}

/// The categories scanned for one changed file: its path-based group
/// (`auto_review_kind`), with a `Code` file downgraded to a skip when its change
/// is binary — git diffed it as binary (`auto_review_binary_patch`), or its
/// on-disk content sniffs as binary (`auto_review_path_is_binary`, when
/// `repo_root` is known). Only the `Code` fallback is reconsidered: a file
/// already recognized as documentation or skipped, or forced to a full review,
/// keeps its group, so the binary check can never override a confident decision.
pub(crate) fn auto_review_entry_categories(
    path: &str,
    patch: &str,
    repo_root: Option<&Path>,
) -> &'static [(usize, &'static str)] {
    let kind = auto_review_kind(path);
    if kind == AutoReviewKind::Code {
        let binary = auto_review_binary_patch(patch)
            || repo_root.is_some_and(|root| auto_review_path_is_binary(&root.join(path)));
        if binary {
            return auto_review_kind_categories(AutoReviewKind::Skip);
        }
    }
    auto_review_kind_categories(kind)
}

/// The Alt+r reject window of `/auto_review`: the category receiving the
/// comment and the multi-line Markdown comment editor. Tab moves the focus
/// between the selector and the editor.
pub(crate) struct AutoReviewReject {
    /// Index of the chosen category, into `AUTO_REVIEW_CATEGORIES`.
    pub(crate) category: usize,
    /// `true` while the focus is on the category selector.
    pub(crate) selector_focused: bool,
    /// The comment editor; `Enter` inserts a newline into its buffer.
    pub(crate) editor: InputState,
}

/// One navigable item in the rendered report: which finding it is (so it can
/// be removed) and the report line range it occupies (so it can be highlighted
/// and scrolled to). Built by `AutoReviewState::build_report`, in the order the
/// items appear down the left pane.
pub(crate) struct AutoReviewItem {
    /// First report line of the item (inclusive).
    pub(crate) start: usize,
    /// One past the last report line of the item (exclusive).
    pub(crate) end: usize,
    pub(crate) kind: AutoReviewItemKind,
}

/// What a navigable report item refers to: a per-category finding, or a
/// `Conclusion` entry standing in for a whole file.
pub(crate) enum AutoReviewItemKind {
    /// A finding in `sections[section]` at the given index.
    Finding { section: usize, index: usize },
    /// A `Conclusion` entry (a rejected or not-reviewed file), carrying the
    /// file path it stands for. Removing it approves that whole file.
    Conclusion { path: String },
}

/// Interactive state for `/auto_review` mode.
pub(crate) struct AutoReviewState {
    pub(crate) files: Vec<ReviewEntry>,
    /// The file highlighted in the right pane: the one being reviewed while
    /// the run is in progress, or the one picked with Alt+j/Alt+k while
    /// browsing afterwards. `None` once the run ends, until the user
    /// navigates.
    pub(crate) selected: Option<usize>,
    /// The report item highlighted in the left pane while browsing: an index
    /// into `build_report`'s item list, moved by Up/Down and acted on by `-`.
    /// `None` until the user navigates (or once the report has no items left).
    pub(crate) selected_item: Option<usize>,
    /// The file whose categories are currently being reviewed; its status box
    /// blinks a white dot. `None` during the whole-change pass and after the
    /// run.
    pub(crate) reviewing: Option<usize>,
    /// Index of the first report line shown in the left pane.
    pub(crate) scroll: usize,
    /// Horizontal pan offset for the left pane.
    pub(crate) x_offset: usize,
    /// Collected findings per category, in `AUTO_REVIEW_CATEGORIES` order.
    pub(crate) sections: [Vec<String>; AUTO_REVIEW_CATEGORIES.len()],
    /// Text for the status area at the top of the screen: the file and
    /// category being worked on while the run is in progress.
    pub(crate) status: String,
    /// When the run started, for the status area's `Time:` element.
    pub(crate) started: std::time::Instant,
    /// When the run ended (done or cancelled); freezes the `Time:` element.
    pub(crate) finished: Option<std::time::Instant>,
    /// Projected instant the run will finish, set after each completed request
    /// from the average time per request so far; drives the `Estimated:`
    /// element, which counts down toward it. `None` until the first request
    /// completes.
    pub(crate) projected_finish: Option<std::time::Instant>,
    /// The run finished every file and the overall pass.
    pub(crate) done: bool,
    /// The run was cancelled with Esc Esc.
    pub(crate) cancelled: bool,
    /// When set, the Alt+r reject window is open over the panes (browse
    /// phase only).
    pub(crate) reject: Option<AutoReviewReject>,
    /// The model performing the review, shown after the `Conclusion` verdict.
    pub(crate) model: String,
}

impl AutoReviewState {
    pub(crate) fn new(launch: ReviewLaunch) -> Self {
        Self {
            files: launch.files,
            selected: None,
            selected_item: None,
            reviewing: None,
            scroll: 0,
            x_offset: 0,
            sections: Default::default(),
            status: "Starting".to_string(),
            started: std::time::Instant::now(),
            finished: None,
            projected_finish: None,
            done: false,
            cancelled: false,
            reject: None,
            model: String::new(),
        }
    }

    /// The total time spent on the run so far, frozen once it ends.
    pub(crate) fn elapsed(&self) -> std::time::Duration {
        self.finished
            .unwrap_or_else(std::time::Instant::now)
            .saturating_duration_since(self.started)
    }

    /// Recompute the projected finish from the work done so far: the average
    /// time per completed request extrapolated over the requests still to run.
    /// Called after each request completes so the `Estimated:` element counts
    /// down toward zero as the run progresses. A no-op until at least one
    /// request has completed, since there is nothing to average yet.
    pub(crate) fn update_estimate(&mut self, completed: usize, total_requests: usize) {
        if completed == 0 {
            return;
        }
        let total = total_requests.max(1);
        // Saturating casts so a pathological request count can't overflow the
        // u32 arithmetic; in practice both are well under a few hundred.
        let remaining = u32::try_from(total.saturating_sub(completed)).unwrap_or(u32::MAX);
        let completed = u32::try_from(completed).unwrap_or(u32::MAX);
        // Multiply the elapsed time before dividing so the per-request average
        // keeps full (nanosecond) resolution instead of truncating up front,
        // and saturate the multiply so the Duration can't overflow.
        let remaining = self.elapsed().saturating_mul(remaining) / completed;
        self.projected_finish = Some(std::time::Instant::now() + remaining);
    }

    /// Time still expected before the run finishes, counting down from the last
    /// projection. `None` until the first request completes, and once the run
    /// ends (the `Estimated:` element drops away, leaving only the frozen
    /// `Time:`).
    pub(crate) fn estimated_remaining(&self) -> Option<std::time::Duration> {
        if self.finished.is_some() {
            return None;
        }
        self.projected_finish
            .map(|finish| finish.saturating_duration_since(std::time::Instant::now()))
    }

    /// The status area's full text: the current activity, the total time spent
    /// on the run, and — while the run is in progress — the estimated time
    /// left, all after the progress information.
    pub(crate) fn status_text(&self) -> String {
        let mut text = format!(
            "{}  Time: {}",
            self.status,
            orangu::tui::format_status_duration(self.elapsed()),
        );
        if let Some(remaining) = self.estimated_remaining() {
            text.push_str("  Estimated: ");
            text.push_str(&orangu::tui::format_status_duration(remaining));
        }
        text
    }

    /// The patch verdict opening the `Conclusion` category: `orangu approves
    /// this patch` when every file is approved, otherwise `orangu rejects
    /// this patch`.
    pub(crate) fn conclusion_verdict(&self) -> &'static str {
        let all_approved = self
            .files
            .iter()
            .all(|file| file.status == ReviewStatus::Approved);
        if all_approved {
            "orangu approves this patch"
        } else {
            "orangu rejects this patch"
        }
    }

    /// The report's closing attribution as Markdown: `Generated by: **orangu
    /// <version>**`, with the reviewing model in parentheses (outside the bold)
    /// when its name is known.
    pub(crate) fn generated_by_markdown(&self) -> String {
        let model = if self.model.is_empty() {
            String::new()
        } else {
            format!(" ({})", self.model)
        };
        format!("Generated by: **orangu {VERSION}**{model}")
    }

    /// The `Conclusion` verdict row as rendered for the console: the verdict
    /// in bold, standing alone.
    pub(crate) fn conclusion_verdict_line(&self) -> String {
        format!("\x1b[1m{}\x1b[0m", self.conclusion_verdict())
    }

    /// The rejected and not-reviewed files listed under the `Conclusion`
    /// verdict, each as its source file path and the rendered line (in Markdown
    /// bold), grouped by their status, rejected first. Empty when every file is
    /// approved.
    pub(crate) fn conclusion_entries(&self) -> Vec<(String, String)> {
        let mut entries = Vec::new();
        for file in &self.files {
            if file.status == ReviewStatus::Rejected {
                entries.push((file.path.clone(), format!("Rejected: **{}**", file.path)));
            }
        }
        for file in &self.files {
            if file.status == ReviewStatus::Unreviewed {
                entries.push((
                    file.path.clone(),
                    format!("Not reviewed: **{}**", file.path),
                ));
            }
        }
        entries
    }

    /// The rejected and not-reviewed files listed under the `Conclusion`
    /// verdict (in Markdown bold), grouped by their status, rejected first.
    /// Empty when every file is approved.
    pub(crate) fn conclusion_findings(&self) -> Vec<String> {
        self.conclusion_entries()
            .into_iter()
            .map(|(_, line)| line)
            .collect()
    }

    /// The left-pane report, rendered for the console: each category as a
    /// bold heading (the `##` markers of the Markdown report are consumed,
    /// not displayed) followed by its findings as a bullet list with the
    /// `**file**` names resolved to bold, with a dimmed placeholder while the
    /// run is still in progress, ending with the synthesized `Conclusion`.
    pub(crate) fn report_lines(&self) -> Vec<String> {
        self.build_report().0
    }

    /// The navigable items of the report — the per-category findings and the
    /// `Conclusion` entries — in the order they appear down the left pane, each
    /// with the report line range it occupies. Empty while the run is still in
    /// progress (only placeholders are shown then).
    pub(crate) fn report_items(&self) -> Vec<AutoReviewItem> {
        self.build_report().1
    }

    /// Render the report lines and, alongside them, the navigable items (each
    /// with its line range). Up/Down move between these items and `-` removes
    /// the highlighted one, so the renderer and the browse loop agree on what a
    /// "line" the user is pointing at maps back to.
    pub(crate) fn build_report(&self) -> (Vec<String>, Vec<AutoReviewItem>) {
        let pending = !(self.done || self.cancelled);
        let mut lines = Vec::new();
        let mut items = Vec::new();
        for (index, name) in AUTO_REVIEW_CATEGORIES.iter().enumerate() {
            lines.push(format!("\x1b[1m{name}\x1b[0m"));
            lines.push(String::new());
            let section = &self.sections[index];
            if section.is_empty() {
                if pending {
                    lines.push("\x1b[2m(pending)\x1b[0m".to_string());
                } else {
                    lines.push("No issues found".to_string());
                }
            } else {
                for (finding_index, finding) in section.iter().enumerate() {
                    let start = lines.len();
                    let bullet = render_markdown_for_console(&auto_review_finding_bullet(finding));
                    lines.extend(bullet.lines().map(str::to_string));
                    items.push(AutoReviewItem {
                        start,
                        end: lines.len(),
                        kind: AutoReviewItemKind::Finding {
                            section: index,
                            index: finding_index,
                        },
                    });
                }
            }
            lines.push(String::new());
        }
        lines.push(format!("\x1b[1m{AUTO_REVIEW_CONCLUSION}\x1b[0m"));
        lines.push(String::new());
        if pending {
            lines.push("\x1b[2m(pending)\x1b[0m".to_string());
        } else {
            // The verdict stands alone in bold; the affected files follow as a
            // bullet list.
            lines.push(self.conclusion_verdict_line());
            let entries = self.conclusion_entries();
            if !entries.is_empty() {
                lines.push(String::new());
                for (path, line) in entries {
                    let start = lines.len();
                    lines.push(render_markdown_for_console(&format!("- {line}")));
                    items.push(AutoReviewItem {
                        start,
                        end: lines.len(),
                        kind: AutoReviewItemKind::Conclusion { path },
                    });
                }
            }
        }
        // The report closes with the orangu version and reviewing model.
        lines.push(String::new());
        lines.push(render_markdown_for_console(&self.generated_by_markdown()));
        (lines, items)
    }

    /// The report line range of the highlighted item, for the renderer to
    /// invert. `None` when no item is highlighted or the index is stale.
    pub(crate) fn selected_item_span(&self) -> Option<(usize, usize)> {
        let index = self.selected_item?;
        self.report_items()
            .get(index)
            .map(|item| (item.start, item.end))
    }

    /// Clamp scroll/pan offsets to the report's size.
    pub(crate) fn clamp(&mut self, body_height: usize, left_width: usize) {
        let lines = self.report_lines();
        self.scroll = self.scroll.min(lines.len().saturating_sub(body_height));
        let content_width = lines
            .iter()
            .map(|line| orangu::tui::visible_line_width(line))
            .max()
            .unwrap_or(0);
        self.x_offset = self.x_offset.min(content_width.saturating_sub(left_width));
    }

    /// Move the highlight to the next file; from no highlight (after the run
    /// ended) Alt+j starts at the first file.
    pub(crate) fn select_next(&mut self) {
        if self.files.is_empty() {
            return;
        }
        self.selected = Some(match self.selected {
            Some(index) => (index + 1).min(self.files.len() - 1),
            None => 0,
        });
    }

    /// Move the highlight to the previous file; from no highlight (after the
    /// run ended) Alt+k starts at the last file.
    pub(crate) fn select_prev(&mut self) {
        if self.files.is_empty() {
            return;
        }
        self.selected = Some(match self.selected {
            Some(index) => index.saturating_sub(1),
            None => self.files.len() - 1,
        });
    }

    /// The path of the file highlighted in the right pane, if any.
    pub(crate) fn selected_path(&self) -> Option<String> {
        self.selected
            .and_then(|index| self.files.get(index))
            .map(|file| file.path.clone())
    }

    /// Alt+a: approve the highlighted file and remove every finding recorded
    /// against it from all report categories — the model's findings and any
    /// Alt+r rejection comments alike. The Conclusion follows the file
    /// statuses, so it updates with the approval.
    pub(crate) fn approve_selected(&mut self) {
        let Some(index) = self.selected else {
            return;
        };
        let Some(path) = self.files.get(index).map(|file| file.path.clone()) else {
            return;
        };
        self.approve_path(&path);
        self.clamp_selected_item();
    }

    /// The two finding prefixes that bind a finding to `path`: `**path**:` (no
    /// line, e.g. an Alt+r comment) and `**path:` (a `**path:line**:`
    /// location), as written by `apply_category_result`, `commit_reject`, and
    /// the failure records.
    fn finding_prefixes(path: &str) -> (String, String) {
        (format!("**{path}**:"), format!("**{path}:"))
    }

    /// Approve `path`: turn its dot green and drop every finding recorded
    /// against it from all report categories. Shared by Alt+a and by removing
    /// the file's `Conclusion` item.
    fn approve_path(&mut self, path: &str) {
        if let Some(file) = self.files.iter_mut().find(|file| file.path == path) {
            file.status = ReviewStatus::Approved;
        }
        let (without_line, with_line) = Self::finding_prefixes(path);
        for section in &mut self.sections {
            section.retain(|finding| {
                !finding.starts_with(&without_line) && !finding.starts_with(&with_line)
            });
        }
    }

    /// Whether any report category still holds a finding bound to `path`.
    fn file_has_findings(&self, path: &str) -> bool {
        let (without_line, with_line) = Self::finding_prefixes(path);
        self.sections.iter().any(|section| {
            section.iter().any(|finding| {
                finding.starts_with(&without_line) || finding.starts_with(&with_line)
            })
        })
    }

    /// The file path a finding is bound to — the `path` inside its leading
    /// `**path**` or `**path:line**` location — or `None` for a finding with no
    /// file location (e.g. a whole-change `Overall` bullet).
    fn finding_path(finding: &str) -> Option<&str> {
        let rest = finding.strip_prefix("**")?;
        let inner = &rest[..rest.find("**")?];
        Some(inner.split_once(':').map_or(inner, |(path, _)| path))
    }

    /// Up/Down while browsing: move the highlight to the next report item,
    /// starting at the first item from no highlight. Also moves the file
    /// highlight to the item's file so Alt+a/Alt+r act on it.
    pub(crate) fn select_next_item(&mut self) {
        let items = self.report_items();
        if items.is_empty() {
            self.selected_item = None;
            return;
        }
        let next = match self.selected_item {
            Some(index) => (index + 1).min(items.len() - 1),
            None => 0,
        };
        self.selected_item = Some(next);
        self.sync_selected_to_item(&items);
    }

    /// Up/Down while browsing: move the highlight to the previous report item,
    /// starting at the last item from no highlight.
    pub(crate) fn select_prev_item(&mut self) {
        let items = self.report_items();
        if items.is_empty() {
            self.selected_item = None;
            return;
        }
        let prev = match self.selected_item {
            Some(index) => index.saturating_sub(1),
            None => items.len() - 1,
        };
        self.selected_item = Some(prev);
        self.sync_selected_to_item(&items);
    }

    /// The item index at which each category begins in `items`, in display
    /// order: a per-category section's first finding, then the `Conclusion`
    /// group's first entry. PageUp/PageDown jump the highlight between these so
    /// a long report can be walked category by category. Categories with no
    /// findings have no item and so contribute no start.
    fn category_starts(items: &[AutoReviewItem]) -> Vec<usize> {
        let mut starts = Vec::new();
        let mut last_key: Option<usize> = None;
        for (index, item) in items.iter().enumerate() {
            // The Conclusion entries share a single key, distinct from every
            // per-category section index, so they form one trailing category.
            let key = match &item.kind {
                AutoReviewItemKind::Finding { section, .. } => *section,
                AutoReviewItemKind::Conclusion { .. } => usize::MAX,
            };
            if last_key != Some(key) {
                starts.push(index);
                last_key = Some(key);
            }
        }
        starts
    }

    /// Move the item highlight to `target`, scrolling the category heading (two
    /// lines above the first finding) to the top so the whole category — not
    /// just the finding — comes into view, and pointing the file highlight at
    /// the item's file.
    fn jump_to_item(&mut self, target: usize, items: &[AutoReviewItem]) {
        self.selected_item = Some(target);
        if let Some(item) = items.get(target) {
            self.scroll = item.start.saturating_sub(2);
        }
        self.sync_selected_to_item(items);
    }

    /// PageDown while browsing: jump the highlight to the first item of the next
    /// category that has findings (the `Conclusion` entries count as the final
    /// category). From no highlight it starts at the first such category. A
    /// no-op once the highlight is already in the last category, so PageDown
    /// never jumps backward.
    pub(crate) fn select_next_category(&mut self) {
        let items = self.report_items();
        let starts = Self::category_starts(&items);
        if starts.is_empty() {
            self.selected_item = None;
            return;
        }
        let target = match self.selected_item {
            Some(current) => starts.iter().copied().find(|&start| start > current),
            None => Some(starts[0]),
        };
        if let Some(target) = target {
            self.jump_to_item(target, &items);
        }
    }

    /// PageUp while browsing: jump the highlight to the first item of the
    /// previous category that has findings. From no highlight it starts at the
    /// last such category. A no-op once the highlight is already in the first
    /// category.
    pub(crate) fn select_prev_category(&mut self) {
        let items = self.report_items();
        let starts = Self::category_starts(&items);
        if starts.is_empty() {
            self.selected_item = None;
            return;
        }
        let target = match self.selected_item {
            Some(current) => {
                // The start of the highlight's own category, then the start of
                // the category before it.
                let current_start = starts
                    .iter()
                    .copied()
                    .rfind(|&start| start <= current)
                    .unwrap_or(starts[0]);
                starts.iter().copied().rfind(|&start| start < current_start)
            }
            None => starts.last().copied(),
        };
        if let Some(target) = target {
            self.jump_to_item(target, &items);
        }
    }

    /// Point the right-pane file highlight at the file owning the highlighted
    /// item, so the panes agree and Alt+a/Alt+r act on the right file. Items
    /// without a file location (whole-change `Overall` bullets) leave it put.
    fn sync_selected_to_item(&mut self, items: &[AutoReviewItem]) {
        let Some(item) = self.selected_item.and_then(|index| items.get(index)) else {
            return;
        };
        let path = match &item.kind {
            AutoReviewItemKind::Conclusion { path } => Some(path.as_str()),
            AutoReviewItemKind::Finding { section, index } => self.sections[*section]
                .get(*index)
                .and_then(|finding| Self::finding_path(finding)),
        };
        if let Some(path) = path
            && let Some(file_index) = self.files.iter().position(|file| file.path == path)
        {
            self.selected = Some(file_index);
        }
    }

    /// `-` while browsing: remove the highlighted item from its list. Removing
    /// a finding that leaves its file with no other findings approves that file
    /// (and so drops it from the Conclusion); removing a `Conclusion` item
    /// approves the whole file it stands for, clearing all of its findings.
    pub(crate) fn remove_selected_item(&mut self) {
        let items = self.report_items();
        let Some(item) = self.selected_item.and_then(|index| items.get(index)) else {
            return;
        };
        match &item.kind {
            AutoReviewItemKind::Finding { section, index } => {
                let (section, index) = (*section, *index);
                let path = self.sections[section]
                    .get(index)
                    .and_then(|finding| Self::finding_path(finding).map(str::to_string));
                self.sections[section].remove(index);
                // Approving the file once its last finding is gone drops it from
                // the Conclusion; a finding with no file location just leaves.
                if let Some(path) = path
                    && !self.file_has_findings(&path)
                {
                    self.approve_path(&path);
                }
            }
            AutoReviewItemKind::Conclusion { path } => {
                let path = path.clone();
                self.approve_path(&path);
            }
        }
        self.clamp_selected_item();
    }

    /// Keep `selected_item` pointing at a real item after the report shrinks:
    /// clamp it to the last item, or clear it when nothing is left.
    fn clamp_selected_item(&mut self) {
        let count = self.report_items().len();
        self.selected_item = match self.selected_item {
            _ if count == 0 => None,
            Some(index) => Some(index.min(count - 1)),
            None => None,
        };
    }

    /// Scroll so the highlighted item's lines sit within the visible body,
    /// nudging the report up or down only as far as needed.
    pub(crate) fn ensure_item_visible(&mut self, body_height: usize) {
        let Some((start, end)) = self.selected_item_span() else {
            return;
        };
        if start < self.scroll {
            self.scroll = start;
        } else if end > self.scroll + body_height {
            self.scroll = end.saturating_sub(body_height);
        }
    }

    /// Alt+r: open the reject window for the highlighted file, starting on
    /// the category selector.
    pub(crate) fn open_reject(&mut self) {
        if self.selected_path().is_some() {
            self.reject = Some(AutoReviewReject {
                category: 0,
                selector_focused: true,
                editor: InputState::default(),
            });
        }
    }

    /// Save the reject window: mark the highlighted file rejected and append
    /// the comment (when non-empty) to the chosen category, prefixed with the
    /// file's path in Markdown bold. Each Alt+r adds another comment.
    pub(crate) fn commit_reject(&mut self) {
        let Some(reject) = self.reject.take() else {
            return;
        };
        let Some(index) = self.selected else {
            return;
        };
        let Some(path) = self.files.get(index).map(|file| file.path.clone()) else {
            return;
        };
        self.set_file_status(index, ReviewStatus::Rejected);
        let text = reject.editor.as_str().trim().to_string();
        if !text.is_empty() {
            self.sections[reject.category].push(format!("**{path}**: {text}"));
        }
    }

    /// Append one category review's findings — prefixed with the file's path
    /// in Markdown bold — to the matching report section, so the left pane
    /// fills in category by category as the run progresses.
    pub(crate) fn apply_category_result(
        &mut self,
        index: usize,
        section: usize,
        findings: Vec<String>,
    ) {
        let Some(path) = self.files.get(index).map(|file| file.path.clone()) else {
            return;
        };
        for finding in findings {
            // The model prefixes each finding with the affected line (or range);
            // fold it into the file's location so the bullet reads
            // `path:line: <finding>`.
            let (line, body) = auto_review_split_line(&finding);
            let location = auto_review_finding_location(&path, line);
            self.sections[section].push(format!("{location}: {body}"));
        }
    }

    /// Auto-mark a file's dot once all its category reviews have run.
    pub(crate) fn set_file_status(&mut self, index: usize, status: ReviewStatus) {
        if let Some(file) = self.files.get_mut(index) {
            file.status = status;
        }
    }

    /// Record a failed per-category request; the failure is noted in the
    /// `Overall` section.
    pub(crate) fn record_failure(&mut self, index: usize, category: &str, error: &Error) {
        if let Some(file) = self.files.get(index) {
            self.sections[0].push(format!(
                "**{}**: {category} review failed: {error:#}",
                file.path
            ));
        }
    }

    /// Record a failed whole-change request in the `Overall` section.
    pub(crate) fn record_overall_failure(&mut self, error: &Error) {
        self.sections[0].push(format!("Overall review failed: {error:#}"));
    }

    /// Append the whole-change pass's findings to the `Overall` category.
    pub(crate) fn apply_overall(&mut self, text: &str) {
        for line in text.lines() {
            if let Some(finding) = auto_review_finding_body(line) {
                self.sections[0].push(finding);
            }
        }
    }

    /// Mark the run cancelled (Esc Esc). The highlight and the blinking dot
    /// are cleared: nothing is being reviewed anymore.
    pub(crate) fn cancel(&mut self) {
        self.cancelled = true;
        self.reviewing = None;
        self.selected = None;
        self.finished = Some(std::time::Instant::now());
        self.status = "Cancelled".to_string();
    }

    /// Mark the run complete. The highlight and the blinking dot are cleared:
    /// nothing is being reviewed anymore.
    pub(crate) fn finish(&mut self) {
        self.done = true;
        self.reviewing = None;
        self.selected = None;
        self.finished = Some(std::time::Instant::now());
        self.status = "Done".to_string();
    }
}

/// The overall-progress part of the status area: completed requests out of
/// the run's total (one request per enabled category per file, plus the final
/// whole-change pass).
pub(crate) fn auto_review_progress_label(completed: usize, total_requests: usize) -> String {
    // The whole-change pass always counts as one request, so the total is
    // never zero; guard anyway.
    let total = total_requests.max(1);
    let percent = completed * 100 / total;
    format!("Progress: {completed}/{total} ({percent}%)")
}

/// A finding as a Markdown bullet. A multi-line finding (an Alt+r rejection
/// comment) keeps its newlines, with the continuation lines indented two
/// spaces so they stay inside the bullet.
pub(crate) fn auto_review_finding_bullet(finding: &str) -> String {
    let mut lines = finding.lines();
    let mut bullet = format!("- {}", lines.next().unwrap_or(""));
    for line in lines {
        bullet.push_str("\n  ");
        bullet.push_str(line);
    }
    bullet
}

/// Split a model finding into its leading line reference — a line number or a
/// `<start>-<end>` range, as `build_auto_review_category_prompt` asks for — and
/// the remaining finding text. Returns `(None, finding)` (trimmed) when the
/// model gave no usable line reference, so a finding without one still records
/// cleanly.
pub(crate) fn auto_review_split_line(finding: &str) -> (Option<&str>, &str) {
    if let Some((head, rest)) = finding.split_once(':') {
        let token = head.trim();
        let is_line_reference = token.starts_with(|ch: char| ch.is_ascii_digit())
            && token.chars().all(|ch| ch.is_ascii_digit() || ch == '-');
        if is_line_reference {
            return (Some(token), rest.trim());
        }
    }
    (None, finding.trim())
}

/// The Markdown-bold location prefix for a finding: `**path:line**` when the
/// model gave a line reference, otherwise `**path**`, so the rendered bullet
/// reads `path:line: <finding>` (or `path: <finding>` without a line).
pub(crate) fn auto_review_finding_location(path: &str, line: Option<&str>) -> String {
    match line {
        Some(line) => format!("**{path}:{line}**"),
        None => format!("**{path}**"),
    }
}

/// Whether `text` is a "no findings" placeholder rather than a real finding:
/// empty, or a `None`/`no issues`/... word — possibly with a trailing
/// parenthetical justification (`None (no direct memory risk)`) or surrounding
/// punctuation. The model emits these when a category is clean, and they must
/// never reach the report.
pub(crate) fn auto_review_is_placeholder(text: &str) -> bool {
    const PHRASES: [&str; 6] = [
        "none",
        "no findings",
        "no issues",
        "no issues found",
        "nothing",
        "n/a",
    ];
    // Drop a trailing `(...)` justification, but only when something precedes
    // it — a finding that is wholly parenthesized is kept as content.
    let core = match text
        .strip_suffix(')')
        .and_then(|head| head.rsplit_once('('))
    {
        Some((before, _)) if !before.trim().is_empty() => before.trim(),
        _ => text.trim(),
    };
    let lower = core.to_ascii_lowercase();
    // The whole text is a bare placeholder (possibly with trailing `.`/`!`).
    let trimmed = lower.trim_end_matches(['.', '!']);
    if trimmed.is_empty() || PHRASES.contains(&trimmed) {
        return true;
    }
    // Or it opens with a placeholder as its own sentence — `None. <prose>` is
    // still a "nothing to report" affirmation, so the whole finding is dropped.
    // A word that merely starts with the phrase (`None of the changes …`) keeps
    // going, so it is not mistaken for one.
    PHRASES.iter().any(|phrase| {
        lower
            .strip_prefix(phrase)
            .is_some_and(|rest| rest.starts_with(['.', '!', ',', ';', ':']))
    })
}

/// The body of a finding line: bullet markers and list numbering stripped;
/// `None` for blank lines and "no findings" placeholders.
pub(crate) fn auto_review_finding_body(line: &str) -> Option<String> {
    let body = line.trim().trim_start_matches(['-', '*', '•']).trim();
    // Strip a "1." / "2)" numbered-list prefix.
    let body = match body.split_once(['.', ')']) {
        Some((number, rest))
            if !number.is_empty() && number.chars().all(|ch| ch.is_ascii_digit()) =>
        {
            rest.trim_start()
        }
        _ => body,
    };
    // A stray `VERDICT: …` line is never a finding — the whole-change pass
    // sometimes emits one even though only findings were asked for, and the
    // verdict is already recorded elsewhere — so drop it.
    if auto_review_header_rest(body, "verdict").is_some() {
        return None;
    }
    // A finding may carry a leading `line:` / `start-end:` reference; look past
    // it for the placeholder check. A clean category may also fill the line
    // slot itself with `None`, leaving the bare line `None: None`, so the
    // actual finding text is whatever follows the last colon — drop the line
    // when that, or the whole body, is a placeholder.
    let (_, text) = auto_review_split_line(body);
    let finding_text = text.rsplit_once(':').map_or(text, |(_, last)| last);
    if auto_review_is_placeholder(text) || auto_review_is_placeholder(finding_text) {
        None
    } else {
        Some(body.to_string())
    }
}

/// Recognize a `name` header line (`VERDICT:`, `**Findings:**`, `## FINDINGS`,
/// ...) and return the rest of the line after the colon. A name followed by
/// anything other than a colon (or end of line) is not a header, so a finding
/// like `verdict handling is wrong ...` stays a finding.
pub(crate) fn auto_review_header_rest<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let cleaned = line.trim_start_matches(['#', '*', '_', '`', ' ']);
    if cleaned.len() < name.len() || !cleaned[..name.len()].eq_ignore_ascii_case(name) {
        return None;
    }
    let rest = cleaned[name.len()..]
        .trim_start_matches(['*', '_', '`'])
        .trim_start();
    if let Some(rest) = rest.strip_prefix(':') {
        return Some(rest.trim_start_matches(['*', '_', '`', ' ']));
    }
    rest.is_empty().then_some("")
}

/// Parse one per-category auto review response in the requested
/// `VERDICT:`/`FINDINGS:` format — the exact format that
/// `build_auto_review_category_prompt` asks the model for: the explicit
/// verdict (when one was found) and the findings list. Markdown decoration
/// around the headers is tolerated and "None" placeholders are dropped.
pub(crate) fn parse_auto_review_category_response(text: &str) -> (Option<bool>, Vec<String>) {
    let mut approved = None;
    let mut findings = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // A `VERDICT:` line carries the approval answer and nothing else.
        if let Some(rest) = auto_review_header_rest(line, "verdict") {
            let verdict = rest.to_ascii_lowercase();
            if verdict.contains("reject") {
                approved = Some(false);
            } else if verdict.contains("approve") {
                approved = Some(true);
            }
            continue;
        }
        // Every other line is a finding: drop a `FINDINGS:` header (keeping
        // any inline finding after the colon), then strip bullet markers and
        // "None" placeholders.
        let body = auto_review_header_rest(line, "findings").unwrap_or(line);
        if let Some(finding) = auto_review_finding_body(body) {
            findings.push(finding);
        }
    }
    (approved, findings)
}

/// The per-file, per-category prompt: ask for a verdict plus findings for one
/// category only, in a fixed plain-text format that
/// `parse_auto_review_category_response` understands. The diff leads and the
/// category instruction follows, so a file's category requests share their
/// prefix and the server's prompt cache (e.g. llama.cpp) can reuse the
/// processed diff across them.
pub(crate) fn build_auto_review_category_prompt(
    path: &str,
    category: &str,
    focus: &str,
    patch: &str,
) -> String {
    format!(
        "You are performing an automated code review of the changes made to `{path}` in the diff below.\n\
         \n\
         ```diff\n{patch}\n```\n\
         \n\
         Review only the changes — the added, removed, and modified lines — for {category} issues ({focus}), and judge how the changes fit into the surrounding context. Do not review pre-existing content the change does not touch.\n\
         \n\
         Respond in exactly this format, with no other prose:\n\
         \n\
         VERDICT: APPROVE or REJECT\n\
         FINDINGS:\n\
         - <line>: <finding, or None>\n\
         \n\
         List at most five findings, one short line each, prefixed with the affected line number — or range, as `<start>-<end>` — in the new version of the file (the right side of the diff, the lines marked with `+` or unchanged). Only report real {category} issues introduced by the changes. Answer REJECT only when a finding must be fixed before merging; otherwise answer APPROVE."
    )
}

/// The whole-change prompt for the final pass of the run: every per-file
/// verdict and finding collected so far is summarized for the model, which
/// answers with a few bullet points on how the changes fit together. The
/// bullets land in the `Overall` category via `AutoReviewState::apply_overall`.
pub(crate) fn build_auto_review_overall_prompt(state: &AutoReviewState) -> String {
    let mut summary = String::new();
    for file in &state.files {
        summary.push_str(&format!(
            "{}: {}\n",
            file.path,
            review_status_label(file.status)
        ));
    }
    for (index, name) in AUTO_REVIEW_CATEGORIES.iter().enumerate().skip(1) {
        for finding in &state.sections[index] {
            summary.push_str(&format!("{name}: {finding}\n"));
        }
    }
    format!(
        "You are performing an automated code review and have reviewed each changed file, with the results below. Describe briefly how the changes fit together as one change set — readiness, risk, and common themes — as 2 to 6 short bullet points, one line each. Respond with only the bullet points.\n\n{summary}"
    )
}

/// Build the auto review exit report: the lines rendered for the output
/// window, and the raw Markdown copied to the clipboard. In the Markdown,
/// each category — `Overall` through `Documentation` — is a `##` heading
/// followed by its findings as a bullet list with the file names in bold,
/// then the `Conclusion` and the patch verdict plus any rejected or
/// not-reviewed files (the per-file statuses live in the `Conclusion`, not in a
/// header), and a closing `Generated by: **orangu <version>** (<model>)` line.
/// The rendered lines display the same report with the Markdown syntax
/// consumed: bold category headings without the `##` markers, and the
/// `**file**` names resolved to bold.
pub(crate) fn auto_review_exit_output(state: &AutoReviewState) -> (Vec<String>, String) {
    // The two variants stay in lockstep: `lines` goes to the output window,
    // `markdown` is what lands on the clipboard.
    let mut lines = Vec::new();
    let mut markdown = Vec::new();
    for (index, name) in AUTO_REVIEW_CATEGORIES.iter().enumerate() {
        lines.push(format!("\x1b[1m{name}\x1b[0m"));
        markdown.push(format!("## {name}"));
        lines.push(String::new());
        markdown.push(String::new());
        let section = &state.sections[index];
        if section.is_empty() {
            lines.push("No issues found".to_string());
            markdown.push("No issues found".to_string());
        } else {
            for finding in section {
                let bullet = auto_review_finding_bullet(finding);
                lines.extend(
                    render_markdown_for_console(&bullet)
                        .lines()
                        .map(str::to_string),
                );
                markdown.push(bullet);
            }
        }
        lines.push(String::new());
        markdown.push(String::new());
    }
    lines.push(format!("\x1b[1m{AUTO_REVIEW_CONCLUSION}\x1b[0m"));
    markdown.push(format!("## {AUTO_REVIEW_CONCLUSION}"));
    lines.push(String::new());
    markdown.push(String::new());
    // The verdict stands alone in bold; the affected files follow as a bullet
    // list.
    lines.push(state.conclusion_verdict_line());
    markdown.push(format!("**{}**", state.conclusion_verdict()));
    let findings = state.conclusion_findings();
    if !findings.is_empty() {
        lines.push(String::new());
        markdown.push(String::new());
        for line in findings {
            lines.push(render_markdown_for_console(&format!("- {line}")));
            markdown.push(format!("- {line}"));
        }
    }
    // The report closes with the orangu version and reviewing model.
    let generated_by = state.generated_by_markdown();
    lines.push(String::new());
    lines.push(render_markdown_for_console(&generated_by));
    markdown.push(String::new());
    markdown.push(generated_by);
    (lines, markdown.join("\n"))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn print_auto_review_screen(
    state: &AutoReviewState,
    viewport: &ViewportState,
    chrome: ReviewChrome<'_>,
    left_status: Option<StatusFragment>,
    blink_on: bool,
    input: &str,
    cursor: usize,
    ghost: &str,
) {
    let report_lines = state.report_lines();
    let status_text = state.status_text();
    let selected_path = state.selected_path();
    let reject = state
        .reject
        .as_ref()
        .zip(selected_path.as_deref())
        .map(|(reject, path)| AutoReviewRejectView {
            path,
            categories: &AUTO_REVIEW_CATEGORIES,
            category: reject.category,
            selector_focused: reject.selector_focused,
            text: reject.editor.as_str(),
            cursor: reject.editor.cursor(),
        });
    print!("{CLEAR_TERMINAL_SEQUENCE}");
    print!(
        "{}",
        render_auto_review_screen(AutoReviewScreenArgs {
            files: &state.files,
            selected: state.selected,
            // Pulsing the index on the render tick makes the dot blink.
            reviewing: state.reviewing.filter(|_| blink_on),
            browsing: state.done || state.cancelled,
            reject,
            report_lines: &report_lines,
            // Highlight the Up/Down item cursor only while browsing the report.
            selected_lines: if state.done || state.cancelled {
                state.selected_item_span()
            } else {
                None
            },
            scroll: state.scroll,
            x_offset: state.x_offset,
            status: &status_text,
            input,
            cursor,
            ghost,
            current_model: chrome.current_model,
            prompt_branch: chrome.prompt_branch,
            left_status,
            pending_count: chrome.pending_count,
            actual_width: viewport.actual_width,
            actual_height: viewport.actual_height,
        })
    );
}

/// Drive a whole `/auto_review` run: each file's per-category requests, the
/// whole-change pass, and the post-run report browsing, until the user leaves
/// the view. Returns the final state — completed, cancelled (Esc Esc), or
/// exited (Alt+x) — for the exit report.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_auto_review_mode(
    launch: ReviewLaunch,
    prompt_profile: &LlmConfiguration,
    usage_stats: &mut UsageStats,
    viewport: &mut ViewportState,
    chrome: ReviewChrome<'_>,
    workspace: &Path,
    terminal: &str,
    feedback: bool,
    skills: &orangu::skills::SkillRegistry,
) -> Result<AutoReviewState> {
    let mut state = AutoReviewState::new(launch);
    state.model = chrome.current_model.to_string();
    let mut exit_requested = false;
    let total = state.files.len();
    // The repository root the file paths are relative to, for the on-disk binary
    // sniff; `None` (and so a patch-marker-only skip) outside a repository.
    let repo_root = discover_git_root(workspace);
    // The categories each file is scanned for, decided once up front so the
    // progress total and the per-file loop below agree. A file is scanned only
    // for the categories its detected group enables: a documentation file skips
    // the code-related checks, and a skipped or binary file gets none.
    let file_categories: Vec<&'static [(usize, &'static str)]> = state
        .files
        .iter()
        .map(|file| auto_review_entry_categories(&file.path, &file.patch, repo_root.as_deref()))
        .collect();
    // The run's request count: every enabled per-file category, plus the final
    // whole-change pass.
    let total_requests: usize = file_categories.iter().map(|cats| cats.len()).sum::<usize>() + 1;
    let mut completed = 0usize;
    // Review each file by itself, one focused request per enabled category.
    // Every request runs in a scratch session so the reviews stay independent
    // and the main chat session is left untouched.
    'auto: for (index, categories) in file_categories.iter().enumerate() {
        state.selected = Some(index);
        let (path, patch) = {
            let file = &state.files[index];
            (file.path.clone(), file.patch.clone())
        };
        state.reviewing = Some(index);
        let mut any_rejected = false;
        let mut any_failed = false;
        for (section, focus) in *categories {
            let section = *section;
            let category = AUTO_REVIEW_CATEGORIES[section];
            state.status = format!(
                "File: {path} ({}/{total})  Category: {category}  {}",
                index + 1,
                auto_review_progress_label(completed, total_requests),
            );
            let prompt = build_auto_review_category_prompt(&path, category, focus, &patch);
            let mut scratch = ChatSession::new(system_prompt(prompt_profile));
            let llm_start = std::time::Instant::now();
            let outcome = run_auto_review_request(
                &mut scratch,
                &prompt,
                prompt_profile,
                &mut state,
                viewport,
                chrome,
                feedback,
            )
            .await?;
            match outcome {
                AutoReviewRequestOutcome::Completed(Ok(text)) => {
                    completed += 1;
                    state.update_estimate(completed, total_requests);
                    // No tools run during auto review requests.
                    usage_stats.record_response(
                        llm_start.elapsed(),
                        &text,
                        std::time::Duration::ZERO,
                    );
                    let (verdict, findings) = parse_auto_review_category_response(&text);
                    // A category passes when it carries an approving verdict, or
                    // when it carries neither a verdict nor findings — a bare
                    // "no verdict and no findings" response (e.g. truncated by
                    // the response cap) counts as clean, not a failure. So a
                    // file whose categories all come back empty is approved.
                    if !verdict.unwrap_or(findings.is_empty()) {
                        any_rejected = true;
                    }
                    state.apply_category_result(index, section, findings);
                }
                AutoReviewRequestOutcome::Completed(Err(err)) => {
                    completed += 1;
                    state.update_estimate(completed, total_requests);
                    any_failed = true;
                    state.record_failure(index, category, &err);
                }
                AutoReviewRequestOutcome::Cancelled => {
                    state.cancel();
                    break 'auto;
                }
                AutoReviewRequestOutcome::Exit => {
                    exit_requested = true;
                    break 'auto;
                }
            }
        }
        // Mark the file: red when any category rejected, white when a request
        // failed, green otherwise.
        let status = if any_rejected {
            ReviewStatus::Rejected
        } else if any_failed {
            ReviewStatus::Unreviewed
        } else {
            ReviewStatus::Approved
        };
        state.set_file_status(index, status);
    }
    // The per-file reviews are over; no file is highlighted and no dot blinks
    // during the whole-change pass.
    state.reviewing = None;
    state.selected = None;
    // Review the changes overall, from the per-file results.
    if !state.cancelled && !exit_requested {
        state.status = format!(
            "Category: Overall (whole change)  {}",
            auto_review_progress_label(completed, total_requests),
        );
        let prompt = build_auto_review_overall_prompt(&state);
        let mut scratch = ChatSession::new(system_prompt(prompt_profile));
        let llm_start = std::time::Instant::now();
        let outcome = run_auto_review_request(
            &mut scratch,
            &prompt,
            prompt_profile,
            &mut state,
            viewport,
            chrome,
            feedback,
        )
        .await?;
        match outcome {
            AutoReviewRequestOutcome::Completed(Ok(text)) => {
                // No tools run during auto review requests.
                usage_stats.record_response(llm_start.elapsed(), &text, std::time::Duration::ZERO);
                state.apply_overall(&text);
                state.finish();
            }
            AutoReviewRequestOutcome::Completed(Err(err)) => {
                state.record_overall_failure(&err);
                state.finish();
            }
            AutoReviewRequestOutcome::Cancelled => state.cancel(),
            AutoReviewRequestOutcome::Exit => exit_requested = true,
        }
    }
    // When feedback is on, announce a completed run with the standard terminal
    // bell and drop the blinking-dot title back to a plain `orangu`. Only a run
    // that actually finished rings — a cancel (Esc Esc) or exit (Alt+x) does
    // not.
    if feedback {
        if state.done {
            ring_terminal_bell();
        }
        set_terminal_title(Some(TERMINAL_TITLE));
        std::io::stdout().flush()?;
    }
    // Keep the report on screen for browsing until Alt+x/Esc Esc.
    if !exit_requested {
        run_auto_review_browse(&mut state, viewport, chrome, workspace, terminal, skills)?;
    }
    Ok(state)
}

/// Result of one auto review LLM request.
pub(crate) enum AutoReviewRequestOutcome {
    /// The model responded (`Ok`) or the request errored (`Err`).
    Completed(Result<String>),
    /// The user pressed Esc twice — stop the auto review run, keeping the
    /// collected report on screen.
    Cancelled,
    /// The user pressed Alt+x — leave auto review mode entirely.
    Exit,
}

/// Drive one auto review request, rendering the screen with a live status
/// (thinking, then the streaming rate once tokens arrive) until the response
/// completes. The requests run without tool definitions and with a capped
/// response length, so a review can neither start tool rounds nor generate
/// unbounded output. The report stays scrollable while the model works;
/// `Esc` `Esc` cancels the run and `Alt+x` exits the mode.
pub(crate) async fn run_auto_review_request(
    scratch: &mut ChatSession,
    prompt: &str,
    profile: &LlmConfiguration,
    state: &mut AutoReviewState,
    viewport: &mut ViewportState,
    chrome: ReviewChrome<'_>,
    feedback: bool,
) -> Result<AutoReviewRequestOutcome> {
    let streamed_state = Arc::new(Mutex::new(StreamRenderState::default()));
    let prompt_output = Arc::clone(&streamed_state);
    let prompt_metrics = Arc::clone(&streamed_state);
    let tokenizer = cl100k_base().ok();
    let mut future = Box::pin(scratch.prompt_without_tools(
        prompt,
        profile,
        // The configured `/auto_review` response cap (0 = no cap), so a
        // review can never generate unbounded output unless asked to.
        profile.review_max_tokens,
        move |delta| {
            if let Ok(mut state) = prompt_output.lock() {
                state.output.push_str(delta);
            }
        },
        move |metrics| {
            if let Ok(mut state) = prompt_metrics.lock() {
                state.metrics.merge(metrics);
            }
        },
    ));
    let mut interval = tokio::time::interval(WAIT_LOOP_POLL_INTERVAL);
    let started = std::time::Instant::now();
    let mut escape_cancel = EscapeCancelState::default();

    loop {
        tokio::select! {
            result = &mut future => return Ok(AutoReviewRequestOutcome::Completed(result)),
            _ = interval.tick() => {
                let body_height = auto_review_pane_body_height(
                    viewport.actual_height,
                    "",
                    chrome.prompt_branch,
                    viewport.actual_width,
                );
                while event::poll(std::time::Duration::ZERO)? {
                    let (code, modifiers) = match event::read()? {
                        Event::Resize(width, height) => {
                            viewport.on_resize(usize::from(width), usize::from(height));
                            continue;
                        }
                        Event::Key(KeyEvent { code, modifiers, kind, .. })
                            if kind == KeyEventKind::Press || kind == KeyEventKind::Repeat =>
                        {
                            (code, modifiers)
                        }
                        _ => continue,
                    };
                    let alt = modifiers.contains(KeyModifiers::ALT)
                        && !modifiers.contains(KeyModifiers::CONTROL);
                    if code == KeyCode::Esc {
                        if escape_cancel.handle_escape(std::time::Instant::now()) {
                            drop(future);
                            return Ok(AutoReviewRequestOutcome::Cancelled);
                        }
                        continue;
                    }
                    escape_cancel.reset();
                    match (code, alt) {
                        (KeyCode::Char('x'), true) => {
                            drop(future);
                            return Ok(AutoReviewRequestOutcome::Exit);
                        }
                        (KeyCode::Up, _) => state.scroll = state.scroll.saturating_sub(1),
                        (KeyCode::Down, _) => state.scroll = state.scroll.saturating_add(1),
                        (KeyCode::Left, _) => state.x_offset = state.x_offset.saturating_sub(1),
                        (KeyCode::Right, _) => state.x_offset = state.x_offset.saturating_add(1),
                        (KeyCode::PageUp, _) => {
                            state.scroll = state.scroll.saturating_sub(body_height);
                        }
                        (KeyCode::PageDown, _) => {
                            state.scroll = state.scroll.saturating_add(body_height);
                        }
                        _ => {}
                    }
                }
                let right_width =
                    orangu::tui::review_right_width(&state.files, viewport.actual_width);
                let left_width = viewport.actual_width.saturating_sub(right_width + 1).max(1);
                state.clamp(body_height, left_width);
                let frame = (started.elapsed().as_millis()
                    / THINKING_FRAME_INTERVAL.as_millis().max(1)) as usize;
                // Thinking until the first token, then the live streaming rate
                // (llama.cpp-native t/s when available).
                let current_state = streamed_state
                    .lock()
                    .map(|state| state.clone())
                    .unwrap_or_default();
                let status = render_left_status(
                    profile,
                    &current_state.output,
                    &current_state.metrics,
                    None,
                    started.elapsed(),
                    frame,
                    tokenizer.as_ref(),
                );
                // The reviewed file's white dot blinks at ~1Hz on the 120ms
                // frame clock: four frames on, four frames off.
                let blink_on = (frame / 4).is_multiple_of(2);
                // With feedback on, mirror the progress in the terminal title so
                // a backgrounded window still shows the run is alive: `orangu ●`
                // with the dot blinking once per second off the whole-run clock.
                if feedback {
                    let dot_on = state.elapsed().as_secs().is_multiple_of(2);
                    let title = if dot_on {
                        format!("{TERMINAL_TITLE} ●")
                    } else {
                        TERMINAL_TITLE.to_string()
                    };
                    set_terminal_title(Some(&title));
                }
                print_auto_review_screen(state, viewport, chrome, status, blink_on, "", 0, "");
                std::io::stdout().flush()?;
            }
        }
    }
}

/// Byte index of the start of the logical line containing `cursor`.
pub(crate) fn multiline_line_start(text: &str, cursor: usize) -> usize {
    text[..cursor.min(text.len())]
        .rfind('\n')
        .map(|index| index + 1)
        .unwrap_or(0)
}

/// Byte index of the end of the logical line containing `cursor` (just
/// before its `\n`, or the end of the text).
pub(crate) fn multiline_line_end(text: &str, cursor: usize) -> usize {
    let cursor = cursor.min(text.len());
    text[cursor..]
        .find('\n')
        .map(|index| cursor + index)
        .unwrap_or(text.len())
}

/// Move a byte cursor to the same column on the previous logical line (or
/// its end when that line is shorter); the first line keeps the cursor put.
pub(crate) fn multiline_cursor_up(text: &str, cursor: usize) -> usize {
    let cursor = cursor.min(text.len());
    let line_start = multiline_line_start(text, cursor);
    if line_start == 0 {
        return cursor;
    }
    let column = text[line_start..cursor].chars().count();
    let prev_start = multiline_line_start(text, line_start - 1);
    let prev_line = &text[prev_start..line_start - 1];
    prev_start
        + prev_line
            .char_indices()
            .nth(column)
            .map(|(index, _)| index)
            .unwrap_or(prev_line.len())
}

/// Move a byte cursor to the same column on the next logical line (or its
/// end when that line is shorter); the last line keeps the cursor put.
pub(crate) fn multiline_cursor_down(text: &str, cursor: usize) -> usize {
    let cursor = cursor.min(text.len());
    let line_start = multiline_line_start(text, cursor);
    let column = text[line_start..cursor].chars().count();
    let line_end = multiline_line_end(text, cursor);
    if line_end == text.len() {
        return cursor;
    }
    let next_start = line_end + 1;
    let next_line = &text[next_start..multiline_line_end(text, next_start)];
    next_start
        + next_line
            .char_indices()
            .nth(column)
            .map(|(index, _)| index)
            .unwrap_or(next_line.len())
}

/// Run the post-run auto review event loop — browsing the report — until the
/// user exits with Alt+x or Esc Esc. Alt+j/Alt+k move through the files;
/// Alt+a approves the highlighted file (dropping its findings from the
/// report), Alt+r opens the reject window, and Alt+e opens the file in the
/// configured editor.
pub(crate) fn run_auto_review_browse(
    state: &mut AutoReviewState,
    viewport: &mut ViewportState,
    chrome: ReviewChrome<'_>,
    workspace: &Path,
    terminal: &str,
    skills: &orangu::skills::SkillRegistry,
) -> Result<()> {
    let mut escape_cancel = EscapeCancelState::default();
    // The browse-phase input window: `/open_file <path>` or `open <path>` here
    // opens any project file in `$EDITOR`. Empty by default, so the report keeps
    // its full height and bare `-` still removes the highlighted item.
    let mut input_state = InputState::default();
    loop {
        let body_height = auto_review_pane_body_height(
            viewport.actual_height,
            input_state.as_str(),
            chrome.prompt_branch,
            viewport.actual_width,
        );
        let right_width = orangu::tui::review_right_width(&state.files, viewport.actual_width);
        let left_width = viewport.actual_width.saturating_sub(right_width + 1).max(1);
        state.clamp(body_height, left_width);
        state.ensure_item_visible(body_height);
        // Preview the file/command Tab would fill in, exactly like `/review`.
        let ghost = crate::completion::input_ghost_suffix(
            input_state.as_str(),
            input_state.cursor(),
            input_state.ghost_index,
            workspace,
            &[],
            &[],
            skills,
        )
        .unwrap_or_default();
        print_auto_review_screen(
            state,
            viewport,
            chrome,
            None,
            false,
            input_state.as_str(),
            input_state.cursor(),
            &ghost,
        );
        std::io::stdout().flush()?;

        let (code, modifiers) = match event::read()? {
            Event::Resize(width, height) => {
                viewport.on_resize(usize::from(width), usize::from(height));
                continue;
            }
            Event::Key(KeyEvent {
                code,
                modifiers,
                kind,
                ..
            }) if kind == KeyEventKind::Press || kind == KeyEventKind::Repeat => (code, modifiers),
            _ => continue,
        };
        let alt =
            modifiers.contains(KeyModifiers::ALT) && !modifiers.contains(KeyModifiers::CONTROL);
        let ctrl = modifiers.contains(KeyModifiers::CONTROL);

        // While the reject window is open it is modal: Tab moves the focus
        // between the category selector and the comment editor, Alt+Enter
        // saves the comment, Esc discards the window.
        if state.reject.is_some() {
            escape_cancel.reset();
            match (code, alt) {
                (KeyCode::Esc, _) => state.reject = None,
                (KeyCode::Enter, true) => state.commit_reject(),
                (KeyCode::Tab, _) | (KeyCode::BackTab, _) => {
                    let reject = state.reject.as_mut().unwrap();
                    reject.selector_focused = !reject.selector_focused;
                }
                _ => {
                    let reject = state.reject.as_mut().unwrap();
                    if reject.selector_focused {
                        match code {
                            KeyCode::Up => reject.category = reject.category.saturating_sub(1),
                            KeyCode::Down => {
                                reject.category =
                                    (reject.category + 1).min(AUTO_REVIEW_CATEGORIES.len() - 1);
                            }
                            // Enter moves on to the comment editor.
                            KeyCode::Enter => reject.selector_focused = false,
                            _ => {}
                        }
                    } else {
                        let editor = &mut reject.editor;
                        match (code, ctrl) {
                            (KeyCode::Enter, _) => editor.insert_char('\n'),
                            (KeyCode::Backspace, _) if alt => {
                                editor.delete_backward_readline_word();
                            }
                            (KeyCode::Backspace, _) => editor.backspace(),
                            (KeyCode::Delete, _) => editor.delete(),
                            (KeyCode::Left, true) => editor.move_backward_readline_word(),
                            (KeyCode::Right, true) => editor.move_forward_readline_word(),
                            (KeyCode::Left, _) => editor.move_left(),
                            (KeyCode::Right, _) => editor.move_right(),
                            (KeyCode::Up, _) => {
                                editor.cursor = multiline_cursor_up(&editor.buffer, editor.cursor);
                            }
                            (KeyCode::Down, _) => {
                                editor.cursor =
                                    multiline_cursor_down(&editor.buffer, editor.cursor);
                            }
                            (KeyCode::Home, _) => {
                                editor.cursor = multiline_line_start(&editor.buffer, editor.cursor);
                            }
                            (KeyCode::End, _) => {
                                editor.cursor = multiline_line_end(&editor.buffer, editor.cursor);
                            }
                            (KeyCode::Char(ch), false) if !alt => editor.insert_char(ch),
                            _ => {}
                        }
                    }
                }
            }
            continue;
        }

        // A second Esc within the timeout leaves auto review; the first arms it.
        if code == KeyCode::Esc {
            if escape_cancel.handle_escape(std::time::Instant::now()) {
                return Ok(());
            }
            continue;
        }
        escape_cancel.reset();

        match (code, alt, ctrl) {
            (KeyCode::Char('x'), true, _) => return Ok(()),
            (KeyCode::Char('j'), true, _) => state.select_next(),
            (KeyCode::Char('k'), true, _) => state.select_prev(),
            (KeyCode::Char('a'), true, _) => state.approve_selected(),
            (KeyCode::Char('r'), true, _) => state.open_reject(),
            (KeyCode::Char('e'), true, _) => {
                if let Some(path) = state.selected_path()
                    && let Err(err) = open_in_editor(workspace, &path, terminal)
                {
                    // No feedback popup in auto review; surface the error in
                    // the status area.
                    state.status = format!("Open {path} failed: {err:#}");
                }
            }
            // Submitting `/open_file <path>` or `open <path>` opens any project
            // file in `$EDITOR`; anything else is ignored (auto review has no
            // chat to send it to).
            (KeyCode::Enter, _, _) => {
                if let Some(path) = crate::commands::parse_open_command_target(input_state.as_str())
                {
                    let path = path.to_string();
                    input_state.clear();
                    if let Err(err) = open_in_editor(workspace, &path, terminal) {
                        state.status = format!("Open {path} failed: {err:#}");
                    }
                }
            }
            // Tab completes the input window over every project file, like the
            // main prompt; Shift+Tab cycles the ghost preview.
            (KeyCode::Tab, _, _) => {
                crate::input::apply_completion(&mut input_state, workspace, &[], &[], skills);
            }
            (KeyCode::BackTab, _, _) => crate::input::cycle_ghost_suggestion(&mut input_state),
            // Once the run is done Up/Down move between report items (not
            // headings); `-` removes the highlighted one only while the input
            // window is empty, so a `-` in a typed path is still editable.
            (KeyCode::Up, false, _) => state.select_prev_item(),
            (KeyCode::Down, false, _) => state.select_next_item(),
            (KeyCode::Char('-'), false, false) if input_state.as_str().is_empty() => {
                state.remove_selected_item();
            }
            // Horizontal report pan moves to Alt+Left/Right, leaving bare
            // Left/Right to the input cursor (like `/review`).
            (KeyCode::Left, true, _) => state.x_offset = state.x_offset.saturating_sub(1),
            (KeyCode::Right, true, _) => state.x_offset = state.x_offset.saturating_add(1),
            // PageUp/PageDown jump the item highlight to the previous/next
            // category that has findings, so a long report is walked category
            // by category.
            (KeyCode::PageUp, _, _) => state.select_prev_category(),
            (KeyCode::PageDown, _, _) => state.select_next_category(),
            // Input window editing, mirroring `/review`.
            (KeyCode::Backspace, true, _) => input_state.delete_backward_readline_word(),
            (KeyCode::Backspace, _, _) => input_state.backspace(),
            (KeyCode::Delete, _, _) => input_state.delete(),
            (KeyCode::Left, _, true) => input_state.move_backward_readline_word(),
            (KeyCode::Right, _, true) => input_state.move_forward_readline_word(),
            (KeyCode::Left, _, _) => input_state.move_left(),
            (KeyCode::Right, _, _) => input_state.move_right(),
            (KeyCode::Home, _, _) | (KeyCode::Char('a'), _, true) => input_state.move_home(),
            (KeyCode::End, _, _) | (KeyCode::Char('e'), _, true) => input_state.move_end(),
            (KeyCode::Char('k'), _, true) => input_state.kill_to_end(),
            (KeyCode::Char('u'), _, true) => input_state.kill_to_start(),
            (KeyCode::Char('w'), _, true) => input_state.delete_prev_word(),
            (KeyCode::Char(ch), false, false) => input_state.insert_char(ch),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn parse_auto_review_category_response_reads_verdict_and_findings() {
        use crate::review::parse_auto_review_category_response;

        let text = "VERDICT: REJECT\n\
                    FINDINGS:\n\
                    - unwrap may panic\n\
                    - missing error context\n";
        let (approved, findings) = parse_auto_review_category_response(text);
        assert_eq!(approved, Some(false));
        assert_eq!(
            findings,
            vec![
                "unwrap may panic".to_string(),
                "missing error context".to_string()
            ]
        );

        // A "None" placeholder yields an approved, empty review — including
        // when the model still prefixed it with a line reference, so a clean
        // category records nothing rather than a `95-96: None` finding.
        for clean in [
            "VERDICT: APPROVE\nFINDINGS:\n- None\n",
            "VERDICT: APPROVE\nFINDINGS:\n- 95-96: None\n",
            "VERDICT: APPROVE\nFINDINGS:\n- 42: none.\n",
            // A "None" with a trailing parenthetical justification is still a
            // placeholder, not a finding.
            "VERDICT: APPROVE\nFINDINGS:\n- 152-155: None (no direct memory risk)\n",
            "VERDICT: APPROVE\nFINDINGS:\n- No issues (documentation only)\n",
            // The model may also fill the line slot with `None`, leaving a bare
            // `None: None` (optionally with a justification).
            "VERDICT: APPROVE\nFINDINGS:\n- None: None\n",
            "VERDICT: APPROVE\nFINDINGS:\n- None: None (nothing to flag)\n",
        ] {
            let (approved, findings) = parse_auto_review_category_response(clean);
            assert_eq!(approved, Some(true));
            assert!(findings.is_empty(), "{clean:?} -> {findings:?}");
        }

        // A genuine finding that merely ends in a parenthetical is kept — only
        // a "None"-style placeholder is stripped.
        let (_, findings) = parse_auto_review_category_response(
            "FINDINGS:\n- 42: unwrap may panic (added on the hot path)\n",
        );
        assert_eq!(
            findings,
            vec!["42: unwrap may panic (added on the hot path)".to_string()]
        );
    }

    #[test]
    fn parse_auto_review_category_response_tolerates_markdown_decoration() {
        use crate::review::parse_auto_review_category_response;

        let text = "**VERDICT:** APPROVE\n\
                    ## Findings:\n\
                    * looks fine\n\
                    1. add a regression test\n";
        let (approved, findings) = parse_auto_review_category_response(text);
        assert_eq!(approved, Some(true));
        assert_eq!(
            findings,
            vec![
                "looks fine".to_string(),
                "add a regression test".to_string()
            ]
        );

        // An inline finding after the header is kept; a finding starting with
        // a header-like word is not mistaken for a header.
        let (approved, findings) = parse_auto_review_category_response(
            "FINDINGS: cache grows without bound\nverdict handling is wrong\n",
        );
        assert_eq!(approved, None);
        assert_eq!(
            findings,
            vec![
                "cache grows without bound".to_string(),
                "verdict handling is wrong".to_string()
            ]
        );
    }

    #[test]
    fn apply_overall_drops_verdict_lines_and_none_affirmations() {
        use crate::commands::ReviewLaunch;
        use crate::review::AutoReviewState;
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let mut state = AutoReviewState::new(ReviewLaunch {
            files: vec![ReviewEntry {
                path: "a.rs".to_string(),
                status: ReviewStatus::Approved,
                diff_lines: vec!["+x".to_string()],
                patch: String::new(),
            }],
        });

        // The whole-change pass sometimes emits a stray verdict and "nothing to
        // report" affirmations alongside the real bullets. The verdict is
        // recorded elsewhere, and a leading `None.`/`Nothing.` sentence is noise,
        // so only the genuine observations reach the `Overall` section.
        state.apply_overall(
            "- None.\n\
             - VERDICT: REJECT\n\
             - None. Everything is consistent.\n\
             - The change set is cohesive and ready to merge.\n\
             - None of the modules grow unbounded.\n",
        );
        assert_eq!(
            state.sections[0],
            vec![
                "The change set is cohesive and ready to merge.".to_string(),
                "None of the modules grow unbounded.".to_string(),
            ]
        );
    }

    #[test]
    fn auto_review_is_placeholder_catches_leading_none_sentence() {
        use crate::review::auto_review_is_placeholder;

        // Bare placeholders, and a placeholder that opens a sentence.
        for text in [
            "None",
            "None.",
            "Nothing!",
            "N/A",
            "None. Everything looks good.",
            "None; the patch is clean.",
        ] {
            assert!(auto_review_is_placeholder(text), "{text:?}");
        }
        // Real content that merely begins with the word `None` is kept.
        for text in [
            "None of the changes are risky",
            "Nonexistent field referenced",
            "unwrap may panic",
        ] {
            assert!(!auto_review_is_placeholder(text), "{text:?}");
        }
    }

    #[test]
    fn auto_review_category_results_prefix_findings_with_the_path() {
        use crate::commands::ReviewLaunch;
        use crate::review::AutoReviewState;
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let entry = |path: &str| ReviewEntry {
            path: path.to_string(),
            status: ReviewStatus::Unreviewed,
            diff_lines: vec!["+x".to_string()],
            patch: String::new(),
        };
        let mut state = AutoReviewState::new(ReviewLaunch {
            files: vec![entry("a.rs"), entry("b.rs")],
        });

        // Findings land in the requested category, prefixed with the file's
        // location in Markdown bold. The model's leading line reference — a
        // single line or a `<start>-<end>` range — is folded into the location
        // as `path:line`; a finding without one keeps just the path.
        state.apply_category_result(1, 1, vec!["42: broken loop".to_string()]);
        state.apply_category_result(1, 2, vec!["10-14: unchecked range".to_string()]);
        state.apply_category_result(1, 6, vec!["update the manual".to_string()]);
        assert_eq!(
            state.sections[1],
            vec!["**b.rs:42**: broken loop".to_string()]
        );
        assert_eq!(
            state.sections[2],
            vec!["**b.rs:10-14**: unchecked range".to_string()]
        );
        assert_eq!(
            state.sections[6],
            vec!["**b.rs**: update the manual".to_string()]
        );

        // The dot is set once per file, after all categories have run.
        state.set_file_status(0, ReviewStatus::Approved);
        state.set_file_status(1, ReviewStatus::Rejected);
        assert_eq!(state.files[0].status, ReviewStatus::Approved);
        assert_eq!(state.files[1].status, ReviewStatus::Rejected);
    }

    #[test]
    fn auto_review_approve_removes_the_files_findings() {
        use crate::commands::ReviewLaunch;
        use crate::review::AutoReviewState;
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let entry = |path: &str| ReviewEntry {
            path: path.to_string(),
            status: ReviewStatus::Rejected,
            diff_lines: vec!["+x".to_string()],
            patch: String::new(),
        };
        let mut state = AutoReviewState::new(ReviewLaunch {
            files: vec![entry("a.rs"), entry("b.rs")],
        });
        // A mix of line-bearing (`**path:line**:`) and line-less (`**path**:`,
        // e.g. an Alt+r comment) findings; approving the file must strip both.
        state.sections[1].push("**a.rs:42**: broken loop".to_string());
        state.sections[2].push("**a.rs**: unsafe input".to_string());
        state.sections[1].push("**b.rs:7**: another issue".to_string());
        state.finish();

        // Alt+a with no highlight does nothing.
        state.approve_selected();
        assert_eq!(state.sections[1].len(), 2);

        // Approving a.rs strips its findings from every category — the other
        // file's findings stay — and turns its dot green.
        state.selected = Some(0);
        state.approve_selected();
        assert_eq!(state.files[0].status, ReviewStatus::Approved);
        assert_eq!(
            state.sections[1],
            vec!["**b.rs:7**: another issue".to_string()]
        );
        assert!(state.sections[2].is_empty());
    }

    #[test]
    fn auto_review_item_navigation_walks_findings_then_conclusion() {
        use crate::commands::ReviewLaunch;
        use crate::review::{AutoReviewItemKind, AutoReviewState};
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let entry = |path: &str| ReviewEntry {
            path: path.to_string(),
            status: ReviewStatus::Rejected,
            diff_lines: vec!["+x".to_string()],
            patch: String::new(),
        };
        let mut state = AutoReviewState::new(ReviewLaunch {
            files: vec![entry("a.rs"), entry("b.rs")],
        });
        state.sections[1].push("**a.rs:42**: broken loop".to_string());
        state.sections[1].push("**b.rs:7**: another issue".to_string());
        state.sections[2].push("**a.rs**: unsafe input".to_string());
        state.finish();

        // The items are the findings in display order, then one Conclusion
        // entry per unapproved file — never the category headings.
        let kinds: Vec<_> = state
            .report_items()
            .into_iter()
            .map(|item| item.kind)
            .collect();
        assert!(matches!(
            kinds[0],
            AutoReviewItemKind::Finding {
                section: 1,
                index: 0
            }
        ));
        assert!(matches!(
            kinds[2],
            AutoReviewItemKind::Finding {
                section: 2,
                index: 0
            }
        ));
        assert!(matches!(&kinds[3], AutoReviewItemKind::Conclusion { path } if path == "a.rs"));
        assert_eq!(kinds.len(), 5);

        // Down from no highlight starts at the first item and moves the file
        // highlight to that item's file; Up from no highlight starts at the
        // last. Both clamp at the ends.
        state.select_next_item();
        assert_eq!(state.selected_item, Some(0));
        assert_eq!(state.selected, Some(0));
        state.select_next_item(); // b.rs:7 finding
        assert_eq!(state.selected, Some(1));
        state.selected_item = None;
        state.select_prev_item();
        assert_eq!(state.selected_item, Some(4));
    }

    #[test]
    fn auto_review_page_keys_jump_between_categories_with_findings() {
        use crate::commands::ReviewLaunch;
        use crate::review::AutoReviewState;
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let entry = |path: &str| ReviewEntry {
            path: path.to_string(),
            status: ReviewStatus::Rejected,
            diff_lines: vec!["+x".to_string()],
            patch: String::new(),
        };
        let mut state = AutoReviewState::new(ReviewLaunch {
            files: vec![entry("a.rs"), entry("b.rs")],
        });
        // Code (two findings), Security (one), and the Conclusion (two rejected
        // files) carry items; the other categories stay empty and are skipped.
        state.sections[1].push("**a.rs:42**: broken loop".to_string());
        state.sections[1].push("**b.rs:7**: another issue".to_string());
        state.sections[2].push("**a.rs**: unsafe input".to_string());
        state.finish();
        // Items: 0,1 = Code findings, 2 = Security, 3,4 = Conclusion entries.

        // PageDown from no highlight lands on the first category's first finding,
        // then jumps category by category — skipping the second Code finding —
        // and lands on the Conclusion last.
        state.select_next_category();
        assert_eq!(state.selected_item, Some(0));
        state.select_next_category();
        assert_eq!(state.selected_item, Some(2)); // Security, not the 2nd Code item
        state.select_next_category();
        assert_eq!(state.selected_item, Some(3)); // Conclusion
        // Already in the last category: PageDown is a no-op (never wraps back).
        state.select_next_category();
        assert_eq!(state.selected_item, Some(3));

        // PageUp walks back to each category's first item.
        state.select_prev_category();
        assert_eq!(state.selected_item, Some(2)); // Security
        state.select_prev_category();
        assert_eq!(state.selected_item, Some(0)); // Code's first finding
        // Already in the first category: PageUp is a no-op.
        state.select_prev_category();
        assert_eq!(state.selected_item, Some(0));

        // From a finding in the middle of a category, PageDown still leaves to
        // the next category; PageUp from no highlight starts at the last.
        state.selected_item = Some(1); // second Code finding
        state.select_next_category();
        assert_eq!(state.selected_item, Some(2));
        state.selected_item = None;
        state.select_prev_category();
        assert_eq!(state.selected_item, Some(3)); // last category (Conclusion)
    }

    #[test]
    fn auto_review_removing_items_can_approve_the_whole_patch() {
        use crate::commands::ReviewLaunch;
        use crate::review::AutoReviewState;
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let entry = |path: &str| ReviewEntry {
            path: path.to_string(),
            status: ReviewStatus::Rejected,
            diff_lines: vec!["+x".to_string()],
            patch: String::new(),
        };
        let mut state = AutoReviewState::new(ReviewLaunch {
            files: vec![entry("a.rs"), entry("b.rs")],
        });
        state.sections[1].push("**a.rs:42**: broken loop".to_string());
        state.sections[1].push("**b.rs:7**: another issue".to_string());
        state.sections[2].push("**a.rs**: unsafe input".to_string());
        state.finish();

        // Removing a.rs's first finding leaves it with another (in Security),
        // so a.rs stays rejected and still appears in the Conclusion.
        state.selected_item = Some(0);
        state.remove_selected_item();
        assert_eq!(state.sections[1], vec!["**b.rs:7**: another issue"]);
        assert_eq!(state.files[0].status, ReviewStatus::Rejected);
        assert!(
            state
                .conclusion_findings()
                .iter()
                .any(|line| line.contains("a.rs"))
        );

        // Removing a.rs's last finding approves it and drops it from the
        // Conclusion — without touching b.rs.
        let security = state
            .report_items()
            .iter()
            .position(|item| {
                matches!(
                    item.kind,
                    crate::review::AutoReviewItemKind::Finding { section: 2, .. }
                )
            })
            .expect("a.rs security finding");
        state.selected_item = Some(security);
        state.remove_selected_item();
        assert_eq!(state.files[0].status, ReviewStatus::Approved);
        assert!(state.sections[2].is_empty());
        assert!(
            !state
                .conclusion_findings()
                .iter()
                .any(|line| line.contains("a.rs"))
        );
        assert_eq!(state.conclusion_verdict(), "orangu rejects this patch");

        // Removing b.rs's Conclusion item approves the whole file (clearing its
        // finding too); with every file approved the patch verdict flips and no
        // items remain, so the highlight clears.
        let conclusion = state
            .report_items()
            .iter()
            .position(|item| {
                matches!(
                    item.kind,
                    crate::review::AutoReviewItemKind::Conclusion { .. }
                )
            })
            .expect("b.rs conclusion item");
        state.selected_item = Some(conclusion);
        state.remove_selected_item();
        assert_eq!(state.files[1].status, ReviewStatus::Approved);
        assert!(state.sections.iter().all(|section| section.is_empty()));
        assert_eq!(state.conclusion_verdict(), "orangu approves this patch");
        assert!(state.report_items().is_empty());
        assert_eq!(state.selected_item, None);
    }

    #[test]
    fn auto_review_reject_records_the_comment_in_the_chosen_category() {
        use crate::commands::ReviewLaunch;
        use crate::review::AutoReviewState;
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let mut state = AutoReviewState::new(ReviewLaunch {
            files: vec![ReviewEntry {
                path: "a.rs".to_string(),
                status: ReviewStatus::Approved,
                diff_lines: vec!["+x".to_string()],
                patch: String::new(),
            }],
        });
        state.finish();

        // Alt+r needs a highlighted file.
        state.open_reject();
        assert!(state.reject.is_none());

        // The comment lands in the chosen category, prefixed with the path,
        // and the file's dot turns red.
        state.selected = Some(0);
        state.open_reject();
        let reject = state.reject.as_mut().expect("reject window open");
        reject.category = 2;
        reject.editor.set_buffer("uses *raw* input".to_string());
        state.commit_reject();
        assert!(state.reject.is_none());
        assert_eq!(state.files[0].status, ReviewStatus::Rejected);
        assert_eq!(
            state.sections[2],
            vec!["**a.rs**: uses *raw* input".to_string()]
        );

        // Rejecting can be repeated; each comment is kept.
        state.open_reject();
        let reject = state.reject.as_mut().expect("reject window open");
        reject.category = 2;
        reject.editor.set_buffer("also unbounded".to_string());
        state.commit_reject();
        assert_eq!(state.sections[2].len(), 2);

        // An empty comment still rejects the file but adds no finding.
        state.approve_selected();
        state.open_reject();
        state.commit_reject();
        assert_eq!(state.files[0].status, ReviewStatus::Rejected);
        assert!(state.sections.iter().all(|section| section.is_empty()));
    }

    #[test]
    fn auto_review_multiline_comments_stay_inside_their_bullet() {
        use crate::commands::ReviewLaunch;
        use crate::review::{AutoReviewState, auto_review_exit_output, auto_review_finding_bullet};
        use orangu::tui::{ReviewEntry, ReviewStatus};

        assert_eq!(auto_review_finding_bullet("one line"), "- one line");
        assert_eq!(
            auto_review_finding_bullet("first\nsecond"),
            "- first\n  second"
        );

        let mut state = AutoReviewState::new(ReviewLaunch {
            files: vec![ReviewEntry {
                path: "a.rs".to_string(),
                status: ReviewStatus::Approved,
                diff_lines: vec!["+x".to_string()],
                patch: String::new(),
            }],
        });
        state.finish();
        state.selected = Some(0);
        state.open_reject();
        let reject = state.reject.as_mut().expect("reject window open");
        reject.category = 1;
        reject
            .editor
            .set_buffer("broken loop\nsee the spec".to_string());
        state.commit_reject();

        // The clipboard Markdown keeps the comment as one bullet with an
        // indented continuation line; the rendered lines split per row.
        let (lines, markdown) = auto_review_exit_output(&state);
        assert!(
            markdown.contains("- **a.rs**: broken loop\n  see the spec"),
            "{markdown:?}"
        );
        assert!(lines.iter().any(|line| line.contains("see the spec")));
        assert!(lines.iter().all(|line| !line.contains('\n')));
    }

    #[test]
    fn multiline_cursor_moves_between_logical_lines() {
        use crate::review::{
            multiline_cursor_down, multiline_cursor_up, multiline_line_end, multiline_line_start,
        };

        let text = "alpha\nbé\ngamma";
        // Down from column 4 of "alpha" clamps to the end of the shorter "bé".
        assert_eq!(multiline_cursor_down(text, 4), 6 + "bé".len());
        // Up from "gamma" lands at the same column of "bé"; columns are
        // characters, not bytes.
        let gamma_start = text.find("gamma").unwrap();
        assert_eq!(
            multiline_cursor_up(text, gamma_start + 1),
            6 + 'b'.len_utf8()
        );
        // The first and last lines keep the cursor put.
        assert_eq!(multiline_cursor_up(text, 2), 2);
        assert_eq!(
            multiline_cursor_down(text, gamma_start + 1),
            gamma_start + 1
        );
        // Line home/end stay within the logical line.
        assert_eq!(multiline_line_start(text, 7), 6);
        assert_eq!(multiline_line_end(text, 7), 6 + "bé".len());
    }

    #[test]
    fn auto_review_category_prompts_share_their_diff_prefix() {
        use crate::review::{AUTO_REVIEW_FILE_CATEGORIES, build_auto_review_category_prompt};

        // The diff leads the prompt and the category instruction follows, so a
        // file's category requests share their prefix and the server's prompt
        // cache can reuse the processed diff across them.
        let patch = "--- a/src/main.rs\n+++ b/src/main.rs\n@@ -1 +1 @@\n-x\n+y\n";
        let (_, code_focus) = AUTO_REVIEW_FILE_CATEGORIES[0];
        let (_, security_focus) = AUTO_REVIEW_FILE_CATEGORIES[1];
        let code = build_auto_review_category_prompt("src/main.rs", "Code", code_focus, patch);
        let security =
            build_auto_review_category_prompt("src/main.rs", "Security", security_focus, patch);

        let diff_end = code.find("```\n\n").expect("diff block") + "```".len();
        assert!(code[..diff_end].contains(patch));
        assert_eq!(code[..diff_end], security[..diff_end]);
        // The category-specific instruction only appears after the diff.
        assert!(code[diff_end..].contains("Code issues"));
        assert!(security[diff_end..].contains("Security issues"));
    }

    #[test]
    fn auto_review_status_text_appends_the_run_time() {
        use crate::commands::ReviewLaunch;
        use crate::review::AutoReviewState;

        // `Time:` follows the progress information and freezes when the run
        // ends. (The duration format itself is covered by the tui tests.)
        let mut state = AutoReviewState::new(ReviewLaunch { files: Vec::new() });
        state.status = "Category: Code  Progress: 1/13 (7%)".to_string();
        let text = state.status_text();
        let progress = text.find("Progress:").expect("progress in status");
        let time = text.find("  Time: ").expect("time in status");
        assert!(time > progress, "expected Time after Progress in {text:?}");

        // No estimate until the first request completes; once one has, the
        // estimate follows `Time:`.
        assert!(!text.contains("Estimated:"), "{text:?}");
        state.update_estimate(1, 13);
        let text = state.status_text();
        let estimated = text.find("  Estimated: ").expect("estimate in status");
        assert!(
            estimated > time,
            "expected Estimated after Time in {text:?}"
        );

        // The estimate drops away once the run ends; `Time:` freezes.
        state.finish();
        let frozen = state.status_text();
        assert!(frozen.starts_with("Done  Time: "));
        assert!(!frozen.contains("Estimated:"), "{frozen:?}");
        assert_eq!(frozen, state.status_text());
    }

    #[test]
    fn auto_review_highlight_clears_when_the_run_ends() {
        use crate::commands::ReviewLaunch;
        use crate::review::AutoReviewState;
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let entry = |path: &str| ReviewEntry {
            path: path.to_string(),
            status: ReviewStatus::Approved,
            diff_lines: vec!["+x".to_string()],
            patch: String::new(),
        };
        let mut state = AutoReviewState::new(ReviewLaunch {
            files: vec![entry("a.rs"), entry("b.rs")],
        });

        // The run highlights the file under review; finishing clears it.
        state.selected = Some(1);
        state.finish();
        assert_eq!(state.selected, None);

        // Browsing brings the highlight back: Alt+j starts at the first file,
        // Alt+k (from none) at the last, and both clamp at the ends.
        state.select_next();
        assert_eq!(state.selected, Some(0));
        state.select_next();
        state.select_next();
        assert_eq!(state.selected, Some(1));

        state.selected = None;
        state.select_prev();
        assert_eq!(state.selected, Some(1));
        state.select_prev();
        state.select_prev();
        assert_eq!(state.selected, Some(0));

        // Cancelling clears the highlight too.
        let mut state = AutoReviewState::new(ReviewLaunch {
            files: vec![entry("a.rs")],
        });
        state.selected = Some(0);
        state.cancel();
        assert_eq!(state.selected, None);
    }

    #[test]
    fn auto_review_empty_response_parses_to_a_clean_pass() {
        use crate::review::parse_auto_review_category_response;

        // An empty (e.g. cap-truncated) response parses to no verdict and no
        // findings. The run treats this as a clean category — not a failure —
        // so a file whose categories all come back empty is approved (the
        // pass test `verdict.unwrap_or(findings.is_empty())` is true and no
        // findings are recorded).
        let (verdict, findings) = parse_auto_review_category_response("");
        assert_eq!(verdict, None);
        assert!(findings.is_empty());
        assert!(verdict.unwrap_or(findings.is_empty()));
    }

    #[test]
    fn auto_review_progress_label_counts_all_requests() {
        use crate::review::auto_review_progress_label;

        // E.g. two code files (6 requests each) plus the overall pass = 13.
        assert_eq!(auto_review_progress_label(0, 13), "Progress: 0/13 (0%)");
        assert_eq!(auto_review_progress_label(6, 13), "Progress: 6/13 (46%)");
        assert_eq!(auto_review_progress_label(12, 13), "Progress: 12/13 (92%)");
    }

    #[test]
    fn auto_review_file_categories_follow_the_file_extension() {
        use crate::review::auto_review_entry_categories;
        use crate::review::{AUTO_REVIEW_CATEGORIES, AUTO_REVIEW_FILE_CATEGORIES};

        // The path-only decision: no patch and no repo root, so the
        // content-based binary sniff is a no-op.
        let categories = |path: &str| auto_review_entry_categories(path, "", None);

        // Code files are scanned for every per-file category.
        assert_eq!(categories("src/main.rs"), &AUTO_REVIEW_FILE_CATEGORIES[..]);
        // Files without an extension too.
        assert_eq!(categories("Makefile"), &AUTO_REVIEW_FILE_CATEGORIES[..]);
        // Code files whose guessed MIME would misclassify them keep the full
        // review: TypeScript reads as an MPEG transport stream (`video/*`), and
        // a `.java` source as `application/octet-stream`. Falling back to a code
        // review is the safe choice when the MIME is not certain.
        for path in ["app/main.ts", "app/main.mts", "src/App.java"] {
            assert_eq!(
                categories(path),
                &AUTO_REVIEW_FILE_CATEGORIES[..],
                "expected the full review for {path:?}"
            );
        }
        // Known documentation extensions go straight to Documentation,
        // case-insensitively — including the extensions added to the list, and
        // the MIME-detected types (`.tex`) that no extension entry covers.
        for path in [
            "README.md",
            "doc/manual.RST",
            "notes.txt",
            "guide.mdx",
            "paper.tex",
        ] {
            let enabled = categories(path);
            assert_eq!(enabled.len(), 1, "expected only Documentation for {path:?}");
            assert_eq!(AUTO_REVIEW_CATEGORIES[enabled[0].0], "Documentation");
        }

        // A skip-list file (lock file or binary asset) is approved at once: no
        // categories, so no requests. Audio and video assets, detected by MIME
        // type, skip too.
        for path in [
            "Cargo.lock",
            "package-lock.json",
            "go.sum",
            "assets/logo.png",
            "fonts/Inter.woff2",
            "media/theme.mp3",
            "media/intro.mp4",
            "dist/app.wasm",
        ] {
            assert!(
                categories(path).is_empty(),
                "expected no categories for {path:?}"
            );
        }

        // A forced-full metadata file takes the full review even though its
        // `.txt` extension would otherwise read as documentation.
        for path in ["CMakeLists.txt", "build/CMakeLists.txt", "requirements.txt"] {
            assert_eq!(
                categories(path),
                &AUTO_REVIEW_FILE_CATEGORIES[..],
                "expected the full review for {path:?}"
            );
        }

        // The detection that decides whether a file's code-related checks
        // can be skipped.
        use crate::review::{auto_review_documentation_file, auto_review_skipped_file};
        assert!(auto_review_documentation_file("README.md"));
        assert!(!auto_review_documentation_file("src/main.rs"));
        assert!(!auto_review_documentation_file("Makefile"));
        // CMakeLists.txt is not documentation despite its extension: the
        // metadata override wins.
        assert!(auto_review_documentation_file("CMakeLists.txt"));
        assert!(!auto_review_skipped_file("CMakeLists.txt"));
        assert!(auto_review_skipped_file("Cargo.lock"));
        assert!(auto_review_skipped_file("assets/logo.PNG"));
        // Audio, video, and font assets are skipped by MIME type, not a curated
        // extension list.
        assert!(auto_review_skipped_file("sound/clip.WAV"));
        assert!(auto_review_skipped_file("clips/demo.mov"));
        assert!(!auto_review_skipped_file("src/main.rs"));
    }

    #[test]
    fn auto_review_binary_changes_skip_by_patch_and_content() {
        use crate::review::{
            AUTO_REVIEW_FILE_CATEGORIES, auto_review_binary_content, auto_review_binary_patch,
            auto_review_entry_categories,
        };

        // git emits this marker instead of a textual hunk for a binary file with
        // an extension we do not otherwise recognize.
        let binary_patch = "diff --git a/blob.dat b/blob.dat\n\
             index 0000000..1111111 100644\n\
             Binary files a/blob.dat and b/blob.dat differ\n";
        assert!(auto_review_binary_patch(binary_patch));
        assert!(
            auto_review_entry_categories("blob.dat", binary_patch, None).is_empty(),
            "a binary patch downgrades a code file to a skip"
        );

        // A textual diff is not a binary change, so it keeps the full review.
        let text_patch = "diff --git a/src/main.rs b/src/main.rs\n\
             @@ -1 +1 @@\n\
             -old\n+new\n";
        assert!(!auto_review_binary_patch(text_patch));
        assert_eq!(
            auto_review_entry_categories("src/main.rs", text_patch, None),
            &AUTO_REVIEW_FILE_CATEGORIES[..]
        );

        // Magic-number detection: a PNG header is binary, plain UTF-8 is not.
        let png_header = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        assert!(auto_review_binary_content(&png_header));
        assert!(!auto_review_binary_content(b"fn main() {}\n"));
    }

    #[test]
    fn auto_review_exit_output_lists_categories_and_conclusion() {
        use crate::commands::ReviewLaunch;
        use crate::review::{AutoReviewState, auto_review_exit_output};
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let mut state = AutoReviewState::new(ReviewLaunch {
            files: vec![ReviewEntry {
                path: "a.rs".to_string(),
                status: ReviewStatus::Approved,
                diff_lines: vec!["+x".to_string()],
                patch: String::new(),
            }],
        });
        state.sections[0].push("ready to merge".to_string());
        state.sections[1].push("**a.rs**: tighten error handling".to_string());
        state.model = "gemma".to_string();
        state.finish();

        // The report is just the categories — no header and no per-file
        // status lines — ending with the Conclusion verdict (standing alone in
        // bold) and a closing `Generated by: orangu <version> (<model>)` line.
        // The output-window lines display the Markdown rendered: bold headings
        // without the `##` markers, `**file**` resolved to bold.
        let (lines, clipboard) = auto_review_exit_output(&state);
        assert_eq!(
            lines,
            vec![
                "\x1b[1mOverall\x1b[0m".to_string(),
                String::new(),
                "- ready to merge".to_string(),
                String::new(),
                "\x1b[1mCode\x1b[0m".to_string(),
                String::new(),
                "- \x1b[1ma.rs\x1b[22m: tighten error handling".to_string(),
                String::new(),
                "\x1b[1mSecurity\x1b[0m".to_string(),
                String::new(),
                "No issues found".to_string(),
                String::new(),
                "\x1b[1mMemory\x1b[0m".to_string(),
                String::new(),
                "No issues found".to_string(),
                String::new(),
                "\x1b[1mPerformance\x1b[0m".to_string(),
                String::new(),
                "No issues found".to_string(),
                String::new(),
                "\x1b[1mTest Suite\x1b[0m".to_string(),
                String::new(),
                "No issues found".to_string(),
                String::new(),
                "\x1b[1mDocumentation\x1b[0m".to_string(),
                String::new(),
                "No issues found".to_string(),
                String::new(),
                "\x1b[1mConclusion\x1b[0m".to_string(),
                String::new(),
                "\x1b[1morangu approves this patch\x1b[0m".to_string(),
                String::new(),
                format!(
                    "Generated by: \x1b[1morangu {}\x1b[22m (gemma)",
                    crate::VERSION
                ),
            ]
        );
        // The clipboard copy is the raw Markdown report.
        assert_eq!(
            clipboard,
            format!(
                "## Overall\n\
             \n\
             - ready to merge\n\
             \n\
             ## Code\n\
             \n\
             - **a.rs**: tighten error handling\n\
             \n\
             ## Security\n\
             \n\
             No issues found\n\
             \n\
             ## Memory\n\
             \n\
             No issues found\n\
             \n\
             ## Performance\n\
             \n\
             No issues found\n\
             \n\
             ## Test Suite\n\
             \n\
             No issues found\n\
             \n\
             ## Documentation\n\
             \n\
             No issues found\n\
             \n\
             ## Conclusion\n\
             \n\
             **orangu approves this patch**\n\
             \n\
             Generated by: **orangu {}** (gemma)",
                crate::VERSION,
            )
        );
    }

    #[test]
    fn auto_review_conclusion_rejects_and_groups_unapproved_files() {
        use crate::commands::ReviewLaunch;
        use crate::review::AutoReviewState;
        use orangu::tui::{ReviewEntry, ReviewStatus};

        let entry = |path: &str, status| ReviewEntry {
            path: path.to_string(),
            status,
            diff_lines: vec!["+x".to_string()],
            patch: String::new(),
        };
        let state = AutoReviewState::new(ReviewLaunch {
            files: vec![
                entry("a.rs", ReviewStatus::Approved),
                entry("b.rs", ReviewStatus::Unreviewed),
                entry("c.rs", ReviewStatus::Rejected),
                entry("d.rs", ReviewStatus::Rejected),
            ],
        });

        // Any rejected or not-reviewed file rejects the patch; the files are
        // listed in bold, grouped by their status, rejected first.
        assert_eq!(state.conclusion_verdict(), "orangu rejects this patch");
        assert_eq!(
            state.conclusion_findings(),
            vec![
                "Rejected: **c.rs**".to_string(),
                "Rejected: **d.rs**".to_string(),
                "Not reviewed: **b.rs**".to_string(),
            ]
        );

        // All approved: a clean verdict with no file list.
        let state = AutoReviewState::new(ReviewLaunch {
            files: vec![entry("a.rs", ReviewStatus::Approved)],
        });
        assert_eq!(state.conclusion_verdict(), "orangu approves this patch");
        assert!(state.conclusion_findings().is_empty());
    }

    #[test]
    fn auto_review_report_lines_show_pending_then_findings() {
        use crate::commands::ReviewLaunch;
        use crate::review::AutoReviewState;

        let mut state = AutoReviewState::new(ReviewLaunch { files: Vec::new() });
        let lines = state.report_lines();
        // Seven findings categories (bold heading without the Markdown `##`
        // markers, blank line, placeholder, blank separator) plus the
        // Conclusion heading, its blank line, and its placeholder, then a blank
        // line and the closing `Generated by:` attribution.
        assert_eq!(lines.len(), 7 * 4 + 3 + 2);
        assert_eq!(lines[0], "\x1b[1mOverall\x1b[0m");
        assert_eq!(lines[2], "\x1b[2m(pending)\x1b[0m");
        assert_eq!(lines[28], "\x1b[1mConclusion\x1b[0m");
        assert_eq!(lines[30], "\x1b[2m(pending)\x1b[0m");
        assert_eq!(
            lines[32],
            format!("Generated by: \x1b[1morangu {}\x1b[22m", crate::VERSION)
        );

        state.sections[0].push("**a.rs**: ready".to_string());
        state.finish();
        let lines = state.report_lines();
        // Findings render as bullets with the bold file name resolved to ANSI.
        assert_eq!(lines[2], "- \x1b[1ma.rs\x1b[22m: ready");
        // Completed categories without findings switch to "No issues found".
        assert_eq!(lines[6], "No issues found");
        // The Conclusion resolves to the patch verdict, standing alone in
        // bold rather than as a list item.
        assert_eq!(lines[30], "\x1b[1morangu approves this patch\x1b[0m");
    }
}
