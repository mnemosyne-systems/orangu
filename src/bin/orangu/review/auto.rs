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
pub(crate) const AUTO_REVIEW_DOCUMENTATION_EXTENSIONS: [&str; 8] = [
    "md", "markdown", "rst", "adoc", "asciidoc", "txt", "org", "tex",
];

/// Whether `path` is detected as documentation, by its file extension.
pub(crate) fn auto_review_documentation_file(path: &str) -> bool {
    let extension = Path::new(path)
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("");
    AUTO_REVIEW_DOCUMENTATION_EXTENSIONS
        .iter()
        .any(|known| extension.eq_ignore_ascii_case(known))
}

/// The categories scanned for `path`, enabled by its file extension: a file
/// detected as documentation skips the code-related checks and is reviewed
/// only for `Documentation`; everything else is scanned for every per-file
/// category.
pub(crate) fn auto_review_file_categories(path: &str) -> &'static [(usize, &'static str)] {
    if auto_review_documentation_file(path) {
        // `Documentation` is the last per-file category.
        &AUTO_REVIEW_FILE_CATEGORIES[AUTO_REVIEW_FILE_CATEGORIES.len() - 1..]
    } else {
        &AUTO_REVIEW_FILE_CATEGORIES[..]
    }
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

/// Interactive state for `/auto_review` mode.
pub(crate) struct AutoReviewState {
    pub(crate) files: Vec<ReviewEntry>,
    /// The file highlighted in the right pane: the one being reviewed while
    /// the run is in progress, or the one picked with Alt+j/Alt+k while
    /// browsing afterwards. `None` once the run ends, until the user
    /// navigates.
    pub(crate) selected: Option<usize>,
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
            reviewing: None,
            scroll: 0,
            x_offset: 0,
            sections: Default::default(),
            status: "Starting".to_string(),
            started: std::time::Instant::now(),
            finished: None,
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

    /// The status area's full text: the current activity, then the total time
    /// spent on the run (after the progress information).
    pub(crate) fn status_text(&self) -> String {
        format!(
            "{}  Time: {}",
            self.status,
            orangu::tui::format_status_duration(self.elapsed()),
        )
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
    /// verdict (in Markdown bold), grouped by their status, rejected first.
    /// Empty when every file is approved.
    pub(crate) fn conclusion_findings(&self) -> Vec<String> {
        let mut lines = Vec::new();
        for file in &self.files {
            if file.status == ReviewStatus::Rejected {
                lines.push(format!("Rejected: **{}**", file.path));
            }
        }
        for file in &self.files {
            if file.status == ReviewStatus::Unreviewed {
                lines.push(format!("Not reviewed: **{}**", file.path));
            }
        }
        lines
    }

    /// The left-pane report, rendered for the console: each category as a
    /// bold heading (the `##` markers of the Markdown report are consumed,
    /// not displayed) followed by its findings as a bullet list with the
    /// `**file**` names resolved to bold, with a dimmed placeholder while the
    /// run is still in progress, ending with the synthesized `Conclusion`.
    pub(crate) fn report_lines(&self) -> Vec<String> {
        let pending = !(self.done || self.cancelled);
        let mut lines = Vec::new();
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
                for finding in section {
                    let bullet = render_markdown_for_console(&auto_review_finding_bullet(finding));
                    lines.extend(bullet.lines().map(str::to_string));
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
            let findings = self.conclusion_findings();
            if !findings.is_empty() {
                lines.push(String::new());
                for line in findings {
                    lines.push(render_markdown_for_console(&format!("- {line}")));
                }
            }
        }
        // The report closes with the orangu version and reviewing model.
        lines.push(String::new());
        lines.push(render_markdown_for_console(&self.generated_by_markdown()));
        lines
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
        self.set_file_status(index, ReviewStatus::Approved);
        // A finding for the file is bold-prefixed with either `**path**:` (no
        // line) or `**path:line**:`, so match both forms.
        let without_line = format!("**{path}**:");
        let with_line = format!("**{path}:");
        for section in &mut self.sections {
            section.retain(|finding| {
                !finding.starts_with(&without_line) && !finding.starts_with(&with_line)
            });
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

    /// Record a category review whose response carried neither a verdict nor
    /// findings — typically truncated by the response cap (see
    /// `review_max_tokens`) or empty. The file keeps its white (unreviewed)
    /// dot and the problem is noted in the `Overall` section.
    pub(crate) fn record_unparseable(&mut self, index: usize, category: &str) {
        if let Some(file) = self.files.get(index) {
            self.sections[0].push(format!(
                "**{}**: {category} review returned no verdict and no findings",
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
    let lower = lower.trim_end_matches(['.', '!']);
    lower.is_empty()
        || matches!(
            lower,
            "none" | "no findings" | "no issues" | "no issues found" | "nothing" | "n/a"
        )
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

pub(crate) fn print_auto_review_screen(
    state: &AutoReviewState,
    viewport: &ViewportState,
    chrome: ReviewChrome<'_>,
    left_status: Option<StatusFragment>,
    blink_on: bool,
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
            scroll: state.scroll,
            x_offset: state.x_offset,
            status: &status_text,
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
) -> Result<AutoReviewState> {
    let mut state = AutoReviewState::new(launch);
    state.model = chrome.current_model.to_string();
    let mut exit_requested = false;
    let total = state.files.len();
    // The run's request count: each file is scanned only for the categories
    // its detected kind enables (a file detected as documentation skips the
    // code-related checks), plus the whole-change pass.
    let total_requests: usize = state
        .files
        .iter()
        .map(|file| auto_review_file_categories(&file.path).len())
        .sum::<usize>()
        + 1;
    let mut completed = 0usize;
    // Review each file by itself, one focused request per enabled category.
    // Every request runs in a scratch session so the reviews stay independent
    // and the main chat session is left untouched.
    'auto: for index in 0..total {
        state.selected = Some(index);
        let (path, patch) = {
            let file = &state.files[index];
            (file.path.clone(), file.patch.clone())
        };
        state.reviewing = Some(index);
        let mut any_rejected = false;
        let mut any_failed = false;
        for (section, focus) in auto_review_file_categories(&path) {
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
            )
            .await?;
            match outcome {
                AutoReviewRequestOutcome::Completed(Ok(text)) => {
                    completed += 1;
                    // No tools run during auto review requests.
                    usage_stats.record_response(
                        llm_start.elapsed(),
                        &text,
                        std::time::Duration::ZERO,
                    );
                    let (verdict, findings) = parse_auto_review_category_response(&text);
                    if verdict.is_none() && findings.is_empty() {
                        // A response carrying neither a verdict nor findings
                        // (e.g. truncated by the response cap) must not pass
                        // silently as a clean review.
                        any_failed = true;
                        state.record_unparseable(index, category);
                    } else {
                        // Without an explicit verdict, a category passes only
                        // when its review found nothing.
                        if !verdict.unwrap_or(findings.is_empty()) {
                            any_rejected = true;
                        }
                        state.apply_category_result(index, section, findings);
                    }
                }
                AutoReviewRequestOutcome::Completed(Err(err)) => {
                    completed += 1;
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
    // Keep the report on screen for browsing until Alt+x/Esc Esc.
    if !exit_requested {
        run_auto_review_browse(&mut state, viewport, chrome, workspace, terminal)?;
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
                print_auto_review_screen(state, viewport, chrome, status, blink_on);
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
) -> Result<()> {
    let mut escape_cancel = EscapeCancelState::default();
    loop {
        let body_height = auto_review_pane_body_height(
            viewport.actual_height,
            chrome.prompt_branch,
            viewport.actual_width,
        );
        let right_width = orangu::tui::review_right_width(&state.files, viewport.actual_width);
        let left_width = viewport.actual_width.saturating_sub(right_width + 1).max(1);
        state.clamp(body_height, left_width);
        print_auto_review_screen(state, viewport, chrome, None, false);
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

        match (code, alt) {
            (KeyCode::Char('x'), true) => return Ok(()),
            (KeyCode::Char('j'), true) => state.select_next(),
            (KeyCode::Char('k'), true) => state.select_prev(),
            (KeyCode::Char('a'), true) => state.approve_selected(),
            (KeyCode::Char('r'), true) => state.open_reject(),
            (KeyCode::Char('e'), true) => {
                if let Some(path) = state.selected_path()
                    && let Err(err) = open_in_editor(workspace, &path, terminal)
                {
                    // No feedback popup in auto review; surface the error in
                    // the status area.
                    state.status = format!("Open {path} failed: {err:#}");
                }
            }
            (KeyCode::Up, _) => state.scroll = state.scroll.saturating_sub(1),
            (KeyCode::Down, _) => state.scroll = state.scroll.saturating_add(1),
            (KeyCode::Left, _) => state.x_offset = state.x_offset.saturating_sub(1),
            (KeyCode::Right, _) => state.x_offset = state.x_offset.saturating_add(1),
            (KeyCode::PageUp, _) => state.scroll = state.scroll.saturating_sub(body_height),
            (KeyCode::PageDown, _) => state.scroll = state.scroll.saturating_add(body_height),
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

        state.finish();
        let frozen = state.status_text();
        assert!(frozen.starts_with("Done  Time: "));
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
    fn auto_review_unparseable_response_is_recorded_not_approved() {
        use crate::commands::ReviewLaunch;
        use crate::review::{AutoReviewState, parse_auto_review_category_response};
        use orangu::tui::{ReviewEntry, ReviewStatus};

        // An empty (e.g. cap-truncated) response parses to no verdict and no
        // findings...
        let (verdict, findings) = parse_auto_review_category_response("");
        assert_eq!(verdict, None);
        assert!(findings.is_empty());

        // ...which is recorded as a failed category review under Overall
        // instead of silently passing as clean.
        let mut state = AutoReviewState::new(ReviewLaunch {
            files: vec![ReviewEntry {
                path: "a.rs".to_string(),
                status: ReviewStatus::Unreviewed,
                diff_lines: vec!["+x".to_string()],
                patch: String::new(),
            }],
        });
        state.record_unparseable(0, "Security");
        assert_eq!(
            state.sections[0],
            vec!["**a.rs**: Security review returned no verdict and no findings".to_string()]
        );
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
        use crate::review::auto_review_file_categories;
        use crate::review::{AUTO_REVIEW_CATEGORIES, AUTO_REVIEW_FILE_CATEGORIES};

        // Code files are scanned for every per-file category.
        assert_eq!(
            auto_review_file_categories("src/main.rs"),
            &AUTO_REVIEW_FILE_CATEGORIES[..]
        );
        // Files without an extension too.
        assert_eq!(
            auto_review_file_categories("Makefile"),
            &AUTO_REVIEW_FILE_CATEGORIES[..]
        );
        // Known documentation extensions go straight to Documentation,
        // case-insensitively.
        for path in ["README.md", "doc/manual.RST", "notes.txt"] {
            let categories = auto_review_file_categories(path);
            assert_eq!(
                categories.len(),
                1,
                "expected only Documentation for {path:?}"
            );
            assert_eq!(AUTO_REVIEW_CATEGORIES[categories[0].0], "Documentation");
        }

        // The detection that decides whether a file's code-related checks
        // can be skipped.
        use crate::review::auto_review_documentation_file;
        assert!(auto_review_documentation_file("README.md"));
        assert!(!auto_review_documentation_file("src/main.rs"));
        assert!(!auto_review_documentation_file("Makefile"));
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
