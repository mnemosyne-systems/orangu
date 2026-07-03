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

use super::*;
use crate::workspaces::WorkspacePlacement;
use std::time::Duration;
use terminal_size::{Height, Width, terminal_size};

/// The workspace tab bar to draw: how many tabs are open, which is active
/// (0-based, left to right) and where the bar sits relative to the screen.
#[derive(Clone, Copy)]
pub struct WorkspaceTabsView {
    pub count: usize,
    pub active: usize,
    pub placement: WorkspacePlacement,
}

/// Status of a single open workspace tab, shown as a colored dot in the tab
/// bar when the `feedback` configuration key is on.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TabStatus {
    /// The workspace directory and branch are valid (green ●).
    Valid,
    /// The live branch no longer matches the one the tab was opened on —
    /// probably deleted or merged (red ●).
    BranchGone,
    /// The active tab is currently streaming a response (blinking white ●).
    Working,
}

pub(crate) const USER_INPUT_BACKGROUND: &str = "\x1b[48;2;44;44;44m";
const GHOST_TEXT: &str = "\x1b[38;2;120;120;120m";
const THINKING_TEXT: &str = "Thinking";
const WORKING_TEXT: &str = "Working";
const THINKING_SHADE_LEVELS: &[u8] = &[230, 210, 190, 170, 150, 130, 110, 90];

pub struct ScreenRenderArgs<'a> {
    pub version: &'a str,
    pub current_model: &'a str,
    pub endpoint: &'a str,
    pub workspace: &'a std::path::Path,
    pub prompt_branch: Option<&'a str>,
    pub status: HeaderStatus,
    pub banner: Banner,
    pub tab_bar: Option<WorkspaceTabsView>,
    /// Per-tab status dots for the tab bar, in left-to-right order. Empty when
    /// `feedback` is off; when non-empty has exactly `tab_bar.count` entries.
    pub tab_statuses: &'a [TabStatus],
    pub transcript: &'a [TranscriptLine],
    pub scroll_offset: usize,
    pub left_status: Option<StatusFragment>,
    pub pending_count: usize,
    pub pending_line: Option<&'a str>,
    pub input: &'a str,
    pub cursor: usize,
    pub ghost: &'a str,
    pub virtual_width: usize,
    pub actual_width: usize,
    pub actual_height: usize,
    pub x_offset: usize,
    pub dropdown_candidates: Option<&'a [(String, String)]>,
    pub dropdown_selected: usize,
    pub valid_command_len: usize,
}

/// Inputs for the bottom prompt frame (separator, input window, status bar),
/// shared by the normal screen, `/review` mode, and the manual viewer.
pub struct PromptFrameArgs<'a> {
    pub header_height: usize,
    pub current_model: &'a str,
    pub left_status: Option<StatusFragment>,
    pub pending_count: usize,
    pub prompt_prefix: &'a str,
    pub input: &'a str,
    pub cursor: usize,
    pub ghost: &'a str,
    pub height: usize,
    pub actual_width: usize,
    pub valid_command_len: usize,
}

fn tab_dot(status: TabStatus) -> &'static str {
    match status {
        TabStatus::Valid => "\x1b[38;2;80;200;120m●\x1b[0m",
        TabStatus::BranchGone => "\x1b[38;2;220;80;80m●\x1b[0m",
        TabStatus::Working => "\x1b[5;38;2;230;230;230m●\x1b[0m",
    }
}

/// The horizontal workspace tab bar (`1 │ 2 │ 3`, active tab bold) for top and
/// bottom placement, clipped to `width`. When `statuses` is non-empty a
/// colored dot precedes each tab number.
fn horizontal_tab_bar(view: WorkspaceTabsView, width: usize, statuses: &[TabStatus]) -> String {
    let cells: Vec<String> = (0..view.count)
        .map(|index| {
            let number = index + 1;
            let dot = statuses.get(index).copied().map(tab_dot).unwrap_or("");
            if index == view.active {
                format!("{dot}\x1b[1m{number}\x1b[0m")
            } else {
                format!("{dot}{GHOST_TEXT}{number}{ANSI_RESET}")
            }
        })
        .collect();
    clip_line(&cells.join(" │ "), 0, width)
}

/// Width of the vertical tab gutter: the widest tab number plus ` │ `, and one
/// extra column when status dots are shown.
fn tab_gutter_width(view: WorkspaceTabsView, has_dots: bool) -> usize {
    view.count.to_string().len() + 3 + usize::from(has_dots)
}

/// The gutter cell for screen `row` on a left/right placement: the tab number
/// (bold when active) on the first `count` rows, blank after, always carrying
/// the separator bar — `N │ ` on the left, ` │ N` on the right. When
/// `statuses` is non-empty, a colored dot precedes each tab number.
fn tab_gutter_cell(
    view: WorkspaceTabsView,
    row: usize,
    left: bool,
    statuses: &[TabStatus],
) -> String {
    let digits = view.count.to_string().len();
    let dot = if statuses.is_empty() {
        String::new()
    } else if let Some(&status) = statuses.get(row) {
        tab_dot(status).to_string()
    } else {
        " ".to_string()
    };
    let label = if row < view.count {
        let number = row + 1;
        if row == view.active {
            format!("\x1b[1m{number:>digits$}\x1b[0m")
        } else {
            format!("{GHOST_TEXT}{number:>digits$}{ANSI_RESET}")
        }
    } else {
        " ".repeat(digits)
    };
    if left {
        format!("{dot}{label} │ ")
    } else {
        format!(" │ {dot}{label}")
    }
}

pub fn render_screen(args: ScreenRenderArgs<'_>) -> String {
    let width = args.virtual_width.max(1);
    let actual_width = args.actual_width.max(1);
    let actual_height = args.actual_height.max(1);

    // Where the workspace tab bar sits. Top/bottom take a row of the screen;
    // left/right take a gutter column from the banner and output region.
    let placement = args.tab_bar.map(|view| view.placement);
    let top_row = usize::from(placement == Some(WorkspacePlacement::Top));
    let bottom_row = usize::from(placement == Some(WorkspacePlacement::Bottom));
    let left = placement == Some(WorkspacePlacement::Left);
    let right = placement == Some(WorkspacePlacement::Right);
    let has_dots = !args.tab_statuses.is_empty();
    let gutter = match args.tab_bar {
        Some(view) if left || right => tab_gutter_width(view, has_dots),
        _ => 0,
    };
    let inner_width = actual_width.saturating_sub(gutter).max(1);
    // The prompt frame keeps the full width; a bottom bar shrinks its height.
    let frame_height = actual_height.saturating_sub(bottom_row).max(1);

    let header = render_header(
        args.version,
        args.current_model,
        args.endpoint,
        args.workspace,
        args.status,
        args.banner,
        inner_width,
    );
    let header_line_count = header.lines().count();
    let prompt_prefix = prompt_prefix(args.prompt_branch);
    let input_lines = wrapped_input_lines(args.input, actual_width, &prompt_prefix);
    let prompt_frame_height = input_lines.len() + 3;

    // Priority: prompt frame first, then banner, then output.
    let rows_above_prompt = frame_height.saturating_sub(prompt_frame_height);
    // Banner = header lines + 1 blank separator line; truncate to what fits.
    let full_banner_height = header_line_count + 1;
    let banner_rows = full_banner_height.min(rows_above_prompt);
    // A top bar takes one row from the output area.
    let available_output_rows =
        available_output_rows(rows_above_prompt, banner_rows).saturating_sub(top_row);

    let mut output_lines = args
        .transcript
        .iter()
        .map(|line| {
            let (rendered, offset) = match line {
                TranscriptLine::UserInput(_) => (render_transcript_line(line, inner_width), 0),
                _ => (render_transcript_line(line, width), args.x_offset),
            };
            clip_line(&rendered, offset, inner_width)
        })
        .collect::<Vec<_>>();
    if let Some(pending_line) = args.pending_line {
        if pending_line.is_empty() {
            output_lines.push(String::new());
        } else {
            output_lines.extend(
                pending_line
                    .lines()
                    .map(|l| clip_line(l, args.x_offset, inner_width)),
            );
        }
    }
    let max_scroll_offset = output_lines.len().saturating_sub(available_output_rows);
    let scroll_offset = args.scroll_offset.min(max_scroll_offset);
    let visible_end = output_lines.len().saturating_sub(scroll_offset);
    let visible_start = visible_end.saturating_sub(available_output_rows);
    let visible_lines = &output_lines[visible_start..visible_end];

    // The banner and output region, as rows at `inner_width`.
    let mut upper: Vec<String> = Vec::new();
    if banner_rows > 0 {
        let shown_header_lines = banner_rows.min(header_line_count);
        for line in header.split("\r\n").take(shown_header_lines) {
            upper.push(clip_line(line, 0, inner_width));
        }
        if banner_rows > header_line_count {
            upper.push(String::new());
        }
    }
    upper.extend(visible_lines.iter().cloned());

    let mut screen = String::new();

    // Pre-compute the horizontal bar string once; used by at most one of the
    // top/bottom branches (they are mutually exclusive), but computing upfront
    // avoids repeating the Vec+join allocation if this function is ever called
    // in a context where placement could be re-evaluated.
    let h_tab_bar: Option<String> = if top_row == 1 || bottom_row == 1 {
        args.tab_bar
            .map(|view| horizontal_tab_bar(view, actual_width, args.tab_statuses))
    } else {
        None
    };

    // Top tab bar.
    if top_row == 1
        && let Some(ref bar) = h_tab_bar
    {
        screen.push_str(bar);
        screen.push_str("\r\n");
    }

    // Banner and output, with a left/right gutter when placed vertically.
    if let (true, Some(view)) = (left || right, args.tab_bar) {
        for row in 0..rows_above_prompt {
            let content = upper.get(row).cloned().unwrap_or_default();
            let cell = tab_gutter_cell(view, row, left, args.tab_statuses);
            if left {
                screen.push_str(&cell);
                screen.push_str(&content);
            } else {
                let pad = inner_width.saturating_sub(visible_line_width(&content));
                screen.push_str(&content);
                for _ in 0..pad {
                    screen.push(' ');
                }
                screen.push_str(&cell);
            }
            screen.push_str("\r\n");
        }
    } else if !upper.is_empty() {
        screen.push_str(&upper.join("\r\n"));
        screen.push_str("\r\n");
    }

    if let Some(candidates) = args.dropdown_candidates
        && !candidates.is_empty()
    {
        let pf_height = frame_height.max(banner_rows + input_lines.len() + 3);
        let pf_top_row = (pf_height.saturating_sub(input_lines.len() + 2)).max(banner_rows + 1);
        screen.push_str(&render_dropdown_popup(
            candidates,
            args.dropdown_selected,
            pf_top_row,
            actual_width,
        ));
    }

    screen.push_str(&render_prompt_frame(PromptFrameArgs {
        header_height: banner_rows,
        current_model: args.current_model,
        left_status: args.left_status,
        pending_count: args.pending_count,
        prompt_prefix: &prompt_prefix,
        input: args.input,
        cursor: args.cursor,
        ghost: args.ghost,
        height: frame_height,
        actual_width,
        valid_command_len: args.valid_command_len,
    }));

    // Bottom tab bar on the last terminal row.
    if bottom_row == 1
        && let Some(ref bar) = h_tab_bar
    {
        screen.push_str(&format!("\x1b[{actual_height};1H{bar}"));
        // Writing the bar moved the cursor to the last row; put it back in the
        // input area so keystrokes land in the right place.
        let input_height = input_lines.len();
        let pf_height = frame_height.max(banner_rows + input_height + 3);
        let pf_top_row = (pf_height.saturating_sub(input_height + 2)).max(banner_rows + 1);
        let pf_input_start = pf_top_row + 1;
        let pf_prompt_width = prompt_prefix.chars().count();
        let (cr_offset, cc_offset) =
            cursor_position(args.input, args.cursor, actual_width, &prompt_prefix);
        let cr = pf_input_start + cr_offset;
        let cc = (1 + pf_prompt_width + cc_offset).max(1);
        screen.push_str(&format!("\x1b[{cr};{cc}H"));
    }

    screen
}

#[allow(clippy::too_many_arguments)]
pub fn output_view_rows(
    version: &str,
    current_model: &str,
    endpoint: &str,
    workspace: &std::path::Path,
    prompt_branch: Option<&str>,
    status: HeaderStatus,
    input: &str,
    actual_width: usize,
    actual_height: usize,
) -> usize {
    let header = render_header(
        version,
        current_model,
        endpoint,
        workspace,
        status,
        Banner::Left,
        actual_width,
    );
    let header_line_count = header.lines().count();
    let prompt_prefix = prompt_prefix(prompt_branch);
    let input_lines = wrapped_input_lines(input, actual_width.max(1), &prompt_prefix);
    let prompt_frame_height = input_lines.len() + 3;
    let rows_above_prompt = actual_height.saturating_sub(prompt_frame_height);
    let banner_rows = (header_line_count + 1).min(rows_above_prompt);
    available_output_rows(rows_above_prompt, banner_rows)
}

pub fn render_thinking_status(frame: usize, elapsed: Duration) -> StatusFragment {
    let suffix = format!(" {}", format_elapsed_timer(elapsed));
    StatusFragment {
        rendered: format!(
            "{}{}{}",
            render_rolling_text(THINKING_TEXT, frame),
            ANSI_RESET,
            suffix
        ),
        visible_width: THINKING_TEXT.chars().count() + suffix.chars().count(),
    }
}

pub fn render_tool_running_status(frame: usize, elapsed: Duration) -> StatusFragment {
    let suffix = format!(" {}", format_elapsed_timer(elapsed));
    StatusFragment {
        rendered: format!(
            "{}{}{}",
            render_rolling_text(WORKING_TEXT, frame),
            ANSI_RESET,
            suffix
        ),
        visible_width: WORKING_TEXT.chars().count() + suffix.chars().count(),
    }
}

/// Like [`render_tool_running_status`] but with a completion percentage and,
/// once there is enough to extrapolate from, an estimate of the time remaining
/// — e.g. `Working (7%) (10m3s, ~2h10m left)`. `permille` is progress in
/// thousandths (0..=1000); `eta_total_ms` is the task's current estimate of the
/// *total* run time. The remaining time is `estimate − elapsed`, so it counts
/// down between updates rather than drifting up.
pub fn render_tool_progress_status(
    frame: usize,
    elapsed: Duration,
    permille: u64,
    eta_total_ms: u64,
) -> StatusFragment {
    let permille = permille.min(1000);
    let percent = permille / 10;
    let eta = if eta_total_ms > 0 && permille < 1000 {
        let elapsed_ms = elapsed.as_millis() as u64;
        let remaining = Duration::from_millis(eta_total_ms.saturating_sub(elapsed_ms));
        format!(", ~{} left", format_status_duration(remaining))
    } else {
        String::new()
    };
    let suffix = format!(" ({percent}%) ({}{eta})", format_status_duration(elapsed));
    StatusFragment {
        rendered: format!(
            "{}{}{}",
            render_rolling_text(WORKING_TEXT, frame),
            ANSI_RESET,
            suffix
        ),
        visible_width: WORKING_TEXT.chars().count() + suffix.chars().count(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatusFragment {
    pub rendered: String,
    pub visible_width: usize,
}

impl StatusFragment {
    pub fn plain(rendered: String) -> Self {
        let visible_width = rendered.chars().count();
        Self {
            rendered,
            visible_width,
        }
    }
}

pub fn render_working_status(frame: usize, rate: f64, elapsed: Duration) -> StatusFragment {
    let suffix = format!(" @ {rate:.1} t/s {}", format_elapsed_timer(elapsed));
    StatusFragment {
        rendered: format!(
            "{}{}{}",
            render_rolling_text(WORKING_TEXT, frame),
            ANSI_RESET,
            suffix
        ),
        visible_width: WORKING_TEXT.chars().count() + suffix.chars().count(),
    }
}

/// An elapsed duration in its shortest form — `5s`, `1m5s`, `1h2m3s` — as used
/// by the Thinking/Working status timers and the auto review status area.
pub fn format_status_duration(elapsed: Duration) -> String {
    let seconds = elapsed.as_secs();
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    let seconds = seconds % 60;
    if hours > 0 {
        format!("{hours}h{minutes}m{seconds}s")
    } else if minutes > 0 {
        format!("{minutes}m{seconds}s")
    } else {
        format!("{seconds}s")
    }
}

fn format_elapsed_timer(elapsed: Duration) -> String {
    format!("({})", format_status_duration(elapsed))
}

fn available_output_rows(rows_above_prompt: usize, banner_rows: usize) -> usize {
    rows_above_prompt.saturating_sub(banner_rows)
}

/// Render the bottom prompt frame with absolute cursor positioning: the top
/// separator, the input window, the bottom separator, and the status line.
pub fn render_prompt_frame(args: PromptFrameArgs<'_>) -> String {
    let input_lines = wrapped_input_lines(args.input, args.actual_width, args.prompt_prefix);
    let input_height = input_lines.len();
    let height = args.height.max(args.header_height + input_height + 3);
    let top_row = (height.saturating_sub(input_height + 2)).max(args.header_height + 1);
    let input_start_row = top_row + 1;
    let bottom_row = input_start_row + input_height;
    let model_row = bottom_row + 1;
    let separator = "━".repeat(args.actual_width);
    let prompt_width = args.prompt_prefix.chars().count();
    let mut frame = format!("\x1b[{top_row};1H{separator}");

    let cmd_len = args.valid_command_len;

    let mut char_offset = 0;
    let last_input_index = input_lines.len().saturating_sub(1);
    for (index, input_line) in input_lines.iter().enumerate() {
        let row = input_start_row + index;
        let content = truncate_to_width(input_line, args.actual_width.saturating_sub(prompt_width));
        let content_width = content.chars().count();

        let highlighted_content = if cmd_len > 0 {
            let mut res = String::new();
            for (i, ch) in content.chars().enumerate() {
                let global_idx = char_offset + i;
                if global_idx == 0 {
                    res.push_str("\x1b[38;2;210;140;70m");
                }
                if global_idx == cmd_len {
                    res.push_str("\x1b[39m");
                }
                res.push(ch);
            }
            if char_offset + content_width <= cmd_len {
                res.push_str("\x1b[39m");
            }
            res
        } else {
            content.clone()
        };
        char_offset += content_width;

        let mut full_line = format!("{}{}", args.prompt_prefix, highlighted_content);
        let mut used = content_width + prompt_width;

        // The ghost suffix trails the input on the cursor's (final) line, in grey.
        if index == last_input_index && !args.ghost.is_empty() {
            let ghost = truncate_to_width(args.ghost, args.actual_width.saturating_sub(used));
            let ghost_width = ghost.chars().count();
            if ghost_width > 0 {
                full_line.push_str(&format!("{GHOST_TEXT}{ghost}{ANSI_RESET}"));
                used += ghost_width;
            }
        }

        if args.actual_width > used {
            full_line.push_str(&" ".repeat(args.actual_width - used));
        }
        frame.push_str(&format!("\x1b[{row};1H{full_line}"));
    }

    let (cursor_row_offset, cursor_col_offset) = cursor_position(
        args.input,
        args.cursor,
        args.actual_width,
        args.prompt_prefix,
    );
    let cursor_row = input_start_row + cursor_row_offset;
    let display_cursor_col = (1 + prompt_width + cursor_col_offset).max(1);

    let status_line = render_status_line(
        args.actual_width,
        args.left_status.as_ref(),
        args.current_model,
        args.pending_count,
    );
    frame.push_str(&format!(
        "\x1b[{bottom_row};1H{separator}\x1b[{model_row};1H{status_line}\x1b[{cursor_row};{display_cursor_col}H"
    ));
    frame
}

fn render_transcript_line(line: &TranscriptLine, width: usize) -> String {
    match line {
        TranscriptLine::Plain(content) | TranscriptLine::Wide(content) => content.clone(),
        TranscriptLine::UserInput(content) => {
            let padding = width.saturating_sub(content.chars().count());
            format!(
                "{USER_INPUT_BACKGROUND}{content}{}{ANSI_RESET}",
                " ".repeat(padding)
            )
        }
    }
}

pub(crate) fn wrapped_input_lines(input: &str, width: usize, prompt_prefix: &str) -> Vec<String> {
    let input_width = width.saturating_sub(prompt_prefix.chars().count()).max(1);
    if input.is_empty() {
        return vec![String::new()];
    }

    let mut lines = Vec::new();
    let mut current = String::new();

    for ch in input.chars() {
        current.push(ch);
        if current.chars().count() == input_width {
            lines.push(current);
            current = String::new();
        }
    }

    if current.is_empty() {
        lines.push(String::new());
    } else {
        lines.push(current);
    }

    lines
}

pub(crate) fn cursor_position(
    input: &str,
    cursor: usize,
    width: usize,
    prompt_prefix: &str,
) -> (usize, usize) {
    let input_width = width.saturating_sub(prompt_prefix.chars().count()).max(1);
    let prefix_chars = input[..cursor.min(input.len())].chars().count();
    (prefix_chars / input_width, prefix_chars % input_width)
}

/// The input-window prompt prefix: `<branch>> ` on a branch, `> ` otherwise.
pub fn prompt_prefix(branch_name: Option<&str>) -> String {
    match branch_name {
        Some(branch_name) if !branch_name.trim().is_empty() => format!("{branch_name}> "),
        _ => "> ".to_string(),
    }
}

fn render_status_line(
    width: usize,
    left_status: Option<&StatusFragment>,
    current_model: &str,
    pending_count: usize,
) -> String {
    // Priority: left_status (Working/Thinking) > model name > Pending.
    let left_visible_width = left_status.map_or(0, |s| s.visible_width);
    let right_space = width.saturating_sub(left_visible_width);

    let model_width = current_model.chars().count();
    let show_model = right_space >= model_width;

    let pending_text = (pending_count > 0).then(|| format!("Pending: {pending_count}"));
    let pending_width = pending_text.as_ref().map_or(0, |s| s.chars().count());
    // Gap is the space between the left status and the model name (or right edge if no model).
    let gap = if show_model {
        right_space.saturating_sub(model_width)
    } else {
        right_space
    };
    let show_pending = show_model && pending_text.is_some() && gap >= pending_width;

    let mut right_cells = vec![' '; right_space];
    if show_model {
        let start = right_space - model_width;
        for (i, ch) in current_model.chars().enumerate() {
            if start + i < right_space {
                right_cells[start + i] = ch;
            }
        }
    }
    if show_pending {
        let pending = pending_text.as_deref().unwrap_or("");
        let start = (gap - pending_width) / 2;
        for (i, ch) in pending.chars().enumerate() {
            if start + i < right_space {
                right_cells[start + i] = ch;
            }
        }
    }

    let right: String = right_cells.into_iter().collect();
    if let Some(left) = left_status.filter(|s| s.visible_width > 0) {
        format!("{}{}", left.rendered, right)
    } else {
        right
    }
}

fn render_rolling_text(text: &str, frame: usize) -> String {
    let mut rendered = String::new();
    let offset = frame % THINKING_SHADE_LEVELS.len();

    for (index, ch) in text.chars().enumerate() {
        let shade_index =
            (index + THINKING_SHADE_LEVELS.len() - offset) % THINKING_SHADE_LEVELS.len();
        let shade = THINKING_SHADE_LEVELS[shade_index];
        rendered.push_str(&format!("\x1b[38;2;{shade};{shade};{shade}m{ch}"));
    }

    rendered
}

fn truncate_to_width(input: &str, width: usize) -> String {
    input.chars().take(width).collect()
}

fn render_dropdown_popup(
    candidates: &[(String, String)],
    selected: usize,
    bottom_row: usize,
    actual_width: usize,
) -> String {
    let mut popup = String::new();
    let max_height = 10.min(candidates.len());
    let start_idx = selected
        .saturating_sub(max_height / 2)
        .min(candidates.len().saturating_sub(max_height));
    let display_candidates = &candidates[start_idx..start_idx + max_height];

    let top_row = bottom_row.saturating_sub(display_candidates.len()).max(1);

    for (i, (cmd, desc)) in display_candidates.iter().enumerate() {
        let row = top_row + i;
        let is_selected = (start_idx + i) == selected;
        let max_width = actual_width.min(100); // don't stretch too far

        let mut cmd_padded = cmd.clone();
        if cmd_padded.chars().count() < 32 {
            cmd_padded.push_str(&" ".repeat(32 - cmd_padded.chars().count()));
        }

        let desc_truncated = truncate_to_width(desc, max_width.saturating_sub(34));
        let visible_len = cmd_padded.chars().count() + 1 + desc_truncated.chars().count();
        let padding = max_width.saturating_sub(visible_len);

        let line = if is_selected {
            format!(
                "\x1b[49m\x1b[38;2;240;160;80m\x1b[1m{} \x1b[22m\x1b[38;2;120;120;120m{}{}\x1b[0m",
                cmd_padded,
                desc_truncated,
                " ".repeat(padding)
            )
        } else {
            format!(
                "\x1b[49m\x1b[38;2;120;120;120m{} \x1b[38;2;120;120;120m{}{}\x1b[0m",
                cmd_padded,
                desc_truncated,
                " ".repeat(padding)
            )
        };

        popup.push_str(&format!("\x1b[{row};1H{line}"));
    }

    popup
}

/// Render `text` as a user-input line — a dark background spanning the full
/// width — matching how submitted prompts appear in the main output window.
pub fn render_user_input_line(text: &str, width: usize) -> String {
    let clipped = clip_line(text, 0, width);
    let padding = width.saturating_sub(visible_line_width(&clipped));
    format!(
        "{USER_INPUT_BACKGROUND}{clipped}{}{ANSI_RESET}",
        " ".repeat(padding)
    )
}

pub fn terminal_width() -> usize {
    terminal_size()
        .map(|(Width(width), _)| usize::from(width))
        .filter(|width| *width > 0)
        .or_else(|| {
            std::env::var("COLUMNS")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|width| *width > 0)
        })
        .unwrap_or(80)
}

pub fn terminal_height() -> usize {
    terminal_size()
        .map(|(_, Height(height))| usize::from(height))
        .filter(|height| *height > 0)
        .or_else(|| {
            std::env::var("LINES")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|height| *height > 0)
        })
        .unwrap_or(24)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn format_status_duration_uses_the_shortest_form() {
        assert_eq!(format_status_duration(Duration::from_secs(0)), "0s");
        assert_eq!(format_status_duration(Duration::from_secs(59)), "59s");
        assert_eq!(format_status_duration(Duration::from_secs(65)), "1m5s");
        assert_eq!(format_status_duration(Duration::from_secs(3723)), "1h2m3s");
    }

    #[test]
    fn thinking_status_rolls_and_formats_elapsed() {
        let frame_zero = render_thinking_status(0, Duration::from_secs(61));
        let frame_one = render_thinking_status(1, Duration::from_secs(61));

        assert!(frame_zero.rendered.contains('T'));
        assert!(frame_zero.rendered.contains('g'));
        assert_ne!(frame_zero.rendered, frame_one.rendered);
        assert!(frame_zero.rendered.contains("(1m1s)"));
        assert!(frame_one.rendered.contains("(1m1s)"));
        assert_eq!(
            frame_zero.visible_width,
            THINKING_TEXT.chars().count() + " (1m1s)".chars().count()
        );
        for ch in THINKING_TEXT.chars() {
            assert!(frame_zero.rendered.contains(ch));
        }
    }

    #[test]
    fn working_status_rolls_and_formats_rate() {
        let frame_zero = render_working_status(0, 42.5, Duration::from_secs(65));
        let frame_one = render_working_status(1, 42.5, Duration::from_secs(65));

        assert!(frame_zero.rendered.contains("42.5 t/s"));
        assert!(frame_zero.rendered.contains("(1m5s)"));
        assert_eq!(
            frame_zero.visible_width,
            WORKING_TEXT.chars().count() + " @ 42.5 t/s (1m5s)".chars().count()
        );
        assert_ne!(frame_zero.rendered, frame_one.rendered);
    }

    #[test]
    fn prompt_prefix_uses_branch_name() {
        assert_eq!(prompt_prefix(Some("main")), "main> ");
        assert_eq!(prompt_prefix(None), "> ");
    }

    #[test]
    fn wrapped_input_lines_respect_prompt_width() {
        assert_eq!(
            wrapped_input_lines("abc", 8, "main> "),
            vec!["ab".to_string(), "c".to_string()]
        );
    }

    #[test]
    fn status_line_centers_pending_count() {
        let line = render_status_line(
            30,
            Some(&StatusFragment::plain("2.5t/s".to_string())),
            "gpt-4.1",
            3,
        );
        assert!(line.starts_with("2.5t/s"));
        assert!(line.contains("Pending: 3"));
        assert!(line.ends_with("gpt-4.1"));
    }

    #[test]
    fn available_output_rows_matches_current_layout_math() {
        // header=8 lines, prompt_frame=4, height=24
        // rows_above_prompt = 24-4 = 20, banner_rows = min(9, 20) = 9
        assert_eq!(available_output_rows(20, 9), 11);
    }

    #[test]
    fn transcript_input_highlight_fills_the_row() {
        let rendered =
            render_transcript_line(&TranscriptLine::UserInput("> Hello World!".to_string()), 20);

        assert_eq!(
            rendered,
            format!("{USER_INPUT_BACKGROUND}> Hello World!      {ANSI_RESET}")
        );
    }
}
