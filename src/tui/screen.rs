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
use crate::llm::StreamPromptProgress;
use crate::workspaces::WorkspacePlacement;
use std::cell::RefCell;
use std::time::Duration;
use terminal_size::{Height, Width, terminal_size};

struct TranscriptRenderCache {
    transcript_ptr: usize,
    transcript_len: usize,
    transcript_epoch: usize,
    width: usize,
    lines: Vec<ratatui::text::Line<'static>>,
    is_user_input: Vec<bool>,
    user_input_ranges: Vec<(usize, usize)>,
}

thread_local! {
    static TRANSCRIPT_RENDER_CACHE: RefCell<Option<TranscriptRenderCache>> = const { RefCell::new(None) };
}

#[derive(Clone, Copy)]
pub struct MainScreenLayout {
    pub horizontal_tab_area: Option<ratatui::layout::Rect>,
    pub above_prompt_content_area: ratatui::layout::Rect,
    pub vertical_tab_area: Option<ratatui::layout::Rect>,
    pub header_area: ratatui::layout::Rect,
    pub output_area: ratatui::layout::Rect,
    pub prompt_area: ratatui::layout::Rect,
    pub padded_prompt_area: ratatui::layout::Rect,
}

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

#[derive(Clone)]
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
    pub transcript_epoch: usize,
    pub scroll_offset: usize,
    pub left_status: Option<StatusFragment>,
    pub pending_count: usize,
    pub pending_lines: &'a [TranscriptLine],
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

fn build_transcript_render_cache(
    transcript: &[TranscriptLine],
    transcript_epoch: usize,
    width: usize,
    theme: &crate::tui::theme::Theme,
) -> TranscriptRenderCache {
    let mut lines = Vec::new();
    let mut is_user_input = Vec::new();
    let mut user_input_ranges = Vec::new();

    for line in transcript {
        let start_index = lines.len();
        let rendered = render_transcript_line_multi(line, width, theme);
        let user_input = matches!(line, TranscriptLine::UserInput(_));
        for rendered_line in rendered {
            lines.push(rendered_line);
            is_user_input.push(user_input);
        }
        if user_input {
            user_input_ranges.push((start_index, lines.len()));
        }
    }

    TranscriptRenderCache {
        transcript_ptr: transcript.as_ptr() as usize,
        transcript_len: transcript.len(),
        transcript_epoch,
        width,
        lines,
        is_user_input,
        user_input_ranges,
    }
}

fn with_transcript_render_cache<R>(
    transcript: &[TranscriptLine],
    transcript_epoch: usize,
    width: usize,
    theme: &crate::tui::theme::Theme,
    f: impl FnOnce(&TranscriptRenderCache) -> R,
) -> R {
    TRANSCRIPT_RENDER_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        let transcript_ptr = transcript.as_ptr() as usize;
        let cache_miss = cache.as_ref().is_none_or(|cached| {
            cached.transcript_ptr != transcript_ptr
                || cached.transcript_len != transcript.len()
                || cached.transcript_epoch != transcript_epoch
                || cached.width != width
        });
        if cache_miss {
            *cache = Some(build_transcript_render_cache(
                transcript,
                transcript_epoch,
                width,
                theme,
            ));
        }
        f(cache.as_ref().expect("transcript cache initialized"))
    })
}

/// Inputs for the bottom prompt frame (separator, input window, status bar),
/// shared by the normal screen, `/review` mode, `/auto_review`, and the
/// manual viewer.
pub struct PromptFrameArgs<'a> {
    pub header_height: usize,
    pub current_model: &'a str,
    pub left_status: Option<StatusFragment>,
    pub pending_count: usize,
    /// The knowledge graph's build status, shown as a `Graph: ●` indicator
    /// centered alongside `Pending: N`. `None` omits it entirely — today only
    /// `/auto_review` wires up a real value; the other screens pass `None`.
    pub graph_status: Option<ConnStatus>,
    pub prompt_prefix: &'a str,
    pub input: &'a str,
    pub cursor: usize,
    pub ghost: &'a str,
    pub height: usize,
    pub actual_width: usize,
    pub valid_command_len: usize,
}

fn tab_dot(status: TabStatus, theme: &crate::tui::Theme) -> ratatui::text::Span<'static> {
    match status {
        TabStatus::Valid => ratatui::text::Span::styled("●", theme.success),
        TabStatus::BranchGone => ratatui::text::Span::styled("●", theme.error),
        TabStatus::Working => ratatui::text::Span::styled(
            "●",
            ratatui::style::Style::default()
                .fg(ratatui::style::Color::Rgb(230, 230, 230))
                .add_modifier(ratatui::style::Modifier::SLOW_BLINK),
        ),
    }
}

/// The horizontal workspace tab bar (`1 │ 2 │ 3`, active tab bold) for top and
/// bottom placement, clipped to `width`. When `statuses` is non-empty a
/// colored dot precedes each tab number.
fn horizontal_tab_bar<'a>(
    view: WorkspaceTabsView,
    width: usize,
    statuses: &[TabStatus],
    theme: &'a crate::tui::Theme,
) -> ratatui::text::Line<'a> {
    let mut spans = Vec::new();
    for index in 0..view.count {
        if index > 0 {
            spans.push(ratatui::text::Span::styled(" │ ", theme.muted));
        }
        if let Some(&status) = statuses.get(index) {
            spans.push(tab_dot(status, theme));
        }
        let number = (index + 1).to_string();
        if index == view.active {
            spans.push(ratatui::text::Span::styled(
                number,
                ratatui::style::Style::default().add_modifier(ratatui::style::Modifier::BOLD),
            ));
        } else {
            spans.push(ratatui::text::Span::styled(number, theme.muted));
        }
    }
    let line = ratatui::text::Line::from(spans);
    crate::tui::text::clip_ratatui_line(&line, 0, width)
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
fn tab_gutter_cell<'a>(
    view: WorkspaceTabsView,
    row: usize,
    left: bool,
    statuses: &[TabStatus],
    theme: &'a crate::tui::Theme,
) -> ratatui::text::Line<'a> {
    let digits = view.count.to_string().len();
    let mut spans = Vec::new();

    if left {
        if !statuses.is_empty() {
            if let Some(&status) = statuses.get(row) {
                spans.push(tab_dot(status, theme));
            } else {
                spans.push(ratatui::text::Span::raw(" "));
            }
        }

        if row < view.count {
            let number = format!("{:>digits$}", row + 1);
            if row == view.active {
                spans.push(ratatui::text::Span::styled(
                    number,
                    ratatui::style::Style::default().add_modifier(ratatui::style::Modifier::BOLD),
                ));
            } else {
                spans.push(ratatui::text::Span::styled(number, theme.muted));
            }
        } else {
            spans.push(ratatui::text::Span::raw(" ".repeat(digits)));
        }

        spans.push(ratatui::text::Span::styled(" │ ", theme.muted));
    } else {
        spans.push(ratatui::text::Span::styled(" │ ", theme.muted));

        if !statuses.is_empty() {
            if let Some(&status) = statuses.get(row) {
                spans.push(tab_dot(status, theme));
            } else {
                spans.push(ratatui::text::Span::raw(" "));
            }
        }

        if row < view.count {
            let number = format!("{:>digits$}", row + 1);
            if row == view.active {
                spans.push(ratatui::text::Span::styled(
                    number,
                    ratatui::style::Style::default().add_modifier(ratatui::style::Modifier::BOLD),
                ));
            } else {
                spans.push(ratatui::text::Span::styled(number, theme.muted));
            }
        } else {
            spans.push(ratatui::text::Span::raw(" ".repeat(digits)));
        }
    }

    ratatui::text::Line::from(spans)
}

pub fn main_screen_layout(
    actual_width: usize,
    actual_height: usize,
    prompt_branch: Option<&str>,
    input: &str,
    tab_bar: Option<WorkspaceTabsView>,
    tab_statuses: &[TabStatus],
    has_output_content: bool,
) -> MainScreenLayout {
    let area = ratatui::layout::Rect {
        x: 0,
        y: 0,
        width: actual_width.max(1) as u16,
        height: actual_height.max(1) as u16,
    };
    let placement = tab_bar.map(|view| view.placement);

    let (horizontal_tab_area, main_area) = match placement {
        Some(WorkspacePlacement::Top) => {
            let chunks = ratatui::layout::Layout::default()
                .direction(ratatui::layout::Direction::Vertical)
                .constraints([
                    ratatui::layout::Constraint::Length(1),
                    ratatui::layout::Constraint::Min(0),
                ])
                .split(area);
            (Some(chunks[0]), chunks[1])
        }
        Some(WorkspacePlacement::Bottom) => {
            let chunks = ratatui::layout::Layout::default()
                .direction(ratatui::layout::Direction::Vertical)
                .constraints([
                    ratatui::layout::Constraint::Min(0),
                    ratatui::layout::Constraint::Length(1),
                ])
                .split(area);
            (Some(chunks[1]), chunks[0])
        }
        _ => (None, area),
    };

    let prompt_prefix = prompt_prefix(prompt_branch);
    let input_lines = wrapped_input_lines(input, actual_width.max(1), &prompt_prefix);
    let prompt_frame_height = (input_lines.len() + 3) as u16;
    let main_chunks = ratatui::layout::Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([
            ratatui::layout::Constraint::Min(0),
            ratatui::layout::Constraint::Length(1),
            ratatui::layout::Constraint::Length(prompt_frame_height),
        ])
        .split(main_area);
    let above_prompt_area = main_chunks[0];
    let prompt_area = main_chunks[2];

    let above_prompt_chunks = ratatui::layout::Layout::default()
        .direction(ratatui::layout::Direction::Vertical)
        .constraints([
            ratatui::layout::Constraint::Length(1),
            ratatui::layout::Constraint::Min(0),
        ])
        .split(above_prompt_area);
    let above_prompt_content_area = above_prompt_chunks[1];

    let has_dots = !tab_statuses.is_empty();
    let (vertical_tab_area, central_area) = match placement {
        Some(WorkspacePlacement::Left) => {
            let w = tab_gutter_width(tab_bar.expect("left placement has tab bar"), has_dots) as u16;
            let chunks = ratatui::layout::Layout::default()
                .direction(ratatui::layout::Direction::Horizontal)
                .constraints([
                    ratatui::layout::Constraint::Length(w),
                    ratatui::layout::Constraint::Min(0),
                ])
                .split(above_prompt_content_area);
            (Some(chunks[0]), chunks[1])
        }
        Some(WorkspacePlacement::Right) => {
            let w =
                tab_gutter_width(tab_bar.expect("right placement has tab bar"), has_dots) as u16;
            let chunks = ratatui::layout::Layout::default()
                .direction(ratatui::layout::Direction::Horizontal)
                .constraints([
                    ratatui::layout::Constraint::Min(0),
                    ratatui::layout::Constraint::Length(w),
                ])
                .split(above_prompt_content_area);
            (Some(chunks[1]), chunks[0])
        }
        _ => (None, above_prompt_content_area),
    };

    let is_landing_mode = !has_output_content && input.is_empty();
    let (header_area, output_area) = if is_landing_mode {
        let v_chunks = ratatui::layout::Layout::default()
            .direction(ratatui::layout::Direction::Vertical)
            .constraints([
                ratatui::layout::Constraint::Min(0),
                ratatui::layout::Constraint::Length(9),
                ratatui::layout::Constraint::Min(0),
            ])
            .split(central_area);
        let area_width = v_chunks[1].width;
        let left_pad = area_width.saturating_sub(85) / 2;
        let h_chunks = ratatui::layout::Layout::default()
            .direction(ratatui::layout::Direction::Horizontal)
            .constraints([
                ratatui::layout::Constraint::Length(left_pad),
                ratatui::layout::Constraint::Length(85),
                ratatui::layout::Constraint::Min(0),
            ])
            .split(v_chunks[1]);
        (h_chunks[1], ratatui::layout::Rect::default())
    } else {
        let margin = ratatui::layout::Margin {
            horizontal: 2,
            vertical: 0,
        };
        let output_area = if central_area.width > 4 {
            central_area.inner(margin)
        } else {
            central_area
        };
        (ratatui::layout::Rect::default(), output_area)
    };

    let padded_prompt_area = ratatui::layout::Rect {
        x: prompt_area.x + 2,
        y: prompt_area.y,
        width: prompt_area.width.saturating_sub(4),
        height: prompt_area.height,
    };

    MainScreenLayout {
        horizontal_tab_area,
        above_prompt_content_area,
        vertical_tab_area,
        header_area,
        output_area,
        prompt_area,
        padded_prompt_area,
    }
}

pub fn draw_screen(frame: &mut ratatui::Frame, args: ScreenRenderArgs<'_>) {
    let theme = crate::tui::theme::Theme::current();

    let area = frame.area();
    frame.render_widget(
        ratatui::widgets::Block::default().style(
            ratatui::style::Style::default()
                .bg(theme.bg_base)
                .fg(theme.text_primary),
        ),
        area,
    );
    let actual_width = args.actual_width.max(1);
    let placement = args.tab_bar.map(|view| view.placement);
    let prompt_prefix = prompt_prefix(args.prompt_branch);
    let layout = main_screen_layout(
        actual_width,
        args.actual_height,
        args.prompt_branch,
        args.input,
        args.tab_bar,
        args.tab_statuses,
        !args.transcript.is_empty() || !args.pending_lines.is_empty(),
    );

    // Render Horizontal Tab Bar
    if let Some(tab_area) = layout.horizontal_tab_area {
        let view = args.tab_bar.unwrap();
        let bar = horizontal_tab_bar(view, actual_width, args.tab_statuses, &theme);
        frame.render_widget(ratatui::widgets::Paragraph::new(bar), tab_area);
    }

    // Render Vertical Tab Bar
    if let Some(tab_area) = layout.vertical_tab_area {
        let view = args.tab_bar.unwrap();
        let left = placement == Some(WorkspacePlacement::Left);
        let mut gutter_lines = Vec::new();
        for row in 0..layout.above_prompt_content_area.height as usize {
            let cell = tab_gutter_cell(view, row, left, args.tab_statuses, &theme);
            gutter_lines.push(cell);
        }
        frame.render_widget(ratatui::widgets::Paragraph::new(gutter_lines), tab_area);
    }

    if layout.header_area.height > 0 {
        let header_widget = crate::tui::widgets::HeaderWidget {
            version: args.version,
            current_model: args.current_model,
            endpoint: args.endpoint,
            workspace: args.workspace,
            status: args.status,
            alignment: args.banner,
        };
        frame.render_widget(header_widget, layout.header_area);
    }

    // Render Output
    let inner_width = layout.output_area.width as usize;
    let available_output_rows = layout.output_area.height as usize;
    let pending_lines = args
        .pending_lines
        .iter()
        .flat_map(|line| render_transcript_line_multi(line, inner_width, &theme))
        .map(|line| crate::tui::text::clip_ratatui_line(&line, args.x_offset, inner_width))
        .collect::<Vec<_>>();

    let (scroll_offset, mut visible_lines) = with_transcript_render_cache(
        args.transcript,
        args.transcript_epoch,
        inner_width,
        &theme,
        |cache| {
            let total_lines = cache.lines.len() + pending_lines.len();
            let has_status_line = args.left_status.is_some() || args.pending_count > 0;
            let max_scroll_offset = if has_status_line {
                total_lines.saturating_sub(available_output_rows.saturating_sub(1))
            } else {
                total_lines.saturating_sub(available_output_rows)
            };
            let scroll_offset = args.scroll_offset.min(max_scroll_offset);

            let text_rows = if has_status_line && scroll_offset == 0 {
                available_output_rows.saturating_sub(1)
            } else {
                available_output_rows
            };

            let visible_end = total_lines.saturating_sub(scroll_offset);
            let visible_start = visible_end.saturating_sub(text_rows);
            let mut visible_lines = Vec::with_capacity(visible_end.saturating_sub(visible_start));

            for global_idx in visible_start..visible_end {
                if global_idx < cache.lines.len() {
                    let line = &cache.lines[global_idx];
                    let offset = if cache.is_user_input[global_idx] {
                        0
                    } else {
                        args.x_offset
                    };
                    visible_lines.push(crate::tui::text::clip_ratatui_line(
                        line,
                        offset,
                        inner_width,
                    ));
                } else {
                    visible_lines.push(pending_lines[global_idx - cache.lines.len()].clone());
                }
            }

            if let Some(sticky_idx) = cache
                .user_input_ranges
                .iter()
                .rposition(|(start, _)| *start < visible_start)
            {
                let (sticky_start, sticky_end) = cache.user_input_ranges[sticky_idx];
                let next_start = cache
                    .user_input_ranges
                    .get(sticky_idx + 1)
                    .map(|(start, _)| *start)
                    .unwrap_or(usize::MAX);
                let sticky_len = sticky_end.saturating_sub(sticky_start);
                let space_available = next_start.saturating_sub(visible_start);
                let draw_count = sticky_len.min(space_available).min(visible_lines.len());
                for (draw_idx, line) in visible_lines.iter_mut().take(draw_count).enumerate() {
                    *line = cache.lines[sticky_end - draw_count + draw_idx].clone();
                }
            }

            (scroll_offset, visible_lines)
        },
    );

    // Append status line directly at the end of the transcript view if we are scrolled to bottom
    if scroll_offset == 0 {
        let mut left_spans = Vec::new();
        let mut left_used = 0;
        if let Some(left) = &args.left_status {
            if let Ok(mut text) = ansi_to_tui::IntoText::into_text(&left.rendered)
                && let Some(line) = text.lines.pop()
            {
                left_spans.extend(line.spans);
            }
            left_used += left.visible_width;
        } else if args.pending_count > 0 {
            let pending = format!("Pending: {}", args.pending_count);
            left_spans.push(ratatui::text::Span::styled(
                pending,
                ratatui::style::Style::default().fg(ratatui::style::Color::Rgb(220, 220, 100)),
            ));
            left_used += format!("Pending: {}", args.pending_count).chars().count();
        }

        if left_used > 0 {
            visible_lines.push(ratatui::text::Line::from(left_spans));
        }
    }

    frame.render_widget(
        ratatui::widgets::Paragraph::new(visible_lines),
        layout.output_area,
    );

    // Render Prompt Frame
    let current_model =
        crate::tui::header::display_model_name(args.status.is_coordinator, args.current_model);
    let prompt_widget = crate::tui::widgets::PromptFrameWidget {
        current_model,
        prompt_prefix: &prompt_prefix,
        input: args.input,
        cursor: args.cursor,
        ghost: args.ghost,
        valid_command_len: args.valid_command_len,
        left_status: args.left_status.as_ref(),
        pending_count: args.pending_count,
        graph_status: Some(args.status.server_ok),
        prompt_branch: args.prompt_branch,
    };
    frame.render_widget(prompt_widget, layout.padded_prompt_area);

    // Render Dropdown natively so it floats correctly
    if let Some(candidates) = args.dropdown_candidates
        && !candidates.is_empty()
    {
        let pf_top_row = layout.padded_prompt_area.y;
        let (dropdown_rect, dropdown_lines) = render_dropdown_popup(
            candidates,
            args.dropdown_selected,
            pf_top_row as usize,
            layout.padded_prompt_area.x as usize,
            actual_width,
        );
        frame.render_widget(ratatui::widgets::Clear, dropdown_rect);
        frame.render_widget(
            ratatui::widgets::Paragraph::new(dropdown_lines),
            dropdown_rect,
        );
    }

    // Set cursor position using Ratatui
    let (cr_offset, cc_offset) = cursor_position(
        args.input,
        args.cursor,
        layout.padded_prompt_area.width.saturating_sub(4) as usize,
        &prompt_prefix,
    );
    let pf_input_start = layout.padded_prompt_area.y + 1;
    let pf_prompt_width = prompt_prefix.chars().count();
    let cr = pf_input_start as usize + cr_offset;
    let cc = (1 + pf_prompt_width + cc_offset).max(1);
    frame.set_cursor_position((layout.padded_prompt_area.x + cc as u16, cr as u16));
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
    let _ = (version, current_model, endpoint, workspace, status);
    main_screen_layout(
        actual_width,
        actual_height,
        prompt_branch,
        input,
        None,
        &[],
        true,
    )
    .output_area
    .height as usize
}

#[allow(clippy::too_many_arguments)]
pub fn output_view_width(
    version: &str,
    current_model: &str,
    endpoint: &str,
    workspace: &std::path::Path,
    prompt_branch: Option<&str>,
    status: HeaderStatus,
    input: &str,
    actual_width: usize,
    actual_height: usize,
    tab_bar: Option<WorkspaceTabsView>,
    tab_statuses: &[TabStatus],
    has_output_content: bool,
) -> usize {
    let _ = (version, current_model, endpoint, workspace, status);
    main_screen_layout(
        actual_width,
        actual_height,
        prompt_branch,
        input,
        tab_bar,
        tab_statuses,
        has_output_content,
    )
    .output_area
    .width as usize
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

/// Like [`render_thinking_status`] but for the prefill phase of an orangu-server
/// request, when the server has reported [`StreamPromptProgress`] — shows how
/// much of the prompt is being served from its KV cache rather than
/// reprocessed, e.g. `Thinking (41% cached, 620/1500 tok) (2s)`. This is the
/// user-visible signal that orangu's append-only prompt discipline (see
/// `doc/manual/en/70-dev.md`) is actually paying off.
pub fn render_prefill_status(
    frame: usize,
    progress: StreamPromptProgress,
    elapsed: Duration,
) -> StatusFragment {
    let total = progress.total.max(0);
    let percent_cached = if total > 0 {
        progress.cache.max(0) * 100 / total
    } else {
        0
    };
    let suffix = format!(
        " ({percent_cached}% cached, {}/{total} tok) {}",
        progress.processed.max(0),
        format_elapsed_timer(elapsed)
    );
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
        args.graph_status,
    );
    frame.push_str(&format!(
        "\x1b[{bottom_row};1H{separator}\x1b[{model_row};1H{status_line}\x1b[{cursor_row};{display_cursor_col}H"
    ));
    frame
}

pub fn render_transcript_line_multi(
    line: &TranscriptLine,
    width: usize,
    theme: &crate::tui::theme::Theme,
) -> Vec<ratatui::text::Line<'static>> {
    use ansi_to_tui::IntoText;
    use ratatui::style::{Color, Style};
    use ratatui::text::{Line, Span};

    match line {
        TranscriptLine::Plain(content) | TranscriptLine::Wide(content) => {
            if let Ok(text) = content.into_text() {
                text.lines
            } else {
                vec![Line::from(content.clone())]
            }
        }
        TranscriptLine::CodeBlock {
            language,
            ansi_lines,
        } => {
            let mut lines = Vec::new();

            // First, find the maximum line width in the code block
            let mut max_line_width = 0;
            let mut parsed_lines = Vec::new();
            for ansi_line in ansi_lines {
                let mut line_text = ansi_line
                    .into_text()
                    .map(|text| text.lines.into_iter().next().unwrap_or_default())
                    .unwrap_or_default();

                // Force background color on all code spans
                for span in &mut line_text.spans {
                    span.style = span.style.patch(theme.code_block_bg);
                }

                let line_width = line_text.width();
                if line_width > max_line_width {
                    max_line_width = line_width;
                }
                parsed_lines.push((line_text, line_width));
            }

            let block_width = width.max(max_line_width + 3);

            let mut top_border_spans = vec![Span::styled("╭─", theme.code_block_bg)];
            let lang_label = language.as_deref().unwrap_or("").trim();
            let label_len = if lang_label.is_empty() {
                0
            } else {
                lang_label.chars().count() + 2
            };

            if !lang_label.is_empty() {
                top_border_spans.push(Span::styled(
                    format!(" {} ", lang_label),
                    theme
                        .code_block_bg
                        .add_modifier(ratatui::style::Modifier::BOLD),
                ));
            }

            let dash_count = block_width.saturating_sub(2 + label_len + 1);
            top_border_spans.push(Span::styled("─".repeat(dash_count), theme.code_block_bg));
            top_border_spans.push(Span::styled("╮", theme.code_block_bg));

            lines.push(Line::from(top_border_spans));

            for (line_text, visible_width) in parsed_lines {
                let mut spans = vec![Span::styled("│ ", theme.code_block_bg)];

                spans.extend(line_text.spans);

                let padding = block_width.saturating_sub(visible_width + 3);
                spans.push(Span::styled(" ".repeat(padding), theme.code_block_bg));
                spans.push(Span::styled("│", theme.code_block_bg));

                lines.push(Line::from(spans));
            }

            let bottom_border_spans = vec![
                Span::styled("╰", theme.code_block_bg),
                Span::styled(
                    "─".repeat(block_width.saturating_sub(2)),
                    theme.code_block_bg,
                ),
                Span::styled("╯", theme.code_block_bg),
            ];
            lines.push(Line::from(bottom_border_spans));

            lines
        }
        TranscriptLine::Collapsible {
            title,
            content,
            expanded,
            ..
        } => {
            let mut lines = Vec::new();
            let marker = if *expanded { "▼" } else { "◈" };
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled(marker.to_string(), Style::default().fg(Color::Indexed(244))),
                Span::styled(
                    format!(" {}", title),
                    Style::default().fg(Color::Indexed(244)),
                ),
            ]));
            if *expanded {
                for line in content.lines() {
                    lines.push(Line::from(vec![
                        Span::raw("    "),
                        Span::styled("│", Style::default().fg(Color::Indexed(240))),
                        Span::raw(" "),
                        Span::styled(line.to_string(), Style::default().fg(Color::Indexed(244))),
                    ]));
                }
            }
            lines
        }
        TranscriptLine::ToolCall { name, arguments } => {
            let mut lines = Vec::new();
            lines.push(Line::from(vec![
                Span::raw("  "),
                Span::styled("🛠️  ", Style::default()),
                Span::styled(
                    format!("Tool: {} ", name),
                    Style::default()
                        .fg(Color::Indexed(214))
                        .add_modifier(ratatui::style::Modifier::BOLD),
                ),
                Span::styled(arguments.clone(), Style::default().fg(Color::Indexed(244))),
            ]));
            lines
        }
        TranscriptLine::UserInput(content) => {
            let mut lines = Vec::new();
            let content = content.trim_start_matches("> ");

            let wrapped = wrapped_input_lines(&format!("  ❯ {}", content), width, "    ");
            let bg = theme.user_input;

            // Gap above
            lines.push(Line::default());

            // Inside top padding
            lines.push(Line::from(Span::styled(
                " ".repeat(width.saturating_sub(1)),
                bg,
            )));

            for l in wrapped {
                let padding = width.saturating_sub(1).saturating_sub(l.chars().count());
                lines.push(Line::from(vec![
                    Span::styled(l, bg),
                    Span::styled(" ".repeat(padding), bg),
                ]));
            }

            // Inside bottom padding
            lines.push(Line::from(Span::styled(
                " ".repeat(width.saturating_sub(1)),
                bg,
            )));

            // Gap below
            lines.push(Line::default());

            lines
        }
    }
}

pub fn wrapped_input_lines(input: &str, width: usize, prompt_prefix: &str) -> Vec<String> {
    let input_width = width.saturating_sub(prompt_prefix.chars().count()).max(1);
    if input.is_empty() {
        return vec![String::new()];
    }

    let mut lines = Vec::new();

    for paragraph in input.split('\n') {
        let mut current = String::new();
        for ch in paragraph.chars() {
            current.push(ch);
            if current.chars().count() == input_width {
                lines.push(current);
                current = String::new();
            }
        }
        if !current.is_empty() || paragraph.is_empty() {
            lines.push(current);
        }
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

/// The input-window prompt prefix: always `❯ ` (branch name is shown in top bar).
pub fn prompt_prefix(_prompt_branch: Option<&str>) -> String {
    "❯ ".to_string()
}

/// Center `text` (which may carry ANSI color codes — its width is measured
/// with `visible_line_width`, not `text.len()`) within `width` visible
/// columns, padding with plain spaces on both sides. Returns `text` unchanged
/// if it doesn't fit — callers gate on `visible_line_width(text) <= width`
/// before calling this, so that's only ever reached defensively.
fn centered(text: &str, width: usize) -> String {
    let visible = visible_line_width(text);
    if visible >= width {
        return text.to_string();
    }
    let total_pad = width - visible;
    let left_pad = total_pad / 2;
    let right_pad = total_pad - left_pad;
    format!("{}{text}{}", " ".repeat(left_pad), " ".repeat(right_pad))
}

fn render_status_line(
    width: usize,
    left_status: Option<&StatusFragment>,
    current_model: &str,
    pending_count: usize,
    graph_status: Option<ConnStatus>,
) -> String {
    // Priority: left_status (Working/Thinking) > model name > Graph/Pending.
    let left_visible_width = left_status.map_or(0, |s| s.visible_width);
    let right_space = width.saturating_sub(left_visible_width);

    let model_width = current_model.chars().count();
    let show_model = right_space >= model_width;
    // Gap is the space between the left status and the model name (or right edge if no model).
    let gap = if show_model {
        right_space.saturating_sub(model_width)
    } else {
        right_space
    };

    // The centered content: `Graph: ●` and `Pending: N`, joined when both are
    // present and both fit; if the pair doesn't fit but the Graph indicator
    // alone would, it's shown alone rather than dropped for a Pending count
    // that only matters while a command queue is building up.
    let graph_text = graph_status.map(|status| format!("Graph: {}", indicator(status)));
    let pending_text = (pending_count > 0).then(|| format!("Pending: {pending_count}"));
    let combined = match (&graph_text, &pending_text) {
        (Some(graph), Some(pending)) => Some(format!("{graph}   {pending}")),
        (Some(graph), None) => Some(graph.clone()),
        (None, Some(pending)) => Some(pending.clone()),
        (None, None) => None,
    };
    let middle = combined
        .filter(|text| show_model && visible_line_width(text) <= gap)
        .or_else(|| graph_text.filter(|text| show_model && visible_line_width(text) <= gap));

    let gap_content = match &middle {
        Some(text) => centered(text, gap),
        None => " ".repeat(gap),
    };
    let right = if show_model {
        format!("{gap_content}{current_model}")
    } else {
        gap_content
    };

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

pub(crate) fn truncate_to_width(input: &str, width: usize) -> String {
    input.chars().take(width).collect()
}

fn render_dropdown_popup(
    candidates: &[(String, String)],
    selected: usize,
    bottom_row: usize,
    x_offset: usize,
    actual_width: usize,
) -> (ratatui::layout::Rect, Vec<ratatui::text::Line<'static>>) {
    use ratatui::style::{Color, Modifier, Style};
    use ratatui::text::{Line, Span};

    let max_height = 10.min(candidates.len());
    let start_idx = selected
        .saturating_sub(max_height / 2)
        .min(candidates.len().saturating_sub(max_height));
    let display_candidates = &candidates[start_idx..start_idx + max_height];

    let top_row = bottom_row.saturating_sub(display_candidates.len());
    let max_width = actual_width.saturating_sub(x_offset).min(100);

    let mut lines = Vec::new();

    for (i, (cmd, desc)) in display_candidates.iter().enumerate() {
        let is_selected = (start_idx + i) == selected;

        let mut cmd_padded = cmd.clone();
        if cmd_padded.chars().count() < 32 {
            cmd_padded.push_str(&" ".repeat(32 - cmd_padded.chars().count()));
        }

        // Subtract extra for the internal padding (1 space left, 1 space between, 1 space right)
        let desc_truncated = truncate_to_width(desc, max_width.saturating_sub(36));
        let visible_len = cmd_padded.chars().count() + 1 + desc_truncated.chars().count();
        let padding = max_width.saturating_sub(visible_len).saturating_sub(2);

        let cmd_style = if is_selected {
            Style::default()
                .fg(Color::Rgb(240, 160, 80))
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::Rgb(120, 120, 120))
        };

        let desc_style = Style::default().fg(Color::Rgb(120, 120, 120));
        let bg_style = if is_selected {
            Style::default().bg(Color::Rgb(40, 40, 40))
        } else {
            Style::default()
        };

        let line = Line::from(vec![
            Span::styled(format!(" {} ", cmd_padded), cmd_style),
            Span::styled(
                format!("{}{}", desc_truncated, " ".repeat(padding + 1)),
                desc_style,
            ),
        ])
        .style(bg_style);

        lines.push(line);
    }

    let rect = ratatui::layout::Rect {
        x: x_offset as u16,
        y: top_row as u16,
        width: max_width as u16,
        height: display_candidates.len() as u16,
    };

    (rect, lines)
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
    fn prefill_status_shows_cache_percent_and_token_counts() {
        let progress = StreamPromptProgress {
            total: 1500,
            cache: 615,
            processed: 620,
            time_ms: 2_000,
        };
        let frame_zero = render_prefill_status(0, progress, Duration::from_secs(2));
        let frame_one = render_prefill_status(1, progress, Duration::from_secs(2));

        // 615 * 100 / 1500 = 41 (integer division).
        assert!(frame_zero.rendered.contains("41% cached"));
        assert!(frame_zero.rendered.contains("620/1500 tok"));
        assert!(frame_zero.rendered.contains("(2s)"));
        assert_ne!(frame_zero.rendered, frame_one.rendered);
        assert_eq!(
            frame_zero.visible_width,
            THINKING_TEXT.chars().count() + " (41% cached, 620/1500 tok) (2s)".chars().count()
        );
    }

    #[test]
    fn prefill_status_handles_zero_total_without_dividing_by_zero() {
        let progress = StreamPromptProgress {
            total: 0,
            cache: 0,
            processed: 0,
            time_ms: 0,
        };
        let status = render_prefill_status(0, progress, Duration::from_secs(1));
        assert!(status.rendered.contains("0% cached, 0/0 tok"));
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
    fn prompt_prefix_omits_branch_name() {
        assert_eq!(prompt_prefix(Some("main")), "❯ ");
        assert_eq!(prompt_prefix(None), "❯ ");
    }

    #[test]
    fn wrapped_input_lines_respect_prompt_width() {
        assert_eq!(
            wrapped_input_lines("abc", 4, "> "),
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
            None,
        );
        assert!(line.starts_with("2.5t/s"));
        assert!(line.contains("Pending: 3"));
        assert!(line.ends_with("gpt-4.1"));
    }

    #[test]
    fn status_line_shows_graph_dot_alongside_pending() {
        let line = render_status_line(40, None, "gpt-4.1", 3, Some(ConnStatus::Ok));
        assert!(line.contains("Graph: "));
        assert!(line.contains(&indicator(ConnStatus::Ok)));
        assert!(line.contains("Pending: 3"));
        assert!(line.ends_with("gpt-4.1"));
        // Graph precedes Pending in the combined centered text.
        assert!(line.find("Graph:").unwrap() < line.find("Pending:").unwrap());
    }

    #[test]
    fn status_line_shows_graph_dot_alone_without_pending() {
        let line = render_status_line(30, None, "gpt-4.1", 0, Some(ConnStatus::Failed));
        assert!(line.contains("Graph: "));
        assert!(line.contains(&indicator(ConnStatus::Failed)));
        assert!(!line.contains("Pending:"));
    }

    #[test]
    fn status_line_omits_graph_dot_when_not_given() {
        let line = render_status_line(30, None, "gpt-4.1", 3, None);
        assert!(!line.contains("Graph:"));
        assert!(line.contains("Pending: 3"));
    }

    #[test]
    fn output_view_rows_matches_current_layout_math() {
        let rows = output_view_rows(
            "0.0.0",
            "model",
            "endpoint",
            std::path::Path::new("."),
            Some("main"),
            HeaderStatus {
                workspace_ok: true,
                server_ok: ConnStatus::Ok,
                model_ok: ConnStatus::Ok,
                is_coordinator: false,
            },
            "abc",
            80,
            24,
        );

        assert_eq!(rows, 18);
    }
}
