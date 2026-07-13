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

use crate::tui::Theme;
use crate::tui::{
    auto_review::{
        AutoReviewDiffView, AutoReviewFileMode, AutoReviewRejectView, AutoReviewScreenArgs,
    },
    review::{ReviewStatus, review_right_width},
    text::{clip_line, clip_ratatui_line},
    widgets::PromptFrameWidget,
};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Clear, Paragraph},
};

fn clip_plain(line: &str, width: usize) -> String {
    clip_line(line, 0, width)
}

fn first_ansi_line(line: &str) -> Line<'static> {
    use ansi_to_tui::IntoText;

    line.into_text()
        .ok()
        .and_then(|text| text.lines.into_iter().next())
        .unwrap_or_else(|| Line::from(line.to_string()))
}

fn review_status_span(status: ReviewStatus, theme: &Theme) -> Span<'static> {
    match status {
        ReviewStatus::Unreviewed => Span::raw(" "),
        ReviewStatus::Approved => Span::styled("●", theme.success),
        ReviewStatus::Rejected => Span::styled("●", theme.error),
    }
}

pub fn draw_auto_review_screen(f: &mut Frame, args: AutoReviewScreenArgs<'_>) {
    let theme = Theme::current();
    let width = args.actual_width as u16;
    let height = args.actual_height as u16;
    let area = f.area();
    f.render_widget(
        ratatui::widgets::Block::default().style(
            ratatui::style::Style::default()
                .bg(theme.bg_base)
                .fg(theme.text_primary),
        ),
        area,
    );

    let prompt_prefix = crate::tui::prompt_prefix(args.prompt_branch);
    let input_lines = crate::tui::wrapped_input_lines(args.input, width as usize, &prompt_prefix);
    let prompt_frame_height = (input_lines.len() + 3) as u16;
    let pane_rows = height.saturating_sub(prompt_frame_height).max(2);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(pane_rows),
            Constraint::Length(prompt_frame_height),
        ])
        .split(Rect::new(0, 0, width, height));

    if let Some(diff) = &args.diff {
        draw_auto_review_diff_panel(f, diff, chunks[0], &theme);
    } else if let Some(reject) = &args.reject {
        draw_auto_review_reject_panel(f, reject, chunks[0], &theme);
    } else {
        draw_auto_review_panes(f, &args, chunks[0], &theme);
    }

    f.render_widget(
        PromptFrameWidget {
            current_model: args.current_model,
            prompt_prefix: &prompt_prefix,
            input: args.input,
            cursor: args.cursor,
            ghost: args.ghost,
            valid_command_len: 0,
            left_status: args.left_status.as_ref(),
            pending_count: args.pending_count,
            graph_status: args.graph_status,
            prompt_branch: args.prompt_branch,
        },
        chunks[1],
    );
}

fn draw_auto_review_panes(
    f: &mut Frame,
    args: &AutoReviewScreenArgs<'_>,
    area: Rect,
    theme: &Theme,
) {
    let right_width = review_right_width(args.files, area.width as usize) as u16;
    let left_width = area.width.saturating_sub(right_width).max(1);

    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(0)])
        .split(area);

    let header_area = main_layout[0];
    let panes_area = main_layout[1];

    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(left_width),
            Constraint::Length(right_width),
        ])
        .split(panes_area);

    let left_area = chunks[0];
    let right_area = chunks[1];
    let left_inner_area = left_area.inner(ratatui::layout::Margin {
        vertical: 1,
        horizontal: 1,
    });

    let body_height = left_inner_area.height as usize;
    let right_body_height = right_area.height.saturating_sub(1) as usize;
    let list_start = args.list_offset;

    let badges = if args.prestart {
        vec![
            ("Alt+s", "Start"),
            ("Alt+j/k", "Switch file"),
            ("Alt+m", "Mode"),
            ("Alt+e", "Diff"),
            ("Esc Esc", "Cancel"),
            ("Alt+x", "Exit"),
        ]
    } else if args.browsing {
        vec![
            ("Alt+j/k", "Switch file"),
            ("Alt+a", "Approve"),
            ("Alt+r", "Reject"),
            ("Alt+e", "Open"),
            ("↑/↓", "Item"),
            ("Enter", "Diff"),
            ("PgUp/PgDn", "Category"),
            ("-", "Remove"),
            ("Alt+x", "Exit"),
        ]
    } else {
        vec![("Esc Esc", "Cancel"), ("Alt+x", "Exit")]
    };

    let mut header_spans = vec![Span::styled(
        "Auto review  ",
        Style::default().fg(theme.text_primary),
    )];
    let badge_style = theme.cursor_line_bg;
    for (i, (key, desc)) in badges.iter().enumerate() {
        header_spans.push(Span::styled(format!(" {key} "), badge_style));
        header_spans.push(Span::styled(
            format!(" {desc}"),
            Style::default().fg(theme.text_primary),
        ));
        if i < badges.len() - 1 {
            header_spans.push(Span::raw("  "));
        }
    }

    f.render_widget(
        Paragraph::new(Line::from(header_spans)).block(
            ratatui::widgets::Block::default()
                .borders(ratatui::widgets::Borders::BOTTOM)
                .border_style(theme.muted),
        ),
        header_area,
    );

    let mut left_lines = Vec::new();

    // Status area
    if body_height > 0 {
        let status = first_ansi_line(&clip_plain(args.status, left_inner_area.width as usize))
            .style(theme.cursor_line_bg);
        left_lines.push(status);
    }

    // Report lines
    for row in 0..body_height.saturating_sub(1) {
        let line_index = args.scroll + row;
        match args.report_lines.get(line_index) {
            Some(line) => {
                let cell = clip_line(line, args.x_offset, left_inner_area.width as usize);
                // Highlight if selected
                let is_selected = match args.selected_lines {
                    Some((start, end)) => line_index >= start && line_index < end,
                    _ => false,
                };
                let mut rendered = first_ansi_line(&cell);
                if is_selected {
                    rendered = rendered.style(theme.cursor_line_bg);
                }
                left_lines.push(rendered);
            }
            None => left_lines.push(Line::from("")),
        }
    }

    f.render_widget(
        Paragraph::new(left_lines)
            .block(ratatui::widgets::Block::bordered().border_style(theme.muted)),
        left_area,
    );

    let right_header = format!("Files ({})", args.files.len());
    let mut right_lines = vec![Line::from(Span::styled(
        clip_plain(&right_header, right_area.width as usize),
        Style::default().fg(theme.text_primary),
    ))];

    for row in 0..right_body_height {
        let file_index = list_start + row;
        match args.files.get(file_index) {
            Some(file) => {
                let mode = args.modes.get(file_index).copied().unwrap_or_default();
                let status_dot = if args.prestart && mode == AutoReviewFileMode::Ignore {
                    Span::styled("●", theme.ignore)
                } else if args.prestart && mode == AutoReviewFileMode::Deep {
                    Span::styled("●", theme.deep)
                } else if (args.prestart && mode == AutoReviewFileMode::Normal)
                    || args.reviewing == Some(file_index)
                {
                    Span::styled("●", Style::default().fg(theme.text_primary))
                } else {
                    review_status_span(file.status, theme)
                };

                let spans = vec![
                    Span::styled("[", Style::default().fg(theme.text_primary)),
                    status_dot,
                    Span::styled("] ", Style::default().fg(theme.text_primary)),
                    Span::styled(file.path.clone(), Style::default().fg(theme.text_primary)),
                ];

                let line = Line::from(spans);
                let clipped = clip_ratatui_line(&line, 0, right_area.width as usize);

                if args.selected == Some(file_index) {
                    right_lines.push(clipped.style(theme.selected_file));
                } else {
                    right_lines.push(clipped);
                }
            }
            None => right_lines.push(Line::from("")),
        }
    }

    f.render_widget(
        Paragraph::new(right_lines).block(
            ratatui::widgets::Block::default().padding(ratatui::widgets::Padding::new(1, 1, 1, 0)),
        ),
        right_area,
    );
}

fn push_reject_section_label(
    lines: &mut Vec<Line>,
    label: &str,
    focused: bool,
    _width: usize,
    theme: &Theme,
) {
    if focused {
        lines.push(Line::from(Span::styled(
            label.to_string(),
            theme.cursor_line_bg,
        )));
    } else {
        lines.push(Line::from(label.to_string()));
    }
    lines.push(Line::from(Span::styled(
        "─".repeat(label.chars().count()),
        theme.muted,
    )));
}

fn draw_auto_review_reject_panel(
    f: &mut Frame,
    reject: &AutoReviewRejectView<'_>,
    area: Rect,
    theme: &Theme,
) {
    f.render_widget(Clear, area);
    let width = area.width as usize;
    let mut lines = Vec::new();

    let header = format!(
        "Reject: {}  (Tab Switch focus · ↑/↓ Category · Alt+Enter Save · Esc Cancel)",
        reject.path
    );
    lines.push(Line::from(Span::styled(
        clip_plain(&header, width),
        theme.cursor_line_bg,
    )));
    lines.push(Line::from(""));

    push_reject_section_label(
        &mut lines,
        "Category:",
        reject.selector_focused,
        width,
        theme,
    );
    for (index, name) in reject.categories.iter().enumerate() {
        let chosen = index == reject.category;
        let marker = if chosen { "[●]" } else { "[ ]" };
        let text = format!("{marker} {name}");
        if chosen && reject.selector_focused {
            lines.push(Line::from(Span::styled(text, theme.cursor_line_bg)));
        } else {
            lines.push(Line::from(text));
        }
    }

    lines.push(Line::from(""));
    push_reject_section_label(
        &mut lines,
        "Comment:",
        !reject.selector_focused,
        width,
        theme,
    );

    let editor_rows = (area.height as usize).saturating_sub(lines.len()).max(1);
    let wrapped = crate::tui::review::wrapped_multiline_lines(reject.text, width);
    let (cursor_row, cursor_col) =
        crate::tui::review::multiline_cursor_position(reject.text, reject.cursor, width);
    let start = cursor_row.saturating_sub(editor_rows - 1);

    for row in 0..editor_rows {
        let index = start + row;
        let content = wrapped.get(index).cloned().unwrap_or_default();
        if index == cursor_row && !reject.selector_focused {
            // Replicate comment_caret logic natively
            let mut spans = vec![];
            let chars: Vec<char> = content.chars().collect();
            if cursor_col < chars.len() {
                let prefix: String = chars[..cursor_col].iter().collect();
                let char_at: String = chars[cursor_col].to_string();
                let suffix: String = chars[cursor_col + 1..].iter().collect();
                if !prefix.is_empty() {
                    spans.push(Span::raw(prefix));
                }
                spans.push(Span::styled(
                    char_at,
                    Style::default().add_modifier(Modifier::REVERSED),
                ));
                if !suffix.is_empty() {
                    spans.push(Span::raw(suffix));
                }
                lines.push(Line::from(spans));
            } else {
                let mut content_str = content.to_string();
                content_str.push(' ');
                let spans = vec![
                    Span::raw(content_str[..content_str.len() - 1].to_string()),
                    Span::styled(" ", Style::default().add_modifier(Modifier::REVERSED)),
                ];
                lines.push(Line::from(spans));
            }
        } else {
            lines.push(Line::from(content));
        }
    }

    f.render_widget(Paragraph::new(lines), area);
}

fn draw_auto_review_diff_panel(
    f: &mut Frame,
    diff: &AutoReviewDiffView<'_>,
    area: Rect,
    theme: &Theme,
) {
    f.render_widget(Clear, area);
    let mut lines = Vec::new();
    let header = format!("{}  (↑/↓ Scroll · Esc Close)", diff.title);
    lines.push(Line::from(Span::styled(
        clip_plain(&header, area.width as usize),
        theme.cursor_line_bg,
    )));

    let body_height = area.height.saturating_sub(1) as usize;
    for row in 0..body_height {
        let line = match diff.lines.get(diff.scroll + row) {
            Some(line) => clip_line(line, diff.x_offset, area.width as usize),
            None => "".to_string(),
        };
        lines.push(first_ansi_line(&line));
    }
    f.render_widget(Paragraph::new(lines), area);
}
