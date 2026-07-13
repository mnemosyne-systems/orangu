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

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
};

use super::*;
use crate::tui::screen::{cursor_position, prompt_prefix, wrapped_input_lines};
use crate::tui::text::clip_line;

pub fn draw_review_screen(frame: &mut ratatui::Frame, args: ReviewScreenArgs<'_>) {
    let theme = Theme::current();
    let area = frame.area();
    let width = area.width.max(1);
    let height = area.height.max(1);
    frame.render_widget(
        ratatui::widgets::Block::default().style(
            ratatui::style::Style::default()
                .bg(theme.bg_base)
                .fg(theme.text_primary),
        ),
        area,
    );

    let prefix = prompt_prefix(args.prompt_branch);
    let input_lines_count = wrapped_input_lines(args.input, width as usize, &prefix).len();
    let prompt_frame_height = (input_lines_count + 3) as u16;
    let pane_rows = height.saturating_sub(prompt_frame_height).max(1);

    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(pane_rows),
            Constraint::Length(prompt_frame_height),
        ])
        .split(area);

    let panes_area = layout[0];
    let prompt_area = layout[1];

    if let Some(feedback) = &args.feedback {
        draw_review_feedback_panel(frame, panes_area, feedback, width as usize, &theme);
    } else {
        draw_review_panes(frame, panes_area, &args, width as usize, &theme);
    }

    let current_model = crate::tui::header::display_model_name(false, args.current_model); // Assuming false for is_coordinator in review mode
    let prompt_widget = crate::tui::widgets::PromptFrameWidget {
        current_model,
        prompt_prefix: &prefix,
        input: args.input,
        cursor: args.cursor,
        ghost: args.ghost,
        valid_command_len: 0,
        left_status: args.left_status.as_ref(),
        pending_count: args.pending_count,
        graph_status: None,
        prompt_branch: args.prompt_branch,
    };
    frame.render_widget(prompt_widget, prompt_area);
}

fn draw_review_feedback_panel(
    frame: &mut ratatui::Frame,
    area: Rect,
    feedback: &ReviewFeedbackView<'_>,
    width: usize,
    theme: &Theme,
) {
    let header_text = format!("{} (x to close · ↑/↓ scroll)", feedback.title);
    let mut lines = Vec::new();

    lines.push(Line::from(Span::styled(
        clip_line(&header_text, 0, width),
        theme.cursor_line_bg,
    )));

    if let Some(question) = feedback.question {
        lines.push(Line::from(Span::styled(
            clip_line(&format!("> {question}"), 0, width),
            theme.highlight,
        )));
    }

    let header_rows = lines.len();
    let body_height = area.height.saturating_sub(header_rows as u16) as usize;

    for row in 0..body_height {
        let content = if let Some(line) = feedback.lines.get(feedback.scroll + row) {
            clip_line(line, feedback.x_offset, width)
        } else {
            String::new()
        };
        if let Ok(text) = ansi_to_tui::IntoText::into_text(&content)
            && let Some(line) = text.lines.into_iter().next()
        {
            lines.push(line);
            continue;
        }
        lines.push(Line::from(content));
    }

    frame.render_widget(Paragraph::new(lines), area);
}

fn draw_review_panes(
    frame: &mut ratatui::Frame,
    area: Rect,
    args: &ReviewScreenArgs<'_>,
    width: usize,
    theme: &Theme,
) {
    let right_width = crate::tui::review::review_right_width(args.files, width) as u16;
    let left_width = width.saturating_sub(right_width as usize).max(1) as u16;

    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(2), Constraint::Min(0)])
        .split(area);

    let header_area = main_layout[0];
    let panes_area = main_layout[1];

    let layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(left_width),
            Constraint::Length(right_width),
        ])
        .split(panes_area);

    let left_area = layout[0];
    let right_area = layout[1];

    let left_inner_area = left_area.inner(ratatui::layout::Margin {
        vertical: 1,
        horizontal: 1,
    });
    let body_height = left_inner_area.height as usize;
    let right_body_height = right_area.height.saturating_sub(1) as usize;

    // Header Badges
    let mut header_spans = vec![Span::styled(
        "Review  ",
        Style::default().fg(theme.text_primary),
    )];
    let badge_style = theme.cursor_line_bg;
    let badges = [
        ("Alt+j/k", "Switch file"),
        ("Alt+a", "Approve"),
        ("Alt+r", "Reject"),
        ("Alt+o", "Review"),
        ("Alt+c", "Comment"),
        ("Alt+e", "Open"),
        ("Alt+x", "Exit"),
    ];
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
    frame.render_widget(
        Paragraph::new(Line::from(header_spans)).block(
            ratatui::widgets::Block::default()
                .borders(ratatui::widgets::Borders::BOTTOM)
                .border_style(theme.muted),
        ),
        header_area,
    );

    // Right Pane
    let mut right_lines = Vec::new();
    let right_header = format!("Files ({})", args.files.len());
    right_lines.push(Line::from(Span::styled(
        clip_line(&right_header, 0, right_area.width as usize),
        Style::default().fg(theme.text_primary),
    )));

    let list_start = args.list_offset;

    let selected_lines: &[String] = args
        .files
        .get(args.selected)
        .map(|file| file.diff_lines.as_slice())
        .unwrap_or(&[]);

    let has_comment = |index: usize| args.commented_lines.contains(&index);
    let mut box_shown = false;
    let mut diff_index = args.scroll;
    let mut lines_pushed = 0;

    for row_idx in 0..right_body_height {
        // Right side
        let file_index = list_start + row_idx;
        if let Some(file) = args.files.get(file_index) {
            let status_box = crate::tui::review::review_status_box(file.status);
            let mut right_spans = Vec::new();
            if let Ok(text) = ansi_to_tui::IntoText::into_text(&status_box) {
                if let Some(line) = text.lines.into_iter().next() {
                    right_spans.extend(line.spans);
                }
            } else {
                right_spans.push(Span::raw("[ ]"));
            }
            right_spans.push(Span::styled(
                format!(" {}", file.path),
                Style::default().fg(theme.text_primary),
            ));

            let mut line = Line::from(right_spans);
            if file_index == args.selected {
                line = line.style(theme.selected_file);
            }
            right_lines.push(line);
        } else {
            right_lines.push(Line::raw(""));
        }
    }

    // Left side
    let mut left_lines = Vec::new();
    for _row_idx in 0..body_height {
        if let Some(editor) = &args.comment_editor
            && diff_index == args.line + 1
            && !box_shown
            && lines_pushed < body_height
        {
            box_shown = true;

            // Add editor rows here directly to left_lines
            let inner_width = left_inner_area.width as usize;

            let chosen = if editor.selector_focused {
                Span::styled(
                    format!(" {} ", editor.category),
                    theme.comment_bg.add_modifier(Modifier::REVERSED),
                )
            } else {
                Span::styled(format!("[{}]", editor.category), theme.comment_bg)
            };

            let mut spans1 = vec![
                Span::styled("▕ ", theme.comment_bg.fg(Color::Rgb(120, 160, 120))),
                Span::styled("Category: ", theme.comment_bg),
                chosen,
                Span::styled("  ", theme.comment_bg),
                Span::styled(
                    "↑/↓ Category · Tab Switch focus",
                    theme.comment_bg.fg(Color::DarkGray),
                ),
            ];
            let w1 = spans1.iter().map(|s| s.width()).sum::<usize>();
            let pad1 = " ".repeat(inner_width.saturating_sub(w1.saturating_sub(2))); // subtract 2 for ▕
            spans1.push(Span::styled(pad1, theme.comment_bg));
            left_lines.push(Line::from(spans1));
            lines_pushed += 1;

            let wrapped = wrapped_input_lines(editor.text, inner_width, "");
            let (cursor_row, cursor_col) =
                cursor_position(editor.text, editor.cursor, inner_width, "");
            let start =
                cursor_row.saturating_sub(crate::tui::review::REVIEW_COMMENT_BOX_HEIGHT - 1);

            for r in 0..crate::tui::review::REVIEW_COMMENT_BOX_HEIGHT {
                if lines_pushed >= body_height {
                    break;
                }
                let idx = start + r;
                let content = wrapped.get(idx).cloned().unwrap_or_default();
                let mut spans = vec![Span::styled(
                    "▕ ",
                    theme.comment_bg.fg(Color::Rgb(120, 160, 120)),
                )];

                if idx == cursor_row && !editor.selector_focused {
                    // caret
                    let chars: Vec<char> = content.chars().collect();
                    if cursor_col < chars.len() {
                        let (before, rest) = content.split_at(
                            content
                                .char_indices()
                                .nth(cursor_col)
                                .map(|(i, _)| i)
                                .unwrap_or(0),
                        );
                        let (caret, after) = rest.split_at(
                            rest.char_indices()
                                .nth(1)
                                .map(|(i, _)| i)
                                .unwrap_or(rest.len()),
                        );
                        spans.push(Span::styled(before.to_string(), theme.comment_bg));
                        spans.push(Span::styled(
                            caret.to_string(),
                            theme.comment_bg.add_modifier(Modifier::REVERSED),
                        ));
                        spans.push(Span::styled(after.to_string(), theme.comment_bg));
                    } else if chars.len() < inner_width {
                        spans.push(Span::styled(content.clone(), theme.comment_bg));
                        spans.push(Span::styled(
                            " ",
                            theme.comment_bg.add_modifier(Modifier::REVERSED),
                        ));
                    } else {
                        spans.push(Span::styled(content.clone(), theme.comment_bg));
                    }
                } else {
                    spans.push(Span::styled(content.clone(), theme.comment_bg));
                }

                let w = spans.iter().map(|s| s.width()).sum::<usize>();
                let pad = " ".repeat(inner_width.saturating_sub(w.saturating_sub(2)));
                spans.push(Span::styled(pad, theme.comment_bg));

                left_lines.push(Line::from(spans));
                lines_pushed += 1;
            }
        }

        if lines_pushed < body_height {
            if let Some(diff_line) = selected_lines.get(diff_index) {
                let mut spans = Vec::new();
                let is_cursor = diff_index == args.line;
                let clipped = clip_line(diff_line, args.x_offset, left_inner_area.width as usize);

                if let Ok(text) = ansi_to_tui::IntoText::into_text(&clipped) {
                    if let Some(parsed_line) = text.lines.into_iter().next() {
                        spans.extend(parsed_line.spans);
                    }
                } else {
                    spans.push(Span::raw(clipped));
                }

                if has_comment(diff_index) {
                    let w = spans.iter().map(|s| s.width()).sum::<usize>();
                    let pad =
                        " ".repeat(
                            left_area.width.saturating_sub(2).saturating_sub(w as u16) as usize
                        );
                    spans.push(Span::raw(pad));
                    spans.push(Span::raw(" "));
                    spans.push(Span::styled("●", theme.warning));
                } else {
                    let w = spans.iter().map(|s| s.width()).sum::<usize>();
                    let pad = " ".repeat(left_inner_area.width.saturating_sub(w as u16) as usize);
                    spans.push(Span::raw(pad));
                }

                let mut line = Line::from(spans);
                if is_cursor {
                    line = line.style(theme.cursor_line_bg);
                }
                left_lines.push(line);
            } else {
                left_lines.push(Line::raw(""));
            }
            diff_index += 1;
            lines_pushed += 1;
        }
    }

    frame.render_widget(
        Paragraph::new(left_lines)
            .block(ratatui::widgets::Block::bordered().border_style(theme.muted)),
        left_area,
    );
    frame.render_widget(
        Paragraph::new(right_lines).block(
            ratatui::widgets::Block::default().padding(ratatui::widgets::Padding::new(1, 1, 1, 0)),
        ),
        right_area,
    );
}
