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
    buffer::Buffer,
    layout::{Alignment, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Paragraph, Widget},
};
use std::path::Path;

use crate::tui::{
    StatusFragment,
    header::{Banner, ConnStatus, HeaderStatus, display_model_name},
    theme::Theme,
};

const CLIENT_LOGO_ART: &[&str] = &[
    " ██████  ██████   █████  ███    ██  ██████  ██    ██ ",
    "██    ██ ██   ██ ██   ██ ████   ██ ██       ██    ██ ",
    "██    ██ ██████  ███████ ██ ██  ██ ██   ███ ██    ██ ",
    "██    ██ ██   ██ ██   ██ ██  ██ ██ ██    ██ ██    ██ ",
    " ██████  ██   ██ ██   ██ ██   ████  ██████   ██████  ",
];

/// The orangu brand brown the banner art is drawn in.
const ORANGU_BROWN: Color = Color::Rgb(139, 90, 43);
/// The `Pending` connectivity dot, matching the shade the classic frame used
/// before the indicators were drawn as Ratatui spans.
const STATUS_PENDING: Color = Color::Rgb(230, 230, 230);
/// The recognized-command prefix typed into the input window.
const VALID_COMMAND: Color = Color::Rgb(210, 140, 70);
/// The inline completion hint trailing the input in the classic frame.
const GHOST_TEXT: Color = Color::Rgb(120, 120, 120);

/// Height of the classic banner box: five rows of logo art against seven
/// status rows, plus the top and bottom borders.
pub const CLASSIC_BANNER_HEIGHT: usize = 9;

pub struct HeaderWidget<'a> {
    pub version: &'a str,
    pub current_model: &'a str,
    pub endpoint: &'a str,
    pub workspace: &'a Path,
    pub status: HeaderStatus,
    pub alignment: Banner,
}

/// The seven `Version:` / `Workspace:` / `Server:` / `Model:` / `Help:` rows
/// shown beside the logo, each paired with its visible width so the classic
/// frame can pad the box to a straight right edge (a status dot is one glyph
/// plus its leading space, which `Span::width` would report correctly but the
/// callers need before the spans are laid out).
fn header_status_lines(
    version: &str,
    current_model: &str,
    endpoint: &str,
    workspace: &Path,
    status: HeaderStatus,
    theme: &Theme,
) -> Vec<(Line<'static>, usize)> {
    let dot = |conn: ConnStatus| match conn {
        ConnStatus::Pending => Span::styled("●", Style::default().fg(STATUS_PENDING)),
        ConnStatus::Ok => Span::styled("●", theme.success),
        ConnStatus::Failed => Span::styled("●", theme.error),
    };
    let row = |text: String, conn: Option<ConnStatus>| {
        let width = text.chars().count() + if conn.is_some() { 2 } else { 0 };
        let mut spans = vec![Span::raw(text)];
        if let Some(conn) = conn {
            spans.push(Span::raw(" "));
            spans.push(dot(conn));
        }
        (Line::from(spans), width)
    };

    vec![
        row(format!("Version: {version}"), None),
        row(String::new(), None),
        row(
            format!("Workspace: {}", workspace.display()),
            Some(ConnStatus::from_bool(status.workspace_ok)),
        ),
        row(format!("Server: {endpoint}"), Some(status.server_ok)),
        row(format!("Model: {current_model}"), Some(status.model_ok)),
        row(String::new(), None),
        row("Help: /help".to_string(), None),
    ]
}

impl<'a> Widget for HeaderWidget<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        let theme = Theme::current();
        let current_model = display_model_name(self.status.is_coordinator, self.current_model);
        let status_lines = header_status_lines(
            self.version,
            current_model,
            self.endpoint,
            self.workspace,
            self.status,
            &theme,
        );

        let logo_width = CLIENT_LOGO_ART[0].chars().count();
        let gap = 2;
        let line_count = CLIENT_LOGO_ART.len().max(status_lines.len());
        let classic = theme.chrome.is_classic();
        // The classic frame boxes the banner, so every row is padded out to the
        // widest one; the modern frame lets each row end where its text does.
        let status_width = if classic {
            status_lines
                .iter()
                .map(|(_, width)| *width)
                .max()
                .unwrap_or(0)
        } else {
            0
        };
        let content_width = logo_width + gap + status_width;

        let mut lines = Vec::new();
        if classic {
            lines.push(Line::from(format!("┏{}┓", "━".repeat(content_width + 2))));
        }

        for index in 0..line_count {
            let mut spans = Vec::new();
            if classic {
                spans.push(Span::raw("┃ "));
            }
            match CLIENT_LOGO_ART.get(index) {
                Some(art) => spans.push(Span::styled(*art, Style::default().fg(ORANGU_BROWN))),
                None => spans.push(Span::raw(" ".repeat(logo_width))),
            }
            spans.push(Span::raw(" ".repeat(gap)));

            let used = match status_lines.get(index) {
                Some((line, width)) => {
                    spans.extend(line.spans.clone());
                    *width
                }
                None => 0,
            };
            if classic {
                spans.push(Span::raw(" ".repeat(status_width.saturating_sub(used))));
                spans.push(Span::raw(" ┃"));
            }
            lines.push(Line::from(spans));
        }

        if classic {
            lines.push(Line::from(format!("┗{}┛", "━".repeat(content_width + 2))));
        }

        let alignment = match self.alignment {
            Banner::Left => Alignment::Left,
            Banner::Center => Alignment::Center,
            Banner::Right => Alignment::Right,
        };

        Paragraph::new(lines).alignment(alignment).render(area, buf);
    }
}

pub struct PromptFrameWidget<'a> {
    pub current_model: &'a str,
    pub prompt_prefix: &'a str,
    pub input: &'a str,
    pub cursor: usize,
    pub ghost: &'a str,
    pub valid_command_len: usize,
    pub left_status: Option<&'a StatusFragment>,
    pub pending_count: usize,
    pub graph_status: Option<ConnStatus>,
    pub prompt_branch: Option<&'a str>,
}

impl<'a> PromptFrameWidget<'a> {
    /// The input rows themselves — identical in both frames, apart from how
    /// many columns the surrounding chrome leaves for them.
    fn input_lines(&self, content_width: usize, ghost_color: Color) -> Vec<Line<'static>> {
        let input_lines_wrapped =
            crate::tui::screen::wrapped_input_lines(self.input, content_width, self.prompt_prefix);
        let prompt_width = self.prompt_prefix.chars().count();
        let cmd_len = self.valid_command_len;
        let last_input_index = input_lines_wrapped.len().saturating_sub(1);
        let mut char_offset = 0;
        let mut lines = Vec::with_capacity(input_lines_wrapped.len());

        for (index, input_line) in input_lines_wrapped.iter().enumerate() {
            let content = crate::tui::screen::truncate_to_width(
                input_line,
                content_width.saturating_sub(prompt_width),
            );
            let used_width = content.chars().count();

            let prefix = if index == 0 {
                self.prompt_prefix.to_string()
            } else {
                " ".repeat(prompt_width)
            };
            let mut spans = vec![Span::raw(prefix)];

            if cmd_len > 0 {
                for (offset, ch) in content.chars().enumerate() {
                    let global_index = char_offset + offset;
                    let style = if global_index < cmd_len {
                        Style::default().fg(VALID_COMMAND)
                    } else {
                        Style::default()
                    };
                    spans.push(Span::styled(ch.to_string(), style));
                }
            } else {
                spans.push(Span::raw(content.clone()));
            }
            char_offset += used_width;

            let used = used_width + prompt_width;

            if index == last_input_index && !self.ghost.is_empty() {
                let ghost = crate::tui::screen::truncate_to_width(
                    self.ghost,
                    content_width.saturating_sub(used),
                );
                if !ghost.is_empty() {
                    spans.push(Span::styled(ghost, Style::default().fg(ghost_color)));
                }
            }

            lines.push(Line::from(spans));
        }

        lines
    }

    /// The classic frame: a full-width separator, the input window, a second
    /// separator, and the status line below them.
    fn render_classic(self, area: Rect, buf: &mut Buffer, theme: &Theme) {
        let width = area.width as usize;
        let separator = "\u{2501}".repeat(width);

        let mut lines = vec![Line::from(separator.clone())];
        lines.extend(self.input_lines(width, GHOST_TEXT));
        lines.push(Line::from(separator));
        lines.push(status_line(
            width,
            self.prompt_branch,
            self.left_status,
            self.current_model,
            self.pending_count,
            self.graph_status,
            theme,
        ));

        Paragraph::new(lines).render(area, buf);
    }

    /// The modern frame: a rounded box around the input window, and the same
    /// status line the classic frame uses on the row below it — opening with
    /// the branch, which this frame keeps out of the prompt prefix.
    fn render_modern(self, area: Rect, buf: &mut Buffer, theme: &Theme) {
        let width = area.width as usize;
        let lines = self.input_lines(width.saturating_sub(4), Color::DarkGray);

        // The box takes everything but the last row, which the status line gets
        // to itself — the same split the classic frame makes below its second
        // separator.
        let box_area = Rect {
            height: area.height.saturating_sub(1),
            ..area
        };
        let status_area = Rect {
            y: area.y + box_area.height,
            height: area.height - box_area.height,
            ..area
        };

        // One column of padding inside the border, so the prompt marker doesn't
        // butt against it. Together with the two borders this is the four
        // columns `input_lines` wraps at.
        let block = ratatui::widgets::Block::default()
            .borders(ratatui::widgets::Borders::ALL)
            .border_type(ratatui::widgets::BorderType::Rounded)
            .border_style(Style::default().fg(Color::DarkGray))
            .padding(ratatui::widgets::Padding::horizontal(1));

        Paragraph::new(lines).block(block).render(box_area, buf);

        if status_area.height > 0 {
            Paragraph::new(status_line(
                width,
                self.prompt_branch,
                self.left_status,
                self.current_model,
                self.pending_count,
                self.graph_status,
                theme,
            ))
            .render(status_area, buf);
        }
    }
}

/// The status line under the input window, shared by both frames: `branch` and
/// the activity status (Working/Thinking) on the left, a centered `Graph:` /
/// `Pending:` group, and the model name flush right. When space runs short the
/// left side is kept first, then the model name, then the centered group; if
/// the `Graph:`/`Pending:` pair doesn't fit but the graph indicator alone
/// would, it's shown alone rather than dropped for a queue count that only
/// matters transiently.
///
/// `branch` is `None` on the classic frame, which carries it in the prompt
/// prefix instead.
fn status_line(
    width: usize,
    branch: Option<&str>,
    left_status: Option<&StatusFragment>,
    current_model: &str,
    pending_count: usize,
    graph_status: Option<ConnStatus>,
    theme: &Theme,
) -> Line<'static> {
    let mut spans = Vec::new();
    let mut left_visible_width = 0;

    if let Some(branch) = branch.filter(|branch| !branch.trim().is_empty()) {
        let label = format!("{branch}  ");
        left_visible_width += label.chars().count();
        spans.push(Span::styled(label, Style::default().fg(Color::DarkGray)));
    }
    if let Some(left) = left_status.filter(|status| status.visible_width > 0) {
        left_visible_width += left.visible_width;
        if let Ok(mut text) = ansi_to_tui::IntoText::into_text(&left.rendered)
            && let Some(line) = text.lines.pop()
        {
            spans.extend(line.spans);
        }
    }

    let right_space = width.saturating_sub(left_visible_width);
    let model_width = current_model.chars().count();
    let show_model = right_space >= model_width;
    let gap = if show_model {
        right_space.saturating_sub(model_width)
    } else {
        right_space
    };

    let graph_width = graph_status.map(|_| "Graph: ".chars().count() + 1);
    let pending_text = (pending_count > 0).then(|| format!("Pending: {pending_count}"));
    let combined_width = match (graph_width, &pending_text) {
        (Some(graph), Some(pending)) => Some(graph + 3 + pending.chars().count()),
        (Some(graph), None) => Some(graph),
        (None, Some(pending)) => Some(pending.chars().count()),
        (None, None) => None,
    };
    let (middle_width, with_pending) = match combined_width {
        Some(combined) if show_model && combined <= gap => (Some(combined), pending_text.is_some()),
        _ => match graph_width {
            Some(graph) if show_model && graph <= gap => (Some(graph), false),
            _ => (None, false),
        },
    };

    match middle_width {
        Some(middle) => {
            let left_pad = (gap - middle) / 2;
            spans.push(Span::raw(" ".repeat(left_pad)));
            if let Some(status) = graph_status {
                spans.push(Span::raw("Graph: "));
                spans.push(Span::styled(
                    "\u{25cf}",
                    match status {
                        ConnStatus::Pending => Style::default().fg(STATUS_PENDING),
                        ConnStatus::Ok => theme.success,
                        ConnStatus::Failed => theme.error,
                    },
                ));
            }
            if with_pending && let Some(pending) = pending_text {
                if graph_status.is_some() {
                    spans.push(Span::raw("   "));
                }
                spans.push(Span::raw(pending));
            }
            spans.push(Span::raw(" ".repeat(gap - middle - left_pad)));
        }
        None => spans.push(Span::raw(" ".repeat(gap))),
    }

    if show_model {
        spans.push(Span::raw(current_model.to_string()));
    }

    Line::from(spans)
}

impl<'a> Widget for PromptFrameWidget<'a> {
    fn render(self, area: Rect, buf: &mut Buffer) {
        if area.height == 0 || area.width == 0 {
            return;
        }

        let theme = Theme::current();
        if theme.chrome.is_classic() {
            self.render_classic(area, buf, &theme);
        } else {
            self.render_modern(area, buf, &theme);
        }
    }
}
