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

use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthChar;

/// Wrap a styled line at word boundaries, preserving every span's style. Long
/// unbroken words are split so no rendered row exceeds `visible_width`.
pub fn wrap_ratatui_line(line: &Line<'_>, visible_width: usize) -> Vec<Line<'static>> {
    let visible_width = visible_width.max(1);
    let mut words: Vec<Vec<(char, ratatui::style::Style)>> = Vec::new();
    let mut word = Vec::new();

    for span in &line.spans {
        for ch in span.content.chars() {
            if ch.is_whitespace() && ch != '\n' {
                if !word.is_empty() {
                    words.push(std::mem::take(&mut word));
                }
                words.push(vec![(ch, span.style)]);
            } else {
                word.push((ch, span.style));
            }
        }
    }
    if !word.is_empty() || words.is_empty() {
        words.push(word);
    }

    let mut rows: Vec<Vec<(char, ratatui::style::Style)>> = vec![Vec::new()];
    let mut row_width = 0;
    for word in words {
        let word_width = word
            .iter()
            .map(|(ch, _)| UnicodeWidthChar::width(*ch).unwrap_or(0))
            .sum::<usize>();
        let whitespace = word.iter().all(|(ch, _)| ch.is_whitespace());
        if row_width > 0 && row_width + word_width > visible_width {
            rows.push(Vec::new());
            row_width = 0;
            if whitespace {
                continue;
            }
        }
        for (ch, style) in word {
            let char_width = UnicodeWidthChar::width(ch).unwrap_or(0);
            if row_width > 0 && row_width + char_width > visible_width {
                rows.push(Vec::new());
                row_width = 0;
            }
            rows.last_mut().expect("wrap row exists").push((ch, style));
            row_width += char_width;
        }
    }

    rows.into_iter()
        .map(|row| {
            let spans: Vec<Span<'static>> = row
                .into_iter()
                .map(|(ch, style)| Span::styled(ch.to_string(), style))
                .collect();
            let mut wrapped = Line::from(spans);
            if let Some(alignment) = line.alignment {
                wrapped = wrapped.alignment(alignment);
            }
            wrapped
        })
        .collect()
}

pub fn clip_ratatui_line<'a>(
    line: &Line<'a>,
    mut x_offset: usize,
    visible_width: usize,
) -> Line<'a> {
    let mut new_spans = Vec::new();
    let mut current_width = 0;

    for span in &line.spans {
        if current_width >= visible_width {
            break;
        }

        let span_width = span.width();
        if x_offset >= span_width {
            x_offset -= span_width;
            continue;
        }

        let mut content = String::new();
        for ch in span.content.chars() {
            let ch_width = UnicodeWidthChar::width(ch).unwrap_or(0);
            if x_offset > 0 {
                // A viewport cannot render half of a wide character, so skip
                // the entire character when its leading cell is clipped.
                x_offset = x_offset.saturating_sub(ch_width);
                continue;
            }

            if current_width + ch_width > visible_width {
                break;
            }

            content.push(ch);
            current_width += ch_width;
        }

        if !content.is_empty() {
            new_spans.push(Span::styled(content, span.style));
        }
    }

    let mut new_line = Line::from(new_spans);
    if let Some(align) = line.alignment {
        new_line = new_line.alignment(align);
    }
    new_line
}

pub fn clip_line(line: &str, x_offset: usize, visible_width: usize) -> String {
    let mut result = String::new();
    let mut col = 0usize;
    let mut pre_clip_ansi = String::new();
    let mut in_visible = false;
    let mut truncated = false;
    let mut chars = line.chars().peekable();

    'outer: while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            let mut seq = String::from('\x1b');
            match chars.peek() {
                Some(&'[') => {
                    seq.push(chars.next().unwrap());
                    loop {
                        match chars.next() {
                            Some(c) => {
                                let done = c.is_ascii_alphabetic() || c == '~' || c == '@';
                                seq.push(c);
                                if done {
                                    break;
                                }
                            }
                            None => break 'outer,
                        }
                    }
                }
                Some(&'O') => {
                    seq.push(chars.next().unwrap());
                    if let Some(c) = chars.next() {
                        seq.push(c);
                    }
                }
                // An OSC sequence (e.g. an OSC 8 hyperlink): `ESC ] ... ST`,
                // where the terminator is BEL or `ESC \`. It draws nothing, so
                // it is carried through but never counts toward a column.
                Some(&']') => {
                    seq.push(chars.next().unwrap());
                    loop {
                        match chars.next() {
                            Some('\x07') => {
                                seq.push('\x07');
                                break;
                            }
                            Some('\x1b') => {
                                seq.push('\x1b');
                                if chars.peek() == Some(&'\\') {
                                    seq.push(chars.next().unwrap());
                                }
                                break;
                            }
                            Some(c) => seq.push(c),
                            None => break 'outer,
                        }
                    }
                }
                _ => {}
            }
            if col < x_offset {
                pre_clip_ansi.push_str(&seq);
            } else {
                result.push_str(&seq);
            }
            continue;
        }

        if col < x_offset {
            col += 1;
            continue;
        }

        let vis_col = col - x_offset;
        if vis_col >= visible_width {
            truncated = true;
            break;
        }

        if !in_visible {
            result.push_str(&pre_clip_ansi);
            in_visible = true;
        }

        result.push(ch);
        col += 1;
    }

    if truncated {
        result.push_str("\x1b[0m");
    }

    result
}

pub fn visible_line_width(line: &str) -> usize {
    let mut col = 0usize;
    let mut chars = line.chars().peekable();
    'outer: while let Some(ch) = chars.next() {
        if ch == '\x1b' {
            match chars.peek() {
                Some(&'[') => {
                    chars.next();
                    loop {
                        match chars.next() {
                            Some(c) => {
                                if c.is_ascii_alphabetic() || c == '~' || c == '@' {
                                    break;
                                }
                            }
                            None => break 'outer,
                        }
                    }
                }
                Some(&'O') => {
                    chars.next();
                    chars.next();
                }
                // An OSC sequence (e.g. an OSC 8 hyperlink) draws nothing, so
                // skip it entirely: `ESC ] ... ST`, terminated by BEL or `ESC \`.
                Some(&']') => {
                    chars.next();
                    loop {
                        match chars.next() {
                            Some('\x07') => break,
                            Some('\x1b') => {
                                if chars.peek() == Some(&'\\') {
                                    chars.next();
                                }
                                break;
                            }
                            Some(_) => {}
                            None => break 'outer,
                        }
                    }
                }
                _ => {}
            }
            continue;
        }
        col += 1;
    }
    col
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An OSC 8 hyperlink: `label` is shown and clickable, the URL is not drawn.
    fn osc8_link(label: &str, url: &str) -> String {
        format!("\x1b]8;;{url}\x1b\\{label}\x1b]8;;\x1b\\")
    }

    #[test]
    fn visible_width_ignores_osc8_hyperlinks() {
        // Only the label's six glyphs count; the OSC 8 control bytes (and the
        // URL they carry) are zero-width.
        let line = osc8_link("orangu", "https://example.com/orangu/");
        assert_eq!(visible_line_width(&line), "orangu".chars().count());

        // The same holds with a BEL terminator instead of ST.
        let bel = "\x1b]8;;https://example.com\x07orangu\x1b]8;;\x07";
        assert_eq!(visible_line_width(bel), "orangu".chars().count());
    }

    #[test]
    fn clip_line_preserves_osc8_hyperlinks_and_their_width() {
        let line = format!("see {} now", osc8_link("orangu", "https://example.com/"));
        // Wide enough to keep the whole line: the visible text is "see orangu now".
        let clipped = clip_line(&line, 0, 40);
        assert_eq!(
            visible_line_width(&clipped),
            "see orangu now".chars().count()
        );
        // The hyperlink's opening and closing control sequences survive.
        assert!(clipped.contains("\x1b]8;;https://example.com/\x1b\\"));
        assert!(clipped.contains("\x1b]8;;\x1b\\"));
    }

    #[test]
    fn wraps_styled_lines_on_word_boundaries() {
        let line = Line::from(vec![
            Span::styled("alpha ", ratatui::style::Style::default().bold()),
            Span::raw("beta gamma"),
        ]);
        let wrapped = wrap_ratatui_line(&line, 6);
        let text = wrapped
            .iter()
            .map(|line| line.to_string())
            .collect::<Vec<_>>();
        assert_eq!(text, ["alpha ", "beta ", "gamma"]);
        assert!(
            wrapped[0].spans[0]
                .style
                .add_modifier
                .contains(ratatui::style::Modifier::BOLD)
        );
    }
}
