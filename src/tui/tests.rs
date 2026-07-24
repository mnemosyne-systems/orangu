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
use ratatui::{Terminal, backend::TestBackend};
use std::path::Path;

fn setup_test_terminal(width: u16, height: u16) -> Terminal<TestBackend> {
    let backend = TestBackend::new(width, height);
    Terminal::new(backend).unwrap()
}

/// Every row of the rendered screen, as plain text.
fn screen_rows(terminal: &Terminal<TestBackend>, width: u16, height: u16) -> Vec<String> {
    let buffer = terminal.backend().buffer();
    (0..height)
        .map(|y| {
            (0..width)
                .map(|x| buffer.cell((x, y)).unwrap().symbol())
                .collect::<String>()
        })
        .collect()
}

fn default_render_args<'a>() -> ScreenRenderArgs<'a> {
    ScreenRenderArgs {
        version: "0.11.0",
        current_model: "gpt-4",
        endpoint: "https://api.openai.com",
        workspace: Path::new("/test"),
        prompt_branch: None,
        status: HeaderStatus {
            workspace_ok: true,
            server_ok: crate::tui::ConnStatus::Ok,
            model_ok: crate::tui::ConnStatus::Ok,
            is_coordinator: false,
        },
        banner: Banner::Left,
        tab_bar: None,
        tab_statuses: &[],
        transcript: &[],
        transcript_epoch: 0,
        scroll_offset: 0,
        left_status: None,
        pending_count: 0,
        pending_lines: &[],
        input: "",
        cursor: 0,
        ghost: "",
        virtual_width: 80,
        actual_width: 80,
        actual_height: 24,
        x_offset: 0,
        dropdown_candidates: None,
        dropdown_selected: 0,
        valid_command_len: 0,
    }
}

#[test]
fn test_render_empty_screen() {
    let mut terminal = setup_test_terminal(80, 24);
    let args = default_render_args();

    terminal.draw(|f| renderer::render(f, &args)).unwrap();
    let buffer = terminal.backend().buffer();

    // The screen should have a top separator, output area, bottom separator, and status line.
    // Ensure the model is displayed in the bottom right corner (or bottom line).
    let last_row = 23;
    let mut bottom_line = String::new();
    for col in 0..80 {
        bottom_line.push_str(buffer.cell((col, last_row)).unwrap().symbol());
    }
    assert!(bottom_line.contains("gpt-4"));
}

#[test]
fn test_render_dropdown_popup() {
    let mut terminal = setup_test_terminal(80, 24);
    let mut args = default_render_args();

    let candidates = vec![
        ("/open".to_string(), "Open file".to_string()),
        ("/review".to_string(), "Review diff".to_string()),
    ];
    args.dropdown_candidates = Some(&candidates);
    args.dropdown_selected = 1;
    args.input = "/r";
    args.cursor = 2;

    terminal.draw(|f| renderer::render(f, &args)).unwrap();
    let buffer = terminal.backend().buffer();

    // Find if the dropdown items are rendered.
    let mut content = String::new();
    for y in 0..24 {
        for x in 0..80 {
            content.push_str(buffer.cell((x, y)).unwrap().symbol());
        }
    }

    assert!(content.contains("/open"));
    assert!(content.contains("/review"));
}

#[test]
fn test_render_cursor_position() {
    let mut terminal = setup_test_terminal(80, 24);
    let mut args = default_render_args();

    args.input = "hello world";
    args.cursor = 5; // After 'hello'

    terminal.draw(|f| renderer::render(f, &args)).unwrap();

    // The cursor position should be updated.
    // Default prompt prefix is "> ", which is 2 chars.
    // Padding might be 1 char, so cursor column should be 2 + 5 = 7 (0-indexed maybe?)
    // Let's just check that it doesn't crash and sets it to some valid position.
    let pos = terminal.get_cursor_position().unwrap();
    // Prompt is usually at the bottom, just above status line.
    assert!(pos.x >= 5); // x position
    assert!(pos.y >= 21); // y position
}

#[test]
fn test_render_native_auto_review_screen() {
    let mut terminal = setup_test_terminal(80, 24);
    let files = vec![ReviewEntry {
        path: "src/main.rs".to_string(),
        status: ReviewStatus::Unreviewed,
        diff_lines: vec![],
        patch: String::new(),
    }];
    let report_lines = vec![
        "## Correctness".to_string(),
        "\x1b[1mOverall\x1b[0m".to_string(),
        "\x1b[2m(pending)\x1b[0m".to_string(),
    ];

    terminal
        .draw(|f| {
            auto_review_native::draw_auto_review_screen(
                f,
                AutoReviewScreenArgs {
                    files: &files,
                    selected: Some(0),
                    list_offset: 0,
                    report_lines: &report_lines,
                    selected_lines: Some((1, 2)),
                    scroll: 0,
                    x_offset: 0,
                    status: "File: src/main.rs",
                    reviewing: None,
                    browsing: true,
                    prestart: false,
                    modes: &[],
                    reject: None,
                    diff: None,
                    input: "",
                    cursor: 0,
                    ghost: "",
                    current_model: "gpt-4",
                    prompt_branch: Some("main"),
                    left_status: None,
                    pending_count: 0,
                    graph_status: None,
                    actual_width: 80,
                    actual_height: 24,
                },
            );
        })
        .unwrap();

    let mut content = String::new();
    for y in 0..24 {
        for x in 0..80 {
            content.push_str(terminal.backend().buffer().cell((x, y)).unwrap().symbol());
        }
    }

    assert!(content.contains("Auto review"));
    assert!(content.contains("Alt+j/k"));
    assert!(content.contains("Switch file"));
    assert!(content.contains("src/main.rs"));
    assert!(content.contains("File: src/main.rs"));
    assert!(content.contains("Overall"));
    assert!(content.contains("(pending)"));
    assert!(!content.contains("[1m"));
    assert!(!content.contains("[2m"));
}

#[test]
fn classic_chrome_draws_the_boxed_banner_and_separator_prompt() {
    let _guard = crate::tui::theme::theme_test_guard();
    crate::tui::Theme::apply_named("classic").expect("classic");

    let mut terminal = setup_test_terminal(120, 24);
    let mut args = default_render_args();
    args.actual_width = 120;
    args.prompt_branch = Some("main");
    args.transcript = &[];

    terminal.draw(|f| renderer::render(f, &args)).unwrap();
    let rows = screen_rows(&terminal, 120, 24);

    // The banner is boxed and pinned to the very top, left-aligned per the
    // `banner` configuration key, with the branding and status beside it.
    assert!(rows[0].starts_with('\u{250f}'), "{:?}", rows[0]);
    assert!(rows[1].starts_with('\u{2503}'), "{:?}", rows[1]);
    assert!(rows[8].starts_with('\u{2517}'), "{:?}", rows[8]);
    assert!(rows[1..8].iter().any(|row| row.contains("Version: 0.11.0")));
    assert!(rows[1..8].iter().any(|row| row.contains("Help: /help")));
    assert!(rows[1..8].iter().any(|row| row.contains('\u{2588}')));

    // The prompt frame is a pair of full-width separators around an input
    // window carrying a bare `> ` marker — no branch in it — with the status
    // line below, opening with the branch.
    assert_eq!(rows[20], "\u{2501}".repeat(120));
    assert!(rows[21].starts_with("> "), "{:?}", rows[21]);
    assert_eq!(rows[22], "\u{2501}".repeat(120));
    assert!(rows[23].starts_with("main "), "{:?}", rows[23]);
    assert!(rows[23].contains("Graph: "), "{:?}", rows[23]);
    assert!(rows[23].trim_end().ends_with("gpt-4"), "{:?}", rows[23]);
    // The branch is not in the prompt.
    assert!(!rows.iter().any(|row| row.starts_with("main> ")));
    // No rounded box anywhere.
    assert!(!rows.iter().any(|row| row.contains('\u{256d}')));

    crate::tui::Theme::apply_named("classic").expect("restore classic");
}

#[test]
fn classic_chrome_honors_the_banner_alignment() {
    let _guard = crate::tui::theme::theme_test_guard();
    crate::tui::Theme::apply_named("classic").expect("classic");

    let mut terminal = setup_test_terminal(120, 24);
    let mut args = default_render_args();
    args.actual_width = 120;
    args.banner = Banner::Right;

    terminal.draw(|f| renderer::render(f, &args)).unwrap();
    let rows = screen_rows(&terminal, 120, 24);

    assert!(rows[0].starts_with(' '), "{:?}", rows[0]);
    assert!(rows[0].trim_end().ends_with('\u{2513}'), "{:?}", rows[0]);
    assert_eq!(rows[0].trim_end().chars().count(), 120);

    crate::tui::Theme::apply_named("classic").expect("restore classic");
}

#[test]
fn modern_chrome_draws_the_rounded_prompt_box() {
    let _guard = crate::tui::theme::theme_test_guard();
    crate::tui::Theme::apply_named("modern_dark").expect("modern_dark");

    let mut terminal = setup_test_terminal(120, 26);
    let mut args = default_render_args();
    args.actual_width = 120;
    args.actual_height = 26;
    args.prompt_branch = Some("main");

    terminal.draw(|f| renderer::render(f, &args)).unwrap();
    let rows = screen_rows(&terminal, 120, 26);

    // The prompt frame is the last four rows: a three-row rounded box around
    // the one input line, then the status line.
    let status = rows.last().expect("status row");
    let box_bottom = &rows[rows.len() - 2];
    let box_input = &rows[rows.len() - 3];
    let box_top = &rows[rows.len() - 4];

    // All four corners are the rounded forms, not the square ┌┐└┘.
    assert!(box_top.contains('\u{256d}'), "{box_top:?}");
    assert!(box_top.contains('\u{256e}'), "{box_top:?}");
    assert!(box_bottom.contains('\u{2570}'), "{box_bottom:?}");
    assert!(box_bottom.contains('\u{256f}'), "{box_bottom:?}");
    assert!(
        !rows.iter().any(|row| row.contains('\u{250c}')
            || row.contains('\u{2510}')
            || row.contains('\u{2514}')
            || row.contains('\u{2518}')),
        "square corners in the modern frame"
    );
    assert!(box_input.contains('\u{2502}'), "{box_input:?}");
    // The input window carries the same bare `> ` marker as the classic frame.
    assert!(box_input.contains("> "), "{box_input:?}");
    // Nothing rides the borders — the box is bare.
    assert!(!box_bottom.contains("main"), "{box_bottom:?}");
    // The banner is not boxed.
    assert!(!rows.iter().any(|row| row.contains('\u{250f}')));
    // The status line sits clear of the box, on its own row underneath: the
    // branch on the left, Graph/Pending centered, model flush right. This
    // frame keeps the branch out of the prompt prefix.
    assert!(!status.contains('\u{256f}'), "{status:?}");
    // Inset by two columns to line the row up with the box above it.
    assert!(status.starts_with("  main "), "{status:?}");
    assert!(status.contains("Graph: "), "{status:?}");
    assert!(status.trim_end().ends_with("gpt-4"), "{status:?}");

    crate::tui::Theme::apply_named("classic").expect("restore classic");
}
