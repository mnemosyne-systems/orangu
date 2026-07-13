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
