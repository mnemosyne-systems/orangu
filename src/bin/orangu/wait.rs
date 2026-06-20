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

pub(crate) const WAIT_LOOP_POLL_INTERVAL: std::time::Duration =
    std::time::Duration::from_millis(50);
pub(crate) const THINKING_FRAME_INTERVAL: std::time::Duration =
    std::time::Duration::from_millis(120);

pub(crate) async fn wait_for_response(
    session: &mut ChatSession,
    user_input: &str,
    profile: &LlmConfiguration,
    tools: &ToolExecutor,
    wait_context: WaitContext<'_>,
) -> Result<WaitResult> {
    let WaitContext {
        mut render,
        history,
        history_path,
        server_names,
        available_models,
        interrupt_state,
        output_state,
        input_state,
        pending_commands,
        thinking_quote,
        viewport,
        skills,
    } = wait_context;
    let streamed_state = Arc::new(Mutex::new(StreamRenderState::default()));
    let prompt_output = Arc::clone(&streamed_state);
    let prompt_metrics = Arc::clone(&streamed_state);
    let prompt_tool_running = Arc::clone(&streamed_state);
    let tokenizer = cl100k_base().ok();
    let mut prompt_future = Box::pin(session.prompt(
        user_input,
        profile,
        tools,
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
        move |running| {
            if let Ok(mut state) = prompt_tool_running.lock() {
                state.tool_running_since = if running {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
            }
        },
    ));
    let mut interval = tokio::time::interval(WAIT_LOOP_POLL_INTERVAL);
    let mut thinking_frame = 0usize;
    let thinking_started = std::time::Instant::now();
    let mut last_rendered_output = String::new();
    let mut last_rendered_metrics = StreamMetrics::default();
    let mut last_tool_was_running = false;
    let mut escape_cancel_state = EscapeCancelState::default();
    let initial_status = Some(render_thinking_status(
        thinking_frame,
        thinking_started.elapsed(),
    ));
    let quote_line = thinking_quote.map(|q| format!("\x1b[2m{q}\x1b[0m"));

    print_screen(
        render,
        ScreenState {
            transcript: output_state.lines(),
            scroll_offset: output_state.scroll_offset(),
            left_status: initial_status,
            pending_count: pending_commands.len(),
            pending_line: quote_line.as_deref(),
            input: input_state.as_str(),
            cursor: input_state.cursor(),
            ghost_index: input_state.ghost_index,
        },
    );
    std::io::stdout().flush()?;

    loop {
        tokio::select! {
            result = &mut prompt_future => {
                let response = match result {
                    Ok(response) => response,
                    Err(error) => {
                        let partial = streamed_state
                            .lock()
                            .map(|state| state.output.clone())
                            .unwrap_or_default();
                        return Ok(WaitResult::Failed { partial, error });
                    }
                };
                let final_state = streamed_state
                    .lock()
                    .map(|state| state.clone())
                    .unwrap_or_default();
                if let Some(pending_line) = final_pending_line(&final_state.output, &response)
                    .map(|line| render_markdown_for_console(&line))
                {
                    print_screen(
                        render,
                        ScreenState {
                            transcript: output_state.lines(),
                            scroll_offset: output_state.scroll_offset(),
                            left_status: None,
                            pending_count: pending_commands.len(),
                            pending_line: Some(pending_line.as_str()),
                            input: input_state.as_str(),
                            cursor: input_state.cursor(),
            ghost_index: input_state.ghost_index,
                        },
                    );
                    std::io::stdout().flush()?;
                }
                return Ok(WaitResult::Response(response));
            }
            _ = interval.tick() => {
                let elapsed = thinking_started.elapsed();
                let next_frame = (elapsed.as_millis() / THINKING_FRAME_INTERVAL.as_millis()) as usize;
                let mut redraw = next_frame != thinking_frame;
                thinking_frame = next_frame;
                let current_state = streamed_state
                    .lock()
                    .map(|state| state.clone())
                    .unwrap_or_default();
                let current_streamed_output = current_state.output;
                let current_stream_metrics = current_state.metrics;
                let current_tool_running_since = current_state.tool_running_since;
                redraw |= current_streamed_output != last_rendered_output;
                redraw |= current_stream_metrics != last_rendered_metrics;
                redraw |= current_tool_running_since.is_some() != last_tool_was_running;

                while event::poll(std::time::Duration::ZERO)? {
                    let event = event::read()?;
                    if is_wait_cancel_escape(&event) {
                        if escape_cancel_state.handle_escape(std::time::Instant::now()) {
                            let partial_output = streamed_state
                                .lock()
                                .map(|state| state.output.clone())
                                .unwrap_or_default();
                            drop(prompt_future);
                            return Ok(WaitResult::Cancelled(partial_output));
                        }
                        continue;
                    }
                    escape_cancel_state.reset();
                    let result = handle_input_event(
                        event,
                        input_state,
                        interrupt_state,
                        output_state,
                        viewport,
                        InputContext {
                            history,
                            workspace: render.workspace,
                            server_names,
                            available_models,
                            render,
                            skills,
                        },
                    );
                    render.actual_width = viewport.actual_width;
                    render.actual_height = viewport.actual_height;
                    render.x_offset = viewport.x_offset;

                    if let Some(outcome) = result.outcome {
                        match outcome {
                            InputResult::Submitted(line) => {
                                match parse_local_command(line.trim()) {
                                    Some(LocalCommand::PendingList) => {
                                        output_state
                                            .push_text(&format_pending_list(pending_commands));
                                        redraw = true;
                                    }
                                    Some(LocalCommand::PendingDelete(Some(index))) => {
                                        apply_pending_delete(
                                            index,
                                            pending_commands,
                                            output_state,
                                        );
                                        redraw = true;
                                    }
                                    Some(LocalCommand::PendingDelete(None)) => {
                                        output_state.push_text(
                                            "Usage: /pending delete <number>. Use /pending to list.",
                                        );
                                        redraw = true;
                                    }
                                    _ => {
                                        let had_pending = pending_commands.len();
                                        let _ = prepare_submitted_input(
                                            &line,
                                            history,
                                            history_path,
                                            output_state,
                                            Some(pending_commands),
                                        )?;
                                        redraw = redraw
                                            || pending_commands.len() != had_pending
                                            || !line.trim().is_empty();
                                    }
                                }
                            }
                            InputResult::Refresh => {}
                            InputResult::Quit => return Ok(WaitResult::Quit),
                        }
                    }
                    redraw |= result.redraw;
                }

                if redraw {
                    last_rendered_output = current_streamed_output;
                    last_rendered_metrics = current_stream_metrics;
                    last_tool_was_running = current_tool_running_since.is_some();
                    let left_status = render_left_status(
                        profile,
                        &last_rendered_output,
                        &last_rendered_metrics,
                        current_tool_running_since,
                        elapsed,
                        thinking_frame,
                        tokenizer.as_ref(),
                    );
                    let pending_line = if last_rendered_output.is_empty() {
                        quote_line.clone().unwrap_or_default()
                    } else {
                        render_markdown_for_console(&last_rendered_output)
                    };
                    print_screen(
                        render,
                        ScreenState {
                            transcript: output_state.lines(),
                            scroll_offset: output_state.scroll_offset(),
                            left_status,
                            pending_count: pending_commands.len(),
                            pending_line: Some(pending_line.as_str()),
                            input: input_state.as_str(),
                            cursor: input_state.cursor(),
            ghost_index: input_state.ghost_index,
                        },
                    );
                    std::io::stdout().flush()?;
                }
            }
        }
    }
}

pub(crate) async fn wait_for_local_command(
    wait_context: WaitContext<'_>,
    mut handle: tokio::task::JoinHandle<anyhow::Result<String>>,
) -> anyhow::Result<anyhow::Result<String>> {
    let WaitContext {
        mut render,
        history,
        history_path: _,
        server_names,
        available_models,
        interrupt_state,
        output_state,
        input_state,
        pending_commands,
        thinking_quote: _,
        viewport,
        skills,
    } = wait_context;
    let started = std::time::Instant::now();
    let mut interval = tokio::time::interval(WAIT_LOOP_POLL_INTERVAL);
    let mut frame = 0usize;
    loop {
        tokio::select! {
            result = &mut handle => {
                return Ok(result?);
            }
            _ = interval.tick() => {
                let elapsed = started.elapsed();
                let next_frame = (elapsed.as_millis() / THINKING_FRAME_INTERVAL.as_millis()) as usize;
                if next_frame != frame {
                    frame = next_frame;
                }
                while event::poll(std::time::Duration::ZERO)? {
                    handle_input_event(
                        event::read()?,
                        input_state,
                        interrupt_state,
                        output_state,
                        viewport,
                        InputContext {
                            history,
                            workspace: render.workspace,
                            server_names,
                            available_models,
                            render,
                            skills,
                        },
                    );
                    render.actual_width = viewport.actual_width;
                    render.actual_height = viewport.actual_height;
                    render.x_offset = viewport.x_offset;
                }
                let left_status = Some(render_tool_running_status(frame, elapsed));
                print_screen(
                    render,
                    ScreenState {
                        transcript: output_state.lines(),
                        scroll_offset: output_state.scroll_offset(),
                        left_status,
                        pending_count: pending_commands.len(),
                        pending_line: None,
                        input: input_state.as_str(),
                        cursor: input_state.cursor(),
            ghost_index: input_state.ghost_index,
                    },
                );
                std::io::stdout().flush()?;
            }
        }
    }
}

/// Wait for a streaming command, draining its line channel into the output
/// window as lines arrive so the build log appears live instead of all at once.
pub(crate) async fn wait_for_streaming_command(
    wait_context: WaitContext<'_>,
    mut handle: tokio::task::JoinHandle<anyhow::Result<()>>,
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<String>,
) -> anyhow::Result<anyhow::Result<()>> {
    let WaitContext {
        mut render,
        history,
        history_path: _,
        server_names,
        available_models,
        interrupt_state,
        output_state,
        input_state,
        pending_commands,
        thinking_quote: _,
        viewport,
        skills,
    } = wait_context;
    let started = std::time::Instant::now();
    let mut interval = tokio::time::interval(WAIT_LOOP_POLL_INTERVAL);
    let mut frame = 0usize;
    loop {
        tokio::select! {
            result = &mut handle => {
                // Drain any lines still buffered before the task finished.
                while let Ok(line) = rx.try_recv() {
                    output_state.push_text(&line);
                }
                return Ok(result?);
            }
            _ = interval.tick() => {
                let elapsed = started.elapsed();
                let next_frame = (elapsed.as_millis() / THINKING_FRAME_INTERVAL.as_millis()) as usize;
                if next_frame != frame {
                    frame = next_frame;
                }
                while let Ok(line) = rx.try_recv() {
                    output_state.push_text(&line);
                }
                while event::poll(std::time::Duration::ZERO)? {
                    handle_input_event(
                        event::read()?,
                        input_state,
                        interrupt_state,
                        output_state,
                        viewport,
                        InputContext {
                            history,
                            workspace: render.workspace,
                            server_names,
                            available_models,
                            render,
                            skills,
                        },
                    );
                    render.actual_width = viewport.actual_width;
                    render.actual_height = viewport.actual_height;
                    render.x_offset = viewport.x_offset;
                }
                let left_status = Some(render_tool_running_status(frame, elapsed));
                print_screen(
                    render,
                    ScreenState {
                        transcript: output_state.lines(),
                        scroll_offset: output_state.scroll_offset(),
                        left_status,
                        pending_count: pending_commands.len(),
                        pending_line: None,
                        input: input_state.as_str(),
                        cursor: input_state.cursor(),
            ghost_index: input_state.ghost_index,
                    },
                );
                std::io::stdout().flush()?;
            }
        }
    }
}

pub(crate) fn render_left_status(
    profile: &LlmConfiguration,
    rendered_output: &str,
    metrics: &StreamMetrics,
    tool_running_since: Option<std::time::Instant>,
    elapsed: std::time::Duration,
    frame: usize,
    tokenizer: Option<&tiktoken_rs::CoreBPE>,
) -> Option<orangu::tui::StatusFragment> {
    if let Some(tool_start) = tool_running_since {
        return Some(render_tool_running_status(frame, tool_start.elapsed()));
    }

    if rendered_output.is_empty() {
        return Some(render_thinking_status(frame, elapsed));
    }

    if profile.provider.eq_ignore_ascii_case("llama.cpp")
        && let Some(rate) = metrics
            .predicted_per_second
            .filter(|rate| *rate > 0.0 && !rendered_output.is_empty())
    {
        return Some(render_working_status(frame, rate, elapsed));
    }

    tokenizer.and_then(|tokenizer| {
        let token_count = tokenizer.encode_with_special_tokens(rendered_output).len();
        let elapsed_secs = elapsed.as_secs_f64();
        (token_count > 0 && elapsed_secs > 0.0).then(|| {
            orangu::tui::StatusFragment::plain(format!(
                "{:.1}t/s",
                token_count as f64 / elapsed_secs
            ))
        })
    })
}

pub(crate) fn is_wait_cancel_escape(event: &Event) -> bool {
    matches!(
        event,
        Event::Key(KeyEvent {
            code: KeyCode::Esc,
            kind: KeyEventKind::Press,
            ..
        })
    )
}

/// Render the queue of pending commands for the output window: a header
/// followed by each command on its own 1-based numbered line, or a short notice
/// when the queue is empty. The numbering matches the index
/// [`apply_pending_delete`] expects.
pub(crate) fn format_pending_list(pending: &VecDeque<String>) -> String {
    if pending.is_empty() {
        "No pending commands.".to_string()
    } else {
        let mut lines = vec!["Pending commands:".to_string()];
        for (i, cmd) in pending.iter().enumerate() {
            lines.push(format!("  {}. {}", i + 1, cmd));
        }
        lines.join("\n")
    }
}

/// Remove the pending command at `index` (1-based, as shown by
/// [`format_pending_list`]) and report the result to the output window. An
/// out-of-range index (including `0`) removes nothing and reports that no such
/// command exists.
pub(crate) fn apply_pending_delete(
    index: usize,
    pending: &mut VecDeque<String>,
    output_state: &mut OutputState,
) {
    if index == 0 || index > pending.len() {
        output_state.push_text(&format!(
            "No pending command at index {index}. Use /pending to list."
        ));
    } else {
        let removed = pending.remove(index - 1).expect("index validated");
        output_state.push_text(&format!("Removed: {removed}"));
    }
}

pub(crate) fn final_pending_line(streamed_output: &str, response: &str) -> Option<String> {
    if !streamed_output.is_empty() {
        Some(streamed_output.to_string())
    } else if !response.is_empty() {
        Some(response.to_string())
    } else {
        None
    }
}

pub(crate) fn request_cancelled_message() -> String {
    format!(
        "{}Request cancelled.{}",
        render::ANSI_FG_LIGHT_RED,
        render::ANSI_RESET
    )
}

pub(crate) fn preserve_cancelled_output(output_state: &mut OutputState, partial_output: &str) {
    if !partial_output.is_empty() {
        output_state.push_markdown(partial_output);
    }
    output_state.push_text(&request_cancelled_message());
}

#[cfg(test)]
mod tests {
    use super::*;
    use orangu::llm::StreamPromptProgress;
    use orangu::tui::TranscriptLine;
    use std::time::Duration;

    #[test]
    fn final_pending_line_keeps_visible_output() {
        assert_eq!(
            final_pending_line("streamed reply", "final reply").as_deref(),
            Some("streamed reply")
        );
        assert_eq!(
            final_pending_line("", "final reply").as_deref(),
            Some("final reply")
        );
        assert_eq!(final_pending_line("", ""), None);
    }

    #[test]
    fn cancelled_output_preserves_partial_reply_and_uses_light_red_notice() {
        let mut output_state = OutputState::default();

        preserve_cancelled_output(&mut output_state, "partial reply");

        assert_eq!(
            output_state.lines(),
            &[
                TranscriptLine::Plain("partial reply".to_string()),
                TranscriptLine::Plain(request_cancelled_message()),
            ]
        );
    }

    #[test]
    fn format_pending_list_numbers_commands_from_one() {
        assert_eq!(
            format_pending_list(&VecDeque::new()),
            "No pending commands."
        );

        let pending = VecDeque::from(vec!["first".to_string(), "second".to_string()]);
        assert_eq!(
            format_pending_list(&pending),
            "Pending commands:\n  1. first\n  2. second"
        );
    }

    #[test]
    fn apply_pending_delete_removes_by_one_based_index() {
        let mut pending = VecDeque::from(vec![
            "first".to_string(),
            "second".to_string(),
            "third".to_string(),
        ]);
        let mut output_state = OutputState::default();

        // The 1-based index matches the displayed numbering: index 2 drops
        // "second", leaving the rest in order.
        apply_pending_delete(2, &mut pending, &mut output_state);
        assert_eq!(
            pending,
            VecDeque::from(vec!["first".to_string(), "third".to_string()])
        );
        assert_eq!(
            output_state.lines(),
            &[TranscriptLine::Plain("Removed: second".to_string())]
        );

        // Out-of-range indices (including 0) remove nothing.
        for index in [0, 3] {
            let before = pending.clone();
            apply_pending_delete(index, &mut pending, &mut output_state);
            assert_eq!(pending, before, "index {index} should not remove anything");
        }
    }

    #[test]
    fn wait_cancel_escape_only_matches_escape_press() {
        assert!(is_wait_cancel_escape(&Event::Key(KeyEvent::new(
            KeyCode::Esc,
            KeyModifiers::NONE
        ))));
        assert!(!is_wait_cancel_escape(&Event::Key(
            KeyEvent::new_with_kind(KeyCode::Esc, KeyModifiers::NONE, KeyEventKind::Repeat)
        )));
        assert!(!is_wait_cancel_escape(&Event::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE
        ))));
    }

    #[test]
    fn llama_cpp_left_status_prefers_native_metrics() {
        let profile = LlmConfiguration {
            provider: "llama.cpp".to_string(),
            model: "model".to_string(),
            endpoint: "http://localhost:8080/v1".to_string(),
            api_key: None,
            request_timeout_seconds: 30,
            max_tool_rounds: 10,
            review_max_tokens: 512,
            code_max_tokens: 0,
            system_prompt: String::new(),
        };

        let thinking = render_left_status(
            &profile,
            "",
            &StreamMetrics {
                prompt_progress: Some(StreamPromptProgress {
                    total: 100,
                    cache: 20,
                    processed: 60,
                    time_ms: 2_000,
                }),
                prompt_per_second: Some(15.0),
                predicted_per_second: None,
            },
            None,
            Duration::from_secs(2),
            0,
            None,
        )
        .expect("thinking status");
        for ch in "Thinking".chars() {
            assert!(thinking.rendered.contains(ch));
        }
        assert!(thinking.rendered.contains("(2s)"));
        assert_eq!(thinking.visible_width, "Thinking (2s)".chars().count());

        let working = render_left_status(
            &profile,
            "hello",
            &StreamMetrics {
                prompt_progress: None,
                prompt_per_second: Some(15.0),
                predicted_per_second: Some(42.5),
            },
            None,
            Duration::from_secs(2),
            1,
            None,
        )
        .expect("working status");
        for ch in "Working".chars() {
            assert!(working.rendered.contains(ch));
        }
        assert!(working.rendered.contains("42.5 t/s"));
        assert!(working.rendered.contains("(2s)"));
        assert_eq!(
            working.visible_width,
            "Working @ 42.5 t/s (2s)".chars().count()
        );
    }
}
