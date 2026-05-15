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

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode},
};
use orangu::{
    config::{LlmConfiguration, default_client_config_path, load_client_configuration},
    llm::normalized_openai_endpoint,
    session::ChatSession,
    tools::ToolExecutor,
    tui::{HeaderStatus, help_text, render_screen},
};
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::ExitCode,
    time::{Duration, Instant},
};

const CLEAR_TERMINAL_SEQUENCE: &str = "\x1b[2J\x1b[H";
const VERSION: &str = env!("CARGO_PKG_VERSION");
const TERMINAL_TITLE: &str = "orangu";
const CTRL_C_EXIT_TIMEOUT: Duration = Duration::from_secs(2);
const CTRL_C_EXIT_MESSAGE: &str = "Press Ctrl+c again to quit";
const HISTORY_DIRECTORY: &str = "orangu";
const HISTORY_FILE: &str = "orangu.history";
const COMMANDS: &[&str] = &[
    "/help",
    "/connect",
    "/disconnect",
    "/reload",
    "/list-models",
    "/tools",
    "/model",
    "/clear",
    "/quit",
];

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    workspace: Option<PathBuf>,
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(()) => ExitCode::SUCCESS,
        Err(err) => {
            eprintln!("error: {err:#}");
            ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<()> {
    let _terminal_title_guard = TerminalTitleGuard::new(TERMINAL_TITLE);
    let args = Args::parse();
    let config_path = match args.config.or_else(default_client_config_path) {
        Some(path) => path,
        None => {
            return Err(anyhow!(
                "missing config file; pass --config or add ./orangu.conf or ~/orangu/orangu.conf"
            ));
        }
    };
    let config = load_client_configuration(&config_path)?;
    let workspace = args
        .workspace
        .unwrap_or(std::env::current_dir().context("failed to resolve current directory")?);
    let tools = ToolExecutor::new(&workspace);

    let model_names = sorted_model_names(&config.llms);
    let startup_model = config.default_model.clone();
    let startup_endpoint = config
        .llms
        .get(&startup_model)
        .ok_or_else(|| anyhow!("missing configured profile {}", startup_model))?
        .endpoint
        .clone();
    let mut active_model = startup_model.clone();
    let mut session = ChatSession::new(system_prompt(
        config
            .llms
            .get(&active_model)
            .ok_or_else(|| anyhow!("missing configured profile {}", active_model))?,
    ));
    let mut current_endpoint = Some(startup_endpoint.clone());
    let _raw_mode_guard = RawModeGuard::new()?;

    let mut transcript = Vec::new();
    let mut interrupt_state = InterruptState::default();
    let mut input_state = InputState::default();
    let history_path = history_file_path()?;
    let mut history = load_history(&history_path)?;
    let status_http_client = reqwest::Client::builder()
        .timeout(Duration::from_secs(3))
        .build()?;

    loop {
        let active_profile = config
            .llms
            .get(&active_model)
            .ok_or_else(|| anyhow!("missing configured profile {}", active_model))?;
        let header_status = probe_header_status(
            &status_http_client,
            tools.workspace(),
            &active_model,
            active_profile,
            current_endpoint.as_deref(),
        )
        .await;
        print_screen(
            &active_model,
            current_endpoint.as_deref().unwrap_or("(disconnected)"),
            tools.workspace(),
            header_status,
            &transcript,
            None,
            input_state.as_str(),
            input_state.cursor(),
        );
        std::io::stdout().flush()?;

        let input = match read_input(
            &mut input_state,
            &history,
            &workspace,
            &model_names,
            &mut interrupt_state,
            &mut transcript,
            &active_model,
            current_endpoint.as_deref().unwrap_or("(disconnected)"),
            header_status,
        )? {
            InputResult::Submitted(line) => line,
            InputResult::Quit => {
                print!("{CLEAR_TERMINAL_SEQUENCE}");
                std::io::stdout().flush()?;
                break;
            }
        };

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        history.push(trimmed.to_string());
        append_history_entry(&history_path, trimmed)?;
        push_transcript(&mut transcript, &format!("> {trimmed}"));
        print_screen(
            &active_model,
            current_endpoint.as_deref().unwrap_or("(disconnected)"),
            tools.workspace(),
            header_status,
            &transcript,
            None,
            input_state.as_str(),
            input_state.cursor(),
        );
        std::io::stdout().flush()?;

        match handle_command(
            trimmed,
            &mut active_model,
            &mut current_endpoint,
            &startup_model,
            &startup_endpoint,
            &config.llms,
            &mut session,
            &tools,
            &workspace,
        )? {
            CommandOutcome::Quit => {
                print!("{CLEAR_TERMINAL_SEQUENCE}");
                std::io::stdout().flush()?;
                break;
            }
            CommandOutcome::Cleared => {
                transcript.clear();
                continue;
            }
            CommandOutcome::Output(output) => {
                push_transcript(&mut transcript, &output);
                continue;
            }
            CommandOutcome::Unhandled => {}
        }

        let profile = config
            .llms
            .get(&active_model)
            .ok_or_else(|| anyhow!("unknown model profile '{active_model}'"))?;
        let Some(endpoint) = current_endpoint.as_deref() else {
            push_transcript(&mut transcript, "Error: Not connected to an LLM server");
            continue;
        };
        let mut prompt_profile = profile.clone();
        prompt_profile.endpoint = endpoint.to_string();
        match wait_for_response(
            &mut session,
            trimmed,
            &prompt_profile,
            &tools,
            &active_model,
            endpoint,
            tools.workspace(),
            header_status,
            &transcript,
        )
        .await
        {
            Ok(answer) => push_transcript(&mut transcript, &answer),
            Err(err) => push_transcript(&mut transcript, &format!("Error: {err:#}")),
        }
    }

    Ok(())
}

enum InterruptAction {
    Continue,
    Exit,
}

#[derive(Debug, Default)]
struct InterruptState {
    last_interrupt: Option<Instant>,
}

impl InterruptState {
    fn reset(&mut self) {
        self.last_interrupt = None;
    }

    fn handle_interrupt(&mut self, now: Instant) -> InterruptAction {
        if let Some(last_interrupt) = self.last_interrupt
            && now.duration_since(last_interrupt) <= CTRL_C_EXIT_TIMEOUT
        {
            self.last_interrupt = None;
            return InterruptAction::Exit;
        }

        self.last_interrupt = Some(now);
        InterruptAction::Continue
    }
}

struct TerminalTitleGuard;

impl TerminalTitleGuard {
    fn new(title: &str) -> Self {
        set_terminal_title(Some(title));
        Self
    }
}

impl Drop for TerminalTitleGuard {
    fn drop(&mut self) {
        set_terminal_title(None);
    }
}

fn set_terminal_title(title: Option<&str>) {
    match title {
        Some(title) => print!("\x1b]0;{title}\x07"),
        None => print!("\x1b]0;\x07"),
    }
}

struct RawModeGuard;

impl RawModeGuard {
    fn new() -> Result<Self> {
        enable_raw_mode()?;
        Ok(Self)
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
    }
}

#[derive(Default)]
struct InputState {
    buffer: String,
    cursor: usize,
    completion: Option<CompletionState>,
    history_index: Option<usize>,
    history_draft: String,
}

impl InputState {
    fn as_str(&self) -> &str {
        &self.buffer
    }

    fn cursor(&self) -> usize {
        self.cursor
    }

    fn clear(&mut self) {
        self.buffer.clear();
        self.cursor = 0;
        self.completion = None;
        self.history_index = None;
        self.history_draft.clear();
    }

    fn set_buffer(&mut self, buffer: String) {
        self.buffer = buffer;
        self.cursor = self.buffer.len();
        self.completion = None;
    }

    fn insert_char(&mut self, ch: char) {
        self.buffer.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
        self.completion = None;
    }

    fn insert_str(&mut self, text: &str) {
        self.buffer.insert_str(self.cursor, text);
        self.cursor += text.len();
        self.completion = None;
    }

    fn backspace(&mut self) {
        if let Some(previous) = previous_boundary(&self.buffer, self.cursor) {
            self.buffer.drain(previous..self.cursor);
            self.cursor = previous;
            self.completion = None;
        }
    }

    fn delete(&mut self) {
        if let Some(next) = next_boundary(&self.buffer, self.cursor) {
            self.buffer.drain(self.cursor..next);
            self.completion = None;
        }
    }

    fn move_left(&mut self) {
        if let Some(previous) = previous_boundary(&self.buffer, self.cursor) {
            self.cursor = previous;
            self.completion = None;
        }
    }

    fn move_right(&mut self) {
        if let Some(next) = next_boundary(&self.buffer, self.cursor) {
            self.cursor = next;
            self.completion = None;
        }
    }

    fn move_home(&mut self) {
        self.cursor = 0;
        self.completion = None;
    }

    fn move_end(&mut self) {
        self.cursor = self.buffer.len();
        self.completion = None;
    }

    fn kill_to_end(&mut self) {
        self.buffer.truncate(self.cursor);
        self.completion = None;
    }

    fn kill_to_start(&mut self) {
        self.buffer.drain(..self.cursor);
        self.cursor = 0;
        self.completion = None;
    }

    fn delete_prev_word(&mut self) {
        if self.cursor == 0 {
            return;
        }

        let mut start = self.cursor;
        while let Some(previous) = previous_boundary(&self.buffer, start) {
            if !self.buffer[previous..start]
                .chars()
                .all(char::is_whitespace)
            {
                start = previous;
                break;
            }
            start = previous;
            if start == 0 {
                break;
            }
        }

        while let Some(previous) = previous_boundary(&self.buffer, start) {
            if self.buffer[previous..start]
                .chars()
                .all(char::is_whitespace)
            {
                break;
            }
            start = previous;
            if start == 0 {
                break;
            }
        }

        self.buffer.drain(start..self.cursor);
        self.cursor = start;
        self.completion = None;
    }
}

struct CompletionState {
    start: usize,
    end: usize,
    original: String,
    candidates: Vec<String>,
    index: usize,
}

enum InputResult {
    Submitted(String),
    Quit,
}

fn read_input(
    input_state: &mut InputState,
    history: &[String],
    workspace: &std::path::Path,
    model_names: &[String],
    interrupt_state: &mut InterruptState,
    transcript: &mut Vec<String>,
    current_model: &str,
    endpoint: &str,
    header_status: HeaderStatus,
) -> Result<InputResult> {
    loop {
        let mut redraw = false;
        match event::read()? {
            Event::Paste(text) => {
                interrupt_state.reset();
                input_state.insert_str(&text);
                redraw = true;
            }
            Event::Key(KeyEvent {
                code,
                modifiers,
                kind,
                ..
            }) if kind == KeyEventKind::Press || kind == KeyEventKind::Repeat => {
                match (code, modifiers) {
                    (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                        match interrupt_state.handle_interrupt(Instant::now()) {
                            InterruptAction::Continue => {
                                push_transcript(transcript, CTRL_C_EXIT_MESSAGE);
                                input_state.clear();
                                return Ok(InputResult::Submitted(String::new()));
                            }
                            InterruptAction::Exit => return Ok(InputResult::Quit),
                        }
                    }
                    (KeyCode::Char('d'), KeyModifiers::CONTROL)
                        if input_state.as_str().is_empty() =>
                    {
                        return Ok(InputResult::Quit);
                    }
                    (KeyCode::Enter, KeyModifiers::NONE) => {
                        interrupt_state.reset();
                        let input = input_state.buffer.clone();
                        input_state.clear();
                        return Ok(InputResult::Submitted(input));
                    }
                    (KeyCode::Backspace, _) => {
                        interrupt_state.reset();
                        input_state.backspace();
                        redraw = true;
                    }
                    (KeyCode::Delete, _) | (KeyCode::Char('d'), KeyModifiers::CONTROL) => {
                        interrupt_state.reset();
                        input_state.delete();
                        redraw = true;
                    }
                    (KeyCode::Left, _) => {
                        interrupt_state.reset();
                        input_state.move_left();
                        redraw = true;
                    }
                    (KeyCode::Right, _) => {
                        interrupt_state.reset();
                        input_state.move_right();
                        redraw = true;
                    }
                    (KeyCode::Home, _) | (KeyCode::Char('a'), KeyModifiers::CONTROL) => {
                        interrupt_state.reset();
                        input_state.move_home();
                        redraw = true;
                    }
                    (KeyCode::End, _) | (KeyCode::Char('e'), KeyModifiers::CONTROL) => {
                        interrupt_state.reset();
                        input_state.move_end();
                        redraw = true;
                    }
                    (KeyCode::Char('k'), KeyModifiers::CONTROL) => {
                        interrupt_state.reset();
                        input_state.kill_to_end();
                        redraw = true;
                    }
                    (KeyCode::Char('u'), KeyModifiers::CONTROL) => {
                        interrupt_state.reset();
                        input_state.kill_to_start();
                        redraw = true;
                    }
                    (KeyCode::Char('w'), KeyModifiers::CONTROL) => {
                        interrupt_state.reset();
                        input_state.delete_prev_word();
                        redraw = true;
                    }
                    (KeyCode::Up, _) => {
                        interrupt_state.reset();
                        history_previous(input_state, history);
                        redraw = true;
                    }
                    (KeyCode::Down, _) => {
                        interrupt_state.reset();
                        history_next(input_state, history);
                        redraw = true;
                    }
                    (KeyCode::Tab, _) => {
                        interrupt_state.reset();
                        apply_completion(input_state, workspace, model_names);
                        redraw = true;
                    }
                    (KeyCode::Char(ch), KeyModifiers::NONE | KeyModifiers::SHIFT) => {
                        interrupt_state.reset();
                        input_state.insert_char(ch);
                        redraw = true;
                    }
                    _ => {}
                }
            }
            _ => {}
        }

        if redraw {
            print_screen(
                current_model,
                endpoint,
                workspace,
                header_status,
                transcript,
                None,
                input_state.as_str(),
                input_state.cursor(),
            );
            std::io::stdout().flush()?;
        }
    }
}

fn history_previous(input_state: &mut InputState, history: &[String]) {
    if history.is_empty() {
        return;
    }

    let new_index = match input_state.history_index {
        Some(0) => 0,
        Some(index) => index.saturating_sub(1),
        None => {
            input_state.history_draft = input_state.buffer.clone();
            history.len() - 1
        }
    };

    input_state.history_index = Some(new_index);
    input_state.set_buffer(history[new_index].clone());
}

fn history_next(input_state: &mut InputState, history: &[String]) {
    let Some(index) = input_state.history_index else {
        return;
    };

    if index + 1 >= history.len() {
        input_state.history_index = None;
        let draft = std::mem::take(&mut input_state.history_draft);
        input_state.set_buffer(draft);
        return;
    }

    let new_index = index + 1;
    input_state.history_index = Some(new_index);
    input_state.set_buffer(history[new_index].clone());
}

fn apply_completion(
    input_state: &mut InputState,
    workspace: &std::path::Path,
    model_names: &[String],
) {
    if let Some(state) = input_state.completion.as_mut()
        && !state.candidates.is_empty()
    {
        state.index = (state.index + 1) % state.candidates.len();
        let start = state.start;
        let end = state.end;
        let original = state.original.clone();
        let candidate = state.candidates[state.index].clone();
        apply_completion_candidate(input_state, start, end, &original, &candidate);
        return;
    }

    let Some((start, end, candidates)) = completion_candidates(
        input_state.as_str(),
        input_state.cursor(),
        workspace,
        model_names,
    ) else {
        return;
    };
    if candidates.is_empty() {
        return;
    }

    let original = input_state.buffer.clone();
    let candidate = candidates[0].clone();
    apply_completion_candidate(input_state, start, end, &original, &candidate);
    input_state.completion = Some(CompletionState {
        start,
        end,
        original,
        candidates,
        index: 0,
    });
}

fn apply_completion_candidate(
    input_state: &mut InputState,
    start: usize,
    end: usize,
    original: &str,
    candidate: &str,
) {
    let mut buffer = String::new();
    buffer.push_str(&original[..start]);
    buffer.push_str(candidate);
    buffer.push_str(&original[end..]);
    input_state.buffer = buffer;
    input_state.cursor = start + candidate.len();
}

fn completion_candidates(
    input: &str,
    cursor: usize,
    workspace: &std::path::Path,
    model_names: &[String],
) -> Option<(usize, usize, Vec<String>)> {
    let cursor = cursor.min(input.len());
    let prefix = &input[..cursor];

    if let Some(model_prefix) = prefix.strip_prefix("/model ") {
        return Some((
            7,
            cursor,
            model_names
                .iter()
                .filter(|model| model.starts_with(model_prefix))
                .cloned()
                .collect(),
        ));
    }

    if prefix.starts_with('/') {
        return Some((
            0,
            cursor,
            COMMANDS
                .iter()
                .filter(|command| command.starts_with(prefix))
                .map(|command| (*command).to_string())
                .collect(),
        ));
    }

    let start = prefix
        .char_indices()
        .rev()
        .find(|(_, ch)| ch.is_whitespace())
        .map(|(index, ch)| index + ch.len_utf8())
        .unwrap_or(0);
    let token = &prefix[start..];
    Some((start, cursor, file_completion_candidates(token, workspace)))
}

fn file_completion_candidates(token: &str, workspace: &std::path::Path) -> Vec<String> {
    let (directory, prefix) = match token.rsplit_once('/') {
        Some((directory, prefix)) => (directory, prefix),
        None => ("", token),
    };
    let search_dir = if directory.is_empty() {
        workspace.to_path_buf()
    } else {
        workspace.join(directory)
    };

    let Ok(entries) = fs::read_dir(search_dir) else {
        return Vec::new();
    };

    let mut matches = entries
        .flatten()
        .filter_map(|entry| {
            let file_name = entry.file_name().to_string_lossy().to_string();
            if !file_name.starts_with(prefix) {
                return None;
            }

            let suffix = if entry.file_type().ok()?.is_dir() {
                "/"
            } else {
                ""
            };
            Some(if directory.is_empty() {
                format!("{file_name}{suffix}")
            } else {
                format!("{directory}/{file_name}{suffix}")
            })
        })
        .collect::<Vec<_>>();
    matches.sort();
    matches
}

fn previous_boundary(input: &str, cursor: usize) -> Option<usize> {
    input[..cursor.min(input.len())]
        .char_indices()
        .last()
        .map(|(index, _)| index)
}

fn next_boundary(input: &str, cursor: usize) -> Option<usize> {
    let cursor = cursor.min(input.len());
    input[cursor..]
        .char_indices()
        .nth(1)
        .map(|(index, _)| cursor + index)
        .or_else(|| (cursor < input.len()).then_some(input.len()))
}

enum CommandOutcome {
    Unhandled,
    Output(String),
    Cleared,
    Quit,
}

fn handle_command(
    input: &str,
    active_model: &mut String,
    current_endpoint: &mut Option<String>,
    startup_model: &str,
    startup_endpoint: &str,
    llms: &HashMap<String, LlmConfiguration>,
    session: &mut ChatSession,
    tools: &ToolExecutor,
    workspace: &std::path::Path,
) -> Result<CommandOutcome> {
    if input == "/help" {
        return Ok(CommandOutcome::Output(help_text().to_string()));
    }
    if input == "/list-models" {
        return Ok(CommandOutcome::Output(format_models(llms)));
    }
    if input == "/tools" {
        return Ok(CommandOutcome::Output(format_tools(tools)));
    }
    if input == "/clear" {
        let prompt = system_prompt(
            llms.get(active_model)
                .ok_or_else(|| anyhow!("unknown model profile '{active_model}'"))?,
        );
        session.clear(prompt);
        return Ok(CommandOutcome::Cleared);
    }
    if input == "/model" {
        return Ok(CommandOutcome::Output(
            "Use /list-models to see configured profiles".to_string(),
        ));
    }
    if let Some(name) = input.strip_prefix("/model ") {
        let name = name.trim();
        if !llms.contains_key(name) {
            return Ok(CommandOutcome::Output(format!(
                "Unknown model profile '{name}'. Available: {}",
                sorted_model_names(llms).join(", ")
            )));
        }
        *active_model = name.to_string();
        session.set_system_prompt(system_prompt(&llms[name]));
        return Ok(CommandOutcome::Output(format!(
            "Switched to model profile '{name}'"
        )));
    }
    if input == "/connect" {
        let endpoint = llms
            .get(active_model)
            .ok_or_else(|| anyhow!("unknown model profile '{active_model}'"))?
            .endpoint
            .clone();
        *current_endpoint = Some(endpoint.clone());
        return Ok(CommandOutcome::Output(format!("Connected to '{endpoint}'")));
    }
    if let Some(endpoint) = input.strip_prefix("/connect ") {
        let endpoint = endpoint.trim();
        if endpoint.is_empty() {
            return Ok(CommandOutcome::Output("Usage: /connect [url]".to_string()));
        }
        *current_endpoint = Some(endpoint.to_string());
        return Ok(CommandOutcome::Output(format!("Connected to '{endpoint}'")));
    }
    if input == "/disconnect" {
        *current_endpoint = None;
        return Ok(CommandOutcome::Output(
            "Disconnected from the current server target".to_string(),
        ));
    }
    if input == "/reload" {
        *active_model = startup_model.to_string();
        *current_endpoint = Some(startup_endpoint.to_string());
        let prompt = system_prompt(
            llms.get(startup_model)
                .ok_or_else(|| anyhow!("unknown model profile '{startup_model}'"))?,
        );
        session.clear(prompt);
        return Ok(CommandOutcome::Output(format!(
            "Reloaded startup configuration: model '{startup_model}', server '{startup_endpoint}'"
        )));
    }
    if input == "/quit" {
        return Ok(CommandOutcome::Quit);
    }

    let _ = workspace;
    Ok(CommandOutcome::Unhandled)
}

fn system_prompt(profile: &LlmConfiguration) -> &str {
    if profile.system_prompt.is_empty() {
        "You are Orangu, a coding environment assistant connected to a local workspace. Use the available local tools to inspect files, edit files on disk, fetch external URLs for knowledge, and run shell commands when needed. Be precise, explain what you changed, and surface tool failures explicitly."
    } else {
        &profile.system_prompt
    }
}

fn sorted_model_names(llms: &HashMap<String, LlmConfiguration>) -> Vec<String> {
    let mut names = llms.keys().cloned().collect::<Vec<_>>();
    names.sort();
    names
}

fn print_screen(
    current_model: &str,
    endpoint: &str,
    workspace: &std::path::Path,
    header_status: HeaderStatus,
    transcript: &[String],
    pending_line: Option<&str>,
    input: &str,
    cursor: usize,
) {
    print!("{CLEAR_TERMINAL_SEQUENCE}");
    print!(
        "{}",
        render_screen(
            VERSION,
            current_model,
            endpoint,
            workspace,
            header_status,
            transcript,
            pending_line,
            input,
            cursor
        )
    );
}

async fn wait_for_response(
    session: &mut ChatSession,
    user_input: &str,
    profile: &LlmConfiguration,
    tools: &ToolExecutor,
    current_model: &str,
    endpoint: &str,
    workspace: &std::path::Path,
    header_status: HeaderStatus,
    transcript: &[String],
) -> Result<String> {
    let mut prompt_future = Box::pin(session.prompt(user_input, profile, tools));
    let mut interval = tokio::time::interval(Duration::from_millis(500));
    let mut thinking_visible = true;

    print_screen(
        current_model,
        endpoint,
        workspace,
        header_status,
        transcript,
        Some("Thinking"),
        "",
        0,
    );
    std::io::stdout().flush()?;

    loop {
        tokio::select! {
            result = &mut prompt_future => return result,
            _ = interval.tick() => {
                let pending_line = if thinking_visible {
                    Some("Thinking")
                } else {
                    Some("")
                };
                print_screen(
                    current_model,
                    endpoint,
                    workspace,
                    header_status,
                    transcript,
                    pending_line,
                    "",
                    0,
                );
                std::io::stdout().flush()?;
                thinking_visible = !thinking_visible;
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    #[serde(default)]
    data: Vec<ModelEntry>,
    #[serde(default)]
    models: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    #[serde(default)]
    id: String,
    #[serde(default)]
    model: String,
    #[serde(default)]
    name: String,
}

async fn probe_header_status(
    http_client: &reqwest::Client,
    workspace: &Path,
    active_model: &str,
    profile: &LlmConfiguration,
    endpoint: Option<&str>,
) -> HeaderStatus {
    let workspace_ok = workspace.exists();
    let mut server_ok = false;
    let mut model_ok = false;

    if let Some(endpoint) = endpoint {
        let models_url = format!("{}/v1/models", normalized_openai_endpoint(endpoint));
        if let Ok(response) = http_client.get(&models_url).send().await
            && response.status().is_success()
        {
            server_ok = true;
            if let Ok(models) = response.json::<ModelsResponse>().await {
                model_ok = models.data.iter().chain(models.models.iter()).any(|entry| {
                    entry.id == profile.model
                        || entry.model == profile.model
                        || entry.name == profile.model
                        || entry.id == active_model
                        || entry.model == active_model
                        || entry.name == active_model
                });
            }
        }
    }

    HeaderStatus {
        workspace_ok,
        server_ok,
        model_ok,
    }
}

fn history_file_path() -> Result<PathBuf> {
    let home = home::home_dir().ok_or_else(|| anyhow!("failed to resolve home directory"))?;
    Ok(home.join(HISTORY_DIRECTORY).join(HISTORY_FILE))
}

fn load_history(path: &Path) -> Result<Vec<String>> {
    match fs::read_to_string(path) {
        Ok(content) => Ok(content
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToOwned::to_owned)
            .collect()),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(err) => {
            Err(err).with_context(|| format!("failed to read history file {}", path.display()))
        }
    }
}

fn append_history_entry(path: &Path, entry: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create history directory {}", parent.display()))?;
    }

    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open history file {}", path.display()))?;
    writeln!(file, "{entry}")
        .with_context(|| format!("failed to write history file {}", path.display()))
}

fn push_transcript(transcript: &mut Vec<String>, text: &str) {
    transcript.extend(text.lines().map(ToOwned::to_owned));
}

fn format_models(llms: &HashMap<String, LlmConfiguration>) -> String {
    let mut names = sorted_model_names(llms);
    let mut lines = Vec::with_capacity(names.len());
    for name in names.drain(..) {
        if let Some(llm) = llms.get(&name) {
            lines.push(format!("- {}: {} ({})", name, llm.model, llm.provider));
        }
    }
    lines.join("\n")
}

fn format_tools(tools: &ToolExecutor) -> String {
    tools
        .definitions()
        .into_iter()
        .map(|tool| format!("- {}: {}", tool.function.name, tool.function.description))
        .collect::<Vec<_>>()
        .join("\n")
}
