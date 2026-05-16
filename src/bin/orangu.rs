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
use ignore::gitignore::{Gitignore, GitignoreBuilder};
use orangu::{
    config::{LlmConfiguration, default_client_config_path, load_client_configuration},
    llm::normalized_openai_endpoint,
    session::ChatSession,
    tools::{ToolExecutor, resolve_workspace_path},
    tui::{HeaderStatus, help_text, render_screen, render_thinking_frame},
};
use serde::Deserialize;
use std::{
    collections::HashMap,
    fs,
    io::Write,
    path::{Component, Path, PathBuf},
    process::ExitCode,
    time::{Duration, Instant},
};
use walkdir::WalkDir;

const CLEAR_TERMINAL_SEQUENCE: &str = "\x1b[2J\x1b[H";
const VERSION: &str = env!("CARGO_PKG_VERSION");
const TERMINAL_TITLE: &str = "orangu";
const CTRL_C_EXIT_TIMEOUT: Duration = Duration::from_secs(2);
const CTRL_C_EXIT_MESSAGE: &str = "Press Ctrl+c again to quit";
const THINKING_FRAME_INTERVAL: Duration = Duration::from_millis(120);
const HISTORY_DIRECTORY: &str = ".orangu";
const HISTORY_FILE: &str = "orangu.history";
const COMMANDS: &[&str] = &[
    "/help",
    "/connect",
    "/disconnect",
    "/reload",
    "/list-models",
    "/tools",
    "/model",
    "/open_file",
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
                "Missing config file; pass --config or add ./orangu.conf or ~/.orangu/orangu.conf"
            ));
        }
    };
    let config = load_client_configuration(&config_path)?;
    let workspace = resolve_workspace_root(args.workspace)?;
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
        let prompt_branch = workspace_branch_name(tools.workspace());
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
            prompt_branch.as_deref(),
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
            prompt_branch.as_deref(),
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
        if trimmed.starts_with('\\') {
            continue;
        }

        history.push(trimmed.to_string());
        append_history_entry(&history_path, trimmed)?;
        push_transcript(&mut transcript, &format!("> {trimmed}"));
        print_screen(
            &active_model,
            current_endpoint.as_deref().unwrap_or("(disconnected)"),
            tools.workspace(),
            prompt_branch.as_deref(),
            header_status,
            &transcript,
            None,
            input_state.as_str(),
            input_state.cursor(),
        );
        std::io::stdout().flush()?;
        if trimmed.starts_with('#') {
            continue;
        }

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
            prompt_branch.as_deref(),
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

struct RawModePauseGuard;

impl RawModePauseGuard {
    fn new() -> Result<Self> {
        disable_raw_mode()?;
        Ok(Self)
    }
}

impl Drop for RawModePauseGuard {
    fn drop(&mut self) {
        let _ = enable_raw_mode();
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
    prompt_branch: Option<&str>,
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
                prompt_branch,
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

    if let Some((start, path_prefix)) = open_file_completion_prefix(prefix) {
        return Some((
            start,
            cursor,
            open_file_completion_candidates(path_prefix, workspace),
        ));
    }

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
    let gitignore = workspace_gitignore(workspace);
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
            let entry_type = entry.file_type().ok()?;
            if !should_include_completion_path(
                workspace,
                &entry.path(),
                entry_type.is_dir(),
                gitignore.as_ref(),
            ) {
                return None;
            }

            let file_name = entry.file_name().to_string_lossy().to_string();
            if !file_name.starts_with(prefix) {
                return None;
            }

            let suffix = if entry_type.is_dir() { "/" } else { "" };
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

fn open_file_completion_prefix(prefix: &str) -> Option<(usize, &str)> {
    if let Some(path_prefix) = prefix.strip_prefix("/open_file ") {
        return Some(("/open_file ".len(), path_prefix));
    }

    for command_prefix in ["open file ", "open ", "edit file ", "edit "] {
        if let Some(path_prefix) = strip_ascii_prefix(prefix, command_prefix) {
            return Some((prefix.len() - path_prefix.len(), path_prefix));
        }
    }

    None
}

fn open_file_completion_candidates(token: &str, workspace: &Path) -> Vec<String> {
    let (quoted, token) = match token.chars().next() {
        Some(quote @ '"') | Some(quote @ '\'') => (Some(quote), &token[quote.len_utf8()..]),
        _ => (None, token),
    };
    let gitignore = workspace_gitignore(workspace);

    let mut matches = WalkDir::new(workspace)
        .into_iter()
        .filter_entry(|entry| {
            should_include_completion_path(
                workspace,
                entry.path(),
                entry.file_type().is_dir(),
                gitignore.as_ref(),
            )
        })
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_file())
        .filter_map(|entry| {
            let relative = entry.path().strip_prefix(workspace).ok()?;
            let relative = relative.to_string_lossy().replace('\\', "/");
            let file_name = entry.file_name().to_string_lossy();
            if !open_file_completion_matches(&relative, &file_name, token) {
                return None;
            }

            Some(match quoted {
                Some(quote) => format!("{quote}{relative}"),
                None => relative,
            })
        })
        .collect::<Vec<_>>();
    matches.sort();
    matches
}

fn open_file_completion_matches(relative: &str, file_name: &str, token: &str) -> bool {
    token.is_empty()
        || relative.starts_with(token)
        || (!token.contains('/') && file_name.starts_with(token))
}

fn workspace_gitignore(workspace: &Path) -> Option<Gitignore> {
    let mut builder = GitignoreBuilder::new(workspace);
    let gitignore_path = workspace.join(".gitignore");
    if gitignore_path.is_file() {
        builder.add(gitignore_path);
    }
    builder.build().ok()
}

fn should_include_completion_path(
    workspace: &Path,
    path: &Path,
    is_dir: bool,
    gitignore: Option<&Gitignore>,
) -> bool {
    let Ok(relative) = path.strip_prefix(workspace) else {
        return false;
    };
    if relative.as_os_str().is_empty() {
        return true;
    }

    let relative = relative.to_string_lossy().replace('\\', "/");
    if relative == ".git" || relative.starts_with(".git/") {
        return false;
    }

    gitignore.is_none_or(|matcher| {
        !matcher
            .matched_path_or_any_parents(path, is_dir)
            .is_ignore()
    })
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

enum LocalCommand<'a> {
    Help,
    ConnectDefault,
    ConnectTo(&'a str),
    Disconnect,
    Reload,
    ListModels,
    Tools,
    ModelInfo,
    SetModel(&'a str),
    OpenFile(&'a str),
    Clear,
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
    let Some(command) = parse_local_command(input) else {
        if input.trim_start().starts_with('/') {
            return Ok(CommandOutcome::Output(format!(
                "Unknown command '{}'. Use /help to see available commands.",
                input.trim()
            )));
        }
        return Ok(CommandOutcome::Unhandled);
    };

    match command {
        LocalCommand::Help => Ok(CommandOutcome::Output(help_text().to_string())),
        LocalCommand::ConnectDefault => {
            let endpoint = llms
                .get(active_model)
                .ok_or_else(|| anyhow!("unknown model profile '{active_model}'"))?
                .endpoint
                .clone();
            *current_endpoint = Some(endpoint.clone());
            Ok(CommandOutcome::Output(format!("Connected to '{endpoint}'")))
        }
        LocalCommand::ConnectTo(endpoint) => {
            *current_endpoint = Some(endpoint.to_string());
            Ok(CommandOutcome::Output(format!("Connected to '{endpoint}'")))
        }
        LocalCommand::Disconnect => Ok(CommandOutcome::Output({
            *current_endpoint = None;
            "Disconnected from the current server target".to_string()
        })),
        LocalCommand::Reload => {
            *active_model = startup_model.to_string();
            *current_endpoint = Some(startup_endpoint.to_string());
            let prompt = system_prompt(
                llms.get(startup_model)
                    .ok_or_else(|| anyhow!("unknown model profile '{startup_model}'"))?,
            );
            session.clear(prompt);
            Ok(CommandOutcome::Output(format!(
                "Reloaded startup configuration: model '{startup_model}', server '{startup_endpoint}'"
            )))
        }
        LocalCommand::ListModels => Ok(CommandOutcome::Output(format_models(llms))),
        LocalCommand::Tools => Ok(CommandOutcome::Output(format_tools(tools))),
        LocalCommand::ModelInfo => Ok(CommandOutcome::Output(
            "Use /list-models to see configured profiles".to_string(),
        )),
        LocalCommand::SetModel(name) => {
            if !llms.contains_key(name) {
                return Ok(CommandOutcome::Output(format!(
                    "Unknown model profile '{name}'. Available: {}",
                    sorted_model_names(llms).join(", ")
                )));
            }
            *active_model = name.to_string();
            session.set_system_prompt(system_prompt(&llms[name]));
            Ok(CommandOutcome::Output(format!(
                "Switched to model profile '{name}'"
            )))
        }
        LocalCommand::OpenFile(path) => {
            open_in_editor(workspace, path)?;
            Ok(CommandOutcome::Output(format!("Opened {}", path)))
        }
        LocalCommand::Clear => {
            let prompt = system_prompt(
                llms.get(active_model)
                    .ok_or_else(|| anyhow!("unknown model profile '{active_model}'"))?,
            );
            session.clear(prompt);
            Ok(CommandOutcome::Cleared)
        }
        LocalCommand::Quit => Ok(CommandOutcome::Quit),
    }
}

fn parse_local_command(input: &str) -> Option<LocalCommand<'_>> {
    let input = input.trim();
    if input.is_empty() {
        return None;
    }

    parse_slash_command(input).or_else(|| parse_natural_language_command(input))
}

fn parse_slash_command(input: &str) -> Option<LocalCommand<'_>> {
    match input {
        "/help" => Some(LocalCommand::Help),
        "/connect" => Some(LocalCommand::ConnectDefault),
        "/disconnect" => Some(LocalCommand::Disconnect),
        "/reload" => Some(LocalCommand::Reload),
        "/list-models" => Some(LocalCommand::ListModels),
        "/tools" => Some(LocalCommand::Tools),
        "/model" => Some(LocalCommand::ModelInfo),
        "/clear" => Some(LocalCommand::Clear),
        "/quit" => Some(LocalCommand::Quit),
        _ => {
            if let Some(endpoint) = input.strip_prefix("/connect ") {
                return Some(LocalCommand::ConnectTo(endpoint.trim()));
            }
            if let Some(name) = input.strip_prefix("/model ") {
                return Some(LocalCommand::SetModel(name.trim()));
            }
            parse_open_file_target(input, "/open_file ").map(LocalCommand::OpenFile)
        }
    }
}

fn parse_natural_language_command(input: &str) -> Option<LocalCommand<'_>> {
    if matches_ci(
        input,
        &[
            "help",
            "show help",
            "show commands",
            "show available commands",
        ],
    ) {
        return Some(LocalCommand::Help);
    }
    if matches_ci(input, &["connect", "reconnect"]) {
        return Some(LocalCommand::ConnectDefault);
    }
    if let Some(endpoint) = strip_ascii_prefix(input, "connect to ") {
        return Some(LocalCommand::ConnectTo(endpoint.trim()));
    }
    if matches_ci(input, &["disconnect"]) {
        return Some(LocalCommand::Disconnect);
    }
    if matches_ci(input, &["reload", "reload configuration", "reset session"]) {
        return Some(LocalCommand::Reload);
    }
    if matches_ci(
        input,
        &[
            "list models",
            "show models",
            "show available models",
            "models",
        ],
    ) {
        return Some(LocalCommand::ListModels);
    }
    if matches_ci(
        input,
        &["show tools", "list tools", "show local tools", "tools"],
    ) {
        return Some(LocalCommand::Tools);
    }
    if matches_ci(
        input,
        &[
            "show model",
            "current model",
            "what model am i using",
            "model",
        ],
    ) {
        return Some(LocalCommand::ModelInfo);
    }
    for prefix in [
        "use model ",
        "switch model to ",
        "set model to ",
        "select model ",
    ] {
        if let Some(name) = strip_ascii_prefix(input, prefix) {
            return Some(LocalCommand::SetModel(name.trim()));
        }
    }
    if let Some(path) = parse_open_file_target(input, "/open_file ") {
        return Some(LocalCommand::OpenFile(path));
    }
    for prefix in ["open file ", "open ", "edit file ", "edit "] {
        if let Some(path) = parse_open_file_target(input, prefix) {
            return Some(LocalCommand::OpenFile(path));
        }
    }
    if matches_ci(
        input,
        &["clear", "clear conversation", "reset conversation"],
    ) {
        return Some(LocalCommand::Clear);
    }
    if matches_ci(input, &["quit", "exit"]) {
        return Some(LocalCommand::Quit);
    }

    None
}

fn parse_open_file_target<'a>(input: &'a str, prefix: &str) -> Option<&'a str> {
    let path = strip_ascii_prefix(input, prefix)?.trim();
    if path.is_empty() {
        return None;
    }
    Some(strip_matching_quotes(path))
}

fn strip_ascii_prefix<'a>(input: &'a str, prefix: &str) -> Option<&'a str> {
    if input.len() >= prefix.len() && input[..prefix.len()].eq_ignore_ascii_case(prefix) {
        Some(&input[prefix.len()..])
    } else {
        None
    }
}

fn matches_ci(input: &str, options: &[&str]) -> bool {
    options
        .iter()
        .any(|option| input.eq_ignore_ascii_case(option))
}

fn strip_matching_quotes(input: &str) -> &str {
    if input.len() >= 2 {
        let bytes = input.as_bytes();
        let first = bytes[0];
        let last = bytes[input.len() - 1];
        if (first == b'"' || first == b'\'') && first == last {
            return &input[1..input.len() - 1];
        }
    }
    input
}

fn open_in_editor(workspace: &Path, raw_path: &str) -> Result<()> {
    let editor = std::env::var("EDITOR").context("EDITOR is not set")?;
    let editor_parts = shell_words(&editor)?;
    let path = resolve_workspace_path(workspace, raw_path)?;
    let (program, args) = editor_parts
        .split_first()
        .ok_or_else(|| anyhow!("EDITOR is empty"))?;

    let _raw_mode_pause_guard = RawModePauseGuard::new()?;
    let _child = std::process::Command::new(program)
        .args(args)
        .arg(&path)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .with_context(|| format!("failed to launch editor '{}'", editor))?;

    Ok(())
}

fn shell_words(input: &str) -> Result<Vec<String>> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut chars = input.chars().peekable();
    let mut quote = None;

    while let Some(ch) = chars.next() {
        match quote {
            Some(active_quote) => {
                if ch == active_quote {
                    quote = None;
                } else if ch == '\\' && active_quote == '"' {
                    if let Some(escaped) = chars.next() {
                        current.push(escaped);
                    }
                } else {
                    current.push(ch);
                }
            }
            None if ch.is_whitespace() => {
                if !current.is_empty() {
                    words.push(std::mem::take(&mut current));
                }
            }
            None if ch == '"' || ch == '\'' => {
                quote = Some(ch);
            }
            None if ch == '\\' => {
                if let Some(escaped) = chars.next() {
                    current.push(escaped);
                }
            }
            None => current.push(ch),
        }
    }

    if quote.is_some() {
        return Err(anyhow!("EDITOR contains unterminated quotes"));
    }
    if !current.is_empty() {
        words.push(current);
    }
    if words.is_empty() {
        return Err(anyhow!("EDITOR is empty"));
    }

    Ok(words)
}

fn workspace_branch_name(workspace: &Path) -> Option<String> {
    let git_dir = discover_git_dir(workspace)?;
    let head = fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let reference = head.trim().strip_prefix("ref: ")?;
    reference.strip_prefix("refs/heads/").map(ToOwned::to_owned)
}

fn discover_git_dir(workspace: &Path) -> Option<PathBuf> {
    for ancestor in workspace.ancestors() {
        let git_entry = ancestor.join(".git");
        if git_entry.is_dir() {
            return Some(git_entry);
        }
        if git_entry.is_file() {
            let gitdir = fs::read_to_string(&git_entry).ok()?;
            let relative = gitdir.trim().strip_prefix("gitdir: ")?.trim();
            let path = Path::new(relative);
            return Some(if path.is_absolute() {
                path.to_path_buf()
            } else {
                ancestor.join(path)
            });
        }
    }
    None
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
    prompt_branch: Option<&str>,
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
            prompt_branch,
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
    prompt_branch: Option<&str>,
    header_status: HeaderStatus,
    transcript: &[String],
) -> Result<String> {
    let mut prompt_future = Box::pin(session.prompt(user_input, profile, tools));
    let mut interval = tokio::time::interval(THINKING_FRAME_INTERVAL);
    let mut thinking_frame = 0usize;
    let thinking_started = Instant::now();
    let initial_frame = render_thinking_frame(thinking_frame, thinking_started.elapsed());

    print_screen(
        current_model,
        endpoint,
        workspace,
        prompt_branch,
        header_status,
        transcript,
        Some(initial_frame.as_str()),
        "",
        0,
    );
    std::io::stdout().flush()?;

    loop {
        tokio::select! {
            result = &mut prompt_future => return result,
            _ = interval.tick() => {
                thinking_frame = thinking_frame.wrapping_add(1);
                let pending_line =
                    render_thinking_frame(thinking_frame, thinking_started.elapsed());
                print_screen(
                    current_model,
                    endpoint,
                    workspace,
                    prompt_branch,
                    header_status,
                    transcript,
                    Some(pending_line.as_str()),
                    "",
                    0,
                );
                std::io::stdout().flush()?;
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

fn resolve_workspace_root(workspace: Option<PathBuf>) -> Result<PathBuf> {
    let current_dir = std::env::current_dir().context("failed to resolve current directory")?;
    let workspace = workspace.unwrap_or_else(|| current_dir.clone());
    let absolute = if workspace.is_absolute() {
        workspace
    } else {
        current_dir.join(workspace)
    };
    Ok(normalize_path(&absolute))
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => result.push(prefix.as_os_str()),
            Component::RootDir => result.push(Path::new("/")),
            Component::CurDir => {}
            Component::ParentDir => {
                result.pop();
            }
            Component::Normal(part) => result.push(part),
        }
    }
    result
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

#[cfg(test)]
mod tests {
    use super::{
        CommandOutcome, LocalCommand, completion_candidates, discover_git_dir, handle_command,
        parse_local_command, resolve_workspace_root, shell_words, system_prompt,
        workspace_branch_name,
    };
    use orangu::{config::LlmConfiguration, session::ChatSession, tools::ToolExecutor};
    use std::collections::HashMap;
    use std::{fs, path::PathBuf};
    use tempfile::tempdir;

    #[test]
    fn resolve_workspace_root_makes_relative_paths_absolute() {
        let current_dir = std::env::current_dir().expect("current directory");
        let resolved = resolve_workspace_root(Some(PathBuf::from("."))).expect("workspace");

        assert_eq!(resolved, current_dir);
        assert!(resolved.is_absolute());
    }

    #[test]
    fn resolve_workspace_root_normalizes_parent_segments() {
        let current_dir = std::env::current_dir().expect("current directory");
        let resolved =
            resolve_workspace_root(Some(PathBuf::from("src/../tests"))).expect("workspace");

        assert_eq!(resolved, current_dir.join("tests"));
    }

    #[test]
    fn parses_open_file_commands() {
        match parse_local_command("/open_file README.md") {
            Some(LocalCommand::OpenFile(path)) => assert_eq!(path, "README.md"),
            _ => panic!("expected open file slash command"),
        }
        match parse_local_command("Open README.md") {
            Some(LocalCommand::OpenFile(path)) => assert_eq!(path, "README.md"),
            _ => panic!("expected open file natural language command"),
        }
        match parse_local_command("open \"docs/user guide.md\"") {
            Some(LocalCommand::OpenFile(path)) => assert_eq!(path, "docs/user guide.md"),
            _ => panic!("expected quoted natural language open file command"),
        }
    }

    #[test]
    fn parses_natural_language_command_aliases() {
        assert!(matches!(
            parse_local_command("show commands"),
            Some(LocalCommand::Help)
        ));
        assert!(matches!(
            parse_local_command("list models"),
            Some(LocalCommand::ListModels)
        ));
        assert!(matches!(
            parse_local_command("show tools"),
            Some(LocalCommand::Tools)
        ));
        assert!(matches!(
            parse_local_command("disconnect"),
            Some(LocalCommand::Disconnect)
        ));
        assert!(matches!(
            parse_local_command("reset conversation"),
            Some(LocalCommand::Clear)
        ));
        assert!(matches!(
            parse_local_command("exit"),
            Some(LocalCommand::Quit)
        ));
    }

    #[test]
    fn parses_natural_language_commands_with_arguments() {
        match parse_local_command("connect to http://localhost:8080/v1") {
            Some(LocalCommand::ConnectTo(endpoint)) => {
                assert_eq!(endpoint, "http://localhost:8080/v1")
            }
            _ => panic!("expected connect command"),
        }
        match parse_local_command("switch model to local") {
            Some(LocalCommand::SetModel(name)) => assert_eq!(name, "local"),
            _ => panic!("expected set model command"),
        }
    }

    #[test]
    fn leaves_regular_prompts_unhandled() {
        assert!(parse_local_command("help me understand this code").is_none());
        assert!(parse_local_command("show me the files in the workspace").is_none());
    }

    #[test]
    fn discovers_git_branch_name_from_workspace() {
        let workspace = tempdir().expect("workspace");
        fs::create_dir(workspace.path().join(".git")).expect("git dir");
        fs::write(workspace.path().join(".git/HEAD"), "ref: refs/heads/main\n").expect("head");

        assert_eq!(
            workspace_branch_name(workspace.path()).as_deref(),
            Some("main")
        );
        assert_eq!(
            discover_git_dir(workspace.path()).as_deref(),
            Some(workspace.path().join(".git").as_path())
        );
    }

    #[test]
    fn unknown_slash_commands_error_locally() {
        let workspace = tempdir().expect("workspace");
        let tools = ToolExecutor::new(workspace.path());
        let mut llms = HashMap::new();
        llms.insert(
            "default".to_string(),
            LlmConfiguration {
                provider: "openai".to_string(),
                model: "gpt-4.1".to_string(),
                endpoint: "http://localhost:11434/v1".to_string(),
                api_key: None,
                request_timeout_seconds: 30,
                max_tool_rounds: 10,
                system_prompt: String::new(),
            },
        );
        let mut session = ChatSession::new(system_prompt(&llms["default"]));
        let mut active_model = "default".to_string();
        let mut current_endpoint = Some("http://localhost:11434/v1".to_string());

        let outcome = handle_command(
            "/unknown",
            &mut active_model,
            &mut current_endpoint,
            "default",
            "http://localhost:11434/v1",
            &llms,
            &mut session,
            &tools,
            workspace.path(),
        )
        .expect("command outcome");

        assert!(matches!(
            outcome,
            CommandOutcome::Output(ref message)
                if message == "Unknown command '/unknown'. Use /help to see available commands."
        ));
    }

    #[test]
    fn completes_open_file_commands_across_workspace() {
        let workspace = tempdir().expect("workspace");
        fs::write(workspace.path().join("README.md"), "").expect("root readme");
        fs::create_dir(workspace.path().join("doc")).expect("doc dir");
        fs::write(workspace.path().join("doc/README.md"), "").expect("doc readme");
        fs::write(workspace.path().join(".gitignore"), "ignored.md\n").expect("gitignore");
        fs::write(workspace.path().join("ignored.md"), "").expect("ignored file");
        fs::create_dir(workspace.path().join(".git")).expect("git dir");
        fs::write(workspace.path().join(".git/config"), "").expect("git config");

        let (_, _, slash_candidates) = completion_candidates(
            "/open_file READ",
            "/open_file READ".len(),
            workspace.path(),
            &[],
        )
        .expect("slash completion");
        assert_eq!(
            slash_candidates,
            vec!["README.md".to_string(), "doc/README.md".to_string()]
        );

        let (start, _, natural_candidates) =
            completion_candidates("Open READ", "Open READ".len(), workspace.path(), &[])
                .expect("natural completion");
        assert_eq!(start, "Open ".len());
        assert_eq!(
            natural_candidates,
            vec!["README.md".to_string(), "doc/README.md".to_string()]
        );

        let (_, _, ignored_candidates) =
            completion_candidates("Open ign", "Open ign".len(), workspace.path(), &[])
                .expect("ignored completion");
        assert!(ignored_candidates.is_empty());

        let (_, _, git_candidates) =
            completion_candidates("Open con", "Open con".len(), workspace.path(), &[])
                .expect("git completion");
        assert!(git_candidates.is_empty());
    }

    #[test]
    fn splits_editor_command_and_flags() {
        assert_eq!(
            shell_words("code --wait").expect("editor command"),
            vec!["code".to_string(), "--wait".to_string()]
        );
        assert_eq!(
            shell_words("\"/tmp/my editor\" --flag").expect("quoted editor command"),
            vec!["/tmp/my editor".to_string(), "--flag".to_string()]
        );
    }
}
