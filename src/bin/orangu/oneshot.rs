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

//! `orangu -p "<prompt>"`: send one prompt to the configured server, stream the
//! answer to stdout and exit — no terminal UI, no session on disk. The prompt is
//! assembled exactly as the interactive client assembles it (system prompt,
//! skills index, `AGENTS.md`, tool definitions), so what it measures is what the
//! prompt area does.

use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Result, anyhow};

use crate::commands::system_prompt;
use orangu::{config::ClientAppConfiguration, llm::StreamMetrics, session::ChatSession};

/// Everything the run loop already resolved before it knew this was a one-shot:
/// the loaded config and the workspace the prompt runs against.
pub(crate) struct OneshotContext {
    pub(crate) config: ClientAppConfiguration,
    pub(crate) workspace: PathBuf,
}

pub(crate) async fn run_prompt(prompt: &str, context: OneshotContext) -> Result<()> {
    let OneshotContext { config, workspace } = context;

    let server = config.default_server.clone();
    let profile = config
        .llms
        .get(&server)
        .ok_or_else(|| anyhow!("missing configured server {server}"))?;

    let tools = orangu::tools::ToolExecutor::with_config(
        &workspace,
        config.compression,
        config.auto_downsample_lines,
        config.diff_file_cap,
        None,
    );
    let skills = orangu::skills::SkillRegistry::discover(&workspace);

    // The same system prompt the interactive client builds for a tab.
    let mut system = system_prompt(profile, None).to_string();
    let index = skills.system_prompt_index();
    if !index.is_empty() {
        system.push_str("\n\n");
        system.push_str(&index);
    }
    system.push_str(&orangu::config::load_agents_instructions(&workspace));

    let definitions = tools.definitions();
    let tools_bytes = serde_json::to_string(&definitions).map_or(0, |json| json.len());
    eprintln!(
        "server {server} ({}) model {} — workspace {}",
        profile.endpoint,
        profile.model,
        workspace.display()
    );
    eprintln!(
        "sending {} chars of system prompt and {} tool definitions ({tools_bytes} chars)",
        system.chars().count(),
        definitions.len(),
    );

    let mut session = ChatSession::new(&system);
    let metrics = Arc::new(Mutex::new(StreamMetrics::default()));
    let collected = Arc::clone(&metrics);
    let started = Instant::now();
    let mut first_delta: Option<std::time::Duration> = None;

    let result = session
        .prompt(
            prompt,
            profile,
            &tools,
            |delta| {
                if first_delta.is_none() {
                    first_delta = Some(started.elapsed());
                }
                print!("{delta}");
                let _ = std::io::stdout().flush();
            },
            move |update| {
                if let Ok(mut state) = collected.lock() {
                    state.merge(update);
                }
            },
            |_running| {},
            |tool_call| {
                eprintln!("[tool] {}", tool_call.function.name);
            },
        )
        .await;

    println!();
    let elapsed = started.elapsed();
    let metrics = metrics.lock().ok().map(|state| state.clone());
    report_timings(elapsed, first_delta, metrics.as_ref());

    result.map(|_| ())
}

/// The line that makes a slow answer diagnosable: how long the server spent on
/// the prompt (and how much of it came from its KV cache) against how long it
/// spent generating.
fn report_timings(
    elapsed: std::time::Duration,
    first_delta: Option<std::time::Duration>,
    metrics: Option<&StreamMetrics>,
) {
    let mut parts = vec![format!("total {:.1}s", elapsed.as_secs_f64())];
    if let Some(first) = first_delta {
        parts.push(format!("first token {:.1}s", first.as_secs_f64()));
    }
    if let Some(progress) = metrics.and_then(|m| m.prompt_progress.as_ref()) {
        parts.push(format!(
            "prompt {} tokens ({} cached) in {:.1}s",
            progress.total,
            progress.cache,
            progress.time_ms as f64 / 1000.0
        ));
    }
    if let Some(rate) = metrics.and_then(|m| m.prompt_per_second) {
        parts.push(format!("prefill {rate:.1} t/s"));
    }
    if let Some(rate) = metrics.and_then(|m| m.predicted_per_second) {
        parts.push(format!("decode {rate:.1} t/s"));
    }
    eprintln!("[{}]", parts.join(" | "));
}
