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

//! Bounded code-and-review loops for the non-interactive CLI.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use crate::workflow::{WorkflowLoop, WorkflowLoopStop};
use anyhow::{Context, Result, anyhow, bail};
use clap::{Args, Subcommand};
use orangu::config::ClientAppConfiguration;
use orangu::llm::{StreamMetrics, normalized_openai_endpoint};
use orangu::session::ChatSession;
use orangu::skills::SkillRegistry;
use orangu::tools::ToolExecutor;
use serde::{Deserialize, Serialize};

use crate::commands::build_workspace_system_prompt;
use crate::git::collect_review_diff;
use crate::models::{coordinator_role_profile, is_active_connection_a_coordinator};

const MAX_REVIEW_INPUT_BYTES: usize = 48_000;
const MAX_CHECK_OUTPUT_BYTES: usize = 12_000;
const MAX_SAVED_REVIEW_BYTES: usize = 4_000;
const LOOP_STATE_VERSION: u32 = 1;
const LOOP_STATE_FILE: &str = "active.yaml";
const LOOP_LOCK_FILE: &str = "running.lock";

/// Run a bounded work-and-review loop from the command line.
#[derive(Args, Debug)]
pub(crate) struct LoopArgs {
    #[command(subcommand)]
    action: Option<LoopAction>,
    /// Stop after this many complete work-and-review iterations.
    #[arg(long, value_name = "COUNT")]
    turns: Option<u32>,
    /// Stop after this amount of active wall-clock time (for example 30m or 2h).
    #[arg(long, value_name = "DURATION")]
    time: Option<String>,
    /// Continue until the reviewer can verify this condition.
    #[arg(long, value_name = "CONDITION")]
    until: Option<String>,
    /// Load a reusable loop definition from YAML.
    #[arg(long, value_name = "FILE")]
    file: Option<PathBuf>,
    /// Run this validation command after every work phase. May be supplied more than once.
    #[arg(long = "check", value_name = "COMMAND")]
    checks: Vec<String>,
    /// Additional review criterion. May be supplied more than once.
    #[arg(long = "review", value_name = "CRITERION")]
    rubric: Vec<String>,
    /// The coding objective. Omit it when --file supplies the objective.
    #[arg(value_name = "OBJECTIVE", trailing_var_arg = true)]
    objective: Vec<String>,
}

/// Manage the single persisted loop for this workspace.
#[derive(Subcommand, Clone, Debug)]
enum LoopAction {
    /// Show the saved loop's objective, progress, and state.
    Status,
    /// Ask a running loop to stop safely after its current phase.
    Pause,
    /// Continue a paused or interrupted loop from its saved objective.
    Resume,
    /// Cancel the saved loop. A new `orangu loop` command may then replace it.
    Clear,
}

pub(crate) struct LoopRunContext {
    pub(crate) config: ClientAppConfiguration,
    pub(crate) workspace: PathBuf,
    /// An explicit workflow role. Standalone loops leave routing to Orangu's
    /// configured default, preserving their original CLI behavior.
    pub(crate) role: Option<String>,
    pub(crate) quiet: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LoopSpec {
    objective: String,
    stop: StopPolicy,
    review: ReviewConfig,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum StopPolicy {
    Turns(u32),
    Time(Duration),
    Goal(String),
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ReviewConfig {
    #[serde(default)]
    checks: Vec<String>,
    #[serde(default)]
    rubric: Vec<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LoopFile {
    version: u32,
    objective: String,
    stop: LoopFileStop,
    #[serde(default)]
    review: ReviewConfig,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum LoopFileStop {
    Turns { count: u32 },
    Time { duration: String },
    Goal { condition: String },
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PersistedLoop {
    version: u32,
    state: LoopState,
    objective: String,
    stop: PersistedStopPolicy,
    review: ReviewConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    iterations: u32,
    active_seconds: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    last_review: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    failure: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum LoopState {
    Active,
    Paused,
    Complete,
    TurnLimitReached,
    TimeLimitReached,
    Failed,
    Cancelled,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum PersistedStopPolicy {
    Turns { count: u32 },
    Time { seconds: u64 },
    Goal { condition: String },
}

impl LoopArgs {
    fn into_spec(self) -> Result<LoopSpec> {
        if let Some(path) = self.file {
            if self.turns.is_some()
                || self.time.is_some()
                || self.until.is_some()
                || !self.checks.is_empty()
                || !self.rubric.is_empty()
                || !self.objective.is_empty()
            {
                bail!("--file cannot be combined with loop flags or an objective");
            }
            return LoopSpec::from_file(&path);
        }

        let objective = self.objective.join(" ").trim().to_string();
        if objective.is_empty() {
            bail!("an objective is required (or supply --file)");
        }
        let specified = usize::from(self.turns.is_some())
            + usize::from(self.time.is_some())
            + usize::from(self.until.is_some());
        if specified != 1 {
            bail!("supply exactly one stopping policy: --turns, --time, or --until");
        }

        let stop = if let Some(turns) = self.turns {
            validate_turns(turns)?;
            StopPolicy::Turns(turns)
        } else if let Some(duration) = self.time {
            StopPolicy::Time(parse_duration(&duration)?)
        } else {
            StopPolicy::Goal(required_text(self.until.as_deref(), "--until")?)
        };
        Ok(LoopSpec {
            objective,
            stop,
            review: ReviewConfig {
                checks: self.checks,
                rubric: self.rubric,
            },
        })
    }

    fn lifecycle_action(&self) -> Result<Option<LoopAction>> {
        let Some(action) = &self.action else {
            return Ok(None);
        };
        if self.turns.is_some()
            || self.time.is_some()
            || self.until.is_some()
            || self.file.is_some()
            || !self.checks.is_empty()
            || !self.rubric.is_empty()
            || !self.objective.is_empty()
        {
            bail!("loop lifecycle commands cannot be combined with loop flags or an objective");
        }
        Ok(Some(action.clone()))
    }
}

impl LoopSpec {
    fn from_file(path: &Path) -> Result<Self> {
        let source = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read loop definition {}", path.display()))?;
        let file: LoopFile = serde_yaml::from_str(&source)
            .with_context(|| format!("invalid loop definition {}", path.display()))?;
        if file.version != 1 {
            bail!(
                "unsupported loop definition version {}; expected 1",
                file.version
            );
        }
        let objective = required_text(Some(&file.objective), "objective")?;
        let stop = match file.stop {
            LoopFileStop::Turns { count } => {
                validate_turns(count)?;
                StopPolicy::Turns(count)
            }
            LoopFileStop::Time { duration } => StopPolicy::Time(parse_duration(&duration)?),
            LoopFileStop::Goal { condition } => {
                StopPolicy::Goal(required_text(Some(&condition), "stop.condition")?)
            }
        };
        Ok(Self {
            objective,
            stop,
            review: file.review,
        })
    }
}

impl From<WorkflowLoop> for LoopSpec {
    fn from(spec: WorkflowLoop) -> Self {
        let stop = match spec.stop {
            WorkflowLoopStop::Turns(count) => StopPolicy::Turns(count),
            WorkflowLoopStop::Time(duration) => StopPolicy::Time(duration),
            WorkflowLoopStop::Goal(condition) => StopPolicy::Goal(condition),
        };
        Self {
            objective: spec.objective,
            stop,
            review: ReviewConfig {
                checks: spec.checks,
                rubric: spec.rubric,
            },
        }
    }
}

impl PersistedLoop {
    fn new(spec: LoopSpec, role: Option<String>) -> Self {
        Self {
            version: LOOP_STATE_VERSION,
            state: LoopState::Active,
            objective: spec.objective,
            stop: PersistedStopPolicy::from_stop(spec.stop),
            review: spec.review,
            role,
            iterations: 0,
            active_seconds: 0,
            last_review: None,
            failure: None,
        }
    }

    fn to_spec(&self) -> LoopSpec {
        LoopSpec {
            objective: self.objective.clone(),
            stop: self.stop.to_stop(),
            review: self.review.clone(),
        }
    }

    fn is_terminal(&self) -> bool {
        matches!(
            self.state,
            LoopState::Complete
                | LoopState::TurnLimitReached
                | LoopState::TimeLimitReached
                | LoopState::Cancelled
        )
    }

    fn status_report(&self) -> String {
        let mut report = format!(
            "Loop status: {}\nObjective: {}\nStopping policy: {}\nCompleted iterations: {}\nActive time: {} second(s)\n",
            self.state.describe(),
            self.objective,
            self.stop.to_stop().describe(),
            self.iterations,
            self.active_seconds,
        );
        if let Some(role) = &self.role {
            report.push_str(&format!("Role: {role}\n"));
        }
        if let Some(failure) = &self.failure {
            report.push_str(&format!("Last failure: {failure}\n"));
        }
        if let Some(review) = &self.last_review {
            report.push_str(&format!("Last review:\n{review}\n"));
        }
        report
    }
}

impl LoopState {
    fn describe(self) -> &'static str {
        match self {
            Self::Active => "active",
            Self::Paused => "paused",
            Self::Complete => "complete",
            Self::TurnLimitReached => "turn limit reached",
            Self::TimeLimitReached => "time limit reached",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

impl PersistedStopPolicy {
    fn from_stop(stop: StopPolicy) -> Self {
        match stop {
            StopPolicy::Turns(count) => Self::Turns { count },
            StopPolicy::Time(duration) => Self::Time {
                seconds: duration.as_secs(),
            },
            StopPolicy::Goal(condition) => Self::Goal { condition },
        }
    }

    fn to_stop(&self) -> StopPolicy {
        match self {
            Self::Turns { count } => StopPolicy::Turns(*count),
            Self::Time { seconds } => StopPolicy::Time(Duration::from_secs(*seconds)),
            Self::Goal { condition } => StopPolicy::Goal(condition.clone()),
        }
    }
}

fn loop_state_path(workspace: &Path) -> PathBuf {
    orangu::workspace_cache::workspace_cache_dir(workspace, "loops").join(LOOP_STATE_FILE)
}

fn loop_lock_path(state_path: &Path) -> PathBuf {
    state_path.with_file_name(LOOP_LOCK_FILE)
}

struct LoopLock {
    path: PathBuf,
}

impl LoopLock {
    fn acquire(state_path: &Path) -> Result<Self> {
        let path = loop_lock_path(state_path);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| {
                format!(
                    "a loop is already running for this workspace; use `orangu loop status` ({})",
                    path.display()
                )
            })?;
        let _ = writeln!(file, "pid={}", std::process::id());
        Ok(Self { path })
    }
}

impl Drop for LoopLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn load_state(path: &Path) -> Result<PersistedLoop> {
    let source = std::fs::read_to_string(path).with_context(|| {
        format!(
            "no saved loop for this workspace; start one with `orangu loop ...` ({})",
            path.display()
        )
    })?;
    let state: PersistedLoop = serde_yaml::from_str(&source)
        .with_context(|| format!("invalid saved loop state {}", path.display()))?;
    if state.version != LOOP_STATE_VERSION {
        bail!(
            "unsupported saved loop version {}; expected {LOOP_STATE_VERSION}",
            state.version
        );
    }
    Ok(state)
}

fn save_state(path: &Path, state: &PersistedLoop) -> Result<()> {
    let parent = path
        .parent()
        .context("saved loop state has no parent directory")?;
    std::fs::create_dir_all(parent)
        .with_context(|| format!("failed to create loop state directory {}", parent.display()))?;
    let content = serde_yaml::to_string(state).context("failed to serialize loop state")?;
    let temporary = path.with_extension("yaml.tmp");
    std::fs::write(&temporary, content)
        .with_context(|| format!("failed to write loop state {}", temporary.display()))?;
    std::fs::rename(&temporary, path)
        .with_context(|| format!("failed to save loop state {}", path.display()))
}

fn active_elapsed(active_seconds_before_resume: u64, resumed_at: Instant) -> Duration {
    Duration::from_secs(active_seconds_before_resume).saturating_add(resumed_at.elapsed())
}

fn checkpoint(
    path: &Path,
    state: &mut PersistedLoop,
    active_seconds_before_resume: u64,
    resumed_at: Instant,
) -> Result<()> {
    state.active_seconds = active_elapsed(active_seconds_before_resume, resumed_at).as_secs();
    if let Ok(saved) = load_state(path)
        && saved.state != LoopState::Active
    {
        state.state = saved.state;
    }
    save_state(path, state)
}

fn validate_turns(turns: u32) -> Result<()> {
    if turns == 0 {
        bail!("--turns must be greater than zero");
    }
    Ok(())
}

fn required_text(value: Option<&str>, field: &str) -> Result<String> {
    value
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow!("{field} must not be empty"))
}

fn parse_duration(input: &str) -> Result<Duration> {
    let input = input.trim();
    let split = input.find(|character: char| !character.is_ascii_digit());
    let Some(index) = split else {
        bail!("invalid duration '{input}'; use a positive value such as 30m or 2h");
    };
    let (number, unit) = input.split_at(index);
    if number.is_empty()
        || unit.is_empty()
        || unit.chars().any(|character| character.is_whitespace())
    {
        bail!("invalid duration '{input}'; use a positive value such as 30m or 2h");
    }
    let value: u64 = number
        .parse()
        .with_context(|| format!("invalid duration '{input}'"))?;
    if value == 0 {
        bail!("duration must be greater than zero");
    }
    let seconds = match unit {
        "s" => value,
        "m" => value.checked_mul(60).context("duration is too large")?,
        "h" => value
            .checked_mul(60 * 60)
            .context("duration is too large")?,
        "d" => value
            .checked_mul(24 * 60 * 60)
            .context("duration is too large")?,
        _ => bail!("invalid duration unit in '{input}'; use s, m, h, or d"),
    };
    Ok(Duration::from_secs(seconds))
}

pub(crate) async fn run(args: LoopArgs, context: LoopRunContext) -> Result<()> {
    if let Some(action) = args.lifecycle_action()? {
        return manage_loop(action, context).await;
    }
    let spec = args.into_spec()?;
    start(spec, context).await
}

/// Run a loop that was validated and expanded by the unified workflow
/// compiler. Lifecycle state is intentionally shared with `orangu loop` so a
/// workflow run can be inspected, paused, and resumed through the existing CLI.
pub(crate) async fn run_workflow(spec: WorkflowLoop, context: LoopRunContext) -> Result<()> {
    start(spec.into(), context).await
}

pub(crate) async fn workflow_action(action: &str, context: LoopRunContext) -> Result<()> {
    let action = match action {
        "status" => LoopAction::Status,
        "pause" => LoopAction::Pause,
        "resume" => LoopAction::Resume,
        "clear" => LoopAction::Clear,
        _ => bail!("unknown workflow action '{action}'"),
    };
    manage_loop(action, context).await
}

async fn start(spec: LoopSpec, context: LoopRunContext) -> Result<()> {
    let state_path = loop_state_path(&context.workspace);
    if let Ok(existing) = load_state(&state_path)
        && matches!(existing.state, LoopState::Active | LoopState::Paused)
    {
        bail!(
            "a {} loop is already saved for this workspace; use `orangu loop status`, `resume`, `pause`, or `clear`",
            existing.state.describe()
        );
    }
    let state = PersistedLoop::new(spec, context.role.clone());
    save_state(&state_path, &state)?;
    run_persisted(state, context, state_path).await
}

async fn manage_loop(action: LoopAction, context: LoopRunContext) -> Result<()> {
    let path = loop_state_path(&context.workspace);
    match action {
        LoopAction::Status => {
            let state = load_state(&path)?;
            if !context.quiet {
                print!("{}", state.status_report());
            }
            Ok(())
        }
        LoopAction::Pause => {
            let mut state = load_state(&path)?;
            if state.is_terminal() {
                bail!(
                    "the saved loop is already {}; start a new loop instead",
                    state.state.describe()
                );
            }
            state.state = LoopState::Paused;
            save_state(&path, &state)?;
            if !context.quiet {
                println!(
                    "Loop pause requested. A running loop will stop at its next safe boundary."
                );
            }
            Ok(())
        }
        LoopAction::Clear => {
            let mut state = load_state(&path)?;
            state.state = LoopState::Cancelled;
            state.failure = None;
            save_state(&path, &state)?;
            if !context.quiet {
                println!("Saved loop cancelled. A new `orangu loop` command may now replace it.");
            }
            Ok(())
        }
        LoopAction::Resume => {
            let mut state = load_state(&path)?;
            if loop_lock_path(&path).exists() {
                bail!(
                    "a loop is already running for this workspace; pause it or wait for its current phase to finish"
                );
            }
            if state.is_terminal() && state.state != LoopState::Failed {
                bail!(
                    "the saved loop is {}; start a new loop instead",
                    state.state.describe()
                );
            }
            state.state = LoopState::Active;
            state.failure = None;
            save_state(&path, &state)?;
            run_persisted(state, context, path).await
        }
    }
}

async fn run_persisted(
    mut state: PersistedLoop,
    context: LoopRunContext,
    state_path: PathBuf,
) -> Result<()> {
    let _lock = LoopLock::acquire(&state_path)?;
    let result = run_persisted_inner(&mut state, &context, &state_path).await;
    if let Err(error) = &result {
        state.state = LoopState::Failed;
        state.failure = Some(format!("{error:#}"));
        let _ = save_state(&state_path, &state);
    }
    result
}

async fn run_persisted_inner(
    state: &mut PersistedLoop,
    context: &LoopRunContext,
    state_path: &Path,
) -> Result<()> {
    let LoopRunContext {
        config,
        workspace,
        role,
        quiet,
    } = context;
    let spec = state.to_spec();
    let role = role.as_deref().or(state.role.as_deref());
    let server = role
        .map(|role| config.find_server_for_role(role))
        .unwrap_or_else(|| config.default_server.clone());
    let configured_profile = config
        .llms
        .get(&server)
        .ok_or_else(|| anyhow!("missing configured server {server}"))?
        .clone();
    let profile = match role {
        Some(role) => {
            let endpoint = normalized_openai_endpoint(&configured_profile.endpoint);
            let http_client = reqwest::Client::builder()
                .timeout(Duration::from_secs(3))
                .build()?;
            if is_active_connection_a_coordinator(&http_client, config, &server, Some(&endpoint))
                .await
            {
                coordinator_role_profile(&configured_profile, &endpoint, role)
            } else {
                configured_profile
            }
        }
        None => configured_profile,
    };
    let (mcp, warnings) =
        orangu::mcp::McpManager::connect_all(&config.mcp_servers, workspace).await?;
    for warning in warnings {
        note(*quiet, &format!("Warning: {warning}"));
    }
    let tools = ToolExecutor::with_config(
        workspace,
        config.compression,
        config.auto_downsample_lines,
        config.diff_file_cap,
        None,
    )
    .with_mcp(mcp.clone());
    let skills = SkillRegistry::discover(workspace);
    let mut system = build_workspace_system_prompt(&profile, &skills, workspace, None);
    if !mcp.instructions().is_empty() {
        system.push_str("\n\n");
        system.push_str(&mcp.instructions());
    }
    let mut work_session = ChatSession::new(&system);
    let review_system = format!(
        "{system}\n\nYou are the review phase of an automated code loop. You are read-only: do not claim to have changed files, and base conclusions only on the supplied evidence."
    );
    let mut review_session = ChatSession::new(&review_system);
    let resumed_at = Instant::now();
    let active_seconds_before_resume = state.active_seconds;

    note(
        *quiet,
        &format!(
            "loop started: {} ({})",
            spec.objective,
            spec.stop.describe()
        ),
    );
    loop {
        if state.state != LoopState::Active {
            finish(*quiet, state.iterations, "pause or cancellation requested");
            return Ok(());
        }
        if let StopPolicy::Time(limit) = spec.stop
            && state.iterations > 0
            && active_elapsed(active_seconds_before_resume, resumed_at) >= limit
        {
            state.state = LoopState::TimeLimitReached;
            checkpoint(state_path, state, active_seconds_before_resume, resumed_at)?;
            finish(*quiet, state.iterations, "time limit reached");
            return Ok(());
        }
        if let StopPolicy::Turns(limit) = spec.stop
            && state.iterations >= limit
        {
            state.state = LoopState::TurnLimitReached;
            checkpoint(state_path, state, active_seconds_before_resume, resumed_at)?;
            finish(*quiet, state.iterations, "turn limit reached");
            return Ok(());
        }

        let iteration = state.iterations + 1;
        heading(*quiet, &format!("Iteration {iteration}: work"));
        let work_prompt = render_work_prompt(
            &spec,
            iteration,
            active_elapsed(active_seconds_before_resume, resumed_at),
            state.last_review.as_deref(),
        );
        let work_answer = work_session
            .prompt(
                &work_prompt,
                &profile,
                &tools,
                |delta| stream(*quiet, delta),
                |_metrics: StreamMetrics| {},
                |_running| {},
                |tool_call| note(*quiet, &format!("[tool] {}", tool_call.function.name)),
                |_| false,
            )
            .await
            .context("work phase failed")?;
        stream(*quiet, "\n");

        let checks = run_checks(workspace, &spec.review.checks).await?;
        let diff = review_diff_evidence(workspace);
        heading(*quiet, &format!("Iteration {iteration}: review"));
        let review_prompt = render_review_prompt(&spec, iteration, &work_answer, &diff, &checks);
        let review = review_session
            .prompt_without_tools(
                &review_prompt,
                &profile,
                0,
                |delta| stream(*quiet, delta),
                |_metrics: StreamMetrics| {},
            )
            .await
            .context("review phase failed")?;
        stream(*quiet, "\n");

        state.iterations = iteration;
        state.last_review = Some(truncate_for_review(&review, MAX_SAVED_REVIEW_BYTES));
        checkpoint(state_path, state, active_seconds_before_resume, resumed_at)?;

        if state.state == LoopState::Active
            && matches!(spec.stop, StopPolicy::Goal(_))
            && reviewer_declared_complete(&review)
        {
            state.state = LoopState::Complete;
            checkpoint(state_path, state, active_seconds_before_resume, resumed_at)?;
            finish(*quiet, iteration, "goal condition verified by review");
            return Ok(());
        }
    }
}

impl StopPolicy {
    fn describe(&self) -> String {
        match self {
            Self::Turns(count) => format!("{count} iteration(s)"),
            Self::Time(duration) => format!("{} active second(s)", duration.as_secs()),
            Self::Goal(condition) => format!("until {condition}"),
        }
    }
}

fn render_work_prompt(
    spec: &LoopSpec,
    iteration: u32,
    elapsed: Duration,
    previous_review: Option<&str>,
) -> String {
    let feedback = previous_review.map_or_else(
        || "No prior review is available; establish a working baseline.".to_string(),
        |review| {
            format!(
                "Address this review feedback before pursuing new work:\n{}",
                truncate_for_review(review, MAX_SAVED_REVIEW_BYTES)
            )
        },
    );
    format!(
        "You are in iteration {iteration} of an automated code-and-review loop.\n\n\
Objective:\n{}\n\n\
Stopping policy: {}\nActive time so far: {} seconds.\n\n\
Previous review:\n{feedback}\n\n\
Make concrete progress toward the objective in the current workspace. Inspect the current state, change code when needed, and run relevant validation. Do not merely describe a proposed change. Keep the full objective intact even if this iteration cannot finish it. End with a concise account of changed files, validation, and unresolved risks; a separate reviewer will examine the result.",
        spec.objective,
        spec.stop.describe(),
        elapsed.as_secs(),
    )
}

fn render_review_prompt(
    spec: &LoopSpec,
    iteration: u32,
    work_answer: &str,
    diff: &str,
    checks: &str,
) -> String {
    let rubric = if spec.review.rubric.is_empty() {
        "correctness, regressions, tests, and maintainability".to_string()
    } else {
        spec.review.rubric.join(", ")
    };
    let condition = match &spec.stop {
        StopPolicy::Goal(condition) => condition.as_str(),
        StopPolicy::Turns(_) | StopPolicy::Time(_) => {
            "not applicable; report remaining work honestly"
        }
    };
    format!(
        "Review iteration {iteration}.\n\n\
Objective:\n{}\n\n\
Completion condition:\n{condition}\n\n\
Review rubric: {rubric}\n\n\
Worker report:\n{}\n\n\
Validation command results:\n{}\n\n\
Current workspace diff:\n{}\n\n\
Find concrete defects, missing verification, regressions, and risks. Do not claim the objective is complete without evidence from the supplied diff and validation output. End with exactly one standalone line: `LOOP_COMPLETE: yes` only when the completion condition is proved; otherwise `LOOP_COMPLETE: no`.",
        spec.objective,
        truncate_for_review(work_answer, MAX_REVIEW_INPUT_BYTES / 4),
        truncate_for_review(checks, MAX_REVIEW_INPUT_BYTES / 4),
        truncate_for_review(diff, MAX_REVIEW_INPUT_BYTES / 2),
    )
}

async fn run_checks(workspace: &Path, commands: &[String]) -> Result<String> {
    let workspace = workspace.to_path_buf();
    let commands = commands.to_vec();
    tokio::task::spawn_blocking(move || {
        let mut results = String::new();
        for command in commands {
            let output = shell_command(&workspace, &command)
                .with_context(|| format!("failed to run loop check '{command}'"))?;
            results.push_str(&format!("$ {command}\n"));
            results.push_str(&String::from_utf8_lossy(&output.stdout));
            results.push_str(&String::from_utf8_lossy(&output.stderr));
            results.push_str(&format!("exit: {}\n\n", output.status));
        }
        Ok::<_, anyhow::Error>(if results.is_empty() {
            "No explicit validation commands were configured.".to_string()
        } else {
            truncate_for_review(&results, MAX_CHECK_OUTPUT_BYTES)
        })
    })
    .await
    .context("loop validation task failed")?
}

fn shell_command(workspace: &Path, command: &str) -> std::io::Result<std::process::Output> {
    let (program, args) = orangu::shell::command_parts();
    Command::new(program)
        .args(args)
        .arg(command)
        .current_dir(workspace)
        .output()
}

/// Build the same committed-plus-local evidence used by `/review`. Falling
/// back to a worktree diff keeps loops useful in a non-Git workspace.
fn review_diff_evidence(workspace: &Path) -> String {
    match collect_review_diff(workspace) {
        Ok(review) if review.files.is_empty() => format!(
            "No changes compared with {} (including committed and local changes).",
            review.base_label
        ),
        Ok(review) => {
            let mut evidence = format!(
                "Changes compared with {} (including committed and local changes):\n",
                review.base_label
            );
            for file in review.files {
                evidence.push_str(&format!("\n--- {} ---\n{}\n", file.path, file.patch));
            }
            evidence
        }
        Err(error) => workspace_diff(workspace, &format!("full branch diff unavailable: {error}")),
    }
}

fn workspace_diff(workspace: &Path, reason: &str) -> String {
    let output = Command::new("git")
        .args(["-C"])
        .arg(workspace)
        .args(["diff", "--no-ext-diff", "--no-color"])
        .output();
    match output {
        Ok(output) if output.status.success() => {
            let diff = String::from_utf8_lossy(&output.stdout).to_string();
            if diff.trim().is_empty() {
                format!("{reason}; no uncommitted Git diff is available.")
            } else {
                diff
            }
        }
        Ok(output) => format!(
            "{reason}; Git diff unavailable: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ),
        Err(err) => format!("{reason}; Git diff unavailable: {err}"),
    }
}

fn reviewer_declared_complete(review: &str) -> bool {
    review
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .is_some_and(|line| line.trim().eq_ignore_ascii_case("LOOP_COMPLETE: yes"))
}

fn truncate_for_review(value: &str, limit: usize) -> String {
    if value.len() <= limit {
        return value.to_string();
    }
    let mut end = limit;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n[truncated by Orangu]", &value[..end])
}

fn heading(quiet: bool, text: &str) {
    if !quiet {
        println!("\n--- {text} ---");
    }
}

fn stream(quiet: bool, text: &str) {
    if !quiet {
        print!("{text}");
        let _ = std::io::stdout().flush();
    }
}

fn note(quiet: bool, text: &str) {
    if !quiet {
        eprintln!("{text}");
    }
}

fn finish(quiet: bool, iterations: u32, reason: &str) {
    if !quiet {
        println!("\nLoop stopped after {iterations} iteration(s): {reason}.");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    use std::net::TcpListener;
    use tempfile::NamedTempFile;

    fn find_subsequence(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        haystack
            .windows(needle.len())
            .position(|window| window == needle)
    }

    fn serve_loop_responses(listener: TcpListener) -> std::thread::JoinHandle<Vec<String>> {
        std::thread::spawn(move || {
            let responses = [
                "data: {\"choices\":[{\"delta\":{\"content\":\"work complete\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"LOOP_COMPLETE: yes\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n",
            ];
            let mut bodies = Vec::new();
            for response_body in responses {
                let (mut stream, _) = listener.accept().expect("accept connection");
                let mut request = Vec::new();
                let mut buffer = [0u8; 4096];
                let header_end = loop {
                    let read = stream.read(&mut buffer).expect("read request");
                    request.extend_from_slice(&buffer[..read]);
                    if let Some(position) = find_subsequence(&request, b"\r\n\r\n") {
                        break position + 4;
                    }
                };
                let headers = String::from_utf8_lossy(&request[..header_end]).to_ascii_lowercase();
                let content_length: usize = headers
                    .lines()
                    .find_map(|line| line.strip_prefix("content-length:"))
                    .and_then(|value| value.trim().parse().ok())
                    .unwrap_or(0);
                while request.len() < header_end + content_length {
                    let read = stream.read(&mut buffer).expect("read body");
                    request.extend_from_slice(&buffer[..read]);
                }
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    response_body.len(),
                    response_body
                );
                stream
                    .write_all(response.as_bytes())
                    .expect("write response");
                bodies.push(String::from_utf8_lossy(&request[header_end..]).to_string());
            }
            bodies
        })
    }

    fn args(objective: &[&str]) -> LoopArgs {
        LoopArgs {
            action: None,
            turns: None,
            time: None,
            until: None,
            file: None,
            checks: Vec::new(),
            rubric: Vec::new(),
            objective: objective.iter().map(|value| (*value).to_string()).collect(),
        }
    }

    #[test]
    fn parses_duration_units() {
        assert_eq!(parse_duration("30s").unwrap(), Duration::from_secs(30));
        assert_eq!(parse_duration("2m").unwrap(), Duration::from_secs(120));
        assert_eq!(parse_duration("3h").unwrap(), Duration::from_secs(10_800));
        assert!(parse_duration("0m").is_err());
        assert!(parse_duration("10").is_err());
        assert!(parse_duration("1w").is_err());
    }

    #[test]
    fn requires_one_stopping_policy() {
        assert!(args(&["fix", "parser"]).into_spec().is_err());
        let mut invalid = args(&["fix", "parser"]);
        invalid.turns = Some(2);
        invalid.until = Some("tests pass".to_string());
        assert!(invalid.into_spec().is_err());
    }

    #[test]
    fn builds_turn_limited_spec() {
        let mut input = args(&["fix", "parser"]);
        input.turns = Some(3);
        input.checks.push("cargo test".to_string());
        let spec = input.into_spec().unwrap();
        assert_eq!(spec.objective, "fix parser");
        assert_eq!(spec.stop, StopPolicy::Turns(3));
        assert_eq!(spec.review.checks, vec!["cargo test"]);
    }

    #[test]
    fn workflow_loop_preserves_the_standalone_loop_specification() {
        let spec = LoopSpec::from(WorkflowLoop {
            objective: "Fix parser".to_string(),
            stop: WorkflowLoopStop::Goal("All tests pass".to_string()),
            checks: vec!["cargo test".to_string()],
            rubric: vec!["correctness".to_string()],
        });
        assert_eq!(spec.objective, "Fix parser");
        assert_eq!(spec.stop, StopPolicy::Goal("All tests pass".to_string()));
        assert_eq!(spec.review.checks, vec!["cargo test"]);
        assert_eq!(spec.review.rubric, vec!["correctness"]);
    }

    #[test]
    fn loads_yaml_definition() {
        let file = NamedTempFile::new().unwrap();
        std::fs::write(
            file.path(),
            "version: 1\nobjective: Harden authentication\nstop:\n  type: goal\n  condition: All tests pass\nreview:\n  checks:\n    - cargo test\n  rubric:\n    - security\n",
        )
        .unwrap();
        let spec = LoopSpec::from_file(file.path()).unwrap();
        assert_eq!(spec.objective, "Harden authentication");
        assert_eq!(spec.stop, StopPolicy::Goal("All tests pass".to_string()));
        assert_eq!(spec.review.rubric, vec!["security"]);
    }

    #[test]
    fn completion_marker_must_be_standalone() {
        assert!(reviewer_declared_complete("LOOP_COMPLETE: yes\n"));
        assert!(!reviewer_declared_complete("I think LOOP_COMPLETE: yes."));
        assert!(!reviewer_declared_complete(
            "LOOP_COMPLETE: yes\nBut there is still unresolved work."
        ));
        assert!(!reviewer_declared_complete("LOOP_COMPLETE: no"));
    }

    #[test]
    fn saves_and_loads_resumable_state() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("active.yaml");
        let state = PersistedLoop::new(
            LoopSpec {
                objective: "Harden authentication".to_string(),
                stop: StopPolicy::Turns(3),
                review: ReviewConfig {
                    checks: vec!["cargo test".to_string()],
                    rubric: vec!["security".to_string()],
                },
            },
            Some("code".to_string()),
        );
        save_state(&path, &state).unwrap();
        let loaded = load_state(&path).unwrap();
        assert_eq!(loaded.objective, "Harden authentication");
        assert!(matches!(
            loaded.stop,
            PersistedStopPolicy::Turns { count: 3 }
        ));
        assert_eq!(loaded.state, LoopState::Active);
        assert_eq!(loaded.role.as_deref(), Some("code"));
        assert!(loaded.status_report().contains("Role: code"));
        assert!(loaded.status_report().contains("Completed iterations: 0"));
    }

    #[test]
    fn checkpoint_preserves_resume_baseline() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("active.yaml");
        let mut state = PersistedLoop::new(
            LoopSpec {
                objective: "Fix parser".to_string(),
                stop: StopPolicy::Turns(2),
                review: ReviewConfig::default(),
            },
            None,
        );
        state.active_seconds = 17;
        save_state(&path, &state).unwrap();
        let resumed_at = Instant::now();
        checkpoint(&path, &mut state, 17, resumed_at).unwrap();
        assert!(state.active_seconds >= 17);
        assert!(state.active_seconds < 19);
    }

    #[test]
    fn next_work_turn_receives_prior_review() {
        let spec = LoopSpec {
            objective: "Fix parser".to_string(),
            stop: StopPolicy::Turns(2),
            review: ReviewConfig::default(),
        };
        let prompt = render_work_prompt(
            &spec,
            2,
            Duration::from_secs(5),
            Some("The parser rejects escaped quotes."),
        );
        assert!(prompt.contains("The parser rejects escaped quotes."));
        assert!(prompt.contains("Address this review feedback"));
    }

    #[tokio::test]
    async fn one_iteration_runs_work_then_read_only_review_and_completes() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let endpoint = format!("http://{}/v1", listener.local_addr().expect("address"));
        let server = serve_loop_responses(listener);
        let root = tempfile::tempdir().expect("workspace");
        let config_file = NamedTempFile::new().expect("config");
        std::fs::write(
            config_file.path(),
            format!(
                "[orangu]\nserver = test\nmodel = test-model\ntimeout = 5\n\n[test]\nrole = all\nendpoint = {endpoint}\nmodel = test-model\n"
            ),
        )
        .expect("write config");
        let config =
            orangu::config::load_client_configuration(config_file.path()).expect("load config");
        let mut state = PersistedLoop::new(
            LoopSpec {
                objective: "Finish the parser".to_string(),
                stop: StopPolicy::Goal("The parser is complete".to_string()),
                review: ReviewConfig::default(),
            },
            None,
        );
        let state_path = root.path().join("active.yaml");
        let context = LoopRunContext {
            config,
            workspace: root.path().to_path_buf(),
            role: None,
            quiet: true,
        };

        run_persisted_inner(&mut state, &context, &state_path)
            .await
            .expect("loop iteration");

        assert_eq!(state.state, LoopState::Complete);
        assert_eq!(state.iterations, 1);
        assert_eq!(state.last_review.as_deref(), Some("LOOP_COMPLETE: yes"));
        let requests = server.join().expect("server thread");
        assert!(requests[0].contains("\"tools\""));
        assert!(!requests[1].contains("\"tools\""));
        assert!(requests[1].contains("work complete"));
    }
}
