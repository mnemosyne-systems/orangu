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

//! Every subagent run of a conversation, as a task: what it is doing, how it
//! ended, its report, and its conversation, kept so it can be continued.
//!
//! A task runs in the foreground (the main agent waits for its report) or in
//! the background (the main agent carries on; the report is queued as a
//! notification for the client to hand to the model as a later turn). Both
//! are listed by `/tasks`; a background one can be stopped; a finished one
//! can be continued with `continue_agent`.

use crate::config::LlmConfiguration;
use crate::session::ChatSession;
use crate::tools::ToolExecutor;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// How many finished tasks are kept; the oldest go first.
const MAX_FINISHED: usize = 50;

/// The longest task summary `/tasks` shows.
const SUMMARY_CHARS: usize = 80;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    Running,
    Completed,
    /// Ran out of model requests; its report says what is left.
    Partial,
    Failed,
    Stopped,
}

impl TaskStatus {
    pub fn name(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Completed => "completed",
            Self::Partial => "partial",
            Self::Failed => "failed",
            Self::Stopped => "stopped",
        }
    }
}

/// A subagent's conversation and what it needs to take another turn.
pub struct Resumable {
    pub session: ChatSession,
    pub tools: ToolExecutor,
    pub profile: LlmConfiguration,
    pub writable: bool,
    pub max_tool_rounds: usize,
    /// The worktree it works in, with `isolation: worktree`.
    pub worktree: Option<crate::subagent_worktree::Worktree>,
    /// How its tools were built, its model's role, and whether it searches
    /// the web: what rebuilding it after a restart needs.
    pub recipe: Recipe,
    pub role: String,
    pub web_search: bool,
}

/// How a subagent's tools were built, so a task restored from disk can
/// have them built again.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Recipe {
    /// A built-in subagent, by its tool.
    BuiltIn(String),
    /// A document writer, by its type.
    Document(String),
    /// A custom agent, by name.
    Custom(String),
    /// A fork of the conversation.
    Fork,
}

/// What a task needs to be continued after a restart: how to build its
/// tools again, and its conversation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SavedResume {
    pub recipe: Recipe,
    pub role: String,
    pub web_search: bool,
    pub writable: bool,
    pub max_tool_rounds: usize,
    pub worktree: Option<crate::subagent_worktree::Worktree>,
    pub transcript: Vec<crate::llm::ChatMessage>,
}

/// A finished task as it is saved with its session.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct SavedTask {
    id: String,
    agent: String,
    summary: String,
    background: bool,
    status: TaskStatus,
    model: String,
    tool_calls: usize,
    prompt_tokens: u64,
    completion_tokens: u64,
    elapsed_ms: u64,
    result: Option<String>,
    resume: Option<SavedResume>,
}

/// One task as `/tasks` shows it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TaskSnapshot {
    pub id: String,
    pub agent: String,
    pub summary: String,
    pub background: bool,
    pub status: TaskStatus,
    pub model: String,
    pub tool_calls: usize,
    /// The tool it called last: what a running task is doing.
    pub last_tool: Option<String>,
    /// Tokens its model requests read and wrote, as the server reported.
    pub usage: crate::llm::TokenUsage,
    /// The end of what a running task has streamed and called.
    pub activity: String,
    pub elapsed: Duration,
    pub result: Option<String>,
    /// Whether `continue_agent` can take it further.
    pub resumable: bool,
}

/// A finished background task's report, for the model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Notification {
    pub id: String,
    pub agent: String,
    pub status: TaskStatus,
    pub report: String,
}

impl Notification {
    /// The text handed to the model as a turn of its own.
    pub fn message(&self) -> String {
        format!(
            "<subagent-notification task=\"{}\" agent=\"{}\" status=\"{}\">\n{}\n</subagent-notification>\n\
             The background subagent task {} has finished. Use its report to carry on with what it was started for.",
            self.id,
            self.agent,
            self.status.name(),
            self.report.trim(),
            self.id
        )
    }
}

struct Entry {
    id: String,
    agent: String,
    summary: String,
    background: bool,
    status: TaskStatus,
    model: String,
    tool_calls: usize,
    last_tool: Option<String>,
    usage: crate::llm::TokenUsage,
    /// The end of what it has streamed and called, while it runs.
    activity: String,
    started: Instant,
    finished: Option<Instant>,
    result: Option<String>,
    abort: Option<tokio::task::AbortHandle>,
    /// How the user moves this foreground task to the background (Ctrl+B):
    /// set for a task the main conversation waits for.
    detach: Option<Arc<tokio::sync::Notify>>,
    resumable: Option<Arc<tokio::sync::Mutex<Resumable>>>,
    /// A task restored from disk: how to continue it, once rebuilt.
    saved: Option<SavedResume>,
}

#[derive(Default)]
struct Inner {
    next: u32,
    entries: Vec<Entry>,
    undelivered: VecDeque<Notification>,
    /// Where finished tasks are saved, with the session.
    file: Option<std::path::PathBuf>,
}

/// The tasks of one conversation, shared by every clone of its executor.
#[derive(Default)]
pub struct TaskRegistry {
    inner: Mutex<Inner>,
}

/// The most of a task's recent activity kept: its end.
const ACTIVITY_CHARS: usize = 4000;

/// Appends `text` to `activity`, keeping only its last [`ACTIVITY_CHARS`]
/// (on a character boundary).
fn push_activity(activity: &mut String, text: &str) {
    activity.push_str(text);
    if activity.len() > ACTIVITY_CHARS {
        let mut cut = activity.len() - ACTIVITY_CHARS;
        while !activity.is_char_boundary(cut) {
            cut += 1;
        }
        activity.drain(..cut);
    }
}

fn summarize(task: &str) -> String {
    let line = task
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();
    match line.char_indices().nth(SUMMARY_CHARS) {
        Some((cut, _)) => format!("{}…", &line[..cut]),
        None => line.to_string(),
    }
}

impl TaskRegistry {
    fn with<T>(&self, f: impl FnOnce(&mut Inner) -> T) -> T {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        f(&mut inner)
    }

    /// Starts a task for `agent` on `task`; returns its id (`t1`, `t2`, …).
    pub fn start(&self, agent: &str, task: &str, model: &str, background: bool) -> String {
        self.with(|inner| {
            inner.next += 1;
            let id = format!("t{}", inner.next);
            inner.entries.push(Entry {
                id: id.clone(),
                agent: agent.to_string(),
                summary: summarize(task),
                background,
                status: TaskStatus::Running,
                model: model.to_string(),
                tool_calls: 0,
                last_tool: None,
                usage: crate::llm::TokenUsage::default(),
                activity: String::new(),
                detach: None,
                saved: None,
                started: Instant::now(),
                finished: None,
                result: None,
                abort: None,
                resumable: None,
            });
            id
        })
    }

    /// Marks finished task `id` running again for `task`: a continuation.
    pub fn restart(&self, id: &str, task: &str, background: bool) -> bool {
        self.with(
            |inner| match inner.entries.iter_mut().find(|e| e.id == id) {
                Some(entry) if entry.status != TaskStatus::Running => {
                    entry.status = TaskStatus::Running;
                    entry.summary = summarize(task);
                    entry.background = background;
                    entry.started = Instant::now();
                    entry.finished = None;
                    entry.result = None;
                    entry.abort = None;
                    entry.activity.clear();
                    true
                }
                _ => false,
            },
        )
    }

    /// How a background task can be stopped.
    pub fn set_abort(&self, id: &str, abort: tokio::task::AbortHandle) {
        self.with(|inner| {
            if let Some(entry) = inner.entries.iter_mut().find(|e| e.id == id) {
                entry.abort = Some(abort);
            }
        });
    }

    /// Saves finished tasks to `file` from now on (with their conversations,
    /// so they can be continued after a restart), and takes back the ones
    /// saved there before. Their ids carry on where they left off.
    pub fn persist_to(&self, file: std::path::PathBuf) {
        let saved: Vec<SavedTask> = std::fs::read_to_string(&file)
            .ok()
            .and_then(|text| serde_json::from_str(&text).ok())
            .unwrap_or_default();
        self.with(|inner| {
            let now = Instant::now();
            for task in saved {
                if inner.entries.iter().any(|e| e.id == task.id) {
                    continue;
                }
                let number = task.id.trim_start_matches('t').parse::<u32>().unwrap_or(0);
                inner.next = inner.next.max(number);
                let elapsed = Duration::from_millis(task.elapsed_ms);
                inner.entries.push(Entry {
                    id: task.id,
                    agent: task.agent,
                    summary: task.summary,
                    background: task.background,
                    status: task.status,
                    model: task.model,
                    tool_calls: task.tool_calls,
                    last_tool: None,
                    usage: crate::llm::TokenUsage {
                        prompt: task.prompt_tokens,
                        completion: task.completion_tokens,
                    },
                    activity: String::new(),
                    started: now.checked_sub(elapsed).unwrap_or(now),
                    finished: Some(now),
                    result: task.result,
                    abort: None,
                    detach: None,
                    resumable: None,
                    saved: task.resume,
                });
            }
            inner.file = Some(file);
        });
    }

    /// Writes the finished tasks to the file set by [`Self::persist_to`].
    fn save(&self) {
        let (file, saved) = self.with(|inner| {
            let saved: Vec<SavedTask> = inner
                .entries
                .iter()
                .filter(|e| e.status != TaskStatus::Running)
                .map(|e| SavedTask {
                    id: e.id.clone(),
                    agent: e.agent.clone(),
                    summary: e.summary.clone(),
                    background: e.background,
                    status: e.status,
                    model: e.model.clone(),
                    tool_calls: e.tool_calls,
                    prompt_tokens: e.usage.prompt,
                    completion_tokens: e.usage.completion,
                    elapsed_ms: e
                        .finished
                        .map_or(Duration::ZERO, |finished| finished - e.started)
                        .as_millis() as u64,
                    result: e.result.clone(),
                    // A conversation busy in another turn is not read; the
                    // next save has it.
                    resume: match &e.resumable {
                        Some(resumable) => resumable.try_lock().ok().map(|r| SavedResume {
                            recipe: r.recipe.clone(),
                            role: r.role.clone(),
                            web_search: r.web_search,
                            writable: r.writable,
                            max_tool_rounds: r.max_tool_rounds,
                            worktree: r.worktree.clone(),
                            transcript: r.session.messages().to_vec(),
                        }),
                        None => e.saved.clone(),
                    },
                })
                .collect();
            (inner.file.clone(), saved)
        });
        let Some(file) = file else {
            return;
        };
        if let Ok(json) = serde_json::to_string(&saved) {
            let temporary = file.with_extension("tmp");
            if std::fs::write(&temporary, json).is_ok() {
                let _ = std::fs::rename(&temporary, &file);
            }
        }
    }

    /// How to continue task `id` restored from disk, if it is one.
    pub fn restorable(&self, id: &str) -> Option<SavedResume> {
        self.with(|inner| {
            inner
                .entries
                .iter()
                .find(|e| e.id == id && e.status != TaskStatus::Running)
                .and_then(|e| e.saved.clone())
        })
    }
    pub fn set_resumable(&self, id: &str, resumable: Arc<tokio::sync::Mutex<Resumable>>) {
        self.with(|inner| {
            if let Some(entry) = inner.entries.iter_mut().find(|e| e.id == id) {
                entry.resumable = Some(resumable);
                entry.saved = None;
            }
        });
    }

    pub fn resumable(&self, id: &str) -> Option<Arc<tokio::sync::Mutex<Resumable>>> {
        self.with(|inner| {
            inner
                .entries
                .iter()
                .find(|e| e.id == id)
                .and_then(|e| e.resumable.clone())
        })
    }

    pub fn count_tool_call(&self, id: &str, tool: &str) {
        self.with(|inner| {
            if let Some(entry) = inner.entries.iter_mut().find(|e| e.id == id) {
                entry.tool_calls += 1;
                entry.last_tool = Some(tool.to_string());
                push_activity(&mut entry.activity, &format!("\n[{tool}]\n"));
            }
        });
    }

    /// Adds text task `id`'s model streamed to its recent activity.
    pub fn add_output(&self, id: &str, text: &str) {
        self.with(|inner| {
            if let Some(entry) = inner.entries.iter_mut().find(|e| e.id == id) {
                push_activity(&mut entry.activity, text);
            }
        });
    }

    /// Adds what one of task `id`'s model requests cost.
    pub fn add_usage(&self, id: &str, usage: crate::llm::TokenUsage) {
        self.with(|inner| {
            if let Some(entry) = inner.entries.iter_mut().find(|e| e.id == id) {
                entry.usage.prompt += usage.prompt;
                entry.usage.completion += usage.completion;
            }
        });
    }

    /// Records how task `id` ended. A background task's report is queued
    /// for [`Self::take_notifications`]; one that was stopped is not.
    pub fn finish(&self, id: &str, status: TaskStatus, result: &str) {
        self.with(|inner| {
            let Some(entry) = inner.entries.iter_mut().find(|e| e.id == id) else {
                return;
            };
            if entry.status == TaskStatus::Stopped {
                return;
            }
            entry.status = status;
            entry.finished = Some(Instant::now());
            entry.result = Some(result.to_string());
            entry.abort = None;
            if entry.background {
                let notification = Notification {
                    id: entry.id.clone(),
                    agent: entry.agent.clone(),
                    status,
                    report: result.to_string(),
                };
                inner.undelivered.push_back(notification);
            }
            // Keep the list bounded: drop the oldest finished tasks.
            while inner
                .entries
                .iter()
                .filter(|e| e.status != TaskStatus::Running)
                .count()
                > MAX_FINISHED
            {
                if let Some(oldest) = inner
                    .entries
                    .iter()
                    .position(|e| e.status != TaskStatus::Running)
                {
                    inner.entries.remove(oldest);
                }
            }
        });
        self.save();
    }

    /// Stops running background task `id`.
    pub fn stop(&self, id: &str) -> Result<(), String> {
        let stopped = self.with(|inner| {
            let Some(entry) = inner.entries.iter_mut().find(|e| e.id == id) else {
                return Err(format!("no task {id}"));
            };
            if entry.status != TaskStatus::Running {
                return Err(format!("task {id} is not running"));
            }
            let Some(abort) = entry.abort.take() else {
                return Err(format!(
                    "task {id} runs in the foreground; stop the turn with Esc"
                ));
            };
            abort.abort();
            entry.status = TaskStatus::Stopped;
            entry.finished = Some(Instant::now());
            entry.result = Some("Stopped by the user.".to_string());
            // Cut off mid-turn, its conversation may end in a tool call
            // without a result: not something to continue.
            entry.resumable = None;
            Ok(())
        });
        self.save();
        stopped
    }

    /// Lets the user move foreground task `id` to the background; the turn
    /// waiting for it waits on the signal returned too.
    pub fn detachable(&self, id: &str) -> Arc<tokio::sync::Notify> {
        let signal = Arc::new(tokio::sync::Notify::new());
        self.with(|inner| {
            if let Some(entry) = inner.entries.iter_mut().find(|e| e.id == id) {
                entry.detach = Some(signal.clone());
            }
        });
        signal
    }

    /// Asks every running foreground task that can be moved to move to the
    /// background (Ctrl+B); how many were asked.
    pub fn detach_foreground(&self) -> usize {
        self.with(|inner| {
            let mut asked = 0;
            for entry in &mut inner.entries {
                if entry.status == TaskStatus::Running
                    && !entry.background
                    && let Some(signal) = &entry.detach
                {
                    signal.notify_one();
                    asked += 1;
                }
            }
            asked
        })
    }

    /// Moves task `id` to the background, if it is still running: its report
    /// then arrives as a notification. `false` when it has already ended.
    pub fn move_to_background(&self, id: &str, abort: tokio::task::AbortHandle) -> bool {
        self.with(
            |inner| match inner.entries.iter_mut().find(|e| e.id == id) {
                Some(entry) if entry.status == TaskStatus::Running => {
                    entry.background = true;
                    entry.detach = None;
                    entry.abort = Some(abort);
                    true
                }
                _ => false,
            },
        )
    }

    /// Marks task `id`, if still running, stopped because the turn that was
    /// waiting for it was cancelled. Like a stop, it cannot be continued.
    pub fn interrupt(&self, id: &str) {
        self.with(|inner| {
            if let Some(entry) = inner.entries.iter_mut().find(|e| e.id == id)
                && entry.status == TaskStatus::Running
            {
                entry.status = TaskStatus::Stopped;
                entry.finished = Some(Instant::now());
                entry.result = Some("Stopped: the turn waiting for it was cancelled.".to_string());
                entry.abort = None;
                entry.resumable = None;
            }
        });
    }

    /// The finished background reports not handed over yet, oldest first.
    pub fn take_notifications(&self) -> Vec<Notification> {
        self.with(|inner| inner.undelivered.drain(..).collect())
    }

    /// Whether any background task is still running.
    pub fn background_running(&self) -> bool {
        self.background_count() > 0
    }

    /// How many background tasks are running.
    pub fn background_count(&self) -> usize {
        self.with(|inner| {
            inner
                .entries
                .iter()
                .filter(|e| e.background && e.status == TaskStatus::Running)
                .count()
        })
    }

    pub fn has_notifications(&self) -> bool {
        self.with(|inner| !inner.undelivered.is_empty())
    }

    pub fn list(&self) -> Vec<TaskSnapshot> {
        self.with(|inner| {
            inner
                .entries
                .iter()
                .map(|e| TaskSnapshot {
                    id: e.id.clone(),
                    agent: e.agent.clone(),
                    summary: e.summary.clone(),
                    background: e.background,
                    status: e.status,
                    model: e.model.clone(),
                    tool_calls: e.tool_calls,
                    last_tool: e.last_tool.clone(),
                    usage: e.usage,
                    activity: e.activity.clone(),
                    elapsed: e.finished.unwrap_or_else(Instant::now) - e.started,
                    result: e.result.clone(),
                    resumable: (e.resumable.is_some() || e.saved.is_some())
                        && e.status != TaskStatus::Running,
                })
                .collect()
        })
    }

    pub fn get(&self, id: &str) -> Option<TaskSnapshot> {
        self.list().into_iter().find(|task| task.id == id)
    }
}

/// The registry whose running background tasks the status line counts: the
/// active tab's. Held weakly, so a closed tab's registry is not kept alive.
static SHOWN: Mutex<Option<std::sync::Weak<TaskRegistry>>> = Mutex::new(None);

/// Makes `tasks` the registry the status line counts.
pub fn show_in_status_line(tasks: &Arc<TaskRegistry>) {
    if let Ok(mut shown) = SHOWN.lock() {
        let same = shown
            .as_ref()
            .is_some_and(|current| std::ptr::eq(current.as_ptr(), Arc::as_ptr(tasks)));
        if !same {
            *shown = Some(Arc::downgrade(tasks));
        }
    }
}

/// Ctrl+B in the active tab: [`TaskRegistry::detach_foreground`] on the
/// registry the status line shows, which is that tab's.
pub fn detach_shown_foreground() -> usize {
    SHOWN
        .lock()
        .ok()
        .and_then(|shown| shown.as_ref().and_then(std::sync::Weak::upgrade))
        .map_or(0, |tasks| tasks.detach_foreground())
}

/// How many background tasks the status line's registry is running.
pub fn shown_background_count() -> usize {
    SHOWN
        .lock()
        .ok()
        .and_then(|shown| shown.as_ref().and_then(std::sync::Weak::upgrade))
        .map_or(0, |tasks| tasks.background_count())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tasks_are_numbered_and_summarised() {
        let tasks = TaskRegistry::default();
        let long = format!("\n  {}\nsecond line", "word ".repeat(40));
        let a = tasks.start("research_web", "What is X?", "code", false);
        let b = tasks.start("api-designer", &long, "code", true);
        assert_eq!((a.as_str(), b.as_str()), ("t1", "t2"));
        let listed = tasks.list();
        assert_eq!(listed[0].summary, "What is X?");
        assert!(listed[1].summary.ends_with('…'));
        assert!(listed[1].summary.chars().count() <= SUMMARY_CHARS + 1);
        assert!(listed.iter().all(|t| t.status == TaskStatus::Running));
        assert!(tasks.background_running());
    }

    #[test]
    fn only_background_reports_become_notifications() {
        let tasks = TaskRegistry::default();
        let foreground = tasks.start("explore_repository", "find it", "explorer", false);
        let background = tasks.start("research_web", "look it up", "code", true);
        tasks.count_tool_call(&background, "show_file");
        tasks.count_tool_call(&background, "show_file");
        tasks.finish(&foreground, TaskStatus::Completed, "src/a.rs:1-9");
        tasks.finish(&background, TaskStatus::Partial, "[Partial: …] half done");
        assert!(tasks.has_notifications());
        let delivered = tasks.take_notifications();
        assert_eq!(delivered.len(), 1);
        assert_eq!(delivered[0].id, background);
        assert_eq!(delivered[0].status, TaskStatus::Partial);
        let message = delivered[0].message();
        assert!(message.starts_with(
            "<subagent-notification task=\"t2\" agent=\"research_web\" status=\"partial\">\n[Partial: …] half done\n</subagent-notification>"
        ));
        assert!(tasks.take_notifications().is_empty());
        let task = tasks.get(&background).unwrap();
        assert_eq!(task.tool_calls, 2);
        assert_eq!(task.result.as_deref(), Some("[Partial: …] half done"));
        assert!(!tasks.background_running());
    }

    #[test]
    fn usage_adds_up_and_the_status_line_counts_the_shown_registry() {
        let tasks = Arc::new(TaskRegistry::default());
        let id = tasks.start("research_web", "look", "code", true);
        for (prompt, completion) in [(1_000, 50), (1_200, 70)] {
            tasks.add_usage(&id, crate::llm::TokenUsage { prompt, completion });
        }
        let usage = tasks.get(&id).unwrap().usage;
        assert_eq!((usage.prompt, usage.completion), (2_200, 120));

        tasks.start("explore_repository", "find", "explorer", false);
        show_in_status_line(&tasks);
        assert_eq!(shown_background_count(), 1);
        tasks.finish(&id, TaskStatus::Completed, "done");
        assert_eq!(shown_background_count(), 0);
        // A closed tab's registry is not kept alive by the status line.
        drop(tasks);
        assert_eq!(shown_background_count(), 0);
    }

    #[test]
    fn a_running_task_keeps_the_end_of_its_activity() {
        let tasks = TaskRegistry::default();
        let id = tasks.start("research_web", "look", "code", true);
        tasks.add_output(&id, "Searching first.");
        tasks.count_tool_call(&id, "web_search");
        tasks.add_output(&id, "Found it.");
        assert_eq!(
            tasks.get(&id).unwrap().activity,
            "Searching first.\n[web_search]\nFound it."
        );
        // Only the end is kept, cut on a character boundary.
        tasks.add_output(&id, &"é".repeat(ACTIVITY_CHARS));
        let activity = tasks.get(&id).unwrap().activity;
        assert!(activity.len() <= ACTIVITY_CHARS && activity.ends_with('é'));
        assert!(!activity.contains("Found it."));
        // A continuation starts afresh.
        tasks.finish(&id, TaskStatus::Completed, "done");
        assert!(tasks.restart(&id, "more", true));
        assert!(tasks.get(&id).unwrap().activity.is_empty());
    }

    #[tokio::test]
    async fn a_background_task_can_be_stopped_and_stays_stopped() {
        let tasks = TaskRegistry::default();
        let id = tasks.start("research_web", "slow", "code", true);
        let handle = tokio::spawn(async {
            tokio::time::sleep(Duration::from_secs(60)).await;
        });
        tasks.set_abort(&id, handle.abort_handle());
        tasks.stop(&id).unwrap();
        assert!(handle.await.unwrap_err().is_cancelled());
        // A late finish does not undo the stop, nor notify.
        tasks.finish(&id, TaskStatus::Completed, "done anyway");
        let task = tasks.get(&id).unwrap();
        assert_eq!(task.status, TaskStatus::Stopped);
        assert!(!tasks.has_notifications());
        assert!(tasks.stop(&id).unwrap_err().contains("not running"));
        assert!(tasks.stop("t99").unwrap_err().contains("no task"));

        let foreground = tasks.start("write_tests", "x", "code", false);
        assert!(tasks.stop(&foreground).unwrap_err().contains("Esc"));
    }

    #[test]
    fn a_finished_task_can_run_again() {
        let tasks = TaskRegistry::default();
        let id = tasks.start("code-reviewer", "review auth", "code", false);
        assert!(!tasks.restart(&id, "again", false));
        tasks.finish(&id, TaskStatus::Completed, "ok");
        assert!(tasks.restart(&id, "now authorization", true));
        let task = tasks.get(&id).unwrap();
        assert_eq!(task.status, TaskStatus::Running);
        assert_eq!(task.summary, "now authorization");
        assert!(task.background);
        assert!(task.result.is_none());
    }

    #[test]
    fn only_the_newest_finished_tasks_are_kept() {
        let tasks = TaskRegistry::default();
        let running = tasks.start("long", "keeps going", "code", false);
        for n in 0..(MAX_FINISHED + 5) {
            let id = tasks.start("quick", &format!("task {n}"), "code", false);
            tasks.finish(&id, TaskStatus::Completed, "ok");
        }
        let listed = tasks.list();
        assert_eq!(listed.len(), MAX_FINISHED + 1);
        assert!(listed.iter().any(|t| t.id == running));
        assert!(!listed.iter().any(|t| t.summary == "task 0"));
    }
}
