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
use orangu::tui::{Banner, HeaderStatus};

pub(crate) use crate::git::init_git_for_test as init_test_git_repo;

/// One merge flow and one prompt mode exist per process, so the tests that
/// write either take this lock (see [`exclusive_prompt_state`]).
static PROMPT_STATE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub(crate) struct PromptStateGuard(#[allow(dead_code)] std::sync::MutexGuard<'static, ()>);

impl Drop for PromptStateGuard {
    fn drop(&mut self) {
        crate::completion::flow::test_support::reset();
        crate::completion::set_active_pull_requests(&[]);
        crate::mode::set(crate::mode::PromptMode::default());
    }
}

/// Take sole ownership of the prompt state — the merge flow, the cached open
/// pull requests, and the prompt mode — for the rest of the test, starting and
/// finishing at the defaults.
pub(crate) fn exclusive_prompt_state() -> PromptStateGuard {
    let guard = PROMPT_STATE_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    crate::completion::flow::test_support::reset();
    crate::completion::set_active_pull_requests(&[]);
    crate::mode::set(crate::mode::PromptMode::default());
    PromptStateGuard(guard)
}

/// [`exclusive_prompt_state`] on the committer prompt — the mode the merge flow
/// is suggested in, and so the one its tests need.
pub(crate) fn exclusive_committer_prompt() -> PromptStateGuard {
    let guard = exclusive_prompt_state();
    crate::mode::set(crate::mode::PromptMode::Committer);
    guard
}

pub(crate) fn test_profile(endpoint: &str, model: &str) -> LlmConfiguration {
    LlmConfiguration {
        endpoint: endpoint.to_string(),
        model: model.to_string(),
        role: "all".to_string(),
        api_key: None,
        request_timeout_seconds: 1800,
        max_tool_rounds: 10,
        review_max_tokens: 512,
        review_confidence_threshold: 80,
        code_max_tokens: 0,
        system_prompt: "".to_string(),
        model_verbosity: None,
    }
}

pub(crate) fn test_input_context<'a>(workspace: &'a std::path::Path) -> InputContext<'a> {
    static EMPTY_STRINGS: Vec<String> = Vec::new();
    static SKILLS: std::sync::OnceLock<orangu::skills::SkillRegistry> = std::sync::OnceLock::new();
    InputContext {
        history: &EMPTY_STRINGS,
        workspace,
        server_names: &EMPTY_STRINGS,
        available_models: &EMPTY_STRINGS,
        render: RenderContext {
            current_model: "default",
            endpoint: "http://localhost:11434/v1",
            workspace,
            prompt_branch: None,
            header_status: HeaderStatus {
                workspace_ok: true,
                server_ok: orangu::tui::ConnStatus::Ok,
                model_ok: orangu::tui::ConnStatus::Ok,
                is_coordinator: false,
            },
            virtual_width: 80,
            actual_width: 80,
            actual_height: 24,
            x_offset: 0,
            banner: Banner::Left,
            drop_down: true,
            word_wrap: true,
            feedback: false,
            server_names: &[],
            available_models: &[],
            skills: SKILLS
                .get_or_init(|| orangu::skills::SkillRegistry::discover(std::path::Path::new("/"))),
            // Tests run with a single workspace; no tab bar or status dots needed.
            tab_bar: None,
            tab_statuses: &[],
        },
        skills: SKILLS
            .get_or_init(|| orangu::skills::SkillRegistry::discover(std::path::Path::new("/"))),
    }
}
