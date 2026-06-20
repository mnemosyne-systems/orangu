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

pub(crate) fn test_profile(provider: &str, endpoint: &str, model: &str) -> LlmConfiguration {
    LlmConfiguration {
        provider: provider.to_string(),
        endpoint: endpoint.to_string(),
        model: model.to_string(),
        api_key: None,
        request_timeout_seconds: 1800,
        max_tool_rounds: 10,
        review_max_tokens: 512,
        code_max_tokens: 0,
        system_prompt: String::new(),
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
                server_ok: true,
                model_ok: true,
            },
            virtual_width: 80,
            actual_width: 80,
            actual_height: 24,
            x_offset: 0,
            banner: Banner::Left,
            feedback: false,
            server_names: &[],
            available_models: &[],
            skills: SKILLS
                .get_or_init(|| orangu::skills::SkillRegistry::discover(std::path::Path::new("/"))),
        },
        skills: SKILLS
            .get_or_init(|| orangu::skills::SkillRegistry::discover(std::path::Path::new("/"))),
    }
}
