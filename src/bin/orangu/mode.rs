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

//! What the prompt is for right now: writing code (`/developer`, the default)
//! or landing a reviewed pull request (`/committer`). The mode decides what an
//! untouched prompt offers — a greeting, or the next step of the merge flow —
//! and nothing else; every command stays available in both.
//!
//! Completion reads this on every keystroke, so it is a plain atomic rather
//! than anything the prompt has to wait on.

use std::sync::atomic::{AtomicU8, Ordering};

/// The greeting the developer prompt draws on an untouched line. Rendered like
/// any other ghost, but it is not a command, so Tab never fills it in.
pub const WELCOME_GHOST: &str = "Welcome, I'm orangu";

/// What the prompt is for.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum PromptMode {
    /// Writing code: the prompt greets you and otherwise stays out of the way.
    #[default]
    Developer,
    /// Landing a reviewed pull request: the prompt walks the merge flow (see
    /// `completion::flow`), hinting each step and completing its arguments.
    Committer,
}

impl PromptMode {
    fn from_repr(value: u8) -> PromptMode {
        match value {
            1 => PromptMode::Committer,
            _ => PromptMode::Developer,
        }
    }

    fn repr(self) -> u8 {
        match self {
            PromptMode::Developer => 0,
            PromptMode::Committer => 1,
        }
    }
}

/// The active mode. Read by the completion code on every keystroke, written by
/// the `/developer` and `/committer` commands.
static MODE: AtomicU8 = AtomicU8::new(0);

/// The mode the prompt is in.
pub fn current() -> PromptMode {
    PromptMode::from_repr(MODE.load(Ordering::Relaxed))
}

/// Switch the prompt to `mode`.
pub fn set(mode: PromptMode) {
    MODE.store(mode.repr(), Ordering::Relaxed);
}

/// The greeting to draw on an untouched developer prompt, or `None` once
/// something has been typed or the prompt is in committer mode — where the
/// merge flow's next step has the line instead.
///
/// This is a rendered hint only: it is deliberately kept out of the Tab and
/// Shift+Tab candidates, since there is no command to accept.
pub fn opening_ghost(input: &str) -> Option<&'static str> {
    (input.is_empty() && current() == PromptMode::Developer).then_some(WELCOME_GHOST)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::test_support::exclusive_prompt_state;

    #[test]
    fn the_developer_prompt_opens_with_a_greeting() {
        let _guard = exclusive_prompt_state();
        // The default mode greets an untouched line...
        assert_eq!(current(), PromptMode::Developer);
        assert_eq!(opening_ghost(""), Some(WELCOME_GHOST));
        // ...and says nothing once anything is typed.
        assert_eq!(opening_ghost("p"), None);
        // The committer prompt has the merge flow to show instead.
        set(PromptMode::Committer);
        assert_eq!(opening_ghost(""), None);
    }

    #[test]
    fn the_greeting_is_never_completed_into_the_line() {
        let _guard = exclusive_prompt_state();
        let workspace = tempfile::tempdir().expect("workspace");
        // It is drawn as a hint, but it is not a candidate: Tab and Shift+Tab
        // have nothing to offer an untouched developer prompt.
        assert!(
            crate::completion::ghost_candidates("", workspace.path()).is_empty(),
            "the greeting must not be Tab-completable"
        );
        let mut input_state = crate::input::InputState::default();
        crate::input::apply_completion(
            &mut input_state,
            workspace.path(),
            &[],
            &[],
            &orangu::skills::SkillRegistry::discover(std::path::Path::new("/")),
        );
        assert_eq!(input_state.as_str(), "");
    }
}
