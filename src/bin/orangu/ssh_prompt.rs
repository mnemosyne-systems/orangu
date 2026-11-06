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

//! The questions an SSH login asks, answered in the input area of the TUI.
//!
//! A login runs inside a tool call, on any tab, and possibly in a background
//! subagent, so the questions wait here for whichever loop is drawing the
//! screen: the idle prompt or a turn in progress.

use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, OnceLock};

/// The label a question is shown under.
pub const PROMPT_LABEL: &str = "SSH";

/// A question waiting for the user.
pub struct SshPromptRequest {
    pub prompt: String,
    reply: tokio::sync::oneshot::Sender<Option<String>>,
}

impl SshPromptRequest {
    /// Whether the login has stopped waiting for the answer.
    pub fn abandoned(&self) -> bool {
        self.reply.is_closed()
    }

    pub fn answer(self, answer: Option<String>) {
        let _ = self.reply.send(answer);
    }
}

type Queue = (Sender<SshPromptRequest>, Mutex<Receiver<SshPromptRequest>>);

fn queue() -> &'static Queue {
    static QUEUE: OnceLock<Queue> = OnceLock::new();
    QUEUE.get_or_init(|| {
        let (sender, receiver) = channel();
        (sender, Mutex::new(receiver))
    })
}

/// The next unanswered question, if any.
pub fn next_request() -> Option<SshPromptRequest> {
    let (_, receiver) = queue();
    receiver.lock().ok()?.try_recv().ok()
}

struct TuiPrompter;

impl orangu::ssh::Prompter for TuiPrompter {
    fn ask(&self, prompt: String) -> tokio::sync::oneshot::Receiver<Option<String>> {
        let (reply, answer) = tokio::sync::oneshot::channel();
        let (sender, _) = queue();
        // Should the queue be gone, the dropped `reply` reads as cancelled.
        let _ = sender.send(SshPromptRequest { prompt, reply });
        answer
    }
}

/// What a TUI conversation's SSH logins ask through.
pub fn prompter() -> Arc<dyn orangu::ssh::Prompter> {
    Arc::new(TuiPrompter)
}

#[cfg(test)]
mod tests {
    use super::*;
    use orangu::ssh::Prompter;

    /// One queue, so these steps run as one test.
    #[test]
    fn a_question_reaches_the_screen_and_its_answer_comes_back() {
        let mut answer = TuiPrompter.ask("Password: ".to_string());
        let request = next_request().expect("queued");
        assert_eq!(request.prompt, "Password: ");
        assert!(!request.abandoned());
        request.answer(Some("x".to_string()));
        assert_eq!(answer.try_recv().unwrap(), Some("x".to_string()));
        assert!(next_request().is_none());

        // Given up on, by a timeout or a cancelled turn.
        let answer = TuiPrompter.ask("Password: ".to_string());
        let request = next_request().expect("queued");
        drop(answer);
        assert!(request.abandoned());
    }
}
