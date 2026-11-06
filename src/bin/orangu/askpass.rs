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

//! Credential prompts from the startup `git`/`gh`/`glab` calls, answered in
//! the input area.
//!
//! Those calls run in the background while the TUI owns the terminal, so a
//! password or SSH passphrase prompt written straight to `/dev/tty` would land
//! in the middle of the screen with raw-mode input it cannot read. Instead
//! orangu registers itself as `GIT_ASKPASS` / `SSH_ASKPASS` for them. The
//! helper invocation ([`run_helper`]) forwards the prompt over a loopback
//! socket to the running TUI, which shows it in the input area (see
//! [`read_answer`]) and sends back what was typed.

use std::cell::Cell;
use std::ffi::OsString;
use std::io::{IsTerminal, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Command, ExitCode};
use std::sync::OnceLock;
use std::sync::mpsc::{Receiver, Sender, channel};

use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};

use crate::input::{RenderContext, ScreenState};
use crate::terminal::TerminalUiGuard;

/// `<address> <token>` of the running TUI's prompt socket, handed to the
/// helper invocation.
const ASKPASS_ENV: &str = "ORANGU_ASKPASS";

/// The environment given to commands spawned inside [`with_prompts`], set
/// once [`start`] has bound the socket.
static COMMAND_ENV: OnceLock<Vec<(&'static str, OsString)>> = OnceLock::new();

thread_local! {
    static PROMPTS_ENABLED: Cell<bool> = const { Cell::new(false) };
}

/// A credential prompt waiting for the user's answer.
pub struct AskpassRequest {
    pub prompt: String,
    reply: Sender<Option<String>>,
}

impl AskpassRequest {
    /// Send the answer back to the waiting helper; `None` cancels the prompt.
    pub fn answer(self, answer: Option<String>) {
        let _ = self.reply.send(answer);
    }
}

/// Bind the prompt socket and return the receiver of incoming prompts. `None`
/// when the socket or the orangu executable cannot be resolved, in which case
/// commands keep their default prompting.
pub fn start() -> Option<Receiver<AskpassRequest>> {
    let exe = std::env::current_exe().ok()?;
    let listener = TcpListener::bind(("127.0.0.1", 0)).ok()?;
    let address = listener.local_addr().ok()?;
    let token = uuid::Uuid::new_v4().simple().to_string();

    let env = vec![
        ("GIT_ASKPASS", exe.clone().into_os_string()),
        ("SSH_ASKPASS", exe.into_os_string()),
        // OpenSSH only consults SSH_ASKPASS without a terminal unless forced.
        ("SSH_ASKPASS_REQUIRE", OsString::from("force")),
        (ASKPASS_ENV, OsString::from(format!("{address} {token}"))),
    ];
    if COMMAND_ENV.set(env).is_err() {
        return None;
    }

    let (sender, receiver) = channel();
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            let sender = sender.clone();
            let token = token.clone();
            std::thread::spawn(move || serve(stream, &token, &sender));
        }
    });
    Some(receiver)
}

/// Read one helper's prompt, queue it for the TUI, and write back the answer:
/// `1\n<answer>` when given, `0` when cancelled.
fn serve(mut stream: TcpStream, token: &str, sender: &Sender<AskpassRequest>) {
    let mut request = String::new();
    if stream.read_to_string(&mut request).is_err() {
        return;
    }
    let Some((sent_token, prompt)) = request.split_once('\n') else {
        return;
    };
    if sent_token != token {
        return;
    }
    let (reply, answer) = channel();
    if sender
        .send(AskpassRequest {
            prompt: prompt.to_string(),
            reply,
        })
        .is_err()
    {
        return;
    }
    let response = match answer.recv() {
        Ok(Some(answer)) => format!("1\n{answer}"),
        _ => "0".to_string(),
    };
    let _ = stream.write_all(response.as_bytes());
}

/// Run `f` with prompts routed to the input area for every command it builds
/// through [`apply`] on this thread. Only background work may do this: a
/// command run on the UI thread would wait on a prompt the UI thread can never
/// show.
pub fn with_prompts<T>(f: impl FnOnce() -> T) -> T {
    PROMPTS_ENABLED.with(|enabled| enabled.set(true));
    let result = f();
    PROMPTS_ENABLED.with(|enabled| enabled.set(false));
    result
}

/// Point `command`'s credential prompts at the input area when called inside
/// [`with_prompts`] after [`start`]; otherwise leave it untouched.
pub fn apply(command: &mut Command) -> &mut Command {
    if PROMPTS_ENABLED.with(Cell::get)
        && let Some(env) = COMMAND_ENV.get()
    {
        command.envs(env.iter().map(|(key, value)| (key, value)));
    }
    command
}

/// When this process was started by git or ssh as the askpass helper, forward
/// the prompt to the TUI, print the answer, and return the exit code. `None`
/// for a normal orangu start.
pub fn run_helper() -> Option<ExitCode> {
    let target = std::env::var(ASKPASS_ENV).ok()?;
    let mut args = std::env::args_os().skip(1);
    let (Some(prompt), None) = (args.next(), args.next()) else {
        return None;
    };
    // git and ssh read the answer from a pipe; a terminal on stdout means
    // orangu was started by hand from a shell that inherited the variable.
    if std::io::stdout().is_terminal() {
        return None;
    }
    let (address, token) = target.split_once(' ')?;
    let answer = ask(address, token, &prompt.to_string_lossy());
    Some(match answer {
        Some(answer) => {
            println!("{answer}");
            ExitCode::SUCCESS
        }
        None => ExitCode::FAILURE,
    })
}

fn ask(address: &str, token: &str, prompt: &str) -> Option<String> {
    let mut stream = TcpStream::connect(address).ok()?;
    stream
        .write_all(format!("{token}\n{prompt}").as_bytes())
        .ok()?;
    stream.shutdown(std::net::Shutdown::Write).ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    response.strip_prefix("1\n").map(str::to_string)
}

/// Whether the typed answer to `prompt` is hidden: everything but a user name
/// or an SSH yes/no confirmation.
fn is_secret(prompt: &str) -> bool {
    let prompt = prompt.to_lowercase();
    !(prompt.starts_with("username") || prompt.contains("yes/no"))
}

/// The input-area text for `prompt`: prefixed with the forge, folded onto one
/// line (SSH host-key prompts span several), and followed by the answer typed
/// so far — masked when it is a secret.
fn prompt_line(forge: &str, prompt: &str, answer: &str) -> String {
    let prompt = prompt.split_whitespace().collect::<Vec<_>>().join(" ");
    let shown = if is_secret(&prompt) {
        "*".repeat(answer.chars().count())
    } else {
        answer.to_string()
    };
    format!("{forge}: {prompt} {shown}")
}

/// Show `request` in the input area, prefixed with `forge`, and read the
/// answer: Enter submits it, Esc or Ctrl+C cancels the prompt. Ends with `None`
/// as well once `abandoned` says nobody waits any more.
pub fn read_answer(
    guard: &mut TerminalUiGuard,
    render: RenderContext<'_>,
    screen: &ScreenState<'_>,
    forge: &str,
    prompt: &str,
    abandoned: impl Fn() -> bool,
) -> anyhow::Result<Option<String>> {
    read_answer_with(
        |line| {
            guard.print_prompt_screen(
                render,
                ScreenState {
                    left_status: None,
                    pending_lines: &[],
                    input: line,
                    cursor: line.len(),
                    ghost_index: 0,
                    dropdown: None,
                    reverse_search: None,
                    ..*screen
                },
            );
        },
        forge,
        prompt,
        abandoned,
    )
}

/// How often an open prompt checks whether it is still wanted.
const PROMPT_POLL: std::time::Duration = std::time::Duration::from_millis(100);

/// [`read_answer`], drawing each state of the input line with `draw`.
pub fn read_answer_with(
    mut draw: impl FnMut(&str),
    forge: &str,
    prompt: &str,
    abandoned: impl Fn() -> bool,
) -> anyhow::Result<Option<String>> {
    let mut answer = String::new();
    loop {
        let line = prompt_line(forge, prompt, &answer);
        draw(&line);
        std::io::stdout().flush()?;

        while !event::poll(PROMPT_POLL)? {
            if abandoned() {
                return Ok(None);
            }
        }
        match event::read()? {
            Event::Key(KeyEvent {
                code,
                modifiers,
                kind: KeyEventKind::Press | KeyEventKind::Repeat,
                ..
            }) => match code {
                KeyCode::Enter => return Ok(Some(answer)),
                KeyCode::Esc => return Ok(None),
                KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                    return Ok(None);
                }
                KeyCode::Char('u') if modifiers.contains(KeyModifiers::CONTROL) => answer.clear(),
                KeyCode::Backspace => {
                    answer.pop();
                }
                KeyCode::Char(ch) if !modifiers.contains(KeyModifiers::CONTROL) => answer.push(ch),
                _ => {}
            },
            Event::Paste(text) => answer.push_str(text.trim_end_matches(['\r', '\n'])),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passwords_and_passphrases_are_masked() {
        assert_eq!(
            prompt_line(
                "GitHub",
                "Password for 'https://me@github.com': ",
                "hunter2"
            ),
            "GitHub: Password for 'https://me@github.com': *******"
        );
        assert_eq!(
            prompt_line(
                "GitLab",
                "Enter passphrase for key '/home/me/.ssh/id_ed25519': ",
                "ab"
            ),
            "GitLab: Enter passphrase for key '/home/me/.ssh/id_ed25519': **"
        );
    }

    #[test]
    fn user_names_and_confirmations_are_shown() {
        assert_eq!(
            prompt_line("GitHub", "Username for 'https://github.com': ", "me"),
            "GitHub: Username for 'https://github.com': me"
        );
        assert_eq!(
            prompt_line(
                "GitLab",
                "The authenticity of host 'gitlab.com' can't be established.\nAre you sure you want to continue connecting (yes/no/[fingerprint])? ",
                "yes"
            ),
            "GitLab: The authenticity of host 'gitlab.com' can't be established. Are you sure you want to continue connecting (yes/no/[fingerprint])? yes"
        );
    }

    #[test]
    fn a_prompt_round_trips_through_the_socket() {
        let listener = TcpListener::bind(("127.0.0.1", 0)).expect("bind");
        let address = listener.local_addr().expect("address").to_string();
        let (sender, receiver) = channel();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten().take(3) {
                serve(stream, "token", &sender);
            }
        });
        let answers = std::thread::spawn(move || {
            for answer in [Some("secret".to_string()), None] {
                let request: AskpassRequest = receiver.recv().expect("request");
                assert_eq!(request.prompt, "Password: ");
                request.answer(answer);
            }
        });

        assert_eq!(
            ask(&address, "token", "Password: ").as_deref(),
            Some("secret")
        );
        assert_eq!(ask(&address, "token", "Password: "), None);
        // A wrong token is dropped without reaching the TUI.
        assert_eq!(ask(&address, "wrong", "Password: "), None);
        answers.join().expect("answers");
    }
}
