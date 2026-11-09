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

//! The Option key on a Mac. Unless the terminal sends Option as Meta, `⌥S`
//! arrives as `ß`, `⌥J` as `∆` and so on, and no `Alt+…` binding fires.
//! Where a screen expects a binding rather than text, such a character is
//! read as the `Alt` key it came from, and the user is told once which
//! terminal setting makes Option work everywhere.

use std::sync::atomic::{AtomicBool, Ordering};

use crossterm::event::{KeyCode, KeyModifiers};

/// Whether Option characters are read as `Alt` bindings: on macOS only,
/// where the keyboard produces them.
pub(crate) const ACTIVE: bool = cfg!(target_os = "macos");

/// The key a US-layout Option chord types, by the character it types. The
/// vowels that start an accent (`⌥E`, `⌥I`, `⌥N`, `⌥U`) type nothing alone
/// and are not listed.
pub(crate) fn option_key(ch: char) -> Option<char> {
    Some(match ch {
        'å' => 'a',
        '∫' => 'b',
        'ç' => 'c',
        '∂' => 'd',
        'ƒ' => 'f',
        '©' => 'g',
        '˙' => 'h',
        '∆' => 'j',
        '˚' => 'k',
        '¬' => 'l',
        'µ' => 'm',
        'ø' => 'o',
        'π' => 'p',
        'œ' => 'q',
        '®' => 'r',
        'ß' => 's',
        '†' => 't',
        '√' => 'v',
        '∑' => 'w',
        '≈' => 'x',
        '¥' => 'y',
        'Ω' => 'z',
        '≤' => ',',
        '≥' => '.',
        _ => return None,
    })
}

/// `code` read as the `Alt` key an Option chord meant, when `active`, the
/// key carries no modifier besides Shift, and `expects_binding` (the screen
/// is not taking text just now). Records that the hint is due.
pub(crate) fn as_binding(
    code: KeyCode,
    modifiers: KeyModifiers,
    active: bool,
    expects_binding: bool,
) -> (KeyCode, KeyModifiers) {
    if !active || !expects_binding || !(modifiers - KeyModifiers::SHIFT).is_empty() {
        return (code, modifiers);
    }
    let KeyCode::Char(ch) = code else {
        return (code, modifiers);
    };
    match option_key(ch) {
        Some(key) => {
            note_option_character();
            (KeyCode::Char(key), KeyModifiers::ALT)
        }
        None => (code, modifiers),
    }
}

/// Whether the hint was shown (or is waiting to be).
static NOTED: AtomicBool = AtomicBool::new(false);
/// Whether the hint is waiting to be shown.
static PENDING: AtomicBool = AtomicBool::new(false);

/// Records that an Option character arrived where a binding was expected.
pub(crate) fn note_option_character() {
    if !NOTED.swap(true, Ordering::Relaxed) {
        PENDING.store(true, Ordering::Relaxed);
    }
}

/// The hint, once per run, after an Option character arrived.
pub(crate) fn take_hint() -> Option<String> {
    PENDING
        .swap(false, Ordering::Relaxed)
        .then(|| hint(&terminal_name()))
}

/// The one-line hint naming the setting of the terminal `name`
/// (`$TERM_PROGRAM`, or `$TERM` for kitty and Alacritty).
pub(crate) fn hint(name: &str) -> String {
    let setting = setting_for(name)
        .unwrap_or("your terminal's setting that sends Option as Meta (or Esc+, or Alt)");
    format!(
        "Option typed a character, so orangu read it as the Alt key it meant here. \
         For Option to work as Alt everywhere, turn on {setting}; \
         see \"Alt on macOS\" in /manual."
    )
}

/// The setting that makes Option send Meta in the terminal `name`.
fn setting_for(name: &str) -> Option<&'static str> {
    let name = name.to_ascii_lowercase();
    Some(if name == "apple_terminal" {
        "Settings → Profiles → Keyboard → \"Use Option as Meta key\""
    } else if name.starts_with("iterm") {
        "Settings → Profiles → Keys → General → Left Option key → \"Esc+\""
    } else if name.contains("ghostty") {
        "`macos-option-as-alt = left` in Ghostty's config"
    } else if name.contains("wezterm") {
        "`send_composed_key_when_left_alt_is_pressed = false` in WezTerm's config"
    } else if name.contains("kitty") {
        "`macos_option_as_alt left` in kitty.conf"
    } else if name.contains("alacritty") {
        "`option_as_alt = \"OnlyLeft\"` under [window] in Alacritty's config"
    } else {
        return None;
    })
}

/// Which terminal this is, as far as the environment says.
fn terminal_name() -> String {
    std::env::var("TERM_PROGRAM")
        .ok()
        .filter(|name| !name.is_empty())
        .or_else(|| std::env::var("TERM").ok())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn option_characters_become_their_alt_keys_where_a_binding_is_expected() {
        let none = KeyModifiers::NONE;
        assert_eq!(
            as_binding(KeyCode::Char('ß'), none, true, true),
            (KeyCode::Char('s'), KeyModifiers::ALT)
        );
        assert_eq!(
            as_binding(KeyCode::Char('≥'), none, true, true),
            (KeyCode::Char('.'), KeyModifiers::ALT)
        );
        // Text is left alone: while typing, off macOS, or for other keys.
        assert_eq!(
            as_binding(KeyCode::Char('ß'), none, true, false),
            (KeyCode::Char('ß'), none)
        );
        assert_eq!(
            as_binding(KeyCode::Char('ß'), none, false, true),
            (KeyCode::Char('ß'), none)
        );
        assert_eq!(
            as_binding(KeyCode::Char('s'), none, true, true),
            (KeyCode::Char('s'), none)
        );
        assert_eq!(
            as_binding(KeyCode::Char('ß'), KeyModifiers::CONTROL, true, true),
            (KeyCode::Char('ß'), KeyModifiers::CONTROL)
        );
    }

    #[test]
    fn the_hint_names_the_terminal_setting() {
        assert!(hint("Apple_Terminal").contains("Use Option as Meta key"));
        assert!(hint("iTerm.app").contains("Esc+"));
        assert!(hint("xterm-kitty").contains("macos_option_as_alt"));
        assert!(hint("something-else").contains("sends Option as Meta"));
    }

    #[test]
    fn the_hint_is_given_once() {
        note_option_character();
        assert!(take_hint().is_some());
        note_option_character();
        assert!(take_hint().is_none());
    }
}
