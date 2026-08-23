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

//! How a boolean `ORANGU_*` tuning flag is read.
//!
//! One function, because the alternative was sixty copies of
//! `std::env::var_os("ORANGU_X").is_some()` — under which **`ORANGU_X=0`
//! switches the feature on**. Every one of those flags exists to be swept, and
//! a sweep of `0,1` under that spelling runs the feature on both arms and
//! reports the difference between a thing and itself. That is the worst shape
//! a measurement bug can take, because every number it produces looks
//! reasonable and nothing ever fails.
//!
//! The value flags have had a shared reader for a while
//! ([`crate::engine::backend::env_tuning_value`], which also warns on a value
//! it cannot parse rather than silently keeping the default). This is that
//! idea's other half, and it lives at `engine` level rather than under
//! `engine::backend` because the flags using it are spread across the whole
//! binary — `main`, the architectures, the backends and the scheduler — not
//! just the two shader modules.

/// Whether a boolean tuning flag is switched **on**.
///
/// `1`, `true`, `yes`, `on`, or any other value reads as on. **`0`, `false`,
/// `no`, `off`, and the empty string read as off**, as does an unset variable.
///
/// The off-list is what makes these flags sweepable: `FLAG=0` has to be a real
/// control arm rather than a second copy of the treatment. It is deliberately
/// generous about what counts as "off" and deliberately permissive about what
/// counts as "on" — a typo like `ORANGU_X=ture` turning the feature on is a
/// visible, self-correcting mistake, where `ORANGU_X=0` turning it on is not.
///
/// Whitespace is trimmed and case is ignored, because these are typed by hand
/// into shell one-liners and pasted into bug reports.
pub(crate) fn flag_on(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "off" | "no"
        ),
        // A variable that is not set, or whose value is not UTF-8, is off.
        // Neither is anyone asking for the feature.
        Err(_) => false,
    }
}

/// Whether a boolean tuning flag is switched **on**, for a feature that is on
/// unless someone turns it off.
///
/// Same off-list as [`flag_on`] — `0`, `false`, `no`, `off`, the empty string —
/// and the same reason for it: these are typed into shell one-liners, and a
/// sweep of `0,1` has to give a real control arm. The difference is only what
/// an *unset* variable means.
///
/// Deliberately a second function rather than a `default` argument, so that
/// reading a call site tells you which way the feature points without having to
/// find the constant.
pub(crate) fn flag_on_unless_disabled(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "" | "0" | "false" | "off" | "no"
        ),
        Err(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `ORANGU_*` flags are process-wide, so the tests that set one cannot run
    /// beside each other.
    static ENV: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// The whole reason this function exists.
    ///
    /// Under the `var_os(..).is_some()` spelling it replaced, every one of
    /// these reads as **on** — so `--sweep FLAG=0,1` measured the feature
    /// against itself and reported a difference of zero as a finding.
    #[test]
    fn zero_and_its_synonyms_are_off() {
        let _guard = ENV.lock().unwrap_or_else(|e| e.into_inner());
        let name = "ORANGU_TEST_ENV_FLAG_OFF";
        for off in ["0", "false", "FALSE", "off", "no", "", "  0  ", " Off "] {
            unsafe { std::env::set_var(name, off) };
            assert!(!flag_on(name), "{off:?} should read as off");
        }
        unsafe { std::env::remove_var(name) };
        assert!(!flag_on(name), "an unset flag is off");
    }

    /// Anything that is not one of the off spellings turns the feature on,
    /// including values nobody intended — a flag that quietly ignored
    /// `FLAG=ture` would be the same silent-no-op failure pointing the other
    /// way.
    #[test]
    fn anything_else_is_on() {
        let _guard = ENV.lock().unwrap_or_else(|e| e.into_inner());
        let name = "ORANGU_TEST_ENV_FLAG_ON";
        for on in ["1", "true", "TRUE", "yes", "on", "ture", "2", "-1"] {
            unsafe { std::env::set_var(name, on) };
            assert!(flag_on(name), "{on:?} should read as on");
        }
        unsafe { std::env::remove_var(name) };
    }
}
