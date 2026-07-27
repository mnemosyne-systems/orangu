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

//! Renders a GGUF model's own `tokenizer.chat_template` (a Jinja2 template,
//! the same one llama.cpp renders with its bundled `minja` engine) via
//! `minijinja` — the closest pure-Rust equivalent. `raise_exception` and
//! `strftime_now` are registered as globals since several widely-used
//! templates (Llama 3, Qwen2.5) call them.

use anyhow::{Result, anyhow};
use minijinja::{Environment, context};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct ChatTemplate {
    source: String,
}

impl ChatTemplate {
    pub fn new(source: String) -> Self {
        Self { source }
    }

    /// `enable_thinking`, when `Some`, is passed into the template as a
    /// same-named variable — the kwarg convention several reasoning-
    /// capable models' own templates check (Qwen3's among them:
    /// `{%- if enable_thinking is defined and enable_thinking is false
    /// %}`) to skip whatever preamble tells the model to think before
    /// answering. `None` omits the variable entirely (leaving it
    /// genuinely undefined, not merely `null`, so an `is defined` check
    /// behaves as if the caller never mentioned it at all) rather than
    /// passing `None`/`null` through — a template checking `is defined`
    /// would otherwise see a *defined* (if null) variable and take the
    /// wrong branch. Harmless no-op for a template that doesn't check it.
    pub fn render(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        bos_token: &str,
        eos_token: &str,
        enable_thinking: Option<bool>,
    ) -> Result<String> {
        let mut env = Environment::new();
        env.add_function("raise_exception", |msg: String| {
            Err::<String, minijinja::Error>(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        });
        env.add_function("strftime_now", |fmt: String| strftime_now(&fmt));
        // Real chat templates (this project's own gemma-4-E2B-it test
        // model's included) are written for Python's Jinja2 and lean on
        // dict/list/str methods minijinja doesn't implement natively —
        // `message.get('reasoning')`, `.strip()`, `.split()`, and so on.
        // `minijinja_contrib::pycompat` fills that gap; without it, any
        // template using `.get()` (a common pattern for optional
        // tool-calling/reasoning fields) fails to render at all.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_template("chat", &self.source)
            .map_err(|err| anyhow!("invalid chat template: {err}"))?;
        let tmpl = env.get_template("chat").expect("just added");
        let rendered = match enable_thinking {
            Some(enable_thinking) => tmpl.render(context! {
                messages => messages,
                add_generation_prompt => add_generation_prompt,
                bos_token => bos_token,
                eos_token => eos_token,
                enable_thinking => enable_thinking,
            }),
            None => tmpl.render(context! {
                messages => messages,
                add_generation_prompt => add_generation_prompt,
                bos_token => bos_token,
                eos_token => eos_token,
            }),
        };
        rendered.map_err(|err| anyhow!("failed to render chat template: {err}"))
    }
}

/// English abbreviated (`%b`) and full (`%B`) month names, indexed by
/// month - 1. Hardcoded rather than locale-derived on purpose: a chat
/// template's date line is model-facing text that the model was trained
/// against in English, so rendering it in the host's locale would be a bug,
/// not a feature.
const MONTH_NAMES: [(&str, &str); 12] = [
    ("Jan", "January"),
    ("Feb", "February"),
    ("Mar", "March"),
    ("Apr", "April"),
    ("May", "May"),
    ("Jun", "June"),
    ("Jul", "July"),
    ("Aug", "August"),
    ("Sep", "September"),
    ("Oct", "October"),
    ("Nov", "November"),
    ("Dec", "December"),
];

/// A minimal `strftime`: only the handful of specifiers real templates
/// actually use (`%Y-%m-%d` for a "knowledge cutoff" style date line), via
/// `SystemTime` rather than pulling in a date/time crate for one function.
///
/// `%b`/`%B` are here because Llama 3.1/3.2's own chat template calls
/// `strftime_now("%d %b %Y")` — an unhandled specifier isn't inert, it
/// reaches the model verbatim: the rendered system block read `Today Date:
/// 27 %b 2026`, tokenizing the literal `%`/`b` into the prompt of every
/// chat request against those models.
fn strftime_now(fmt: &str) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days_since_epoch = now / 86400;
    // Civil-from-days (Howard Hinnant's algorithm) — no calendar crate
    // needed for a plain proleptic-Gregorian Y/M/D.
    let z = days_since_epoch as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    // `%B` before `%b` is irrelevant here (distinct keys), but `%%` is
    // handled first and restored last so an escaped percent can't be
    // re-interpreted as a specifier by a later replacement.
    const PERCENT: &str = "\u{0}orangu-percent\u{0}";
    let (abbrev, full) = MONTH_NAMES[(m as usize).clamp(1, 12) - 1];
    fmt.replace("%%", PERCENT)
        .replace("%Y", &y.to_string())
        .replace("%m", &format!("{m:02}"))
        .replace("%d", &format!("{d:02}"))
        .replace("%e", &format!("{d:2}"))
        .replace("%b", abbrev)
        .replace("%B", full)
        .replace(PERCENT, "%")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_a_simple_template() {
        let tmpl = ChatTemplate::new(
            "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}\
             {% if add_generation_prompt %}assistant:{% endif %}"
                .to_string(),
        );
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];
        let out = tmpl.render(&messages, true, "<s>", "</s>", None).unwrap();
        assert_eq!(out, "user: hi\nassistant:");
    }

    #[test]
    fn exposes_bos_and_eos_tokens() {
        let tmpl = ChatTemplate::new("{{ bos_token }}...{{ eos_token }}".to_string());
        let out = tmpl.render(&[], false, "<BOS>", "<EOS>", None).unwrap();
        assert_eq!(out, "<BOS>...<EOS>");
    }

    /// Regression test: real chat templates (this project's own
    /// `gemma-4-E2B-it` test model's included) are written for Python's
    /// Jinja2 and call `.get()` on messages — a dict method minijinja
    /// doesn't implement natively, unlike `pycompat`. Without `pycompat`
    /// wired in, this fails with "map has no method named get" instead of
    /// rendering; sending any message through the web UI against such a
    /// model 400'd until this was fixed.
    #[test]
    fn supports_python_style_dict_get_on_messages() {
        let tmpl = ChatTemplate::new(
            "{% for m in messages %}{{ m.get('role') }}={{ m.get('missing', 'default') }} \
             {% endfor %}"
                .to_string(),
        );
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];
        let out = tmpl.render(&messages, false, "", "", None).unwrap();
        assert_eq!(out, "user=default ");
    }

    #[test]
    fn raise_exception_surfaces_as_an_error() {
        let tmpl = ChatTemplate::new(
            "{% if messages[0].role != 'system' %}{{ raise_exception('need a system message') }}{% endif %}"
                .to_string(),
        );
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hi".to_string(),
        }];
        assert!(tmpl.render(&messages, false, "", "", None).is_err());
    }

    /// `enable_thinking: Some(false)` reaches the template as a real
    /// variable a `{%- if enable_thinking is defined and enable_thinking
    /// is false %}`-style check (Qwen3's own template convention) can see;
    /// `None` leaves it genuinely undefined rather than passing `null`
    /// through, so `is defined` correctly evaluates false too.
    #[test]
    fn enable_thinking_is_only_defined_when_given() {
        let tmpl = ChatTemplate::new(
            "{%- if enable_thinking is defined and enable_thinking is false -%}\
             no-think\
             {%- else -%}\
             think\
             {%- endif -%}"
                .to_string(),
        );
        assert_eq!(
            tmpl.render(&[], false, "", "", Some(false)).unwrap(),
            "no-think"
        );
        assert_eq!(
            tmpl.render(&[], false, "", "", Some(true)).unwrap(),
            "think"
        );
        assert_eq!(tmpl.render(&[], false, "", "", None).unwrap(), "think");
    }

    #[test]
    fn strftime_now_formats_year_month_day() {
        let out = strftime_now("%Y-%m-%d");
        let parts: Vec<&str> = out.split('-').collect();
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].len(), 4);
    }

    /// Llama 3.1/3.2's own chat template calls `strftime_now("%d %b %Y")`.
    /// An unimplemented specifier is not inert — it reaches the model as a
    /// literal `%b` inside the system block of every chat request — so the
    /// assertion is that *no* unexpanded specifier survives, not merely
    /// that the output is non-empty.
    #[test]
    fn strftime_now_expands_llama3_style_month_names() {
        let out = strftime_now("%d %b %Y");
        assert!(!out.contains('%'), "unexpanded specifier in {out:?}");
        let parts: Vec<&str> = out.split(' ').collect();
        assert_eq!(parts.len(), 3, "{out:?}");
        assert!(
            MONTH_NAMES.iter().any(|(abbrev, _)| *abbrev == parts[1]),
            "{:?} is not an abbreviated month name",
            parts[1]
        );

        let full = strftime_now("%B");
        assert!(
            MONTH_NAMES.iter().any(|(_, name)| *name == full),
            "{full:?} is not a full month name"
        );
    }

    /// An escaped `%%` must survive as a single literal `%` without its
    /// output being rescanned — otherwise `%%b` would expand to a month.
    #[test]
    fn strftime_now_leaves_an_escaped_percent_alone() {
        assert_eq!(strftime_now("100%%"), "100%");
        assert_eq!(strftime_now("%%b"), "%b");
    }
}
