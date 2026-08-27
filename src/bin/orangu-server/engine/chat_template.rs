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
use minijinja::Environment;
use serde::{Deserialize, Serialize};

/// One message as the OpenAI Chat Completions API shapes it.
///
/// Everything past `role`/`content` exists for tool calling and is carried
/// through to the chat template untouched — a template is the only thing that
/// knows how its model wants a tool call written, and every mainstream one
/// reads these exact field names (`message.get('tool_calls')`,
/// `follow.get('tool_call_id')`, `follow.get('name')`). Dropping them, as this
/// struct used to, does not merely lose formatting: a model that called a tool
/// on turn N sees no record of having done so on turn N+1, and calls it again.
#[derive(Serialize, Deserialize, Clone, Default)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
    /// Assistant messages: the calls this turn made. Left as raw JSON rather
    /// than a typed struct because templates reach into it in
    /// model-specific ways (`function['arguments']` as a mapping *or* a
    /// pre-serialized string, `tool_call['id']`, vendor extensions), and
    /// re-shaping it here would only lose whatever a given template needs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<serde_json::Value>,
    /// Tool messages: which call this is the result of.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Tool messages: the function's name. Some templates use it directly;
    /// others resolve it from `tool_call_id` against the preceding assistant
    /// message, so both are carried.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    /// A plain text message — the shape every non-tool caller wants.
    pub fn text(role: &str, content: &str) -> Self {
        Self {
            role: role.to_string(),
            content: content.to_string(),
            ..Default::default()
        }
    }
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
    ///
    /// `tools`, when `Some`, is the request's OpenAI-shaped tool array, passed
    /// into the template as the same-named variable every tool-capable
    /// template gates on (`{%- if tools -%}`). `None` omits the variable
    /// entirely rather than passing an empty list, so a template that only
    /// checks truthiness and one that checks `is defined` both behave as if
    /// the caller never mentioned tools — the same care
    /// [`Self::render`]'s `enable_thinking` takes, and for the same reason.
    pub fn render_with_tools(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        bos_token: &str,
        eos_token: &str,
        enable_thinking: Option<bool>,
        tools: Option<&serde_json::Value>,
    ) -> Result<String> {
        self.render_inner(
            messages,
            add_generation_prompt,
            bos_token,
            eos_token,
            enable_thinking,
            tools,
        )
    }

    pub fn render(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        bos_token: &str,
        eos_token: &str,
        enable_thinking: Option<bool>,
    ) -> Result<String> {
        self.render_inner(
            messages,
            add_generation_prompt,
            bos_token,
            eos_token,
            enable_thinking,
            None,
        )
    }

    fn render_inner(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        bos_token: &str,
        eos_token: &str,
        enable_thinking: Option<bool>,
        tools: Option<&serde_json::Value>,
    ) -> Result<String> {
        let mut env = Environment::new();
        env.add_function("raise_exception", |msg: String| {
            Err::<String, minijinja::Error>(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        });
        env.add_function("strftime_now", |fmt: String| strftime_now(&fmt));
        env.add_filter("tojson", tojson);
        // Real chat templates (this project's own gemma-4-E2B-it test
        // model's included) are written for Python's Jinja2 and lean on
        // dict/list/str methods minijinja doesn't implement natively —
        // `message.get('reasoning')`, `.strip()`, `.split()`, and so on.
        // `minijinja_contrib::pycompat` fills that gap; without it, any
        // template using `.get()` (a common pattern for optional
        // tool-calling/reasoning fields) fails to render at all.
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        let source = parenthesize_call_kwarg_conditionals(&self.source);
        env.add_template("chat", &source)
            .map_err(|err| anyhow!("invalid chat template: {err}"))?;
        let tmpl = env.get_template("chat").expect("just added");
        // Built additively rather than as one `context!` per combination:
        // two optional variables that must each be *absent* (not null) when
        // unset would otherwise need four literals kept in step.
        let mut ctx = std::collections::BTreeMap::<&str, minijinja::Value>::new();
        ctx.insert("messages", minijinja::Value::from_serialize(messages));
        ctx.insert(
            "add_generation_prompt",
            minijinja::Value::from(add_generation_prompt),
        );
        ctx.insert("bos_token", minijinja::Value::from(bos_token));
        ctx.insert("eos_token", minijinja::Value::from(eos_token));
        if let Some(enable_thinking) = enable_thinking {
            ctx.insert("enable_thinking", minijinja::Value::from(enable_thinking));
        }
        if let Some(tools) = tools {
            ctx.insert("tools", minijinja::Value::from_serialize(tools));
        }
        tmpl.render(minijinja::Value::from_serialize(&ctx))
            .map_err(|err| anyhow!("failed to render chat template: {err}"))
    }
}

/// `tojson`, as `transformers` defines it for chat templates: a thin
/// wrapper over Python's `json.dumps` that takes `ensure_ascii`, `indent`,
/// `separators` and `sort_keys` as keyword arguments and — unlike Jinja2's
/// own HTML-safe `tojson` — escapes nothing beyond what JSON requires.
///
/// This build's `minijinja` has no `tojson` at all (its JSON filters are
/// behind a feature this project does not enable), so a template that calls
/// one fails to render: `Inkling-Small`'s does, in the recursive
/// `canonical_json` macro that writes its tool declarations, and that
/// failure is a 500 on every tool-carrying chat request rather than
/// anything visible at load. Registering it here is purely additive —
/// there is no existing behavior to change.
///
/// The defaults are `transformers`', not Python's: `ensure_ascii` defaults
/// to **false** and the separators to the compact `(',', ':')`, which is
/// what every chat template written against `transformers` assumes.
/// Object keys keep the order the template produced them in (Python's
/// behavior); `sort_keys` is accepted and rejected rather than ignored,
/// since silently leaving keys unsorted would be a difference nothing
/// downstream could see.
fn tojson(
    value: minijinja::Value,
    kwargs: minijinja::value::Kwargs,
) -> Result<minijinja::Value, minijinja::Error> {
    use minijinja::{Error, ErrorKind};
    use serde::Serialize;

    let ensure_ascii: bool = kwargs.get::<Option<bool>>("ensure_ascii")?.unwrap_or(false);
    let indent: Option<usize> = kwargs.get::<Option<usize>>("indent")?;
    // A Python 2-tuple, which reaches minijinja as a two-element sequence.
    let separators = match kwargs.get::<Option<minijinja::Value>>("separators")? {
        Some(pair) if !pair.is_none() => {
            let parts: Vec<String> = pair
                .try_iter()
                .map_err(|err| Error::new(ErrorKind::InvalidOperation, format!("tojson: {err}")))?
                .map(|v| v.to_string())
                .collect();
            let [item, key] = <[String; 2]>::try_from(parts).map_err(|parts| {
                Error::new(
                    ErrorKind::InvalidOperation,
                    format!("tojson: separators wants 2 values, got {}", parts.len()),
                )
            })?;
            Some((item, key))
        }
        _ => None,
    };
    if kwargs.get::<Option<bool>>("sort_keys")?.unwrap_or(false) {
        return Err(Error::new(
            ErrorKind::InvalidOperation,
            "tojson(sort_keys=true) is not implemented",
        ));
    }
    kwargs.assert_all_used()?;

    let mut buf = Vec::new();
    match indent {
        // Python's `indent` overrides the separators with newline-and-pad,
        // which is exactly what serde_json's pretty formatter writes.
        Some(indent) => {
            let pad = " ".repeat(indent);
            let formatter = serde_json::ser::PrettyFormatter::with_indent(pad.as_bytes());
            let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
            value.serialize(&mut ser)
        }
        None => {
            let (item, key) = separators.unwrap_or_else(|| (",".to_string(), ":".to_string()));
            let mut ser =
                serde_json::Serializer::with_formatter(&mut buf, PythonSeparators { item, key });
            value.serialize(&mut ser)
        }
    }
    .map_err(|err| Error::new(ErrorKind::InvalidOperation, format!("tojson: {err}")))?;

    let json = String::from_utf8(buf)
        .map_err(|err| Error::new(ErrorKind::InvalidOperation, format!("tojson: {err}")))?;
    Ok(minijinja::Value::from(if ensure_ascii {
        escape_non_ascii(&json)
    } else {
        json
    }))
}

/// A `serde_json` formatter that writes Python's `separators=(item, key)`
/// pair instead of serde's fixed `,`/`:`.
///
/// Only the three hooks that emit a separator are overridden; everything
/// else — number and string formatting, escaping — stays serde_json's, which
/// already matches `json.dumps` for every value a chat template can hold.
struct PythonSeparators {
    item: String,
    key: String,
}

impl serde_json::ser::Formatter for PythonSeparators {
    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(self.item.as_bytes())
        }
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if first {
            Ok(())
        } else {
            writer.write_all(self.item.as_bytes())
        }
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(self.key.as_bytes())
    }
}

/// `json.dumps(..., ensure_ascii=True)`'s escaping: every non-ASCII
/// character as `\uXXXX`, with astral ones as a surrogate pair.
///
/// Safe to run over the finished document rather than per string, because
/// every character JSON gives structural meaning is ASCII — a non-ASCII
/// character can only ever be inside a string literal.
fn escape_non_ascii(json: &str) -> String {
    if json.is_ascii() {
        return json.to_string();
    }
    let mut out = String::with_capacity(json.len());
    for ch in json.chars() {
        if ch.is_ascii() {
            out.push(ch);
        } else {
            let mut units = [0u16; 2];
            for unit in ch.encode_utf16(&mut units) {
                out.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    out
}

/// Rewrites `f(k=a if b else c)` to `f(k=(a if b else c))` throughout a
/// template's tags, leaving everything else byte-for-byte alone.
///
/// A **parser** gap, not a template bug: minijinja parses a call's keyword
/// argument with an expression grammar that stops short of the inline
/// conditional, so it reads `k=a`, meets `if`, and reports "unexpected
/// identifier, expected `,`" — at *compile* time, which fails the whole
/// template for every request rather than only the branch containing it.
/// Python's Jinja2 and llama.cpp's own `minja` both accept the bare form,
/// so a template written against either can carry it. `Muse-Glimmer-30B`'s
/// does (`namespace(name=tcid if tcid else '')`, in its tool-message
/// branch), and without this its chat endpoints are unusable — the model
/// loads and generates fine, but no chat request can be rendered.
///
/// The parenthesized form is accepted (inside brackets minijinja runs its
/// full expression parser) and means exactly the same thing, so this is a
/// pure widening: every source that parsed before still parses, unchanged.
/// It only ever fires on a keyword argument whose value contains a
/// *top-level* `if`, which is precisely the shape that could not have
/// parsed before. Borrowed and untouched when nothing matches, which is
/// every template this project had before Muse-Glimmer.
///
/// Deliberately a source rewrite rather than a Jinja dialect of our own:
/// the alternative is forking the parser, and this is one production with
/// one shape.
fn parenthesize_call_kwarg_conditionals(source: &str) -> std::borrow::Cow<'_, str> {
    let mut current = std::borrow::Cow::Borrowed(source);
    // One pass wraps a set of non-overlapping spans. A keyword argument
    // nested inside another one's value overlaps it, so only the outer is
    // taken this time round and the inner on the next — hence the loop.
    // It terminates because each pass either wraps at least one span
    // (strictly reducing how many are left unwrapped) or finds none.
    loop {
        let mut spans = call_kwarg_conditional_spans(current.as_bytes());
        spans.sort_unstable();
        let mut copied = 0;
        let mut out = String::with_capacity(current.len() + 2 * spans.len());
        let mut wrapped = 0;
        for (start, end) in spans {
            if start < copied {
                continue;
            }
            out.push_str(&current[copied..start]);
            out.push('(');
            out.push_str(&current[start..end]);
            out.push(')');
            copied = end;
            wrapped += 1;
        }
        if wrapped == 0 {
            return current;
        }
        out.push_str(&current[copied..]);
        current = std::borrow::Cow::Owned(out);
    }
}

/// One keyword argument being scanned by [`call_kwarg_conditional_spans`]:
/// where its value started, how deeply nested the call was at that point,
/// and whether a bare `if` has been seen at that same nesting depth (an
/// `if` inside a nested call or literal belongs to that inner expression,
/// which parses fine on its own).
struct PendingKwarg {
    depth: usize,
    start: usize,
    has_if: bool,
}

/// Every keyword-argument value in `bytes` that is a bare inline
/// conditional, as a byte span.
///
/// Innermost-first, and a nested one is *contained* in its enclosing
/// argument's span rather than disjoint from it — which is why the caller
/// sorts, wraps a non-overlapping subset, and comes round again.
///
/// A single left-to-right pass over the source. Only the inside of a Jinja
/// `{% %}`/`{{ }}` tag is examined; template *text* and `{# #}` comments
/// are passed over, which is what keeps a document that merely mentions
/// `f(a=b if c else d)` in prose or in a comment from being rewritten.
/// String literals are skipped wholesale, so a `,`, a bracket or the word
/// `if` inside one cannot move the scan.
fn call_kwarg_conditional_spans(bytes: &[u8]) -> Vec<(usize, usize)> {
    let is_ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_';
    let mut spans = Vec::new();
    let mut i = 0;
    while i + 1 < bytes.len() {
        if bytes[i] != b'{' || !matches!(bytes[i + 1], b'%' | b'{' | b'#') {
            i += 1;
            continue;
        }
        // A comment holds no expressions at all — not even a string
        // literal that could hide a `#}` — so it is skipped whole.
        if bytes[i + 1] == b'#' {
            i = bytes[i + 2..]
                .windows(2)
                .position(|w| w == b"#}")
                .map_or(bytes.len(), |p| i + 2 + p + 2);
            continue;
        }
        // Inside a tag. `depth` counts every kind of bracket together:
        // what matters to an argument's extent is only whether it is
        // nested, not in what.
        let kind = bytes[i + 1];
        let mut depth = 0usize;
        let mut pending: Vec<PendingKwarg> = Vec::new();
        let mut j = i + 2;
        while j < bytes.len() {
            let c = bytes[j];
            match c {
                b'\'' | b'"' => {
                    j += 1;
                    while j < bytes.len() && bytes[j] != c {
                        // A backslash escapes the next byte, quote included.
                        j += if bytes[j] == b'\\' { 2 } else { 1 };
                    }
                    j += 1;
                    continue;
                }
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => {
                    // `%}` / `}}` at the outermost level closes the tag;
                    // `}}` inside a dict literal does not.
                    let closes_tag = match kind {
                        b'%' => false,
                        _ => c == b'}' && depth == 0 && bytes.get(j + 1) == Some(&b'}'),
                    };
                    if closes_tag {
                        j += 2;
                        break;
                    }
                    depth = depth.saturating_sub(1);
                    while pending.last().is_some_and(|k| k.depth > depth) {
                        finish_kwarg(&mut pending, bytes, j, &mut spans);
                    }
                }
                b'%' if kind == b'%' && bytes.get(j + 1) == Some(&b'}') => {
                    j += 2;
                    break;
                }
                b',' => {
                    while pending.last().is_some_and(|k| k.depth == depth) {
                        finish_kwarg(&mut pending, bytes, j, &mut spans);
                    }
                }
                b'i' if bytes.get(j + 1) == Some(&b'f')
                    && !bytes[..j].last().is_some_and(|&p| is_ident(p))
                    && !bytes.get(j + 2).copied().is_some_and(is_ident) =>
                {
                    if let Some(k) = pending.last_mut()
                        && k.depth == depth
                    {
                        k.has_if = true;
                    }
                }
                // A keyword argument's `=`: inside a call (`depth > 0`),
                // preceded by an identifier, and not half of `==`/`!=`/
                // `<=`/`>=`. Assignment in `{% set x = ... %}` is at depth
                // 0 and is left alone — minijinja parses *that* position
                // with the full grammar already.
                b'=' if depth > 0
                    && bytes.get(j + 1) != Some(&b'=')
                    && !matches!(bytes[j - 1], b'=' | b'!' | b'<' | b'>')
                    && bytes[..j]
                        .iter()
                        .rposition(|&p| p != b' ')
                        .is_some_and(|p| is_ident(bytes[p])) =>
                {
                    let mut start = j + 1;
                    while bytes.get(start) == Some(&b' ') {
                        start += 1;
                    }
                    pending.push(PendingKwarg {
                        depth,
                        start,
                        has_if: false,
                    });
                }
                _ => {}
            }
            j += 1;
        }
        i = j.max(i + 2);
    }
    spans
}

/// Closes the innermost pending keyword argument at `end`, recording its
/// value's span when that value turned out to be a bare inline
/// conditional. Trailing spaces are left outside the parentheses so the
/// rewrite stays as close to the original text as it can.
fn finish_kwarg(
    pending: &mut Vec<PendingKwarg>,
    bytes: &[u8],
    end: usize,
    spans: &mut Vec<(usize, usize)>,
) {
    let kwarg = pending.pop().expect("caller checked `last`");
    let mut end = end;
    while end > kwarg.start && bytes[end - 1] == b' ' {
        end -= 1;
    }
    if kwarg.has_if && end > kwarg.start {
        spans.push((kwarg.start, end));
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
        let messages = vec![ChatMessage::text("user", "hi")];
        let out = tmpl.render(&messages, true, "<s>", "</s>", None).unwrap();
        assert_eq!(out, "user: hi\nassistant:");
    }

    /// `{% break %}` — GLM-5.3-Flash's own construct, reduced. minijinja
    /// gates loop controls behind a Cargo feature that is *not* in its
    /// default set, so without `features = ["loop_controls"]` this fails at
    /// **compile** time: the template does not parse, and every chat
    /// request against such a model answers `invalid chat template: syntax
    /// error: unknown statement break` instead of generating. Nothing else
    /// catches it — `/v1/completions` applies no template at all, so the
    /// same model answers perfectly there while the web console and every
    /// OpenAI-shaped client see only the error.
    #[test]
    fn renders_a_template_that_breaks_out_of_a_loop() {
        let tmpl = ChatTemplate::new(
            "{%- set ns = namespace(n=0) -%}\
             {%- for m in messages -%}\
             {%- set ns.n = ns.n + 1 -%}\
             {%- if ns.n > 1 -%}{%- break -%}{%- endif -%}\
             {{- m.content -}}\
             {%- endfor -%}"
                .to_string(),
        );
        let messages = vec![
            ChatMessage::text("user", "first"),
            ChatMessage::text("assistant", "second"),
        ];
        let out = tmpl.render(&messages, false, "<s>", "</s>", None).unwrap();
        assert_eq!(out, "first");
    }

    /// `{% continue %}` rides on the same feature, and a template that used
    /// it would fail the same way.
    #[test]
    fn renders_a_template_that_continues_a_loop() {
        let tmpl = ChatTemplate::new(
            "{%- for m in messages -%}\
             {%- if m.role == 'assistant' -%}{%- continue -%}{%- endif -%}\
             {{- m.content -}}\
             {%- endfor -%}"
                .to_string(),
        );
        let messages = vec![
            ChatMessage::text("user", "a"),
            ChatMessage::text("assistant", "b"),
            ChatMessage::text("user", "c"),
        ];
        let out = tmpl.render(&messages, false, "<s>", "</s>", None).unwrap();
        assert_eq!(out, "ac");
    }

    /// The whole point of [`parenthesize_call_kwarg_conditionals`]: a
    /// keyword argument whose value is a bare inline conditional. This is
    /// `Muse-Glimmer-30B`'s own construct, reduced — without the rewrite
    /// minijinja rejects it at *compile* time, so the template fails for
    /// every request, not only for the tool-message branch it sits in.
    #[test]
    fn renders_a_conditional_keyword_argument() {
        let tmpl = ChatTemplate::new(
            "{%- set ns = namespace(name=messages[0].content if messages else 'none') -%}\
             {{- ns.name -}}"
                .to_string(),
        );
        let messages = vec![ChatMessage::text("user", "hi")];
        assert_eq!(tmpl.render(&messages, false, "", "", None).unwrap(), "hi");
        assert_eq!(tmpl.render(&[], false, "", "", None).unwrap(), "none");
    }

    /// The rewrite is a widening, so a template that never needed it must
    /// come through byte-for-byte — including the shapes that look like
    /// its trigger and are not: an assignment outside a call, a comparison
    /// (`==`, `!=`), an `if` belonging to a nested expression rather than
    /// to the argument, and template *text* that merely reads like code.
    #[test]
    fn leaves_everything_that_already_parsed_untouched() {
        for source in [
            "{% set x = a if b else c %}",
            "{% if a == b %}{{ f(k=1) }}{% endif %}",
            "{% if a != b %}{{ f(k=[1, 2], j='x') }}{% endif %}",
            "{{ f(k=g(a if b else c)) }}",
            "call f(k=a if b else c) in prose",
            "{# f(k=a if b else c) #}",
            "{{ f(k='a if b else c') }}",
        ] {
            assert!(
                matches!(
                    parenthesize_call_kwarg_conditionals(source),
                    std::borrow::Cow::Borrowed(_)
                ),
                "rewrote {source:?}"
            );
        }
    }

    /// Two arguments, only one of them conditional, and a conditional
    /// nested one call deep — the cases the single-pass span collection
    /// cannot get right on its own.
    #[test]
    fn wraps_each_conditional_argument_and_nothing_else() {
        assert_eq!(
            parenthesize_call_kwarg_conditionals("{% set n = f(a=1, b=x if y else z) %}"),
            "{% set n = f(a=1, b=(x if y else z)) %}"
        );
        assert_eq!(
            parenthesize_call_kwarg_conditionals(
                "{% set n = f(a=g(b=x if y else z) if q else w) %}"
            ),
            "{% set n = f(a=(g(b=(x if y else z)) if q else w)) %}"
        );
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
        let messages = vec![ChatMessage::text("user", "hi")];
        let out = tmpl.render(&messages, false, "", "", None).unwrap();
        assert_eq!(out, "user=default ");
    }

    /// The whole point of T1: a tool-capable template gates its declaration
    /// block on `{%- if tools -%}`, so a server that never passes `tools`
    /// renders a prompt in which no tool exists — and the model, correctly,
    /// never calls one.
    #[test]
    fn tools_reach_the_template_and_gate_its_declaration_block() {
        let tmpl = ChatTemplate::new(
            "{% if tools %}{% for t in tools %}TOOL:{{ t.function.name }};{% endfor %}\
             {% else %}NONE{% endif %}"
                .to_string(),
        );
        let tools = serde_json::json!([
            {"type": "function", "function": {"name": "show_file"}},
            {"type": "function", "function": {"name": "run_shell_command"}},
        ]);
        let with = tmpl
            .render_with_tools(&[], false, "", "", None, Some(&tools))
            .unwrap();
        assert_eq!(with, "TOOL:show_file;TOOL:run_shell_command;");

        let without = tmpl
            .render_with_tools(&[], false, "", "", None, None)
            .unwrap();
        assert_eq!(without, "NONE");
    }

    /// A template must be able to tell "no tools were offered" from "tools
    /// were offered", `is defined` included — the same distinction
    /// `enable_thinking` needs and for the same reason.
    #[test]
    fn absent_tools_are_undefined_rather_than_null() {
        let tmpl = ChatTemplate::new(
            "{% if tools is defined %}DEFINED{% else %}UNDEFINED{% endif %}".to_string(),
        );
        assert_eq!(
            tmpl.render_with_tools(&[], false, "", "", None, None)
                .unwrap(),
            "UNDEFINED"
        );
    }

    /// Turn N+1 has to show the model that turn N called a tool and what came
    /// back. With `tool_calls`/`tool_call_id` dropped from `ChatMessage`, the
    /// transcript said only that the assistant produced empty content — so
    /// the model called the same tool again, forever.
    #[test]
    fn a_tool_call_and_its_result_survive_into_the_next_turn() {
        let tmpl = ChatTemplate::new(
            "{% for m in messages %}\
             {% if m.get('tool_calls') %}CALL:{{ m.tool_calls[0].function.name }};{% endif %}\
             {% if m.role == 'tool' %}RESULT[{{ m.get('name') }}/{{ m.get('tool_call_id') }}]\
             ={{ m.content }};{% endif %}\
             {% endfor %}"
                .to_string(),
        );
        let messages = vec![
            ChatMessage::text("user", "weather?"),
            ChatMessage {
                role: "assistant".into(),
                tool_calls: Some(serde_json::json!([
                    {"id": "call-1", "type": "function",
                     "function": {"name": "get_weather", "arguments": "{}"}}
                ])),
                ..Default::default()
            },
            ChatMessage {
                role: "tool".into(),
                content: "17C".into(),
                tool_call_id: Some("call-1".into()),
                name: Some("get_weather".into()),
                ..Default::default()
            },
        ];
        let out = tmpl.render(&messages, false, "", "", None).unwrap();
        assert_eq!(out, "CALL:get_weather;RESULT[get_weather/call-1]=17C;");
    }

    #[test]
    fn raise_exception_surfaces_as_an_error() {
        let tmpl = ChatTemplate::new(
            "{% if messages[0].role != 'system' %}{{ raise_exception('need a system message') }}{% endif %}"
                .to_string(),
        );
        let messages = vec![ChatMessage::text("user", "hi")];
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

    /// Renders `{{ value | tojson(<args>) }}` and returns what came out.
    fn render_tojson(args: &str, value: serde_json::Value) -> Result<String> {
        let tmpl = ChatTemplate::new(format!("{{{{ value | tojson({args}) }}}}"));
        // The value rides in as a message field, since `render` builds its
        // own context — `content` is passed through untouched.
        let source = tmpl.source.replace("value", "messages[0].tool_calls");
        let mut env = Environment::new();
        env.add_filter("tojson", tojson);
        env.add_template("t", &source)
            .map_err(|err| anyhow!("{err}"))?;
        env.get_template("t")
            .unwrap()
            .render(
                minijinja::context! { messages => vec![serde_json::json!({"tool_calls": value})] },
            )
            .map_err(|err| anyhow!("{err}"))
    }

    /// The default has to be `transformers`' default, not Python's or
    /// Jinja2's: compact separators, no HTML escaping, and non-ASCII left
    /// as itself. A template that passes no keyword arguments at all still
    /// expects that.
    #[test]
    fn tojson_defaults_to_compact_unescaped_json() {
        let out = render_tojson("", serde_json::json!({"a": 1, "b": ["x", "<y>"]})).unwrap();
        assert_eq!(out, r#"{"a":1,"b":["x","<y>"]}"#);
        assert_eq!(
            render_tojson("", serde_json::json!("héllo")).unwrap(),
            r#""héllo""#
        );
    }

    /// The keyword arguments real templates pass. `Inkling-Small`'s writes
    /// every one of its tool declarations through
    /// `tojson(ensure_ascii=false, separators=(',', ':'))`.
    #[test]
    fn tojson_honors_pythons_keyword_arguments() {
        let value = serde_json::json!({"a": 1, "b": 2});
        assert_eq!(
            render_tojson("ensure_ascii=false, separators=(',', ':')", value.clone()).unwrap(),
            r#"{"a":1,"b":2}"#
        );
        assert_eq!(
            render_tojson("separators=(', ', ': ')", value.clone()).unwrap(),
            r#"{"a": 1, "b": 2}"#
        );
        assert_eq!(
            render_tojson("indent=2", value).unwrap(),
            "{\n  \"a\": 1,\n  \"b\": 2\n}"
        );
        // `ensure_ascii=true` is Python's own default and the one no chat
        // template asks for; it escapes past ASCII, astral planes as a
        // surrogate pair.
        assert_eq!(
            render_tojson("ensure_ascii=true", serde_json::json!("héllo ☃ 😀")).unwrap(),
            r#""h\u00e9llo \u2603 \ud83d\ude00""#
        );
    }

    /// A keyword argument that would change the output and is not
    /// implemented must fail loudly. Accepting and ignoring `sort_keys`
    /// would produce a differently-ordered document that nothing
    /// downstream could notice.
    #[test]
    fn tojson_rejects_the_keyword_arguments_it_does_not_implement() {
        let err = render_tojson("sort_keys=true", serde_json::json!({"b": 1, "a": 2}))
            .expect_err("sort_keys is not implemented");
        assert!(err.to_string().contains("sort_keys"), "{err}");
        let err = render_tojson("wat=1", serde_json::json!(1)).expect_err("unknown kwarg");
        assert!(err.to_string().contains("wat"), "{err}");
    }
}

#[cfg(test)]
mod real_model_tests {
    use super::*;

    /// A model's *own* `tokenizer.chat_template`, rendered.
    ///
    /// The compatibility shims here (`pycompat`, the keyword-argument
    /// rewrite, [`tojson`]) each exist because one real template needed
    /// one, and a template that fails to render fails at *request* time —
    /// the model loads, generates, and every chat call returns a 500. A
    /// unit test over a reduced construct proves the shim works; only the
    /// real file proves the shim was enough.
    fn render_real_template(env_var: &str, tools: Option<&serde_json::Value>) -> String {
        let path = std::env::var(env_var).unwrap_or_else(|_| panic!("set {env_var}"));
        let gguf = orangu::gguf::GgufFile::open(std::path::Path::new(&path)).expect("open gguf");
        let source = gguf
            .metadata
            .iter()
            .find_map(|(k, v)| match (k.as_str(), v) {
                ("tokenizer.chat_template", orangu::gguf::GgufValue::String(s)) => Some(s.clone()),
                _ => None,
            })
            .expect("the file has no tokenizer.chat_template");
        let messages = vec![
            ChatMessage::text("system", "You are terse."),
            ChatMessage::text("user", "Name three primes over 20."),
            ChatMessage::text("assistant", "23, 29, 31."),
            ChatMessage::text("user", "And one more?"),
        ];
        ChatTemplate::new(source)
            .render_with_tools(&messages, true, "", "", None, tools)
            .expect("render")
    }

    /// `unsloth/Inkling-Small-GGUF`'s template, which leans harder on
    /// Jinja than any other model here: recursive macros, `{% set %}`
    /// blocks, string slicing, and `tojson` called with Python's keyword
    /// arguments.
    ///
    /// Run with `ORANGU_TEST_INKLING_MODEL=/path/to/Inkling-Small-...-00001-of-00005.gguf
    /// cargo test --bin orangu-server chat_template::real_model_tests --
    /// --ignored`.
    #[test]
    #[ignore]
    fn the_inkling_template_renders_a_conversation() {
        let out = render_real_template("ORANGU_TEST_INKLING_MODEL", None);
        assert!(out.contains("<|message_user|><|content_text|>And one more?<|end_message|>"));
        assert!(out.contains("<|message_model|><|content_text|>23, 29, 31."));
        // The generation prompt: an open model turn with no content marker,
        // so the model picks the kind of body it writes next.
        assert!(
            out.ends_with("<|message_model|>"),
            "tail: {:?}",
            &out[out.len().saturating_sub(64)..]
        );
    }

    /// The same template with tools, which is the branch that reaches the
    /// recursive `canonical_json` macro and its `tojson(ensure_ascii=false,
    /// separators=(',', ':'))` calls.
    #[test]
    #[ignore]
    fn the_inkling_template_renders_tool_declarations() {
        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Current weather for a place",
                "parameters": {
                    "type": "object",
                    "properties": {"place": {"type": "string"}},
                    "required": ["place"],
                },
            },
        }]);
        let out = render_real_template("ORANGU_TEST_INKLING_MODEL", Some(&tools));
        assert!(out.contains("<|message_system|>tool_declare<|content_xml|>["));
        // Compact separators and no HTML escaping — Python's `json.dumps`
        // as transformers configures it, which is what `tojson` reproduces.
        assert!(
            out.contains(r#""name":"get_weather""#),
            "tool declaration was: {out}"
        );
    }
}
