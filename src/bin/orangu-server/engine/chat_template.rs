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
use std::borrow::Cow;

/// One message as the OpenAI Chat Completions API shapes it.
///
/// Everything past `role`/`content` exists for tool calling and is carried
/// through to the chat template untouched — a template is the only thing that
/// knows how its model wants a tool call written, and every mainstream one
/// reads these exact field names (`message.get('tool_calls')`,
/// `follow.get('tool_call_id')`, `follow.get('name')`). Dropping them, as this
/// struct used to, does not merely lose formatting: a model that called a tool
/// on turn N sees no record of having done so on turn N+1, and calls it again.
///
/// `content` may arrive as OpenAI's *array of parts* rather than a string —
/// `[{"type": "text", "text": ...}, {"type": "image_url", "image_url":
/// {"url": "data:image/png;base64,..."}}]`, which is how a client attaches
/// a picture. The text parts are joined into [`content`](Self::content)
/// (so every chat template keeps seeing a string) and the pictures land in
/// [`images`](Self::images), which is never serialized and so never reaches
/// a template, with where in the text each one stood. A chat model with a
/// vision tower reads them in place (`http::openai::render_chat`); a
/// `qwen_image` model starts its picture from the last one. See
/// [`ChatMessageWire`], which is what is actually deserialized.
#[derive(Serialize, Deserialize, Clone, Default)]
#[serde(from = "ChatMessageWire")]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
    /// The `image_url` parts of an array-shaped `content`, as given (data
    /// URLs or plain URLs). See the struct's own doc.
    #[serde(skip)]
    pub images: Vec<String>,
    /// Where in [`content`](Self::content) each of [`images`](Self::images)
    /// stood, as a byte offset — what puts a picture back between the
    /// texts it was sent between.
    #[serde(skip)]
    pub image_at: Vec<usize>,
    /// Assistant messages: the calls this turn made. Left as raw JSON rather
    /// than a typed struct because templates reach into it in
    /// model-specific ways (`function['arguments']`, `tool_call['id']`,
    /// vendor extensions), and re-shaping it here would only lose whatever a
    /// given template needs. A template sees `arguments` as a mapping and
    /// ids in the form it accepts: see [`ChatTemplate::for_template`].
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

    /// Adds a picture at the end of the text so far.
    pub fn push_image(&mut self, url: String) {
        self.image_at.push(self.content.len());
        self.images.push(url);
    }
}

/// [`ChatMessage`] as it is read off the wire: `content` still a JSON value,
/// because it is a string *or* an array of parts, and serde's derive can
/// only fill one field from one key. The conversion below is where the
/// parts are split into text and pictures.
#[derive(Deserialize)]
struct ChatMessageWire {
    role: String,
    #[serde(default)]
    content: serde_json::Value,
    #[serde(default)]
    tool_calls: Option<serde_json::Value>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
}

impl From<ChatMessageWire> for ChatMessage {
    /// A string is the text; `null` is empty; an array's `text` parts are
    /// joined with newlines and its `image_url` parts collected. Any other
    /// part type is ignored, as OpenAI's own servers ignore what they do not
    /// know, and any other JSON value is read as its text — a number sent
    /// as content is a caller's mistake, not a reason to refuse the turn.
    fn from(wire: ChatMessageWire) -> Self {
        let mut images = Vec::new();
        let mut image_at = Vec::new();
        let content = match wire.content {
            serde_json::Value::Null => String::new(),
            serde_json::Value::String(s) => s,
            serde_json::Value::Array(parts) => {
                let mut text = String::new();
                for part in parts {
                    match part.get("type").and_then(|t| t.as_str()) {
                        Some("text") => {
                            if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                                if !text.is_empty() {
                                    text.push('\n');
                                }
                                text.push_str(t);
                            }
                        }
                        Some("image_url") => {
                            let url = part.get("image_url").and_then(|u| {
                                u.get("url").and_then(|s| s.as_str()).or_else(|| u.as_str())
                            });
                            if let Some(url) = url {
                                image_at.push(text.len());
                                images.push(url.to_string());
                            }
                        }
                        _ => {}
                    }
                }
                text
            }
            other => other.to_string(),
        };
        Self {
            role: wire.role,
            content,
            images,
            image_at,
            tool_calls: wire.tool_calls,
            tool_call_id: wire.tool_call_id,
            name: wire.name,
        }
    }
}

/// A model's chat template, compiled once.
///
/// The environment — its helper functions, filters and the parsed template —
/// is built when the template is, and every render reuses it. A template that
/// fails to compile keeps its error, reported by each render as it would have
/// been had it compiled there.
pub struct ChatTemplate {
    env: std::result::Result<Environment<'static>, String>,
    /// Whether the template refuses a tool call id that is not nine
    /// alphanumeric characters (Mistral's do).
    nine_char_ids: bool,
    /// Whether the template takes a tool call's arguments only as JSON text
    /// (it joins them to a string with `+`).
    text_arguments: bool,
    /// The call syntax this template teaches, when no parser knows it: read
    /// once, on the first request that declares tools.
    learned_calls: std::sync::OnceLock<Option<std::sync::Arc<super::tool_learn::LearnedCalls>>>,
}

/// What a request wants from a reasoning model: whether it should think at
/// all, and how hard.
///
/// One parameter rather than two because it is one decision, taken in one
/// place (`config::Role` plus `[orangu-server].reasoning_effort`) and read
/// by the three endpoints that render a prompt. Both fields are `Option`
/// and both mean the same thing when `None`: leave the variable *undefined*
/// so the template's own `default(...)` applies, rather than passing a null
/// a template would have to treat as a value.
#[derive(Clone, Copy, Default)]
pub struct Reasoning<'a> {
    /// `enable_thinking` — `Some(false)` for a reasoning-suppressing role,
    /// `None` to leave the template's own default alone.
    pub enabled: Option<bool>,
    /// `reasoning_effort`, verbatim: the levels are the template's
    /// vocabulary, not this server's.
    pub effort: Option<&'a str>,
    /// Whether [`Self::effort`] is this server's default rather than a
    /// value the deployment asked for.
    ///
    /// It decides what happens when a template *rejects* the level. A
    /// default this server picked has no business turning someone's request
    /// into a `400` because their template spells its levels differently,
    /// so it is dropped and the prompt rendered again without it. A value
    /// from `[orangu-server].reasoning_effort` is a stated intention, and a
    /// typo in it should say so rather than be quietly ignored.
    pub effort_is_default: bool,
}

/// The largest template this compiles, some five times the largest in use. The
/// compiler's constant folding is quadratic in a chain of operators, so this is
/// also what bounds how long a hostile template can hold up a server's start.
const MAX_TEMPLATE_BYTES: usize = 256 * 1024;

/// A stack for compiling a template of [`MAX_TEMPLATE_BYTES`], at about a
/// kilobyte a level in a debug build. The compiler
/// recurses once per operator in a chain like `a - a - a`, which has no limit
/// but the source's length. Only what a compile touches is ever committed.
const COMPILE_STACK_BYTES: usize = 512 * 1024 * 1024;

/// Compiles on a stack that a hostile template cannot run off the end of.
fn compile_template(
    mut env: Environment<'static>,
    source: String,
) -> std::result::Result<Environment<'static>, String> {
    if source.len() > MAX_TEMPLATE_BYTES {
        return Err(format!(
            "invalid chat template: {} bytes is more than the {MAX_TEMPLATE_BYTES} allowed",
            source.len()
        ));
    }
    std::thread::Builder::new()
        .name("template-compile".to_string())
        .stack_size(COMPILE_STACK_BYTES)
        .spawn(move || match env.add_template_owned("chat", source) {
            Ok(()) => Ok(env),
            Err(err) => Err(format!("invalid chat template: {err}")),
        })
        .map_err(|err| format!("invalid chat template: no thread to compile it on: {err}"))?
        .join()
        .unwrap_or_else(|_| Err("invalid chat template: compiling it panicked".to_string()))
}

/// How many instructions one render may run: what stops a template that never
/// finishes, in about ten seconds. A real template can be quadratic in the
/// messages, as `gemma-4`'s is at some 30 million instructions for 5000 of them,
/// and refusing a real conversation is worse than a hostile model being slow.
const TEMPLATE_FUEL: u64 = 100_000_000;

impl ChatTemplate {
    /// The call syntax read off this template, for a model no hand-written
    /// parser reads. See [`super::tool_learn`].
    pub fn learned_calls(&self) -> Option<std::sync::Arc<super::tool_learn::LearnedCalls>> {
        self.learned_calls
            .get_or_init(|| super::tool_learn::LearnedCalls::learn(self).map(std::sync::Arc::new))
            .clone()
    }

    pub fn new(source: String) -> Self {
        let mut env = Environment::new();
        // A template comes in a model file and runs on every request.
        env.set_fuel(Some(TEMPLATE_FUEL));
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
        let source = generation_tags_as_if(&source);
        let source = parenthesize_call_kwarg_conditionals(&source).into_owned();
        let env = compile_template(env, source);
        let mut template = Self {
            env,
            nine_char_ids: false,
            text_arguments: false,
            learned_calls: std::sync::OnceLock::new(),
        };
        let object = || serde_json::json!({"k": "v"});
        let text = || serde_json::json!("{\"k\": \"v\"}");
        template.nine_char_ids = !template.renders_a_call("call-1-0", object())
            && template.renders_a_call("a1b2c3d4e", object());
        let id = if template.nine_char_ids {
            "a1b2c3d4e"
        } else {
            "call-1-0"
        };
        template.text_arguments =
            !template.renders_a_call(id, object()) && template.renders_a_call(id, text());
        template
    }

    /// Whether a conversation with one tool call, by `id` and with
    /// `arguments`, and its result renders.
    fn renders_a_call(&self, id: &str, arguments: serde_json::Value) -> bool {
        let session = |id: &str| {
            let mut call = ChatMessage::text("assistant", "");
            call.tool_calls = Some(serde_json::json!([{
                "id": id,
                "type": "function",
                "function": {"name": "f", "arguments": arguments},
            }]));
            vec![
                ChatMessage::text("user", "u"),
                call,
                ChatMessage {
                    role: "tool".to_string(),
                    content: "r".to_string(),
                    tool_call_id: Some(id.to_string()),
                    name: Some("f".to_string()),
                    ..Default::default()
                },
            ]
        };
        self.render_messages(&session(id), true, "", "", Reasoning::default(), None)
            .is_ok()
    }

    /// `messages` as the template is given them. A tool call's `arguments`
    /// arrive as the JSON text the client sent; templates read them as a
    /// mapping (`arguments.items()`, `arguments | tojson`), so text that
    /// parses as a JSON object is passed as that object, unless the template
    /// renders a call only with text arguments. For a template that
    /// accepts only nine-character alphanumeric call ids, every other id is
    /// replaced by one derived from it, the same one in each request, so a
    /// call and its result still match and the prompt still extends the
    /// last one.
    fn for_template<'m>(&self, messages: &'m [ChatMessage]) -> Cow<'m, [ChatMessage]> {
        let rewrite = |m: &ChatMessage| {
            let calls_need = m
                .tool_calls
                .as_ref()
                .and_then(|c| c.as_array())
                .is_some_and(|calls| {
                    calls.iter().any(|call| {
                        (!self.text_arguments
                            && call
                                .pointer("/function/arguments")
                                .is_some_and(|a| a.is_string()))
                            || (self.nine_char_ids
                                && call
                                    .get("id")
                                    .and_then(|id| id.as_str())
                                    .is_some_and(|id| nine_char_id(id) != id))
                    })
                });
            calls_need
                || (self.nine_char_ids
                    && m.tool_call_id
                        .as_deref()
                        .is_some_and(|id| nine_char_id(id) != id))
        };
        if !messages.iter().any(rewrite) {
            return Cow::Borrowed(messages);
        }
        let mut out = messages.to_vec();
        for m in &mut out {
            if let Some(calls) = m.tool_calls.as_mut().and_then(|c| c.as_array_mut()) {
                for call in calls {
                    if !self.text_arguments
                        && let Some(arguments) = call.pointer_mut("/function/arguments")
                        && let Some(text) = arguments.as_str()
                        && let Ok(object @ serde_json::Value::Object(_)) =
                            serde_json::from_str::<serde_json::Value>(text)
                    {
                        *arguments = object;
                    }
                    if self.nine_char_ids
                        && let Some(id) = call.get_mut("id")
                        && let Some(text) = id.as_str()
                    {
                        *id = serde_json::Value::String(nine_char_id(text));
                    }
                }
            }
            if self.nine_char_ids
                && let Some(id) = m.tool_call_id.as_mut()
            {
                *id = nine_char_id(id);
            }
        }
        Cow::Owned(out)
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
        reasoning: Reasoning<'_>,
        tools: Option<&serde_json::Value>,
    ) -> Result<String> {
        self.render_inner(
            messages,
            add_generation_prompt,
            bos_token,
            eos_token,
            reasoning,
            tools,
        )
    }

    pub fn render(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        bos_token: &str,
        eos_token: &str,
        reasoning: Reasoning<'_>,
    ) -> Result<String> {
        self.render_inner(
            messages,
            add_generation_prompt,
            bos_token,
            eos_token,
            reasoning,
            None,
        )
    }

    /// Renders, and — when the level was this server's own default and the
    /// template refused it — renders again without one. See
    /// [`Reasoning::effort_is_default`].
    fn render_inner(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        bos_token: &str,
        eos_token: &str,
        reasoning: Reasoning<'_>,
        tools: Option<&serde_json::Value>,
    ) -> Result<String> {
        match self.render_once(
            messages,
            add_generation_prompt,
            bos_token,
            eos_token,
            reasoning,
            tools,
        ) {
            Err(err) if reasoning.effort_is_default && reasoning.effort.is_some() => self
                .render_once(
                    messages,
                    add_generation_prompt,
                    bos_token,
                    eos_token,
                    Reasoning {
                        effort: None,
                        ..reasoning
                    },
                    tools,
                )
                .map_err(|_| err),
            other => other,
        }
    }

    fn render_once(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        bos_token: &str,
        eos_token: &str,
        reasoning: Reasoning<'_>,
        tools: Option<&serde_json::Value>,
    ) -> Result<String> {
        self.render_messages(
            &self.for_template(messages),
            add_generation_prompt,
            bos_token,
            eos_token,
            reasoning,
            tools,
        )
    }

    fn render_messages(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
        bos_token: &str,
        eos_token: &str,
        reasoning: Reasoning<'_>,
        tools: Option<&serde_json::Value>,
    ) -> Result<String> {
        let env = self.env.as_ref().map_err(|err| anyhow!("{err}"))?;
        let tmpl = env.get_template("chat").expect("added at construction");
        // Built additively rather than as one `context!` per combination:
        // two optional variables that must each be *absent* (not null) when
        // unset would otherwise need four literals kept in step.
        let mut ctx = std::collections::BTreeMap::<&str, minijinja::Value>::new();
        ctx.insert(
            "messages",
            minijinja::Value::from(minijinja::value::Serde(messages)),
        );
        ctx.insert(
            "add_generation_prompt",
            minijinja::Value::from(add_generation_prompt),
        );
        ctx.insert("bos_token", minijinja::Value::from(bos_token));
        ctx.insert("eos_token", minijinja::Value::from(eos_token));
        if let Some(enable_thinking) = reasoning.enabled {
            ctx.insert("enable_thinking", minijinja::Value::from(enable_thinking));
        }
        // Absent unless configured, for the same reason `enable_thinking`
        // is: a template asks `reasoning_effort|default('xhigh')` or
        // `is defined`, and a null would be neither what it asked for nor
        // silence. Passed through verbatim — every template that reads it
        // spells its own levels ('xhigh'/'medium'/'low' for Qwen3.x,
        // 'high'/'medium'/'low' elsewhere) and validates them itself, with
        // an error this server surfaces as a 400.
        if let Some(reasoning_effort) = reasoning.effort {
            ctx.insert("reasoning_effort", minijinja::Value::from(reasoning_effort));
        }
        if let Some(tools) = tools {
            ctx.insert(
                "tools",
                minijinja::Value::from(minijinja::value::Serde(tools)),
            );
        }
        tmpl.render(minijinja::Value::from(minijinja::value::Serde(&ctx)))
            .map_err(|err| anyhow!("failed to render chat template: {err}"))
    }
}

/// `id` itself when it is nine ASCII letters and digits, otherwise nine
/// derived from it.
fn nine_char_id(id: &str) -> String {
    if id.len() == 9 && id.bytes().all(|b| b.is_ascii_alphanumeric()) {
        return id.to_string();
    }
    const DIGITS: &[u8; 62] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    let mut h = rustc_hash::FxHasher::default();
    std::hash::Hasher::write(&mut h, id.as_bytes());
    let mut n = std::hash::Hasher::finish(&h);
    (0..9)
        .map(|_| {
            let digit = DIGITS[(n % 62) as usize] as char;
            n /= 62;
            digit
        })
        .collect()
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

/// Rewrites Hugging Face's `{% generation %}` … `{% endgeneration %}` to
/// `{% if true %}` … `{% endif %}`, keeping each tag's whitespace-control
/// markers, and leaves everything else byte-for-byte alone.
///
/// The pair marks an assistant turn for `transformers`' training-time
/// assistant-token mask and renders as nothing more than its body, but
/// minijinja has no such statement and rejects the whole template at
/// compile time ("unknown statement generation") — so every chat request
/// fails, not only those with an assistant turn. `Muse-Spark-Open-816B`'s
/// template wraps its assistant branch in one. An always-true `if` is the
/// same block with the same nesting, and keeping the `-`/`+` markers keeps
/// the surrounding whitespace exactly as the original would trim it.
/// Borrowed and untouched when the source never names the statement.
fn generation_tags_as_if(source: &str) -> std::borrow::Cow<'_, str> {
    if !source.contains("generation") {
        return std::borrow::Cow::Borrowed(source);
    }
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while let Some(open) = rest.find("{%") {
        out.push_str(&rest[..open]);
        let tag = &rest[open..];
        // After the opener: in `{%}` the two overlap.
        let Some(close) = tag[2..].find("%}").map(|at| at + 2) else {
            rest = tag;
            break;
        };
        let inner = &tag[2..close];
        let lead = if inner.starts_with(['-', '+']) {
            &inner[..1]
        } else {
            ""
        };
        let trail = if inner.len() > lead.len() && inner.ends_with(['-', '+']) {
            &inner[inner.len() - 1..]
        } else {
            ""
        };
        let word = inner[lead.len()..inner.len() - trail.len()].trim();
        match word {
            "generation" | "endgeneration" => {
                let statement = if word == "generation" {
                    "if true"
                } else {
                    "endif"
                };
                out.push_str(&format!("{{%{lead} {statement} {trail}%}}"));
            }
            _ => out.push_str(&tag[..close + 2]),
        }
        rest = &tag[close + 2..];
    }
    out.push_str(rest);
    std::borrow::Cow::Owned(out)
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
mod hostile_template_tests {
    use super::*;

    /// A chain of operators nests as deeply as it is long, which no bracket
    /// count would see, and every kind of chain has to compile or be refused.
    #[test]
    fn a_long_chain_cannot_overflow_the_stack() {
        let length = MAX_TEMPLATE_BYTES / 32;
        for link in ["- 1 ", "|e ", ".a ", "~ 1 ", "and 1 "] {
            let chain = link.repeat(length / link.len());
            let template = ChatTemplate::new(format!("{{{{ x {chain} }}}}"));
            let _ = template.render(&[], true, "", "", Default::default());
        }
    }

    #[test]
    fn a_template_past_the_size_limit_is_refused() {
        let template = ChatTemplate::new("x".repeat(MAX_TEMPLATE_BYTES + 1));
        let error = template
            .render(&[], true, "", "", Default::default())
            .unwrap_err();
        assert!(format!("{error:#}").contains("allowed"), "{error:#}");
    }

    /// The fuel has to stop only what never finishes: a conversation longer than
    /// a context window holds, through each real template, must still render.
    ///
    /// `ORANGU_TEST_TEMPLATES=<dir> cargo test --release --bin orangu-server
    /// fuel_leaves_real -- --ignored --nocapture`
    #[test]
    #[ignore = "needs ORANGU_TEST_TEMPLATES"]
    fn fuel_leaves_real_templates_alone() {
        let dir = std::env::var("ORANGU_TEST_TEMPLATES").expect("a directory of templates");
        let tools: serde_json::Value =
            serde_json::json!((0..64).map(|i| serde_json::json!({
            "type": "function", "function": {"name": format!("tool{i}"), "description": "d",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}},
        })).collect::<Vec<_>>());
        let mut messages = vec![ChatMessage::text("system", "be brief")];
        for i in 0..2_500 {
            messages.push(ChatMessage::text("user", &format!("question {i}")));
            messages.push(ChatMessage::text("assistant", "an answer"));
        }
        messages.push(ChatMessage::text("user", "last"));
        let (mut rendered, mut refused, mut out_of_fuel) = (0, 0, Vec::new());
        let mut slowest = std::time::Duration::ZERO;
        for entry in std::fs::read_dir(dir).unwrap().flatten() {
            let Ok(source) = std::fs::read_to_string(entry.path()) else {
                continue;
            };
            let started = std::time::Instant::now();
            let result = ChatTemplate::new(source).render_with_tools(
                &messages,
                true,
                "<s>",
                "</s>",
                Reasoning::default(),
                Some(&tools),
            );
            slowest = slowest.max(started.elapsed());
            match result {
                Ok(_) => rendered += 1,
                Err(error) if format!("{error:#}").contains("fuel") => {
                    out_of_fuel.push(entry.file_name())
                }
                Err(_) => refused += 1,
            }
        }
        println!("{rendered} rendered, {refused} refused otherwise, slowest {slowest:?}");
        assert!(out_of_fuel.is_empty(), "out of fuel: {out_of_fuel:?}");
    }

    /// An opener and closer that share a character are not a tag to read, in a
    /// template the generation rewrite looks through.
    #[test]
    fn a_tag_opener_overlapping_its_closer_is_only_a_bad_template() {
        let template = ChatTemplate::new("{% generation %}{%}{% endgeneration %}".to_string());
        let _ = template.render(&[], true, "", "", Default::default());
    }

    /// A loop that would not end in a lifetime ends with an error instead.
    #[test]
    fn a_template_that_never_finishes_runs_out_of_fuel() {
        let forever = "{% for a in range(10000) %}{% for b in range(10000) %}\
                       {% for c in range(10000) %}x{% endfor %}{% endfor %}{% endfor %}";
        let template = ChatTemplate::new(forever.to_string());
        assert!(
            template
                .render(&[], true, "", "", Default::default())
                .is_err()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An agent turn as a client sends it back: the call's arguments as
    /// JSON text, the server's own call id.
    fn a_called_tool(id: &str) -> Vec<ChatMessage> {
        let mut call = ChatMessage::text("assistant", "");
        call.tool_calls = Some(serde_json::json!([{
            "id": id,
            "type": "function",
            "function": {"name": "read", "arguments": "{\"path\": \"a.rs\"}"},
        }]));
        vec![
            ChatMessage::text("user", "Read a.rs"),
            call,
            ChatMessage {
                role: "tool".to_string(),
                content: "fn main() {}".to_string(),
                tool_call_id: Some(id.to_string()),
                name: Some("read".to_string()),
                ..Default::default()
            },
        ]
    }

    #[test]
    fn a_template_reads_text_arguments_as_a_mapping() {
        let template = ChatTemplate::new(
            "{% for m in messages %}{% if m.tool_calls %}{% for c in m.tool_calls %}\
             {% for k, v in c.function.arguments.items() %}{{ k }}={{ v }};{% endfor %}\
             {{ c.function.arguments | tojson }}{% endfor %}{% endif %}{% endfor %}"
                .to_string(),
        );
        let out = template
            .render(
                &a_called_tool("call-1-0"),
                false,
                "",
                "",
                Reasoning::default(),
            )
            .unwrap();
        assert_eq!(out, r#"path=a.rs;{"path":"a.rs"}"#);
        // Text that is not a JSON object stays text.
        let mut odd = a_called_tool("call-1-0");
        odd[1].tool_calls = Some(serde_json::json!([{
            "id": "call-1-0", "type": "function",
            "function": {"name": "read", "arguments": "not json"},
        }]));
        let template = ChatTemplate::new(
            "{% for m in messages %}{% if m.tool_calls %}{{ m.tool_calls[0].function.arguments }}\
             {% endif %}{% endfor %}"
                .to_string(),
        );
        assert_eq!(
            template
                .render(&odd, false, "", "", Reasoning::default())
                .unwrap(),
            "not json"
        );
    }

    #[test]
    fn a_template_that_joins_arguments_as_text_keeps_them_text() {
        let template = ChatTemplate::new(
            "{% for m in messages %}{% if m.tool_calls %}{% for c in m.tool_calls %}\
             {{ '>>>' + c.function.name + ' ' + c.function.arguments }}{% endfor %}\
             {% endif %}{% endfor %}"
                .to_string(),
        );
        assert!(template.text_arguments);
        let out = template
            .render(
                &a_called_tool("call-1-0"),
                false,
                "",
                "",
                Reasoning::default(),
            )
            .unwrap();
        assert_eq!(out, r#">>>read {"path": "a.rs"}"#);
    }

    /// Mistral's check, reduced.
    const NINE_CHAR_IDS: &str = "{% for m in messages %}\
        {% if m.tool_calls %}{% for c in m.tool_calls %}\
        {% if c.id is not string or c.id|length != 9 %}{{ raise_exception('ids') }}{% endif %}\
        CALL[{{ c.id }}]{% endfor %}{% elif m.role == 'tool' %}\
        {% if m.tool_call_id|length != 9 %}{{ raise_exception('ids') }}{% endif %}\
        RESULT[{{ m.tool_call_id }}]{% endif %}{% endfor %}";

    #[test]
    fn a_template_that_wants_nine_character_ids_gets_them_and_they_match() {
        let template = ChatTemplate::new(NINE_CHAR_IDS.to_string());
        assert!(template.nine_char_ids);
        let out = template
            .render(
                &a_called_tool("call-1700000000-0"),
                false,
                "",
                "",
                Reasoning::default(),
            )
            .unwrap();
        let id = nine_char_id("call-1700000000-0");
        assert_eq!(id.len(), 9);
        assert!(id.bytes().all(|b| b.is_ascii_alphanumeric()));
        assert_eq!(out, format!("CALL[{id}]RESULT[{id}]"));
        // The same id each request; another call, another id.
        assert_eq!(nine_char_id("call-1700000000-0"), id);
        assert_ne!(nine_char_id("call-1700000000-1"), id);
        // An id already in the form is kept.
        assert_eq!(nine_char_id("a1B2c3D4e"), "a1B2c3D4e");
    }

    #[test]
    fn a_template_that_takes_any_id_sees_the_one_sent() {
        let template = ChatTemplate::new(
            "{% for m in messages %}{% if m.tool_calls %}CALL[{{ m.tool_calls[0].id }}]\
             {% elif m.role == 'tool' %}RESULT[{{ m.tool_call_id }}]{% endif %}{% endfor %}"
                .to_string(),
        );
        assert!(!template.nine_char_ids);
        let out = template
            .render(
                &a_called_tool("toolu_01"),
                false,
                "",
                "",
                Reasoning::default(),
            )
            .unwrap();
        assert_eq!(out, "CALL[toolu_01]RESULT[toolu_01]");
    }

    #[test]
    fn renders_a_simple_template() {
        let tmpl = ChatTemplate::new(
            "{% for m in messages %}{{ m.role }}: {{ m.content }}\n{% endfor %}\
             {% if add_generation_prompt %}assistant:{% endif %}"
                .to_string(),
        );
        let messages = vec![ChatMessage::text("user", "hi")];
        let out = tmpl
            .render(&messages, true, "<s>", "</s>", Reasoning::default())
            .unwrap();
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
        let out = tmpl
            .render(&messages, false, "<s>", "</s>", Reasoning::default())
            .unwrap();
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
        let out = tmpl
            .render(&messages, false, "<s>", "</s>", Reasoning::default())
            .unwrap();
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
        assert_eq!(
            tmpl.render(&messages, false, "", "", Reasoning::default())
                .unwrap(),
            "hi"
        );
        assert_eq!(
            tmpl.render(&[], false, "", "", Reasoning::default())
                .unwrap(),
            "none"
        );
    }

    /// [`generation_tags_as_if`]: the training-mask block renders as its
    /// body, with the whitespace its `-` markers trim still trimmed —
    /// `Muse-Spark-Open-816B`'s assistant branch, reduced.
    #[test]
    fn renders_a_generation_block_as_its_body() {
        let tmpl = ChatTemplate::new(
            "{%- for m in messages -%}\n  {%- generation %}\n  [{{ m.content }}]\n  \
             {%- endgeneration %}\n{%- endfor -%}"
                .to_string(),
        );
        let messages = vec![
            ChatMessage::text("user", "a"),
            ChatMessage::text("assistant", "b"),
        ];
        assert_eq!(
            tmpl.render(&messages, false, "", "", Reasoning::default())
                .unwrap(),
            "\n  [a]\n  [b]"
        );
        // Text that merely names the statement is not a tag.
        for source in [
            "generation",
            "{{ 'generation' }}",
            "{% if generation %}x{% endif %}",
        ] {
            assert_eq!(generation_tags_as_if(source), source);
        }
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
        let out = tmpl
            .render(&[], false, "<BOS>", "<EOS>", Reasoning::default())
            .unwrap();
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
        let out = tmpl
            .render(&messages, false, "", "", Reasoning::default())
            .unwrap();
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
            .render_with_tools(&[], false, "", "", Reasoning::default(), Some(&tools))
            .unwrap();
        assert_eq!(with, "TOOL:show_file;TOOL:run_shell_command;");

        let without = tmpl
            .render_with_tools(&[], false, "", "", Reasoning::default(), None)
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
            tmpl.render_with_tools(&[], false, "", "", Reasoning::default(), None)
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
        let out = tmpl
            .render(&messages, false, "", "", Reasoning::default())
            .unwrap();
        assert_eq!(out, "CALL:get_weather;RESULT[get_weather/call-1]=17C;");
    }

    /// OpenAI's array-of-parts `content`: the text parts become the string
    /// every template reads, the pictures are kept aside, and a plain string
    /// still deserializes as it always did.
    #[test]
    fn content_parts_split_into_text_and_images() {
        let message: ChatMessage = serde_json::from_str(
            r#"{"role": "user", "content": [
                {"type": "text", "text": "a cat"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": "on a mat"},
                {"type": "input_audio", "input_audio": {}}
            ]}"#,
        )
        .unwrap();
        assert_eq!(message.content, "a cat\non a mat");
        assert_eq!(message.images, vec!["data:image/png;base64,AAAA"]);
        // What a template sees: no `images` key at all.
        let json = serde_json::to_value(&message).unwrap();
        assert!(json.get("images").is_none());
        assert_eq!(json["content"], "a cat\non a mat");

        let plain: ChatMessage =
            serde_json::from_str(r#"{"role": "user", "content": "hi"}"#).unwrap();
        assert_eq!(plain.content, "hi");
        assert!(plain.images.is_empty());
        let none: ChatMessage =
            serde_json::from_str(r#"{"role": "assistant", "content": null}"#).unwrap();
        assert_eq!(none.content, "");
    }

    #[test]
    fn raise_exception_surfaces_as_an_error() {
        let tmpl = ChatTemplate::new(
            "{% if messages[0].role != 'system' %}{{ raise_exception('need a system message') }}{% endif %}"
                .to_string(),
        );
        let messages = vec![ChatMessage::text("user", "hi")];
        assert!(
            tmpl.render(&messages, false, "", "", Reasoning::default())
                .is_err()
        );
    }

    /// `reasoning_effort` reaches the template, and is *undefined* rather
    /// than null when unset — the difference between a template picking its
    /// own default and one seeing a value it has to validate. Qwen3.x's
    /// unset default is `xhigh`, the most expensive setting there is, which
    /// is why this knob exists at all.
    #[test]
    fn reasoning_effort_is_only_defined_when_given() {
        let tmpl = ChatTemplate::new("{{- reasoning_effort|default('xhigh') -}}".to_string());
        assert_eq!(
            tmpl.render(&[], false, "", "", Reasoning::default())
                .unwrap(),
            "xhigh"
        );
        assert_eq!(
            tmpl.render(
                &[],
                false,
                "",
                "",
                Reasoning {
                    enabled: None,
                    effort: Some("low"),
                    effort_is_default: false,
                }
            )
            .unwrap(),
            "low"
        );
    }

    /// A template that refuses this server's *default* level gets the
    /// prompt without one rather than turning the request into a `400`.
    /// The levels are the template's vocabulary — Qwen3.x raises on
    /// anything but `xhigh`/`medium`/`low` — and a default this server
    /// chose has no business breaking a model whose scale is spelled
    /// differently.
    #[test]
    fn a_default_level_the_template_refuses_is_dropped() {
        let tmpl = ChatTemplate::new(
            "{%- if reasoning_effort is defined -%}\
             {%- if reasoning_effort != 'thorough' -%}\
             {{- raise_exception('Unexpected reasoning effort') -}}\
             {%- endif -%}\
             {{- reasoning_effort -}}\
             {%- else -%}\
             none\
             {%- endif -%}"
                .to_string(),
        );
        let default = Reasoning {
            enabled: None,
            effort: Some("medium"),
            effort_is_default: true,
        };
        assert_eq!(tmpl.render(&[], false, "", "", default).unwrap(), "none");

        // A level the deployment asked for is a stated intention: it fails
        // loudly, with the template's own complaint, rather than being
        // silently swapped for something that renders.
        let asked_for = Reasoning {
            effort_is_default: false,
            ..default
        };
        let err = tmpl.render(&[], false, "", "", asked_for).unwrap_err();
        assert!(err.to_string().contains("Unexpected reasoning effort"));

        // And a default the template *accepts* is used, not dropped.
        let accepted = Reasoning {
            effort: Some("thorough"),
            ..default
        };
        assert_eq!(
            tmpl.render(&[], false, "", "", accepted).unwrap(),
            "thorough"
        );
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
            tmpl.render(
                &[],
                false,
                "",
                "",
                Reasoning {
                    enabled: Some(false),
                    effort: None,
                    effort_is_default: false,
                }
            )
            .unwrap(),
            "no-think"
        );
        assert_eq!(
            tmpl.render(
                &[],
                false,
                "",
                "",
                Reasoning {
                    enabled: Some(true),
                    effort: None,
                    effort_is_default: false,
                }
            )
            .unwrap(),
            "think"
        );
        assert_eq!(
            tmpl.render(&[], false, "", "", Reasoning::default())
                .unwrap(),
            "think"
        );
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
        let source = format!("{{{{ messages[0].tool_calls | tojson({args}) }}}}");
        let mut env = Environment::new();
        env.add_filter("tojson", tojson);
        env.add_template("t", &source)
            .map_err(|err| anyhow!("{err}"))?;
        env.get_template("t")
            .unwrap()
            .render(
                minijinja::context! { messages => minijinja::value::Serde(vec![serde_json::json!({"tool_calls": value})]) },
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
            .render_with_tools(&messages, true, "", "", Reasoning::default(), tools)
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

    /// `unsloth/Muse-Spark-Open-816B-A42B-GGUF`'s template, with tools so
    /// the ATEM declaration macros run as well: `namespace` flags read
    /// across a loop, `is sameas false`, `strftime_now`, and string
    /// slicing on `tojson` output.
    ///
    /// Run with `ORANGU_TEST_MUSE_SPARK_MODEL=/path/to/Muse-Spark-Open-816B-A42B-UD-Q4_K_XL-00001-of-00012.gguf
    /// cargo test --bin orangu-server chat_template::real_model_tests --
    /// --ignored`.
    #[test]
    #[ignore]
    fn the_muse_spark_template_renders_a_conversation_with_tools() {
        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Current weather for a place",
                "parameters": {
                    "type": "object",
                    "properties": {"place": {"type": "string"}},
                },
            },
        }]);
        let out = render_real_template("ORANGU_TEST_MUSE_SPARK_MODEL", Some(&tools));
        assert!(out.starts_with("<|start|>system<|message|>"), "{out}");
        assert!(out.contains("<|start|>user<|message|>And one more?<|eot|>"));
        assert!(out.contains("<|start|>assistant to=user<|message|>23, 29, 31.<|eot|>"));
        assert!(out.contains(r#"{"name": "get_weather""#), "{out}");
        // No effort asked for: the template's own default, `xhigh`.
        assert!(out.contains("Reasoning strength: 512."), "{out}");
        // The generation prompt stops before the recipient, which the
        // model writes itself.
        assert!(out.ends_with("<|eot|><|start|>assistant"), "{out}");
    }
}

#[cfg(test)]
mod prefix_tests {
    use super::*;

    /// orangu-bench's coding session as an agent runs it: the fixture's
    /// system prompt and tools, and each turn a user message, a tool call,
    /// its result and the answer. One message list per request: one after
    /// the user's message, which the model answers with the call, and one
    /// after the result, which it answers in prose.
    fn agent_turns() -> (serde_json::Value, Vec<Vec<ChatMessage>>) {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../orangu-bench/fixtures/coding-session.json"
        ))
        .unwrap();
        let tools = fixture["tools"].clone();
        let tool = tools[0]["function"]["name"].as_str().unwrap().to_string();
        let mut messages = vec![ChatMessage::text(
            "system",
            fixture["system"].as_str().unwrap(),
        )];
        let mut turns = Vec::new();
        for (i, turn) in fixture["turns"].as_array().unwrap().iter().enumerate() {
            messages.push(ChatMessage::text("user", turn.as_str().unwrap()));
            turns.push(messages.clone());
            // Named as this server names a call (`http::openai`).
            let id = format!("call-1700000000-{i}");
            messages.push(ChatMessage {
                role: "assistant".into(),
                tool_calls: Some(serde_json::json!([
                    {"id": id, "type": "function",
                     "function": {"name": tool, "arguments": "{\"path\": \"src/store.rs\"}"}}
                ])),
                ..Default::default()
            });
            messages.push(ChatMessage {
                role: "tool".into(),
                content: format!("contents of src/store.rs, read {i}"),
                tool_call_id: Some(id),
                name: Some(tool.clone()),
                ..Default::default()
            });
            turns.push(messages.clone());
            messages.push(ChatMessage::text(
                "assistant",
                &format!("Answer {i}: here is the change."),
            ));
        }
        (tools, turns)
    }

    /// One request's prompt against the previous request's, from
    /// [`shared_prefixes`].
    struct Shared {
        request: usize,
        /// The previous prompt's length, and how much of it this one starts
        /// with, in bytes.
        previous: usize,
        shared: usize,
        /// The text either side of where the two part.
        was: String,
        now: String,
    }

    /// Each request's rendered prompt against the previous request's: how
    /// much of it the new one starts with — what the server's prefix cache
    /// can answer of the new request before its new tail. A template that
    /// rewrites history (drops an earlier turn's reasoning, moves the tools,
    /// re-renders the system block) shares less than all of it. `Err` names
    /// the request that does not render.
    fn shared_prefixes(source: &str) -> Result<Vec<Shared>, String> {
        let (tools, turns) = agent_turns();
        let template = ChatTemplate::new(source.to_string());
        let mut previous: Option<String> = None;
        let mut out = Vec::new();
        for (i, messages) in turns.iter().enumerate() {
            let prompt = template
                .render_with_tools(
                    messages,
                    true,
                    "<bos>",
                    "<eos>",
                    Reasoning::default(),
                    Some(&tools),
                )
                .map_err(|e| format!("request {i} does not render: {e:#}"))?;
            if let Some(previous) = &previous {
                let shared = prompt
                    .char_indices()
                    .zip(previous.chars())
                    .find(|((_, a), b)| a != b)
                    .map_or(previous.len().min(prompt.len()), |((at, _), _)| at);
                let near = |text: &str| {
                    let from = text.floor_char_boundary(shared.saturating_sub(60));
                    let to = text.floor_char_boundary((shared + 60).min(text.len()));
                    text[from..to].to_string()
                };
                out.push(Shared {
                    request: i,
                    previous: previous.len(),
                    shared,
                    was: near(previous),
                    now: near(&prompt),
                });
            }
            previous = Some(prompt);
        }
        Ok(out)
    }

    /// Every request's rendered prompt starts with the whole of the previous
    /// request's; fails with the request it first parts at and the text on
    /// either side.
    fn assert_each_request_extends_the_last(name: &str, source: &str) {
        for s in shared_prefixes(source).unwrap_or_else(|e| panic!("{name}: {e}")) {
            assert_eq!(
                s.shared,
                s.previous,
                "{name}: request {} does not start with request {}'s prompt; they part at \
                 byte {}:\n  was: {:?}\n  now: {:?}",
                s.request,
                s.request - 1,
                s.shared,
                s.was,
                s.now
            );
        }
    }

    /// The check itself, on a template shaped like the tool-capable ones —
    /// tools declared in the opening block, calls and results in the
    /// transcript — and on one that rewrites history, which it must catch.
    #[test]
    fn the_prefix_check_passes_a_stable_template_and_catches_a_rewrite() {
        const STABLE: &str = "{{ bos_token }}{% for m in messages %}\
            {% if m.role == 'system' %}<sys>{{ m.content }}{% if tools %}<tools>\
            {% for t in tools %}{{ t.function.name }};{% endfor %}</tools>{% endif %}</sys>\
            {% elif m.get('tool_calls') %}<assistant><call>{{ m.tool_calls[0].function.name }}\
            {{ m.tool_calls[0].function.arguments }}</call>\
            {% elif m.role == 'tool' %}<result>{{ m.content }}</result>\
            {% else %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>{% endif %}\
            {% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}";
        assert_each_request_extends_the_last("stable", STABLE);
        // Numbers the turns from the end: every request renumbers them all.
        const REWRITES: &str = "{% for m in messages %}[{{ messages|length - loop.index }}]\
            {{ m.content }}{% endfor %}";
        let caught =
            std::panic::catch_unwind(|| assert_each_request_extends_the_last("rewrites", REWRITES));
        assert!(caught.is_err(), "a template that rewrites history passed");
    }

    /// The same session through real templates — every `.jinja` in
    /// `ORANGU_TEST_TEMPLATES` (`a.jinja:b.jinja`), and the template of the
    /// model in `ORANGU_TEST_MODEL` when that is set — as a report: how many
    /// requests rewrite the previous prompt, and the least of it one reuses,
    /// with the text where they part. Rewriting is the template's own
    /// choice, so only a template that cannot render the session fails.
    ///
    /// Run with `ORANGU_TEST_TEMPLATES=a.jinja:b.jinja cargo test --bin
    /// orangu-server how_much_of_each_prompt -- --ignored --nocapture`.
    #[test]
    #[ignore = "needs ORANGU_TEST_TEMPLATES or ORANGU_TEST_MODEL"]
    fn how_much_of_each_prompt_real_templates_reuse() {
        let mut sources: Vec<(String, String)> = Vec::new();
        if let Ok(paths) = std::env::var("ORANGU_TEST_TEMPLATES") {
            for path in paths.split(':').filter(|p| !p.is_empty()) {
                sources.push((path.to_string(), std::fs::read_to_string(path).unwrap()));
            }
        }
        if let Ok(path) = std::env::var("ORANGU_TEST_MODEL") {
            let gguf =
                orangu::gguf::GgufFile::open(std::path::Path::new(&path)).expect("open gguf");
            let source = gguf
                .metadata
                .iter()
                .find_map(|(k, v)| match (k.as_str(), v) {
                    ("tokenizer.chat_template", orangu::gguf::GgufValue::String(s)) => {
                        Some(s.clone())
                    }
                    _ => None,
                })
                .expect("the file has no tokenizer.chat_template");
            sources.push((path, source));
        }
        assert!(
            !sources.is_empty(),
            "set ORANGU_TEST_TEMPLATES or ORANGU_TEST_MODEL"
        );
        let mut unrendered = Vec::new();
        for (name, source) in &sources {
            match shared_prefixes(source) {
                Ok(shared) => {
                    let ratio = |s: &Shared| s.shared as f64 / s.previous.max(1) as f64;
                    let worst = shared
                        .iter()
                        .min_by(|a, b| ratio(a).total_cmp(&ratio(b)))
                        .expect("the session has more than one request");
                    let rewrites = shared.iter().filter(|s| s.shared < s.previous).count();
                    println!(
                        "{name}: {rewrites} of {} requests rewrite the previous prompt; the \
                         worst, request {}, reuses {} of {} bytes ({:.1}%)",
                        shared.len(),
                        worst.request,
                        worst.shared,
                        worst.previous,
                        ratio(worst) * 100.0
                    );
                    if worst.shared < worst.previous {
                        println!("    was: {:?}\n    now: {:?}", worst.was, worst.now);
                    }
                }
                Err(why) => {
                    println!("{name}: {why}");
                    unrendered.push(name.clone());
                }
            }
        }
        assert!(
            unrendered.is_empty(),
            "templates that cannot render the session: {unrendered:?}"
        );
    }
}

#[cfg(test)]
mod render_cost_tests {
    use super::*;

    /// What one chat request pays for its template: building the environment
    /// and parsing the template, then rendering, against rendering alone on a
    /// template compiled once. The conversation is orangu-bench's built-in
    /// coding session at its deepest turn, tools included.
    ///
    /// Run with `ORANGU_TEST_TEMPLATES=a.jinja:b.jinja cargo test --release
    /// --bin orangu-server render_cost -- --ignored --nocapture`.
    #[test]
    #[ignore = "a timing report; needs ORANGU_TEST_TEMPLATES"]
    fn render_cost() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../orangu-bench/fixtures/coding-session.json"
        ))
        .unwrap();
        let tools = fixture["tools"].clone();
        let mut messages = vec![ChatMessage::text(
            "system",
            fixture["system"].as_str().unwrap(),
        )];
        for turn in fixture["turns"].as_array().unwrap() {
            messages.push(ChatMessage::text("user", turn.as_str().unwrap()));
            messages.push(ChatMessage::text(
                "assistant",
                "Understood; here is the change.",
            ));
        }
        messages.pop();
        let paths = std::env::var("ORANGU_TEST_TEMPLATES").expect("set ORANGU_TEST_TEMPLATES");
        const N: u32 = 200;
        for path in paths.split(':') {
            let source = std::fs::read_to_string(path).unwrap();
            let render = |t: &ChatTemplate| {
                t.render_with_tools(&messages, true, "", "", Reasoning::default(), Some(&tools))
                    .unwrap()
            };
            let compiled = ChatTemplate::new(source.clone());
            let bytes = render(&compiled).len();
            let t0 = std::time::Instant::now();
            for _ in 0..N {
                std::hint::black_box(render(&ChatTemplate::new(source.clone())));
            }
            let fresh = t0.elapsed() / N;
            let t0 = std::time::Instant::now();
            for _ in 0..N {
                std::hint::black_box(render(&compiled));
            }
            let reused = t0.elapsed() / N;
            println!(
                "{:<40} {:>7} B prompt  compile+render {:>9.1?}  render {:>9.1?}  saved {:>9.1?}",
                std::path::Path::new(path)
                    .file_name()
                    .unwrap()
                    .to_string_lossy(),
                bytes,
                fresh,
                reused,
                fresh.saturating_sub(reused),
            );
        }
    }
}
