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

//! Recovering structured tool calls from what a model actually generated.
//!
//! A chat template writes tool *declarations* into the prompt, and the model
//! answers in whatever syntax that template taught it. Nothing about that
//! syntax is standard: gemma-4 emits a brace DSL between `<|tool_call>` and
//! `<tool_call|>` **special tokens**, Qwen/Hermes emit JSON between literal
//! `<tool_call>` tags, Mistral emits a JSON array after `[TOOL_CALLS]`. The
//! OpenAI API this server speaks has exactly one shape, so something has to
//! translate — this module.
//!
//! Only delimiter-anchored formats are recognised. Guessing that a bare JSON
//! object in an answer "looks like" a tool call is how a model that was asked
//! to *write* some JSON ends up having it executed instead, so a format
//! without an unambiguous opening marker is deliberately not supported.
//!
//! # Several bodies, one opener
//!
//! `<tool_call>` is not one format. Supported architectures write four
//! different things inside that same span — JSON, `<arg_key>`/`<arg_value>`
//! pairs, a `<function=…>`/`<parameter=…>` nest, and MiniMax's
//! `<invoke name=…>` with one element per argument — so the *body* is what
//! distinguishes them, discriminated by its own leading structure once the
//! span is already anchored. Whether the delimiters themselves arrive as
//! text or as special tokens is a third, independent axis, and a property
//! of the vocabulary rather than the format: Qwen writes `<tool_call>` as
//! text, Ling 3.0 spells the same six delimiters as tokens. See
//! [`marker_text`]. That is not a relaxation of the rule above: no
//! text outside a delimited span is ever examined, and a body that matches
//! none of the shapes stays in the answer as prose.
//!
//! Every format here was read off a real model's own chat template — the
//! branch that renders `tool_calls` back into the prompt, which is by
//! construction the syntax that model was trained to emit.

use serde_json::{Map, Value, json};

/// One call recovered from generated text.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    pub name: String,
    /// The arguments as a JSON **string**, which is the shape the OpenAI API
    /// puts on the wire (`function.arguments`) — not an object. Kept as the
    /// wire shape rather than a `Value` so nothing re-serialises it with
    /// different key ordering on the way out.
    pub arguments: String,
}

/// A generated span split into the prose the user should see and the calls the
/// caller should execute.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Split {
    pub content: String,
    pub calls: Vec<ParsedToolCall>,
}

impl Split {
    pub fn has_calls(&self) -> bool {
        !self.calls.is_empty()
    }
}

/// The delimiter pairs understood, longest opener first so that a format whose
/// opener is a prefix of another's cannot shadow it.
const FORMATS: &[(&str, &str, Syntax)] = [
    // gemma-4. The markers are special tokens; `engine::generate` re-renders
    // them into the stream as their literal text precisely so this can run.
    ("<|tool_call>", "<tool_call|>", Syntax::GemmaDsl),
    // Qwen2.5/3 and Hermes wrote JSON here; GLM and Nemotron reused the same
    // delimiters for entirely different bodies. See [`parse_tool_call_body`].
    ("<tool_call>", "</tool_call>", Syntax::ToolCallSpan),
    // Mistral: a JSON *array* with no closing marker — it runs to the end.
    ("[TOOL_CALLS]", "", Syntax::JsonArray),
    // Muse-Glimmer's ATEM block.
    (
        "<atem:function_calls>",
        "</atem:function_calls>",
        Syntax::InvokeTags("atem:"),
    ),
    // DeepSeek-V4's DSML block. The marker is spelled with U+FF5C fullwidth
    // vertical lines, not ASCII pipes — the template builds it from a
    // `dsml_token` variable, and a lookalike ASCII spelling would never match.
    (
        "<\u{ff5c}DSML\u{ff5c}tool_calls>",
        "</\u{ff5c}DSML\u{ff5c}tool_calls>",
        Syntax::InvokeTags("\u{ff5c}DSML\u{ff5c}"),
    ),
]
.as_slice();

#[derive(Debug, Clone, Copy, PartialEq, Default)]
enum Syntax {
    /// `call:NAME{key:value,...}` — see [`parse_gemma_dsl`].
    #[default]
    GemmaDsl,
    /// A `<tool_call>` span, whose body may be any of four shapes — see
    /// [`parse_tool_call_body`].
    ToolCallSpan,
    /// `[{"name": "...", "arguments": {...}}, ...]` to end of text.
    JsonArray,
    /// `<PREFIXinvoke name="...">` wrapping `<PREFIXparameter name="...">`
    /// elements, where `PREFIX` is the tag namespace this format spells its
    /// markers with. Two models use this shape with different prefixes, so
    /// the prefix travels with the format rather than being baked into the
    /// parser — see [`parse_invoke_tags`].
    InvokeTags(&'static str),
}

/// Split `text` into user-visible content and tool calls.
///
/// Unterminated spans are left in `content` untouched: a call the model did
/// not finish writing is not a call, and silently swallowing the partial text
/// would lose it from the answer as well.
///
/// `content` is returned **exactly** as the model wrote it, minus whole
/// tool-call spans — no trimming. The streaming path calls this once per token
/// as tokens arrive, so a `trim()` here is not tidying the edges of an answer,
/// it is deleting the leading space of every word in it. (It did: for one
/// release every streamed reply arrived as `Ihavelistedthefiles…`.) Callers
/// that want a tidy whole answer trim the assembled result themselves.
pub fn split(text: &str) -> Split {
    let mut content = String::new();
    let mut calls = Vec::new();
    let mut rest = text;

    // Whether the loop below gave up on a span rather than running out of
    // them — an opener that never closed, or a body that would not parse.
    // Both mean "stop reading calls and keep the rest as prose", and neither
    // may fall through to the recovery pass afterwards, which exists for a
    // different failure entirely.
    let mut gave_up = false;

    while let Some((at, open, close, syntax)) = next_opener(rest) {
        let body_start = at + open.len();
        let (body, after) = if close.is_empty() {
            (&rest[body_start..], "")
        } else {
            match rest[body_start..].find(close) {
                Some(end) => (
                    &rest[body_start..body_start + end],
                    &rest[body_start + end + close.len()..],
                ),
                // Opened and never closed — not a call.
                None => {
                    gave_up = true;
                    break;
                }
            }
        };
        let parsed = match syntax {
            Syntax::GemmaDsl => parse_gemma_dsl(body),
            Syntax::ToolCallSpan => parse_tool_call_body(body),
            Syntax::JsonArray => parse_json_array(body),
            Syntax::InvokeTags(prefix) => parse_invoke_tags(body, prefix),
        };
        if parsed.is_empty() {
            // A well-delimited span we could not read. Keeping the raw text is
            // the honest outcome: the user sees what the model produced rather
            // than a silently truncated answer.
            gave_up = true;
            break;
        }
        content.push_str(&rest[..at]);
        calls.extend(parsed);
        rest = after;
    }

    // A call whose opener the model forgot, once it has already shown it can
    // write one. See [`recover_unopened`].
    if !gave_up {
        while !calls.is_empty()
            && let Some((at, parsed, after)) = recover_unopened(rest)
        {
            content.push_str(&rest[..at]);
            calls.extend(parsed);
            rest = after;
        }
    }

    content.push_str(rest);
    Split { content, calls }
}

/// A gemma call whose *opening* marker the model forgot.
///
/// Having written one well-formed `<|tool_call>call:…{…}<tool_call|>` span,
/// gemma-4 will sometimes write the next call's body straight out and close
/// it — `call:create_file{…}<tool_call|>` with nothing opening it. Measured on
/// a three-file answer where only the first file was ever written: the model
/// asked for `index.html`, `style.css` and `game.js`, and two of the three
/// reached the caller as prose.
///
/// This is not a relaxation of the module's "only delimiter-anchored formats"
/// rule. `<tool_call|>` is a special token the model writes to *end* a call
/// and for nothing else, and `call:` is the format's own body opener; between
/// the two the span is as anchored as one whose opener arrived. A body with
/// neither is still nothing, and one that does not parse is still prose.
///
/// Only after a call has already been recovered from this message, so the
/// format is established rather than guessed at.
fn recover_unopened(text: &str) -> Option<(usize, Vec<ParsedToolCall>, &str)> {
    const CLOSE: &str = "<tool_call|>";
    let at = text.find(CLOSE)?;
    // The last `call:` before the closer: anything earlier is the prose the
    // model wrote between the two calls, and stays in the answer.
    let body_start = text[..at].rfind("call:")?;
    let parsed = parse_gemma_dsl(&text[body_start..at]);
    if parsed.is_empty() {
        return None;
    }
    Some((body_start, parsed, &text[at + CLOSE.len()..]))
}

/// Where the first tool-call opener in `text` starts, and which one it is.
pub fn first_opener(text: &str) -> Option<(usize, &'static str)> {
    next_opener(text).map(|(at, open, _, _)| (at, open))
}

/// `text` with its last call closed, when the model ended on a call it never
/// wrote the closer of and the call reads whole once closed; `text` as it
/// is otherwise.
pub fn close_trailing(text: &str) -> String {
    let Some((at, opener)) = last_opener(text) else {
        return text.to_string();
    };
    let Some(close) = FORMATS
        .iter()
        .find(|(open, _, _)| *open == opener)
        .map(|(_, close, _)| *close)
        .filter(|close| !close.is_empty() && !text[at..].contains(close))
    else {
        return text.to_string();
    };
    let closed = format!("{}{close}", text.trim_end());
    if split(&closed).calls.len() > split(text).calls.len() {
        closed
    } else {
        text.to_string()
    }
}

/// Where the last tool-call opener in `text` starts, and which one it is.
pub fn last_opener(text: &str) -> Option<(usize, &'static str)> {
    FORMATS
        .iter()
        .filter_map(|(open, _, _)| text.rfind(open).map(|at| (at, *open)))
        .max_by_key(|(at, open)| (*at, open.len()))
}

fn next_opener(text: &str) -> Option<(usize, &'static str, &'static str, Syntax)> {
    FORMATS
        .iter()
        .filter_map(|(open, close, syntax)| text.find(open).map(|at| (at, *open, *close, *syntax)))
        .min_by_key(|(at, open, _, _)| (*at, std::cmp::Reverse(open.len())))
}

/// gemma-4's string delimiter, as [`marker_text`] renders it — the literal
/// spelling of the vocabulary's own `<|"|>` token.
///
/// Five characters rather than one so that a `"` occurring *inside* a string
/// value cannot be mistaken for the end of it. The template writes the value
/// between two of these and escapes nothing, so this is the only thing that
/// says where the value stops.
const DSL_QUOTE: &str = "<|\"|>";

/// gemma-4's `call:NAME{key:value,...}`.
///
/// The value grammar is the one the template's own `format_argument` macro
/// writes: strings delimited by [`DSL_QUOTE`], `true`/`false`/`null` bare,
/// numbers bare, nested objects with **unquoted** keys, and arrays. That last
/// part is why this cannot simply be handed to `serde_json`. A bare `"` is
/// accepted as a delimiter too, for a model that spells one instead of
/// emitting the token.
fn parse_gemma_dsl(body: &str) -> Vec<ParsedToolCall> {
    let body = body.trim();
    // The `call:` prefix is what the template writes, and the model usually
    // writes it back — but not always, and a span missing it was a file the
    // model asked for and never got. Nothing is being guessed at by accepting
    // the bare form: this text is already anchored between `<|tool_call>` and
    // `<tool_call|>`, so it is a call or it is nothing. Without the prefix the
    // name still has to look like one, which is the whole of the difference.
    let (rest, prefixed) = match body.strip_prefix("call:") {
        Some(rest) => (rest, true),
        None => (body, false),
    };
    let Some(brace) = rest.find('{') else {
        return Vec::new();
    };
    let name = rest[..brace].trim();
    if name.is_empty() {
        return Vec::new();
    }
    if !prefixed
        && !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-')
    {
        return Vec::new();
    }
    let args = rest[brace..].trim();
    let Some(inner) = args.strip_prefix('{').and_then(|s| s.strip_suffix('}')) else {
        return Vec::new();
    };
    let mut parser = DslParser { s: inner, i: 0 };
    let Some(object) = parser.object_body() else {
        return Vec::new();
    };
    vec![ParsedToolCall {
        name: name.to_string(),
        arguments: Value::Object(object).to_string(),
    }]
}

/// A `<tool_call>` span's body, whichever of the four shapes it is.
///
/// Discriminated by leading structure, and each test is unambiguous: JSON
/// opens with a brace or a bracket, the key/value shape has an `<arg_key>`
/// element, the nested shape has a `<function=` element, and MiniMax's
/// element-per-argument shape opens with `<invoke name="`. A body matching
/// none of them yields nothing, which leaves the whole span in the answer as
/// prose — the right outcome for text that merely happens to sit between
/// those tags.
fn parse_tool_call_body(body: &str) -> Vec<ParsedToolCall> {
    let trimmed = body.trim();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        // A single object is the common case; an array here is not Mistral's
        // format but is unambiguous and free to accept.
        if let Some(call) = parse_json_object(trimmed) {
            return vec![call];
        }
        return parse_json_array(trimmed);
    }
    if trimmed.contains("<arg_key>") {
        return parse_arg_key_values(trimmed);
    }
    if trimmed.contains("<function=") {
        return parse_function_parameters(trimmed);
    }
    if trimmed.starts_with(INVOKE_ELEMENT_OPEN) {
        return parse_invoke_elements(trimmed);
    }
    Vec::new()
}

/// The opening of a MiniMax-M3 call inside a `<tool_call>` span.
const INVOKE_ELEMENT_OPEN: &str = "<invoke name=\"";

/// `<invoke name="NAME"><KEY>value</KEY>…</invoke>` — MiniMax-M3, whose
/// template writes each argument as an element named for its key, and
/// expands a structured value the same way: a mapping's entries as
/// elements, a list's entries as `<item>` elements. Every tag is preceded
/// by the `]<]minimax[>[` namespace token, which is special and so already
/// gone by the time the text reaches this parser.
///
/// A leaf is [`scalar_or_json`]'d, as in the other tag formats: the
/// template writes strings and numbers bare and booleans as JSON.
fn parse_invoke_elements(body: &str) -> Vec<ParsedToolCall> {
    let mut calls = Vec::new();
    let mut rest = body.trim_start();
    while let Some(after_open) = rest.strip_prefix(INVOKE_ELEMENT_OPEN) {
        let Some(name_end) = after_open.find("\">") else {
            return Vec::new();
        };
        let name = &after_open[..name_end];
        let inner_start = &after_open[name_end + 2..];
        let Some(end) = inner_start.find("</invoke>") else {
            return Vec::new();
        };
        let Some(arguments) = xml_elements(&inner_start[..end]) else {
            return Vec::new();
        };
        let arguments = arguments.into_iter().collect::<Map<String, Value>>();
        calls.push(ParsedToolCall {
            name: name.to_string(),
            arguments: Value::Object(arguments).to_string(),
        });
        rest = inner_start[end + "</invoke>".len()..].trim_start();
    }
    calls
}

/// A run of sibling elements, `<a>…</a><b>…</b>`, as `(name, value)`
/// pairs; `None` when `text` is not wholly that. Whitespace between
/// elements is ignored. An element whose content is itself a run of
/// elements becomes an array when every child is `<item>` and an object
/// otherwise; any other content is a leaf.
fn xml_elements(text: &str) -> Option<Vec<(String, Value)>> {
    let mut out = Vec::new();
    let mut rest = text.trim_start();
    while !rest.is_empty() {
        let after_lt = rest.strip_prefix('<')?;
        let name_end = after_lt.find('>')?;
        let name = &after_lt[..name_end];
        if name.is_empty() || name.starts_with('/') || name.contains(char::is_whitespace) {
            return None;
        }
        let content_start = &after_lt[name_end + 1..];
        let content_len = matching_close(content_start, name)?;
        let content = &content_start[..content_len];
        out.push((name.to_string(), element_value(content)));
        rest = content_start[content_len + name.len() + 3..].trim_start();
    }
    Some(out)
}

/// Where `<name>`'s matching `</name>` starts in `text` (the content after
/// the opening tag), counting nested elements of the same name — a list of
/// lists nests `<item>` in `<item>`.
fn matching_close(text: &str, name: &str) -> Option<usize> {
    let open = format!("<{name}>");
    let close = format!("</{name}>");
    let mut depth = 0usize;
    let mut at = 0;
    loop {
        let next_close = at + text[at..].find(&close)?;
        match text[at..next_close].find(&open) {
            Some(rel) => {
                depth += 1;
                at += rel + open.len();
            }
            None if depth == 0 => return Some(next_close),
            None => {
                depth -= 1;
                at = next_close + close.len();
            }
        }
    }
}

/// One element's content: nested elements as an array or an object, text
/// as a leaf.
fn element_value(content: &str) -> Value {
    if content.trim_start().starts_with('<')
        && let Some(children) = xml_elements(content)
        && !children.is_empty()
    {
        if children.iter().all(|(name, _)| name == "item") {
            return Value::Array(children.into_iter().map(|(_, v)| v).collect());
        }
        return Value::Object(children.into_iter().collect());
    }
    scalar_or_json(content)
}

/// `NAME<arg_key>k</arg_key><arg_value>v</arg_value>…`
///
/// The name is whatever precedes the first `<arg_key>`, and each value is
/// JSON when it parses as JSON and a plain string otherwise — which is what
/// the template does from the other side, writing values through a JSON
/// filter except where they are already strings.
fn parse_arg_key_values(body: &str) -> Vec<ParsedToolCall> {
    let Some(first) = body.find("<arg_key>") else {
        return Vec::new();
    };
    let name = body[..first].trim();
    if name.is_empty() {
        return Vec::new();
    }
    let mut arguments = Map::new();
    let mut rest = &body[first..];
    while let Some(key) = take_between(&mut rest, "<arg_key>", "</arg_key>") {
        let Some(value) = take_between(&mut rest, "<arg_value>", "</arg_value>") else {
            // A key with no value is a call the model did not finish writing.
            return Vec::new();
        };
        arguments.insert(key.trim().to_string(), scalar_or_json(value));
    }
    vec![ParsedToolCall {
        name: name.to_string(),
        arguments: Value::Object(arguments).to_string(),
    }]
}

/// `<function=NAME><parameter=KEY>value</parameter>…</function>`
fn parse_function_parameters(body: &str) -> Vec<ParsedToolCall> {
    let mut calls = Vec::new();
    let mut rest = body;
    while let Some(name) = take_between(&mut rest, "<function=", ">") {
        let name = name.trim();
        if name.is_empty() {
            return Vec::new();
        }
        // Bounded to this function's own span, so two calls in one body
        // cannot pull each other's parameters.
        let (mut params, after) = match rest.find("</function>") {
            Some(end) => (&rest[..end], &rest[end + "</function>".len()..]),
            None => return Vec::new(),
        };
        let mut arguments = Map::new();
        while let Some(key) = take_between(&mut params, "<parameter=", ">") {
            let Some(value) = take_between(&mut params, "", "</parameter>") else {
                return Vec::new();
            };
            arguments.insert(key.trim().to_string(), scalar_or_json(value));
        }
        calls.push(ParsedToolCall {
            name: name.to_string(),
            arguments: Value::Object(arguments).to_string(),
        });
        rest = after;
    }
    calls
}

/// `<PREFIXinvoke name="NAME"><PREFIXparameter name="KEY" …>value</…>`
///
/// One parser for two models, because the shape is identical and only the tag
/// namespace differs. One of them additionally marks each parameter
/// `string="true"` or `string="false"`, and that attribute is honoured where
/// present: a value declared a string stays a string even when it would parse
/// as a number, which is the distinction the attribute exists to carry.
fn parse_invoke_tags(body: &str, prefix: &str) -> Vec<ParsedToolCall> {
    let invoke_open = format!("<{prefix}invoke ");
    let invoke_close = format!("</{prefix}invoke>");
    let param_open = format!("<{prefix}parameter ");
    let param_close = format!("</{prefix}parameter>");

    let mut calls = Vec::new();
    let mut rest = body;
    while let Some(at) = rest.find(&invoke_open) {
        let head_start = at + invoke_open.len();
        let Some(head_end) = rest[head_start..].find('>') else {
            return Vec::new();
        };
        let Some(name) = attribute(&rest[head_start..head_start + head_end], "name") else {
            return Vec::new();
        };
        let body_start = head_start + head_end + 1;
        let (mut params, after) = match rest[body_start..].find(&invoke_close) {
            Some(end) => (
                &rest[body_start..body_start + end],
                &rest[body_start + end + invoke_close.len()..],
            ),
            // Opened and never closed — not a call, and the caller keeps the
            // raw text rather than inventing the missing half.
            None => return Vec::new(),
        };
        let mut arguments = Map::new();
        while let Some(at) = params.find(&param_open) {
            let head_start = at + param_open.len();
            let Some(head_end) = params[head_start..].find('>') else {
                return Vec::new();
            };
            let head = &params[head_start..head_start + head_end];
            let Some(key) = attribute(head, "name") else {
                return Vec::new();
            };
            let value_start = head_start + head_end + 1;
            let Some(end) = params[value_start..].find(&param_close) else {
                return Vec::new();
            };
            let raw = &params[value_start..value_start + end];
            let value = if attribute(head, "string").as_deref() == Some("true") {
                Value::String(raw.to_string())
            } else {
                scalar_or_json(raw)
            };
            arguments.insert(key, value);
            params = &params[value_start + end + param_close.len()..];
        }
        calls.push(ParsedToolCall {
            name,
            arguments: Value::Object(arguments).to_string(),
        });
        rest = after;
    }
    calls
}

/// The value of `name="…"` in a tag's attribute text.
fn attribute(head: &str, key: &str) -> Option<String> {
    let at = head.find(&format!("{key}=\""))? + key.len() + 2;
    let end = head[at..].find('"')?;
    Some(head[at..at + end].to_string())
}

/// Consumes up to and including `close`, returning what sat between `open`
/// (skipped if empty) and it.
fn take_between<'a>(rest: &mut &'a str, open: &str, close: &str) -> Option<&'a str> {
    let start = if open.is_empty() {
        0
    } else {
        rest.find(open)? + open.len()
    };
    let end = rest[start..].find(close)?;
    let value = &rest[start..start + end];
    *rest = &rest[start + end + close.len()..];
    Some(value)
}

/// A tag body as JSON where it is JSON, and as a plain string otherwise.
///
/// These formats write scalars bare — `true`, `42`, a sentence — and
/// structures as JSON, with nothing marking which is which. Trying JSON first
/// keeps an object an object; falling back to a string keeps a sentence a
/// sentence rather than dropping the argument.
fn scalar_or_json(raw: &str) -> Value {
    let trimmed = raw.trim();
    match serde_json::from_str::<Value>(trimmed) {
        Ok(value) => value,
        Err(_) => Value::String(trimmed.to_string()),
    }
}

fn parse_json_object(body: &str) -> Option<ParsedToolCall> {
    let value: Value = serde_json::from_str(body.trim()).ok()?;
    from_named_value(&value)
}

fn parse_json_array(body: &str) -> Vec<ParsedToolCall> {
    let Ok(Value::Array(items)) = serde_json::from_str::<Value>(body.trim()) else {
        return Vec::new();
    };
    items.iter().filter_map(from_named_value).collect()
}

/// `{"name": ..., "arguments"|"parameters": {...}}` in either spelling — both
/// are in the wild, sometimes from the same model family.
fn from_named_value(value: &Value) -> Option<ParsedToolCall> {
    let name = value.get("name")?.as_str()?.to_string();
    let arguments = value
        .get("arguments")
        .or_else(|| value.get("parameters"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    // Already-serialised arguments are passed through; an object is rendered.
    let arguments = match arguments {
        Value::String(s) => s,
        other => other.to_string(),
    };
    Some(ParsedToolCall { name, arguments })
}

/// A hand-written reader for gemma's brace DSL. Small enough to be obvious and
/// specific enough that no JSON parser would accept the input.
struct DslParser<'a> {
    s: &'a str,
    i: usize,
}

impl<'a> DslParser<'a> {
    fn peek(&self) -> Option<char> {
        self.s[self.i..].chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.i += c.len_utf8();
        Some(c)
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_whitespace()) {
            self.bump();
        }
    }

    /// Whether a string starts here, in either of the two spellings the
    /// delimiter arrives in — see [`Self::string`].
    fn at_quote(&self) -> bool {
        self.s[self.i..].starts_with(DSL_QUOTE) || self.peek() == Some('"')
    }

    /// `key:value(,key:value)*` with no surrounding braces. Keys are bare.
    fn object_body(&mut self) -> Option<Map<String, Value>> {
        let mut out = Map::new();
        self.skip_ws();
        if self.i >= self.s.len() {
            return Some(out);
        }
        loop {
            self.skip_ws();
            let key = self.key()?;
            self.skip_ws();
            if self.bump()? != ':' {
                return None;
            }
            self.skip_ws();
            let value = self.value()?;
            out.insert(key, value);
            self.skip_ws();
            match self.peek() {
                Some(',') => {
                    self.bump();
                }
                None => return Some(out),
                _ => return None,
            }
        }
    }

    /// A bare key, or a quoted one — templates differ on `escape_keys`.
    ///
    /// A lone `"` before a bare key (`{"command:<|"|>ls<|"|>}`) is a slip
    /// models make; read as an opening quote it swallows the `:` and the
    /// call is lost, so a quoted key that is not followed by `:` is read
    /// again as a bare one with the stray quote dropped.
    fn key(&mut self) -> Option<String> {
        if self.at_quote() {
            let start = self.i;
            if let Some(key) = self.string() {
                let after = self.i;
                self.skip_ws();
                let followed = self.peek() == Some(':');
                self.i = after;
                if followed {
                    return Some(key);
                }
            }
            self.i = start;
            if self.peek() != Some('"') {
                return None;
            }
            self.bump();
        }
        let start = self.i;
        while matches!(self.peek(), Some(c) if c != ':' && c != ',' && !c.is_whitespace()) {
            self.bump();
        }
        (self.i > start)
            .then(|| self.s[start..self.i].trim_end_matches('"').to_string())
            .filter(|key| !key.is_empty())
    }

    fn value(&mut self) -> Option<Value> {
        if self.at_quote() {
            return self.string().map(Value::String);
        }
        match self.peek()? {
            '{' => {
                self.bump();
                let inner_start = self.i;
                let inner_end = self.matching(inner_start, '{', '}')?;
                let mut inner = DslParser {
                    s: &self.s[inner_start..inner_end],
                    i: 0,
                };
                let object = inner.object_body()?;
                self.i = inner_end + 1;
                Some(Value::Object(object))
            }
            '[' => {
                self.bump();
                let inner_start = self.i;
                let inner_end = self.matching(inner_start, '[', ']')?;
                let mut inner = DslParser {
                    s: &self.s[inner_start..inner_end],
                    i: 0,
                };
                let items = inner.array_body()?;
                self.i = inner_end + 1;
                Some(Value::Array(items))
            }
            _ => self.scalar(),
        }
    }

    fn array_body(&mut self) -> Option<Vec<Value>> {
        let mut out = Vec::new();
        self.skip_ws();
        if self.i >= self.s.len() {
            return Some(out);
        }
        loop {
            self.skip_ws();
            out.push(self.value()?);
            self.skip_ws();
            match self.peek() {
                Some(',') => {
                    self.bump();
                }
                None => return Some(out),
                _ => return None,
            }
        }
    }

    /// Index of the `close` that matches an already-consumed `open`, honouring
    /// nesting and ignoring delimiters inside strings.
    fn matching(&self, from: usize, open: char, close: char) -> Option<usize> {
        let hay = &self.s[from..];
        let mut depth = 1usize;
        let mut offset = 0usize;
        while offset < hay.len() {
            let rest = &hay[offset..];
            // A `<|"|>` string is literal to its closing delimiter — braces,
            // brackets and quotes inside it are content, not structure.
            if let Some(after) = rest.strip_prefix(DSL_QUOTE) {
                offset += DSL_QUOTE.len() + after.find(DSL_QUOTE)? + DSL_QUOTE.len();
                continue;
            }
            let c = rest.chars().next()?;
            if c == '"' {
                offset += c.len_utf8();
                let mut escaped = false;
                loop {
                    let inner = hay[offset..].chars().next()?;
                    offset += inner.len_utf8();
                    if escaped {
                        escaped = false;
                    } else if inner == '\\' {
                        escaped = true;
                    } else if inner == '"' {
                        break;
                    }
                }
                continue;
            }
            if c == open {
                depth += 1;
            } else if c == close {
                depth -= 1;
                if depth == 0 {
                    return Some(from + offset);
                }
            }
            offset += c.len_utf8();
        }
        None
    }

    /// A string value, in either spelling of its delimiter.
    ///
    /// [`DSL_QUOTE`] is what the template writes and what the model emits, and
    /// it is the one that makes a value containing `"` — which is to say
    /// almost any file this server is asked to write — readable at all. A bare
    /// `"` is still accepted, for a model that spells the delimiter instead of
    /// emitting the token, and only there can a quote inside the value end it
    /// early; nothing can be done about that from here.
    ///
    /// The two also differ in what a backslash means. The template writes a
    /// `<|"|>` value **verbatim** — `'<|"|>' + argument + '<|"|>'`, no escaping
    /// on the way in — so nothing may be unescaped on the way out: a file
    /// containing `print("\n")` was arriving with that `\n` turned into a real
    /// newline, which is a syntax error in the file the model asked for. The
    /// bare-`"` form keeps the JSON-style escapes it always had, since a model
    /// spelling its own delimiter is following that convention rather than the
    /// template's.
    fn string(&mut self) -> Option<String> {
        if let Some(rest) = self.s[self.i..].strip_prefix(DSL_QUOTE) {
            let end = rest.find(DSL_QUOTE)?;
            self.i += DSL_QUOTE.len() + end + DSL_QUOTE.len();
            return Some(rest[..end].to_string());
        }
        if self.bump()? != '"' {
            return None;
        }
        let mut out = String::new();
        loop {
            match self.bump()? {
                '"' => return Some(out),
                '\\' => match self.bump()? {
                    'n' => out.push('\n'),
                    't' => out.push('\t'),
                    'r' => out.push('\r'),
                    other => out.push(other),
                },
                c => out.push(c),
            }
        }
    }

    /// An unquoted `true`/`false`/`null`/number, or — as a last resort — a
    /// bare word, which some templates emit for enum values.
    fn scalar(&mut self) -> Option<Value> {
        let start = self.i;
        while matches!(self.peek(), Some(c) if c != ',' && c != '}' && c != ']') {
            self.bump();
        }
        let raw = self.s[start..self.i].trim();
        if raw.is_empty() {
            return None;
        }
        Some(match raw {
            "true" => Value::Bool(true),
            "false" => Value::Bool(false),
            "null" => Value::Null,
            _ => match serde_json::from_str::<Value>(raw) {
                Ok(v @ Value::Number(_)) => v,
                _ => Value::String(raw.to_string()),
            },
        })
    }
}

/// The name of the call `opener` opens, once the call has written both it
/// and the start of its arguments: `span` is the text after the opener.
pub fn call_name(opener: &str, span: &str) -> Option<String> {
    let syntax = FORMATS.iter().find(|(open, _, _)| *open == opener)?.2;
    let name = match syntax {
        Syntax::GemmaDsl => gemma_call_name(span)?.to_string(),
        Syntax::JsonArray => json_call_head(span)?.0,
        Syntax::ToolCallSpan => {
            let body = span.trim_start();
            if body.starts_with(['{', '[']) {
                json_call_head(span)?.0
            } else if let Some(at) = body.find("<function=") {
                let rest = &body[at + "<function=".len()..];
                rest[..rest.find('>')?].trim().to_string()
            } else if let Some(rest) = body.strip_prefix(INVOKE_ELEMENT_OPEN) {
                rest[..rest.find("\">")?].to_string()
            } else {
                body[..body.find("<arg_key>")?].trim().to_string()
            }
        }
        Syntax::InvokeTags(prefix) => {
            let open = format!("<{prefix}invoke ");
            let at = span.find(&open)? + open.len();
            let head = &span[at..at + span[at..].find('>')?];
            attribute(head, "name")?
        }
    };
    let plain = !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-');
    plain.then_some(name)
}

/// How much of `text` could still begin a JSON document: up to the first
/// control character inside a string, which JSON forbids.
fn valid_json_prefix(text: &str) -> usize {
    let mut in_string = false;
    let mut escaped = false;
    for (i, c) in text.char_indices() {
        if in_string {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                c if (c as u32) < 0x20 => return i,
                _ => {}
            }
        } else if c == '"' {
            in_string = true;
        }
    }
    text.len()
}

/// What closes the JSON text `partial` was cut off in: a string it is in,
/// a value a key still waits for, and each open object and array.
fn json_closers(partial: &str) -> String {
    let mut stack = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for c in partial.chars() {
        if in_string {
            match c {
                _ if escaped => escaped = false,
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                stack.pop();
            }
            _ => {}
        }
    }
    let mut out = String::new();
    if in_string {
        out.push('"');
    }
    if escaped {
        out.insert(0, '\\');
    }
    let tail = format!("{partial}{out}");
    let last = tail.trim_end().chars().last();
    if last == Some(':') {
        out.push_str("null");
    }
    while let Some(close) = stack.pop() {
        out.push(close);
    }
    out
}

/// The end of the JSON value that starts at `start` in `text` — past its
/// closing brace, bracket or quote — or `None` while it is still being
/// written.
fn json_value_end(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            match b {
                _ if escaped => escaped = false,
                b'\\' => escaped = true,
                b'"' => {
                    in_string = false;
                    if depth == 0 {
                        return Some(i + 1);
                    }
                }
                _ => {}
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' | b'[' => depth += 1,
            // A bare scalar ends where its container goes on.
            b'}' | b']' | b',' if depth == 0 => return Some(i),
            b'}' | b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i + 1);
                }
            }
            _ => {}
        }
    }
    None
}

/// A JSON call's name, and where its arguments object starts in `span`,
/// once both are written — the name first, as every JSON call format
/// writes it. `None` for arguments given as a string or written before the
/// name.
fn json_call_head(span: &str) -> Option<(String, usize)> {
    let text = span.trim_start();
    let offset = span.len() - text.len();
    let mut i = match text.as_bytes().first()? {
        b'{' => 1,
        b'[' => text[1..].find('{')? + 2,
        _ => return None,
    };
    let mut name = None;
    loop {
        let rest = &text[i..];
        let key_at = i + rest.find(|c: char| !c.is_whitespace())?;
        if text.as_bytes()[key_at] != b'"' {
            return None;
        }
        let key_end = json_value_end(text, key_at)?;
        let key: String = serde_json::from_str(&text[key_at..key_end]).ok()?;
        let colon = key_end + text[key_end..].find(':')?;
        let value_at = colon + 1 + text[colon + 1..].find(|c: char| !c.is_whitespace())?;
        match key.as_str() {
            "name" => {
                let end = json_value_end(text, value_at)?;
                name = Some(serde_json::from_str::<String>(&text[value_at..end]).ok()?);
                i = end;
            }
            "arguments" | "parameters" => {
                return (text.as_bytes()[value_at] == b'{')
                    .then(|| name.map(|n| (n, offset + value_at)))
                    .flatten();
            }
            _ => i = json_value_end(text, value_at)?,
        }
        let rest = &text[i..];
        let next = i + rest.find(|c: char| !c.is_whitespace())?;
        if text.as_bytes()[next] != b',' {
            return None;
        }
        i = next + 1;
    }
}

/// A tag format's call so far, cut after its last whole parameter and
/// closed: its arguments as its own parser reads them. `None` until the
/// call has a whole parameter.
fn finished_tag_pairs(syntax: Syntax, span: &str) -> Option<Map<String, Value>> {
    let closed = match syntax {
        Syntax::InvokeTags(prefix) => {
            let close = format!("</{prefix}parameter>");
            let end = span.rfind(&close)? + close.len();
            format!("{}</{prefix}invoke>", &span[..end])
        }
        _ if span.contains("<function=") => {
            let end = span.rfind("</parameter>")? + "</parameter>".len();
            format!("{}</function>", &span[..end])
        }
        _ => {
            let end = span.rfind("</arg_value>")? + "</arg_value>".len();
            span[..end].to_string()
        }
    };
    let calls = match syntax {
        Syntax::InvokeTags(prefix) => parse_invoke_tags(&closed, prefix),
        _ => parse_tool_call_body(&closed),
    };
    match serde_json::from_str::<Value>(&calls.first()?.arguments).ok()? {
        Value::Object(map) => Some(map),
        _ => None,
    }
}

/// A gemma-4 call's name, once its `{` has arrived: `span` is the text after
/// its opener.
fn gemma_call_name(span: &str) -> Option<&str> {
    let body = span.trim_start();
    let rest = body.strip_prefix("call:").unwrap_or(body);
    let name = rest[..rest.find('{')?].trim();
    (!name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-'))
    .then_some(name)
}

/// The JSON text of `s` without its surrounding quotes.
fn json_escaped(s: &str) -> String {
    let quoted = Value::String(s.to_string()).to_string();
    quoted[1..quoted.len() - 1].to_string()
}

/// How far a call's arguments have been written, read at the top level.
enum Tail {
    /// Mid-value, or between values.
    Open,
    /// Inside a top-level `<|"|>` string, whose opening delimiter is here.
    String(usize),
    /// The object's closing `}` is here.
    Ended(usize),
    /// A bare `"` at the top level — a spelling only the whole call can be
    /// read in.
    Unclear,
}

/// The last top-level `,` in `inner` (a call's arguments after their `{`)
/// and where the text ends — the same lexing as [`DslParser::matching`].
fn scan_arguments(inner: &str) -> (Option<usize>, Tail) {
    let mut i = 0;
    let mut depth = 0usize;
    let mut last = None;
    while i < inner.len() {
        let rest = &inner[i..];
        if let Some(after) = rest.strip_prefix(DSL_QUOTE) {
            match after.find(DSL_QUOTE) {
                Some(end) => {
                    i += 2 * DSL_QUOTE.len() + end;
                    continue;
                }
                None if depth == 0 => return (last, Tail::String(i)),
                None => return (last, Tail::Open),
            }
        }
        let Some(c) = rest.chars().next() else { break };
        match c {
            '"' if depth == 0 => return (last, Tail::Unclear),
            '"' => {
                let mut escaped = false;
                let mut closed = false;
                for (n, inner_c) in rest.char_indices().skip(1) {
                    if escaped {
                        escaped = false;
                    } else if inner_c == '\\' {
                        escaped = true;
                    } else if inner_c == '"' {
                        i += n + 1;
                        closed = true;
                        break;
                    }
                }
                if !closed {
                    return (last, Tail::Open);
                }
                continue;
            }
            '{' | '[' => depth += 1,
            '}' if depth == 0 => return (last, Tail::Ended(i)),
            ']' if depth == 0 => return (last, Tail::Unclear),
            '}' | ']' => depth -= 1,
            ',' if depth == 0 => last = Some(i),
            _ => {}
        }
        i += c.len_utf8();
    }
    (last, Tail::Open)
}

/// One call's arguments as JSON text, a piece at a time while the model
/// writes them. For gemma-4's DSL: each finished `key:value` pair once
/// [`DslParser`] has read it, and a top-level `<|"|>` string — a file's
/// content, a command — a few characters at a time. For a JSON call: its
/// arguments object's own text as it is written. For the tag formats: each
/// parameter once the format's parser reads it whole. The pieces join into
/// the JSON the whole call parses to; [`finish`](Self::finish) writes the
/// rest and says whether what was sent agrees with that parse.
#[derive(Default)]
pub struct ArgumentStream {
    syntax: Syntax,
    /// A JSON call's arguments: the bytes of their text already sent.
    raw_sent: usize,
    /// ...and that text.
    raw: String,
    opened: bool,
    /// The pairs sent whole, or whose string was streamed and closed.
    sent: Vec<(String, Value)>,
    /// The key whose string is being streamed, and the text sent of it.
    open: Option<(String, String)>,
    /// A spelling that only the whole call can be read in was met: nothing
    /// more is sent until [`finish`](Self::finish).
    stalled: bool,
    diverged: bool,
}

impl ArgumentStream {
    fn separator(&self) -> &'static str {
        if self.sent.is_empty() && self.open.is_none() {
            ""
        } else {
            ","
        }
    }

    /// The pairs in `completed` not sent yet, closing the streamed string
    /// when it is one of them.
    fn send_pairs(&mut self, completed: &str, out: &mut String) -> bool {
        let mut parser = DslParser { s: completed, i: 0 };
        let Some(pairs) = parser.object_body() else {
            return false;
        };
        self.send_map(pairs, out)
    }

    /// The pairs of `pairs` not sent yet, closing the streamed string when
    /// it is one of them.
    fn send_map(&mut self, mut pairs: Map<String, Value>, out: &mut String) -> bool {
        for (key, previous) in &self.sent {
            if let Some(value) = pairs.remove(key) {
                self.diverged |= *previous != value;
            }
        }
        if let Some((key, streamed)) = self.open.take() {
            let Some(value) = pairs.remove(&key) else {
                self.open = Some((key, streamed));
                // A pair finished while another's string is still open is
                // not a reading this stream can follow.
                self.stalled |= !pairs.is_empty();
                return pairs.is_empty();
            };
            match value.as_str() {
                Some(full) if full.starts_with(&streamed) => {
                    out.push_str(&json_escaped(&full[streamed.len()..]));
                }
                _ => self.diverged = true,
            }
            out.push('"');
            self.sent.push((key, value));
        }
        for (key, value) in pairs {
            out.push_str(self.separator());
            out.push_str(&Value::String(key.clone()).to_string());
            out.push(':');
            out.push_str(&value.to_string());
            self.sent.push((key, value));
        }
        true
    }

    /// A stream for the call `opener` (one of [`FORMATS`]' openers) opens.
    pub fn new(opener: &str) -> Self {
        let syntax = FORMATS
            .iter()
            .find(|(open, _, _)| *open == opener)
            .map_or(Syntax::GemmaDsl, |(_, _, syntax)| *syntax);
        Self {
            syntax,
            ..Self::default()
        }
    }

    /// What to send now: `span` is the call's text so far, after its
    /// opener.
    pub fn advance(&mut self, span: &str) -> String {
        match self.syntax {
            Syntax::GemmaDsl => self.advance_gemma(span),
            Syntax::JsonArray => self.advance_json(span),
            Syntax::ToolCallSpan if span.trim_start().starts_with(['{', '[']) => {
                self.advance_json(span)
            }
            _ => self.advance_tags(span),
        }
    }

    /// A JSON call's arguments object, as much of its text as is written.
    fn advance_json(&mut self, span: &str) -> String {
        let Some((_, start)) = json_call_head(span) else {
            return String::new();
        };
        self.opened = true;
        if self.stalled {
            return String::new();
        }
        let end = json_value_end(span, start).unwrap_or(span.len());
        let written = &span[start..end];
        // Only what is still JSON goes out: a client parses the pieces as
        // they come.
        let valid = valid_json_prefix(written);
        self.stalled = valid < written.len();
        let ready = &written[..valid];
        let out = ready.get(self.raw_sent..).unwrap_or_default().to_string();
        self.raw_sent = self.raw_sent.max(ready.len());
        self.raw = ready.to_string();
        out
    }

    /// A tag format's parameters, each once it is whole.
    fn advance_tags(&mut self, span: &str) -> String {
        let Some(pairs) = finished_tag_pairs(self.syntax, span) else {
            return String::new();
        };
        let mut out = String::new();
        if !self.opened {
            self.opened = true;
            out.push('{');
        }
        if !self.stalled && !self.send_map(pairs, &mut out) {
            self.stalled = true;
        }
        out
    }

    fn advance_gemma(&mut self, span: &str) -> String {
        let mut out = String::new();
        let body = span.trim_start();
        let rest = body.strip_prefix("call:").unwrap_or(body);
        let Some(brace) = rest.find('{') else {
            return out;
        };
        if !self.opened {
            self.opened = true;
            out.push('{');
        }
        if self.stalled {
            return out;
        }
        let inner = &rest[brace + 1..];
        let (last_comma, tail) = scan_arguments(inner);
        let segment_start = match tail {
            Tail::Ended(end) => {
                self.send_pairs(&inner[..end], &mut out);
                return out;
            }
            _ => match last_comma {
                Some(comma) => {
                    if !self.send_pairs(&inner[..comma], &mut out) {
                        return out;
                    }
                    comma + 1
                }
                None => 0,
            },
        };
        let quote = match tail {
            Tail::String(at) => at - segment_start,
            Tail::Unclear => {
                self.stalled = true;
                return out;
            }
            _ => return out,
        };
        let segment = &inner[segment_start..];
        let opening = &segment[..quote + DSL_QUOTE.len()];
        // The key, read by the parser itself from the pair with its string
        // closed empty.
        let mut parser = DslParser {
            s: &format!("{opening}{DSL_QUOTE}"),
            i: 0,
        };
        let Some(key) = parser
            .object_body()
            .filter(|pairs| pairs.len() == 1)
            .and_then(|pairs| pairs.into_iter().next())
            .filter(|(_, value)| value.as_str() == Some(""))
            .map(|(key, _)| key)
        else {
            return out;
        };
        if self.sent.iter().any(|(k, _)| *k == key) {
            self.stalled = true;
            return out;
        }
        match &self.open {
            Some((open, _)) if *open != key => {
                self.stalled = true;
                return out;
            }
            Some(_) => {}
            None => {
                out.push_str(self.separator());
                out.push_str(&Value::String(key.clone()).to_string());
                out.push_str(":\"");
                self.open = Some((key, String::new()));
            }
        }
        let raw = &segment[quote + DSL_QUOTE.len()..];
        // A closing delimiter that has begun to arrive is not content.
        let held = (1..DSL_QUOTE.len())
            .rev()
            .find(|&n| raw.ends_with(&DSL_QUOTE[..n]))
            .unwrap_or(0);
        let ready = &raw[..raw.len() - held];
        if let Some((_, streamed)) = &mut self.open
            && ready.len() > streamed.len()
            && ready.starts_with(streamed.as_str())
        {
            out.push_str(&json_escaped(&ready[streamed.len()..]));
            streamed.push_str(&ready[streamed.len()..]);
        }
        out
    }

    /// The rest of the JSON, once the whole call parsed to `arguments`, and
    /// whether everything sent before agrees with them.
    pub fn finish(mut self, span: &str, arguments: &str) -> (String, bool) {
        if self.raw_sent > 0 {
            let written = json_call_head(span).and_then(|(_, start)| {
                let end = json_value_end(span, start)?;
                Some(&span[start..end])
            });
            let parsed = |text: &str| serde_json::from_str::<Value>(text).ok();
            return match written {
                Some(text) => (
                    text.get(self.raw_sent..).unwrap_or_default().to_string(),
                    text.len() >= self.raw_sent && parsed(text) == parsed(arguments),
                ),
                None => (String::new(), false),
            };
        }
        let whole = match serde_json::from_str::<Value>(arguments) {
            Ok(Value::Object(map)) => map,
            _ => Map::new(),
        };
        let mut out = String::new();
        if !self.opened {
            out.push('{');
        }
        if let Some((key, streamed)) = self.open.take() {
            match whole.get(&key).and_then(Value::as_str) {
                Some(full) if full.starts_with(&streamed) => {
                    out.push_str(&json_escaped(&full[streamed.len()..]));
                }
                _ => self.diverged = true,
            }
            out.push('"');
            let value = whole.get(&key).cloned().unwrap_or(Value::Null);
            self.sent.push((key, value));
        }
        for (key, value) in &self.sent {
            self.diverged |= whole.get(key) != Some(value);
        }
        for (key, value) in &whole {
            if self.sent.iter().any(|(k, _)| k == key) {
                continue;
            }
            out.push_str(if self.sent.is_empty() { "" } else { "," });
            out.push_str(&Value::String(key.clone()).to_string());
            out.push(':');
            out.push_str(&value.to_string());
            self.sent.push((key.clone(), value.clone()));
        }
        out.push('}');
        (out, !self.diverged)
    }

    /// The rest of the JSON for a call that never parsed: what was sent,
    /// closed.
    pub fn abandon(self) -> String {
        if self.raw_sent > 0 {
            return json_closers(&self.raw);
        }
        let mut out = String::new();
        if !self.opened {
            out.push('{');
        }
        if self.open.is_some() {
            out.push('"');
        }
        out.push('}');
        out
    }
}

/// The literal text a special token should be rendered as when it carries
/// tool-call structure, or `None` for every other special token (which stays
/// suppressed, as all of them used to be).
///
/// `<|"|>` is gemma's string delimiter *inside* a call's arguments, and it is
/// rendered as its own literal spelling rather than as `"`. Flattening it to
/// `"` is what [`split`] used to receive, and it made the delimiter
/// indistinguishable from a `"` the model wrote *inside* the value: a
/// `create_file` whose content contained `print("hi")` ended its string at
/// that quote, failed to parse, and reached the caller as prose. Nearly every
/// source file has a quote in it, so nearly every file the model tried to
/// write was silently never written. Kept whole, the delimiter is
/// unambiguous — see [`DslParser::string`]. It has no meaning outside a call
/// and a model does not emit it elsewhere.
///
/// Whether a given format's delimiters need an entry here is a property of
/// the *vocabulary*, not of the format. Qwen writes `<tool_call>` as ordinary
/// text, so it arrives through `decode` and needs nothing; Ling 3.0
/// (`bailingmoe3`) spells the same six delimiters as USER_DEFINED tokens, so
/// without these entries the model's call reaches [`split`] as
/// `get_weather\ncity\nParis` — prose with nothing left to say it was a call,
/// which is not a parse failure the caller can see but a tool invocation that
/// silently became chat.
///
/// `<tool_response>`/`</tool_response>` are deliberately absent: those frame a
/// tool's *result* on the way in, and a model has no reason to write them.
pub fn marker_text(token: &str) -> Option<&'static str> {
    match token {
        "<|tool_call>" => Some("<|tool_call>"),
        "<tool_call|>" => Some("<tool_call|>"),
        "<tool_call>" => Some("<tool_call>"),
        "</tool_call>" => Some("</tool_call>"),
        "<arg_key>" => Some("<arg_key>"),
        "</arg_key>" => Some("</arg_key>"),
        "<arg_value>" => Some("<arg_value>"),
        "</arg_value>" => Some("</arg_value>"),
        "<|\"|>" => Some(DSL_QUOTE),
        _ => None,
    }
}

/// Whether `text` could still grow into a tool call — used by the streaming
/// path to hold back a partial opener instead of showing the user half a
/// delimiter and then taking it away.
pub fn may_be_partial(text: &str) -> bool {
    FORMATS.iter().any(|(open, close, _)| {
        // An opener that has started but not finished arriving. Split at
        // **character** boundaries, not byte indices: one opener is spelled
        // with fullwidth vertical lines, and slicing into the middle of one
        // panics rather than failing to match.
        open.char_indices()
            .skip(1)
            .any(|(n, _)| text.ends_with(&open[..n]))
            // ...or a complete, still-unclosed span.
            || (text.contains(open) && !close.is_empty() && !text.contains(close))
            || (text.contains(open) && close.is_empty())
    })
}

#[cfg(test)]
mod tests {
    /// Ling 3.0's six tool-call delimiters are USER_DEFINED tokens in its
    /// vocabulary, so `engine::generate` suppresses them like every other
    /// special token unless [`marker_text`] claims them back.
    ///
    /// Without them the model's call arrives as `get_weather\ncity\nParis`
    /// — which [`split`] correctly refuses to treat as a call, so the
    /// caller silently gets chat where it asked for a tool invocation. The
    /// second half of this test is the part that matters: reassembled
    /// through `marker_text`, the same token stream parses.
    #[test]
    fn the_bailing_tool_call_delimiters_survive_special_token_suppression() {
        let tokens = [
            "<tool_call>",
            "get_weather",
            "<arg_key>",
            "city",
            "</arg_key>",
            "\n",
            "<arg_value>",
            "Paris",
            "</arg_value>",
            "\n",
            "</tool_call>",
        ];

        // What the stream would carry if every special token stayed hidden.
        let suppressed: String = tokens
            .iter()
            .filter(|t| marker_text(t).is_none())
            .copied()
            .collect();
        assert!(
            split(&suppressed).calls.is_empty(),
            "without the markers this is prose, and must not be mistaken for a call"
        );

        let rendered: String = tokens.iter().map(|t| marker_text(t).unwrap_or(t)).collect();
        let parsed = split(&rendered);
        assert_eq!(parsed.content, "");
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].name, "get_weather");
        assert_eq!(parsed.calls[0].arguments, r#"{"city":"Paris"}"#);
    }

    use super::*;

    /// `span`'s call streamed as it grows `step` characters at a time,
    /// then finished from the whole call's parse: the JSON sent, and
    /// whether the stream agreed with the parse.
    fn streamed(span: &str, step: usize) -> (String, bool) {
        let mut stream = ArgumentStream::default();
        let mut out = String::new();
        let ends: Vec<usize> = span.char_indices().map(|(i, _)| i).collect();
        for &end in ends.iter().step_by(step) {
            out.push_str(&stream.advance(&span[..end]));
        }
        out.push_str(&stream.advance(span));
        let parsed = parse_gemma_dsl(span);
        let (rest, agreed) = stream.finish(span, &parsed[0].arguments);
        out.push_str(&rest);
        (out, agreed)
    }

    /// However a call arrives, the streamed pieces join into JSON equal to
    /// what the whole call parses to.
    #[test]
    fn streamed_arguments_join_into_the_parsed_call() {
        for span in [
            "call:create_file{path:<|\"|>/a.py<|\"|>,content:<|\"|>print(\"hi, {x}\")\n<|\"|>}",
            "call:f{opts:{a:1,b:[1,2]},name:<|\"|>x<|\"|>,n:5}",
            "write{path:<|\"|>p<|\"|>}",
            "call:bash{\"command:<|\"|>ls<|\"|>}",
            "call:f{}",
            "call:f{s:\"a,b\",t:<|\"|>c<|\"|>}",
            "call:f{z:<|\"|>last<|\"|>,a:true}",
        ] {
            let whole: Value = serde_json::from_str(&parse_gemma_dsl(span)[0].arguments).unwrap();
            for step in [1, 3, 7, 1000] {
                let (json, agreed) = streamed(span, step);
                assert!(agreed, "{span} by {step}: {json}");
                let value: Value = serde_json::from_str(&json)
                    .unwrap_or_else(|err| panic!("{span} by {step}: {json}: {err}"));
                assert_eq!(value, whole, "{span} by {step}");
            }
        }
    }

    /// A file's content is sent while it is written, not when the call ends.
    #[test]
    fn a_string_argument_is_sent_as_it_is_written() {
        let mut stream = ArgumentStream::default();
        let mut out = String::new();
        let mut span = String::new();
        for part in [
            "call:create_file{",
            "path:<|\"|>/a.py<|\"|>,",
            "content:<|\"|>import os\n",
            "print(\"x\")",
            "<|",
        ] {
            span.push_str(part);
            out.push_str(&stream.advance(&span));
        }
        assert_eq!(
            out,
            "{\"path\":\"/a.py\",\"content\":\"import os\\nprint(\\\"x\\\")"
        );
    }

    /// The opener's name is read once its `{` arrives, and a name that does
    /// not look like one is not a call.
    #[test]
    fn a_gemma_call_is_named_once_its_brace_arrives() {
        assert_eq!(gemma_call_name("call:create_file"), None);
        assert_eq!(gemma_call_name("call:create_file{pa"), Some("create_file"));
        assert_eq!(gemma_call_name("bash{"), Some("bash"));
        assert_eq!(gemma_call_name("some prose {"), None);
    }

    /// Every other format, however its call arrives: the streamed pieces
    /// join into JSON equal to the whole call's parse, and something was
    /// sent before the call was whole.
    #[test]
    fn every_format_streams_into_the_parsed_call() {
        let calls = [
            (
                "<tool_call>",
                "</tool_call>",
                "\n{\"name\": \"write\", \"arguments\": {\"path\": \"/a\", \"content\": \"x, \\\"y\\\" {z}\", \"n\": 5}}\n",
            ),
            (
                "[TOOL_CALLS]",
                "",
                "[{\"name\": \"write\", \"arguments\": {\"n\": 5, \"b\": true, \"l\": [1, {\"a\": \"]\"}]}}]",
            ),
            (
                "<tool_call>",
                "</tool_call>",
                "\n<function=write>\n<parameter=path>\n/a\n</parameter>\n<parameter=n>\n5\n</parameter>\n</function>\n",
            ),
            (
                "<tool_call>",
                "</tool_call>",
                "write<arg_key>path</arg_key><arg_value>/a</arg_value><arg_key>n</arg_key><arg_value>5</arg_value>",
            ),
            (
                "<atem:function_calls>",
                "</atem:function_calls>",
                "\n<atem:invoke name=\"write\">\n<atem:parameter name=\"path\">/a</atem:parameter>\n<atem:parameter name=\"n\">5</atem:parameter>\n</atem:invoke>\n",
            ),
        ];
        for (opener, close, span) in calls {
            let parsed = split(&format!("{opener}{span}{close}"));
            let whole: Value = serde_json::from_str(&parsed.calls[0].arguments).unwrap();
            assert_eq!(call_name(opener, span).as_deref(), Some("write"), "{span}");
            for step in [1, 4, 1000] {
                let mut stream = ArgumentStream::new(opener);
                let mut out = String::new();
                let ends: Vec<usize> = span.char_indices().map(|(i, _)| i).collect();
                for &end in ends.iter().step_by(step) {
                    out.push_str(&stream.advance(&span[..end]));
                }
                let early = out.len();
                let (rest, agreed) = stream.finish(span, &parsed.calls[0].arguments);
                out.push_str(&rest);
                assert!(agreed, "{span} by {step}: {out}");
                assert!(step == 1000 || early > 0, "{span}: nothing sent early");
                let value: Value = serde_json::from_str(&out)
                    .unwrap_or_else(|err| panic!("{span} by {step}: {out}: {err}"));
                assert_eq!(value, whole, "{span} by {step}");
            }
        }
    }

    /// A JSON call whose arguments come before its name, or are a string,
    /// is not streamed.
    #[test]
    fn a_json_call_is_streamed_only_name_first_with_object_arguments() {
        assert!(json_call_head(r#"{"arguments": {"a": 1}, "name": "f"}"#).is_none());
        assert!(json_call_head(r#"{"name": "f", "arguments": "{\"a\": 1}"}"#).is_none());
        let (name, at) = json_call_head(r#" {"name": "f", "id": 3, "arguments": {"a"#).unwrap();
        assert_eq!((name.as_str(), at), ("f", 37));
    }

    /// A call cut off in a string, or after a key, closes into valid JSON;
    /// a call the model ended without its closer is closed when it reads
    /// whole.
    #[test]
    fn a_cut_off_json_call_closes_and_an_unclosed_whole_one_counts() {
        for cut in [
            r#"{"a": "half"#,
            r#"{"a": ["x", {"b": "#,
            r#"{"a": 1, "b": "esc\"#,
        ] {
            let closed = format!("{cut}{}", json_closers(cut));
            assert!(serde_json::from_str::<Value>(&closed).is_ok(), "{closed}");
        }
        let mut raw = ArgumentStream::new("<tool_call>");
        let mut sent = raw.advance("{\"name\": \"f\", \"arguments\": {\"a\": \"line\nnext");
        assert_eq!(sent, "{\"a\": \"line");
        sent.push_str(&raw.abandon());
        assert!(serde_json::from_str::<Value>(&sent).is_ok(), "{sent}");
        let whole = "<tool_call>\n{\"name\": \"f\", \"arguments\": {}}";
        assert_eq!(split(&close_trailing(whole)).calls.len(), 1);
        let cut = "<tool_call>\n{\"name\": \"f\", \"argu";
        assert_eq!(close_trailing(cut), cut);
        assert_eq!(close_trailing("plain text"), "plain text");
    }

    /// A call that never parsed is closed as it stood.
    #[test]
    fn an_abandoned_call_is_closed_as_it_stood() {
        let mut stream = ArgumentStream::default();
        let mut out = stream.advance("call:f{a:<|\"|>half");
        out.push_str(&stream.abandon());
        assert_eq!(out, "{\"a\":\"half\"}");
    }

    #[test]
    fn a_gemma_dsl_call_becomes_openai_shaped_arguments() {
        let split = split("<|tool_call>call:show_file{path:\"src/main.rs\",lines:40}<tool_call|>");
        assert_eq!(split.content, "");
        assert_eq!(split.calls.len(), 1);
        assert_eq!(split.calls[0].name, "show_file");
        let args: Value = serde_json::from_str(&split.calls[0].arguments).unwrap();
        assert_eq!(args["path"], "src/main.rs");
        assert_eq!(args["lines"], 40);
    }

    #[test]
    fn a_stray_quote_before_a_key_does_not_lose_the_call() {
        let q = DSL_QUOTE;
        let split = split(&format!(
            "<|tool_call>call:bash{{\"command:{q}ls /skills{q},\"n\":2}}<tool_call|>"
        ));
        assert_eq!(split.content, "");
        let args: Value = serde_json::from_str(&split.calls[0].arguments).unwrap();
        assert_eq!(args["command"], "ls /skills");
        assert_eq!(args["n"], 2);
    }

    #[test]
    fn prose_around_a_call_is_kept_as_content() {
        let split = split("Let me look.<|tool_call>call:ls{}<tool_call|>");
        assert_eq!(split.content, "Let me look.");
        assert_eq!(split.calls.len(), 1);
        assert_eq!(split.calls[0].arguments, "{}");
    }

    #[test]
    fn nested_objects_and_arrays_survive_the_dsl() {
        let split = split(
            "<|tool_call>call:edit{file:\"a.rs\",opts:{dry:true,tags:[\"x\",\"y\"]},n:null}<tool_call|>",
        );
        let args: Value = serde_json::from_str(&split.calls[0].arguments).unwrap();
        assert_eq!(args["opts"]["dry"], true);
        assert_eq!(args["opts"]["tags"][1], "y");
        assert_eq!(args["n"], Value::Null);
    }

    #[test]
    fn a_brace_inside_a_string_does_not_end_the_call() {
        let split = split("<|tool_call>call:run{cmd:\"echo {hi}\"}<tool_call|>");
        let args: Value = serde_json::from_str(&split.calls[0].arguments).unwrap();
        assert_eq!(args["cmd"], "echo {hi}");
    }

    /// The file a coding agent is actually asked to write: source, with
    /// quotes and braces in it, delimited by the vocabulary's own `<|"|>`.
    /// Flattened to a bare `"` this ended its string at `print("`, failed to
    /// parse, and reached the caller as prose with no file written.
    #[test]
    fn a_file_whose_content_contains_quotes_still_parses() {
        let content = "def main():\n    print(\"hi {there}\")\n";
        let span = format!(
            "<|tool_call>call:create_file{{content:{q}{content}{q},path:{q}main.py{q}}}<tool_call|>",
            q = DSL_QUOTE
        );
        let split = split(&span);
        assert_eq!(split.content, "");
        assert_eq!(split.calls.len(), 1, "call was not recovered: {split:?}");
        assert_eq!(split.calls[0].name, "create_file");
        let args: Value = serde_json::from_str(&split.calls[0].arguments).unwrap();
        assert_eq!(args["content"], content);
        assert_eq!(args["path"], "main.py");
    }

    /// Three files asked for, one written: after the first well-formed span
    /// gemma-4 stopped writing the opening marker, and the remaining two
    /// calls reached the caller as prose. The closer and `call:` anchor them.
    #[test]
    fn a_call_whose_opener_the_model_forgot_is_still_recovered() {
        let span = concat!(
            "Here you go.",
            "<|tool_call>call:create_file{path:\"index.html\",content:\"<html/>\"}<tool_call|>",
            "Now the stylesheet.",
            "call:create_file{path:\"style.css\",content:\"body{}\"}<tool_call|>",
            "call:create_file{path:\"game.js\",content:\"let x=1\"}<tool_call|>",
        );
        let split = split(span);
        assert_eq!(split.calls.len(), 3, "{split:?}");
        let paths: Vec<String> = split
            .calls
            .iter()
            .map(|call| {
                serde_json::from_str::<Value>(&call.arguments).unwrap()["path"]
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        assert_eq!(paths, ["index.html", "style.css", "game.js"]);
        // The prose between the calls survives; the call bodies do not.
        assert_eq!(split.content, "Here you go.Now the stylesheet.");
    }

    /// The recovery needs the format to have been established first, and it
    /// needs a body that parses. Neither a stray closer in prose nor an
    /// unopened body on its own is a call.
    #[test]
    fn an_unopened_body_is_not_a_call_on_its_own() {
        // No call has been recovered yet, so nothing is established.
        let first = "call:create_file{path:\"a.txt\"}<tool_call|>";
        assert!(split(first).calls.is_empty(), "{:?}", split(first));

        // Established, but the trailing text is prose rather than a body.
        let prose = concat!(
            "<|tool_call>call:create_file{path:\"a.txt\"}<tool_call|>",
            "and that closes it out<tool_call|>",
        );
        let split = split(prose);
        assert_eq!(split.calls.len(), 1, "{split:?}");
        assert!(
            split.content.contains("and that closes it out"),
            "{split:?}"
        );
    }

    /// A span that opened and never closed still stops the read dead: the
    /// recovery pass must not pick over the remains of a truncated answer and
    /// execute half a call.
    #[test]
    fn an_unterminated_span_is_not_salvaged_by_the_recovery_pass() {
        let truncated = concat!(
            "<|tool_call>call:create_file{path:\"a.txt\"}<tool_call|>",
            "<|tool_call>call:create_file{path:\"b.txt\",content:\"half a fi",
        );
        let split = split(truncated);
        assert_eq!(split.calls.len(), 1, "only the finished call: {split:?}");
        assert!(split.content.contains("half a fi"), "{split:?}");
    }

    /// The model does not always write the template's `call:` prefix. The
    /// span is anchored either way, so the bare form is read too — as a span
    /// missing it was a `create_file` that reached the caller as prose.
    #[test]
    fn a_gemma_span_without_the_call_prefix_is_still_a_call() {
        let split = split("<|tool_call>create_file{path:\"a.py\",content:\"x = 1\"}<tool_call|>");
        assert_eq!(split.calls.len(), 1, "call was not recovered: {split:?}");
        assert_eq!(split.calls[0].name, "create_file");
        let args: Value = serde_json::from_str(&split.calls[0].arguments).unwrap();
        assert_eq!(args["path"], "a.py");
    }

    /// …but only when what precedes the brace actually looks like a name.
    /// Prose that happens to contain a brace is left in the answer.
    #[test]
    fn prose_with_a_brace_in_a_gemma_span_is_not_a_call() {
        let span = "<|tool_call>here is a set {a, b}<tool_call|>";
        let split = split(span);
        assert!(split.calls.is_empty(), "{split:?}");
        assert_eq!(split.content, span);
    }

    /// A `<|"|>` value is verbatim: the template escapes nothing writing it,
    /// so unescaping it here turned `print("\\n")` in the file the model asked
    /// for into a real newline and a syntax error.
    #[test]
    fn a_backslash_in_a_quote_delimited_value_stays_a_backslash() {
        let content = "print(\"\\n\")\nre.match(r\"\\d+\", s)";
        let span = format!(
            "<|tool_call>call:create_file{{content:{q}{content}{q}}}<tool_call|>",
            q = DSL_QUOTE
        );
        let split = split(&span);
        assert_eq!(split.calls.len(), 1, "call was not recovered: {split:?}");
        let args: Value = serde_json::from_str(&split.calls[0].arguments).unwrap();
        assert_eq!(args["content"], content);
    }

    /// The same delimiter around a *nested* value: the object scan has to skip
    /// the string whole, or a brace inside the source closes the call early.
    #[test]
    fn a_quote_delimited_string_inside_a_nested_object_is_skipped_whole() {
        let span = format!(
            "<|tool_call>call:write{{edits:{{body:{q}if (x) {{ y(\"z\"); }}{q}}},n:1}}<tool_call|>",
            q = DSL_QUOTE
        );
        let split = split(&span);
        assert_eq!(split.calls.len(), 1, "call was not recovered: {split:?}");
        let args: Value = serde_json::from_str(&split.calls[0].arguments).unwrap();
        assert_eq!(args["edits"]["body"], "if (x) { y(\"z\"); }");
        assert_eq!(args["n"], 1);
    }

    /// The vocabulary's delimiter token reaches [`split`] as its own literal
    /// spelling, not as `"` — which is what keeps the two distinguishable.
    #[test]
    fn the_dsl_quote_token_renders_as_itself() {
        assert_eq!(marker_text("<|\"|>"), Some(DSL_QUOTE));
    }

    #[test]
    fn the_qwen_hermes_json_form_is_understood() {
        let split =
            split("<tool_call>{\"name\": \"ls\", \"arguments\": {\"path\": \".\"}}</tool_call>");
        assert_eq!(split.calls.len(), 1);
        assert_eq!(split.calls[0].name, "ls");
        assert_eq!(
            serde_json::from_str::<Value>(&split.calls[0].arguments).unwrap()["path"],
            "."
        );
    }

    #[test]
    fn the_mistral_array_form_is_understood() {
        let split = split("[TOOL_CALLS][{\"name\": \"a\", \"arguments\": {}}, {\"name\": \"b\"}]");
        assert_eq!(split.calls.len(), 2);
        assert_eq!(split.calls[1].name, "b");
        assert_eq!(split.calls[1].arguments, "{}");
    }

    #[test]
    fn two_calls_in_one_turn_are_both_recovered() {
        let split =
            split("<|tool_call>call:a{x:1}<tool_call|><|tool_call>call:b{y:\"two\"}<tool_call|>");
        assert_eq!(split.calls.len(), 2);
        assert_eq!(split.calls[0].name, "a");
        assert_eq!(split.calls[1].name, "b");
    }

    /// The failure mode that matters most: an answer that merely *contains*
    /// tool-call-looking JSON must not be executed. Only a delimiter counts.
    #[test]
    fn json_in_an_ordinary_answer_is_never_mistaken_for_a_call() {
        let text = "You can post {\"name\": \"delete_everything\", \"arguments\": {}} to that API.";
        let split = split(text);
        assert!(split.calls.is_empty());
        assert_eq!(split.content, text);
    }

    /// A call the model never finished is not a call, and its text must not
    /// vanish from the answer either.
    #[test]
    fn an_unterminated_span_stays_as_content() {
        let text = "hmm <|tool_call>call:ls{";
        let split = split(text);
        assert!(split.calls.is_empty());
        assert_eq!(split.content, text);
    }

    #[test]
    fn a_delimited_span_we_cannot_parse_is_left_visible() {
        let text = "<|tool_call>total nonsense<tool_call|>";
        let split = split(text);
        assert!(split.calls.is_empty());
        assert_eq!(split.content, text);
    }

    /// Drive `split` the way the streaming endpoint does — one token at a
    /// time, holding back only what might still become a delimiter — and
    /// require the pieces to reassemble into exactly what the model wrote.
    ///
    /// Every other test here passes a whole string, which is why a `trim()`
    /// inside `split` looked harmless and shipped: per-token it deleted the
    /// leading space of every word, and the client rendered
    /// `Ihavelistedthefiles…`.
    fn stream(tokens: &[&str]) -> (String, Vec<ParsedToolCall>) {
        let mut seen = String::new();
        let mut calls = Vec::new();
        let mut pending = String::new();
        for token in tokens {
            pending.push_str(token);
            if may_be_partial(&pending) {
                continue;
            }
            let split = split(&pending);
            pending.clear();
            seen.push_str(&split.content);
            calls.extend(split.calls);
        }
        let tail = split(&pending);
        seen.push_str(&tail.content);
        calls.extend(tail.calls);
        (seen, calls)
    }

    #[test]
    fn streaming_token_by_token_preserves_every_space() {
        let tokens = ["Hello", " there", ",", " how", " are", " you", "?"];
        let (seen, calls) = stream(&tokens);
        assert_eq!(seen, "Hello there, how are you?");
        assert!(calls.is_empty());
    }

    #[test]
    fn streaming_reassembles_prose_around_a_call() {
        let tokens = [
            "Let",
            " me",
            " look",
            ".",
            "<|tool_call>",
            "call:ls{",
            "path:",
            "\".\"",
            "}",
            "<tool_call|>",
            " Done",
            ".",
        ];
        let (seen, calls) = stream(&tokens);
        assert_eq!(seen, "Let me look. Done.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "ls");
    }

    /// Newlines are content too — a code block streamed a token at a time
    /// must not come back as one line.
    #[test]
    fn streaming_preserves_newlines_and_indentation() {
        let tokens = ["fn main() {", "\n", "    ", "println!();", "\n", "}"];
        let (seen, _) = stream(&tokens);
        assert_eq!(seen, "fn main() {\n    println!();\n}");
    }

    #[test]
    fn a_half_arrived_opener_is_held_back() {
        assert!(may_be_partial("thinking… <|tool_ca"));
        assert!(may_be_partial("<|tool_call>call:ls{"));
        assert!(!may_be_partial("an ordinary answer"));
        assert!(!may_be_partial("<|tool_call>call:ls{}<tool_call|>"));
    }

    /// GLM writes the name bare and then `<arg_key>`/`<arg_value>` pairs,
    /// inside the *same* `<tool_call>` delimiters Hermes uses for JSON.
    ///
    /// Before the body was discriminated, this span was found, failed to
    /// parse as JSON, and was left in the answer as raw markup — a tool call
    /// the user saw and nothing executed.
    #[test]
    fn a_key_value_body_in_a_tool_call_span_is_read() {
        let split = split(
            "sure<tool_call>get_weather<arg_key>city</arg_key><arg_value>\"Paris\"</arg_value>\
             <arg_key>days</arg_key><arg_value>3</arg_value></tool_call>",
        );
        assert_eq!(split.content, "sure");
        assert_eq!(split.calls.len(), 1);
        assert_eq!(split.calls[0].name, "get_weather");
        let args: Value = serde_json::from_str(&split.calls[0].arguments).unwrap();
        assert_eq!(args["city"], "Paris");
        // A bare number stays a number rather than becoming the string "3".
        assert_eq!(args["days"], 3);
    }

    /// Nemotron nests `<function=…>` and `<parameter=…>` inside the same
    /// delimiters again — a third body behind one opener.
    #[test]
    fn a_function_parameter_body_in_a_tool_call_span_is_read() {
        let split = split(
            "<tool_call>\n<function=list_files>\n<parameter=path>\n/tmp\n</parameter>\n\
             <parameter=recursive>\ntrue\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(split.calls.len(), 1);
        assert_eq!(split.calls[0].name, "list_files");
        let args: Value = serde_json::from_str(&split.calls[0].arguments).unwrap();
        assert_eq!(args["path"], "/tmp");
        assert_eq!(args["recursive"], true);
    }

    /// MiniMax-M3's body: each argument an element named for its key, a
    /// mapping expanded into elements and a list into `<item>`s — nested
    /// `<item>` in `<item>` included. The namespace token before every tag
    /// is special and arrives already removed.
    #[test]
    fn an_invoke_element_body_in_a_tool_call_span_is_read() {
        let split = split(
            "Checking.<tool_call>\n<invoke name=\"edit\"><path>src/a.rs</path>\
             <count>3</count><dry>true</dry><opts><mode>fast</mode><tags><item>x</item>\
             <item>y</item></tags></opts><grid><item><item>1</item><item>2</item></item>\
             </grid></invoke>\n<invoke name=\"ls\"><dir>/tmp</dir></invoke>\n</tool_call>",
        );
        assert_eq!(split.content, "Checking.");
        assert_eq!(split.calls.len(), 2);
        assert_eq!(split.calls[0].name, "edit");
        let args: Value = serde_json::from_str(&split.calls[0].arguments).unwrap();
        assert_eq!(
            args,
            serde_json::json!({
                "path": "src/a.rs",
                "count": 3,
                "dry": true,
                "opts": {"mode": "fast", "tags": ["x", "y"]},
                "grid": [[1, 2]],
            })
        );
        assert_eq!(split.calls[1].name, "ls");
        assert_eq!(
            call_name("<tool_call>", "\n<invoke name=\"edit\"><pa").as_deref(),
            Some("edit")
        );
    }

    /// A leaf that is prose containing angle brackets stays a string, and a
    /// call left unclosed is not a call.
    #[test]
    fn an_invoke_element_leaf_with_markup_is_text_and_an_open_invoke_is_prose() {
        let parsed = split(
            "<tool_call><invoke name=\"write\"><body>if a < b { x }</body></invoke></tool_call>",
        );
        let args: Value = serde_json::from_str(&parsed.calls[0].arguments).unwrap();
        assert_eq!(args["body"], "if a < b { x }");

        let open = split("<tool_call><invoke name=\"write\"><body>x</body></tool_call>");
        assert!(open.calls.is_empty());
    }

    /// Two calls in one span must not pool their parameters.
    #[test]
    fn two_functions_in_one_span_keep_their_own_parameters() {
        let split = split(
            "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n\
             <function=b>\n<parameter=y>\n2\n</parameter>\n</function>\n</tool_call>",
        );
        assert_eq!(split.calls.len(), 2);
        let first: Value = serde_json::from_str(&split.calls[0].arguments).unwrap();
        let second: Value = serde_json::from_str(&split.calls[1].arguments).unwrap();
        assert_eq!(first, serde_json::json!({"x": 1}));
        assert_eq!(second, serde_json::json!({"y": 2}));
    }

    /// The `<PREFIXinvoke>` shape, in both spellings that use it.
    #[test]
    fn an_invoke_tag_block_is_read_in_either_namespace() {
        let atem = split(
            "<atem:function_calls>\n<atem:invoke name=\"read\">\n\
             <atem:parameter name=\"path\">/etc/hosts</atem:parameter>\n\
             </atem:invoke>\n</atem:function_calls>",
        );
        assert_eq!(atem.calls.len(), 1);
        assert_eq!(atem.calls[0].name, "read");
        let args: Value = serde_json::from_str(&atem.calls[0].arguments).unwrap();
        assert_eq!(args["path"], "/etc/hosts");

        let dsml = split(
            "<\u{ff5c}DSML\u{ff5c}tool_calls>\n\
             <\u{ff5c}DSML\u{ff5c}invoke name=\"read\">\n\
             <\u{ff5c}DSML\u{ff5c}parameter name=\"path\" string=\"true\">/etc/hosts\
             </\u{ff5c}DSML\u{ff5c}parameter>\n\
             </\u{ff5c}DSML\u{ff5c}invoke>\n</\u{ff5c}DSML\u{ff5c}tool_calls>",
        );
        assert_eq!(dsml.calls.len(), 1);
        assert_eq!(dsml.calls[0].name, "read");
        let args: Value = serde_json::from_str(&dsml.calls[0].arguments).unwrap();
        assert_eq!(args["path"], "/etc/hosts");
    }

    /// `string="true"` is load-bearing: a parameter declared a string stays
    /// one even when it would parse as a number. Ignoring the attribute turns
    /// a version like `1.20` into the float `1.2`.
    #[test]
    fn a_parameter_declared_a_string_is_not_reinterpreted_as_a_number() {
        let call = &split(
            "<\u{ff5c}DSML\u{ff5c}tool_calls>\
             <\u{ff5c}DSML\u{ff5c}invoke name=\"pin\">\
             <\u{ff5c}DSML\u{ff5c}parameter name=\"version\" string=\"true\">1.20\
             </\u{ff5c}DSML\u{ff5c}parameter>\
             <\u{ff5c}DSML\u{ff5c}parameter name=\"count\" string=\"false\">7\
             </\u{ff5c}DSML\u{ff5c}parameter>\
             </\u{ff5c}DSML\u{ff5c}invoke></\u{ff5c}DSML\u{ff5c}tool_calls>",
        )
        .calls[0];
        let args: Value = serde_json::from_str(&call.arguments).unwrap();
        assert_eq!(args["version"], "1.20");
        assert_eq!(args["count"], 7);
    }

    /// **The posture, restated as a test.** Nothing outside a delimited span
    /// is ever a call, however much it looks like one — a model asked to
    /// *write* a tool call must be able to.
    #[test]
    fn nothing_outside_a_delimited_span_is_ever_a_call() {
        for text in [
            r#"{"name": "rm", "arguments": {"path": "/"}}"#,
            r#"Here is an example: {"name": "rm", "parameters": {}}"#,
            "get_weather<arg_key>city</arg_key><arg_value>\"Paris\"</arg_value>",
            "<function=rm>\n<parameter=path>\n/\n</parameter>\n</function>",
            "<atem:invoke name=\"rm\"><atem:parameter name=\"p\">/</atem:parameter></atem:invoke>",
        ] {
            let split = split(text);
            assert!(
                split.calls.is_empty(),
                "unanchored text became a call: {text}"
            );
            assert_eq!(split.content, text, "and it must survive in the answer");
        }
    }

    /// A body matching none of the four shapes is prose, and stays in the
    /// answer rather than vanishing.
    #[test]
    fn an_unreadable_tool_call_body_is_left_in_the_answer() {
        let text = "<tool_call>something we do not understand</tool_call>";
        let split = split(text);
        assert!(split.calls.is_empty());
        assert_eq!(split.content, text);
    }
}
