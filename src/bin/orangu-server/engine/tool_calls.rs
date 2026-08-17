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
//! `<tool_call>` is not one format. Three supported architectures write three
//! different things inside that same span — JSON, `<arg_key>`/`<arg_value>`
//! pairs, and a `<function=…>`/`<parameter=…>` nest — so the *body* is what
//! distinguishes them, discriminated by its own leading structure once the
//! span is already anchored. That is not a relaxation of the rule above: no
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

#[derive(Debug, Clone, Copy, PartialEq)]
enum Syntax {
    /// `call:NAME{key:value,...}` — see [`parse_gemma_dsl`].
    GemmaDsl,
    /// A `<tool_call>` span, whose body may be any of three shapes — see
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
                None => break,
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
            break;
        }
        content.push_str(&rest[..at]);
        calls.extend(parsed);
        rest = after;
    }

    content.push_str(rest);
    Split { content, calls }
}

fn next_opener(text: &str) -> Option<(usize, &'static str, &'static str, Syntax)> {
    FORMATS
        .iter()
        .filter_map(|(open, close, syntax)| text.find(open).map(|at| (at, *open, *close, *syntax)))
        .min_by_key(|(at, open, _, _)| (*at, std::cmp::Reverse(open.len())))
}

/// gemma-4's `call:NAME{key:value,...}`.
///
/// The value grammar is the one the template's own `format_argument` macro
/// writes: strings quoted (the `<|"|>` token having been rendered back to `"`
/// upstream), `true`/`false`/`null` bare, numbers bare, nested objects with
/// **unquoted** keys, and arrays. That last part is why this cannot simply be
/// handed to `serde_json`.
fn parse_gemma_dsl(body: &str) -> Vec<ParsedToolCall> {
    let body = body.trim();
    let Some(rest) = body.strip_prefix("call:") else {
        return Vec::new();
    };
    let Some(brace) = rest.find('{') else {
        return Vec::new();
    };
    let name = rest[..brace].trim();
    if name.is_empty() {
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

/// A `<tool_call>` span's body, whichever of the three shapes it is.
///
/// Discriminated by leading structure, and each test is unambiguous: JSON
/// opens with a brace or a bracket, the key/value shape has an `<arg_key>`
/// element, and the nested shape has a `<function=` element. A body matching
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
    Vec::new()
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
    fn key(&mut self) -> Option<String> {
        if self.peek() == Some('"') {
            return self.string();
        }
        let start = self.i;
        while matches!(self.peek(), Some(c) if c != ':' && c != ',' && !c.is_whitespace()) {
            self.bump();
        }
        (self.i > start).then(|| self.s[start..self.i].to_string())
    }

    fn value(&mut self) -> Option<Value> {
        match self.peek()? {
            '"' => self.string().map(Value::String),
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
        let mut depth = 1usize;
        let mut in_string = false;
        let mut escaped = false;
        for (offset, c) in self.s[from..].char_indices() {
            if in_string {
                if escaped {
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                } else if c == '"' {
                    in_string = false;
                }
                continue;
            }
            match c {
                '"' => in_string = true,
                _ if c == open => depth += 1,
                _ if c == close => {
                    depth -= 1;
                    if depth == 0 {
                        return Some(from + offset);
                    }
                }
                _ => {}
            }
        }
        None
    }

    fn string(&mut self) -> Option<String> {
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

/// The literal text a special token should be rendered as when it carries
/// tool-call structure, or `None` for every other special token (which stays
/// suppressed, as all of them used to be).
///
/// `<|"|>` is gemma's string delimiter *inside* a call's arguments; mapping it
/// to `"` is what lets [`split`] tell a string value from a bare one. It has
/// no meaning outside a call and a model does not emit it elsewhere.
pub fn marker_text(token: &str) -> Option<&'static str> {
    match token {
        "<|tool_call>" => Some("<|tool_call>"),
        "<tool_call|>" => Some("<tool_call|>"),
        "<|\"|>" => Some("\""),
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
    use super::*;

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

    /// A body matching none of the three shapes is prose, and stays in the
    /// answer rather than vanishing.
    #[test]
    fn an_unreadable_tool_call_body_is_left_in_the_answer() {
        let text = "<tool_call>something we do not understand</tool_call>";
        let split = split(text);
        assert!(split.calls.is_empty());
        assert_eq!(split.content, text);
    }
}
