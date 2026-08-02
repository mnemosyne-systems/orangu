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
    // Qwen2.5/3, Hermes, and everything that copied their template.
    ("<tool_call>", "</tool_call>", Syntax::Json),
    // Mistral: a JSON *array* with no closing marker — it runs to the end.
    ("[TOOL_CALLS]", "", Syntax::JsonArray),
]
.as_slice();

#[derive(Debug, Clone, Copy, PartialEq)]
enum Syntax {
    /// `call:NAME{key:value,...}` — see [`parse_gemma_dsl`].
    GemmaDsl,
    /// `{"name": "...", "arguments": {...}}`, one object per span.
    Json,
    /// `[{"name": "...", "arguments": {...}}, ...]` to end of text.
    JsonArray,
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
            Syntax::Json => parse_json_object(body).into_iter().collect(),
            Syntax::JsonArray => parse_json_array(body),
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
        // An opener that has started but not finished arriving...
        (1..open.len()).any(|n| text.ends_with(&open[..n]))
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
}
