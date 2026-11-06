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

//! Reading tool calls in a syntax [`super::tool_calls`] has no parser for, by
//! asking the model's own chat template how it writes one.
//!
//! A template's branch that renders `tool_calls` into the prompt is by
//! construction the syntax the model was trained to emit, and it can be run.
//! Rendering a known call and looking at where the known name, keys and values
//! land gives the delimiters and the shape of the body: a JSON object naming
//! the function, a name followed by JSON arguments, or a name followed by
//! key/value tags.
//!
//! What this learns is the fallback. A template whose call is already readable
//! learns nothing, so a hand-written parser always wins. The rule of
//! [`super::tool_calls`] holds here too: a syntax with no opening delimiter is
//! not learned, because guessing that bare JSON is a call is how a model asked
//! to write some JSON ends up having it executed.
//!
//! Learned calls leave [`LearnedReader`] rewritten as `<tool_call>` JSON, so
//! everything downstream has one form to read.

use super::chat_template::{ChatMessage, ChatTemplate};
use super::tool_calls::{self, ParsedToolCall};
use serde_json::{Value, json};
use std::sync::Arc;

const NAME: &str = "orangu_probe_tool";
const KEY_A: &str = "orangu_probe_a";
const VALUE_A: &str = "orangu_value_a";
const KEY_B: &str = "orangu_probe_b";
const VALUE_B: &str = "orangu_value_b";

/// How a body names its function and carries its arguments.
#[derive(Clone, Debug, PartialEq)]
enum Shape {
    /// `{"<name_field>": NAME, "<args_field>": {…}}`.
    JsonObject {
        name_field: String,
        args_field: String,
    },
    /// `NAME<gap>{…}`.
    NameThenJson { gap: String },
    /// `NAME<name_end>` then `<key_open>KEY<key_end>VALUE<value_end>` per
    /// argument.
    Pairs {
        name_end: String,
        key_open: String,
        key_end: String,
        value_end: String,
    },
}

/// A call syntax read off a chat template.
#[derive(Clone, Debug, PartialEq)]
pub struct LearnedCalls {
    open: String,
    close: String,
    /// What stands between two calls in one block, when calls share their
    /// delimiters rather than each having its own.
    separator: Option<String>,
    shape: Shape,
    /// The rendered probe call, whose special tokens a model has to be allowed
    /// to write visibly.
    rendered: String,
}

/// The span of `text` one JSON value starting at `start` occupies, and that
/// value.
fn json_value_at(text: &str, start: usize) -> Option<(Value, usize)> {
    let mut values = serde_json::Deserializer::from_str(&text[start..]).into_iter::<Value>();
    let value = values.next()?.ok()?;
    Some((value, start + values.byte_offset()))
}

fn common_suffix_len(a: &str, b: &str) -> usize {
    a.chars()
        .rev()
        .zip(b.chars().rev())
        .take_while(|(x, y)| x == y)
        .map(|(x, _)| x.len_utf8())
        .sum()
}

/// One call the probe makes, for a template to render.
fn probe_call() -> Value {
    json!({"id": "probe", "type": "function", "function": {
        "name": NAME,
        "arguments": json!({KEY_A: VALUE_A, KEY_B: VALUE_B}).to_string(),
    }})
}

/// Renders probe turns through a template, and takes off what comes before a
/// reply: what comes before says nothing about the call, and can share text
/// with what ends the turn.
struct CallProbe<'a> {
    template: &'a ChatTemplate,
    tools: Value,
    user: ChatMessage,
    head: String,
    plain: String,
}

impl<'a> CallProbe<'a> {
    fn new(template: &'a ChatTemplate) -> Option<Self> {
        let tools = json!([{"type": "function", "function": {
            "name": NAME,
            "parameters": {"type": "object", "properties": {
                KEY_A: {"type": "string"}, KEY_B: {"type": "string"},
            }},
        }}]);
        let user = ChatMessage::text("user", "probe");
        let head = template
            .render_with_tools(
                std::slice::from_ref(&user),
                true,
                "",
                "",
                Default::default(),
                Some(&tools),
            )
            .ok()?;
        let mut probe = Self {
            template,
            tools,
            user,
            head,
            plain: String::new(),
        };
        probe.plain = probe.reply(probe.render(None)?);
        Some(probe)
    }

    fn render(&self, calls: Option<Value>) -> Option<String> {
        let mut turn = ChatMessage::text("assistant", "");
        turn.tool_calls = calls;
        self.template
            .render_with_tools(
                &[self.user.clone(), turn],
                false,
                "",
                "",
                Default::default(),
                Some(&self.tools),
            )
            .ok()
    }

    fn reply(&self, rendered: String) -> String {
        match rendered.strip_prefix(self.head.as_str()) {
            Some(reply) => reply.to_string(),
            None => rendered,
        }
    }
}

/// Where a reply already stands when a call's grammar takes over.
#[derive(Clone, Debug, PartialEq)]
pub enum CallStart<'a> {
    /// Nowhere: the model chooses whether to call.
    Free,
    /// Forced up to where any tool's name goes.
    AtName,
    /// Forced through this tool's name.
    AfterName(&'a str),
}

/// What holds a tool call to the tools a request declared: once `trigger` has
/// been written, `after_trigger` says what may follow it, each tool's name
/// with arguments its schema describes.
#[derive(Debug)]
pub struct CallGrammar {
    pub trigger: String,
    pub after_trigger: Arc<crate::engine::grammar::Grammar>,
    /// What a reply forced to begin with a call is held to from its start.
    pub initial: Option<Arc<crate::engine::grammar::Grammar>>,
}

impl CallGrammar {
    /// The grammar for `tools`, in the request's shape, on the call syntax
    /// `template` renders. `None` where its arguments are not JSON, as a
    /// grammar over them is not something this can write. A tool whose schema
    /// cannot be followed takes any object as arguments.
    pub fn build(template: &ChatTemplate, tools: &Value, start: CallStart<'_>) -> Option<Self> {
        use crate::engine::schema_text::quoted;
        let probe = CallProbe::new(template)?;
        let one = probe.reply(probe.render(Some(json!([probe_call()])))?);
        let region = call_text(&one, &probe.plain)?.trim();
        if region.is_empty() || region.len() > MAX_CALL_BYTES {
            return None;
        }
        let (open, close, shape) = read_shape(region)?;
        let trigger = open.trim().to_string();
        if trigger.is_empty() {
            return None;
        }
        let closing = match close.trim() {
            "" => "\"\"".to_string(),
            close => quoted(close),
        };
        let json_key = |key: &str| quoted(&Value::String(key.to_string()).to_string());
        // Per tool: the rule for what follows its name, and the whole call.
        let mut rules = crate::engine::json_schema::common_rules();
        let mut whole = Vec::new();
        let mut at_name = Vec::new();
        let mut after_named = None;
        let declared = tools.as_array()?;
        for (index, tool) in declared.iter().enumerate() {
            let function = tool.get("function")?;
            let name = function.get("name")?.as_str()?;
            let prefix = format!("t{index}");
            let parameters = function.get("parameters").unwrap_or(&Value::Null);
            let args = match crate::engine::json_schema::schema_rules(parameters, &prefix) {
                Ok((schema_rules, root)) => {
                    rules.push_str(&schema_rules);
                    root
                }
                Err(_) => "anyobject".to_string(),
            };
            let json_name = Value::String(name.to_string()).to_string();
            let bare_name = &json_name[1..json_name.len() - 1];
            let after = format!("{prefix}after");
            let (call, after_body) = match &shape {
                Shape::JsonObject {
                    name_field,
                    args_field,
                } => {
                    let name_key = json_key(name_field);
                    let args_key = json_key(args_field);
                    if region.find(name_field.as_str()) < region.find(args_field.as_str()) {
                        (
                            format!(
                                "\"{{\" ws {name_key} ws \":\" ws \"\\\"\" {} {after}",
                                quoted(bare_name)
                            ),
                            format!(
                                "\"\\\"\" ws \",\" ws {args_key} ws \":\" ws {args} ws \"}}\" ws {closing}"
                            ),
                        )
                    } else {
                        (
                            format!(
                                "\"{{\" ws {args_key} ws \":\" ws {args} ws \",\" ws {name_key} ws \":\" ws {} ws \"}}\" ws {closing}",
                                quoted(&json_name)
                            ),
                            String::new(),
                        )
                    }
                }
                Shape::NameThenJson { gap } => (
                    format!("{} {after}", quoted(name)),
                    format!("{} ws {args} ws {closing}", quoted(gap)),
                ),
                Shape::Pairs { .. } => return None,
            };
            if !after_body.is_empty() {
                rules.push_str(&format!("{after} ::= {after_body}\n"));
                at_name.push(format!("{} {after}", quoted(bare_name)));
                if start == CallStart::AfterName(name) {
                    after_named = Some(after.clone());
                }
            }
            let rule = format!("{prefix}call");
            rules.push_str(&format!("{rule} ::= {call}\n"));
            whole.push(rule);
        }
        if whole.is_empty() {
            return None;
        }
        let parse = |root: &str| {
            crate::engine::grammar::Grammar::parse(&format!("root ::= {root}\n{rules}")).ok()
        };
        let any_call = whole.join(" | ");
        let after_trigger = parse(&format!("ws ({any_call})"))?;
        let initial = match start {
            CallStart::Free => None,
            CallStart::AtName if at_name.len() == whole.len() => parse(&at_name.join(" | ")),
            // The arguments come first, so the reply is forced only that far.
            CallStart::AtName => parse(&format!("ws ({any_call})")),
            CallStart::AfterName(_) => after_named.as_deref().and_then(parse),
        };
        Some(Self {
            trigger,
            after_trigger,
            initial,
        })
    }
}

/// The most text one rendered call may be: a real one is a few dozen bytes, and
/// reading its shape looks at every brace in it.
const MAX_CALL_BYTES: usize = 16 * 1024;

/// What a call adds to a turn: `with` is the turn rendered with the call and
/// `without` the same turn without it. What surrounds a turn's text can be on
/// both sides of the call, and the shortest head that fits is the right split:
/// a turn's closing `</s>` and a call's opening `<call>` start alike.
///
/// A head can be as long as what the two share at the start, and a tail as long
/// as what they share at the end, so the shortest head is what the tail leaves.
fn call_text<'a>(with: &'a str, without: &str) -> Option<&'a str> {
    if with.len() < without.len() {
        return None;
    }
    let shared = |a: &mut dyn Iterator<Item = (u8, u8)>| a.take_while(|(x, y)| x == y).count();
    let start = shared(&mut with.bytes().zip(without.bytes()));
    let end = shared(&mut with.bytes().rev().zip(without.bytes().rev()));
    let mut head = without.len().saturating_sub(end);
    while !without.is_char_boundary(head) {
        head += 1;
    }
    (head <= start).then(|| &with[head..with.len() - (without.len() - head)])
}

/// Where the probe's arguments are in `text`, as a JSON object starting at or
/// after `from`: the object's bounds.
fn probe_arguments(text: &str, from: usize) -> Option<(usize, usize)> {
    text[from..].match_indices('{').find_map(|(at, _)| {
        let start = from + at;
        let (value, end) = json_value_at(text, start)?;
        (value.get(KEY_A)?.as_str()? == VALUE_A).then_some((start, end))
    })
}

fn read_shape(region: &str) -> Option<(String, String, Shape)> {
    let name_at = region.find(NAME)?;
    // The whole call as one object.
    for (start, _) in region[..name_at].match_indices('{') {
        let Some((Value::Object(object), end)) = json_value_at(region, start) else {
            continue;
        };
        let name_field = object.iter().find(|(_, v)| v.as_str() == Some(NAME))?.0;
        let args_field = object.iter().find(|(_, v)| v.get(KEY_A).is_some())?.0;
        return Some((
            region[..start].to_string(),
            region[end..].to_string(),
            Shape::JsonObject {
                name_field: name_field.clone(),
                args_field: args_field.clone(),
            },
        ));
    }
    let open = region[..name_at].to_string();
    let after_name = name_at + NAME.len();
    if let Some((start, end)) = probe_arguments(region, after_name) {
        return Some((
            open,
            region[end..].to_string(),
            Shape::NameThenJson {
                gap: region[after_name..start].to_string(),
            },
        ));
    }
    // Tags: `gap` and `between` both end in what opens a key.
    let key_a = after_name + region[after_name..].find(KEY_A)?;
    let value_a = key_a + region[key_a..].find(VALUE_A)?;
    let key_b = value_a + region[value_a..].find(KEY_B)?;
    let value_b = key_b + region[key_b..].find(VALUE_B)?;
    let gap = &region[after_name..key_a];
    let key_end = &region[key_a + KEY_A.len()..value_a];
    let between = &region[value_a + VALUE_A.len()..key_b];
    if key_end.is_empty() || key_end != &region[key_b + KEY_B.len()..value_b] {
        return None;
    }
    let key_open_len = common_suffix_len(gap, between);
    let (name_end, key_open) = gap.split_at(gap.len() - key_open_len);
    let value_end = &between[..between.len() - key_open_len];
    if value_end.is_empty() && key_open.is_empty() {
        return None;
    }
    let close = region[value_b + VALUE_B.len()..].strip_prefix(value_end)?;
    Some((
        open,
        close.to_string(),
        Shape::Pairs {
            name_end: name_end.to_string(),
            key_open: key_open.to_string(),
            key_end: key_end.to_string(),
            value_end: value_end.to_string(),
        },
    ))
}

impl LearnedCalls {
    /// Learns `template`'s call syntax. `None` when its call is one
    /// [`tool_calls`] reads already, when it has none, or when the syntax has
    /// no opening delimiter to anchor on.
    pub fn learn(template: &ChatTemplate) -> Option<Self> {
        let probe = CallProbe::new(template)?;
        let one = probe.reply(probe.render(Some(json!([probe_call()])))?);
        let region = call_text(&one, &probe.plain)?.trim();
        if region.is_empty()
            || region.len() > MAX_CALL_BYTES
            || tool_calls::first_opener(region).is_some()
        {
            return None;
        }
        let call = probe_call;
        let (render, reply, plain) = (
            |calls: Option<Value>| probe.render(calls),
            |rendered: String| probe.reply(rendered),
            probe.plain.clone(),
        );
        let (open, close, shape) = read_shape(region)?;
        let open = open.trim_start().to_string();
        if open.trim().is_empty() {
            return None;
        }
        let close = close.trim_end().to_string();
        let separator = render(Some(json!([call(), call()])))
            .map(reply)
            .and_then(|two| {
                let region = call_text(&two, &plain)?.trim();
                let body = region.strip_prefix(open.as_str())?;
                let body = if close.is_empty() {
                    body
                } else {
                    body.strip_suffix(close.as_str())?
                };
                separator_of(body, &shape)
            })
            .filter(|between| between.trim() != format!("{close}{open}").trim());
        Some(Self {
            open,
            close,
            separator,
            shape,
            rendered: region.to_string(),
        })
    }

    /// What opens a call in this syntax.
    pub fn opener(&self) -> &str {
        &self.open
    }

    /// What closes a call in this syntax; empty when it runs to the end.
    pub fn closer(&self) -> &str {
        &self.close
    }

    /// The text a special token has to occur in for a model to be allowed to
    /// write it visibly — `GenerateRequest::call_markers`.
    pub fn markers(&self) -> Vec<String> {
        vec![self.rendered.clone()]
    }

    fn parse_one(&self, body: &str) -> Option<ParsedToolCall> {
        let body = body.trim();
        match &self.shape {
            Shape::JsonObject {
                name_field,
                args_field,
            } => {
                let value: Value = serde_json::from_str(body).ok()?;
                let arguments = match value.get(args_field)? {
                    Value::String(text) => text.clone(),
                    other => other.to_string(),
                };
                Some(ParsedToolCall {
                    name: value.get(name_field)?.as_str()?.to_string(),
                    arguments,
                })
            }
            Shape::NameThenJson { gap } => {
                let (name, rest) = if gap.trim().is_empty() {
                    let at = body.find('{')?;
                    (body[..at].trim(), &body[at..])
                } else {
                    body.split_once(gap.as_str())?
                };
                let (arguments, end) = json_value_at(rest.trim_start(), 0)?;
                (arguments.is_object()
                    && !name.is_empty()
                    && rest.trim_start()[end..].trim().is_empty())
                .then(|| ParsedToolCall {
                    name: name.trim().to_string(),
                    arguments: arguments.to_string(),
                })
            }
            Shape::Pairs {
                name_end,
                key_open,
                key_end,
                value_end,
            } => {
                let first = if name_end.is_empty() {
                    key_open.as_str()
                } else {
                    name_end.as_str()
                };
                let (name, mut rest) = match body.split_once(first) {
                    Some((name, _)) if name_end.is_empty() => (name, &body[name.len()..]),
                    Some(split) => split,
                    None => (body, ""),
                };
                let mut arguments = serde_json::Map::new();
                while !rest.trim().is_empty() {
                    let after_open = rest.trim_start().strip_prefix(key_open.as_str())?;
                    let (key, after_key) = after_open.split_once(key_end.as_str())?;
                    let (value, after_value) = if value_end.is_empty() {
                        match after_key.find(key_open.as_str()) {
                            Some(at) => after_key.split_at(at),
                            None => (after_key, ""),
                        }
                    } else {
                        after_key.split_once(value_end.as_str())?
                    };
                    arguments.insert(key.trim().to_string(), tool_calls::scalar_or_json(value));
                    rest = after_value;
                }
                (!name.trim().is_empty()).then(|| ParsedToolCall {
                    name: name.trim().to_string(),
                    arguments: Value::Object(arguments).to_string(),
                })
            }
        }
    }

    /// The calls of a span's body, or `None` when it does not read as this
    /// syntax.
    fn parse_body(&self, body: &str) -> Option<Vec<ParsedToolCall>> {
        match self.separator.as_deref().filter(|s| !s.trim().is_empty()) {
            Some(separator) => body
                .split(separator.trim())
                .map(|one| self.parse_one(one))
                .collect(),
            None => Some(vec![self.parse_one(body)?]),
        }
    }

    fn canonical(calls: &[ParsedToolCall]) -> String {
        calls
            .iter()
            .map(|call| {
                let arguments = serde_json::from_str::<Value>(&call.arguments)
                    .unwrap_or_else(|_| Value::String(call.arguments.clone()));
                format!(
                    "<tool_call>{}</tool_call>",
                    json!({"name": call.name, "arguments": arguments})
                )
            })
            .collect()
    }
}

/// What stands between the first and second call of `body`.
fn separator_of(body: &str, shape: &Shape) -> Option<String> {
    let first_end = match shape {
        Shape::JsonObject { .. } => json_value_at(body, body.find('{')?)?.1,
        Shape::NameThenJson { .. } => probe_arguments(body, 0)?.1,
        Shape::Pairs { value_end, .. } => {
            let after = body.find(VALUE_B)? + VALUE_B.len();
            after + value_end.len()
        }
    };
    let rest = body.get(first_end..)?;
    let second = match shape {
        Shape::JsonObject { .. } => rest.find('{')?,
        _ => rest.find(NAME)?,
    };
    Some(rest[..second].to_string())
}

/// [`LearnedCalls`] over a stream of text: a learned call's span is held until
/// it closes and comes out as `<tool_call>` JSON; everything else goes straight
/// through. A span that does not read as a call comes out as it was written.
pub struct LearnedReader {
    calls: Arc<LearnedCalls>,
    held: String,
}

impl LearnedReader {
    pub fn new(calls: Arc<LearnedCalls>) -> Self {
        Self {
            calls,
            held: String::new(),
        }
    }

    pub fn push(&mut self, text: &str) -> String {
        self.held.push_str(text);
        let mut out = String::new();
        let open = self.calls.open.as_str();
        let close = self.calls.close.as_str();
        loop {
            let Some(at) = self.held.find(open) else {
                // Keep the longest tail that an opener could still grow from.
                let keep = (1..open.len().min(self.held.len() + 1))
                    .rev()
                    .find(|&n| open.is_char_boundary(n) && self.held.ends_with(&open[..n]))
                    .unwrap_or(0);
                let cut = self.held.len() - keep;
                out.extend(self.held.drain(..cut));
                return out;
            };
            out.extend(self.held.drain(..at));
            // A span with no closer runs to the end of the reply.
            if close.is_empty() {
                return out;
            }
            let Some(end) = self.held[open.len()..].find(close) else {
                return out;
            };
            let body = &self.held[open.len()..open.len() + end];
            match self.calls.parse_body(body) {
                Some(calls) => out.push_str(&LearnedCalls::canonical(&calls)),
                None => out.push_str(&self.held[..open.len() + end + close.len()]),
            }
            self.held.drain(..open.len() + end + close.len());
        }
    }

    /// The reply is over: what is held is a call the model ended on, closer or
    /// not, or text.
    pub fn finish(&mut self) -> String {
        let held = std::mem::take(&mut self.held);
        let Some(body) = held.strip_prefix(self.calls.open.as_str()) else {
            return held;
        };
        match self.calls.parse_body(body) {
            Some(calls) => LearnedCalls::canonical(&calls),
            None => held,
        }
    }
}

#[cfg(test)]
mod call_grammar_tests {
    use super::*;
    use crate::engine::grammar::{Grammar, Matcher};

    const JSON_CALLS: &str = "{% for m in messages %}<{{ m.role }}>{% if m.tool_calls %}{% for c in m.tool_calls %}\
        <tool_call>\n{\"name\": \"{{ c.function.name }}\", \"arguments\": {{ c.function.arguments | tojson }}}\n</tool_call>\
        {% endfor %}{% else %}{{ m.content }}{% endif %}</s>{% endfor %}\
        {% if add_generation_prompt %}<assistant>{% endif %}";

    const NAME_THEN_JSON: &str = "{% for m in messages %}<{{ m.role }}>{% if m.tool_calls %}{% for c in m.tool_calls %}\
        [CALL]{{ c.function.name }}[ARGS]{{ c.function.arguments | tojson }}\
        {% endfor %}{% else %}{{ m.content }}{% endif %}</s>{% endfor %}\
        {% if add_generation_prompt %}<assistant>{% endif %}";

    fn tools() -> Value {
        json!([
            {"type": "function", "function": {"name": "read", "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "line": {"type": "integer", "minimum": 1}},
                "required": ["path"],
            }}},
            {"type": "function", "function": {"name": "ping", "parameters": {
                "type": "object", "not": {"type": "null"},
            }}},
        ])
    }

    fn calls(template: &str, start: CallStart<'_>) -> Option<CallGrammar> {
        CallGrammar::build(&ChatTemplate::new(template.to_string()), &tools(), start)
    }

    fn takes(grammar: &Arc<Grammar>, text: &str) -> bool {
        let mut matcher = Matcher::new(grammar.clone());
        matcher.push_bytes(text.as_bytes()) && matcher.is_complete()
    }

    #[test]
    fn a_call_is_held_to_a_declared_tool_and_its_schema() {
        let calls = calls(JSON_CALLS, CallStart::Free).unwrap();
        assert_eq!(calls.trigger, "<tool_call>");
        assert!(calls.initial.is_none());
        let g = &calls.after_trigger;
        let call = |body: &str| format!("\n{body}\n</tool_call>");
        assert!(takes(
            g,
            &call(r#"{"name": "read", "arguments": {"path": "a"}}"#)
        ));
        assert!(takes(
            g,
            &call(r#"{"name": "read", "arguments": {"path": "a", "line": 3}}"#)
        ));
        // A tool nobody declared, a missing argument, a wrong type, a bound.
        assert!(!takes(
            g,
            &call(r#"{"name": "write", "arguments": {"path": "a"}}"#)
        ));
        assert!(!takes(g, &call(r#"{"name": "read", "arguments": {}}"#)));
        assert!(!takes(
            g,
            &call(r#"{"name": "read", "arguments": {"path": 1}}"#)
        ));
        assert!(!takes(
            g,
            &call(r#"{"name": "read", "arguments": {"path": "a", "line": 0}}"#)
        ));
        // Nor can it stop before the closer.
        assert!(!takes(g, r#"{"name": "read", "arguments": {"path": "a"}}"#));
    }

    /// A schema that cannot be followed does not lose the tool: its name is
    /// still held, and any object is its arguments.
    #[test]
    fn a_schema_it_cannot_follow_leaves_the_tool_any_arguments() {
        let calls = calls(JSON_CALLS, CallStart::Free).unwrap();
        let g = &calls.after_trigger;
        assert!(takes(
            g,
            "{\"name\": \"ping\", \"arguments\": {\"x\": [1]}}</tool_call>"
        ));
        assert!(!takes(
            g,
            "{\"name\": \"ping\", \"arguments\": 1}</tool_call>"
        ));
    }

    #[test]
    fn a_forced_reply_is_held_from_where_the_forcing_ended() {
        let any = calls(JSON_CALLS, CallStart::AtName).unwrap();
        let g = any.initial.as_ref().unwrap();
        assert!(takes(
            g,
            "read\", \"arguments\": {\"path\": \"a\"}}</tool_call>"
        ));
        assert!(!takes(g, "write\", \"arguments\": {}}</tool_call>"));

        let named = calls(JSON_CALLS, CallStart::AfterName("read")).unwrap();
        let g = named.initial.as_ref().unwrap();
        assert!(takes(
            g,
            "\", \"arguments\": {\"path\": \"a\"}}</tool_call>"
        ));
        assert!(!takes(g, "\", \"arguments\": {}}</tool_call>"));
    }

    #[test]
    fn a_name_then_arguments_syntax_is_followed_too() {
        let calls = calls(NAME_THEN_JSON, CallStart::Free).unwrap();
        assert_eq!(calls.trigger, "[CALL]");
        let g = &calls.after_trigger;
        assert!(takes(g, "read[ARGS]{\"path\": \"a\"}"));
        assert!(!takes(g, "write[ARGS]{\"path\": \"a\"}"));
        assert!(!takes(g, "read[ARGS]{}"));
    }

    /// What building the grammar costs a request: `orangu` sends its tools on
    /// every turn.
    #[test]
    #[ignore = "a timing report"]
    fn call_grammar_cost() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../orangu-bench/fixtures/coding-session.json"
        ))
        .unwrap();
        let tools = fixture["tools"].clone();
        let count = tools.as_array().unwrap().len();
        let template = ChatTemplate::new(JSON_CALLS.to_string());
        let started = std::time::Instant::now();
        const N: u32 = 200;
        for _ in 0..N {
            std::hint::black_box(CallGrammar::build(&template, &tools, CallStart::Free).unwrap());
        }
        println!("{count} tools: {:?} a build", started.elapsed() / N);
    }

    #[test]
    fn a_syntax_whose_arguments_are_not_json_has_no_grammar() {
        let pairs = "{% for m in messages %}<{{ m.role }}>{% if m.tool_calls %}{% for c in m.tool_calls %}\
            <call name={{ c.function.name }}>\
            {% for k, v in c.function.arguments | items %}<{{ k }}>{{ v }}</{{ k }}>{% endfor %}</call>\
            {% endfor %}{% else %}{{ m.content }}{% endif %}</s>{% endfor %}\
            {% if add_generation_prompt %}<assistant>{% endif %}";
        assert!(calls(pairs, CallStart::Free).is_none());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every split, the slow way: what [`call_text`] has to agree with.
    fn call_text_by_trying<'a>(with: &'a str, without: &str) -> Option<&'a str> {
        (0..=without.len())
            .filter(|&i| without.is_char_boundary(i))
            .find_map(|i| {
                let (head, tail) = without.split_at(i);
                (with.len() >= without.len() && with.starts_with(head) && with.ends_with(tail))
                    .then(|| &with[head.len()..with.len() - tail.len()])
            })
    }

    #[test]
    fn a_call_is_what_the_turn_gains_at_any_split() {
        let pieces = ["", "a", "ab", "</s>", "<call>", "\u{e9}", "\u{4f60}x"];
        for head in pieces {
            for call in pieces {
                for tail in pieces {
                    let with = format!("{head}{call}{tail}");
                    for without in [format!("{head}{tail}"), format!("{tail}{head}")] {
                        assert_eq!(
                            call_text(&with, &without),
                            call_text_by_trying(&with, &without),
                            "{with:?} {without:?}"
                        );
                    }
                }
            }
        }
    }

    /// A template can render as much as it likes around its call.
    #[test]
    fn a_large_render_takes_no_longer_than_reading_it() {
        let without = "x".repeat(4 << 20);
        let with = format!("{}CALL{}", &without[..1 << 20], &without[1 << 20..]);
        let started = std::time::Instant::now();
        assert_eq!(call_text(&with, &without).map(str::len), Some(4));
        assert!(started.elapsed().as_secs() < 30, "{:?}", started.elapsed());
    }

    /// A template teaching one call syntax, through its `tool_calls` branch.
    fn template(call: &str) -> ChatTemplate {
        ChatTemplate::new(format!(
            "{{% for m in messages %}}<|{{{{ m.role }}}}|>{{{{ m.content }}}}\
             {{% if m.tool_calls %}}{{% for c in m.tool_calls %}}{call}{{% endfor %}}{{% endif %}}\
             </s>{{% endfor %}}{{% if add_generation_prompt %}}<|assistant|>{{% endif %}}"
        ))
    }

    /// Runs `text` through the reader one character at a time, the way a slow
    /// tokenizer would deliver it — the shape a per-token transform has to
    /// survive — and reads the calls out of what comes back.
    fn read(calls: &LearnedCalls, text: &str) -> tool_calls::Split {
        let mut reader = LearnedReader::new(Arc::new(calls.clone()));
        let mut out = String::new();
        for c in text.chars() {
            out.push_str(&reader.push(&c.to_string()));
        }
        out.push_str(&reader.finish());
        tool_calls::split(&out)
    }

    fn weather(args: &str) -> ParsedToolCall {
        ParsedToolCall {
            name: "get_weather".to_string(),
            arguments: args.to_string(),
        }
    }

    #[test]
    fn a_call_written_as_one_json_object_is_learned() {
        let learned = LearnedCalls::learn(&template(
            "<<CALL>>{\"fn\": {{ c.function.name | tojson }}, \"args\": {{ c.function.arguments | tojson }}}<<END>>",
        ))
        .expect("learned");
        assert_eq!(learned.opener(), "<<CALL>>");
        assert_eq!(learned.closer(), "<<END>>");
        let split = read(
            &learned,
            "Sure. <<CALL>>{\"fn\": \"get_weather\", \"args\": {\"city\": \"Paris\"}}<<END>> done",
        );
        assert_eq!(split.content, "Sure.  done");
        assert_eq!(split.calls, [weather("{\"city\":\"Paris\"}")]);
    }

    #[test]
    fn a_name_followed_by_json_arguments_is_learned() {
        let learned = LearnedCalls::learn(&template(
            "[CALL]{{ c.function.name }}[ARGS]{{ c.function.arguments | tojson }}[/CALL]",
        ))
        .expect("learned");
        let split = read(
            &learned,
            "[CALL]get_weather[ARGS]{\"city\": \"Paris\", \"n\": 3}[/CALL]",
        );
        assert_eq!(split.content, "");
        assert_eq!(split.calls.len(), 1);
        assert_eq!(split.calls[0].name, "get_weather");
        let arguments: Value = serde_json::from_str(&split.calls[0].arguments).unwrap();
        assert_eq!(arguments, json!({"city": "Paris", "n": 3}));
    }

    #[test]
    fn a_name_followed_by_key_value_tags_is_learned() {
        let learned = LearnedCalls::learn(&template(
            "<call name=\"{{ c.function.name }}\">{% for k, v in c.function.arguments | items %}<k>{{ k }}</k><v>{{ v }}</v>{% endfor %}</call>",
        ))
        .expect("learned");
        let split = read(
            &learned,
            "<call name=\"get_weather\"><k>city</k><v>Paris</v><k>n</k><v>3</v></call>",
        );
        assert_eq!(split.calls.len(), 1);
        assert_eq!(split.calls[0].name, "get_weather");
        let arguments: Value = serde_json::from_str(&split.calls[0].arguments).unwrap();
        assert_eq!(arguments, json!({"city": "Paris", "n": 3}));
    }

    /// Calls that share one block's delimiters, with a separator between them.
    #[test]
    fn several_calls_in_one_block_are_each_read() {
        let learned = LearnedCalls::learn(&ChatTemplate::new(
            "{% for m in messages %}{{ m.content }}{% if m.tool_calls %}<<CALLS>>\
             {% for c in m.tool_calls %}{% if not loop.first %}<<AND>>{% endif %}\
             {{ c.function.name }}={{ c.function.arguments | tojson }}{% endfor %}\
             <</CALLS>>{% endif %}</s>{% endfor %}"
                .to_string(),
        ))
        .expect("learned");
        let split = read(
            &learned,
            "<<CALLS>>a={\"x\": 1}<<AND>>b={\"y\": 2}<</CALLS>>",
        );
        let names: Vec<&str> = split.calls.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, ["a", "b"]);
    }

    /// A model that ended the reply on a call it never closed still made it.
    #[test]
    fn a_call_the_reply_ended_on_is_read_without_its_closer() {
        let learned = LearnedCalls::learn(&template(
            "[CALL]{{ c.function.name }}[ARGS]{{ c.function.arguments | tojson }}[/CALL]",
        ))
        .unwrap();
        let split = read(&learned, "[CALL]get_weather[ARGS]{\"city\": \"Paris\"}");
        assert_eq!(split.calls.len(), 1);
    }

    /// A span that is not a call is prose, and ordinary text is untouched
    /// including what looks like the start of an opener.
    #[test]
    fn what_is_not_a_call_comes_out_as_it_was_written() {
        let learned = LearnedCalls::learn(&template(
            "[CALL]{{ c.function.name }}[ARGS]{{ c.function.arguments | tojson }}[/CALL]",
        ))
        .unwrap();
        let split = read(&learned, "a [CAL b [CALL]not a call[/CALL] c é");
        assert!(split.calls.is_empty());
        assert_eq!(split.content, "a [CAL b [CALL]not a call[/CALL] c é");
    }

    /// A parser written by hand always wins over what a template could teach.
    #[test]
    fn a_syntax_with_its_own_parser_learns_nothing() {
        for call in [
            "<tool_call>{\"name\": {{ c.function.name | tojson }}, \"arguments\": {{ c.function.arguments | tojson }}}</tool_call>",
            "[TOOL_CALLS][{\"name\": {{ c.function.name | tojson }}, \"arguments\": {{ c.function.arguments | tojson }}}]",
        ] {
            assert_eq!(LearnedCalls::learn(&template(call)), None, "{call}");
        }
    }

    /// With nothing to anchor on, a learned parser would read any JSON the
    /// model wrote as a call.
    #[test]
    fn a_syntax_with_no_opening_delimiter_is_not_learned() {
        let learned = LearnedCalls::learn(&template(
            "{\"name\": {{ c.function.name | tojson }}, \"arguments\": {{ c.function.arguments | tojson }}}",
        ));
        assert_eq!(learned, None);
    }

    #[test]
    fn a_template_without_tool_calls_learns_nothing() {
        let template = ChatTemplate::new(
            "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}</s>{% endfor %}".to_string(),
        );
        assert_eq!(LearnedCalls::learn(&template), None);
    }

    /// A turn's closing marker and a call's opening one can begin alike, and
    /// the call must not lose the characters they share.
    #[test]
    fn the_call_keeps_what_it_shares_with_the_end_of_the_turn() {
        assert_eq!(call_text("<call>x</s>", "</s>"), Some("<call>x"));
        assert_eq!(
            call_text("<think></think><call>x</s>", "<think></think></s>"),
            Some("<call>x")
        );
        assert_eq!(call_text("abc", "xyz"), None);
    }

    /// A special token spelled in the learned syntax has to reach the reader.
    #[test]
    fn the_markers_hold_the_spelling_of_the_syntax() {
        let learned = LearnedCalls::learn(&template(
            "<|call|>{{ c.function.name }}<|args|>{{ c.function.arguments | tojson }}<|end_call|>",
        ))
        .unwrap();
        let markers = learned.markers();
        for token in ["<|call|>", "<|args|>", "<|end_call|>"] {
            assert!(markers.iter().any(|m| m.contains(token)), "{token}");
        }
    }
}
