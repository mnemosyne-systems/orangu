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

//! One `choices[]` entry of `/v1/chat/completions`: the model's text as it
//! arrives, turned into OpenAI's content and tool-call deltas.
//!
//! A call's arguments go out as they are written. A call is only known to be
//! one when it closes, so a call that never closes has already been partly
//! sent: its arguments stay unfinished JSON, which no client can run, and the
//! span follows as ordinary content. A client tells it from a finished call
//! the way it does with any OpenAI model cut off by its length limit.

use serde_json::{Value, json};
use std::sync::Arc;

use crate::engine::generate::FinishReason;
use crate::engine::tool_calls::{self, ArgumentStream, ParsedToolCall, Split};
use crate::engine::tool_learn::{LearnedCalls, LearnedReader};

/// How a request wants its tool calls handled.
#[derive(Clone, Default)]
pub(super) struct CallRules {
    /// `tool_choice: "none"`: nothing the reply writes is a call.
    pub(super) withheld: bool,
    /// `parallel_tool_calls: false`: the reply makes one call at most.
    pub(super) single: bool,
    /// The syntax read from the chat template, for a model with no parser.
    pub(super) learned: Option<Arc<LearnedCalls>>,
    /// What the reply was made to begin with, which the model will not repeat.
    pub(super) prefix: Option<String>,
}

/// The id of call `index` of choice `choice`. Choice 0 keeps the short form.
pub(super) fn call_id(created: u64, choice: usize, index: usize) -> String {
    match choice {
        0 => format!("call-{created}-{index}"),
        _ => format!("call-{created}-{choice}-{index}"),
    }
}

/// OpenAI's `tool_calls` array for finished calls.
pub(super) fn whole_calls(
    calls: &[ParsedToolCall],
    created: u64,
    choice: usize,
    first_index: usize,
) -> Value {
    Value::Array(
        calls
            .iter()
            .enumerate()
            .map(|(n, call)| {
                let index = first_index + n;
                json!({
                    "index": index,
                    "id": call_id(created, choice, index),
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                })
            })
            .collect(),
    )
}

/// A whole reply's text read for calls, for a response that is not streamed.
pub(super) fn split_whole(
    text: &str,
    rules: &CallRules,
    finish: FinishReason,
    tidy: impl Fn(&str) -> String,
) -> Split {
    if rules.withheld {
        return Split {
            content: tidy(text),
            calls: Vec::new(),
        };
    }
    let mut written = rules.prefix.clone().unwrap_or_default();
    match &rules.learned {
        Some(learned) => {
            let mut reader = LearnedReader::new(learned.clone());
            let mut read = reader.push(&written);
            read.push_str(&reader.push(text));
            read.push_str(&reader.finish());
            written = read;
        }
        None => written.push_str(text),
    }
    if finish == FinishReason::Stop {
        written = tool_calls::close_trailing(&written);
    }
    let mut split = tool_calls::split(&tidy(&written));
    if rules.single {
        split.calls.truncate(1);
    }
    split
}

/// The call whose arguments are being sent.
struct OpenCall {
    name: String,
    opener: &'static str,
    stream: ArgumentStream,
}

/// One choice's text on its way out as deltas.
pub(super) struct ChoiceStream {
    rules: CallRules,
    reader: Option<LearnedReader>,
    created: u64,
    choice: usize,
    /// Text that might still turn into a call.
    pending: String,
    open: Option<OpenCall>,
    /// Call indexes handed out, the abandoned ones included: a client
    /// accumulates by index, so one is never used twice.
    calls: usize,
    finished_calls: usize,
}

impl ChoiceStream {
    pub(super) fn new(rules: CallRules, created: u64, choice: usize) -> Self {
        Self {
            reader: rules
                .learned
                .clone()
                .map(LearnedReader::new)
                .filter(|_| !rules.withheld),
            rules,
            created,
            choice,
            pending: String::new(),
            open: None,
            calls: 0,
            finished_calls: 0,
        }
    }

    /// Deltas for what the reply was made to begin with; it is read as if the
    /// model had written it.
    pub(super) fn start(&mut self) -> Vec<Value> {
        match self.rules.prefix.clone() {
            Some(prefix) if !self.rules.withheld => self.push(&prefix),
            _ => Vec::new(),
        }
    }

    /// Whether the reply made a call, which decides its `finish_reason`.
    pub(super) fn made_calls(&self) -> bool {
        self.finished_calls > 0
    }

    /// The deltas `text` makes.
    pub(super) fn push(&mut self, text: &str) -> Vec<Value> {
        let mut deltas = Vec::new();
        if self.rules.withheld {
            if !text.is_empty() {
                deltas.push(json!({"content": text}));
            }
            return deltas;
        }
        let read;
        let text = match self.reader.as_mut() {
            Some(reader) => {
                read = reader.push(text);
                read.as_str()
            }
            None => text,
        };
        self.pending.push_str(text);
        if tool_calls::may_be_partial(&self.pending) {
            self.stream_call(&mut deltas);
            return deltas;
        }
        let written = std::mem::take(&mut self.pending);
        self.absorb(tool_calls::split(&written), &written, &mut deltas);
        deltas
    }

    /// The last deltas, and the choice's `finish_reason`.
    pub(super) fn finish(&mut self, finish: FinishReason) -> (Vec<Value>, &'static str) {
        let mut deltas = Vec::new();
        if let Some(reader) = self.reader.as_mut() {
            let rest = reader.finish();
            self.pending.push_str(&rest);
        }
        // A call the model ended the reply on is a call, closer or not.
        if finish == FinishReason::Stop {
            self.pending = tool_calls::close_trailing(&self.pending);
        }
        let written = std::mem::take(&mut self.pending);
        if !written.is_empty() || self.open.is_some() {
            self.absorb(tool_calls::split(&written), &written, &mut deltas);
        }
        let reason = if self.made_calls() {
            "tool_calls"
        } else {
            super::native::finish_reason_str(finish)
        };
        (deltas, reason)
    }

    fn call_delta(&self, index: usize, call: Value) -> Value {
        let mut call = call;
        call["index"] = json!(index);
        json!({"tool_calls": [call]})
    }

    /// Once a call has named itself, what stands before it goes out, its delta
    /// opens, and its arguments follow as they grow.
    fn stream_call(&mut self, deltas: &mut Vec<Value>) {
        if self.open.is_none() {
            if self.rules.single && self.calls > 0 {
                return;
            }
            let Some((at, opener)) = tool_calls::first_opener(&self.pending) else {
                return;
            };
            let Some(name) = tool_calls::call_name(opener, &self.pending[at + opener.len()..])
            else {
                return;
            };
            let before: String = self.pending.drain(..at).collect();
            self.absorb(tool_calls::split(&before), &before, deltas);
            let index = self.calls;
            deltas.push(self.call_delta(
                index,
                json!({
                    "id": call_id(self.created, self.choice, index),
                    "type": "function",
                    "function": {"name": name, "arguments": ""},
                }),
            ));
            self.open = Some(OpenCall {
                name,
                opener,
                stream: ArgumentStream::new(opener),
            });
        }
        let index = self.calls;
        let piece = match &mut self.open {
            Some(open) => match self.pending.strip_prefix(open.opener) {
                Some(span) => open.stream.advance(span),
                None => String::new(),
            },
            None => String::new(),
        };
        if !piece.is_empty() {
            deltas.push(self.call_delta(index, json!({"function": {"arguments": piece}})));
        }
    }

    /// Text that is settled: the open call ends, its prose goes out, and any
    /// calls that were written whole follow.
    fn absorb(&mut self, split: Split, written: &str, deltas: &mut Vec<Value>) {
        let mut split = split;
        if let Some(open) = self.open.take() {
            let index = self.calls;
            self.calls += 1;
            let finished = split
                .calls
                .first()
                .is_some_and(|call| call.name == open.name)
                .then(|| split.calls.remove(0));
            if let Some(call) = finished {
                let span = written.strip_prefix(open.opener).unwrap_or(written);
                let (rest, agreed) = open.stream.finish(span, &call.arguments);
                if !agreed {
                    eprintln!(
                        "orangu-server: [chat] the streamed arguments of a `{}` call differ from its whole parse",
                        call.name
                    );
                }
                if !rest.is_empty() {
                    deltas.push(self.call_delta(index, json!({"function": {"arguments": rest}})));
                }
                self.finished_calls += 1;
            }
        }
        if !split.content.is_empty() {
            deltas.push(json!({"content": split.content}));
        }
        for call in &split.calls {
            if self.rules.single && self.calls > 0 {
                break;
            }
            let calls = whole_calls(
                std::slice::from_ref(call),
                self.created,
                self.choice,
                self.calls,
            );
            deltas.push(json!({"tool_calls": calls}));
            self.calls += 1;
            self.finished_calls += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::chat_template::ChatTemplate;
    use std::collections::BTreeMap;

    /// What a client accumulates from a stream of deltas: content, and each
    /// call by index.
    #[derive(Default, Debug, PartialEq)]
    struct Assembled {
        content: String,
        calls: BTreeMap<u64, (String, String, String)>,
    }

    impl Assembled {
        fn apply(&mut self, deltas: Vec<Value>) {
            for delta in deltas {
                if let Some(text) = delta["content"].as_str() {
                    self.content.push_str(text);
                }
                for call in delta["tool_calls"].as_array().into_iter().flatten() {
                    let entry = self
                        .calls
                        .entry(call["index"].as_u64().expect("every delta has an index"))
                        .or_default();
                    if let Some(id) = call["id"].as_str() {
                        entry.0.push_str(id);
                    }
                    if let Some(name) = call["function"]["name"].as_str() {
                        entry.1.push_str(name);
                    }
                    if let Some(arguments) = call["function"]["arguments"].as_str() {
                        entry.2.push_str(arguments);
                    }
                }
            }
        }
    }

    /// Streams `pieces`, as tokens arrive.
    fn stream(
        rules: CallRules,
        pieces: &[&str],
        finish: FinishReason,
    ) -> (Assembled, &'static str) {
        let mut choice = ChoiceStream::new(rules, 1000, 0);
        let mut got = Assembled::default();
        got.apply(choice.start());
        for piece in pieces {
            got.apply(choice.push(piece));
        }
        let (deltas, reason) = choice.finish(finish);
        got.apply(deltas);
        (got, reason)
    }

    /// `text` cut into `n`-character pieces.
    fn pieces(text: &str, n: usize) -> Vec<String> {
        text.chars()
            .collect::<Vec<_>>()
            .chunks(n)
            .map(|c| c.iter().collect())
            .collect()
    }

    /// Whatever the pieces, a stream assembles into the call a whole parse
    /// reads: every format the server has a parser for, at several widths.
    #[test]
    fn a_streamed_call_assembles_into_what_its_whole_parse_reads() {
        for (label, text) in [
            (
                "gemma",
                "Reading.<|tool_call>call:read_file{path:<|\"|>a \"b\".rs<|\"|>,limit:3}<tool_call|>",
            ),
            (
                "hermes",
                "Reading.<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"a.rs\", \"limit\": 3}}</tool_call>",
            ),
            (
                "glm",
                "Reading.<tool_call>read_file<arg_key>path</arg_key><arg_value>a.rs</arg_value><arg_key>limit</arg_key><arg_value>3</arg_value></tool_call>",
            ),
            (
                "nemotron",
                "Reading.<tool_call><function=read_file><parameter=path>a.rs</parameter><parameter=limit>3</parameter></function></tool_call>",
            ),
        ] {
            let whole = tool_calls::split(text);
            assert_eq!(whole.calls.len(), 1, "{label}");
            for width in [1, 2, 5, 13, 1000] {
                let cut = pieces(text, width);
                let cut: Vec<&str> = cut.iter().map(String::as_str).collect();
                let (got, reason) = stream(CallRules::default(), &cut, FinishReason::Stop);
                assert_eq!(reason, "tool_calls", "{label} at {width}");
                assert_eq!(got.content, "Reading.", "{label} at {width}");
                assert_eq!(got.calls.len(), 1, "{label} at {width}");
                let (id, name, arguments) = &got.calls[&0];
                assert_eq!(id, "call-1000-0");
                assert_eq!(name, &whole.calls[0].name, "{label} at {width}");
                let parsed: Value = serde_json::from_str(arguments)
                    .unwrap_or_else(|err| panic!("{label} at {width}: {arguments}: {err}"));
                assert_eq!(
                    parsed,
                    serde_json::from_str::<Value>(&whole.calls[0].arguments).unwrap(),
                    "{label} at {width}"
                );
            }
        }
    }

    /// The call's arguments have to arrive as they are written, or streaming
    /// them is a claim and not a behaviour.
    #[test]
    fn a_long_argument_arrives_before_the_call_is_finished() {
        let mut choice = ChoiceStream::new(CallRules::default(), 1, 0);
        let mut deltas = choice.push("<|tool_call>call:create_file{content:<|\"|>");
        for _ in 0..40 {
            deltas.extend(choice.push("line of code\n"));
        }
        let sent: usize = deltas
            .iter()
            .filter_map(|d| d["tool_calls"][0]["function"]["arguments"].as_str())
            .map(str::len)
            .sum();
        assert!(sent > 200, "only {sent} bytes went out before the close");
        assert_eq!(
            deltas
                .iter()
                .filter(|d| d["tool_calls"][0]["function"]["name"].as_str().is_some())
                .count(),
            1,
            "the name goes out once"
        );
    }

    /// Two calls keep separate indexes and ids, with prose between them.
    #[test]
    fn two_calls_stream_under_their_own_indexes() {
        let text = "<tool_call>{\"name\": \"a\", \"arguments\": {\"x\": 1}}</tool_call>then<tool_call>{\"name\": \"b\", \"arguments\": {\"y\": 2}}</tool_call>";
        for width in [1, 7, 1000] {
            let cut = pieces(text, width);
            let cut: Vec<&str> = cut.iter().map(String::as_str).collect();
            let (got, reason) = stream(CallRules::default(), &cut, FinishReason::Stop);
            assert_eq!(reason, "tool_calls");
            assert_eq!(got.content, "then", "at {width}");
            assert_eq!(got.calls[&0].1, "a");
            assert_eq!(got.calls[&1].1, "b");
            assert_eq!(got.calls[&1].0, "call-1000-1");
            assert_eq!(
                serde_json::from_str::<Value>(&got.calls[&1].2).unwrap(),
                json!({"y": 2}),
                "at {width}"
            );
        }
    }

    /// **The rule the whole module rests on.** A call cut off by the length
    /// limit has been partly sent, and it must not read as a call: a client
    /// that ran it would write half a file. Its arguments stay unparseable, the
    /// reason stays `length`, and the text is kept as content.
    #[test]
    fn a_call_cut_off_by_the_length_limit_cannot_be_run() {
        let text = "<|tool_call>call:create_file{path:<|\"|>a.rs<|\"|>,content:<|\"|>fn main() {";
        let cut = pieces(text, 3);
        let cut: Vec<&str> = cut.iter().map(String::as_str).collect();
        let (got, reason) = stream(CallRules::default(), &cut, FinishReason::Length);
        assert_eq!(reason, "length");
        let (_, name, arguments) = &got.calls[&0];
        assert_eq!(name, "create_file");
        assert!(
            serde_json::from_str::<Value>(arguments).is_err(),
            "a truncated call must not parse: {arguments}"
        );
        assert!(got.content.contains("fn main()"), "{:?}", got.content);
    }

    /// A span the parsers cannot read is the same: its reason is not
    /// `tool_calls`, and the text reaches the reader.
    #[test]
    fn a_span_that_does_not_parse_is_not_a_call() {
        let text = "<|tool_call>call:broken{path:<|\"|>a<tool_call|>after";
        let (got, reason) = stream(CallRules::default(), &[text], FinishReason::Stop);
        assert_eq!(reason, "stop");
        assert!(got.content.contains("after"));
    }

    /// Ordinary text is not held back, or delayed by a character that could
    /// have started an opener and did not.
    #[test]
    fn plain_text_goes_straight_through() {
        let mut choice = ChoiceStream::new(CallRules::default(), 1, 0);
        assert_eq!(choice.push("Hello"), [json!({"content": "Hello"})]);
        assert_eq!(choice.push(" world"), [json!({"content": " world"})]);
        assert!(choice.push(" <").is_empty(), "could be an opener");
        assert_eq!(choice.push("b"), [json!({"content": " <b"})]);
        let (deltas, reason) = choice.finish(FinishReason::Stop);
        assert!(deltas.is_empty());
        assert_eq!(reason, "stop");
    }

    /// `tool_choice: "none"`: what looks like a call is text, on both paths.
    #[test]
    fn nothing_is_a_call_when_the_request_withheld_the_tools() {
        let text = "<tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call>";
        let rules = CallRules {
            withheld: true,
            ..Default::default()
        };
        let (got, reason) = stream(rules.clone(), &[text], FinishReason::Stop);
        assert_eq!(reason, "stop");
        assert_eq!(got.content, text);
        assert!(got.calls.is_empty());

        let split = split_whole(text, &rules, FinishReason::Stop, str::to_string);
        assert_eq!(split.content, text);
        assert!(split.calls.is_empty());
    }

    /// `parallel_tool_calls: false`: one call, whichever way they arrive.
    #[test]
    fn a_single_call_reply_makes_one_call() {
        let text = "<tool_call>{\"name\": \"a\", \"arguments\": {}}</tool_call><tool_call>{\"name\": \"b\", \"arguments\": {}}</tool_call>";
        let rules = CallRules {
            single: true,
            ..Default::default()
        };
        for width in [1, 1000] {
            let cut = pieces(text, width);
            let cut: Vec<&str> = cut.iter().map(String::as_str).collect();
            let (got, _) = stream(rules.clone(), &cut, FinishReason::Stop);
            assert_eq!(got.calls.len(), 1, "at {width}");
            assert_eq!(got.calls[&0].1, "a");
        }
        let split = split_whole(text, &rules, FinishReason::Stop, str::to_string);
        assert_eq!(split.calls.len(), 1);
    }

    /// A reply cut at a call's closer, which is how a single-call reply ends,
    /// is read whole.
    #[test]
    fn a_reply_that_ends_on_a_call_without_its_closer_made_it() {
        let text = "<tool_call>{\"name\": \"a\", \"arguments\": {\"x\": 1}}";
        let (got, reason) = stream(CallRules::default(), &[text], FinishReason::Stop);
        assert_eq!(reason, "tool_calls");
        assert_eq!(got.calls[&0].1, "a");
        let split = split_whole(
            text,
            &CallRules::default(),
            FinishReason::Stop,
            str::to_string,
        );
        assert_eq!(split.calls.len(), 1);
    }

    /// `tool_choice` naming a function: the call's head is in the prompt, so
    /// the model only writes the rest. Both paths read it as written.
    #[test]
    fn the_prefix_a_reply_was_made_to_begin_with_counts_as_written() {
        let rules = CallRules {
            prefix: Some("<|tool_call>call:read_file".to_string()),
            ..Default::default()
        };
        let text = "{path:<|\"|>a.rs<|\"|>}<tool_call|>";
        let cut = pieces(text, 4);
        let cut: Vec<&str> = cut.iter().map(String::as_str).collect();
        let (got, reason) = stream(rules.clone(), &cut, FinishReason::Stop);
        assert_eq!(reason, "tool_calls");
        assert_eq!(got.calls[&0].1, "read_file");
        assert_eq!(
            serde_json::from_str::<Value>(&got.calls[&0].2).unwrap(),
            json!({"path": "a.rs"})
        );
        let split = split_whole(text, &rules, FinishReason::Stop, str::to_string);
        assert_eq!(split.calls[0].name, "read_file");
    }

    /// Under `required` the head stops where the name goes, and the name the
    /// model then writes is what opens the call.
    #[test]
    fn a_required_call_reads_the_name_the_model_chose() {
        let rules = CallRules {
            prefix: Some("<tool_call>\n{\"name\": \"".to_string()),
            ..Default::default()
        };
        let text = "get_time\", \"arguments\": {\"city\": \"Rome\"}}\n</tool_call>";
        let cut = pieces(text, 3);
        let cut: Vec<&str> = cut.iter().map(String::as_str).collect();
        let (got, reason) = stream(rules.clone(), &cut, FinishReason::Stop);
        assert_eq!(reason, "tool_calls");
        assert_eq!(got.calls[&0].1, "get_time");
        assert_eq!(
            serde_json::from_str::<Value>(&got.calls[&0].2).unwrap(),
            json!({"city": "Rome"})
        );
        assert_eq!(got.content, "");
    }

    /// A syntax no parser knows, taught by the template alone, reaches a
    /// client the same way as every other.
    #[test]
    fn a_learned_syntax_reads_through_both_paths() {
        let template = ChatTemplate::new(
            "{% for m in messages %}{{ m.content }}{% if m.tool_calls %}{% for c in m.tool_calls %}\
             [CALL]{{ c.function.name }}[ARGS]{{ c.function.arguments | tojson }}[/CALL]\
             {% endfor %}{% endif %}</s>{% endfor %}"
                .to_string(),
        );
        let rules = CallRules {
            learned: template.learned_calls(),
            ..Default::default()
        };
        assert!(rules.learned.is_some());
        let text = "Sure.[CALL]read_file[ARGS]{\"path\": \"a.rs\"}[/CALL]";
        let cut = pieces(text, 3);
        let cut: Vec<&str> = cut.iter().map(String::as_str).collect();
        let (got, reason) = stream(rules.clone(), &cut, FinishReason::Stop);
        assert_eq!(reason, "tool_calls");
        assert_eq!(got.content, "Sure.");
        assert_eq!(got.calls[&0].1, "read_file");
        assert_eq!(
            serde_json::from_str::<Value>(&got.calls[&0].2).unwrap(),
            json!({"path": "a.rs"})
        );
        let split = split_whole(text, &rules, FinishReason::Stop, str::to_string);
        assert_eq!(split.calls[0].name, "read_file");
        assert_eq!(split.content, "Sure.");
    }

    /// A later choice's ids are its own: two choices answer one request.
    #[test]
    fn call_ids_differ_between_choices() {
        assert_eq!(call_id(5, 0, 1), "call-5-1");
        assert_ne!(call_id(5, 1, 1), call_id(5, 0, 1));
    }
}
