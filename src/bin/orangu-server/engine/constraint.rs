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

//! Constrained decoding: which tokens the sampler is still allowed to pick.
//!
//! Without this, "return JSON" is a request the model may decline. Everything
//! this project builds on structured output — tool calls, `/auto_review`'s
//! per-category verdicts, duplicate reports — depends on the model choosing to
//! comply, and a model that opens with "Sure! Here's the JSON:" has produced
//! something no parser accepts. A constraint makes the malformed output
//! unreachable rather than unlikely.
//!
//! # What it constrains
//!
//! [`JsonPrefix`] accepts exactly the byte strings that are a **prefix of some
//! valid JSON document**. It is a recogniser for prefixes, not for documents:
//! at every point it answers "could this still become valid JSON", which is
//! the question a decoder has to ask, and separately reports whether what it
//! has *is already* a complete document ([`JsonPrefix::is_complete`]), which
//! is the question the stop condition has to ask.
//!
//! It deliberately does not know about a *schema*. `response_format`'s
//! `json_object` mode is the half of the OpenAI contract that says "valid
//! JSON" and nothing more, and it is the half that makes the existing
//! delimiter-anchored tool-call parsers reliable rather than hopeful. Schema
//! shape (`json_schema` mode) is a strictly larger job and is not this.
//!
//! # Where the cost goes
//!
//! A mask over a 262k-token vocabulary, recomputed per step, is not something
//! to do naively. Two things keep it affordable: the automaton is a few bytes
//! of state that is cheap to clone and replay, and the sampler only ever asks
//! about tokens it is still considering — the argmax candidate on the greedy
//! path, or the post-`top_k` set on the sampled one. See
//! [`crate::engine::sampling::Sampler::sample`].

/// What the automaton is in the middle of reading.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Frame {
    /// Inside `{ ... }`, between members.
    Object,
    /// Inside `[ ... ]`, between elements.
    Array,
}

/// Where the next byte lands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    /// A value is **required** here: the start of the document, after `:`,
    /// and after `,` in an array. A closing bracket is not a value, so it
    /// cannot appear — which is what rejects the trailing comma in `[1,]`.
    Value,
    /// The document has not started and must begin with `{` — the start state
    /// of [`JsonPrefix::object`].
    RootObject,
    /// A value is expected, but the array may also close: immediately after
    /// `[`, and only there. Splitting this from [`State::Value`] is what
    /// separates the empty array `[]` from the trailing comma `[1,]`, which
    /// are otherwise the same "a `]` arrived where a value was expected".
    ValueOrClose,
    /// A value has been read; what may follow is `,`, a closer, or — at the
    /// top level — nothing more.
    AfterValue,
    /// Inside a string literal.
    InString { escape: bool, unicode: u8 },
    /// Inside a number. Tracks what is still legal so `1..2` and `01` are
    /// rejected while `1`, `-1.5e+3` and `0` are not.
    InNumber(NumberState),
    /// A bare word (`true`, `false`, `null`) is being matched against
    /// `LITERALS[which]`, `matched` bytes in.
    InLiteral { which: u8, matched: u8 },
    /// An object key is **required** — after `,`. A `}` here is a trailing
    /// comma.
    ObjectKey,
    /// A key is expected, but the object may also close: immediately after
    /// `{`, and only there.
    ObjectFirstKey,
    /// A key has been read; `:` must follow.
    ObjectColon,
}

/// The parts of a JSON number that have been seen. JSON's grammar forbids a
/// leading zero, a bare `-`, a trailing `.`, and an exponent with no digits,
/// and a prefix recogniser has to allow each of those *while they are still
/// incomplete* and refuse them as final.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumberState {
    /// `-` seen, no digits yet. Not a complete number.
    Minus,
    /// Digits seen, and the integer part started with `0`, so no more digits
    /// may follow it.
    Zero,
    /// Digits seen, integer part.
    Int,
    /// `.` seen, no fraction digits yet. Not complete.
    Dot,
    /// Fraction digits seen.
    Frac,
    /// `e`/`E` seen, no exponent digits yet. Not complete.
    Exp,
    /// `+`/`-` after the exponent marker, no digits yet. Not complete.
    ExpSign,
    /// Exponent digits seen.
    ExpDigits,
}

impl NumberState {
    /// Whether a number stopping here is a whole number rather than a
    /// half-typed one.
    fn is_complete(self) -> bool {
        matches!(self, Self::Zero | Self::Int | Self::Frac | Self::ExpDigits)
    }
}

const LITERALS: [&[u8]; 3] = [b"true", b"false", b"null"];

/// A recogniser for prefixes of valid JSON.
///
/// Cheap to clone, which is the point: testing whether a token is allowed is
/// "clone, feed its bytes, see if it survived", and that happens many times
/// per generated token.
#[derive(Clone, Debug)]
pub struct JsonPrefix {
    stack: Vec<Frame>,
    state: State,
    /// Set once the document is finished and only trailing whitespace may
    /// follow. Distinct from an empty stack, which is also true *before*
    /// anything has been read.
    done: bool,
    /// Whether the string currently open is an object *key* rather than a
    /// value. Tracked rather than inferred: "inside an object, reading a
    /// string" is ambiguous, because keys and values are both strings there,
    /// and getting it wrong would let `{"a" "b"}` through.
    reading_key: bool,
}

impl Default for JsonPrefix {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonPrefix {
    /// Accepts any JSON *value* at the top level — an object, an array, a
    /// string, a number, or a literal.
    pub fn new() -> Self {
        Self {
            stack: Vec::new(),
            state: State::Value,
            done: false,
            reading_key: false,
        }
    }

    /// Accepts only a JSON **object** at the top level.
    ///
    /// What `response_format: {"type": "json_object"}` should mean, and what
    /// its name says. A bare string is valid JSON, so the looser
    /// [`new`](Self::new) is satisfied by `"Name: Sophia, Age: 32"` — which is
    /// exactly what a model asked for a record produced in the first run of
    /// this, prose wrapped in quotes. It parses, and it breaks the caller at
    /// `result["name"]` instead of at the parse, which is worse than failing
    /// outright.
    pub fn object() -> Self {
        Self {
            state: State::RootObject,
            ..Self::new()
        }
    }

    /// Whether what has been read so far is already a complete JSON document,
    /// so generation may stop here.
    ///
    /// Separate from "could still become valid", and both are needed: the mask
    /// uses the first to decide what may come next, and the stop condition
    /// uses this one to decide whether an end-of-sequence token is allowed
    /// yet. Without it a model can emit `{"a":` and stop, which is a prefix of
    /// valid JSON and not valid JSON.
    pub fn is_complete(&self) -> bool {
        if self.done {
            return true;
        }
        self.stack.is_empty()
            && match self.state {
                State::AfterValue => true,
                State::InNumber(n) => n.is_complete(),
                // A document that is just `null`/`true`/`false` never sees a
                // byte after the literal, so nothing ever ends the value for
                // it — the literal being fully matched *is* the document.
                State::InLiteral { which, matched } => {
                    usize::from(matched) == LITERALS[which as usize].len()
                }
                _ => false,
            }
    }

    /// Feeds one byte. Returns `false` if it cannot appear here, in which case
    /// `self` is left in an unspecified state and must be discarded.
    fn push_byte(&mut self, b: u8) -> bool {
        // Whitespace is allowed between tokens, but not inside a string, a
        // number, or a bare literal — those are terminated by it instead.
        if matches!(b, b' ' | b'\t' | b'\n' | b'\r') {
            return match self.state {
                State::InString { .. } => self.push_string_byte(b),
                State::InNumber(n) => {
                    if !n.is_complete() {
                        return false;
                    }
                    self.end_value();
                    true
                }
                State::InLiteral { which, matched } => {
                    if usize::from(matched) != LITERALS[which as usize].len() {
                        return false;
                    }
                    self.end_value();
                    true
                }
                _ => true,
            };
        }
        if self.done {
            // Only whitespace may follow a finished document, and that was
            // handled above.
            return false;
        }
        match self.state {
            State::InString { .. } => self.push_string_byte(b),
            State::InNumber(n) => self.push_number_byte(n, b),
            State::InLiteral { which, matched } => {
                let want = LITERALS[which as usize];
                if usize::from(matched) < want.len() {
                    if want[usize::from(matched)] != b {
                        return false;
                    }
                    self.state = State::InLiteral {
                        which,
                        matched: matched + 1,
                    };
                    return true;
                }
                // The literal is finished; this byte belongs to whatever
                // follows it.
                self.end_value();
                self.push_byte(b)
            }
            State::RootObject => {
                if b == b'{' {
                    self.push_value_start(b)
                } else {
                    false
                }
            }
            State::Value => self.push_value_start(b),
            // Only difference from `Value`: the array may close here.
            State::ValueOrClose => {
                if b == b']' && self.stack.last() == Some(&Frame::Array) {
                    self.stack.pop();
                    self.end_value();
                    true
                } else {
                    self.push_value_start(b)
                }
            }
            State::AfterValue => self.push_after_value(b),
            State::ObjectKey | State::ObjectFirstKey => match b {
                b'"' => {
                    self.reading_key = true;
                    self.state = State::InString {
                        escape: false,
                        unicode: 0,
                    };
                    true
                }
                // Only an object that has just opened may close without a
                // member; after a comma this is a trailing comma.
                b'}' if self.state == State::ObjectFirstKey => {
                    self.stack.pop();
                    self.end_value();
                    true
                }
                _ => false,
            },
            State::ObjectColon => {
                if b == b':' {
                    self.state = State::Value;
                    true
                } else {
                    false
                }
            }
        }
    }

    fn push_value_start(&mut self, b: u8) -> bool {
        match b {
            b'"' => {
                self.state = State::InString {
                    escape: false,
                    unicode: 0,
                };
                true
            }
            b'{' => {
                self.stack.push(Frame::Object);
                self.state = State::ObjectFirstKey;
                true
            }
            b'[' => {
                self.stack.push(Frame::Array);
                self.state = State::ValueOrClose;
                true
            }
            b'-' => {
                self.state = State::InNumber(NumberState::Minus);
                true
            }
            b'0' => {
                self.state = State::InNumber(NumberState::Zero);
                true
            }
            b'1'..=b'9' => {
                self.state = State::InNumber(NumberState::Int);
                true
            }
            b't' | b'f' | b'n' => {
                let which = match b {
                    b't' => 0,
                    b'f' => 1,
                    _ => 2,
                };
                self.state = State::InLiteral { which, matched: 1 };
                true
            }
            _ => false,
        }
    }

    fn push_after_value(&mut self, b: u8) -> bool {
        match (b, self.stack.last().copied()) {
            (b',', Some(Frame::Object)) => {
                self.state = State::ObjectKey;
                true
            }
            (b',', Some(Frame::Array)) => {
                self.state = State::Value;
                true
            }
            (b'}', Some(Frame::Object)) | (b']', Some(Frame::Array)) => {
                self.stack.pop();
                self.end_value();
                true
            }
            _ => false,
        }
    }

    fn push_string_byte(&mut self, b: u8) -> bool {
        let State::InString { escape, unicode } = self.state else {
            return false;
        };
        if unicode > 0 {
            if !b.is_ascii_hexdigit() {
                return false;
            }
            self.state = State::InString {
                escape: false,
                unicode: unicode - 1,
            };
            return true;
        }
        if escape {
            return match b {
                b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => {
                    self.state = State::InString {
                        escape: false,
                        unicode: 0,
                    };
                    true
                }
                b'u' => {
                    self.state = State::InString {
                        escape: false,
                        unicode: 4,
                    };
                    true
                }
                _ => false,
            };
        }
        match b {
            b'"' => {
                // Closing quote. In an object this may have been the key.
                if self.reading_key {
                    self.reading_key = false;
                    self.state = State::ObjectColon;
                } else {
                    self.end_value();
                }
                true
            }
            b'\\' => {
                self.state = State::InString {
                    escape: true,
                    unicode: 0,
                };
                true
            }
            // JSON forbids raw control characters inside a string.
            0x00..=0x1f => false,
            _ => true,
        }
    }

    fn push_number_byte(&mut self, n: NumberState, b: u8) -> bool {
        use NumberState::*;
        let next = match (n, b) {
            (Minus, b'0') => Zero,
            (Minus, b'1'..=b'9') => Int,
            (Int, b'0'..=b'9') => Int,
            (Zero | Int, b'.') => Dot,
            (Dot | Frac, b'0'..=b'9') => Frac,
            (Zero | Int | Frac, b'e' | b'E') => Exp,
            (Exp, b'+' | b'-') => ExpSign,
            (Exp | ExpSign | ExpDigits, b'0'..=b'9') => ExpDigits,
            _ => {
                // Not part of the number. It terminates here if it can, and
                // the byte belongs to whatever follows.
                if !n.is_complete() {
                    return false;
                }
                self.end_value();
                return self.push_byte(b);
            }
        };
        self.state = State::InNumber(next);
        true
    }

    /// Finishes the value just read and moves to whatever may follow it.
    fn end_value(&mut self) {
        self.reading_key = false;
        self.done = self.stack.is_empty();
        self.state = State::AfterValue;
    }

    /// Feeds every byte of `bytes`, returning `false` at the first one that
    /// cannot appear. `self` is only usable afterwards if this returned
    /// `true`.
    pub fn push_bytes(&mut self, bytes: &[u8]) -> bool {
        bytes.iter().all(|&b| self.push_byte(b))
    }

    /// Whether `bytes` could follow what has been read, without committing to
    /// them.
    pub fn allows(&self, bytes: &[u8]) -> bool {
        let mut probe = self.clone();
        probe.push_bytes(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feeds a whole string and reports `(still a valid prefix, complete)`.
    fn run(s: &str) -> (bool, bool) {
        let mut p = JsonPrefix::new();
        let ok = p.push_bytes(s.as_bytes());
        (ok, ok && p.is_complete())
    }

    /// Real documents are accepted and recognised as finished.
    #[test]
    fn complete_documents_are_accepted_and_complete() {
        for doc in [
            "{}",
            "[]",
            "null",
            "true",
            "false",
            "0",
            "-1.5e+10",
            "\"\"",
            "\"hi\"",
            "{\"a\":1}",
            "{\"a\":[1,2,{\"b\":null}]}",
            "[ 1 , 2 ]",
            "  {\n  \"k\" : \"v\"\n}  ",
            "\"\\u00e9\\n\\\\\\\"\"",
            "[[[[]]]]",
        ] {
            assert_eq!(run(doc), (true, true), "{doc:?}");
        }
    }

    /// Prefixes of valid documents are accepted but **not** complete — the
    /// distinction the stop condition depends on. A model that emits `{"a":`
    /// and stops has produced a valid prefix and invalid JSON.
    #[test]
    fn partial_documents_are_accepted_but_incomplete() {
        for part in [
            "",
            "{",
            "{\"",
            "{\"a",
            "{\"a\"",
            "{\"a\":",
            "{\"a\":1",
            "[",
            "[1,",
            "-",
            "1.",
            "1e",
            "1e+",
            "\"unterminated",
            "\"\\",
            "\"\\u00",
            "tru",
            // A valid prefix of `null`, and emphatically not a document.
            "nul",
        ] {
            let (valid, complete) = run(part);
            assert!(valid, "{part:?} should still be a valid prefix");
            assert!(!complete, "{part:?} must not read as a complete document");
        }
    }

    /// Byte strings that can never become valid JSON are rejected outright.
    /// These are the ones a model actually produces when asked for JSON.
    #[test]
    fn malformed_documents_are_rejected() {
        for bad in [
            "Sure! Here's the JSON: {}",
            "```json\n{}\n```",
            "{,}",
            "{\"a\" 1}",
            "{\"a\":1,}",
            "{'a':1}",
            "[1,]",
            "[}",
            "{]",
            "01",
            "1..2",
            "-.5",
            "+1",
            "tru3",
            "\"\\x\"",
            "\"\\u00zz\"",
            "{}}",
            "[]]",
            "{\"a\":1}{",
        ] {
            assert!(!run(bad).0, "{bad:?} should have been rejected");
        }
    }

    /// A raw control byte inside a string is invalid JSON and has to be
    /// refused — it is the one thing inside a string that is not simply
    /// "any byte until the closing quote".
    #[test]
    fn raw_control_bytes_in_strings_are_rejected() {
        let mut p = JsonPrefix::new();
        assert!(p.push_bytes(b"\""));
        assert!(!p.clone().push_bytes(&[0x0a]), "raw newline");
        assert!(!p.clone().push_bytes(&[0x00]), "raw NUL");
        assert!(p.clone().push_bytes(b"\\n"), "escaped newline is fine");
    }

    /// Keys and values are both strings, so the automaton has to remember
    /// which one it is reading — otherwise `{"a" "b"}` looks like two values
    /// in a row and is accepted.
    #[test]
    fn an_object_key_must_be_followed_by_a_colon() {
        assert!(!run("{\"a\" \"b\"}").0);
        assert!(!run("{\"a\",\"b\"}").0);
        assert!(run("{\"a\":\"b\"}").1);
        // ... and the same string inside an array is a value, not a key.
        assert!(run("[\"a\",\"b\"]").1);
    }

    /// `allows` must not disturb the state it is asked about — it is called
    /// once per candidate token per step, and a probe that committed would
    /// corrupt the very state the next probe reads.
    #[test]
    fn allows_does_not_mutate() {
        let mut p = JsonPrefix::new();
        assert!(p.push_bytes(b"{\"a\":"));
        assert!(p.allows(b"1"));
        assert!(p.allows(b"\"x\""));
        assert!(!p.allows(b"}"));
        // Still exactly where it was, and still able to take either value.
        assert!(p.allows(b"1"));
        assert!(!p.is_complete());
        assert!(p.push_bytes(b"1}"));
        assert!(p.is_complete());
    }

    /// Nothing may follow a finished document except whitespace. Without this
    /// a model can emit a valid object and then keep talking.
    #[test]
    fn a_finished_document_accepts_only_trailing_whitespace() {
        let mut p = JsonPrefix::new();
        assert!(p.push_bytes(b"{}"));
        assert!(p.is_complete());
        assert!(p.allows(b"  \n"));
        assert!(!p.allows(b"x"));
        assert!(!p.allows(b"{"));
        assert!(p.push_bytes(b" \n "));
        assert!(p.is_complete(), "whitespace must not un-finish it");
    }
}

#[cfg(test)]
mod object_mode_tests {
    use super::*;

    fn run_object(s: &str) -> (bool, bool) {
        let mut p = JsonPrefix::object();
        let ok = p.push_bytes(s.as_bytes());
        (ok, ok && p.is_complete())
    }

    /// `json_object` means an object. A bare string is valid JSON and is not
    /// what a caller asking for a record wants — the model's first constrained
    /// answer here was `"Name: Sophia Patel, Age: 32"`, which parses and then
    /// breaks the caller at `result["name"]`.
    #[test]
    fn object_mode_rejects_every_other_top_level_value() {
        assert_eq!(run_object("{}"), (true, true));
        assert_eq!(run_object("  {\"a\":1}"), (true, true));
        for not_an_object in ["\"a string\"", "[]", "[1,2]", "42", "true", "null"] {
            assert!(
                !run_object(not_an_object).0,
                "{not_an_object:?} is not an object"
            );
        }
        // Nested values are still values — the restriction is on the root.
        assert_eq!(run_object("{\"a\":[1,\"b\",null]}"), (true, true));
    }

    /// The looser mode still exists and still takes any value, so the two are
    /// a real choice rather than one spelling.
    #[test]
    fn value_mode_still_accepts_any_value() {
        for doc in ["\"s\"", "[]", "42", "true", "{}"] {
            let mut p = JsonPrefix::new();
            assert!(p.push_bytes(doc.as_bytes()) && p.is_complete(), "{doc:?}");
        }
    }
}
