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

//! Ending a generation on what it **wrote** rather than on a token id: a
//! request's stop strings, and the rule that keeps a fill-in-the-middle
//! completion from running past the code around the cursor.
//!
//! Both have the same shape. A stop string can span tokens, and a line's
//! indentation is not known until its first visible character, so text has to
//! be held back until it is ruled out. Nothing a stop would remove ever reaches
//! a client.

/// What ends a generation, besides its stop tokens and its length.
#[derive(Clone, Debug, Default)]
pub struct StopRules {
    /// Text that ends the generation when written, and is not sent.
    pub strings: Vec<String>,
    pub infill: Option<InfillRule>,
}

/// Where a fill-in-the-middle completion ends.
///
/// Both tests apply from the first whole line on: the rest of the cursor's own
/// line is whatever the model makes it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InfillRule {
    /// A line indented less than this has left the cursor's block. `0` never
    /// applies.
    pub indent: usize,
    /// The first non-blank line after the cursor, trimmed. A line equal to it
    /// is the model writing what already follows.
    pub next_line: Option<String>,
    /// The text after the cursor starts with a line break, so a line break in
    /// front of the line a completion stops at would double it.
    pub suffix_opens_line: bool,
}

impl InfillRule {
    /// The rule for a cursor between `prefix` and `suffix`, with `indent`
    /// overriding the indentation of the cursor's own line.
    pub fn at_cursor(prefix: &str, suffix: &str, indent: Option<usize>) -> Self {
        let cursor_line = prefix.rsplit('\n').next().unwrap_or_default();
        let own = cursor_line.len() - cursor_line.trim_start_matches([' ', '\t']).len();
        Self {
            indent: indent.unwrap_or(own),
            next_line: suffix
                .lines()
                .map(str::trim)
                .find(|line| !line.is_empty())
                .map(str::to_string),
            suffix_opens_line: suffix.starts_with(['\n', '\r']),
        }
    }

    fn verdict(&self, line: &str, whole: bool) -> LineVerdict {
        let visible = line.trim_start_matches([' ', '\t', '\r']);
        if visible.is_empty() {
            return if whole {
                LineVerdict::Pass
            } else {
                LineVerdict::Hold
            };
        }
        if self.indent > 0 && line.len() - visible.len() < self.indent {
            return LineVerdict::Stop;
        }
        if let Some(next) = &self.next_line {
            let trimmed = visible.trim_end();
            if trimmed == next {
                return if whole {
                    LineVerdict::Stop
                } else {
                    LineVerdict::Hold
                };
            }
            if !whole && next.starts_with(trimmed) {
                return LineVerdict::Hold;
            }
        }
        LineVerdict::Pass
    }
}

enum LineVerdict {
    Pass,
    Hold,
    Stop,
}

/// Holds back generated text that might be the start of a stop string, and
/// reports the first one that is completed.
pub(crate) struct StopMatcher {
    stops: Vec<String>,
    held: String,
}

impl StopMatcher {
    pub(crate) fn new(stops: &[String]) -> Self {
        Self {
            stops: stops.iter().filter(|s| !s.is_empty()).cloned().collect(),
            held: String::new(),
        }
    }

    /// Takes the next piece of text. Returns what is safe to emit and, once a
    /// stop string has been written, which one — the text before it is the
    /// last there is.
    pub(crate) fn push(&mut self, text: &str) -> (String, Option<String>) {
        if self.stops.is_empty() {
            return (text.to_string(), None);
        }
        self.held.push_str(text);
        let first = self
            .stops
            .iter()
            .filter_map(|s| self.held.find(s.as_str()).map(|at| (at, s)))
            .min_by_key(|(at, _)| *at);
        if let Some((at, stop)) = first {
            let stop = stop.clone();
            let emit = self.held[..at].to_string();
            self.held.clear();
            return (emit, Some(stop));
        }
        // Keep the longest tail of `held` that some stop string starts with.
        let keep = self
            .held
            .char_indices()
            .map(|(i, _)| i)
            .find(|&i| {
                let tail = &self.held[i..];
                self.stops.iter().any(|s| s.starts_with(tail))
            })
            .unwrap_or(self.held.len());
        let emit = self.held[..keep].to_string();
        self.held.drain(..keep);
        (emit, None)
    }

    /// What is still held back when generation ends: it never became a stop
    /// string, so it belongs to the answer.
    pub(crate) fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

/// [`InfillRule`] over a stream of text.
struct InfillGate {
    rule: InfillRule,
    held: String,
    /// In the middle of a line that has been let through, so everything up to
    /// the next line break is free to go.
    mid_line: bool,
}

impl InfillGate {
    fn push(&mut self, text: &str) -> (String, bool) {
        self.held.push_str(text);
        let mut out = String::new();
        loop {
            if self.mid_line {
                match self.held.find('\n') {
                    None => {
                        out.push_str(&self.held);
                        self.held.clear();
                        return (out, false);
                    }
                    Some(at) => {
                        out.extend(self.held.drain(..at));
                        self.mid_line = false;
                    }
                }
            }
            if self.held.is_empty() {
                return (out, false);
            }
            // `held` starts at a line break: the line after it is the one being
            // judged, and the break stays held with it so that a line that
            // stops the completion takes its break along.
            let line_end = self.held[1..].find('\n');
            let line = match line_end {
                Some(end) => &self.held[1..1 + end],
                None => &self.held[1..],
            };
            match self.rule.verdict(line, line_end.is_some()) {
                LineVerdict::Hold => return (out, false),
                LineVerdict::Stop => {
                    if !self.rule.suffix_opens_line {
                        out.push('\n');
                    }
                    self.held.clear();
                    return (out, true);
                }
                LineVerdict::Pass => match line_end {
                    Some(end) => out.extend(self.held.drain(..1 + end)),
                    None => {
                        out.push_str(&self.held);
                        self.held.clear();
                        self.mid_line = true;
                        return (out, false);
                    }
                },
            }
        }
    }

    /// What is held when generation ends. The last line is whole by then, so
    /// it gets its verdict; a line that repeats the suffix is still dropped.
    fn finish(&mut self) -> String {
        let held = std::mem::take(&mut self.held);
        if held.is_empty() || self.mid_line {
            return held;
        }
        match self.rule.verdict(&held[1..], true) {
            LineVerdict::Stop if self.rule.suffix_opens_line => String::new(),
            LineVerdict::Stop => "\n".to_string(),
            _ => held,
        }
    }
}

/// Everything a reply's text passes through on its way to the client.
pub struct TextGate {
    matcher: StopMatcher,
    infill: Option<InfillGate>,
    has_rules: bool,
}

/// What [`TextGate::push`] let through, and whether the generation ends here.
pub struct Passed {
    pub text: String,
    pub stop: bool,
}

impl TextGate {
    pub fn new(rules: &StopRules) -> Self {
        Self {
            matcher: StopMatcher::new(&rules.strings),
            has_rules: !rules.strings.is_empty() || rules.infill.is_some(),
            infill: rules.infill.clone().map(|rule| InfillGate {
                rule,
                held: String::new(),
                mid_line: true,
            }),
        }
    }

    /// Whether this gate can never hold anything back.
    pub fn is_open(&self) -> bool {
        !self.has_rules
    }

    pub fn push(&mut self, text: &str) -> Passed {
        let (safe, hit) = self.matcher.push(text);
        let Some(infill) = self.infill.as_mut() else {
            return Passed {
                text: safe,
                stop: hit.is_some(),
            };
        };
        let (mut out, stop) = infill.push(&safe);
        if stop {
            return Passed { text: out, stop };
        }
        // A stop string written: what the infill rule still holds is whole.
        if hit.is_some() {
            out.push_str(&infill.finish());
        }
        Passed {
            text: out,
            stop: hit.is_some(),
        }
    }

    /// What is still held when generation ends for another reason.
    pub fn finish(&mut self) -> String {
        let rest = self.matcher.finish();
        let Some(infill) = self.infill.as_mut() else {
            return rest;
        };
        let (mut out, stop) = infill.push(&rest);
        if !stop {
            out.push_str(&infill.finish());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feeds `pieces` through `rules` one at a time, as tokens arrive: the text
    /// let through, and whether a rule ended it.
    fn run(rules: &StopRules, pieces: &[&str]) -> (String, bool) {
        let mut gate = TextGate::new(rules);
        let mut out = String::new();
        for piece in pieces {
            let passed = gate.push(piece);
            out.push_str(&passed.text);
            if passed.stop {
                return (out, true);
            }
        }
        out.push_str(&gate.finish());
        (out, false)
    }

    fn strings(stops: &[&str]) -> StopRules {
        StopRules {
            strings: stops.iter().map(|s| s.to_string()).collect(),
            infill: None,
        }
    }

    /// The whole point of holding text back: a stop string split across tokens
    /// is caught, and none of it reaches the client.
    #[test]
    fn a_stop_string_split_across_tokens_is_caught_and_not_sent() {
        assert_eq!(
            run(&strings(&["\nEND"]), &["ab", "c\n", "E", "ND", "more"]),
            ("abc".to_string(), true)
        );
    }

    /// Text held in case it became a stop string is not lost when it does not.
    #[test]
    fn held_text_that_never_becomes_a_stop_string_is_sent() {
        assert_eq!(
            run(&strings(&["STOP"]), &["a S", "T", "x", " ST"]),
            ("a STx ST".to_string(), false)
        );
    }

    /// Of several stop strings, the one written first wins, wherever it stands
    /// in the list.
    #[test]
    fn the_earliest_stop_string_in_the_text_wins() {
        assert_eq!(
            run(&strings(&["cc", "bb"]), &["aabbcc"]),
            ("aa".to_string(), true)
        );
    }

    /// A multi-byte character must not be cut when the tail is searched for a
    /// stop string's start.
    #[test]
    fn a_stop_string_is_found_among_multi_byte_text() {
        assert_eq!(
            run(&strings(&["é!"]), &["caf", "é", "!", "x"]),
            ("caf".to_string(), true)
        );
    }

    #[test]
    fn no_rules_is_an_open_gate() {
        let mut gate = TextGate::new(&StopRules::default());
        assert!(gate.is_open());
        let passed = gate.push("anything\n");
        assert_eq!(passed.text, "anything\n");
        assert!(!passed.stop);
        assert_eq!(gate.finish(), "");
    }

    fn infill(indent: usize, next_line: Option<&str>) -> StopRules {
        StopRules {
            strings: Vec::new(),
            infill: Some(InfillRule {
                indent,
                next_line: next_line.map(str::to_string),
                suffix_opens_line: false,
            }),
        }
    }

    /// Leaving the cursor's block ends the completion, on the line that left
    /// it, however the indentation arrives across tokens.
    #[test]
    fn a_line_less_indented_than_the_cursor_ends_a_completion() {
        let (text, stopped) = run(
            &infill(4, None),
            &["x = 1\n", "    y", " = 2\n", "  ", "z", " = 3\n"],
        );
        assert_eq!(text, "x = 1\n    y = 2\n");
        assert!(stopped);
    }

    /// A blank line is not a line that left the block.
    #[test]
    fn a_blank_line_does_not_end_a_completion() {
        let (text, stopped) = run(&infill(4, None), &["a\n", "\n", "  \n", "    b"]);
        assert_eq!(text, "a\n\n  \n    b");
        assert!(!stopped);
    }

    /// The cursor's own line is not judged: it continues text that is already
    /// there, so its indentation says nothing.
    #[test]
    fn the_cursors_own_line_is_never_judged() {
        let (text, stopped) = run(&infill(8, None), &["foo()"]);
        assert_eq!(text, "foo()");
        assert!(!stopped);
    }

    /// An indent of zero has nothing to leave.
    #[test]
    fn a_cursor_at_column_zero_has_no_block_to_leave() {
        let (text, stopped) = run(&infill(0, None), &["a\nb\nc"]);
        assert_eq!(text, "a\nb\nc");
        assert!(!stopped);
    }

    /// Writing the line that already follows the cursor would duplicate it, so
    /// the completion ends before it.
    #[test]
    fn a_line_repeating_the_suffix_ends_a_completion_without_being_sent() {
        let (text, stopped) = run(
            &infill(0, Some("return x")),
            &["x = 1\n", "    ret", "urn x", "\n", "more"],
        );
        assert_eq!(text, "x = 1\n");
        assert!(stopped);
    }

    /// A line that only starts like the suffix's is held and then released.
    #[test]
    fn a_line_that_only_starts_like_the_suffix_is_sent() {
        let (text, stopped) = run(
            &infill(0, Some("return x")),
            &["a\n", "ret", "urn xs + 1\n", "b"],
        );
        assert_eq!(text, "a\nreturn xs + 1\nb");
        assert!(!stopped);
    }

    /// The generation ending on the repeating line drops it too.
    #[test]
    fn a_repeat_of_the_suffix_at_the_very_end_is_dropped() {
        let (text, stopped) = run(&infill(0, Some("}")), &["a\n", "}"]);
        assert_eq!(text, "a\n");
        assert!(!stopped);
    }

    /// When the text after the cursor starts its own line, the line break in
    /// front of the line the completion stops at would double it.
    #[test]
    fn the_line_break_is_dropped_when_the_suffix_opens_a_line() {
        let rules = StopRules {
            strings: Vec::new(),
            infill: Some(InfillRule::at_cursor("    ", "\n    return x\n", None)),
        };
        let (text, stopped) = run(&rules, &["x = 1\n    return x\n"]);
        assert_eq!(text, "x = 1");
        assert!(stopped);
    }

    #[test]
    fn the_rule_reads_the_cursor_from_the_text_around_it() {
        let rule = InfillRule::at_cursor("def f():\n\tif x:\n\t\ty = ", "\n\n\treturn y\n", None);
        assert_eq!(rule.indent, 2);
        assert_eq!(rule.next_line.as_deref(), Some("return y"));
        assert!(rule.suffix_opens_line);

        // An explicit indent wins, and `0` turns the test off.
        assert_eq!(InfillRule::at_cursor("    a", "", Some(0)).indent, 0);
        assert_eq!(InfillRule::at_cursor("", "", None).next_line, None);
    }

    /// A stop string and the infill rule together, each able to end it.
    #[test]
    fn stop_strings_and_the_infill_rule_work_together() {
        let rules = StopRules {
            strings: vec!["<END>".to_string()],
            infill: Some(InfillRule {
                indent: 2,
                next_line: None,
                suffix_opens_line: false,
            }),
        };
        assert_eq!(
            run(&rules, &["a\n  b<", "END>c"]),
            ("a\n  b".to_string(), true)
        );
        assert_eq!(run(&rules, &["a\nb<END>"]), ("a\n".to_string(), true));
    }
}
