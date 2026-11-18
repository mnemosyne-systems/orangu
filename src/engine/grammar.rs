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

//! Grammars in GBNF: what a request's `grammar` field holds, and what a JSON
//! Schema is compiled to.
//!
//! [`Grammar::parse`] reads the text into flat rules, and a [`Matcher`] follows
//! one through a reply a byte at a time, answering what the sampler asks:
//! could these bytes come next, and is what has been read already a whole
//! sentence of the grammar.
//!
//! # How it matches
//!
//! The grammar comes from outside, and the matcher runs on every token, which
//! decides its shape:
//!
//! - A way of having read the input is a *head*: the element to match next, and
//!   a stack of where to return to. Stacks are hash-consed, so a stack is a
//!   number and two ways of reading the input that have reached the same place
//!   are one head. Ambiguity costs what it costs to compare two numbers.
//! - A call in tail position pushes nothing, so `x*` reads forever in constant
//!   space.
//! - Nothing recurses on the input or on the grammar's depth: closures run on a
//!   work list, so no grammar can overflow the stack.
//! - The first bytes each state allows are a table, so most tokens a sampler
//!   asks about are turned down by one lookup.
//! - [`MAX_HEADS`] bounds what is left: a grammar can still be built to keep
//!   many genuinely different stacks alive, and past the bound the newest are
//!   dropped, so it can refuse text it describes but cannot take the server's
//!   time.
//!
//! Token references (`<[1000]>`, `<think>`) are refused when parsing: a grammar
//! here is over the text of a reply.

use anyhow::{Result, anyhow, bail};
use rustc_hash::{FxHashMap, FxHashSet};
use std::sync::Arc;

/// The longest grammar text accepted.
const MAX_SOURCE_BYTES: usize = 1 << 20;
/// The most elements a grammar may expand to, repetitions included.
const MAX_ELEMENTS: usize = 1 << 21;
/// The largest count in a repetition.
const MAX_REPETITION: usize = 1 << 14;
/// How deeply groups may nest in the text.
const MAX_GROUP_DEPTH: usize = 64;
/// The most ways of reading the input carried at once.
const MAX_HEADS: usize = 1024;

/// One step of a rule. Alternatives of a rule are separated by [`Elem::Alt`],
/// and a rule ends with [`Elem::End`].
#[derive(Clone, Copy, Debug, PartialEq)]
enum Elem {
    End,
    Alt,
    Char(u32),
    Class(u32),
    Rule(u32),
}

/// A set of characters, as sorted ranges, with the ASCII ones as a bitmap.
#[derive(Clone, Debug)]
struct CharClass {
    ranges: Vec<(u32, u32)>,
    ascii: [u64; 2],
    /// The bytes a character of this class can begin with.
    first_bytes: [u64; 4],
}

impl CharClass {
    fn new(mut ranges: Vec<(u32, u32)>, negated: bool) -> Self {
        ranges.sort_unstable();
        let mut merged: Vec<(u32, u32)> = Vec::with_capacity(ranges.len());
        for (low, high) in ranges {
            match merged.last_mut() {
                Some(last) if low <= last.1.saturating_add(1) => last.1 = last.1.max(high),
                _ => merged.push((low, high)),
            }
        }
        if negated {
            let mut inverse = Vec::with_capacity(merged.len() + 1);
            let mut next = 0;
            for (low, high) in merged {
                if low > next {
                    inverse.push((next, low - 1));
                }
                next = high.saturating_add(1);
            }
            if next <= 0x10_FFFF {
                inverse.push((next, 0x10_FFFF));
            }
            merged = inverse;
        }
        let mut ascii = [0u64; 2];
        let mut first_bytes = [0u64; 4];
        for &(low, high) in &merged {
            for cp in low..=high.min(0x7F) {
                ascii[(cp / 64) as usize] |= 1 << (cp % 64);
            }
            // A multi-byte range: every lead byte from its first character's to
            // its last's.
            if high >= 0x80 {
                let first = lead_byte(low.max(0x80));
                let last = lead_byte(high.min(0x10_FFFF));
                for byte in first..=last {
                    first_bytes[(byte / 64) as usize] |= 1 << (byte % 64);
                }
            }
        }
        first_bytes[0] |= ascii[0];
        first_bytes[1] |= ascii[1];
        Self {
            ranges: merged,
            ascii,
            first_bytes,
        }
    }

    fn contains(&self, cp: u32) -> bool {
        if cp < 128 {
            return self.ascii[(cp / 64) as usize] & (1 << (cp % 64)) != 0;
        }
        let at = self.ranges.partition_point(|&(_, high)| high < cp);
        self.ranges.get(at).is_some_and(|&(low, _)| low <= cp)
    }

    /// Whether the class holds any character from `low` to `high`.
    fn intersects(&self, low: u32, high: u32) -> bool {
        let at = self.ranges.partition_point(|&(_, end)| end < low);
        self.ranges.get(at).is_some_and(|&(start, _)| start <= high)
    }
}

/// The first byte of a character's UTF-8 form.
fn lead_byte(cp: u32) -> u8 {
    match cp {
        0..=0x7F => cp as u8,
        0x80..=0x7FF => 0xC0 | (cp >> 6) as u8,
        0x800..=0xFFFF => 0xE0 | (cp >> 12) as u8,
        _ => 0xF0 | (cp >> 18) as u8,
    }
}

/// A parsed grammar: flat, and shared by every request that uses it.
#[derive(Debug)]
pub struct Grammar {
    elements: Vec<Elem>,
    classes: Vec<CharClass>,
    /// Each rule's alternatives, as where each starts in `elements`.
    alternatives: Vec<Vec<u32>>,
    root: u32,
}

/// What a rule is built of while parsing.
type Sequence = Vec<Elem>;

struct Parser<'a> {
    text: &'a [u8],
    at: usize,
    names: FxHashMap<String, u32>,
    /// Each rule's alternatives; `None` until it is defined.
    rules: Vec<Option<Vec<Sequence>>>,
    /// Each rule's name, for reporting an undefined one.
    rule_names: Vec<String>,
    classes: Vec<CharClass>,
    elements: usize,
    depth: usize,
}

impl Grammar {
    /// Reads GBNF text whose starting rule is `root`.
    pub fn parse(text: &str) -> Result<Arc<Self>> {
        if text.len() > MAX_SOURCE_BYTES {
            bail!(
                "a grammar of {} bytes is more than the {MAX_SOURCE_BYTES} allowed",
                text.len()
            );
        }
        let mut parser = Parser {
            text: text.as_bytes(),
            at: 0,
            names: FxHashMap::default(),
            rules: Vec::new(),
            rule_names: Vec::new(),
            classes: Vec::new(),
            elements: 0,
            depth: 0,
        };
        parser.grammar()?;
        let root = *parser
            .names
            .get("root")
            .filter(|&&id| parser.rules[id as usize].is_some())
            .ok_or_else(|| anyhow!("a grammar needs a `root` rule"))?;
        let mut elements = Vec::with_capacity(parser.elements + parser.rules.len());
        let mut alternatives = Vec::with_capacity(parser.rules.len());
        let (rules, rule_names, classes) = (parser.rules, parser.rule_names, parser.classes);
        for (id, rule) in rules.iter().enumerate() {
            let Some(rule) = rule else {
                bail!("the grammar's rule `{}` is not defined", rule_names[id]);
            };
            let mut starts = Vec::with_capacity(rule.len());
            for (index, sequence) in rule.iter().enumerate() {
                if index > 0 {
                    elements.push(Elem::Alt);
                }
                starts.push(elements.len() as u32);
                elements.extend_from_slice(sequence);
            }
            elements.push(Elem::End);
            alternatives.push(starts);
        }
        let grammar = Self {
            elements,
            classes,
            alternatives,
            root,
        };
        grammar.refuse_left_recursion(rules.as_slice(), &rule_names)?;
        Ok(Arc::new(grammar))
    }

    /// A rule that can reach itself before reading anything would call itself
    /// for ever, so a grammar with one is refused.
    fn refuse_left_recursion(
        &self,
        rules: &[Option<Vec<Sequence>>],
        names: &[String],
    ) -> Result<()> {
        let rule = |id: usize| rules[id].as_deref().unwrap_or(&[]);
        // Which rules can match nothing: to a fixed point, as each answer can
        // depend on others.
        let mut nullable = vec![false; rules.len()];
        let mut work: Vec<u32> = Vec::new();
        let mut callers: Vec<Vec<u32>> = vec![Vec::new(); rules.len()];
        for id in 0..rules.len() {
            for sequence in rule(id) {
                for element in sequence {
                    if let Elem::Rule(called) = element {
                        callers[*called as usize].push(id as u32);
                    }
                }
            }
            work.push(id as u32);
        }
        while let Some(id) = work.pop() {
            let id = id as usize;
            if nullable[id] {
                continue;
            }
            let empty = rule(id).iter().any(|sequence| {
                sequence
                    .iter()
                    .all(|element| matches!(element, Elem::Rule(r) if nullable[*r as usize]))
            });
            if empty {
                nullable[id] = true;
                work.extend_from_slice(&callers[id]);
            }
        }
        // Each rule's left edges: what it can call before reading a character,
        // and whether the call leaves anything to come back for. A call with
        // nothing after it pushes nothing, so a cycle of those, as in `(x*)*`,
        // only comes back to where it was, which a closure already handles.
        let mut edges: Vec<Vec<(u32, bool)>> = vec![Vec::new(); rules.len()];
        for (id, targets) in edges.iter_mut().enumerate() {
            for sequence in rule(id) {
                for (index, element) in sequence.iter().enumerate() {
                    let Elem::Rule(called) = element else {
                        break;
                    };
                    targets.push((*called, index + 1 < sequence.len()));
                    if !nullable[*called as usize] {
                        break;
                    }
                }
            }
        }
        // Components of the left edges, without recursing (Tarjan's, with the
        // call stack written out): a stack-growing edge inside one is a rule
        // that calls itself deeper for ever.
        const UNSEEN: u32 = u32::MAX;
        let count = rules.len();
        let (mut index, mut low, mut on_stack) =
            (vec![UNSEEN; count], vec![0u32; count], vec![false; count]);
        let mut component = vec![UNSEEN; count];
        let (mut next_index, mut next_component) = (0u32, 0u32);
        let mut stack: Vec<u32> = Vec::new();
        for start in 0..count {
            if index[start] != UNSEEN {
                continue;
            }
            let mut path: Vec<(usize, usize)> = vec![(start, 0)];
            index[start] = next_index;
            low[start] = next_index;
            next_index += 1;
            stack.push(start as u32);
            on_stack[start] = true;
            while let Some(&mut (id, ref mut edge)) = path.last_mut() {
                if let Some(&(called, _)) = edges[id].get(*edge) {
                    *edge += 1;
                    let called = called as usize;
                    if index[called] == UNSEEN {
                        index[called] = next_index;
                        low[called] = next_index;
                        next_index += 1;
                        stack.push(called as u32);
                        on_stack[called] = true;
                        path.push((called, 0));
                    } else if on_stack[called] {
                        low[id] = low[id].min(index[called]);
                    }
                    continue;
                }
                path.pop();
                if let Some(&(parent, _)) = path.last() {
                    low[parent] = low[parent].min(low[id]);
                }
                if low[id] == index[id] {
                    while let Some(member) = stack.pop() {
                        on_stack[member as usize] = false;
                        component[member as usize] = next_component;
                        if member as usize == id {
                            break;
                        }
                    }
                    next_component += 1;
                }
            }
        }
        for (id, targets) in edges.iter().enumerate() {
            for &(called, grows) in targets {
                if grows && component[id] == component[called as usize] {
                    let named = if names[id].is_empty() {
                        "a group"
                    } else {
                        &names[id]
                    };
                    bail!("the grammar's rule `{named}` is left-recursive");
                }
            }
        }
        Ok(())
    }
}

impl Parser<'_> {
    fn peek(&self) -> Option<u8> {
        self.text.get(self.at).copied()
    }

    fn line(&self) -> usize {
        self.text[..self.at].iter().filter(|&&b| b == b'\n').count() + 1
    }

    fn fail<T>(&self, what: &str) -> Result<T> {
        bail!("grammar line {}: {what}", self.line())
    }

    /// Skips spaces and comments, and newlines where they mean nothing.
    fn space(&mut self, newlines: bool) {
        while let Some(byte) = self.peek() {
            match byte {
                b' ' | b'\t' => self.at += 1,
                b'\r' | b'\n' if newlines => self.at += 1,
                b'#' => {
                    while self.peek().is_some_and(|b| b != b'\n' && b != b'\r') {
                        self.at += 1;
                    }
                }
                _ => break,
            }
        }
    }

    fn name(&mut self) -> Result<String> {
        let start = self.at;
        while self
            .peek()
            .is_some_and(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
        {
            self.at += 1;
        }
        if start == self.at {
            return self.fail("expecting a rule name");
        }
        Ok(String::from_utf8_lossy(&self.text[start..self.at]).into_owned())
    }

    fn rule_id(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.names.get(name) {
            return id;
        }
        let id = self.new_rule(name.to_string());
        self.names.insert(name.to_string(), id);
        id
    }

    fn new_rule(&mut self, name: String) -> u32 {
        self.rules.push(None);
        self.rule_names.push(name);
        (self.rules.len() - 1) as u32
    }

    fn charge(&mut self, elements: usize) -> Result<()> {
        self.elements += elements;
        if self.elements > MAX_ELEMENTS {
            return self.fail("the grammar expands to too many elements");
        }
        Ok(())
    }

    fn grammar(&mut self) -> Result<()> {
        self.space(true);
        while self.peek().is_some() {
            let name = self.name()?;
            self.space(false);
            if !self.text[self.at..].starts_with(b"::=") {
                return self.fail("expecting `::=`");
            }
            self.at += 3;
            self.space(true);
            let rule = self.alternatives(false)?;
            let id = self.rule_id(&name);
            self.rules[id as usize] = Some(rule);
            match self.peek() {
                Some(b'\r' | b'\n') | None => {}
                Some(_) => return self.fail("expecting the end of the rule"),
            }
            self.space(true);
        }
        Ok(())
    }

    fn alternatives(&mut self, nested: bool) -> Result<Vec<Sequence>> {
        let mut rule = vec![self.sequence(nested)?];
        while self.peek() == Some(b'|') {
            self.at += 1;
            self.space(true);
            rule.push(self.sequence(nested)?);
        }
        Ok(rule)
    }

    fn sequence(&mut self, nested: bool) -> Result<Sequence> {
        let mut sequence: Sequence = Vec::new();
        // Where the last item starts: what a repetition applies to.
        let mut last = 0;
        while let Some(byte) = self.peek() {
            match byte {
                b'"' => {
                    self.at += 1;
                    last = sequence.len();
                    while self.peek() != Some(b'"') {
                        if self.peek().is_none() {
                            return self.fail("a string is not closed");
                        }
                        sequence.push(Elem::Char(self.character()?));
                    }
                    self.at += 1;
                    self.charge(sequence.len() - last)?;
                }
                b'[' => {
                    self.at += 1;
                    last = sequence.len();
                    sequence.push(self.class()?);
                    self.charge(1)?;
                }
                b'.' => {
                    self.at += 1;
                    last = sequence.len();
                    let id = self.classes.len() as u32;
                    self.classes
                        .push(CharClass::new(vec![(0, 0x10_FFFF)], false));
                    sequence.push(Elem::Class(id));
                    self.charge(1)?;
                }
                b'(' => {
                    self.at += 1;
                    self.depth += 1;
                    if self.depth > MAX_GROUP_DEPTH {
                        return self.fail("groups are nested too deeply");
                    }
                    self.space(true);
                    let group = self.alternatives(true)?;
                    self.depth -= 1;
                    if self.peek() != Some(b')') {
                        return self.fail("expecting `)`");
                    }
                    self.at += 1;
                    let id = self.new_rule(String::new());
                    self.rules[id as usize] = Some(group);
                    last = sequence.len();
                    sequence.push(Elem::Rule(id));
                    self.charge(1)?;
                }
                b'*' | b'+' | b'?' | b'{' => {
                    if last >= sequence.len() {
                        return self.fail("a repetition needs something to repeat");
                    }
                    let (min, max) = self.repetition()?;
                    let item = sequence.split_off(last);
                    self.repeat(&mut sequence, item, min, max)?;
                    // A repetition of a repetition applies to what it made.
                    last = last.min(sequence.len());
                }
                b'<' | b'!' => {
                    return self.fail(
                        "token references are not supported: a grammar is over a reply's text",
                    );
                }
                b'|' | b')' | b'\r' | b'\n' => break,
                _ if byte.is_ascii_alphanumeric() || byte == b'-' || byte == b'_' => {
                    let start = self.at;
                    let name = self.name()?;
                    // A name followed by `::=` starts the next rule, whether or
                    // not a newline came first.
                    let after_name = self.at;
                    self.space(true);
                    if self.text[self.at..].starts_with(b"::=") {
                        self.at = start;
                        return self.fail("expecting the end of the rule before the next one");
                    }
                    self.at = after_name;
                    last = sequence.len();
                    let id = self.rule_id(&name);
                    sequence.push(Elem::Rule(id));
                    self.charge(1)?;
                }
                _ => return self.fail("unexpected character"),
            }
            self.space(nested);
        }
        Ok(sequence)
    }

    fn repetition(&mut self) -> Result<(usize, Option<usize>)> {
        let byte = self.text[self.at];
        self.at += 1;
        match byte {
            b'*' => Ok((0, None)),
            b'+' => Ok((1, None)),
            b'?' => Ok((0, Some(1))),
            _ => {
                self.space(false);
                let min = self.count()?;
                self.space(false);
                let bounds = match self.peek() {
                    Some(b'}') => (min, Some(min)),
                    Some(b',') => {
                        self.at += 1;
                        self.space(false);
                        if self.peek() == Some(b'}') {
                            (min, None)
                        } else {
                            let max = self.count()?;
                            self.space(false);
                            if max < min {
                                return self.fail("a repetition's bounds are the wrong way round");
                            }
                            (min, Some(max))
                        }
                    }
                    _ => return self.fail("expecting `,` or `}`"),
                };
                if self.peek() != Some(b'}') {
                    return self.fail("expecting `}`");
                }
                self.at += 1;
                Ok(bounds)
            }
        }
    }

    fn count(&mut self) -> Result<usize> {
        let start = self.at;
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.at += 1;
        }
        let digits = std::str::from_utf8(&self.text[start..self.at]).unwrap_or("");
        match digits.parse::<usize>() {
            Ok(count) if count <= MAX_REPETITION => Ok(count),
            Ok(_) | Err(_) if !digits.is_empty() => {
                self.fail(&format!("a repetition is limited to {MAX_REPETITION}"))
            }
            _ => self.fail("expecting a number"),
        }
    }

    /// Appends `item` repeated: `min` copies, then a rule for what may follow.
    fn repeat(
        &mut self,
        sequence: &mut Sequence,
        item: Sequence,
        min: usize,
        max: Option<usize>,
    ) -> Result<()> {
        self.charge(item.len().saturating_mul(min + 1))?;
        for _ in 0..min {
            sequence.extend_from_slice(&item);
        }
        match max {
            None => {
                // star ::= item star |
                let star = self.new_rule(String::new());
                let mut body = item;
                body.push(Elem::Rule(star));
                self.charge(body.len())?;
                self.rules[star as usize] = Some(vec![body, Vec::new()]);
                sequence.push(Elem::Rule(star));
            }
            Some(max) => {
                // optional_n ::= item optional_n-1 |
                let mut next: Option<u32> = None;
                for _ in min..max {
                    let optional = self.new_rule(String::new());
                    let mut body = item.clone();
                    body.extend(next.map(Elem::Rule));
                    self.charge(body.len())?;
                    self.rules[optional as usize] = Some(vec![body, Vec::new()]);
                    next = Some(optional);
                }
                sequence.extend(next.map(Elem::Rule));
            }
        }
        Ok(())
    }

    /// One character of a string or class, escape or not.
    fn character(&mut self) -> Result<u32> {
        let Some(byte) = self.peek() else {
            return self.fail("unexpected end of the grammar");
        };
        if byte == b'\\' {
            self.at += 1;
            let Some(escape) = self.peek() else {
                return self.fail("unexpected end of the grammar");
            };
            self.at += 1;
            return match escape {
                b'x' => self.hex(2),
                b'u' => self.hex(4),
                b'U' => self.hex(8),
                b't' => Ok(u32::from(b'\t')),
                b'r' => Ok(u32::from(b'\r')),
                b'n' => Ok(u32::from(b'\n')),
                b'\\' | b'"' | b'[' | b']' | b'-' | b'^' => Ok(u32::from(escape)),
                _ => self.fail("unknown escape"),
            };
        }
        // The text came from a `str`, so this is a whole character.
        let rest = std::str::from_utf8(&self.text[self.at..]).unwrap_or_default();
        let Some(c) = rest.chars().next() else {
            return self.fail("expecting a character");
        };
        self.at += c.len_utf8();
        Ok(c as u32)
    }

    fn hex(&mut self, digits: usize) -> Result<u32> {
        let end = self.at + digits;
        let parsed = self
            .text
            .get(self.at..end)
            .and_then(|hex| std::str::from_utf8(hex).ok())
            .and_then(|hex| u32::from_str_radix(hex, 16).ok())
            .filter(|&cp| cp <= 0x10_FFFF);
        let Some(cp) = parsed else {
            return self.fail(&format!("expecting {digits} hex digits of a character"));
        };
        self.at = end;
        Ok(cp)
    }

    /// A class, after its `[`.
    fn class(&mut self) -> Result<Elem> {
        let negated = self.peek() == Some(b'^');
        if negated {
            self.at += 1;
        }
        let mut ranges = Vec::new();
        while self.peek() != Some(b']') {
            if self.peek().is_none() {
                return self.fail("a character class is not closed");
            }
            let low = self.character()?;
            let mut high = low;
            if self.peek() == Some(b'-') && self.text.get(self.at + 1) != Some(&b']') {
                self.at += 1;
                high = self.character()?;
                if high < low {
                    return self.fail("a character range is the wrong way round");
                }
            }
            ranges.push((low, high));
        }
        self.at += 1;
        if !negated
            && let [(low, high)] = ranges.as_slice()
            && low == high
        {
            return Ok(Elem::Char(*low));
        }
        let id = self.classes.len() as u32;
        self.classes.push(CharClass::new(ranges, negated));
        Ok(Elem::Class(id))
    }
}

const NO_PARENT: u32 = u32::MAX;

/// A way of having read the input: the character element next to match, and
/// the stack to return to after its rule.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct Head {
    at: u32,
    stack: u32,
}

/// A character partly read: what it can still turn out to be.
#[derive(Clone, Copy, Debug, Default)]
struct Partial {
    /// The bytes still to come, or 0 when no character is open.
    left: u8,
    value: u32,
    /// The smallest code point of this length, for refusing overlong forms.
    least: u32,
}

/// Follows a grammar through a reply. Cheap to probe: [`Self::allows`] leaves
/// the state as it found it.
#[derive(Clone, Debug)]
pub struct Matcher {
    grammar: Arc<Grammar>,
    heads: Vec<Head>,
    accepting: bool,
    partial: Partial,
    /// Stack frames, each a place to return to and the frame under it.
    frames: Vec<(u32, u32)>,
    frame_ids: FxHashMap<(u32, u32), u32>,
    first_bytes: [u64; 4],
}

impl Matcher {
    pub fn new(grammar: Arc<Grammar>) -> Self {
        let mut matcher = Self {
            heads: Vec::new(),
            accepting: false,
            partial: Partial::default(),
            frames: Vec::new(),
            frame_ids: FxHashMap::default(),
            first_bytes: [0; 4],
            grammar,
        };
        let starts = matcher.grammar.alternatives[matcher.grammar.root as usize].clone();
        let seeds: Vec<Head> = starts
            .into_iter()
            .map(|at| Head {
                at,
                stack: NO_PARENT,
            })
            .collect();
        matcher.close(seeds);
        matcher
    }

    /// A frame's number, the same for the same place and stack.
    fn frame(&mut self, at: u32, under: u32) -> u32 {
        // A call in tail position has nothing to come back for.
        if matches!(self.grammar.elements[at as usize], Elem::End | Elem::Alt) {
            return under;
        }
        if let Some(&id) = self.frame_ids.get(&(at, under)) {
            return id;
        }
        let id = self.frames.len() as u32;
        self.frames.push((at, under));
        self.frame_ids.insert((at, under), id);
        id
    }

    /// Expands `seeds` to the character elements they can reach, without
    /// reading anything, and makes those the state.
    fn close(&mut self, seeds: Vec<Head>) {
        let grammar = self.grammar.clone();
        let mut work = seeds;
        let mut seen: FxHashSet<Head> = FxHashSet::default();
        let mut heads = Vec::new();
        let mut accepting = false;
        let mut first_bytes = [0u64; 4];
        while let Some(head) = work.pop() {
            if !seen.insert(head) {
                continue;
            }
            match grammar.elements[head.at as usize] {
                Elem::End | Elem::Alt => {
                    if head.stack == NO_PARENT {
                        accepting = true;
                    } else {
                        let (at, stack) = self.frames[head.stack as usize];
                        work.push(Head { at, stack });
                    }
                }
                Elem::Rule(rule) => {
                    let stack = self.frame(head.at + 1, head.stack);
                    for &at in &grammar.alternatives[rule as usize] {
                        work.push(Head { at, stack });
                    }
                }
                Elem::Char(cp) => {
                    let byte = lead_byte(cp);
                    first_bytes[(byte / 64) as usize] |= 1 << (byte % 64);
                    heads.push(head);
                }
                Elem::Class(class) => {
                    let table = &grammar.classes[class as usize].first_bytes;
                    for (all, some) in first_bytes.iter_mut().zip(table) {
                        *all |= some;
                    }
                    heads.push(head);
                }
            }
            if heads.len() >= MAX_HEADS {
                break;
            }
        }
        self.heads = heads;
        self.accepting = accepting;
        self.first_bytes = first_bytes;
    }

    fn step(&mut self, cp: u32) -> bool {
        let grammar = self.grammar.clone();
        let next: Vec<Head> = self
            .heads
            .iter()
            .filter(|head| match grammar.elements[head.at as usize] {
                Elem::Char(c) => c == cp,
                Elem::Class(class) => grammar.classes[class as usize].contains(cp),
                _ => false,
            })
            .map(|head| Head {
                at: head.at + 1,
                stack: head.stack,
            })
            .collect();
        if next.is_empty() {
            self.heads.clear();
            self.accepting = false;
            self.first_bytes = [0; 4];
            return false;
        }
        self.close(next);
        !self.heads.is_empty() || self.accepting
    }

    /// Whether any head can still match a character from `low` to `high`.
    fn can_match(&self, low: u32, high: u32) -> bool {
        self.heads
            .iter()
            .any(|head| match self.grammar.elements[head.at as usize] {
                Elem::Char(c) => low <= c && c <= high,
                Elem::Class(class) => self.grammar.classes[class as usize].intersects(low, high),
                _ => false,
            })
    }

    fn push_byte(&mut self, byte: u8) -> bool {
        if self.partial.left > 0 {
            if byte & 0xC0 != 0x80 {
                return false;
            }
            self.partial.left -= 1;
            self.partial.value = (self.partial.value << 6) | u32::from(byte & 0x3F);
            let left = u32::from(self.partial.left) * 6;
            let low = self.partial.value << left;
            let high = low | ((1 << left) - 1);
            if self.partial.left > 0 {
                return self.can_match(low.max(self.partial.least), high.min(0x10_FFFF));
            }
            let cp = self.partial.value;
            let surrogate = (0xD800..=0xDFFF).contains(&cp);
            if cp < self.partial.least || cp > 0x10_FFFF || surrogate {
                return false;
            }
            return self.step(cp);
        }
        if self.first_bytes[(byte / 64) as usize] & (1 << (byte % 64)) == 0 {
            return false;
        }
        let (left, value, least) = match byte {
            0..=0x7F => return self.step(u32::from(byte)),
            0xC2..=0xDF => (1, byte & 0x1F, 0x80),
            0xE0..=0xEF => (2, byte & 0x0F, 0x800),
            0xF0..=0xF4 => (3, byte & 0x07, 0x1_0000),
            _ => return false,
        };
        self.partial = Partial {
            left,
            value: u32::from(value),
            least,
        };
        true
    }

    /// Feeds `bytes`, returning `false` at the first that cannot come next.
    /// `self` is only usable afterwards if this returned `true`.
    pub fn push_bytes(&mut self, bytes: &[u8]) -> bool {
        bytes.iter().all(|&byte| self.push_byte(byte))
    }

    /// Whether `bytes` could come next, leaving the state as it was.
    pub fn allows(&mut self, bytes: &[u8]) -> bool {
        let Some(&first) = bytes.first() else {
            return true;
        };
        if self.partial.left == 0
            && self.first_bytes[(first / 64) as usize] & (1 << (first % 64)) == 0
        {
            return false;
        }
        // One character needs no new frames unless it closes into a call, and
        // the frames a probe adds are only extra entries, so this keeps them:
        // they are the same ones the token, if chosen, would add.
        let (heads, accepting, partial, first_bytes) = (
            self.heads.clone(),
            self.accepting,
            self.partial,
            self.first_bytes,
        );
        let allowed = self.push_bytes(bytes);
        self.heads = heads;
        self.accepting = accepting;
        self.partial = partial;
        self.first_bytes = first_bytes;
        allowed
    }

    /// Whether what has been read is a whole sentence of the grammar.
    pub fn is_complete(&self) -> bool {
        self.accepting && self.partial.left == 0
    }

    /// Whether anything more can be read.
    pub fn can_continue(&self) -> bool {
        !self.heads.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matches(grammar: &str, text: &str) -> bool {
        let mut matcher = Matcher::new(Grammar::parse(grammar).unwrap());
        matcher.push_bytes(text.as_bytes()) && matcher.is_complete()
    }

    #[test]
    fn literals_classes_and_alternatives() {
        let g = "root ::= \"a\" [b-d] (\"x\" | \"yz\")\n";
        assert!(matches(g, "abx"));
        assert!(matches(g, "adyz"));
        assert!(!matches(g, "aex"));
        assert!(!matches(g, "ab"));
        assert!(!matches(g, "abxx"));
    }

    /// Nothing read and nothing to come back for: this only comes back to where
    /// it was, and is not the recursion that has to be refused.
    #[test]
    fn a_repetition_of_what_can_be_empty_is_only_a_repetition() {
        let g = "root ::= ([x]*)*\n";
        assert!(matches(g, ""));
        assert!(matches(g, "xxx"));
        assert!(!matches(g, "y"));
        assert!(matches("root ::= o root | \"z\"\no ::= \"a\"?\n", "aaz"));
    }

    #[test]
    fn repetitions_count_what_they_repeat() {
        for (g, good, bad) in [
            ("root ::= \"a\"*", vec!["", "a", "aaaa"], vec!["b"]),
            ("root ::= \"a\"+", vec!["a", "aaa"], vec![""]),
            ("root ::= \"a\"?", vec!["", "a"], vec!["aa"]),
            ("root ::= \"ab\"{2}", vec!["abab"], vec!["ab", "ababab"]),
            ("root ::= [a]{1,3}", vec!["a", "aaa"], vec!["", "aaaa"]),
            ("root ::= [a]{2,}", vec!["aa", "aaaaa"], vec!["a"]),
            ("root ::= [a]{0,2}", vec!["", "aa"], vec!["aaa"]),
            (
                "root ::= (\"a\" | \"b\"){2}",
                vec!["ab", "ba"],
                vec!["a", "abb"],
            ),
        ] {
            for text in good {
                assert!(matches(g, text), "{g} should take {text:?}");
            }
            for text in bad {
                assert!(!matches(g, text), "{g} should refuse {text:?}");
            }
        }
    }

    #[test]
    fn rules_may_recurse_on_the_right_and_call_each_other() {
        let g = "root ::= item (\",\" item)*\nitem ::= \"[\" root? \"]\" | [0-9]+\n";
        assert!(matches(g, "1,22,[3,[4]],[]"));
        assert!(!matches(g, "1,"));
        assert!(!matches(g, "[1"));
    }

    #[test]
    fn comments_and_newlines_mean_nothing_where_they_may_appear() {
        let g =
            "# a comment\nroot ::= ( # inside\n  \"a\" |\n  \"b\"\n) c\n\nc ::=\n  \"c\" # end\n";
        assert!(matches(g, "ac"));
        assert!(matches(g, "bc"));
    }

    #[test]
    fn characters_past_ascii_match_whole_and_may_arrive_in_pieces() {
        let g = "root ::= [α-ω]+ \"é\" [^a]\n";
        let mut matcher = Matcher::new(Grammar::parse(g).unwrap());
        let text = "αβéb".as_bytes();
        for byte in text {
            assert!(matcher.allows(std::slice::from_ref(byte)));
            assert!(!matcher.is_complete());
            assert!(matcher.push_bytes(std::slice::from_ref(byte)));
        }
        assert!(matcher.is_complete());
        // The first byte of an impossible character is refused at once.
        let mut matcher = Matcher::new(Grammar::parse("root ::= [α-ω]\n").unwrap());
        assert!(!matcher.allows("a".as_bytes()));
        assert!(!matcher.allows(&"é".as_bytes()[..1]));
        assert!(matcher.allows(&"β".as_bytes()[..1]));
        // Bytes that are not UTF-8 are not a character of any class.
        let mut any = Matcher::new(Grammar::parse("root ::= .*\n").unwrap());
        assert!(!any.allows(&[0xFF]));
        assert!(!any.allows(&[0xC0, 0x80]));
        assert!(!any.allows(&[0xED, 0xA0, 0x80]));
        assert!(any.allows("\u{1F600}".as_bytes()));
    }

    #[test]
    fn escapes_read_as_the_characters_they_name() {
        let g = r#"root ::= "\x41\u00e9\U0001F600\t\n\"\\" [\]\-\^]
"#;
        assert!(matches(g, "Aé\u{1F600}\t\n\"\\]"));
        assert!(matches(g, "Aé\u{1F600}\t\n\"\\-"));
    }

    #[test]
    fn what_is_not_a_grammar_is_refused_with_a_reason() {
        for (g, says) in [
            ("a ::= \"x\"\n", "root"),
            ("root ::= missing\n", "not defined"),
            ("root ::= root \"a\"\n", "left-recursive"),
            ("root ::= a\na ::= b\nb ::= a \"x\"\n", "left-recursive"),
            ("root ::= o root \"x\"\no ::= \"a\"?\n", "left-recursive"),
            ("root ::= \"a", "not closed"),
            ("root ::= [a", "not closed"),
            ("root ::= (\"a\"", "`)`"),
            ("root ::= *", "repeat"),
            ("root ::= [z-a]", "wrong way"),
            ("root ::= \"a\"{3,2}", "wrong way"),
            ("root ::= \"a\"{999999}", "limited"),
            ("root ::= <[1000]>", "token references"),
            ("root ::= !<think>", "token references"),
            ("root = \"a\"", "::="),
            ("root ::= \"\\q\"", "escape"),
        ] {
            let error = Grammar::parse(g).unwrap_err().to_string();
            assert!(error.contains(says), "{g:?} said {error:?}");
        }
    }

    /// What a hostile grammar can reach for, all at once: every one has to come
    /// back quickly, with an error or a matcher.
    #[test]
    fn expansion_ambiguity_and_depth_stay_bounded() {
        let started = std::time::Instant::now();
        // Repetitions of repetitions.
        let nested = format!(
            "root ::= {}\"a\"{}\n",
            "(".repeat(40),
            "){16000}".repeat(40)
        );
        let _ = Grammar::parse(&nested);
        // Groups past the limit.
        let deep = format!("root ::= {}\"a\"{}\n", "(".repeat(5000), ")".repeat(5000));
        assert!(Grammar::parse(&deep).is_err());
        // A chain of rules that match nothing, deeper than any stack.
        let mut chain = String::from("root ::= r0\n");
        for i in 0..50_000 {
            chain.push_str(&format!("r{i} ::= r{}?\n", i + 1));
        }
        chain.push_str("r50000 ::= \"a\"\n");
        assert!(matches(&chain, "a"));
        // Exponentially many ways to read the same text.
        assert!(matches(
            "root ::= (\"a\" | \"a\" | \"a\"){0,64}\n",
            &"a".repeat(64)
        ));
        let ambiguous = "root ::= (\"a\" | \"aa\" | \"aaa\" | root \"\" )*\n";
        let _ = Grammar::parse(ambiguous);
        let split = "root ::= x* \"b\"\nx ::= \"a\" | \"a\" \"a\" | \"a\" \"a\" \"a\"\n";
        assert!(matches(split, &format!("{}b", "a".repeat(4000))));
        // Many genuinely different stacks, up to the bound.
        let wide = "root ::= \"a\" root \"b\" | \"a\" root \"c\" | \"\"\n";
        let mut matcher = Matcher::new(Grammar::parse(wide).unwrap());
        assert!(matcher.push_bytes(&"a".repeat(2000).into_bytes()));
        assert!(started.elapsed().as_secs() < 60, "{:?}", started.elapsed());
    }

    #[test]
    fn a_long_repetition_reads_in_constant_space() {
        let grammar = Grammar::parse("root ::= ([a-z] | \" \")*\n").unwrap();
        let mut matcher = Matcher::new(grammar);
        assert!(matcher.push_bytes(&b"lorem ipsum ".repeat(5000)));
        assert!(matcher.is_complete());
        assert!(matcher.frames.len() < 16, "{} frames", matcher.frames.len());
    }

    /// Asking about a token changes nothing, whether the answer is yes or no.
    #[test]
    fn a_probe_leaves_the_state_alone() {
        let g = "root ::= \"ab\" [c-d]* \"e\"\n";
        let mut matcher = Matcher::new(Grammar::parse(g).unwrap());
        assert!(matcher.allows(b"abcc"));
        assert!(!matcher.allows(b"abx"));
        assert!(matcher.allows(b"ab"));
        assert!(matcher.push_bytes(b"ab"));
        assert!(!matcher.allows(b"x"));
        assert!(matcher.allows(b"e"));
        assert!(!matcher.is_complete());
        assert!(matcher.push_bytes(b"e"));
        assert!(matcher.is_complete());
        assert!(!matcher.can_continue());
    }
}
