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

//! The value checks of a JSON Schema, written as GBNF: numeric ranges, the
//! common string formats, and `pattern`.

/// A JSON string's text as GBNF: a character that needs no escape, or an escape.
pub const JSON_CHAR_RULES: &str = "char ::= [^\"\\\\\\x00-\\x1F] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\" hex hex hex hex)\n\
                                   hex ::= [0-9a-fA-F]\n";

/// A GBNF string for `text`.
pub fn quoted(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 || c as u32 == 0x7F => {
                out.push_str(&format!("\\x{:02X}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// GBNF for the numbers from `low` to `high`, both the same length in digits.
fn digits_between(low: &[u8], high: &[u8], out: &mut Vec<String>, prefix: &str) {
    let Some((&first_low, rest_low)) = low.split_first() else {
        out.push(prefix.trim_end().to_string());
        return;
    };
    let (&first_high, rest_high) = high.split_first().expect("the same length");
    let free = |n: usize| match n {
        0 => String::new(),
        n => format!(" [0-9]{{{n}}}"),
    };
    let class = |a: u8, b: u8| {
        if a == b {
            format!("\"{}\"", a as char)
        } else {
            format!("[{}-{}]", a as char, b as char)
        }
    };
    let n = rest_low.len();
    let low_full = rest_low.iter().all(|&d| d == b'0');
    let high_full = rest_high.iter().all(|&d| d == b'9');
    if first_low == first_high {
        let next = format!("{prefix}{} ", class(first_low, first_low));
        if low_full && high_full {
            out.push(format!("{}{}", next.trim_end(), free(n)));
        } else {
            digits_between(rest_low, rest_high, out, &next);
        }
        return;
    }
    // The first digit's low end, whole digits between, and its high end.
    let mut middle_low = first_low;
    if !low_full {
        let nines = vec![b'9'; n];
        let next = format!("{prefix}{} ", class(first_low, first_low));
        digits_between(rest_low, &nines, out, &next);
        middle_low += 1;
    }
    let mut middle_high = first_high;
    if !high_full {
        middle_high -= 1;
    }
    if middle_low <= middle_high {
        out.push(format!(
            "{prefix}{}{}",
            class(middle_low, middle_high),
            free(n)
        ));
    }
    if !high_full {
        let zeros = vec![b'0'; n];
        let next = format!("{prefix}{} ", class(first_high, first_high));
        digits_between(&zeros, rest_high, out, &next);
    }
}

/// GBNF alternatives for the naturals from `low` to `high`, no leading zeros.
fn natural_range(low: u128, high: Option<u128>) -> Vec<String> {
    let mut out = Vec::new();
    let low_text = low.to_string();
    let high_text = high.map(|h| h.to_string());
    let last_length = high_text.as_ref().map_or(low_text.len(), String::len);
    for length in low_text.len()..=last_length {
        let start = if length == low_text.len() {
            low_text.clone()
        } else {
            format!("1{}", "0".repeat(length - 1))
        };
        let end = match &high_text {
            Some(text) if text.len() == length => text.clone(),
            _ => "9".repeat(length),
        };
        digits_between(start.as_bytes(), end.as_bytes(), &mut out, "");
    }
    if high.is_none() {
        out.push(format!("[1-9] [0-9]{{{},}}", low_text.len()));
    }
    out
}

/// GBNF for the integers from `low` to `high`, either side open.
pub fn integer_range(low: Option<i128>, high: Option<i128>) -> Option<String> {
    let mut alternatives = Vec::new();
    if high.is_none_or(|h| h >= 0) {
        let from = low.map_or(0, |l| l.max(0)) as u128;
        alternatives.extend(natural_range(from, high.map(|h| h as u128)));
    }
    if low.is_none_or(|l| l < 0) {
        let from = high.map_or(1, |h| h.min(-1).unsigned_abs());
        for alternative in natural_range(from, low.map(i128::unsigned_abs)) {
            alternatives.push(format!("\"-\" {alternative}"));
        }
    }
    (!alternatives.is_empty()).then(|| alternatives.join(" | "))
}

/// A bound: a whole number, and whether the bound itself is left out.
pub type Bound = (i128, bool);

const FRACTION: &str = "(\".\" [0-9]+)?";
const ZERO_FRACTION: &str = "(\".\" \"0\"+)?";
const NONZERO_FRACTION: &str = "\".\" [0-9]* [1-9] [0-9]*";

/// Alternatives for magnitudes from `low` to `high`, fractions included.
fn magnitudes(low: Bound, high: Option<Bound>) -> Vec<String> {
    let mut out = Vec::new();
    let (low_value, low_strict) = low;
    let first_whole = low_value + i128::from(low_strict);
    let last_whole = high.map(|(value, _)| value - 1);
    if last_whole.is_none_or(|last| first_whole <= last) {
        for alternative in natural_range(first_whole as u128, last_whole.map(|l| l as u128)) {
            out.push(format!("({alternative}) {FRACTION}"));
        }
    }
    let below_high = high.is_none_or(|(value, _)| low_value < value);
    if low_strict && below_high {
        out.push(format!("\"{low_value}\" {NONZERO_FRACTION}"));
    }
    if let Some((value, strict)) = high
        && !strict
        && (value > low_value || (value == low_value && !low_strict))
    {
        out.push(format!("\"{value}\" {ZERO_FRACTION}"));
    }
    out
}

/// GBNF for the decimals between two whole bounds, either side open. No
/// exponents: a range is only checked as the digits go by.
pub fn number_range(low: Option<Bound>, high: Option<Bound>) -> Option<String> {
    let mut alternatives = Vec::new();
    // Zero and up.
    if high.is_none_or(|(value, strict)| value > 0 || (value == 0 && !strict)) {
        let from = match low {
            Some((value, strict)) if value >= 0 => (value, strict),
            _ => (0, false),
        };
        alternatives.extend(magnitudes(from, high.filter(|(value, _)| *value >= 0)));
    }
    // Below zero, as magnitudes: the bounds change places.
    if low.is_none_or(|(value, _)| value < 0) {
        let from = match high {
            Some((value, strict)) if value < 0 => (-value, strict),
            _ => (0, true),
        };
        let to = low.map(|(value, strict)| (-value, strict));
        for alternative in magnitudes(from, to) {
            alternatives.push(format!("\"-\" {alternative}"));
        }
    }
    (!alternatives.is_empty()).then(|| alternatives.join(" | "))
}

/// A string `format` that is checked, as GBNF for what goes between the quotes.
/// Any other `format` is an annotation, as the JSON Schema specification has it.
pub fn format_body(format: &str) -> Option<&'static str> {
    const DATE: &str =
        "[0-9]{4} \"-\" (\"0\" [1-9] | \"1\" [0-2]) \"-\" (\"0\" [1-9] | [12] [0-9] | \"3\" [01])";
    const TIME: &str = "([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] (\".\" [0-9]+)? (\"Z\" | \"z\" | [+-] ([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9])";
    match format {
        "date" => Some(DATE),
        "time" => Some(TIME),
        "date-time" => Some(
            "[0-9]{4} \"-\" (\"0\" [1-9] | \"1\" [0-2]) \"-\" (\"0\" [1-9] | [12] [0-9] | \"3\" [01]) [Tt] ([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9] \":\" [0-5] [0-9] (\".\" [0-9]+)? (\"Z\" | \"z\" | [+-] ([01] [0-9] | \"2\" [0-3]) \":\" [0-5] [0-9])",
        ),
        "uuid" => Some("hex{8} \"-\" hex{4} \"-\" hex{4} \"-\" hex{4} \"-\" hex{12}"),
        "ipv4" => Some("octet \".\" octet \".\" octet \".\" octet"),
        _ => None,
    }
}

/// What [`format_body`] calls beyond a JSON string's own rules.
pub const FORMAT_RULES: &str =
    "octet ::= [0-9] | [1-9] [0-9] | \"1\" [0-9] [0-9] | \"2\" [0-4] [0-9] | \"25\" [0-5]\n";

/// What a pattern's character class holds, as ranges of characters.
struct Class {
    ranges: Vec<(u32, u32)>,
    negated: bool,
}

const DIGITS: &[(u32, u32)] = &[(0x30, 0x39)];
const WORD: &[(u32, u32)] = &[(0x30, 0x39), (0x41, 0x5A), (0x5F, 0x5F), (0x61, 0x7A)];
const SPACE: &[(u32, u32)] = &[(0x09, 0x0D), (0x20, 0x20)];

impl Class {
    /// GBNF for one character of the class, as JSON writes it in a string: what
    /// needs an escape can only appear as its escape.
    fn body(&self) -> String {
        fn escape(cp: u32) -> String {
            match char::from_u32(cp) {
                Some(c) if !matches!(c, '\\' | ']' | '[' | '^' | '-' | '"') && cp >= 0x20 => {
                    c.to_string()
                }
                _ if cp <= 0xFF => format!("\\x{cp:02X}"),
                _ if cp <= 0xFFFF => format!("\\u{cp:04X}"),
                _ => format!("\\U{cp:08X}"),
            }
        }
        // The escaped forms of what is in the class, and the rest as a class.
        let holds = |cp: u32| {
            let inside = self.ranges.iter().any(|&(l, h)| l <= cp && cp <= h);
            inside != self.negated
        };
        let mut alternatives = Vec::new();
        for (cp, text) in [
            (0x22, "\\\""),
            (0x5C, "\\\\"),
            (0x0A, "\\n"),
            (0x0D, "\\r"),
            (0x09, "\\t"),
            (0x08, "\\b"),
            (0x0C, "\\f"),
        ] {
            if holds(cp) {
                alternatives.push(quoted(text));
            }
        }
        let mut class = String::from("[");
        if self.negated {
            class.push('^');
        }
        if self.negated {
            class.push_str("\\x00-\\x1F\\x22\\x5C");
            for &(low, high) in &self.ranges {
                class.push_str(&format!("{}-{}", escape(low), escape(high)));
            }
        } else {
            for &(low, high) in &self.ranges {
                // What only JSON's escapes can write is left to them above.
                let mut from = low.max(0x20);
                let to = high;
                for gap in [0x22, 0x5C] {
                    if from <= gap && gap <= to {
                        if from < gap {
                            class.push_str(&format!("{}-{}", escape(from), escape(gap - 1)));
                        }
                        from = gap + 1;
                    }
                }
                if from <= to {
                    class.push_str(&format!("{}-{}", escape(from), escape(to)));
                }
            }
        }
        class.push(']');
        if class != "[]" {
            alternatives.push(class);
        }
        if alternatives.len() == 1 {
            alternatives.remove(0)
        } else {
            format!("({})", alternatives.join(" | "))
        }
    }
}

struct PatternParser<'a> {
    pattern: &'a str,
    chars: Vec<char>,
    at: usize,
    depth: usize,
}

const MAX_PATTERN_DEPTH: usize = 32;

impl PatternParser<'_> {
    fn fail<T>(&self, why: &str) -> Result<T, String> {
        Err(format!(
            "`pattern` \"{}\" cannot be enforced: {why}",
            self.pattern
        ))
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.at).copied()
    }

    fn alternatives(&mut self) -> Result<String, String> {
        let mut out = vec![self.sequence()?];
        while self.peek() == Some('|') {
            self.at += 1;
            out.push(self.sequence()?);
        }
        Ok(out.join(" | "))
    }

    fn sequence(&mut self) -> Result<String, String> {
        let mut items: Vec<String> = Vec::new();
        while let Some(c) = self.peek() {
            let item = match c {
                '|' | ')' => break,
                '^' | '$' => return self.fail("`^` and `$` are only read at its two ends"),
                '(' => {
                    self.at += 1;
                    if self.peek() == Some('?') {
                        if self.chars.get(self.at + 1) != Some(&':') {
                            return self.fail("look-around and named groups are not supported");
                        }
                        self.at += 2;
                    }
                    self.depth += 1;
                    if self.depth > MAX_PATTERN_DEPTH {
                        return self.fail("its groups nest too deeply");
                    }
                    let inner = self.alternatives()?;
                    self.depth -= 1;
                    if self.peek() != Some(')') {
                        return self.fail("a group is not closed");
                    }
                    self.at += 1;
                    format!("({inner})")
                }
                '[' => {
                    self.at += 1;
                    self.class()?.body()
                }
                '.' => {
                    self.at += 1;
                    // Any character but a newline, which only its escape writes.
                    Class {
                        ranges: vec![(0x0A, 0x0A)],
                        negated: true,
                    }
                    .body()
                }
                '\\' => {
                    self.at += 1;
                    self.escape()?
                }
                '*' | '+' | '?' | '{' => return self.fail("a repetition has nothing to repeat"),
                c => {
                    self.at += 1;
                    literal(c)
                }
            };
            items.push(self.repetition(item)?);
        }
        Ok(if items.is_empty() {
            "\"\"".to_string()
        } else {
            items.join(" ")
        })
    }

    fn repetition(&mut self, item: String) -> Result<String, String> {
        let suffix = match self.peek() {
            Some(c @ ('*' | '+' | '?')) => {
                self.at += 1;
                c.to_string()
            }
            Some('{') => {
                let end = self.chars[self.at..].iter().position(|&c| c == '}');
                let Some(end) = end else {
                    return self.fail("a repetition is not closed");
                };
                let inner: String = self.chars[self.at + 1..self.at + end].iter().collect();
                let good = !inner.is_empty()
                    && inner.starts_with(|c: char| c.is_ascii_digit())
                    && inner.chars().all(|c| c.is_ascii_digit() || c == ',');
                if !good {
                    return self.fail("a repetition's bounds are not numbers");
                }
                self.at += end + 1;
                format!("{{{inner}}}")
            }
            _ => return Ok(item),
        };
        // A lazy repetition describes the same strings.
        if self.peek() == Some('?') {
            self.at += 1;
        } else if self.peek() == Some('+') {
            return self.fail("possessive repetitions are not supported");
        }
        Ok(format!("{item}{suffix}"))
    }

    fn escape(&mut self) -> Result<String, String> {
        let Some(c) = self.peek() else {
            return self.fail("it ends in the middle of an escape");
        };
        self.at += 1;
        if let Some((ranges, negated)) = shorthand(c) {
            return Ok(Class {
                ranges: ranges.to_vec(),
                negated,
            }
            .body());
        }
        match self.escaped_char(c)? {
            Some(cp) => {
                let Some(c) = char::from_u32(cp) else {
                    return self.fail("an escape names no character");
                };
                Ok(literal(c))
            }
            None => self.fail("word boundaries and back-references are not supported"),
        }
    }

    /// The character an escape names, or `None` for one that names no character.
    fn escaped_char(&mut self, c: char) -> Result<Option<u32>, String> {
        Ok(Some(match c {
            'n' => 0x0A,
            'r' => 0x0D,
            't' => 0x09,
            'f' => 0x0C,
            'v' => 0x0B,
            'x' | 'u' => {
                let digits = if c == 'x' { 2 } else { 4 };
                let hex: String = self.chars.iter().skip(self.at).take(digits).collect();
                let Ok(cp) = u32::from_str_radix(&hex, 16) else {
                    return self.fail("an escape's digits are not hexadecimal");
                };
                if hex.len() != digits {
                    return self.fail("an escape's digits are not hexadecimal");
                }
                self.at += digits;
                cp
            }
            c if c.is_ascii_alphanumeric() => return Ok(None),
            c => c as u32,
        }))
    }

    /// A class, after its `[`.
    fn class(&mut self) -> Result<Class, String> {
        let negated = self.peek() == Some('^');
        if negated {
            self.at += 1;
        }
        let mut ranges = Vec::new();
        let mut first = true;
        loop {
            let Some(c) = self.peek() else {
                return self.fail("a character class is not closed");
            };
            self.at += 1;
            if c == ']' && !first {
                break;
            }
            first = false;
            let low = if c == '\\' {
                let Some(e) = self.peek() else {
                    return self.fail("it ends in the middle of an escape");
                };
                self.at += 1;
                if let Some((set, false)) = shorthand(e) {
                    ranges.extend_from_slice(set);
                    continue;
                }
                if shorthand(e).is_some() {
                    return self.fail("a negated shorthand inside a class is not supported");
                }
                match self.escaped_char(e)? {
                    Some(cp) => cp,
                    None => return self.fail("an escape in a class names no character"),
                }
            } else {
                c as u32
            };
            let mut high = low;
            if self.peek() == Some('-') && self.chars.get(self.at + 1).is_some_and(|&n| n != ']') {
                self.at += 1;
                let Some(end) = self.peek() else {
                    return self.fail("a character class is not closed");
                };
                self.at += 1;
                high = if end == '\\' {
                    let Some(e) = self.peek() else {
                        return self.fail("it ends in the middle of an escape");
                    };
                    self.at += 1;
                    match self.escaped_char(e)? {
                        Some(cp) => cp,
                        None => return self.fail("an escape in a class names no character"),
                    }
                } else {
                    end as u32
                };
                if high < low {
                    return self.fail("a character range is the wrong way round");
                }
            }
            ranges.push((low, high));
        }
        Ok(Class { ranges, negated })
    }
}

fn shorthand(c: char) -> Option<(&'static [(u32, u32)], bool)> {
    match c {
        'd' => Some((DIGITS, false)),
        'D' => Some((DIGITS, true)),
        'w' => Some((WORD, false)),
        'W' => Some((WORD, true)),
        's' => Some((SPACE, false)),
        'S' => Some((SPACE, true)),
        _ => None,
    }
}

/// GBNF for one character as a JSON string writes it.
fn literal(c: char) -> String {
    let written = match c {
        '"' => "\\\"".to_string(),
        '\\' => "\\\\".to_string(),
        '\n' => "\\n".to_string(),
        '\r' => "\\r".to_string(),
        '\t' => "\\t".to_string(),
        '\u{8}' => "\\b".to_string(),
        '\u{c}' => "\\f".to_string(),
        c if (c as u32) < 0x20 => format!("\\u{:04x}", c as u32),
        c => c.to_string(),
    };
    quoted(&written)
}

/// GBNF for what a `pattern` lets between a string's quotes. A pattern matches
/// anywhere in a string unless it says otherwise, so an end without an anchor
/// takes any characters.
pub fn pattern_body(pattern: &str) -> Result<String, String> {
    let mut chars: Vec<char> = pattern.chars().collect();
    let anchored_start = chars.first() == Some(&'^');
    if anchored_start {
        chars.remove(0);
    }
    let escaped_end = chars
        .iter()
        .rev()
        .skip(1)
        .take_while(|&&c| c == '\\')
        .count()
        % 2
        == 1;
    let anchored_end = chars.last() == Some(&'$') && !escaped_end;
    if anchored_end {
        chars.pop();
    }
    let mut parser = PatternParser {
        pattern,
        chars,
        at: 0,
        depth: 0,
    };
    let body = parser.alternatives()?;
    if parser.at != parser.chars.len() {
        return parser.fail("a `)` closes nothing");
    }
    Ok(format!(
        "{}({body}){}",
        if anchored_start { "" } else { "char* " },
        if anchored_end { "" } else { " char*" },
    ))
}
