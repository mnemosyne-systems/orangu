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

//! Constrained decoding to a JSON Schema: [`SchemaPrefix`] accepts exactly
//! the byte strings that are a prefix of some JSON document the schema
//! describes, and says when what it has is a whole one.
//!
//! # The schema it reads
//!
//! [`compile`] turns a schema into a small graph of [`Node`]s, and the graph
//! into a [grammar](super::grammar), which is what follows the reply:
//!
//! - `type` (one, or a list of them), and the shape each implies:
//!   `properties`/`required`/`additionalProperties` for an object,
//!   `items`/`minItems`/`maxItems` for an array.
//! - `enum` and `const`, matched as their compact JSON text.
//! - `anyOf`/`oneOf` (any branch may match), a one-schema `allOf`, and
//!   `$ref` into the same document (`#/$defs/…`, `#/definitions/…`, `#`),
//!   recursion included.
//! - `minLength`/`maxLength`, `pattern`, and the `format`s
//!   [`schema_text::format_body`] names, for a string. Any other `format` is an
//!   annotation.
//! - `minimum`/`maximum` and their exclusive forms, for a number. A `number`
//!   bounded this way is written without an exponent, and its bounds must be
//!   whole.
//! - `true`, `{}` and a schema with no `type` or shape: any JSON value.
//!
//! An object with `properties` is written **in the order they are
//! declared**: each property may be skipped unless it is `required`, and no
//! key outside them is written. Generating keys in a fixed order is what lets
//! a required property be enforced with a few bytes of state rather than a
//! set of what has been seen; it is also what every other schema-constrained
//! decoder does. An object without `properties` takes any keys, with values
//! as `additionalProperties` says.
//!
//! Annotations such as `description` and `title` do not change what can be
//! written. A keyword that does and cannot be followed (`not`, `if`,
//! `patternProperties`, `multipleOf`, a many-schema `allOf`, …) is refused at
//! compile time rather than silently dropped.

use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::grammar::{Grammar, Matcher};
use super::schema_text;

type NodeId = usize;

/// The most nodes a compiled schema may have, so a request cannot make the
/// server build an arbitrarily large graph.
const MAX_NODES: usize = 4096;

/// The most whitespace bytes in a row. Whitespace keeps the model free to
/// lay a document out, but an unbounded run is how a constrained model can
/// spend its whole budget on newlines.
#[cfg(test)]
const MAX_WHITESPACE: u8 = 32;

/// The most sub-schemas [`validate`] applies to one value, however they nest.
const MAX_CHECK_STEPS: usize = 1 << 20;

#[derive(Debug)]
enum Node {
    /// Any JSON value.
    Any,
    /// `properties` in declared order; no other keys.
    Object {
        props: Vec<Prop>,
    },
    /// Any keys, each value matching the node; `None` allows only `{}`.
    Map {
        values: Option<NodeId>,
    },
    Array {
        items: NodeId,
        min: usize,
        max: Option<usize>,
    },
    String,
    Number {
        integer: bool,
    },
    /// A value given as grammar text.
    Text(String),
    /// The compact JSON texts the value may be.
    Literals(Vec<Vec<u8>>),
    /// Any of these.
    Union(Vec<NodeId>),
}

#[derive(Debug)]
struct Prop {
    /// The key as JSON writes it, quotes included.
    key: Vec<u8>,
    value: NodeId,
    required: bool,
}

/// A schema compiled for [`SchemaPrefix`].
#[derive(Debug)]
pub struct Schema {
    grammar: Arc<Grammar>,
}

/// Compiles a JSON Schema, or says which part of it cannot be followed.
pub fn compile(schema: &Value) -> Result<Schema, String> {
    let (rules, root) = schema_rules(schema, "")?;
    let text = format!("root ::= ws {root}\n{}{rules}", common_rules());
    let grammar =
        Grammar::parse(&text).map_err(|error| format!("the schema is too large: {error}"))?;
    Ok(Schema { grammar })
}

struct Compiler<'a> {
    document: &'a Value,
    nodes: Vec<Node>,
    refs: HashMap<String, NodeId>,
}

/// Keywords that change which documents a schema accepts in ways the
/// recogniser cannot follow.
const UNSUPPORTED: &[&str] = &[
    "not",
    "if",
    "then",
    "else",
    "patternProperties",
    "propertyNames",
    "dependentSchemas",
    "dependentRequired",
    "unevaluatedProperties",
    "unevaluatedItems",
    "contains",
    "prefixItems",
    "$dynamicRef",
    "multipleOf",
    "minProperties",
    "maxProperties",
];

impl Compiler<'_> {
    fn push(&mut self, node: Node) -> Result<NodeId, String> {
        if self.nodes.len() >= MAX_NODES {
            return Err(format!("the schema is larger than {MAX_NODES} nodes"));
        }
        self.nodes.push(node);
        Ok(self.nodes.len() - 1)
    }

    fn node(&mut self, schema: &Value) -> Result<NodeId, String> {
        let object = match schema {
            Value::Bool(true) => return self.push(Node::Any),
            Value::Bool(false) => return Err("a `false` schema accepts nothing".into()),
            Value::Object(object) => object,
            other => return Err(format!("a schema must be an object, not {other}")),
        };
        if let Some(keyword) = UNSUPPORTED.iter().find(|k| object.contains_key(**k)) {
            return Err(format!("`{keyword}` is not supported in a schema"));
        }
        if object
            .get("uniqueItems")
            .is_some_and(|unique| unique != false)
        {
            return Err("`uniqueItems` is not supported in a schema".into());
        }
        if let Some(reference) = object.get("$ref") {
            let reference = reference
                .as_str()
                .ok_or("`$ref` must be a string")?
                .to_string();
            return self.reference(&reference);
        }
        if let Some(value) = object.get("const") {
            return self.push(Node::Literals(vec![literal(value)]));
        }
        if let Some(values) = object.get("enum") {
            let values = values.as_array().ok_or("`enum` must be an array")?;
            if values.is_empty() {
                return Err("an empty `enum` accepts nothing".into());
            }
            return self.push(Node::Literals(values.iter().map(literal).collect()));
        }
        for keyword in ["anyOf", "oneOf"] {
            if let Some(branches) = object.get(keyword) {
                let branches = branches
                    .as_array()
                    .filter(|b| !b.is_empty())
                    .ok_or_else(|| format!("`{keyword}` must be a non-empty array"))?;
                let ids = branches
                    .iter()
                    .map(|b| self.node(b))
                    .collect::<Result<Vec<_>, _>>()?;
                return self.push(Node::Union(ids));
            }
        }
        if let Some(all) = object.get("allOf") {
            return match all.as_array().map(Vec::as_slice) {
                Some([only]) => self.node(only),
                _ => Err("`allOf` is supported with exactly one schema".into()),
            };
        }
        match object.get("type") {
            Some(Value::String(kind)) => self.typed(object, kind),
            Some(Value::Array(kinds)) => {
                let ids = kinds
                    .iter()
                    .map(|k| {
                        let k = k.as_str().ok_or("`type` entries must be strings")?;
                        self.typed(object, k)
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                self.push(Node::Union(ids))
            }
            Some(_) => Err("`type` must be a string or an array of strings".into()),
            None if object.contains_key("properties") => self.typed(object, "object"),
            None if object.contains_key("items") => self.typed(object, "array"),
            // A value check says what kind of value it is about.
            None if ["pattern", "minLength", "maxLength"]
                .iter()
                .any(|key| object.contains_key(*key))
                || object
                    .get("format")
                    .and_then(Value::as_str)
                    .is_some_and(|format| schema_text::format_body(format).is_some()) =>
            {
                self.typed(object, "string")
            }
            None if ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]
                .iter()
                .any(|key| object.contains_key(*key)) =>
            {
                self.typed(object, "number")
            }
            None => self.push(Node::Any),
        }
    }

    fn typed(
        &mut self,
        object: &serde_json::Map<String, Value>,
        kind: &str,
    ) -> Result<NodeId, String> {
        match kind {
            "object" => {
                let required: Vec<&str> = object
                    .get("required")
                    .and_then(Value::as_array)
                    .map(|r| r.iter().filter_map(Value::as_str).collect())
                    .unwrap_or_default();
                match object.get("properties").and_then(Value::as_object) {
                    Some(properties) if !properties.is_empty() => {
                        let mut props = Vec::new();
                        for (name, schema) in properties {
                            props.push(Prop {
                                key: literal(&Value::String(name.clone())),
                                value: self.node(schema)?,
                                required: required.contains(&name.as_str()),
                            });
                        }
                        if let Some(missing) =
                            required.iter().find(|r| !properties.contains_key(**r))
                        {
                            return Err(format!(
                                "`required` names `{missing}`, which `properties` does not declare"
                            ));
                        }
                        self.push(Node::Object { props })
                    }
                    _ => {
                        let values = match object.get("additionalProperties") {
                            Some(Value::Bool(false)) => None,
                            None | Some(Value::Bool(true)) => Some(self.push(Node::Any)?),
                            Some(schema) => Some(self.node(schema)?),
                        };
                        self.push(Node::Map { values })
                    }
                }
            }
            "array" => {
                let items = match object.get("items") {
                    Some(schema) => self.node(schema)?,
                    None => self.push(Node::Any)?,
                };
                let count = |key: &str| object.get(key).and_then(Value::as_u64).map(|n| n as usize);
                let min = count("minItems").unwrap_or(0);
                let max = count("maxItems");
                if max.is_some_and(|max| max < min) {
                    return Err("`maxItems` is smaller than `minItems`".into());
                }
                self.push(Node::Array { items, min, max })
            }
            "string" => match string_body(object)? {
                Some(body) => self.push(Node::Text(body)),
                None => self.push(Node::String),
            },
            "number" => match number_body(object, false)? {
                Some(body) => self.push(Node::Text(body)),
                None => self.push(Node::Number { integer: false }),
            },
            "integer" => match number_body(object, true)? {
                Some(body) => self.push(Node::Text(body)),
                None => self.push(Node::Number { integer: true }),
            },
            "boolean" => self.push(Node::Literals(vec![b"true".to_vec(), b"false".to_vec()])),
            "null" => self.push(Node::Literals(vec![b"null".to_vec()])),
            other => Err(format!("unknown `type` \"{other}\"")),
        }
    }

    /// A `$ref`, compiled once however often it is named. Its node is made
    /// before what it points to is compiled, so a schema that refers back to
    /// itself becomes a cycle in the graph rather than an endless compile.
    fn reference(&mut self, reference: &str) -> Result<NodeId, String> {
        if let Some(&id) = self.refs.get(reference) {
            return Ok(id);
        }
        let pointer = reference
            .strip_prefix('#')
            .ok_or_else(|| format!("`$ref` \"{reference}\" must point into this schema (`#…`)"))?;
        let target = self
            .document
            .pointer(pointer)
            .ok_or_else(|| format!("`$ref` \"{reference}\" points at nothing"))?;
        let id = self.push(Node::Union(Vec::new()))?;
        self.refs.insert(reference.to_string(), id);
        let compiled = self.node(target)?;
        self.nodes[id] = Node::Union(vec![compiled]);
        Ok(id)
    }
}

fn literal(value: &Value) -> Vec<u8> {
    serde_json::to_vec(value).unwrap_or_default()
}

/// What a string's checks let between its quotes, as grammar text. `None` for
/// a string with no checks.
fn string_body(object: &serde_json::Map<String, Value>) -> Result<Option<String>, String> {
    let count = |key: &str| object.get(key).and_then(Value::as_u64).map(|n| n as usize);
    let (min, max) = (count("minLength"), count("maxLength"));
    if let (Some(min), Some(max)) = (min, max)
        && max < min
    {
        return Err("`maxLength` is smaller than `minLength`".into());
    }
    let format = object
        .get("format")
        .and_then(Value::as_str)
        .and_then(schema_text::format_body);
    let pattern = object.get("pattern").map(|pattern| {
        pattern
            .as_str()
            .ok_or_else(|| "`pattern` must be a string".to_string())
            .and_then(schema_text::pattern_body)
    });
    let lengths = min.is_some() || max.is_some();
    let body = match (pattern, format, lengths) {
        (None, None, false) => return Ok(None),
        (Some(pattern), None, false) => pattern?,
        (None, Some(format), false) => format.to_string(),
        (None, None, true) => {
            let min = min.unwrap_or(0);
            match max {
                Some(max) => format!("char{{{min},{max}}}"),
                None => format!("char{{{min},}}"),
            }
        }
        _ => {
            return Err(
                "`pattern`, a checked `format` and a length cannot be enforced together".into(),
            );
        }
    };
    Ok(Some(format!("\"\\\"\" ({body}) \"\\\"\"")))
}

/// The bounds a number's checks set, as grammar text. `None` for a number with
/// no bounds.
fn number_body(
    object: &serde_json::Map<String, Value>,
    integer: bool,
) -> Result<Option<String>, String> {
    let number = |key: &str| {
        object
            .get(key)
            .filter(|v| !v.is_boolean())
            .and_then(Value::as_f64)
    };
    let flag = |key: &str| object.get(key).and_then(Value::as_bool).unwrap_or(false);
    // The older form says `exclusiveMinimum: true` beside `minimum`.
    let low = match (number("minimum"), number("exclusiveMinimum")) {
        (Some(a), Some(b)) if b >= a => Some((b, true)),
        (Some(a), _) => Some((a, flag("exclusiveMinimum"))),
        (None, Some(b)) => Some((b, true)),
        (None, None) => None,
    };
    let high = match (number("maximum"), number("exclusiveMaximum")) {
        (Some(a), Some(b)) if b <= a => Some((b, true)),
        (Some(a), _) => Some((a, flag("exclusiveMaximum"))),
        (None, Some(b)) => Some((b, true)),
        (None, None) => None,
    };
    if low.is_none() && high.is_none() {
        return Ok(None);
    }
    const LIMIT: f64 = 1e30;
    let whole = |value: f64| {
        (value.abs() < LIMIT)
            .then_some(value as i128)
            .ok_or_else(|| format!("the bound {value} is too large to enforce"))
    };
    let body = if integer {
        // A bound past a whole number is the next whole number inside it.
        let low = low
            .map(|(value, strict)| {
                let ceiling = value.ceil();
                let shift = i128::from(strict && ceiling == value);
                whole(ceiling).map(|v| v + shift)
            })
            .transpose()?;
        let high = high
            .map(|(value, strict)| {
                let floor = value.floor();
                let shift = i128::from(strict && floor == value);
                whole(floor).map(|v| v - shift)
            })
            .transpose()?;
        schema_text::integer_range(low, high)
    } else {
        let bound = |bound: Option<(f64, bool)>| {
            bound
                .map(|(value, strict)| {
                    if value.fract() != 0.0 {
                        return Err(format!(
                            "the bound {value} cannot be enforced on a `number`: only whole bounds can"
                        ));
                    }
                    Ok((whole(value)?, strict))
                })
                .transpose()
        };
        schema_text::number_range(bound(low)?, bound(high)?)
    };
    body.map(Some)
        .ok_or_else(|| "no number lies between the schema's bounds".to_string())
}

/// Writes a compiled graph as grammar text.
struct Writer<'a> {
    nodes: &'a [Node],
    /// What this schema's rule names start with, so that several can share a
    /// grammar.
    prefix: &'a str,
    text: String,
}

/// What every schema's grammar can call on.
const COMMON_RULES: &str = "ws ::= [ \\t\\n\\r]{0,32}\n\
    string ::= \"\\\"\" char* \"\\\"\"\n\
    number ::= integer (\".\" [0-9]+)? ([eE] [+-]? [0-9]+)?\n\
    integer ::= \"-\"? (\"0\" | [1-9] [0-9]*)\n\
    any ::= anyobject | anyarray | string | number | \"true\" | \"false\" | \"null\"\n\
    anyobject ::= \"{\" ws (member ws (\",\" ws member ws)*)? \"}\"\n\
    member ::= string ws \":\" ws any\n\
    anyarray ::= \"[\" ws (any ws (\",\" ws any ws)*)? \"]\"\n";

impl Writer<'_> {
    fn rule(&mut self, name: &str, body: &str) {
        self.text.push_str(name);
        self.text.push_str(" ::= ");
        self.text.push_str(body);
        self.text.push('\n');
    }

    /// What a union can be, through unions inside it, each once: a union
    /// naming itself adds nothing, and a grammar could not hold the call.
    fn union_members(&self, id: NodeId) -> Vec<NodeId> {
        let mut members = Vec::new();
        let mut seen = HashSet::new();
        let mut work = vec![id];
        while let Some(id) = work.pop() {
            if !seen.insert(id) {
                continue;
            }
            match &self.nodes[id] {
                Node::Union(branches) => work.extend(branches.iter().rev()),
                _ => members.push(id),
            }
        }
        members
    }

    fn member(&self, key: &[u8], value: NodeId) -> String {
        let key = String::from_utf8_lossy(key);
        let value = self.name(value);
        format!("{} ws \":\" ws {value} ws", schema_text::quoted(&key))
    }

    fn name(&self, id: NodeId) -> String {
        format!("{}n{id}", self.prefix)
    }

    fn node(&mut self, id: NodeId) {
        let name = self.name(id);
        match &self.nodes[id] {
            Node::Any => self.rule(&name, "any"),
            Node::String => self.rule(&name, "string"),
            Node::Number { integer } => {
                self.rule(&name, if *integer { "integer" } else { "number" });
            }
            Node::Text(body) => {
                let body = body.clone();
                self.rule(&name, &body);
            }
            Node::Literals(values) => {
                let alternatives: Vec<String> = values
                    .iter()
                    .map(|v| schema_text::quoted(&String::from_utf8_lossy(v)))
                    .collect();
                self.rule(&name, &alternatives.join(" | "));
            }
            Node::Union(_) => {
                let members: Vec<String> = self
                    .union_members(id)
                    .iter()
                    .map(|m| self.name(*m))
                    .collect();
                if members.is_empty() {
                    // Nothing but itself: there is no such value.
                    self.rule(&name, "[^\\x00-\\U0010FFFF]");
                } else {
                    self.rule(&name, &members.join(" | "));
                }
            }
            Node::Map { values } => match values {
                None => self.rule(&name, "\"{\" ws \"}\""),
                Some(values) => {
                    let member = format!("string ws \":\" ws {} ws", self.name(*values));
                    let body = format!("\"{{\" ws ({member} (\",\" ws {member})*)? \"}}\"");
                    self.rule(&name, &body);
                }
            },
            Node::Array { items, min, max } => {
                let (items, min, max) = (*items, *min, *max);
                let item = format!("{} ws", self.name(items));
                let more = format!("\",\" ws {item}");
                let rest = match max {
                    Some(max) => format!(
                        "({more}){{{},{}}}",
                        min.saturating_sub(1),
                        max.saturating_sub(1)
                    ),
                    None => format!("({more}){{{},}}", min.saturating_sub(1)),
                };
                let body = match (min, max) {
                    (_, Some(0)) => "\"[\" ws \"]\"".to_string(),
                    (0, _) => format!("\"[\" ws ({item} {rest})? \"]\""),
                    _ => format!("\"[\" ws {item} {rest} \"]\""),
                };
                self.rule(&name, &body);
            }
            Node::Object { props } => {
                // One rule per property, for having written none so far (`s`)
                // and some (`t`), so that the grammar grows with the properties
                // however many are optional.
                let count = props.len();
                let lines: Vec<(String, String)> = props
                    .iter()
                    .enumerate()
                    .flat_map(|(i, prop)| {
                        let member = self.member(&prop.key, prop.value);
                        let first = format!("{member} {name}t{}", i + 1);
                        let later = format!("\",\" ws {member} {name}t{}", i + 1);
                        let (first, later) = if prop.required {
                            (first, later)
                        } else {
                            (
                                format!("{first} | {name}s{}", i + 1),
                                format!("{later} | {name}t{}", i + 1),
                            )
                        };
                        [
                            (format!("{name}s{i}"), first),
                            (format!("{name}t{i}"), later),
                        ]
                    })
                    .collect();
                for (rule, body) in lines {
                    self.rule(&rule, &body);
                }
                self.rule(&format!("{name}s{count}"), "\"}\"");
                self.rule(&format!("{name}t{count}"), "\"}\"");
                self.rule(&name, &format!("\"{{\" ws {name}s0"));
            }
        }
    }
}

/// The rules every schema's grammar can call on.
pub(crate) fn common_rules() -> String {
    format!(
        "{COMMON_RULES}{}{}",
        schema_text::JSON_CHAR_RULES,
        schema_text::FORMAT_RULES
    )
}

/// `schema` as rules whose names start with `prefix`, for a grammar that also
/// holds [`common_rules`], and the name of the rule that is the schema.
pub(crate) fn schema_rules(schema: &Value, prefix: &str) -> Result<(String, String), String> {
    let mut compiler = Compiler {
        document: schema,
        nodes: Vec::new(),
        refs: HashMap::new(),
    };
    let root = compiler.node(schema)?;
    let mut writer = Writer {
        nodes: &compiler.nodes,
        prefix,
        text: String::new(),
    };
    for id in 0..compiler.nodes.len() {
        writer.node(id);
    }
    let name = writer.name(root);
    Ok((writer.text, name))
}

/// The schema-shaped counterpart of `JsonPrefix`: what a reply held to a schema
/// has reached.
#[derive(Clone, Debug)]
pub struct SchemaPrefix {
    matcher: Matcher,
}

impl SchemaPrefix {
    pub fn new(schema: Arc<Schema>) -> Self {
        Self {
            matcher: Matcher::new(schema.grammar.clone()),
        }
    }

    /// Feeds every byte of `bytes`, returning `false` once no reading of
    /// them fits the schema. `self` is only usable afterwards if this
    /// returned `true`.
    pub fn push_bytes(&mut self, bytes: &[u8]) -> bool {
        self.matcher.push_bytes(bytes)
    }

    /// Whether `bytes` could follow what has been read, without committing
    /// to them.
    pub fn allows(&mut self, bytes: &[u8]) -> bool {
        self.matcher.allows(bytes)
    }

    /// Whether what has been read is already a whole document the schema
    /// accepts.
    pub fn is_complete(&self) -> bool {
        self.matcher.is_complete()
    }
}

/// Whether `value` is one `schema` describes: `Err` names where it is not
/// and why. Checks a whole value, in any key order — what a tool's
/// `strict: true` holds a finished call's input to — and reads `type`,
/// `properties`/`required`/`additionalProperties`,
/// `items`/`minItems`/`maxItems`, `enum`, `const`, `anyOf`/`oneOf`/`allOf`,
/// `$ref` into the same document, `minLength`/`maxLength`, `pattern` and
/// `minimum`/`maximum`.
pub fn validate(schema: &Value, value: &Value) -> Result<(), String> {
    check(schema, schema, value, "", 0, &mut { MAX_CHECK_STEPS })
}

fn kind_of(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn is_type(value: &Value, kind: &str) -> bool {
    match kind {
        "integer" => value.as_f64().is_some_and(|n| n.fract() == 0.0),
        "number" => value.is_number(),
        other => kind_of(value) == other,
    }
}

fn check(
    root: &Value,
    schema: &Value,
    value: &Value,
    at: &str,
    depth: usize,
    steps: &mut usize,
) -> Result<(), String> {
    let place = if at.is_empty() { "the input" } else { at };
    if depth > 64 {
        return Err(format!("{place}: the schema nests too deeply to check"));
    }
    // Trying each branch of a union is exponential in the depth where the
    // branches lead back to the union, which a request's schema can arrange.
    let Some(left) = steps.checked_sub(1) else {
        return Err(format!("{place}: the schema is too complex to check"));
    };
    *steps = left;
    let schema = match schema {
        Value::Bool(true) => return Ok(()),
        Value::Bool(false) => return Err(format!("{place}: no value is allowed here")),
        Value::Object(map) => map,
        _ => return Ok(()),
    };
    if let Some(reference) = schema.get("$ref").and_then(Value::as_str) {
        let target = reference
            .strip_prefix('#')
            .and_then(|pointer| root.pointer(pointer))
            .ok_or_else(|| format!("{place}: the schema's $ref {reference} names nothing"))?;
        check(root, target, value, at, depth + 1, steps)?;
    }
    if let Some(types) = schema.get("type") {
        let allowed: Vec<&str> = match types {
            Value::String(t) => vec![t.as_str()],
            Value::Array(ts) => ts.iter().filter_map(Value::as_str).collect(),
            _ => Vec::new(),
        };
        if !allowed.is_empty() && !allowed.iter().any(|t| is_type(value, t)) {
            return Err(format!(
                "{place}: expected {}, got {}",
                allowed.join(" or "),
                kind_of(value)
            ));
        }
    }
    if let Some(options) = schema.get("enum").and_then(Value::as_array)
        && !options.contains(value)
    {
        return Err(format!(
            "{place}: must be one of {}",
            Value::Array(options.clone())
        ));
    }
    if let Some(constant) = schema.get("const")
        && constant != value
    {
        return Err(format!("{place}: must be {constant}"));
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(branches) = schema.get(key).and_then(Value::as_array)
            && !branches
                .iter()
                .any(|b| check(root, b, value, at, depth + 1, steps).is_ok())
        {
            return Err(format!(
                "{place}: matches none of the schema's alternatives"
            ));
        }
    }
    if let Some(all) = schema.get("allOf").and_then(Value::as_array) {
        for branch in all {
            check(root, branch, value, at, depth + 1, steps)?;
        }
    }
    match value {
        Value::Object(fields) => {
            let properties = schema.get("properties").and_then(Value::as_object);
            for required in schema
                .get("required")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
            {
                if !fields.contains_key(required) {
                    return Err(format!(
                        "{place}: the required property `{required}` is missing"
                    ));
                }
            }
            for (key, field) in fields {
                let path = format!("{at}/{key}");
                match properties.and_then(|p| p.get(key)) {
                    Some(sub) => check(root, sub, field, &path, depth + 1, steps)?,
                    None => match schema.get("additionalProperties") {
                        Some(Value::Bool(false)) => {
                            return Err(format!("{place}: `{key}` is not one of its properties"));
                        }
                        Some(sub @ Value::Object(_)) => {
                            check(root, sub, field, &path, depth + 1, steps)?
                        }
                        _ => {}
                    },
                }
            }
        }
        Value::Array(items) => {
            let count = |key: &str| schema.get(key).and_then(Value::as_u64);
            if let Some(min) = count("minItems")
                && (items.len() as u64) < min
            {
                return Err(format!("{place}: needs at least {min} items"));
            }
            if let Some(max) = count("maxItems")
                && (items.len() as u64) > max
            {
                return Err(format!("{place}: takes at most {max} items"));
            }
            if let Some(sub) = schema.get("items") {
                for (i, item) in items.iter().enumerate() {
                    check(root, sub, item, &format!("{at}/{i}"), depth + 1, steps)?;
                }
            }
        }
        Value::String(text) => {
            let length = text.chars().count() as u64;
            if let Some(min) = schema.get("minLength").and_then(Value::as_u64)
                && length < min
            {
                return Err(format!("{place}: needs at least {min} characters"));
            }
            if let Some(max) = schema.get("maxLength").and_then(Value::as_u64)
                && length > max
            {
                return Err(format!("{place}: takes at most {max} characters"));
            }
            if let Some(pattern) = schema.get("pattern").and_then(Value::as_str)
                && let Ok(re) = regex::Regex::new(pattern)
                && !re.is_match(text)
            {
                return Err(format!("{place}: must match the pattern {pattern}"));
            }
        }
        Value::Number(n) => {
            let n = n.as_f64().unwrap_or_default();
            if let Some(min) = schema.get("minimum").and_then(Value::as_f64)
                && n < min
            {
                return Err(format!("{place}: must be at least {min}"));
            }
            if let Some(max) = schema.get("maximum").and_then(Value::as_f64)
                && n > max
            {
                return Err(format!("{place}: must be at most {max}"));
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// A union naming itself many times is many paths to the same place: it
    /// has to take about as long as naming itself once.
    #[test]
    fn a_union_of_references_to_itself_is_not_exponential() {
        let schema = serde_json::json!({
            "$defs": {"n": {"anyOf": [
                {"type": "null"},
                {"$ref": "#/$defs/n"}, {"$ref": "#/$defs/n"},
                {"$ref": "#/$defs/n"}, {"$ref": "#/$defs/n"},
            ]}},
            "$ref": "#/$defs/n",
        });
        let started = std::time::Instant::now();
        let mut prefix =
            super::SchemaPrefix::new(std::sync::Arc::new(super::compile(&schema).unwrap()));
        assert!(prefix.push_bytes(b"null"));
        assert!(prefix.is_complete());
        // Null satisfies it, at once; anything else has to try the branches.
        assert!(super::validate(&schema, &serde_json::json!(null)).is_ok());
        assert!(super::validate(&schema, &serde_json::json!(1)).is_err());
        assert!(started.elapsed().as_secs() < 10, "{:?}", started.elapsed());
    }

    use super::*;

    /// A finished value is checked whole, in any key order, and the error
    /// says where.
    #[test]
    fn a_value_is_validated_against_its_schema() {
        let schema = serde_json::json!({
            "type": "object", "required": ["path", "mode"], "additionalProperties": false,
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "mode": {"enum": ["read", "write"]},
                "tags": {"type": "array", "items": {"$ref": "#/$defs/tag"}, "maxItems": 2},
                "size": {"type": "integer", "minimum": 0},
            },
            "$defs": {"tag": {"type": "string", "pattern": "^[a-z]+$"}},
        });
        let ok = serde_json::json!({"mode": "read", "path": "/a", "tags": ["x"], "size": 3});
        assert_eq!(validate(&schema, &ok), Ok(()));
        for (bad, says) in [
            (serde_json::json!({"path": "/a"}), "`mode` is missing"),
            (
                serde_json::json!({"path": 1, "mode": "read"}),
                "/path: expected string",
            ),
            (
                serde_json::json!({"path": "/a", "mode": "x"}),
                "/mode: must be one of",
            ),
            (
                serde_json::json!({"path": "/a", "mode": "read", "extra": 1}),
                "`extra` is not",
            ),
            (
                serde_json::json!({"path": "/a", "mode": "read", "tags": ["A"]}),
                "/tags/0: must match",
            ),
            (
                serde_json::json!({"path": "/a", "mode": "read", "size": 1.5}),
                "/size: expected integer",
            ),
            (
                serde_json::json!({"path": "/a", "mode": "read", "size": -1}),
                "/size: must be at least 0",
            ),
        ] {
            let err = validate(&schema, &bad).unwrap_err();
            assert!(err.contains(says), "{bad}: {err}");
        }
    }
    use serde_json::json;

    fn fixture() -> Vec<Value> {
        serde_json::from_str(include_str!("fixtures/grammar-vectors.json")).unwrap()
    }

    fn prefix(schema: Value) -> SchemaPrefix {
        SchemaPrefix::new(Arc::new(compile(&schema).unwrap()))
    }

    /// `(still a valid prefix, complete)` after feeding `text`.
    fn run(schema: &Value, text: &str) -> (bool, bool) {
        let mut p = prefix(schema.clone());
        let ok = p.push_bytes(text.as_bytes());
        (ok, ok && p.is_complete())
    }

    fn person() -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string"},
                "role": {"enum": ["admin", "user"]},
            },
            "required": ["name", "role"],
            "additionalProperties": false,
        })
    }

    #[test]
    fn an_object_follows_its_properties_in_order() {
        let s = person();
        assert_eq!(
            run(&s, r#"{"name": "Ada", "age": 36, "role": "admin"}"#),
            (true, true)
        );
        // Optional properties may be skipped…
        assert_eq!(run(&s, r#"{"name":"Ada","role":"user"}"#), (true, true));
        // …required ones may not, and nothing undeclared may appear.
        assert!(!run(&s, r#"{"name":"Ada"}"#).0);
        assert!(!run(&s, r#"{"role""#).0);
        assert!(!run(&s, r#"{"nickname""#).0);
        // Declared order is the only order.
        assert!(!run(&s, r#"{"name":"Ada","role":"user","age""#).0);
        // A prefix is allowed and is not yet complete.
        assert_eq!(run(&s, r#"{"name":"Ad"#), (true, false));
    }

    #[test]
    fn values_have_their_declared_types() {
        let s = person();
        assert!(!run(&s, r#"{"name":"Ada","age":36.5"#).0);
        assert!(!run(&s, r#"{"name":"Ada","age":"36""#).0);
        assert!(!run(&s, r#"{"name":"Ada","role":"root""#).0);
        assert!(!run(&s, r#"{"name":7"#).0);
    }

    #[test]
    fn arrays_count_their_items() {
        let s = json!({"type": "array", "items": {"type": "number"}, "minItems": 1, "maxItems": 2});
        assert!(!run(&s, "[]").0);
        assert_eq!(run(&s, "[1.5]"), (true, true));
        assert_eq!(run(&s, "[1, -2e3]"), (true, true));
        assert!(!run(&s, "[1, 2, 3").0);
        assert!(!run(&s, r#"["a""#).0);
    }

    #[test]
    fn unions_nulls_and_type_lists_read_every_way() {
        let s = json!({"type": "object", "properties": {
            "v": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
            "n": {"type": ["string", "null"]},
        }, "required": ["v", "n"]});
        assert_eq!(run(&s, r#"{"v": 3, "n": null}"#), (true, true));
        assert_eq!(run(&s, r#"{"v": "x", "n": "y"}"#), (true, true));
        assert!(!run(&s, r#"{"v": true"#).0);
    }

    #[test]
    fn enums_sharing_a_prefix_stay_open_until_they_diverge() {
        let s = json!({"enum": [1, 12, "a"]});
        assert_eq!(run(&s, "1"), (true, true));
        assert_eq!(run(&s, "12"), (true, true));
        assert!(!run(&s, "13").0);
        assert_eq!(run(&s, "\"a\""), (true, true));
    }

    #[test]
    fn a_reference_may_recurse() {
        let s = json!({
            "$defs": {"node": {"type": "object", "properties": {
                "value": {"type": "integer"},
                "children": {"type": "array", "items": {"$ref": "#/$defs/node"}},
            }, "required": ["value"]}},
            "$ref": "#/$defs/node",
        });
        assert_eq!(
            run(
                &s,
                r#"{"value":1,"children":[{"value":2},{"value":3,"children":[]}]}"#
            ),
            (true, true)
        );
        assert!(!run(&s, r#"{"value":1,"children":[{"x""#).0);
    }

    #[test]
    fn free_maps_and_any_values() {
        let s = json!({"type": "object", "additionalProperties": {"type": "integer"}});
        assert_eq!(run(&s, r#"{"a": 1, "b": 2}"#), (true, true));
        assert!(!run(&s, r#"{"a": "x""#).0);
        let any = json!({"type": "object", "properties": {"meta": {}}, "required": ["meta"]});
        assert_eq!(
            run(&any, r#"{"meta": {"deep": [1, {"x": null}]}}"#),
            (true, true)
        );
        assert!(!run(&any, r#"{"meta": 12, "#).0);
        let closed = json!({"type": "object", "additionalProperties": false});
        assert_eq!(run(&closed, "{}"), (true, true));
        assert!(!run(&closed, r#"{"a""#).0);
    }

    #[test]
    fn strings_take_escapes_and_refuse_control_bytes() {
        let s = json!({"type": "string"});
        assert_eq!(run(&s, r#""a\"bé\n""#), (true, true));
        assert!(!run(&s, "\"a\u{1}").0);
        assert!(!run(&s, r#""\q"#).0);
    }

    #[test]
    fn whitespace_runs_are_bounded() {
        let s = json!({"type": "object", "properties": {"a": {"type": "integer"}}});
        let long = format!("{{{}", " ".repeat(MAX_WHITESPACE as usize + 1));
        assert!(!run(&s, &long).0);
        let fine = format!("{{{}", "\n ".repeat(4));
        assert_eq!(run(&s, &fine), (true, false));
    }

    #[test]
    fn what_cannot_be_followed_is_refused_at_compile_time() {
        for schema in [
            json!({"not": {"type": "string"}}),
            json!({"allOf": [{"type": "string"}, {"minLength": 2}]}),
            json!({"type": "object", "properties": {"a": {}}, "required": ["b"]}),
            json!({"$ref": "https://example.com/schema"}),
            json!({"type": "date"}),
            json!(false),
        ] {
            assert!(compile(&schema).is_err(), "{schema}");
        }
        // A `format` nothing checks is an annotation.
        assert!(compile(&json!({"type": "string", "format": "email", "minLength": 3})).is_ok());
    }

    /// The published grammar and schema vectors: strings each must take, and
    /// strings each must refuse. Where this differs on purpose, the vector is
    /// named in `DIFFERS`.
    #[test]
    fn the_published_vectors_read_as_they_should() {
        // What this does differently on purpose: integers are not stopped at
        // fifteen digits, a schema is an object, a key outside `properties` is
        // not written even where the schema would take it (the document is
        // still one it accepts), and `uniqueItems` cannot be enforced.
        const DIFFERS: &[&str] = &[
            "schema min 0: fail \"100000000000000000000000000000000\"",
            "schema min 2: fail \"12345678900000000\"",
            "schema exotic formats (list)",
            "schema additional properties can't override other properties: pass",
            "schema object properties, additionalProperties: true: pass",
            "schema required props: refused",
        ];
        let mut wrong = Vec::new();
        for vector in fixture() {
            let (kind, name, source) = (
                vector["kind"].as_str().unwrap(),
                vector["name"].as_str().unwrap(),
                vector["source"].as_str().unwrap(),
            );
            let grammar = if kind == "grammar" {
                crate::engine::grammar::Grammar::parse(source).map_err(|e| e.to_string())
            } else {
                serde_json::from_str::<Value>(source)
                    .map_err(|e| e.to_string())
                    .and_then(|schema| compile(&schema))
                    .map(|schema| schema.grammar)
            };
            let grammar = match grammar {
                Ok(grammar) => grammar,
                Err(error) => {
                    wrong.push(format!("{kind} {name}: refused: {error}"));
                    continue;
                }
            };
            for (list, want) in [("pass", true), ("fail", false)] {
                for text in vector[list].as_array().unwrap() {
                    let text = text.as_str().unwrap();
                    let mut matcher = crate::engine::grammar::Matcher::new(grammar.clone());
                    let got = matcher.push_bytes(text.as_bytes()) && matcher.is_complete();
                    if got != want {
                        wrong.push(format!("{kind} {name}: {list} {text:?}"));
                    }
                }
            }
        }
        wrong.retain(|line| !DIFFERS.iter().any(|known| line.starts_with(known)));
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    #[test]
    fn a_reference_to_itself_alone_is_harmless() {
        let s = json!({"$defs": {"a": {"$ref": "#/$defs/a"}}, "$ref": "#/$defs/a"});
        assert!(!run(&s, "1").0);
    }
}
