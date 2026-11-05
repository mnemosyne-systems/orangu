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
//! It answers the same two questions as [`JsonPrefix`] — "could this still
//! become valid", for the mask, and "is this already complete", for the stop
//! condition — with the schema's shape on top of JSON's.
//!
//! # The schema it reads
//!
//! [`compile`] turns a schema into a small graph of [`Node`]s:
//!
//! - `type` (one, or a list of them), and the shape each implies:
//!   `properties`/`required`/`additionalProperties` for an object,
//!   `items`/`minItems`/`maxItems` for an array.
//! - `enum` and `const`, matched as their compact JSON text.
//! - `anyOf`/`oneOf` (any branch may match), a one-schema `allOf`, and
//!   `$ref` into the same document (`#/$defs/…`, `#/definitions/…`, `#`),
//!   recursion included.
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
//! Annotations and value checks — `description`, `title`, `format`,
//! `pattern`, `minLength`, `minimum` and the like — do not change what can
//! be written and are not enforced. Keywords that change the shape in ways
//! this cannot follow (`not`, `if`, `patternProperties`, a many-schema
//! `allOf`, …) are refused at compile time rather than silently dropped.
//!
//! # How it recognises
//!
//! A [`Thread`] is one way of having read the bytes so far: a stack of
//! [`Frame`]s, one per open value. A union or an `enum` whose members share a
//! prefix can be read more than one way, so [`SchemaPrefix`] carries every
//! thread still alive and a byte is allowed while any survives. Each thread
//! is a few frames, cheap to clone — which is what the sampler does to test
//! a token — and branches end within a few bytes as the alternatives
//! diverge.

use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::engine::constraint::JsonPrefix;

type NodeId = usize;

/// The most nodes a compiled schema may have, so a request cannot make the
/// server build an arbitrarily large graph.
const MAX_NODES: usize = 4096;

/// The most parse threads carried at once. Beyond it the newest are dropped:
/// a schema that ambiguous is one whose later alternatives are not reached.
const MAX_THREADS: usize = 64;

/// The most whitespace bytes in a row. Whitespace keeps the model free to
/// lay a document out, but an unbounded run is how a constrained model can
/// spend its whole budget on newlines.
const MAX_WHITESPACE: u8 = 32;

/// How deep unions and references may nest before a value starts — the
/// guard against a reference that names only itself.
const MAX_UNION_DEPTH: usize = 32;

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
    nodes: Vec<Node>,
    root: NodeId,
}

/// Compiles a JSON Schema, or says which part of it cannot be followed.
pub fn compile(schema: &Value) -> Result<Schema, String> {
    let mut compiler = Compiler {
        document: schema,
        nodes: Vec::new(),
        refs: HashMap::new(),
    };
    let root = compiler.node(schema)?;
    Ok(Schema {
        nodes: compiler.nodes,
        root,
    })
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
            "string" => self.push(Node::String),
            "number" => self.push(Node::Number { integer: false }),
            "integer" => self.push(Node::Number { integer: true }),
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

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Number {
    Minus,
    Zero,
    Int,
    Dot,
    Frac,
    E,
    ESign,
    Exp,
}

impl Number {
    fn start(b: u8) -> Option<Self> {
        match b {
            b'-' => Some(Self::Minus),
            b'0' => Some(Self::Zero),
            b'1'..=b'9' => Some(Self::Int),
            _ => None,
        }
    }

    fn next(self, b: u8, integer: bool) -> Option<Self> {
        let digit = b.is_ascii_digit();
        let exponent = matches!(b, b'e' | b'E') && !integer;
        match self {
            Self::Minus if b == b'0' => Some(Self::Zero),
            Self::Minus if digit => Some(Self::Int),
            Self::Zero | Self::Int if b == b'.' && !integer => Some(Self::Dot),
            Self::Zero | Self::Int | Self::Frac if exponent => Some(Self::E),
            Self::Int if digit => Some(Self::Int),
            Self::Dot | Self::Frac if digit => Some(Self::Frac),
            Self::E if matches!(b, b'+' | b'-') => Some(Self::ESign),
            Self::E | Self::ESign | Self::Exp if digit => Some(Self::Exp),
            _ => None,
        }
    }

    fn complete(self) -> bool {
        matches!(self, Self::Zero | Self::Int | Self::Frac | Self::Exp)
    }
}

#[derive(Clone, Copy, Debug)]
enum Escape {
    None,
    Backslash,
    Hex(u8),
}

#[derive(Clone, Copy, Debug)]
enum Phase {
    /// After `{` or `[`: a member, or the close.
    Open,
    /// After a key: the `:`. For an object, which property it named.
    Colon(usize),
    /// After a member: `,` or the close.
    AfterValue,
    /// After `,`: a member.
    AfterComma,
}

#[derive(Clone, Debug)]
enum Frame {
    /// A value of this node, not yet begun.
    Value(NodeId),
    /// An object with declared properties; `next` is the first property not
    /// yet passed.
    Object {
        node: NodeId,
        next: usize,
        phase: Phase,
    },
    Map {
        values: Option<NodeId>,
        phase: Phase,
    },
    Array {
        node: NodeId,
        count: usize,
        phase: Phase,
    },
    /// A declared key being read: the properties it could still be, and how
    /// many bytes of their keys have matched.
    Key {
        node: NodeId,
        alive: Vec<usize>,
        pos: usize,
    },
    Str(Escape),
    Num {
        integer: bool,
        state: Number,
    },
    Lit {
        node: NodeId,
        alive: Vec<usize>,
        pos: usize,
    },
    Any(JsonPrefix),
}

#[derive(Clone, Debug)]
struct Thread {
    stack: Vec<Frame>,
    whitespace: u8,
}

impl Thread {
    /// Takes one more whitespace byte, if the run is not too long already.
    fn space(mut self, out: &mut Vec<Thread>) {
        self.whitespace += 1;
        if self.whitespace <= MAX_WHITESPACE {
            out.push(self);
        }
    }
}

impl Schema {
    fn props(&self, node: NodeId) -> &[Prop] {
        match &self.nodes[node] {
            Node::Object { props } => props,
            _ => &[],
        }
    }

    fn literals(&self, node: NodeId) -> &[Vec<u8>] {
        match &self.nodes[node] {
            Node::Literals(literals) => literals,
            _ => &[],
        }
    }

    /// The properties the next key may name: the first not yet passed, and
    /// each after it up to and including the first required one.
    fn next_keys(&self, node: NodeId, next: usize) -> Vec<usize> {
        let props = self.props(node);
        let mut keys = Vec::new();
        for (index, prop) in props.iter().enumerate().skip(next) {
            keys.push(index);
            if prop.required {
                break;
            }
        }
        keys
    }

    /// Whether the object may close with `next` the first property not yet
    /// passed: no required property is left.
    fn may_close(&self, node: NodeId, next: usize) -> bool {
        !self.props(node).iter().skip(next).any(|p| p.required)
    }

    /// Begins a value of `node` with byte `b`.
    fn start(&self, t: Thread, node: NodeId, b: u8, depth: usize, out: &mut Vec<Thread>) {
        self.start_once(t, node, b, depth, out, &mut HashSet::new());
    }

    /// [`Self::start`], going through each union once. Descending into a union
    /// changes nothing about the thread, so a second visit starts the same
    /// threads again, and a union that names itself several times would make
    /// the visits exponential in the depth allowed.
    fn start_once(
        &self,
        mut t: Thread,
        node: NodeId,
        b: u8,
        depth: usize,
        out: &mut Vec<Thread>,
        seen: &mut HashSet<NodeId>,
    ) {
        match &self.nodes[node] {
            Node::Union(branches) => {
                if depth < MAX_UNION_DEPTH && seen.insert(node) {
                    for &branch in branches {
                        self.start_once(t.clone(), branch, b, depth + 1, out, seen);
                    }
                }
            }
            Node::Any => {
                let mut json = JsonPrefix::new();
                if json.push_bytes(&[b]) {
                    t.stack.push(Frame::Any(json));
                    out.push(t);
                }
            }
            Node::Object { .. } if b == b'{' => {
                t.stack.push(Frame::Object {
                    node,
                    next: 0,
                    phase: Phase::Open,
                });
                out.push(t);
            }
            Node::Map { values } if b == b'{' => {
                t.stack.push(Frame::Map {
                    values: *values,
                    phase: Phase::Open,
                });
                out.push(t);
            }
            Node::Array { .. } if b == b'[' => {
                t.stack.push(Frame::Array {
                    node,
                    count: 0,
                    phase: Phase::Open,
                });
                out.push(t);
            }
            Node::String if b == b'"' => {
                t.stack.push(Frame::Str(Escape::None));
                out.push(t);
            }
            Node::Number { integer } => {
                if let Some(state) = Number::start(b) {
                    t.stack.push(Frame::Num {
                        integer: *integer,
                        state,
                    });
                    out.push(t);
                }
            }
            Node::Literals(literals) => {
                let alive: Vec<usize> = (0..literals.len())
                    .filter(|&i| literals[i].first() == Some(&b))
                    .collect();
                if !alive.is_empty() {
                    t.stack.push(Frame::Lit {
                        node,
                        alive,
                        pos: 1,
                    });
                    out.push(t);
                }
            }
            _ => {}
        }
    }

    /// Every way `t` can go on after reading `b`.
    fn feed(&self, mut t: Thread, b: u8, out: &mut Vec<Thread>) {
        let Some(top) = t.stack.last_mut() else {
            // The document is complete; only trailing whitespace may follow.
            if is_ws(b) {
                t.space(out);
            }
            return;
        };
        if !is_ws(b) {
            t.whitespace = 0;
        }
        match top {
            Frame::Value(node) => {
                let node = *node;
                if is_ws(b) {
                    return t.space(out);
                }
                t.stack.pop();
                self.start(t, node, b, 0, out);
            }
            Frame::Str(escape) => {
                *escape = match (*escape, b) {
                    (Escape::None, b'"') => {
                        t.stack.pop();
                        return out.push(t);
                    }
                    (Escape::None, b'\\') => Escape::Backslash,
                    (Escape::None, b) if b < 0x20 => return,
                    (Escape::None, _) => Escape::None,
                    (Escape::Backslash, b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't') => {
                        Escape::None
                    }
                    (Escape::Backslash, b'u') => Escape::Hex(4),
                    (Escape::Hex(n), b) if b.is_ascii_hexdigit() => {
                        if n == 1 {
                            Escape::None
                        } else {
                            Escape::Hex(n - 1)
                        }
                    }
                    _ => return,
                };
                out.push(t);
            }
            Frame::Num { integer, state } => match state.next(b, *integer) {
                Some(next) => {
                    *state = next;
                    out.push(t);
                }
                None if state.complete() => {
                    t.stack.pop();
                    self.feed(t, b, out);
                }
                None => {}
            },
            Frame::Lit { node, alive, pos } => {
                let literals = self.literals(*node);
                let at = *pos;
                let complete = alive.iter().any(|&i| literals[i].len() == at);
                let going_on: Vec<usize> = alive
                    .iter()
                    .copied()
                    .filter(|&i| literals[i].get(at) == Some(&b))
                    .collect();
                if !going_on.is_empty() {
                    let mut next = t.clone();
                    if let Some(Frame::Lit { alive, pos, .. }) = next.stack.last_mut() {
                        *alive = going_on;
                        *pos += 1;
                    }
                    out.push(next);
                }
                if complete {
                    t.stack.pop();
                    self.feed(t, b, out);
                }
            }
            Frame::Any(json) => {
                if json.is_complete() && (is_ws(b) || !json.allows(&[b])) {
                    t.stack.pop();
                    self.feed(t, b, out);
                } else if json.push_bytes(&[b]) {
                    out.push(t);
                }
            }
            Frame::Key { node, alive, pos } => {
                let props = self.props(*node);
                let at = *pos;
                alive.retain(|&i| props[i].key.get(at) == Some(&b));
                *pos += 1;
                let finished = alive
                    .iter()
                    .copied()
                    .find(|&i| props[i].key.len() == at + 1);
                match (alive.is_empty(), finished) {
                    (true, _) => {}
                    (false, Some(prop)) => {
                        t.stack.pop();
                        if let Some(Frame::Object { phase, .. }) = t.stack.last_mut() {
                            *phase = Phase::Colon(prop);
                        }
                        out.push(t);
                    }
                    (false, None) => out.push(t),
                }
            }
            Frame::Object { node, next, phase } => {
                let (node, next_prop) = (*node, *next);
                match (*phase, b) {
                    (_, b) if is_ws(b) => t.space(out),
                    (Phase::Open | Phase::AfterComma, b'"') => {
                        let alive = self.next_keys(node, next_prop);
                        if !alive.is_empty() {
                            t.stack.push(Frame::Key {
                                node,
                                alive,
                                pos: 1,
                            });
                            out.push(t);
                        }
                    }
                    (Phase::Open | Phase::AfterValue, b'}') if self.may_close(node, next_prop) => {
                        t.stack.pop();
                        out.push(t);
                    }
                    (Phase::Colon(prop), b':') => {
                        *phase = Phase::AfterValue;
                        *next = prop + 1;
                        let value = self.props(node)[prop].value;
                        t.stack.push(Frame::Value(value));
                        out.push(t);
                    }
                    (Phase::AfterValue, b',') if !self.next_keys(node, next_prop).is_empty() => {
                        *phase = Phase::AfterComma;
                        out.push(t);
                    }
                    _ => {}
                }
            }
            Frame::Map { values, phase } => match (*phase, b, *values) {
                (_, b, _) if is_ws(b) => t.space(out),
                (Phase::Open | Phase::AfterComma, b'"', Some(_)) => {
                    *phase = Phase::Colon(0);
                    t.stack.push(Frame::Str(Escape::None));
                    out.push(t);
                }
                (Phase::Open | Phase::AfterValue, b'}', _) => {
                    t.stack.pop();
                    out.push(t);
                }
                (Phase::Colon(_), b':', Some(value)) => {
                    *phase = Phase::AfterValue;
                    t.stack.push(Frame::Value(value));
                    out.push(t);
                }
                (Phase::AfterValue, b',', _) => {
                    *phase = Phase::AfterComma;
                    out.push(t);
                }
                _ => {}
            },
            Frame::Array { node, count, phase } => {
                let Node::Array { items, min, max } = self.nodes[*node] else {
                    return;
                };
                match (*phase, b) {
                    (_, b) if is_ws(b) => t.space(out),
                    (Phase::Open, b']') if min == 0 => {
                        t.stack.pop();
                        out.push(t);
                    }
                    (Phase::AfterValue, b']') if *count >= min => {
                        t.stack.pop();
                        out.push(t);
                    }
                    (Phase::AfterValue, b',') if max.is_none_or(|max| *count < max) => {
                        *phase = Phase::AfterComma;
                        out.push(t);
                    }
                    (Phase::Open | Phase::AfterComma, _)
                        if b != b']' && max.is_none_or(|max| *count < max) =>
                    {
                        *phase = Phase::AfterValue;
                        *count += 1;
                        t.stack.push(Frame::Value(items));
                        self.feed(t, b, out);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Whether a thread holds a whole document: nothing open, or only a
    /// top-level number or literal that nothing has come after to end.
    fn thread_complete(&self, t: &Thread) -> bool {
        match t.stack.as_slice() {
            [] => true,
            [Frame::Num { state, .. }] => state.complete(),
            [Frame::Lit { node, alive, pos }] => {
                alive.iter().any(|&i| self.literals(*node)[i].len() == *pos)
            }
            [Frame::Any(json)] => json.is_complete(),
            _ => false,
        }
    }
}

/// The schema-shaped counterpart of [`JsonPrefix`] — see the module docs.
#[derive(Clone, Debug)]
pub struct SchemaPrefix {
    schema: Arc<Schema>,
    threads: Vec<Thread>,
}

impl SchemaPrefix {
    pub fn new(schema: Arc<Schema>) -> Self {
        let root = schema.root;
        Self {
            schema,
            threads: vec![Thread {
                stack: vec![Frame::Value(root)],
                whitespace: 0,
            }],
        }
    }

    /// Feeds every byte of `bytes`, returning `false` once no reading of
    /// them fits the schema. `self` is only usable afterwards if this
    /// returned `true`.
    pub fn push_bytes(&mut self, bytes: &[u8]) -> bool {
        for &b in bytes {
            let mut next = Vec::new();
            for thread in std::mem::take(&mut self.threads) {
                self.schema.feed(thread, b, &mut next);
                if next.len() >= MAX_THREADS {
                    break;
                }
            }
            next.truncate(MAX_THREADS);
            self.threads = next;
            if self.threads.is_empty() {
                return false;
            }
        }
        true
    }

    /// Whether `bytes` could follow what has been read, without committing
    /// to them.
    pub fn allows(&self, bytes: &[u8]) -> bool {
        self.clone().push_bytes(bytes)
    }

    /// Whether what has been read is already a whole document the schema
    /// accepts.
    pub fn is_complete(&self) -> bool {
        self.threads.iter().any(|t| self.schema.thread_complete(t))
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
        // Annotations and value checks compile, and are not enforced.
        assert!(compile(&json!({"type": "string", "format": "email", "minLength": 3})).is_ok());
    }

    #[test]
    fn a_reference_to_itself_alone_is_harmless() {
        let s = json!({"$defs": {"a": {"$ref": "#/$defs/a"}}, "$ref": "#/$defs/a"});
        assert!(!run(&s, "1").0);
    }
}
