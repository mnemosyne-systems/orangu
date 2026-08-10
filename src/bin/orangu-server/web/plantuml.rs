// Copyright (C) 2026 The orangu community
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! A small, clean-room PlantUML-compatible renderer for the web console.
//!
//! PlantUML's reference implementation depends on a Java runtime and, for
//! many diagram families, Graphviz. The console must remain a single offline
//! binary, so this module parses the commonly generated UML syntax itself and
//! lays it out directly as SVG. PNGs are rasterized from that SVG in-process.
//!
//! This intentionally fails closed. Unsupported structural syntax returns
//! `None`, allowing the Markdown renderer to preserve the original code block
//! instead of displaying a plausible but incomplete diagram.

use base64::Engine as _;
use rustc_hash::FxHasher;
use std::{
    collections::{HashMap, VecDeque},
    fmt::Write as _,
    hash::{Hash as _, Hasher as _},
    sync::{Mutex, OnceLock},
};

const CACHE_LIMIT: usize = 256;
const MAX_SOURCE_BYTES: usize = 256 * 1024;
const MAX_ITEMS: usize = 512;
const MAX_DIMENSION: f64 = 4096.0;
pub const MAX_PER_ATTACHMENT: usize = 32;

type Cache = HashMap<u64, Option<&'static Diagram>>;

/// A rendered PlantUML diagram in both console themes and both requested
/// output formats.
pub struct Diagram {
    pub light: String,
    pub dark: String,
    pub light_png: String,
    pub dark_png: String,
    pub width: f64,
    pub height: f64,
}

pub struct Found {
    pub diagram: &'static Diagram,
    pub source: String,
}

#[derive(Clone, Copy)]
struct Palette {
    canvas: &'static str,
    surface: &'static str,
    surface_alt: &'static str,
    text: &'static str,
    subtle: &'static str,
    border: &'static str,
    accent: &'static str,
    note: &'static str,
}

fn palette(dark: bool) -> Palette {
    if dark {
        Palette {
            canvas: "#23272e",
            surface: "#2a2f38",
            surface_alt: "#2b3a55",
            text: "#e6e8eb",
            subtle: "#9aa3b2",
            border: "#343a45",
            accent: "#5b9dff",
            note: "#403d2d",
        }
    } else {
        Palette {
            canvas: "#ffffff",
            surface: "#eef0f3",
            surface_alt: "#dbe6ff",
            text: "#1b1e23",
            subtle: "#62697a",
            border: "#d7dbe0",
            accent: "#2563eb",
            note: "#fff8c5",
        }
    }
}

fn cache() -> &'static Mutex<Cache> {
    static CACHE: OnceLock<Mutex<Cache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// PlantUML source is unambiguous only when it has one of its `@start...`
/// guards. Requiring the guard prevents prose containing arrows from being
/// converted into a picture when it appears in an untagged fence or file.
pub fn looks_like_diagram(text: &str) -> bool {
    first_meaningful_line(text).is_some_and(|line| {
        let line = line.to_ascii_lowercase();
        line == "@startuml"
            || line.starts_with("@startuml ")
            || line == "@startmindmap"
            || line == "@startwbs"
    })
}

fn first_meaningful_line(text: &str) -> Option<&str> {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty() && !line.starts_with('\''))
}

pub fn render(source: &str) -> Option<&'static Diagram> {
    if source.len() > MAX_SOURCE_BYTES {
        return None;
    }
    let mut hasher = FxHasher::default();
    source.hash(&mut hasher);
    let key = hasher.finish();
    if let Ok(cache) = cache().lock()
        && let Some(hit) = cache.get(&key)
    {
        return *hit;
    }

    let rendered = render_uncached(source);
    if let Ok(mut cache) = cache().lock() {
        if cache.len() >= CACHE_LIMIT {
            cache.clear();
        }
        cache.insert(key, rendered);
    }
    rendered
}

fn render_uncached(source: &str) -> Option<&'static Diagram> {
    let body = diagram_body(source)?;
    let parsed = parse(&body)?;
    let light_svg = parsed.svg(palette(false))?;
    let dark_svg = parsed.svg(palette(true))?;
    let (width, height) = parsed.size();
    let light_png = rasterize(&light_svg)?;
    let dark_png = rasterize(&dark_svg)?;
    Some(Box::leak(Box::new(Diagram {
        light: svg_uri(&light_svg),
        dark: svg_uri(&dark_svg),
        light_png: png_uri(&light_png),
        dark_png: png_uri(&dark_png),
        width,
        height,
    })))
}

fn diagram_body(source: &str) -> Option<Vec<String>> {
    let lines: Vec<_> = source.lines().map(str::trim).collect();
    let start = lines
        .iter()
        .position(|line| !line.is_empty() && !line.starts_with('\''))?;
    let opening = lines[start].to_ascii_lowercase();
    if opening != "@startuml"
        && !opening.starts_with("@startuml ")
        && opening != "@startmindmap"
        && opening != "@startwbs"
    {
        return None;
    }
    // Mindmap and WBS have distinct indentation grammars. Detection is kept
    // for future compatibility, but partial rendering would be misleading.
    if !opening.starts_with("@startuml") {
        return None;
    }
    let end = lines
        .iter()
        .enumerate()
        .skip(start + 1)
        .find(|(_, line)| line.eq_ignore_ascii_case("@enduml"))?
        .0;
    if lines[end + 1..]
        .iter()
        .any(|line| !line.is_empty() && !line.starts_with('\''))
    {
        return None;
    }
    let mut body = Vec::new();
    let mut index = start + 1;
    while index < end {
        let line = lines[index];
        if line.to_ascii_lowercase().starts_with("skinparam ") && line.ends_with('{') {
            index += 1;
            while index < end && lines[index] != "}" {
                index += 1;
            }
            if index == end {
                return None;
            }
        } else {
            body.push(line.to_string());
        }
        index += 1;
    }
    Some(body)
}

enum Parsed {
    Sequence(Sequence),
    Graph(Graph),
}

impl Parsed {
    fn size(&self) -> (f64, f64) {
        match self {
            Self::Sequence(value) => value.size(),
            Self::Graph(value) => value.size(),
        }
    }

    fn svg(&self, palette: Palette) -> Option<String> {
        let (width, height) = self.size();
        if width <= 0.0 || height <= 0.0 || width > MAX_DIMENSION || height > MAX_DIMENSION {
            return None;
        }
        Some(match self {
            Self::Sequence(value) => value.svg(palette),
            Self::Graph(value) => value.svg(palette),
        })
    }
}

fn parse(lines: &[String]) -> Option<Parsed> {
    if looks_like_activity(lines) {
        return parse_activity(lines).map(Parsed::Graph);
    }
    if looks_like_graph(lines) {
        return parse_graph(lines).map(Parsed::Graph);
    }
    if looks_like_sequence(lines) {
        return parse_sequence(lines).map(Parsed::Sequence);
    }
    parse_graph(lines).map(Parsed::Graph)
}

fn looks_like_graph(lines: &[String]) -> bool {
    lines.iter().any(|line| {
        let Some(line) = significant(line) else {
            return false;
        };
        let lower = line.to_ascii_lowercase();
        [
            "abstract class ",
            "class ",
            "interface ",
            "enum ",
            "annotation ",
            "object ",
            "state ",
            "component ",
            "node ",
            "cloud ",
            "artifact ",
            "rectangle ",
            "usecase ",
            "package ",
            "namespace ",
        ]
        .iter()
        .any(|prefix| lower.starts_with(prefix))
            || line.contains("<|")
            || line.contains("|>")
            || line.contains("*--")
            || line.contains("o--")
            || line.contains("..>")
            || line.contains("<..")
            || (split_relation(line).is_some()
                && (line.trim_start().starts_with(['[', '(']) || line.contains("[*]")))
    })
}

fn significant(line: &str) -> Option<&str> {
    let line = line.trim();
    (!line.is_empty() && !line.starts_with('\'')).then_some(line)
}

fn is_cosmetic(line: &str) -> bool {
    let lower = line.to_ascii_lowercase();
    lower.starts_with("skinparam ")
        || lower.starts_with("!pragma ")
        || lower.starts_with("!theme ")
        || lower.starts_with("hide ")
        || lower.starts_with("show ")
        || lower.starts_with("scale ")
        || lower == "left to right direction"
        || lower == "top to bottom direction"
        || lower.starts_with("caption ")
        || lower.starts_with("header ")
        || lower.starts_with("footer ")
        || lower == "allow_mixing"
}

fn title(lines: &[String]) -> Option<String> {
    lines.iter().find_map(|line| {
        significant(line)?
            .strip_prefix("title ")
            .or_else(|| significant(line)?.strip_prefix("Title "))
            .map(clean_label)
    })
}

fn looks_like_sequence(lines: &[String]) -> bool {
    lines.iter().any(|line| {
        let Some(line) = significant(line) else {
            return false;
        };
        is_participant_declaration(line) || split_message(line).is_some()
    })
}

fn looks_like_activity(lines: &[String]) -> bool {
    lines.iter().any(|line| {
        let lower = line.trim().to_ascii_lowercase();
        lower == "start"
            || lower == "stop"
            || (line.trim().starts_with(':') && line.trim().ends_with(';'))
            || lower.starts_with("if (")
            || lower.starts_with("while (")
            || lower == "repeat"
    })
}

#[derive(Clone)]
struct Participant {
    id: String,
    label: String,
    kind: String,
}

enum SeqEvent {
    Message {
        from: String,
        to: String,
        label: String,
        dashed: bool,
        open: bool,
    },
    Note {
        target: String,
        label: String,
    },
    GroupStart(String),
    GroupElse(String),
    GroupEnd,
    Divider(String),
}

struct Sequence {
    participants: Vec<Participant>,
    events: Vec<SeqEvent>,
    title: Option<String>,
}

fn is_participant_declaration(line: &str) -> bool {
    [
        "actor ",
        "participant ",
        "boundary ",
        "control ",
        "entity ",
        "database ",
        "collections ",
        "queue ",
    ]
    .iter()
    .any(|prefix| line.to_ascii_lowercase().starts_with(prefix))
}

fn parse_sequence(lines: &[String]) -> Option<Sequence> {
    let mut participants = Vec::new();
    let mut indexes = HashMap::new();
    let mut events = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        let Some(line) = significant(&lines[index]) else {
            index += 1;
            continue;
        };
        let lower = line.to_ascii_lowercase();
        if is_participant_declaration(line) {
            let (kind, rest) = line.split_once(char::is_whitespace)?;
            let (id, label) = parse_decl_name(rest)?;
            add_participant(&mut participants, &mut indexes, &id, &label, kind);
        } else if let Some((mut from, mut to, label, arrow)) = split_message(line) {
            if arrow.starts_with('<') {
                std::mem::swap(&mut from, &mut to);
            }
            add_participant(&mut participants, &mut indexes, &from, &from, "participant");
            add_participant(&mut participants, &mut indexes, &to, &to, "participant");
            events.push(SeqEvent::Message {
                from,
                to,
                label,
                dashed: arrow.contains("--"),
                open: arrow.contains(">>") || arrow.contains('o'),
            });
        } else if lower.starts_with("note ") {
            let (target, mut label) = parse_sequence_note(line)?;
            if label.is_empty() {
                index += 1;
                let mut body = Vec::new();
                while index < lines.len() && !lines[index].eq_ignore_ascii_case("end note") {
                    body.push(lines[index].clone());
                    index += 1;
                }
                if index == lines.len() {
                    return None;
                }
                label = body.join("\\n");
            }
            if !target.is_empty() {
                add_participant(
                    &mut participants,
                    &mut indexes,
                    &target,
                    &target,
                    "participant",
                );
            }
            events.push(SeqEvent::Note { target, label });
        } else if ["alt", "opt", "loop", "par", "break", "critical", "group"]
            .iter()
            .any(|word| lower == *word || lower.starts_with(&format!("{word} ")))
        {
            events.push(SeqEvent::GroupStart(clean_label(
                line.split_once(' ').map_or(line, |(_, value)| value),
            )));
        } else if lower == "else" || lower.starts_with("else ") {
            events.push(SeqEvent::GroupElse(clean_label(
                line.split_once(' ').map_or("else", |(_, value)| value),
            )));
        } else if lower == "end" {
            events.push(SeqEvent::GroupEnd);
        } else if line.starts_with("==") && line.ends_with("==") {
            events.push(SeqEvent::Divider(clean_label(line.trim_matches('='))));
        } else if lower.starts_with("title ")
            || is_cosmetic(line)
            || lower.starts_with("activate ")
            || lower.starts_with("deactivate ")
            || lower.starts_with("destroy ")
            || lower.starts_with("create ")
            || lower == "autonumber"
            || lower.starts_with("autonumber ")
            || lower == "return"
            || lower.starts_with("return ")
        {
            // Presentation/lifecycle directives do not change the message
            // topology drawn by this compact renderer.
        } else {
            return None;
        }
        if participants.len() > 32 || events.len() > MAX_ITEMS {
            return None;
        }
        index += 1;
    }
    (!participants.is_empty() && !events.is_empty()).then(|| Sequence {
        participants,
        events,
        title: title(lines),
    })
}

fn add_participant(
    participants: &mut Vec<Participant>,
    indexes: &mut HashMap<String, usize>,
    id: &str,
    label: &str,
    kind: &str,
) {
    if let Some(existing) = indexes.get(id).copied() {
        if participants[existing].label == participants[existing].id && label != id {
            participants[existing].label = label.to_string();
            participants[existing].kind = kind.to_ascii_lowercase();
        }
        return;
    }
    indexes.insert(id.to_string(), participants.len());
    participants.push(Participant {
        id: id.to_string(),
        label: label.to_string(),
        kind: kind.to_ascii_lowercase(),
    });
}

fn parse_decl_name(rest: &str) -> Option<(String, String)> {
    let rest = rest.trim();
    if let Some(stripped) = rest.strip_prefix('"') {
        let quote = stripped.find('"')?;
        let label = clean_label(&stripped[..quote]);
        let after = stripped[quote + 1..].trim();
        let id = after
            .strip_prefix("as ")
            .or_else(|| after.strip_prefix("AS "))
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .unwrap_or(&label);
        return Some((clean_id(id), label));
    }
    if let Some((left, right)) = split_ascii_case(rest, " as ") {
        let left = clean_label(left);
        let right = clean_label(right);
        // PlantUML commonly writes `participant "Label" as id`; for the
        // unquoted form, the right-hand side is still the stable identifier.
        return Some((clean_id(&right), left));
    }
    let id = clean_id(rest.split_whitespace().next()?);
    Some((id.clone(), clean_label(rest)))
}

fn split_ascii_case<'a>(value: &'a str, needle: &str) -> Option<(&'a str, &'a str)> {
    let at = value.to_ascii_lowercase().find(needle)?;
    Some((&value[..at], &value[at + needle.len()..]))
}

fn split_message(line: &str) -> Option<(String, String, String, String)> {
    const ARROWS: &[&str] = &[
        "-->>", "<<--", "->>", "<<-", "-->", "<--", "-\\", "\\-", "-/", "/-", "->o", "o<-", "-x",
        "x-", "->", "<-",
    ];
    for arrow in ARROWS {
        if let Some(at) = line.find(arrow) {
            let left = clean_id(line[..at].trim());
            let rest = line[at + arrow.len()..].trim();
            let (right, label) = rest
                .split_once(':')
                .map_or((rest, ""), |(right, label)| (right, label));
            let right = clean_id(right.trim());
            if valid_id(&left) && valid_id(&right) {
                return Some((left, right, clean_label(label), (*arrow).to_string()));
            }
        }
    }
    None
}

fn parse_sequence_note(line: &str) -> Option<(String, String)> {
    let rest = line.get(5..)?.trim();
    let (head, label) = rest.split_once(':').map_or((rest, ""), |parts| parts);
    let lower = head.to_ascii_lowercase();
    let target = if let Some(at) = lower.find(" over ") {
        head[at + 6..]
            .split(',')
            .next()
            .map(str::trim)
            .unwrap_or("")
    } else if let Some(at) = lower.find(" of ") {
        head[at + 4..].trim()
    } else {
        ""
    };
    Some((clean_id(target), clean_label(label)))
}

impl Sequence {
    fn size(&self) -> (f64, f64) {
        let width = (self.participants.len() as f64 * 170.0 + 60.0).max(300.0);
        let title = if self.title.is_some() { 36.0 } else { 0.0 };
        let height = 120.0 + title + self.events.len() as f64 * 58.0;
        (width, height)
    }

    fn svg(&self, p: Palette) -> String {
        let (width, height) = self.size();
        let mut out = svg_start(width, height, p);
        let title_offset = if let Some(title) = &self.title {
            svg_text(&mut out, width / 2.0, 28.0, title, "middle", 18, p.text);
            36.0
        } else {
            0.0
        };
        let step = (width - 60.0) / self.participants.len() as f64;
        let xs: HashMap<_, _> = self
            .participants
            .iter()
            .enumerate()
            .map(|(i, participant)| (participant.id.as_str(), 30.0 + step * (i as f64 + 0.5)))
            .collect();
        let top = 26.0 + title_offset;
        let bottom = height - 28.0;
        for participant in &self.participants {
            let x = xs[participant.id.as_str()];
            let box_width = text_width(&participant.label, 14.0).clamp(86.0, step - 14.0);
            let _ = write!(
                out,
                "<line x1=\"{x:.1}\" y1=\"{:.1}\" x2=\"{x:.1}\" y2=\"{bottom:.1}\" stroke=\"{}\" stroke-dasharray=\"5 5\"/>",
                top + 42.0,
                p.subtle
            );
            let radius = if participant.kind == "actor" { 18 } else { 5 };
            let _ = write!(
                out,
                "<rect x=\"{:.1}\" y=\"{top:.1}\" width=\"{box_width:.1}\" height=\"42\" rx=\"{radius}\" fill=\"{}\" stroke=\"{}\"/>",
                x - box_width / 2.0,
                p.surface,
                p.accent
            );
            svg_text(
                &mut out,
                x,
                top + 26.0,
                &participant.label,
                "middle",
                14,
                p.text,
            );
        }

        let mut y = top + 72.0;
        let mut event_svg = String::new();
        let mut group_svg = String::new();
        let mut groups: Vec<(f64, String)> = Vec::new();
        for event in &self.events {
            match event {
                SeqEvent::Message {
                    from,
                    to,
                    label,
                    dashed,
                    open,
                } => {
                    let (Some(&from_x), Some(&to_x)) = (xs.get(from.as_str()), xs.get(to.as_str()))
                    else {
                        continue;
                    };
                    let dash = if *dashed {
                        " stroke-dasharray=\"6 4\""
                    } else {
                        ""
                    };
                    let marker = if *open {
                        "url(#open-arrow)"
                    } else {
                        "url(#arrow)"
                    };
                    if from == to {
                        let _ = write!(
                            event_svg,
                            "<path d=\"M {from_x:.1} {y:.1} h 48 v 25 h -48\" fill=\"none\" stroke=\"{}\"{dash} marker-end=\"{marker}\"/>",
                            p.subtle
                        );
                    } else {
                        let _ = write!(
                            event_svg,
                            "<line x1=\"{from_x:.1}\" y1=\"{y:.1}\" x2=\"{to_x:.1}\" y2=\"{y:.1}\" stroke=\"{}\"{dash} marker-end=\"{marker}\"/>",
                            p.subtle
                        );
                    }
                    svg_text(
                        &mut event_svg,
                        (from_x + to_x) / 2.0,
                        y - 8.0,
                        label,
                        "middle",
                        13,
                        p.text,
                    );
                }
                SeqEvent::Note { target, label } => {
                    let x = xs.get(target.as_str()).copied().unwrap_or(width / 2.0);
                    let note_width = text_width(label, 13.0).clamp(100.0, 280.0);
                    let _ = write!(
                        event_svg,
                        "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{note_width:.1}\" height=\"36\" rx=\"4\" fill=\"{}\" stroke=\"{}\"/>",
                        (x + 12.0).min(width - note_width - 16.0),
                        y - 23.0,
                        p.note,
                        p.border
                    );
                    svg_text(
                        &mut event_svg,
                        (x + 22.0).min(width - note_width - 6.0),
                        y,
                        label,
                        "start",
                        13,
                        p.text,
                    );
                }
                SeqEvent::GroupStart(label) => groups.push((y - 28.0, label.clone())),
                SeqEvent::GroupElse(label) => {
                    let _ = write!(
                        event_svg,
                        "<line x1=\"20\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" stroke=\"{}\" stroke-dasharray=\"4 3\"/>",
                        width - 20.0,
                        p.border
                    );
                    svg_text(&mut event_svg, 30.0, y - 7.0, label, "start", 12, p.subtle);
                }
                SeqEvent::GroupEnd => {
                    if let Some((start, label)) = groups.pop() {
                        let _ = write!(
                            group_svg,
                            "<rect x=\"18\" y=\"{start:.1}\" width=\"{:.1}\" height=\"{:.1}\" rx=\"4\" fill=\"none\" stroke=\"{}\"/>",
                            width - 36.0,
                            y - start + 18.0,
                            p.border
                        );
                        svg_text(
                            &mut group_svg,
                            27.0,
                            start + 17.0,
                            &label,
                            "start",
                            12,
                            p.accent,
                        );
                    }
                }
                SeqEvent::Divider(label) => {
                    let _ = write!(
                        event_svg,
                        "<line x1=\"20\" y1=\"{y:.1}\" x2=\"{:.1}\" y2=\"{y:.1}\" stroke=\"{}\"/>",
                        width - 20.0,
                        p.border
                    );
                    svg_text(
                        &mut event_svg,
                        width / 2.0,
                        y - 7.0,
                        label,
                        "middle",
                        13,
                        p.text,
                    );
                }
            }
            y += 58.0;
        }
        out.push_str(&group_svg);
        out.push_str(&event_svg);
        out.push_str("</svg>");
        out
    }
}

#[derive(Clone)]
struct GraphNode {
    label: String,
    kind: String,
    members: Vec<String>,
}

#[derive(Clone)]
struct GraphEdge {
    from: usize,
    to: usize,
    label: String,
    dashed: bool,
    inheritance: bool,
    tail_diamond: bool,
}

struct Graph {
    nodes: Vec<GraphNode>,
    edges: Vec<GraphEdge>,
    title: Option<String>,
    left_to_right: bool,
    positions: Vec<(f64, f64)>,
    width: f64,
    height: f64,
}

fn parse_graph(lines: &[String]) -> Option<Graph> {
    let mut nodes = Vec::new();
    let mut indexes = HashMap::new();
    let mut raw_edges = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        let Some(line) = significant(&lines[index]) else {
            index += 1;
            continue;
        };
        let lower = line.to_ascii_lowercase();
        if is_graph_container(line) {
            // Containers are layout hints. Their children are still parsed;
            // the closing brace is accepted below.
        } else if let Some((kind, rest)) = declaration(line) {
            let (id, label) = parse_graph_name(rest)?;
            let mut members = Vec::new();
            if line.ends_with('{') {
                index += 1;
                while index < lines.len() && lines[index].trim() != "}" {
                    if let Some(member) = significant(&lines[index]) {
                        members.push(clean_label(member));
                    }
                    index += 1;
                }
                if index == lines.len() {
                    return None;
                }
            }
            upsert_node(&mut nodes, &mut indexes, id, label, kind, members);
        } else if let Some(edge) = split_relation(line) {
            raw_edges.push(edge);
        } else if lower.starts_with("title ") || is_cosmetic(line) {
        } else if lower.starts_with("package ")
            || lower.starts_with("namespace ")
            || lower.starts_with("folder ")
            || lower.starts_with("frame ")
            || line == "{"
            || line == "}"
        {
            // Containers are layout hints. Their child declarations and
            // relationships remain faithfully represented.
        } else if lower.starts_with("note ") || lower == "end note" {
            // Notes do not alter graph topology; multiline note bodies are
            // accepted as presentation-only until their terminator.
            if !line.contains(':') && lower != "end note" {
                index += 1;
                while index < lines.len() && !lines[index].eq_ignore_ascii_case("end note") {
                    index += 1;
                }
                if index == lines.len() {
                    return None;
                }
            }
        } else {
            return None;
        }
        if nodes.len() + raw_edges.len() > MAX_ITEMS {
            return None;
        }
        index += 1;
    }

    let mut edges = Vec::new();
    for raw in raw_edges {
        let from = ensure_node(&mut nodes, &mut indexes, &raw.from);
        let to = ensure_node(&mut nodes, &mut indexes, &raw.to);
        edges.push(GraphEdge {
            from,
            to,
            label: raw.label,
            dashed: raw.arrow.contains(".."),
            inheritance: raw.arrow.contains("<|") || raw.arrow.contains("|>"),
            tail_diamond: raw.arrow.contains("*--") || raw.arrow.contains("o--"),
        });
    }
    if nodes.is_empty() || (nodes.len() == 1 && edges.is_empty()) {
        return None;
    }
    Graph::layout(
        nodes,
        edges,
        title(lines),
        lines
            .iter()
            .any(|line| line.eq_ignore_ascii_case("left to right direction")),
    )
}

fn is_graph_container(line: &str) -> bool {
    if !line.trim_end().ends_with('{') {
        return false;
    }
    let lower = line.trim_start().to_ascii_lowercase();
    [
        "package ",
        "namespace ",
        "folder ",
        "frame ",
        "rectangle ",
        "component ",
        "node ",
        "cloud ",
    ]
    .iter()
    .any(|prefix| lower.starts_with(prefix))
}

fn declaration(line: &str) -> Option<(&str, &str)> {
    const KINDS: &[&str] = &[
        "abstract class",
        "class",
        "interface",
        "enum",
        "annotation",
        "entity",
        "object",
        "state",
        "component",
        "database",
        "node",
        "cloud",
        "artifact",
        "rectangle",
        "usecase",
        "actor",
        "queue",
        "collections",
    ];
    let lower = line.to_ascii_lowercase();
    KINDS.iter().find_map(|kind| {
        lower
            .strip_prefix(kind)
            .filter(|rest| rest.starts_with(char::is_whitespace))
            .map(|_| {
                (
                    *kind,
                    line[kind.len()..].trim().trim_end_matches('{').trim(),
                )
            })
    })
}

fn parse_graph_name(rest: &str) -> Option<(String, String)> {
    let rest = rest.trim();
    if rest.starts_with('[') && rest.contains(']') {
        let end = rest.find(']')?;
        let label = clean_label(&rest[..=end]);
        let after = rest[end + 1..].trim();
        let id = after
            .strip_prefix("as ")
            .or_else(|| after.strip_prefix("AS "))
            .map(clean_id)
            .unwrap_or_else(|| clean_id(&label));
        return Some((id, label));
    }
    parse_decl_name(rest)
}

fn upsert_node(
    nodes: &mut Vec<GraphNode>,
    indexes: &mut HashMap<String, usize>,
    id: String,
    label: String,
    kind: &str,
    members: Vec<String>,
) -> usize {
    if let Some(&index) = indexes.get(&id) {
        nodes[index].label = label;
        nodes[index].kind = kind.to_string();
        nodes[index].members = members;
        return index;
    }
    let index = nodes.len();
    indexes.insert(id.clone(), index);
    nodes.push(GraphNode {
        label,
        kind: kind.to_string(),
        members,
    });
    index
}

fn ensure_node(
    nodes: &mut Vec<GraphNode>,
    indexes: &mut HashMap<String, usize>,
    value: &str,
) -> usize {
    if let Some(&index) = indexes.get(value) {
        return index;
    }
    upsert_node(
        nodes,
        indexes,
        value.to_string(),
        clean_label(value),
        if value == "[*]" { "endpoint" } else { "class" },
        Vec::new(),
    )
}

struct RawEdge {
    from: String,
    to: String,
    label: String,
    arrow: String,
}

fn split_relation(line: &str) -> Option<RawEdge> {
    const ARROWS: &[&str] = &[
        "<|..", "..|>", "<|--", "--|>", "*--", "--*", "o--", "--o", "..>", "<..", "-->", "<--",
        "-down->", "-up->", "-left->", "-right->", "--", "..", "->", "<-",
    ];
    for arrow in ARROWS {
        let Some(at) = line.find(arrow) else {
            continue;
        };
        let left = relation_endpoint(&line[..at], true)?;
        let rest = line[at + arrow.len()..].trim();
        let (right, label) = rest.split_once(':').map_or((rest, ""), |parts| parts);
        let right = relation_endpoint(right, false)?;
        let (from, to) = if arrow.starts_with('<') || arrow.ends_with('*') || arrow.ends_with('o') {
            (right, left)
        } else {
            (left, right)
        };
        return Some(RawEdge {
            from,
            to,
            label: clean_label(label),
            arrow: (*arrow).to_string(),
        });
    }
    None
}

fn relation_endpoint(value: &str, take_last: bool) -> Option<String> {
    let value = value.trim();
    if let Some(start) = value.find('[')
        && let Some(end) = value.rfind(']')
        && end > start
    {
        return Some(clean_id(&value[start..=end]));
    }
    if let Some(start) = value.find('(')
        && let Some(end) = value.rfind(')')
        && end > start
    {
        return Some(clean_id(&value[start..=end]));
    }
    let tokens: Vec<_> = value
        .split_whitespace()
        .filter(|token| !(token.starts_with('"') && token.ends_with('"')))
        .collect();
    let token = if take_last {
        tokens.last()?
    } else {
        tokens.first()?
    };
    let id = clean_id(token);
    valid_id(&id).then_some(id)
}

impl Graph {
    fn layout(
        nodes: Vec<GraphNode>,
        edges: Vec<GraphEdge>,
        title: Option<String>,
        left_to_right: bool,
    ) -> Option<Self> {
        let mut indegree = vec![0usize; nodes.len()];
        let mut outgoing = vec![Vec::new(); nodes.len()];
        for edge in &edges {
            if edge.from != edge.to {
                indegree[edge.to] += 1;
                outgoing[edge.from].push(edge.to);
            }
        }
        let mut queue: VecDeque<_> = indegree
            .iter()
            .enumerate()
            .filter_map(|(index, degree)| (*degree == 0).then_some(index))
            .collect();
        let mut layers = vec![0usize; nodes.len()];
        let mut visited = vec![false; nodes.len()];
        while let Some(node) = queue.pop_front() {
            visited[node] = true;
            for &next in &outgoing[node] {
                layers[next] = layers[next].max(layers[node] + 1);
                indegree[next] -= 1;
                if indegree[next] == 0 {
                    queue.push_back(next);
                }
            }
        }
        // Cycles have no topological root. Put each remaining node one layer
        // after the previous one, producing a stable layout without looping.
        let mut cycle_layer = layers.iter().copied().max().unwrap_or(0);
        for index in 0..nodes.len() {
            if !visited[index] {
                layers[index] = cycle_layer;
                cycle_layer += 1;
            }
        }
        let max_layer = layers.iter().copied().max().unwrap_or(0);
        let mut per_layer = vec![Vec::new(); max_layer + 1];
        for (node, layer) in layers.iter().copied().enumerate() {
            per_layer[layer].push(node);
        }
        let max_across = per_layer.iter().map(Vec::len).max().unwrap_or(1);
        let title_height = if title.is_some() { 42.0 } else { 0.0 };
        let node_step = nodes
            .iter()
            .map(|node| 120.0 + node.members.len() as f64 * 21.0)
            .fold(150.0, f64::max);
        let (width, height) = if left_to_right {
            (
                80.0 + per_layer.len() as f64 * 230.0,
                80.0 + title_height + max_across as f64 * node_step,
            )
        } else {
            (
                80.0 + max_across as f64 * 230.0,
                80.0 + title_height + per_layer.len() as f64 * node_step,
            )
        };
        if width > MAX_DIMENSION || height > MAX_DIMENSION {
            return None;
        }
        let mut positions = vec![(0.0, 0.0); nodes.len()];
        for (layer, group) in per_layer.iter().enumerate() {
            for (slot, node) in group.iter().copied().enumerate() {
                let centered_x = (slot as f64 + 0.5 - group.len() as f64 / 2.0) * 230.0;
                let centered_y = (slot as f64 + 0.5 - group.len() as f64 / 2.0) * node_step;
                positions[node] = if left_to_right {
                    (
                        155.0 + layer as f64 * 230.0,
                        height / 2.0 + centered_y + title_height / 2.0,
                    )
                } else {
                    (
                        width / 2.0 + centered_x,
                        110.0 + title_height + layer as f64 * node_step,
                    )
                };
            }
        }
        Some(Self {
            nodes,
            edges,
            title,
            left_to_right,
            positions,
            width,
            height,
        })
    }

    fn size(&self) -> (f64, f64) {
        (self.width, self.height)
    }

    fn svg(&self, p: Palette) -> String {
        let mut out = svg_start(self.width, self.height, p);
        if let Some(title) = &self.title {
            svg_text(
                &mut out,
                self.width / 2.0,
                29.0,
                title,
                "middle",
                18,
                p.text,
            );
        }
        for edge in &self.edges {
            let (from_x, from_y) = self.positions[edge.from];
            let (to_x, to_y) = self.positions[edge.to];
            let dash = if edge.dashed {
                " stroke-dasharray=\"6 4\""
            } else {
                ""
            };
            let marker = if edge.inheritance {
                "url(#triangle)"
            } else {
                "url(#arrow)"
            };
            let (sx, sy, ex, ey) = if self.left_to_right {
                (from_x + 95.0, from_y, to_x - 95.0, to_y)
            } else {
                (from_x, from_y + 42.0, to_x, to_y - 42.0)
            };
            if edge.from == edge.to {
                let _ = write!(
                    out,
                    "<path d=\"M {from_x:.1} {from_y:.1} h 125 v 80 h -125\" fill=\"none\" stroke=\"{}\"{dash} marker-end=\"{marker}\"/>",
                    p.subtle
                );
            } else {
                let mid = if self.left_to_right {
                    format!(
                        "M {sx:.1} {sy:.1} H {:.1} V {ey:.1} H {ex:.1}",
                        (sx + ex) / 2.0
                    )
                } else {
                    format!(
                        "M {sx:.1} {sy:.1} V {:.1} H {ex:.1} V {ey:.1}",
                        (sy + ey) / 2.0
                    )
                };
                let _ = write!(
                    out,
                    "<path d=\"{mid}\" fill=\"none\" stroke=\"{}\"{dash} marker-end=\"{marker}\"/>",
                    p.subtle
                );
                if edge.tail_diamond {
                    let _ = write!(
                        out,
                        "<path d=\"M {sx:.1} {sy:.1} l -7 -5 l -7 5 l 7 5 z\" fill=\"{}\" stroke=\"{}\"/>",
                        p.surface, p.subtle
                    );
                }
            }
            svg_text(
                &mut out,
                (from_x + to_x) / 2.0 + 6.0,
                (from_y + to_y) / 2.0 - 7.0,
                &edge.label,
                "start",
                12,
                p.subtle,
            );
        }
        for (index, node) in self.nodes.iter().enumerate() {
            let (x, y) = self.positions[index];
            if node.kind == "endpoint" {
                let _ = write!(
                    out,
                    "<circle cx=\"{x:.1}\" cy=\"{y:.1}\" r=\"10\" fill=\"{}\" stroke=\"{}\"/>",
                    p.text, p.text
                );
                continue;
            }
            if node.kind == "decision" {
                let _ = write!(
                    out,
                    "<polygon points=\"{x:.1},{:.1} {:.1},{y:.1} {x:.1},{:.1} {:.1},{y:.1}\" fill=\"{}\" stroke=\"{}\"/>",
                    y - 42.0,
                    x + 82.0,
                    y + 42.0,
                    x - 82.0,
                    p.surface_alt,
                    p.accent
                );
                svg_text(&mut out, x, y + 5.0, &node.label, "middle", 13, p.text);
                continue;
            }
            let node_width = text_width(&node.label, 14.0).clamp(150.0, 210.0);
            let node_height = 50.0 + node.members.len() as f64 * 21.0;
            let rounded = matches!(node.kind.as_str(), "state" | "usecase" | "actor");
            let fill = if matches!(node.kind.as_str(), "interface" | "abstract class") {
                p.surface_alt
            } else {
                p.surface
            };
            let _ = write!(
                out,
                "<rect x=\"{:.1}\" y=\"{:.1}\" width=\"{node_width:.1}\" height=\"{node_height:.1}\" rx=\"{}\" fill=\"{fill}\" stroke=\"{}\"/>",
                x - node_width / 2.0,
                y - node_height / 2.0,
                if rounded { 16 } else { 5 },
                p.accent
            );
            if !matches!(node.kind.as_str(), "class" | "state" | "object") {
                svg_text(
                    &mut out,
                    x,
                    y - node_height / 2.0 + 17.0,
                    &format!("«{}»", node.kind),
                    "middle",
                    10,
                    p.subtle,
                );
            }
            svg_text(
                &mut out,
                x,
                y - node_height / 2.0 + 36.0,
                &node.label,
                "middle",
                14,
                p.text,
            );
            if !node.members.is_empty() {
                let line_y = y - node_height / 2.0 + 45.0;
                let _ = write!(
                    out,
                    "<line x1=\"{:.1}\" y1=\"{line_y:.1}\" x2=\"{:.1}\" y2=\"{line_y:.1}\" stroke=\"{}\"/>",
                    x - node_width / 2.0,
                    x + node_width / 2.0,
                    p.border
                );
                for (member, text) in node.members.iter().enumerate() {
                    svg_text(
                        &mut out,
                        x - node_width / 2.0 + 9.0,
                        line_y + 17.0 + member as f64 * 21.0,
                        text,
                        "start",
                        12,
                        p.text,
                    );
                }
            }
        }
        out.push_str("</svg>");
        out
    }
}

enum ActivityControl {
    If {
        decision: usize,
        ends: Vec<usize>,
        had_else: bool,
    },
    While {
        decision: usize,
    },
}

fn activity_edge(edges: &mut Vec<GraphEdge>, from: usize, to: usize, label: String) {
    edges.push(GraphEdge {
        from,
        to,
        label,
        dashed: false,
        inheritance: false,
        tail_diamond: false,
    });
}

fn parse_activity(lines: &[String]) -> Option<Graph> {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut current = None;
    let mut controls = Vec::new();
    let mut repeats = Vec::new();
    let mut pending_label = String::new();
    for line in lines {
        let Some(line) = significant(line) else {
            continue;
        };
        let lower = line.to_ascii_lowercase();
        let make_node = |nodes: &mut Vec<GraphNode>, label: String, kind: &str| {
            let index = nodes.len();
            nodes.push(GraphNode {
                label,
                kind: kind.to_string(),
                members: Vec::new(),
            });
            index
        };
        let next = if lower == "start" {
            Some(make_node(&mut nodes, "start".into(), "endpoint"))
        } else if lower == "stop" || lower == "end" {
            Some(make_node(&mut nodes, "stop".into(), "endpoint"))
        } else if line.starts_with(':') && line.ends_with(';') {
            Some(make_node(
                &mut nodes,
                clean_label(line.trim_start_matches(':').trim_end_matches(';')),
                "state",
            ))
        } else if lower.starts_with("if (") {
            let label = between_parentheses(line).unwrap_or_else(|| "condition".into());
            let decision = make_node(&mut nodes, label, "decision");
            controls.push(ActivityControl::If {
                decision,
                ends: Vec::new(),
                had_else: false,
            });
            if let Some(previous) = current.replace(decision) {
                activity_edge(
                    &mut edges,
                    previous,
                    decision,
                    std::mem::take(&mut pending_label),
                );
            }
            pending_label = "yes".into();
            None
        } else if lower.starts_with("while (") {
            let label = between_parentheses(line).unwrap_or_else(|| "condition".into());
            let decision = make_node(&mut nodes, label, "decision");
            controls.push(ActivityControl::While { decision });
            if let Some(previous) = current.replace(decision) {
                activity_edge(
                    &mut edges,
                    previous,
                    decision,
                    std::mem::take(&mut pending_label),
                );
            }
            pending_label = "yes".into();
            None
        } else if lower == "else" || lower.starts_with("else ") {
            let ActivityControl::If {
                decision,
                ends,
                had_else,
            } = controls.last_mut()?
            else {
                return None;
            };
            if let Some(end) = current.take()
                && end != *decision
            {
                ends.push(end);
            }
            *had_else = true;
            current = Some(*decision);
            pending_label = between_parentheses(line).unwrap_or_else(|| "no".into());
            None
        } else if lower == "endif" {
            let ActivityControl::If {
                decision,
                mut ends,
                had_else,
            } = controls.pop()?
            else {
                return None;
            };
            if let Some(end) = current.take()
                && end != decision
            {
                ends.push(end);
            }
            let merge = make_node(&mut nodes, String::new(), "endpoint");
            for end in ends {
                activity_edge(&mut edges, end, merge, String::new());
            }
            if !had_else {
                activity_edge(&mut edges, decision, merge, "no".into());
            }
            current = Some(merge);
            None
        } else if lower == "endwhile" {
            let ActivityControl::While { decision } = controls.pop()? else {
                return None;
            };
            if let Some(end) = current.take()
                && end != decision
            {
                activity_edge(&mut edges, end, decision, String::new());
            }
            let merge = make_node(&mut nodes, String::new(), "endpoint");
            activity_edge(&mut edges, decision, merge, "no".into());
            current = Some(merge);
            None
        } else if lower == "repeat" {
            repeats.push(current?);
            None
        } else if lower.starts_with("repeat while") {
            let start = repeats.pop()?;
            let decision = make_node(
                &mut nodes,
                between_parentheses(line).unwrap_or_else(|| "condition".into()),
                "decision",
            );
            activity_edge(&mut edges, current?, decision, String::new());
            activity_edge(&mut edges, decision, start, "yes".into());
            let merge = make_node(&mut nodes, String::new(), "endpoint");
            activity_edge(&mut edges, decision, merge, "no".into());
            current = Some(merge);
            None
        } else if lower.starts_with("title ") || is_cosmetic(line) {
            None
        } else {
            return None;
        };
        if let Some(next) = next {
            if let Some(previous) = current {
                activity_edge(
                    &mut edges,
                    previous,
                    next,
                    std::mem::take(&mut pending_label),
                );
            }
            current = Some(next);
        }
        if nodes.len() > MAX_ITEMS {
            return None;
        }
    }
    if !controls.is_empty() || !repeats.is_empty() || nodes.len() < 2 {
        return None;
    }
    Graph::layout(nodes, edges, title(lines), false)
}

fn between_parentheses(line: &str) -> Option<String> {
    let start = line.find('(')?;
    let end = line[start + 1..].find(')')? + start + 1;
    Some(clean_label(&line[start + 1..end]))
}

fn clean_id(value: &str) -> String {
    if value.trim() == "[*]" {
        return "[*]".to_string();
    }
    value
        .trim()
        .trim_matches('"')
        .trim_matches('[')
        .trim_matches(']')
        .trim_matches('(')
        .trim_matches(')')
        .trim()
        .to_string()
}

fn clean_label(value: &str) -> String {
    let mut value = value.trim();
    if (value.starts_with('"') && value.ends_with('"'))
        || (value.starts_with('[') && value.ends_with(']'))
        || (value.starts_with('(') && value.ends_with(')'))
    {
        value = &value[1..value.len() - 1];
    }
    value
        .trim()
        .replace("\\n", " · ")
        .replace("<b>", "")
        .replace("</b>", "")
}

fn valid_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 256
        && !value.chars().any(|ch| ch.is_control())
        && !value.contains(['<', '>', '{', '}'])
}

fn text_width(value: &str, font_size: f64) -> f64 {
    value.chars().count() as f64 * font_size * 0.58 + 26.0
}

fn svg_start(width: f64, height: f64, p: Palette) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width:.0} {height:.0}\" width=\"{width:.0}\" height=\"{height:.0}\">\
         <rect width=\"100%\" height=\"100%\" fill=\"{}\"/>\
         <defs>\
         <marker id=\"arrow\" markerWidth=\"10\" markerHeight=\"10\" refX=\"9\" refY=\"3\" orient=\"auto\" markerUnits=\"strokeWidth\"><path d=\"M0,0 L0,6 L9,3 z\" fill=\"{}\"/></marker>\
         <marker id=\"open-arrow\" markerWidth=\"10\" markerHeight=\"10\" refX=\"9\" refY=\"3\" orient=\"auto\" markerUnits=\"strokeWidth\"><path d=\"M0,0 L9,3 L0,6\" fill=\"none\" stroke=\"{}\"/></marker>\
         <marker id=\"triangle\" markerWidth=\"12\" markerHeight=\"12\" refX=\"10\" refY=\"5\" orient=\"auto\" markerUnits=\"strokeWidth\"><path d=\"M0,0 L10,5 L0,10 z\" fill=\"{}\" stroke=\"{}\"/></marker>\
         </defs>",
        p.canvas, p.subtle, p.subtle, p.canvas, p.subtle
    )
}

fn svg_text(out: &mut String, x: f64, y: f64, value: &str, anchor: &str, size: u32, colour: &str) {
    if value.is_empty() {
        return;
    }
    let _ = write!(
        out,
        "<text x=\"{x:.1}\" y=\"{y:.1}\" text-anchor=\"{anchor}\" font-family=\"Red Hat Text, sans-serif\" font-size=\"{size}\" fill=\"{colour}\">{}</text>",
        escape_xml(value)
    );
}

fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

fn svg_uri(svg: &str) -> String {
    format!(
        "data:image/svg+xml;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(svg.as_bytes())
    )
}

fn png_uri(png: &[u8]) -> String {
    format!(
        "data:image/png;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(png)
    )
}

fn rasterize(svg: &str) -> Option<Vec<u8>> {
    static FONT: &[u8] = include_bytes!("../../../../assets/fonts/RedHatText-Regular.ttf");
    let mut options = resvg::usvg::Options {
        font_family: "Red Hat Text".to_string(),
        ..resvg::usvg::Options::default()
    };
    options.fontdb_mut().load_font_data(FONT.to_vec());
    let tree = resvg::usvg::Tree::from_str(svg, &options).ok()?;
    let size = tree.size().to_int_size();
    if u64::from(size.width()) * u64::from(size.height()) > 16_777_216 {
        return None;
    }
    let mut pixmap = resvg::tiny_skia::Pixmap::new(size.width(), size.height())?;
    resvg::render(
        &tree,
        resvg::tiny_skia::Transform::default(),
        &mut pixmap.as_mut(),
    );
    pixmap.encode_png().ok()
}

pub fn find_in_text(text: &str) -> Vec<Found> {
    if looks_like_diagram(text)
        && let Some(diagram) = render(text.trim())
    {
        return vec![Found {
            diagram,
            source: text.trim().to_string(),
        }];
    }
    let Ok(tree) = markdown::to_mdast(text, &markdown::ParseOptions::gfm()) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    collect_from_nodes(&tree, &mut found);
    found
}

fn collect_from_nodes(node: &markdown::mdast::Node, found: &mut Vec<Found>) {
    if found.len() >= MAX_PER_ATTACHMENT {
        return;
    }
    if let markdown::mdast::Node::Code(code) = node {
        let language = code.lang.as_deref().map(str::trim);
        let tagged = matches!(language, Some("plantuml" | "puml" | "pu"));
        let untagged = language.unwrap_or("").is_empty();
        if (tagged || (untagged && looks_like_diagram(&code.value)))
            && let Some(diagram) = render(&code.value)
        {
            found.push(Found {
                diagram,
                source: code.value.clone(),
            });
        }
        return;
    }
    if let Some(children) = node.children() {
        for child in children {
            collect_from_nodes(child, found);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEQUENCE: &str = "@startuml\nAlice -> Bob: Authentication Request\nBob --> Alice: Authentication Response\n@enduml";

    #[test]
    fn renders_sequence_to_svg_and_png_for_both_themes() {
        let diagram = render(SEQUENCE).expect("sequence renders");
        assert!(diagram.light.starts_with("data:image/svg+xml;base64,"));
        assert!(diagram.dark.starts_with("data:image/svg+xml;base64,"));
        assert!(diagram.light_png.starts_with("data:image/png;base64,"));
        assert!(diagram.dark_png.starts_with("data:image/png;base64,"));
        assert_ne!(diagram.light, diagram.dark);
        let png = base64::engine::general_purpose::STANDARD
            .decode(
                diagram
                    .light_png
                    .strip_prefix("data:image/png;base64,")
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
    }

    #[test]
    fn renders_class_relationships_and_members() {
        let source = "@startuml\nclass Animal {\n  +name: String\n  +move()\n}\nclass Dog\nAnimal <|-- Dog : extends\n@enduml";
        let diagram = render(source).expect("class diagram renders");
        let svg = decode_svg(&diagram.light);
        assert!(svg.contains("Animal"));
        assert!(svg.contains("Dog"));
        assert!(svg.contains("+move()"));
        assert!(svg.contains("triangle"));
    }

    #[test]
    fn graph_declarations_disambiguate_component_arrows_from_messages() {
        let source = "@startuml\ncomponent [Web console] as web\ndatabase Storage as db\nweb --> db : writes\n@enduml";
        let diagram = render(source).expect("component diagram renders");
        let svg = decode_svg(&diagram.light);
        assert!(svg.contains("Web console"));
        assert!(svg.contains("Storage"));
        assert!(svg.contains("writes"));

        let implicit = "@startuml\n[Web] --> [API]\n@enduml";
        let svg = decode_svg(&render(implicit).expect("implicit components render").light);
        assert!(svg.contains("Web"));
        assert!(svg.contains("API"));

        let state = "@startuml\nskinparam state {\n  BackgroundColor white\n}\n[*] --> Idle\nIdle --> [*]\n@enduml";
        let svg = decode_svg(&render(state).expect("state graph renders").light);
        assert!(svg.contains("Idle"));
        assert_eq!(svg.matches("<circle").count(), 1, "{svg}");

        let usecase = "@startuml\nactor User\nrectangle System {\n  usecase (Log in) as Login\n}\nUser --> Login\n@enduml";
        let svg = decode_svg(&render(usecase).expect("use-case container renders").light);
        assert!(svg.contains("User"));
        assert!(svg.contains("Log in"));
    }

    #[test]
    fn renders_activity_syntax() {
        let source = "@startuml\nstart\n:Read configuration;\nif (valid?) then (yes)\n:Run server;\nelse (no)\n:Report error;\nendif\nstop\n@enduml";
        let diagram = render(source).expect("activity renders");
        let svg = decode_svg(&diagram.dark);
        assert!(svg.contains("Read configuration"));
        assert!(svg.contains("Report error"));
        assert!(
            svg.contains("<polygon"),
            "decision should be a diamond: {svg}"
        );

        let loops = "@startuml\nstart\nwhile (more?)\n:Process item;\nendwhile\nrepeat\n:Wait;\nrepeat while (retry?)\nstop\n@enduml";
        let svg = decode_svg(&render(loops).expect("activity loops render").light);
        assert!(svg.contains("more?"));
        assert!(svg.contains("retry?"));
        assert!(svg.contains("Process item"));
    }

    #[test]
    fn rejects_missing_guards_unknown_syntax_and_external_includes() {
        assert!(render("Alice -> Bob: hi").is_none());
        assert!(render("@startuml\n@enduml").is_none());
        assert!(
            render("@startuml\n!include https://example.com/x.puml\nA -> B\n@enduml").is_none()
        );
        assert!(!looks_like_diagram("the value @startuml appears in prose"));
    }

    #[test]
    fn escapes_untrusted_labels() {
        let diagram = render("@startuml\nA -> B: </text><script>alert(1)</script>\n@enduml")
            .expect("renders safely");
        let svg = decode_svg(&diagram.light);
        assert!(!svg.contains("<script>"));
        assert!(svg.contains("&lt;/text&gt;"));
    }

    #[test]
    fn caches_and_finds_attached_diagrams() {
        assert!(std::ptr::eq(
            render(SEQUENCE).unwrap(),
            render(SEQUENCE).unwrap()
        ));
        let doc = format!(
            "# Design\n\n```plantuml\n{SEQUENCE}\n```\n\n```rust\n@startuml\nA -> B\n@enduml\n```\n"
        );
        let found = find_in_text(&doc);
        assert_eq!(found.len(), 1);
        assert!(found[0].source.contains("Alice -> Bob"));
    }

    fn decode_svg(uri: &str) -> String {
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(uri.strip_prefix("data:image/svg+xml;base64,").unwrap())
            .unwrap();
        String::from_utf8(bytes).unwrap()
    }
}
