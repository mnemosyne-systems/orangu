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

//! Knowledge bundles in the Open Knowledge Format (OKF, v0.3): directories
//! of Markdown concepts with YAML frontmatter, which the model looks things
//! up in with the `knowledge` tool. orangu only reads them.
//!
//! The spec's consumer rules (§11.2) are kept here rather than left to the
//! model: the trust tier comes only from verifications newer than the last
//! change, staleness only from `stale_after` (and a value that does not
//! parse is not fresh), `index.md` and `log.md` are never concepts, unknown
//! keys and types are accepted, and nothing outside a bundle's root is read.

use anyhow::{Result, anyhow};
use chrono::{DateTime, NaiveDate, Utc};
use serde_yaml::{Mapping, Value};
use std::{
    collections::HashSet,
    fmt, fs,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

/// The most of a concept's body `read` returns.
const MAX_READ_BYTES: usize = 64 * 1024;
/// The most concepts `search` lists.
const MAX_SEARCH_RESULTS: usize = 20;
/// The type of a concept that holds a sanctioned computation (§10).
const ATTESTED_COMPUTATION: &str = "Attested Computation";

/// Where a bundle was found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BundleSource {
    /// `[orangu].knowledge`.
    Configured,
    /// `~/.orangu/knowledge/`.
    User,
    /// `<workspace>/.orangu/knowledge/`.
    Project,
    /// The workspace is a bundle itself.
    Workspace,
}

impl fmt::Display for BundleSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            BundleSource::Configured => "configured",
            BundleSource::User => "user",
            BundleSource::Project => "project",
            BundleSource::Workspace => "workspace",
        })
    }
}

/// One bundle: a directory, and what its root `index.md` says about it
/// (§8.1).
#[derive(Debug, Clone)]
pub struct Bundle {
    /// What the tool and `/knowledge` call it: the directory's name.
    pub name: String,
    pub root: PathBuf,
    pub title: Option<String>,
    pub description: Option<String>,
    pub okf_version: Option<String>,
    pub source: BundleSource,
}

/// How far to trust a concept, from its current verifications (§5.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrustTier {
    Unverified,
    MachineConfirmed,
    HumanReviewed,
}

impl fmt::Display for TrustTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            TrustTier::Unverified => "unverified",
            TrustTier::MachineConfirmed => "machine-confirmed",
            TrustTier::HumanReviewed => "human-reviewed",
        })
    }
}

/// Whether a concept needs checking again (§5.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Staleness {
    /// No `stale_after`.
    Fresh,
    FreshUntil(DateTime<Utc>),
    Stale(DateTime<Utc>),
    /// A `stale_after` that does not parse, which is not fresh.
    Malformed(String),
}

impl Staleness {
    pub fn is_fresh(&self) -> bool {
        matches!(self, Staleness::Fresh | Staleness::FreshUntil(_))
    }
}

/// One concept: a Markdown file under a bundle's root with a `type` in its
/// frontmatter (§3.2).
#[derive(Debug, Clone)]
pub struct Concept {
    /// Its path in the bundle without `.md`, e.g. `metrics/revenue`.
    pub id: String,
    pub path: PathBuf,
    /// Its `type`, which can be anything (§4.1).
    pub kind: String,
    pub title: Option<String>,
    pub description: Option<String>,
    pub tags: Vec<String>,
    /// `stable` when absent; any other value is kept as it is (§5.4).
    pub status: String,
    pub superseded_by: Option<String>,
    pub tier: TrustTier,
    /// Verifications older than the last change, which no longer count.
    pub earlier_verifications: usize,
    stale_after: Option<std::result::Result<DateTime<Utc>, String>>,
    /// `(id, title, resource)` of each `sources` entry.
    pub sources: Vec<(Option<String>, Option<String>, String)>,
}

impl Concept {
    pub fn staleness(&self, now: DateTime<Utc>) -> Staleness {
        match &self.stale_after {
            None => Staleness::Fresh,
            Some(Ok(at)) if now >= *at => Staleness::Stale(*at),
            Some(Ok(at)) => Staleness::FreshUntil(*at),
            Some(Err(raw)) => Staleness::Malformed(raw.clone()),
        }
    }

    pub fn is_deprecated(&self) -> bool {
        self.status == "deprecated"
    }

    /// One line on how far to trust it, e.g. `human-reviewed, stable` or
    /// `unverified, draft, stale since 2026-09-23`.
    pub fn trust_line(&self, now: DateTime<Utc>) -> String {
        let mut line = format!("{}, {}", self.tier, self.status);
        match self.staleness(now) {
            Staleness::Fresh => {}
            Staleness::FreshUntil(at) => {
                line.push_str(&format!(", fresh until {}", at.format("%Y-%m-%d")))
            }
            Staleness::Stale(at) => {
                line.push_str(&format!(", stale since {}", at.format("%Y-%m-%d")))
            }
            Staleness::Malformed(raw) => line.push_str(&format!(
                ", stale_after '{raw}' does not parse, so treat it as stale"
            )),
        }
        if self.earlier_verifications > 0 {
            line.push_str(&format!(
                " ({} earlier verification{} before the last change)",
                self.earlier_verifications,
                if self.earlier_verifications == 1 {
                    ""
                } else {
                    "s"
                }
            ));
        }
        line
    }

    fn display_title(&self) -> &str {
        self.title
            .as_deref()
            .unwrap_or_else(|| self.id.rsplit('/').next().unwrap_or(&self.id))
    }
}

/// What `/knowledge` lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Report {
    Bundles,
    Stale,
    Deprecated,
    Unverified,
}

impl Report {
    pub fn parse(word: &str) -> Option<Self> {
        match word {
            "" => Some(Report::Bundles),
            "stale" => Some(Report::Stale),
            "deprecated" => Some(Report::Deprecated),
            "unverified" => Some(Report::Unverified),
            _ => None,
        }
    }
}

/// The bundles orangu found, and what went wrong finding the configured ones.
#[derive(Debug, Clone, Default)]
pub struct KnowledgeRegistry {
    bundles: Vec<Bundle>,
    warnings: Vec<String>,
}

impl KnowledgeRegistry {
    /// The bundles in `[orangu].knowledge` (`configured`, relative paths from
    /// the workspace), in `~/.orangu/knowledge/` and
    /// `<workspace>/.orangu/knowledge/` (each directory in them is one), and
    /// the workspace itself when its `index.md` declares `okf_version`.
    pub fn discover(workspace: &Path, configured: &[PathBuf]) -> Self {
        let mut candidates = Vec::new();
        for path in configured {
            let path = if path.is_relative() {
                workspace.join(path)
            } else {
                path.clone()
            };
            candidates.push((path, BundleSource::Configured));
        }
        if let Some(home) = home::home_dir() {
            candidates.extend(
                subdirectories(&home.join(".orangu").join("knowledge"))
                    .into_iter()
                    .map(|dir| (dir, BundleSource::User)),
            );
        }
        candidates.extend(
            subdirectories(&workspace.join(".orangu").join("knowledge"))
                .into_iter()
                .map(|dir| (dir, BundleSource::Project)),
        );
        if root_metadata(workspace).is_some_and(|meta| get(&meta, "okf_version").is_some()) {
            candidates.push((workspace.to_path_buf(), BundleSource::Workspace));
        }
        Self::from_roots(candidates)
    }

    /// The bundles at `roots`, the first of any that are the same directory.
    pub fn from_roots(roots: impl IntoIterator<Item = (PathBuf, BundleSource)>) -> Self {
        let mut registry = Self::default();
        let mut seen = HashSet::new();
        let mut names = HashSet::new();
        for (path, source) in roots {
            let root = match fs::canonicalize(&path) {
                Ok(root) if root.is_dir() => root,
                _ => {
                    if source == BundleSource::Configured {
                        registry.warnings.push(format!(
                            "[orangu].knowledge names {}, which is not a directory",
                            path.display()
                        ));
                    }
                    continue;
                }
            };
            if !seen.insert(root.clone()) {
                continue;
            }
            let base = root
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| "bundle".to_string());
            let mut name = base.clone();
            let mut n = 2;
            while !names.insert(name.clone()) {
                name = format!("{base}-{n}");
                n += 1;
            }
            let meta = root_metadata(&root).unwrap_or_default();
            registry.bundles.push(Bundle {
                name,
                title: get(&meta, "title").and_then(text),
                description: get(&meta, "description").and_then(text),
                okf_version: get(&meta, "okf_version").and_then(text),
                root,
                source,
            });
        }
        registry
    }

    pub fn bundles(&self) -> &[Bundle] {
        &self.bundles
    }

    pub fn is_empty(&self) -> bool {
        self.bundles.is_empty()
    }

    /// The `knowledge` tool's description: what it does, and each bundle's
    /// name, title and description, which is all the model sees of them
    /// until it asks.
    pub fn tool_description(&self) -> String {
        let mut out = String::from(
            "Look things up in knowledge bundles (Open Knowledge Format). action=list shows a \
             directory's index (`path`, the bundle root when left out); action=search finds \
             concepts matching `query`; action=read returns the concept at `path` (as in the \
             bundle, e.g. metrics/revenue) with how far to trust it. `bundle` names one; leave \
             it out when there is only one, or to search them all. Prefer a human-reviewed, \
             fresh concept, and say so when an answer rests on one that is not. Bundles:",
        );
        for bundle in &self.bundles {
            out.push_str(&format!("\n- {}", bundle.name));
            if let Some(title) = &bundle.title {
                out.push_str(&format!(": {title}"));
            }
            if let Some(description) = &bundle.description {
                out.push_str(&format!(" ({description})"));
            }
        }
        out
    }

    fn bundle(&self, name: Option<&str>) -> Result<&Bundle> {
        let names = || {
            self.bundles
                .iter()
                .map(|bundle| bundle.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        };
        match name.map(str::trim).filter(|name| !name.is_empty()) {
            Some(name) => self
                .bundles
                .iter()
                .find(|bundle| bundle.name.eq_ignore_ascii_case(name))
                .ok_or_else(|| anyhow!("no bundle '{name}'; there are: {}", names())),
            None if self.bundles.len() == 1 => Ok(&self.bundles[0]),
            None if self.bundles.is_empty() => Err(anyhow!("there are no knowledge bundles")),
            None => Err(anyhow!("name the bundle; there are: {}", names())),
        }
    }

    /// A directory's `index.md`, or one made from its concepts when it has
    /// none (§8).
    pub fn list(&self, bundle: Option<&str>, path: Option<&str>) -> Result<String> {
        let bundle = self.bundle(bundle)?;
        let relative = normalize(path.unwrap_or(""))
            .ok_or_else(|| anyhow!("'{}' is not in the bundle", path.unwrap_or("")))?;
        let dir = inside(&bundle.root, &bundle.root.join(&relative))
            .filter(|dir| dir.is_dir())
            .ok_or_else(|| anyhow!("no directory '{relative}' in {}", bundle.name))?;
        let mut out = format!("Bundle {}", bundle.name);
        if let Some(title) = &bundle.title {
            out.push_str(&format!(": {title}"));
        }
        out.push_str(&format!("\nDirectory: /{relative}\n"));
        if relative.is_empty()
            && let Some(description) = &bundle.description
        {
            out.push_str(&format!("{description}\n"));
        }
        out.push_str(
            "Links below are relative to this directory; give read the path from the bundle root.\n\n",
        );
        let index = dir.join("index.md");
        if let Some(index) = inside(&bundle.root, &index).filter(|index| index.is_file()) {
            let text = fs::read_to_string(&index)?;
            let body = split_frontmatter(&text).map_or(text.as_str(), |(_, body)| body);
            out.push_str(body.trim());
            return Ok(out);
        }
        let concepts = scan(bundle);
        let prefix = if relative.is_empty() {
            String::new()
        } else {
            format!("{relative}/")
        };
        let mut lines = Vec::new();
        for concept in &concepts {
            if let Some(name) = concept.id.strip_prefix(&prefix)
                && !name.contains('/')
            {
                let mut line = format!("* [{}]({name}.md)", concept.display_title());
                if let Some(description) = &concept.description {
                    line.push_str(&format!(" - {description}"));
                }
                lines.push(line);
            }
        }
        let mut subdirectories = subdirectories(&dir)
            .into_iter()
            .filter(|sub| inside(&bundle.root, sub).is_some())
            .filter_map(|sub| sub.file_name().map(|n| n.to_string_lossy().into_owned()))
            .collect::<Vec<_>>();
        subdirectories.sort();
        lines.extend(
            subdirectories
                .iter()
                .map(|sub| format!("* [{sub}]({sub}/)")),
        );
        if lines.is_empty() {
            out.push_str("(nothing here)");
        } else {
            out.push_str(&lines.join("\n"));
        }
        Ok(out)
    }

    /// The concepts whose id, type, title, description or tags hold every
    /// word of `query`, best first, current ones before deprecated ones.
    pub fn search(&self, bundle: Option<&str>, query: &str, now: DateTime<Utc>) -> Result<String> {
        let words: Vec<String> = query.split_whitespace().map(str::to_lowercase).collect();
        if words.is_empty() {
            return Err(anyhow!("search needs a 'query'"));
        }
        let bundles: Vec<&Bundle> = match bundle.map(str::trim).filter(|name| !name.is_empty()) {
            Some(_) => vec![self.bundle(bundle)?],
            None => self.bundles.iter().collect(),
        };
        let mut hits = Vec::new();
        for bundle in bundles {
            for concept in scan(bundle) {
                if let Some(score) = score(&concept, &words) {
                    hits.push((bundle, concept, score));
                }
            }
        }
        hits.sort_by(|(a_bundle, a, a_score), (b_bundle, b, b_score)| {
            a.is_deprecated()
                .cmp(&b.is_deprecated())
                .then(b_score.cmp(a_score))
                .then(a_bundle.name.cmp(&b_bundle.name))
                .then(a.id.cmp(&b.id))
        });
        if hits.is_empty() {
            return Ok(format!("Nothing matches '{query}'."));
        }
        let total = hits.len();
        let mut out = String::new();
        for (bundle, concept, _) in hits.iter().take(MAX_SEARCH_RESULTS) {
            out.push_str(&format!(
                "- [{}] {}: {} ({})",
                bundle.name,
                concept.id,
                concept.display_title(),
                concept.kind
            ));
            if let Some(description) = &concept.description {
                out.push_str(&format!(" - {description}"));
            }
            out.push_str(&format!(" [{}]\n", concept.trust_line(now)));
        }
        if total > MAX_SEARCH_RESULTS {
            out.push_str(&format!(
                "({} more; narrow the query)\n",
                total - MAX_SEARCH_RESULTS
            ));
        }
        Ok(out)
    }

    /// The concept at `path`, under a header that says how far to trust it.
    pub fn read(&self, bundle: Option<&str>, path: &str, now: DateTime<Utc>) -> Result<String> {
        let bundle = self.bundle(bundle)?;
        let relative = normalize(path).ok_or_else(|| anyhow!("'{path}' is not in the bundle"))?;
        let name = relative.rsplit('/').next().unwrap_or(&relative);
        if is_reserved(name) || is_reserved(&format!("{name}.md")) {
            return Err(anyhow!(
                "'{relative}' is an index or a log, not a concept; use action=list"
            ));
        }
        let file = [relative.clone(), format!("{relative}.md")]
            .into_iter()
            .filter(|candidate| candidate.ends_with(".md"))
            .find_map(|candidate| {
                inside(&bundle.root, &bundle.root.join(&candidate)).filter(|file| file.is_file())
            })
            .ok_or_else(|| anyhow!("no concept '{relative}' in {}", bundle.name))?;
        let text = fs::read_to_string(&file)?;
        let concept = parse_concept(bundle, &file, &text).ok_or_else(|| {
            anyhow!("'{relative}' has no `type` in its frontmatter, so it is not a concept")
        })?;
        let body = split_frontmatter(&text).map_or("", |(_, body)| body).trim();

        let mut out = format!(
            "Concept: [{}] {} ({})\nTitle: {}\nTrust: {}\n",
            bundle.name,
            concept.id,
            concept.kind,
            concept.display_title(),
            concept.trust_line(now)
        );
        if let Some(description) = &concept.description {
            out.push_str(&format!("Description: {description}\n"));
        }
        if !concept.tags.is_empty() {
            out.push_str(&format!("Tags: {}\n", concept.tags.join(", ")));
        }
        if concept.is_deprecated() {
            match concept
                .superseded_by
                .as_deref()
                .map(|target| resolve_link(bundle, &concept.id, target))
            {
                Some(Link::Concept(id)) => out.push_str(&format!(
                    "Deprecated: superseded by {id}; read that instead.\n"
                )),
                Some(Link::External(target)) => out.push_str(&format!(
                    "Deprecated: superseded by {target}, outside this bundle.\n"
                )),
                None => out.push_str("Deprecated, with nothing named to replace it.\n"),
            }
        }
        if concept.kind == ATTESTED_COMPUTATION {
            out.push_str(
                "Not attested: orangu does not run or attest computations, so a value from it \
                 has not been confirmed.\n",
            );
        }
        if !concept.sources.is_empty() {
            out.push_str("Sources:\n");
            for (id, title, resource) in &concept.sources {
                let label = id.as_deref().unwrap_or("-");
                match title {
                    Some(title) => out.push_str(&format!("  [{label}] {title}: {resource}\n")),
                    None => out.push_str(&format!("  [{label}] {resource}\n")),
                }
            }
        }
        out.push_str("---\n");
        if body.len() > MAX_READ_BYTES {
            let mut cut = MAX_READ_BYTES;
            while !body.is_char_boundary(cut) {
                cut -= 1;
            }
            out.push_str(&body[..cut]);
            out.push_str("\n[the rest is cut off]");
        } else {
            out.push_str(body);
        }
        Ok(out)
    }

    /// What `/knowledge` prints.
    pub fn report(&self, report: Report, now: DateTime<Utc>) -> String {
        let mut out = String::new();
        for warning in &self.warnings {
            out.push_str(&format!("Warning: {warning}\n"));
        }
        if self.bundles.is_empty() {
            out.push_str(
                "No knowledge bundles. Put one in ~/.orangu/knowledge/ or \
                 <workspace>/.orangu/knowledge/, or name it in [orangu].knowledge.\n",
            );
            return out;
        }
        let mut found = false;
        for bundle in &self.bundles {
            let concepts = scan(bundle);
            match report {
                Report::Bundles => {
                    out.push_str(&format!("{} [{}]", bundle.name, bundle.source));
                    if let Some(title) = &bundle.title {
                        out.push_str(&format!(": {title}"));
                    }
                    out.push_str(&format!("\n  {}\n", bundle.root.display()));
                    let count = |keep: &dyn Fn(&Concept) -> bool| {
                        concepts.iter().filter(|concept| keep(concept)).count()
                    };
                    out.push_str(&format!(
                        "  {} concepts: {} stale, {} deprecated, {} unverified, {} draft\n",
                        concepts.len(),
                        count(&|concept| !concept.staleness(now).is_fresh()),
                        count(&|concept| concept.is_deprecated()),
                        count(&|concept| concept.tier == TrustTier::Unverified),
                        count(&|concept| concept.status == "draft"),
                    ));
                    found = true;
                }
                Report::Stale => {
                    for concept in &concepts {
                        let line = match concept.staleness(now) {
                            Staleness::Stale(at) => {
                                format!("stale since {}", at.format("%Y-%m-%d"))
                            }
                            Staleness::Malformed(raw) => {
                                format!("stale_after '{raw}' does not parse")
                            }
                            Staleness::Fresh | Staleness::FreshUntil(_) => continue,
                        };
                        out.push_str(&format!("[{}] {}: {line}\n", bundle.name, concept.id));
                        found = true;
                    }
                }
                Report::Deprecated => {
                    for concept in concepts.iter().filter(|concept| concept.is_deprecated()) {
                        out.push_str(&format!("[{}] {}", bundle.name, concept.id));
                        match concept
                            .superseded_by
                            .as_deref()
                            .map(|target| resolve_link(bundle, &concept.id, target))
                        {
                            Some(Link::Concept(id)) => out.push_str(&format!(" -> {id}")),
                            Some(Link::External(target)) => out.push_str(&format!(" -> {target}")),
                            None => {}
                        }
                        out.push('\n');
                        found = true;
                    }
                }
                Report::Unverified => {
                    for concept in concepts
                        .iter()
                        .filter(|concept| concept.tier == TrustTier::Unverified)
                    {
                        out.push_str(&format!(
                            "[{}] {} ({}, {})\n",
                            bundle.name, concept.id, concept.kind, concept.status
                        ));
                        found = true;
                    }
                }
            }
        }
        if !found {
            out.push_str(match report {
                Report::Stale => "Nothing is stale.\n",
                Report::Deprecated => "Nothing is deprecated.\n",
                Report::Unverified => "Everything has a current verification.\n",
                Report::Bundles => "",
            });
        }
        out
    }
}

/// The frontmatter and body of `text`: it starts with a line that is
/// exactly `---`, and the block ends at the next such line (§4).
fn split_frontmatter(text: &str) -> Option<(&str, &str)> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let rest = text
        .strip_prefix("---\n")
        .or_else(|| text.strip_prefix("---\r\n"))?;
    let mut offset = 0;
    for line in rest.split_inclusive('\n') {
        if line.trim_end_matches(['\r', '\n']) == "---" {
            return Some((&rest[..offset], &rest[offset + line.len()..]));
        }
        offset += line.len();
    }
    None
}

fn frontmatter(text: &str) -> Option<Mapping> {
    let (yaml, _) = split_frontmatter(text)?;
    match serde_yaml::from_str::<Value>(yaml).ok()? {
        Value::Mapping(map) => Some(map),
        _ => None,
    }
}

/// The frontmatter of a bundle's root `index.md`.
fn root_metadata(root: &Path) -> Option<Mapping> {
    frontmatter(&fs::read_to_string(root.join("index.md")).ok()?)
}

/// `key`'s value, with null, an empty string and an empty list read as no
/// value (§11.2).
fn get<'a>(map: &'a Mapping, key: &str) -> Option<&'a Value> {
    map.get(key).filter(|value| match value {
        Value::Null => false,
        Value::String(text) => !text.trim().is_empty(),
        Value::Sequence(items) => !items.is_empty(),
        _ => true,
    })
}

/// A scalar as text.
fn text(value: &Value) -> Option<String> {
    let text = match value {
        Value::String(text) => text.trim().to_string(),
        Value::Number(number) => number.to_string(),
        Value::Bool(flag) => flag.to_string(),
        Value::Tagged(tagged) => return text(&tagged.value),
        _ => return None,
    };
    (!text.is_empty()).then_some(text)
}

/// A list, or one item read as a list of one (§11.2).
fn items(value: &Value) -> Vec<&Value> {
    match value {
        Value::Sequence(items) => items.iter().collect(),
        value => vec![value],
    }
}

/// An RFC 3339 date-time, or a date meaning the start of that day in UTC
/// (§5).
fn timestamp(value: &Value) -> Option<DateTime<Utc>> {
    let raw = text(value)?;
    if let Ok(at) = DateTime::parse_from_rfc3339(&raw).or_else(|_| {
        // YAML's own timestamps may put a space where RFC 3339 has the `T`.
        DateTime::parse_from_rfc3339(&raw.replacen(' ', "T", 1))
    }) {
        return Some(at.with_timezone(&Utc));
    }
    NaiveDate::parse_from_str(&raw, "%Y-%m-%d")
        .ok()
        .and_then(|date| date.and_hms_opt(0, 0, 0))
        .map(|at| at.and_utc())
}

fn is_reserved(name: &str) -> bool {
    name == "index.md" || name == "log.md"
}

/// The directories in `dir`, leaving out the hidden ones (§3.2).
fn subdirectories(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = entries
        .filter_map(|entry| entry.ok())
        .filter(|entry| !entry.file_name().to_string_lossy().starts_with('.'))
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    dirs.sort();
    dirs
}

/// `path` when it is under `root` once links are followed, so a symbolic
/// link out of the bundle is never read (§3.2, §6). `root` is canonical.
fn inside(root: &Path, path: &Path) -> Option<PathBuf> {
    let resolved = fs::canonicalize(path).ok()?;
    resolved.starts_with(root).then_some(resolved)
}

/// A path in a bundle as `a/b`, from the root: a leading `/`, `.` and a
/// `#fragment` dropped, `..` applied and `%XX` decoded. `None` when it
/// leaves the root, names a hidden file or is a URL.
fn normalize(path: &str) -> Option<String> {
    let path = path.trim();
    if has_scheme(path) {
        return None;
    }
    let path = path.split('#').next().unwrap_or("");
    let mut parts: Vec<String> = Vec::new();
    for part in path.split(['/', '\\']) {
        let part = percent_decode(part)?;
        match part.as_str() {
            "" | "." => {}
            ".." => {
                parts.pop()?;
            }
            hidden if hidden.starts_with('.') => return None,
            _ => parts.push(part),
        }
    }
    Some(parts.join("/"))
}

fn has_scheme(value: &str) -> bool {
    value.split_once(':').is_some_and(|(scheme, _)| {
        scheme.len() > 1
            && scheme.starts_with(|c: char| c.is_ascii_alphabetic())
            && scheme
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
    })
}

fn percent_decode(part: &str) -> Option<String> {
    if !part.contains('%') {
        return Some(part.to_string());
    }
    let bytes = part.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            let hex = part.get(i + 1..i + 3)?;
            out.push(u8::from_str_radix(hex, 16).ok()?);
            i += 3;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

enum Link {
    /// A concept in the bundle, by id.
    Concept(String),
    /// A URL, or a path that leaves the bundle or is not a concept in it.
    External(String),
}

/// Where `target`, written in concept `from`, points (§6): a URL, a path
/// from the root when it starts with `/`, and from `from`'s directory
/// otherwise.
fn resolve_link(bundle: &Bundle, from: &str, target: &str) -> Link {
    let target = target.trim();
    let joined = if target.starts_with('/') || has_scheme(target) {
        target.to_string()
    } else {
        match from.rsplit_once('/') {
            Some((dir, _)) => format!("{dir}/{target}"),
            None => target.to_string(),
        }
    };
    let Some(relative) = normalize(&joined) else {
        return Link::External(target.to_string());
    };
    let file = if relative.ends_with(".md") {
        relative.clone()
    } else {
        format!("{relative}.md")
    };
    match inside(&bundle.root, &bundle.root.join(&file)) {
        Some(path) if path.is_file() => Link::Concept(file.trim_end_matches(".md").to_string()),
        _ => Link::External(target.to_string()),
    }
}

/// Every concept in `bundle`, by id.
fn scan(bundle: &Bundle) -> Vec<Concept> {
    let mut concepts = Vec::new();
    for entry in WalkDir::new(&bundle.root)
        .follow_links(false)
        .into_iter()
        .filter_entry(|entry| {
            entry.depth() == 0 || !entry.file_name().to_string_lossy().starts_with('.')
        })
        .filter_map(|entry| entry.ok())
    {
        let name = entry.file_name().to_string_lossy();
        if !name.ends_with(".md") || is_reserved(&name) || entry.file_type().is_dir() {
            continue;
        }
        let Some(path) = inside(&bundle.root, entry.path()).filter(|path| path.is_file()) else {
            continue;
        };
        if let Ok(text) = fs::read_to_string(&path)
            && let Some(concept) = parse_concept(bundle, entry.path(), &text)
        {
            concepts.push(concept);
        }
    }
    concepts.sort_by(|a, b| a.id.cmp(&b.id));
    concepts
}

/// The concept in `text`, at `path` under `bundle`; `None` when it has no
/// frontmatter or no `type` (§3.2).
fn parse_concept(bundle: &Bundle, path: &Path, text: &str) -> Option<Concept> {
    let map = frontmatter(text)?;
    let kind = get(&map, "type").and_then(self::text)?;
    // `path` is under the canonical root: a walk from it, or a canonical path.
    let id = path
        .strip_prefix(&bundle.root)
        .ok()?
        .to_string_lossy()
        .replace('\\', "/");
    let (tier, earlier_verifications) = trust(&map);
    let sources = get(&map, "sources")
        .map(items)
        .unwrap_or_default()
        .into_iter()
        .filter_map(Value::as_mapping)
        .filter_map(|source| {
            // An entry without its required `resource` is ignored (§11.2).
            let resource = get(source, "resource").and_then(self::text)?;
            Some((
                get(source, "id").and_then(self::text),
                get(source, "title").and_then(self::text),
                resource,
            ))
        })
        .collect();
    Some(Concept {
        id: id.trim_end_matches(".md").to_string(),
        path: path.to_path_buf(),
        kind,
        title: get(&map, "title").and_then(self::text),
        description: get(&map, "description").and_then(self::text),
        // A scalar is one tag, never split at commas (§11.2).
        tags: get(&map, "tags")
            .map(items)
            .unwrap_or_default()
            .into_iter()
            .filter_map(self::text)
            .collect(),
        status: get(&map, "status")
            .and_then(self::text)
            .unwrap_or_else(|| "stable".to_string()),
        superseded_by: get(&map, "superseded_by").and_then(self::text),
        tier,
        earlier_verifications,
        stale_after: get(&map, "stale_after")
            .map(|value| timestamp(value).ok_or_else(|| self::text(value).unwrap_or_default())),
        sources,
    })
}

/// The trust tier from the verifications that are current, those at or
/// after `generated.at` (all of them without `generated`), and how many
/// are not (§5.3).
fn trust(map: &Mapping) -> (TrustTier, usize) {
    let generated = get(map, "generated").and_then(Value::as_mapping);
    // `None`: nothing to compare with. `Some(None)`: a time that does not
    // parse, so no verification can be shown to be newer.
    let generated_at = generated
        .and_then(|generated| get(generated, "at"))
        .map(timestamp);
    let mut current = Vec::new();
    let mut earlier = 0;
    for event in get(map, "verified").map(items).unwrap_or_default() {
        let Some(event) = event.as_mapping() else {
            continue;
        };
        // An event needs both `by` and `at` (§5.2).
        let (Some(by), Some(at)) = (
            get(event, "by").and_then(text),
            get(event, "at").and_then(timestamp),
        ) else {
            continue;
        };
        let is_current = match generated_at {
            None => true,
            Some(Some(generated_at)) => at >= generated_at,
            Some(None) => false,
        };
        if is_current {
            current.push(by);
        } else {
            earlier += 1;
        }
    }
    let tier = if current.iter().any(|by| by.starts_with("human:")) {
        TrustTier::HumanReviewed
    } else if current.is_empty() {
        TrustTier::Unverified
    } else {
        TrustTier::MachineConfirmed
    };
    (tier, earlier)
}

/// How well `concept` matches: every word must be somewhere, and one in the
/// title or id counts most.
fn score(concept: &Concept, words: &[String]) -> Option<u32> {
    let title = format!("{} {}", concept.display_title(), concept.id).to_lowercase();
    let labels = format!("{} {}", concept.kind, concept.tags.join(" ")).to_lowercase();
    let description = concept
        .description
        .as_deref()
        .unwrap_or_default()
        .to_lowercase();
    let mut total = 0;
    for word in words {
        total += if title.contains(word) {
            3
        } else if labels.contains(word) {
            2
        } else if description.contains(word) {
            1
        } else {
            return None;
        };
    }
    Some(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn now() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-11-25T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    /// A bundle in a temporary directory, from `(path, text)` pairs.
    fn bundle(files: &[(&str, &str)]) -> (TempDir, KnowledgeRegistry) {
        let dir = TempDir::new().unwrap();
        let root = dir.path().join("retail");
        for (path, text) in files {
            let file = root.join(path);
            fs::create_dir_all(file.parent().unwrap()).unwrap();
            fs::write(file, text).unwrap();
        }
        fs::create_dir_all(&root).unwrap();
        let registry = KnowledgeRegistry::from_roots([(root, BundleSource::Project)]);
        (dir, registry)
    }

    const ROOT_INDEX: &str = "---\nokf_version: \"0.3\"\ntitle: Retail\ndescription: Sales and margin.\n---\n\n# Subdirectories\n\n* [metrics](metrics/index.md) - Headline numbers.\n";

    const MARGIN: &str = "---\ntype: Metric\ntitle: Gross Margin\ndescription: Revenue minus full COGS.\ntags: [finance, margin]\ngenerated: { by: reference_agent/gemini, at: 2026-06-30T14:00:00Z }\nverified:\n  - { by: human:jsmith, at: 2026-07-01T09:00:00Z }\nstale_after: 2026-12-31T00:00:00Z\nx-acme-owner: finance\nsources:\n  - id: standard\n    resource: /policies/standard.md\n    title: Margin standard\n  - title: no resource, ignored\n---\n\n# Definition\n\nRevenue minus COGS.\n";

    const LEGACY: &str = "---\ntype: Metric\ntitle: Gross Margin (legacy)\ndescription: Retired definition.\nstatus: deprecated\nsuperseded_by: /metrics/margin.md\n---\nOld.\n";

    #[test]
    fn frontmatter_starts_and_ends_at_lines_that_are_exactly_dashes() {
        assert_eq!(
            split_frontmatter("---\ntype: A\n---\nbody\n"),
            Some(("type: A\n", "body\n"))
        );
        assert_eq!(
            split_frontmatter("---\r\ntype: A\r\n---\r\nbody"),
            Some(("type: A\r\n", "body"))
        );
        // `----` and `--- x` do not end it, and a file without the opening
        // line has none.
        assert_eq!(split_frontmatter("---\na: ----\n--- x\n"), None);
        assert_eq!(split_frontmatter("# Title\n---\ntype: A\n---\n"), None);
    }

    #[test]
    fn the_bundle_title_and_description_are_what_the_tool_shows() {
        let (_dir, registry) = bundle(&[("index.md", ROOT_INDEX)]);
        let bundle = &registry.bundles()[0];
        assert_eq!(bundle.name, "retail");
        assert_eq!(bundle.title.as_deref(), Some("Retail"));
        assert_eq!(bundle.okf_version.as_deref(), Some("0.3"));
        assert!(
            registry
                .tool_description()
                .ends_with("Bundles:\n- retail: Retail (Sales and margin.)")
        );
    }

    #[test]
    fn index_and_log_files_and_files_without_a_type_are_not_concepts() {
        let (_dir, registry) = bundle(&[
            ("index.md", ROOT_INDEX),
            ("log.md", "---\ntype: Metric\n---\n"),
            ("metrics/index.md", "---\ntype: Metric\n---\n"),
            ("README.md", "# Readme\n"),
            ("notes.md", "---\ntitle: No type\n---\n"),
            ("metrics/margin.md", MARGIN),
            ("metrics/.draft/hidden.md", "---\ntype: Metric\n---\n"),
        ]);
        let ids: Vec<String> = scan(&registry.bundles()[0])
            .into_iter()
            .map(|concept| concept.id)
            .collect();
        assert_eq!(ids, ["metrics/margin"]);
    }

    #[test]
    fn unknown_keys_types_and_status_values_are_kept() {
        let (_dir, registry) = bundle(&[(
            "x.md",
            "---\ntype: Quantum Widget\nstatus: experimental\nx-acme: { nested: [1, 2] }\n---\n",
        )]);
        let concepts = scan(&registry.bundles()[0]);
        assert_eq!(concepts[0].kind, "Quantum Widget");
        assert_eq!(concepts[0].status, "experimental");
        assert_eq!(concepts[0].trust_line(now()), "unverified, experimental");
    }

    #[test]
    fn only_verifications_after_the_last_change_count() {
        let tier =
            |frontmatter: &str| trust(&serde_yaml::from_str::<Mapping>(frontmatter).unwrap());
        // A human checked it, then an agent rewrote it.
        assert_eq!(
            tier(
                "generated: { by: agent/x, at: \"2026-07-02\" }\nverified: { by: human:a, at: \"2026-07-01T09:00:00Z\" }"
            ),
            (TrustTier::Unverified, 1)
        );
        assert_eq!(
            tier(
                "generated: { by: agent/x, at: \"2026-07-01\" }\nverified:\n  - { by: process:nightly, at: \"2026-07-02\" }\n  - { by: human:a, at: \"2026-06-01\" }"
            ),
            (TrustTier::MachineConfirmed, 1)
        );
        // A team is not a person, and without `generated` every event counts.
        assert_eq!(
            tier("verified: [{ by: team:finance, at: \"2026-07-01\" }]"),
            (TrustTier::MachineConfirmed, 0)
        );
        assert_eq!(
            tier("verified: [{ by: human:a, at: \"2026-07-01\" }]"),
            (TrustTier::HumanReviewed, 0)
        );
        // An event without `at` is ignored, and when the last change's time
        // does not parse no event can be shown to be newer.
        assert_eq!(
            tier("verified: [{ by: human:a }]"),
            (TrustTier::Unverified, 0)
        );
        assert_eq!(
            tier(
                "generated: { by: agent/x, at: yesterday }\nverified: [{ by: human:a, at: \"2026-07-01\" }]"
            ),
            (TrustTier::Unverified, 1)
        );
    }

    #[test]
    fn staleness_comes_from_stale_after_and_a_bad_value_is_not_fresh() {
        let stale_after = |value: &str| {
            let (_dir, registry) = bundle(&[(
                "c.md",
                &format!("---\ntype: T\nstale_after: {value}\n---\n"),
            )]);
            scan(&registry.bundles()[0])[0].staleness(now())
        };
        assert!(matches!(
            stale_after("2026-12-31T00:00:00Z"),
            Staleness::FreshUntil(_)
        ));
        assert!(matches!(stale_after("\"2026-09-23\""), Staleness::Stale(_)));
        assert_eq!(stale_after("~"), Staleness::Fresh);
        assert_eq!(stale_after("\"\""), Staleness::Fresh);
        assert_eq!(
            stale_after("next spring"),
            Staleness::Malformed("next spring".into())
        );
        assert!(!Staleness::Malformed(String::new()).is_fresh());
    }

    #[test]
    fn a_scalar_tag_is_one_tag_and_never_split() {
        let (_dir, registry) = bundle(&[("c.md", "---\ntype: T\ntags: sales, margin\n---\n")]);
        assert_eq!(scan(&registry.bundles()[0])[0].tags, ["sales, margin"]);
    }

    #[test]
    fn read_says_how_far_to_trust_the_concept() {
        let (_dir, registry) = bundle(&[("index.md", ROOT_INDEX), ("metrics/margin.md", MARGIN)]);
        let read = registry.read(None, "/metrics/margin.md", now()).unwrap();
        assert!(read.starts_with(
            "Concept: [retail] metrics/margin (Metric)\nTitle: Gross Margin\nTrust: human-reviewed, stable, fresh until 2026-12-31\n"
        ));
        assert!(read.contains(
            "Sources:\n  [standard] Margin standard: /policies/standard.md\n---\n# Definition"
        ));
        assert!(!read.contains("no resource"));
        // The same concept by its id, and once it is stale.
        let later = DateTime::parse_from_rfc3339("2027-01-05T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        assert!(
            registry
                .read(Some("retail"), "metrics/margin", later)
                .unwrap()
                .contains("Trust: human-reviewed, stable, stale since 2026-12-31\n")
        );
    }

    #[test]
    fn a_deprecated_concept_points_at_its_replacement() {
        let (_dir, registry) =
            bundle(&[("metrics/margin.md", MARGIN), ("metrics/legacy.md", LEGACY)]);
        let read = registry.read(None, "metrics/legacy", now()).unwrap();
        assert!(read.contains("Deprecated: superseded by metrics/margin; read that instead.\n"));
        assert_eq!(
            registry.report(Report::Deprecated, now()),
            "[retail] metrics/legacy -> metrics/margin\n"
        );
    }

    #[test]
    fn an_attested_computation_is_never_shown_as_attested() {
        let (_dir, registry) = bundle(&[(
            "computations/revenue.md",
            "---\ntype: Attested Computation\nruntime: bigquery\nattester: { resource: /attesters/sql.py }\n---\n```sql\nSELECT 1\n```\n",
        )]);
        let read = registry.read(None, "computations/revenue", now()).unwrap();
        assert!(read.contains("Not attested: orangu does not run or attest computations"));
    }

    #[test]
    fn nothing_outside_the_bundle_is_read() {
        let (dir, registry) = bundle(&[("metrics/margin.md", MARGIN)]);
        fs::write(
            dir.path().join("secret.md"),
            "---\ntype: Secret\n---\nkey\n",
        )
        .unwrap();
        for path in [
            "../secret.md",
            "/../secret",
            "metrics/../../secret",
            "file:///etc/passwd",
            ".git/config",
        ] {
            assert!(registry.read(None, path, now()).is_err(), "{path} was read");
        }
        assert!(registry.list(None, Some("..")).is_err());
        assert!(registry.read(None, "index", now()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn a_symbolic_link_out_of_the_bundle_is_not_followed() {
        let (dir, registry) = bundle(&[("metrics/margin.md", MARGIN)]);
        let outside = dir.path().join("outside");
        fs::create_dir_all(&outside).unwrap();
        fs::write(outside.join("secret.md"), "---\ntype: Secret\n---\nkey\n").unwrap();
        let root = &registry.bundles()[0].root;
        std::os::unix::fs::symlink(outside.join("secret.md"), root.join("linked.md")).unwrap();
        std::os::unix::fs::symlink(&outside, root.join("linked")).unwrap();
        assert!(registry.read(None, "linked", now()).is_err());
        assert!(registry.read(None, "linked/secret", now()).is_err());
        let ids: Vec<String> = scan(&registry.bundles()[0])
            .into_iter()
            .map(|c| c.id)
            .collect();
        assert_eq!(ids, ["metrics/margin"]);
    }

    #[test]
    fn list_shows_the_index_or_makes_one() {
        let (_dir, registry) = bundle(&[
            ("index.md", ROOT_INDEX),
            ("metrics/margin.md", MARGIN),
            ("metrics/legacy.md", LEGACY),
            ("metrics/derived/ratio.md", "---\ntype: Metric\n---\n"),
        ]);
        let root = registry.list(None, None).unwrap();
        assert!(root.starts_with("Bundle retail: Retail\nDirectory: /\nSales and margin.\n"));
        assert!(root.ends_with("* [metrics](metrics/index.md) - Headline numbers."));
        // `metrics/` has no index.md, so one is made from its concepts.
        let metrics = registry.list(None, Some("metrics/")).unwrap();
        assert!(metrics.ends_with(
            "* [Gross Margin (legacy)](legacy.md) - Retired definition.\n* [Gross Margin](margin.md) - Revenue minus full COGS.\n* [derived](derived/)"
        ));
    }

    #[test]
    fn search_matches_every_word_and_puts_deprecated_concepts_last() {
        let (_dir, registry) =
            bundle(&[("metrics/margin.md", MARGIN), ("metrics/legacy.md", LEGACY)]);
        let found = registry.search(None, "gross MARGIN", now()).unwrap();
        let lines: Vec<&str> = found.lines().collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].starts_with("- [retail] metrics/margin: Gross Margin (Metric) - Revenue minus full COGS. [human-reviewed, stable"));
        assert!(lines[1].starts_with("- [retail] metrics/legacy:"));
        assert!(
            registry
                .search(None, "finance", now())
                .unwrap()
                .contains("metrics/margin")
        );
        assert_eq!(
            registry.search(None, "margin payroll", now()).unwrap(),
            "Nothing matches 'margin payroll'."
        );
    }

    #[test]
    fn the_bundle_must_be_named_when_there_are_several() {
        let dir = TempDir::new().unwrap();
        for name in ["a", "b"] {
            fs::create_dir_all(dir.path().join(name)).unwrap();
        }
        let registry = KnowledgeRegistry::from_roots(
            ["a", "b", "a"].map(|name| (dir.path().join(name), BundleSource::User)),
        );
        assert_eq!(registry.bundles().len(), 2);
        let error = registry.list(None, None).unwrap_err().to_string();
        assert_eq!(error, "name the bundle; there are: a, b");
        assert!(registry.list(Some("B"), None).is_ok());
    }

    #[test]
    fn discovery_finds_project_bundles_and_a_workspace_that_is_one() {
        let workspace = TempDir::new().unwrap();
        fs::create_dir_all(workspace.path().join(".orangu/knowledge/retail")).unwrap();
        fs::write(workspace.path().join("index.md"), ROOT_INDEX).unwrap();
        let registry = KnowledgeRegistry::discover(workspace.path(), &[PathBuf::from("missing")]);
        let mine: Vec<(String, BundleSource)> = registry
            .bundles()
            .iter()
            .filter(|bundle| bundle.source != BundleSource::User)
            .map(|bundle| (bundle.name.clone(), bundle.source))
            .collect();
        let workspace_name = workspace
            .path()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        assert_eq!(
            mine,
            [
                ("retail".to_string(), BundleSource::Project),
                (workspace_name, BundleSource::Workspace)
            ]
        );
        assert!(
            registry
                .report(Report::Bundles, now())
                .starts_with("Warning: [orangu].knowledge names ")
        );
    }

    #[test]
    fn the_reports_list_what_needs_a_look() {
        let (_dir, registry) = bundle(&[
            ("metrics/margin.md", MARGIN),
            ("metrics/legacy.md", LEGACY),
            (
                "old.md",
                "---\ntype: T\nstale_after: \"2026-01-01\"\nstatus: draft\n---\n",
            ),
        ]);
        assert_eq!(
            registry.report(Report::Stale, now()),
            "[retail] old: stale since 2026-01-01\n"
        );
        assert_eq!(
            registry.report(Report::Unverified, now()),
            "[retail] metrics/legacy (Metric, deprecated)\n[retail] old (T, draft)\n"
        );
        assert!(
            registry
                .report(Report::Bundles, now())
                .ends_with("  3 concepts: 1 stale, 1 deprecated, 2 unverified, 1 draft\n")
        );
        assert_eq!(Report::parse("stale"), Some(Report::Stale));
        assert_eq!(Report::parse("fresh"), None);
    }
}
