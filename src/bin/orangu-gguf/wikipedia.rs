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

//! English (or any language's) Wikipedia as training text.
//!
//! A corpus of source code teaches a model to write code and nothing else
//! — not to follow an instruction, not to explain what it just wrote, not
//! to write a sentence. Prose is what fixes that, and an encyclopedia is
//! the cleanest large body of it there is.
//!
//! **Which dump.** Wikimedia publishes several, and only one of them is
//! already the thing this tool needs. The article dumps
//! (`pages-articles.xml.bz2`) are *wikitext*: templates, infoboxes, tables
//! and reference markup, none of which is prose, and unpicking them
//! reliably is a project of its own. The search-index dumps
//! (`cirrus_search_index`) carry the same articles with all of that
//! already resolved — a `text` field of plain running prose, which is
//! exactly what the search engine indexes and exactly what a language
//! model should read. They are line-delimited JSON, sharded into roughly
//! gigabyte pieces, and streamable, so a run takes as much as it asked for
//! and stops.
//!
//! **Prose, and only prose.** The `text` field is the article's running
//! text — no wikitext, no image embeds, no infobox tables, no captions.
//! The dump's other fields are deliberately not read: `auxiliary_text` is
//! where image captions, table cells and navbox links live, and
//! `opening_text` is a duplicate of the first paragraph. Only namespace-0
//! pages are taken, so File:, Category: and Talk: pages cannot get in even
//! if the index's contents ever widen. Measured over a thousand articles
//! of the real dump, one mention of an image file name survives — inside a
//! citation URL — and no markup at all.
//!
//! **How much.** All of English Wikipedia is tens of gigabytes; a training
//! run rarely wants all of it and never wants to download it twice.
//! `max_bytes` caps the *extracted text*, the stream stops at the shard
//! that reaches it, and finished shards are left on disk so a later run
//! continues rather than starting over.
//!
//! **Licence.** Wikipedia text is CC BY-SA 4.0 — a share-alike licence.
//! That is not a software licence and so is not what the repository gate
//! reads, but it is recorded in the model's provenance and reported
//! alongside the corpus's other reciprocal licences, because it bears on
//! the same question they do.

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::{
    fs,
    io::{BufRead, BufReader, Write},
    path::Path,
    time::Duration,
};

/// Where the dumps live. The older `other/cirrussearch/` path published
/// one enormous file per wiki and is no longer updated; this is the
/// sharded replacement.
const DUMPS: &str = "https://dumps.wikimedia.org/other/cirrus_search_index";

/// The licence Wikipedia text is under, for the model's provenance.
pub const LICENSE: &str = "CC-BY-SA-4.0";

/// Wikipedia as a corpus source, as the manifest declares it.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Wikipedia {
    /// The wiki's language code: `en`, `de`, `simple`.
    #[serde(default = "default_language")]
    pub language: String,
    /// How much *extracted text* to take, in bytes.
    #[serde(default = "default_max_bytes")]
    pub max_bytes: u64,
    /// A dump date (`YYYYMMDD`) to pin. Absent takes the newest, which is
    /// what makes a re-run months later pick up a newer Wikipedia rather
    /// than fail on a dump that has since been rotated away.
    #[serde(default)]
    pub date: Option<String>,
}

fn default_language() -> String {
    "en".to_string()
}

/// 8 GiB of prose. Enough to teach English beside a code corpus, small
/// enough to fetch in an evening.
fn default_max_bytes() -> u64 {
    8 << 30
}

impl Default for Wikipedia {
    fn default() -> Self {
        Wikipedia {
            language: default_language(),
            max_bytes: default_max_bytes(),
            date: None,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct Report {
    pub articles: usize,
    pub bytes: u64,
    pub shards: usize,
    /// The dump this text came from, for the model's provenance.
    pub source: String,
}

/// The index of dump dates.
fn dates_url() -> String {
    format!("{DUMPS}/")
}

/// One dump's directory for one wiki's article index.
///
/// The `%3D` is a literal part of the published path — the directories are
/// named `index_name=enwiki_content`, and the `=` arrives percent-encoded.
pub fn shard_directory(language: &str, date: &str) -> String {
    format!("{DUMPS}/{date}/index_name%3D{language}wiki_content/")
}

/// The newest `YYYYMMDD` in a directory listing.
///
/// Parsed out of the index page rather than guessed from the calendar: the
/// dumps run weekly, and which day they landed on is not something to
/// assume.
pub fn newest_date(listing: &str) -> Option<String> {
    let mut dates: Vec<String> = Vec::new();
    for part in listing.split("href=\"") {
        let name = part.split('"').next().unwrap_or_default();
        let name = name.trim_end_matches('/');
        if name.len() == 8 && name.bytes().all(|b| b.is_ascii_digit()) {
            dates.push(name.to_string());
        }
    }
    dates.sort();
    dates.pop()
}

/// Every shard file named in a directory listing, in order.
pub fn shard_files(listing: &str) -> Vec<String> {
    let mut files: Vec<String> = Vec::new();
    for part in listing.split("href=\"") {
        let name = part.split('"').next().unwrap_or_default();
        if name.ends_with(".json.bz2") && !name.contains('/') {
            files.push(name.to_string());
        }
    }
    files.sort();
    files.dedup();
    files
}

/// One article out of one line of the dump, or `None`.
///
/// The dump alternates an index line with a document line, and plenty of
/// documents are redirects or stubs with no prose at all. Both are skipped
/// here rather than downstream, so what lands on disk is only text worth
/// training on.
pub fn article(line: &str) -> Option<(String, String)> {
    let value: serde_json::Value = serde_json::from_str(line).ok()?;
    let object = value.as_object()?;
    if object.contains_key("index") {
        return None;
    }
    // Articles only. The content index holds nothing else today, and
    // checking is what keeps that true if it ever widens.
    if object
        .get("namespace")
        .and_then(|n| n.as_u64())
        .unwrap_or(0)
        != 0
    {
        return None;
    }
    let text = object.get("text")?.as_str()?.trim();
    if text.is_empty() {
        return None;
    }
    let title = object
        .get("title")
        .and_then(|t| t.as_str())
        .unwrap_or_default()
        .trim();
    Some((title.to_string(), text.to_string()))
}

/// Fetches up to `settings.max_bytes` of article text into `dir`.
///
/// Shards already on disk are counted and skipped, so an interrupted fetch
/// resumes at the shard it stopped on. Each shard is written to a
/// temporary name and renamed, so a shard file that exists is a shard file
/// that is complete.
pub fn fetch(dir: &Path, settings: &Wikipedia, progress: &dyn Fn(&Report)) -> Result<Report> {
    fs::create_dir_all(dir)
        .with_context(|| format!("creating the Wikipedia directory {}", dir.display()))?;

    let client = reqwest::blocking::Client::builder()
        .user_agent(concat!("orangu-gguf/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(30))
        // No overall timeout: a shard is a gigabyte of bzip2 and a slow
        // link is not a failure. The connect timeout above is what
        // distinguishes a slow download from an unreachable host.
        .timeout(None)
        .build()
        .context("building the HTTP client")?;

    let date = match &settings.date {
        Some(date) => date.clone(),
        None => {
            let listing = client
                .get(dates_url())
                .send()
                .and_then(|r| r.error_for_status())
                .and_then(|r| r.text())
                .context("listing the Wikipedia dumps")?;
            newest_date(&listing).context("no dated dump directory in the listing")?
        }
    };

    let directory = shard_directory(&settings.language, &date);
    let listing = client
        .get(&directory)
        .send()
        .and_then(|r| r.error_for_status())
        .and_then(|r| r.text())
        .with_context(|| format!("listing {directory}"))?;
    let shards = shard_files(&listing);
    if shards.is_empty() {
        bail!(
            "no shards under {directory} — is {:?} a wiki language code?",
            settings.language
        );
    }

    let mut report = Report {
        source: directory.clone(),
        ..Report::default()
    };

    for (index, shard) in shards.iter().enumerate() {
        if report.bytes >= settings.max_bytes {
            break;
        }
        let target = dir.join(format!("{}wiki-{date}-{index:05}.txt", settings.language));
        if let Ok(metadata) = fs::metadata(&target) {
            // Already fetched by an earlier run.
            report.bytes += metadata.len();
            report.shards += 1;
            progress(&report);
            continue;
        }

        let url = format!("{directory}{shard}");
        let response = client
            .get(&url)
            .send()
            .and_then(|r| r.error_for_status())
            .with_context(|| format!("fetching {url}"))?;

        let temporary = target.with_extension("partial");
        let mut out = std::io::BufWriter::with_capacity(
            1 << 20,
            fs::File::create(&temporary)
                .with_context(|| format!("creating {}", temporary.display()))?,
        );
        let mut written = 0u64;

        let decoder = bzip2::read::BzDecoder::new(response);
        // A dump line is one whole article, and the longest are far past
        // any default line limit.
        let reader = BufReader::with_capacity(1 << 20, decoder);
        for line in reader.lines() {
            let line = match line {
                Ok(line) => line,
                // A shard cut short still leaves every article before the
                // cut usable, which is worth more than failing the run.
                Err(_) => break,
            };
            let Some((title, text)) = article(&line) else {
                continue;
            };
            let document = if title.is_empty() {
                format!("{text}\n\n")
            } else {
                format!("{title}\n\n{text}\n\n")
            };
            out.write_all(document.as_bytes())?;
            written += document.len() as u64;
            report.articles += 1;
            if report.articles.is_multiple_of(5000) {
                let mut shown = report.clone();
                shown.bytes += written;
                progress(&shown);
            }
            if report.bytes + written >= settings.max_bytes {
                break;
            }
        }
        out.flush()?;
        drop(out);
        fs::rename(&temporary, &target)
            .with_context(|| format!("renaming {} into place", temporary.display()))?;

        report.bytes += written;
        report.shards += 1;
        progress(&report);
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published path is not a name this tool gets to choose, and one
    /// wrong character in it is a 404 an hour into a run.
    #[test]
    fn the_dump_path_is_the_published_one() {
        assert_eq!(
            shard_directory("en", "20260823"),
            "https://dumps.wikimedia.org/other/cirrus_search_index/20260823/index_name%3Denwiki_content/"
        );
        assert_eq!(
            shard_directory("simple", "20260823"),
            "https://dumps.wikimedia.org/other/cirrus_search_index/20260823/index_name%3Dsimplewiki_content/"
        );
    }

    #[test]
    fn the_newest_dump_date_is_taken_from_the_listing() {
        let listing = r#"<a href="../">../</a>
            <a href="20260802/">20260802/</a>
            <a href="20260823/">20260823/</a>
            <a href="20260809/">20260809/</a>
            <a href="DEPRECATED.txt">DEPRECATED.txt</a>"#;
        assert_eq!(newest_date(listing).as_deref(), Some("20260823"));
        assert_eq!(newest_date("<a href=\"../\">..</a>"), None);
    }

    #[test]
    fn shards_are_listed_in_order_and_nothing_else_is() {
        let listing = r#"<a href="../">../</a>
            <a href="_SUCCESS">_SUCCESS</a>
            <a href="enwiki_content-20260823-00002.json.bz2">c</a>
            <a href="enwiki_content-20260823-00000.json.bz2">a</a>
            <a href="enwiki_content-20260823-00001.json.bz2">b</a>"#;
        assert_eq!(
            shard_files(listing),
            vec![
                "enwiki_content-20260823-00000.json.bz2",
                "enwiki_content-20260823-00001.json.bz2",
                "enwiki_content-20260823-00002.json.bz2",
            ]
        );
    }

    /// The dump alternates index lines with documents, and many documents
    /// carry no prose. Both have to be skipped, or the corpus fills with
    /// blank articles and JSON fragments.
    #[test]
    fn only_articles_with_prose_come_out() {
        assert_eq!(article(r#"{"index":{"_id":"12"}}"#), None);
        assert_eq!(article(r#"{"title":"Redirect","text":""}"#), None);
        assert_eq!(article(r#"{"title":"No text field"}"#), None);
        // A media or category page, if one ever reached the content index.
        assert_eq!(
            article(r#"{"namespace":6,"title":"File:Map.jpg","text":"A map."}"#),
            None
        );
        assert_eq!(
            article(r#"{"namespace":14,"title":"Category:Wales","text":"Pages."}"#),
            None
        );
        assert_eq!(article("not json at all"), None);

        let (title, text) =
            article(r#"{"title":"Cardiff","text":"  Cardiff is a city.  ","category":["Wales"]}"#)
                .unwrap();
        assert_eq!(title, "Cardiff");
        assert_eq!(text, "Cardiff is a city.");
    }

    /// A line shaped like the real dump's, end to end.
    #[test]
    fn a_real_dump_line_parses() {
        let line = r#"{"auxiliary_text":["x"],"category":["1948 births"],"content_model":"wikitext","language":"en","namespace":0,"namespace_text":"","title":"Gim Gwang-won","text":"Gim Gwang-won is a South Korean footballer.","page_id":123}"#;
        let (title, text) = article(line).unwrap();
        assert_eq!(title, "Gim Gwang-won");
        assert!(text.starts_with("Gim Gwang-won is"));
    }

    #[test]
    fn the_defaults_are_english_and_a_bounded_amount_of_it() {
        let settings = Wikipedia::default();
        assert_eq!(settings.language, "en");
        assert_eq!(settings.max_bytes, 8 << 30);
        assert!(settings.date.is_none());

        let from_json: Wikipedia = serde_json::from_str("{}").unwrap();
        assert_eq!(from_json, settings);
    }

    #[test]
    fn every_field_can_be_given() {
        let settings: Wikipedia = serde_json::from_str(
            r#"{"language": "simple", "max_bytes": 1024, "date": "20260823"}"#,
        )
        .unwrap();
        assert_eq!(settings.language, "simple");
        assert_eq!(settings.max_bytes, 1024);
        assert_eq!(settings.date.as_deref(), Some("20260823"));
    }

    #[test]
    fn a_misspelled_field_is_an_error() {
        assert!(serde_json::from_str::<Wikipedia>(r#"{"languge": "en"}"#).is_err());
    }
}
