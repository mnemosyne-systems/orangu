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

//! The `browse` tool: reading the web a page at a time, for a subagent that
//! has to move between pages quickly.
//!
//! A page is rendered to text once, split into parts of about
//! [`PART_CHARS`], and its links are numbered and made absolute, so the next
//! step is `link: 7` rather than a URL copied out by hand. Every page read
//! stays in the [`Browser`]'s cache for the rest of the run: going back to
//! one, or to another part of it, costs no request.

use anyhow::{Context, Result, anyhow};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

/// About how much text one part of a page holds.
pub const PART_CHARS: usize = 6_000;

/// The most a page may weigh.
const MAX_BYTES: usize = 5 * 1024 * 1024;

/// How many pages the cache keeps.
const MAX_CACHED: usize = 64;

/// The most links listed for one part.
const MAX_LISTED_LINKS: usize = 80;

/// One page, rendered.
#[derive(Debug, Clone, PartialEq)]
pub struct Page {
    /// Where the page ended up, after redirects.
    pub url: String,
    pub title: Option<String>,
    /// The text, in parts of about [`PART_CHARS`]. Links appear as
    /// `[text][n]`.
    pub parts: Vec<String>,
    /// Link `n` is `links[n - 1]`, absolute.
    pub links: Vec<String>,
}

/// `url` without its `#fragment`: the same page as far as fetching goes.
fn without_fragment(url: &str) -> &str {
    url.split_once('#').map_or(url, |(page, _)| page)
}

/// Splits `text` into parts of about `limit` characters on line boundaries.
fn split_parts(text: &str, limit: usize) -> Vec<String> {
    let mut parts = Vec::new();
    let mut part = String::new();
    for line in text.lines() {
        if !part.is_empty() && part.chars().count() + line.chars().count() > limit {
            parts.push(std::mem::take(&mut part));
        }
        part.push_str(line);
        part.push('\n');
    }
    if !part.trim().is_empty() || parts.is_empty() {
        parts.push(part);
    }
    parts
}

/// The `<title>` of `html`, if it has one.
fn title_of(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let start = lower.find("<title")?;
    let open = start + lower[start..].find('>')? + 1;
    let close = open + lower[open..].find("</title>")?;
    let title = crate::web_search::html_text(&html[open..close]);
    (!title.is_empty()).then_some(title)
}

/// Renders an HTML page found at `url`: its text with `[text][n]` link
/// references, and its links, numbered as the references number them and
/// resolved against `url`.
pub fn render_html(html: &str, url: &str) -> Result<Page> {
    let rendered = html2text::from_read(html.as_bytes(), 100)?;
    let base = reqwest::Url::parse(url).ok();

    // `html2text` lists the links as `[n]: target` lines at the end, from
    // `[1]: ` on, wrapping a long target onto the following lines.
    let lines: Vec<&str> = rendered.lines().collect();
    let footnote = |line: &str| -> Option<(usize, String)> {
        let (number, target) = line.strip_prefix('[')?.split_once("]: ")?;
        Some((number.parse().ok()?, target.trim().to_string()))
    };
    let body_end = lines
        .iter()
        .rposition(|line| line.starts_with("[1]: "))
        .unwrap_or(lines.len());
    let mut raw: Vec<(usize, String)> = Vec::new();
    for line in &lines[body_end..] {
        match footnote(line) {
            Some(link) => raw.push(link),
            None => {
                if let Some((_, target)) = raw.last_mut() {
                    target.push_str(line.trim());
                }
            }
        }
    }
    let links: Vec<(usize, String)> = raw
        .into_iter()
        .map(|(number, target)| {
            let absolute = base
                .as_ref()
                .and_then(|base| base.join(&target).ok())
                .map(|url| url.to_string())
                .unwrap_or(target);
            (number, absolute)
        })
        .collect();
    let mut numbered = vec![String::new(); links.iter().map(|(n, _)| *n).max().unwrap_or(0)];
    for (number, target) in links {
        if number > 0 {
            numbered[number - 1] = target;
        }
    }
    Ok(Page {
        url: url.to_string(),
        title: title_of(html),
        parts: split_parts(&lines[..body_end].join("\n"), PART_CHARS),
        links: numbered,
    })
}

/// A page that is not HTML: plain text as it is.
fn render_text(text: &str, url: &str) -> Page {
    Page {
        url: url.to_string(),
        title: None,
        parts: split_parts(text, PART_CHARS),
        links: Vec::new(),
    }
}

/// Fetches and renders `url`.
async fn fetch(http: &reqwest::Client, url: &str) -> Result<Page> {
    let parsed = reqwest::Url::parse(url).with_context(|| format!("'{url}' is not a URL"))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(anyhow!("only http and https pages can be browsed"));
    }
    let response = http
        .get(parsed)
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .with_context(|| format!("failed to open {url}"))?;
    let final_url = response.url().to_string();
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    if response
        .content_length()
        .is_some_and(|n| n as usize > MAX_BYTES)
    {
        return Err(anyhow!(
            "{url} is larger than {} MB",
            MAX_BYTES / 1024 / 1024
        ));
    }
    let bytes = response.bytes().await?;
    if bytes.len() > MAX_BYTES {
        return Err(anyhow!(
            "{url} is larger than {} MB",
            MAX_BYTES / 1024 / 1024
        ));
    }
    if content_type.contains("pdf") || bytes.starts_with(b"%PDF-") {
        let text = tokio::task::spawn_blocking(move || crate::pdf::extract_text(&bytes)).await??;
        return Ok(render_text(&text, &final_url));
    }
    let is_text = content_type.is_empty()
        || content_type.starts_with("text/")
        || content_type.contains("json")
        || content_type.contains("xml")
        || content_type.contains("javascript");
    if !is_text {
        return Err(anyhow!("{url} is not a text page ({content_type})"));
    }
    let body = String::from_utf8_lossy(&bytes);
    if content_type.contains("html") || body.trim_start().starts_with('<') {
        render_html(&body, &final_url)
    } else {
        Ok(render_text(&body, &final_url))
    }
}

/// What the subagent is looking at, and every page it has read.
#[derive(Debug, Default)]
pub struct Browser {
    cache: HashMap<String, Arc<Page>>,
    /// Cached URLs, oldest first, for eviction.
    order: VecDeque<String>,
    current: Option<Arc<Page>>,
}

impl Browser {
    /// Opens `url`, or link `link` of the current page, or else stays on
    /// the current page; then shows its part `part` (1 by default).
    pub async fn browse(
        &mut self,
        http: &reqwest::Client,
        url: Option<&str>,
        link: Option<usize>,
        part: Option<usize>,
    ) -> Result<String> {
        let target = match (url, link) {
            (Some(_), Some(_)) => return Err(anyhow!("give either 'url' or 'link', not both")),
            (Some(url), None) => Some(url.trim().to_string()),
            (None, Some(link)) => {
                let page = self
                    .current
                    .as_ref()
                    .ok_or_else(|| anyhow!("no page is open yet; give a 'url'"))?;
                let target = link
                    .checked_sub(1)
                    .and_then(|index| page.links.get(index))
                    .filter(|target| !target.is_empty())
                    .ok_or_else(|| {
                        anyhow!(
                            "{} has no link {link} (it has {})",
                            page.url,
                            page.links.len()
                        )
                    })?;
                Some(target.clone())
            }
            (None, None) => None,
        };
        let cached = match target {
            Some(target) => {
                let key = without_fragment(&target);
                let (page, cached) = match self.cache.get(key) {
                    Some(page) => (page.clone(), true),
                    None => {
                        let page = Arc::new(fetch(http, key).await?);
                        self.remember(key, &page);
                        (page, false)
                    }
                };
                self.current = Some(page);
                cached
            }
            None if part.is_some() => true,
            None => return Err(anyhow!("give a 'url', a 'link' or a 'part'")),
        };
        let page = self
            .current
            .as_ref()
            .ok_or_else(|| anyhow!("no page is open yet; give a 'url'"))?;
        show(page, part.unwrap_or(1), cached)
    }

    fn remember(&mut self, requested: &str, page: &Arc<Page>) {
        for key in [
            requested.to_string(),
            without_fragment(&page.url).to_string(),
        ] {
            if self.cache.insert(key.clone(), page.clone()).is_none() {
                self.order.push_back(key);
            }
        }
        while self.order.len() > MAX_CACHED {
            if let Some(oldest) = self.order.pop_front() {
                self.cache.remove(&oldest);
            }
        }
    }
}

/// Part `part` of `page` as the model reads it, with the links it refers to.
pub fn show(page: &Page, part: usize, cached: bool) -> Result<String> {
    let total = page.parts.len();
    if part == 0 || part > total {
        return Err(anyhow!("{} has parts 1 to {total}", page.url));
    }
    let text = &page.parts[part - 1];
    let mut out = format!(
        "URL: {}{}\n",
        page.url,
        if cached { " (cached)" } else { "" }
    );
    if let Some(title) = &page.title {
        out.push_str(&format!("Title: {title}\n"));
    }
    out.push_str(&format!("Part {part} of {total}"));
    if part < total {
        out.push_str(&format!(" (next: part={})", part + 1));
    }
    out.push_str("\n\n");
    out.push_str(text.trim_end());
    out.push('\n');

    // The links this part refers to, `[text][n]`, in order of appearance.
    let mut seen = Vec::new();
    let mut rest = text.as_str();
    while let Some(at) = rest.find("][") {
        rest = &rest[at + 2..];
        if let Some(end) = rest.find(']')
            && let Ok(number) = rest[..end].parse::<usize>()
            && !seen.contains(&number)
        {
            seen.push(number);
        }
    }
    // Anchors into this same page lead nowhere new.
    let here = without_fragment(&page.url);
    let listed: Vec<(usize, &String)> = seen
        .into_iter()
        .filter_map(|n| {
            page.links
                .get(n.wrapping_sub(1))
                .filter(|t| !t.is_empty() && without_fragment(t) != here)
                .map(|t| (n, t))
        })
        .take(MAX_LISTED_LINKS)
        .collect();
    if !listed.is_empty() {
        out.push_str("\nLinks in this part (open one with link=N):\n");
        for (number, target) in listed {
            out.push_str(&format!("[{number}] {target}\n"));
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HTML: &str = r#"<html><head><title>The &amp; Docs</title></head><body>
<h1>Guide</h1><p>Read <a href="/start">the start</a>, then <a href="https://other.example/x">X</a>.</p>
<ul><li><a href="next.html">Next</a></li></ul></body></html>"#;

    #[test]
    fn renders_text_title_and_absolute_numbered_links() {
        let page = render_html(HTML, "https://docs.example/guide/index.html").unwrap();
        assert_eq!(page.title.as_deref(), Some("The & Docs"));
        assert_eq!(page.parts.len(), 1);
        assert!(
            page.parts[0].contains("Read [the start][1], then [X][2]."),
            "{:?}",
            page.parts
        );
        assert!(!page.parts[0].contains("]: "), "{:?}", page.parts);
        assert_eq!(
            page.links,
            [
                "https://docs.example/start",
                "https://other.example/x",
                "https://docs.example/guide/next.html"
            ]
        );
    }

    /// Long targets wrap in `html2text`'s footnotes; they come back whole.
    #[test]
    fn wrapped_link_targets_are_joined() {
        let long = format!("https://docs.example/{}/page.html#section", "a".repeat(150));
        let html =
            format!(r#"<p>See <a href="{long}">the long one</a> and <a href="/b">B</a>.</p>"#);
        let page = render_html(&html, "https://docs.example/").unwrap();
        assert_eq!(page.links, [long.as_str(), "https://docs.example/b"]);
        assert!(!page.parts[0].contains("aaaa"), "{:?}", page.parts);
    }

    #[test]
    fn show_lists_the_links_of_the_part_and_points_to_the_next() {
        let page = Page {
            url: "https://a.example/".into(),
            title: None,
            parts: vec![
                "See [B][2] and [A][1] and [B][2], [top][3].\n".into(),
                "End.\n".into(),
            ],
            links: vec![
                "https://a.example/a".into(),
                "https://a.example/b".into(),
                "https://a.example/#top".into(),
            ],
        };
        // The anchor into the same page is not listed.
        let out = show(&page, 1, false).unwrap();
        assert_eq!(
            out,
            "URL: https://a.example/\nPart 1 of 2 (next: part=2)\n\nSee [B][2] and [A][1] and [B][2], [top][3].\n\n\
             Links in this part (open one with link=N):\n[2] https://a.example/b\n[1] https://a.example/a\n"
        );
        let out = show(&page, 2, true).unwrap();
        assert!(
            out.starts_with("URL: https://a.example/ (cached)\nPart 2 of 2\n"),
            "{out}"
        );
        assert!(show(&page, 3, false).is_err());
        assert!(show(&page, 0, false).is_err());
    }

    #[test]
    fn long_text_is_split_on_line_boundaries() {
        let text = "line of text\n".repeat(1_000);
        let parts = split_parts(&text, 1_000);
        assert!(parts.len() > 10);
        assert!(parts.iter().all(|p| p.chars().count() <= 1_000 + 13));
        assert_eq!(parts.concat(), text);
        assert_eq!(split_parts("", 10), [""]);
    }

    /// A tiny site: `/` links to `/b`; `/b` is plain text. Each path may be
    /// requested once, so a second visit has to come from the cache.
    async fn site() -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            let mut served = Vec::new();
            loop {
                let Ok((mut socket, _)) = listener.accept().await else {
                    break;
                };
                let mut request = vec![0u8; 4096];
                let n = socket.read(&mut request).await.unwrap_or(0);
                let request = String::from_utf8_lossy(&request[..n]).to_string();
                let path = request.split_whitespace().nth(1).unwrap_or("/").to_string();
                assert!(!served.contains(&path), "{path} fetched twice");
                served.push(path.clone());
                let (kind, body) = match path.as_str() {
                    "/" => (
                        "text/html",
                        r#"<title>Home</title><p>Go to <a href="/b">B</a>.</p>"#,
                    ),
                    "/b" => ("text/plain", "Plain page B."),
                    _ => ("image/png", "PNG"),
                };
                let _ = socket
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\ncontent-type: {kind}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .await;
            }
        });
        base
    }

    #[tokio::test]
    async fn browses_by_url_and_link_and_reuses_the_cache() {
        let base = site().await;
        let http = crate::web_search::client();
        let mut browser = Browser::default();

        let home = browser
            .browse(&http, Some(&format!("{base}/")), None, None)
            .await
            .unwrap();
        assert!(home.contains("Title: Home"), "{home}");
        assert!(home.contains(&format!("[1] {base}/b")), "{home}");

        let b = browser.browse(&http, None, Some(1), None).await.unwrap();
        assert!(b.starts_with(&format!("URL: {base}/b\n")), "{b}");
        assert!(b.contains("Plain page B."), "{b}");

        // An anchor on a page already read is the same page: cached.
        let anchored = browser
            .browse(&http, Some(&format!("{base}/b#section")), None, None)
            .await
            .unwrap();
        assert!(
            anchored.starts_with(&format!("URL: {base}/b (cached)")),
            "{anchored}"
        );

        // Back to the start: from the cache, so no second request.
        let again = browser
            .browse(&http, Some(&format!("{base}/")), None, None)
            .await
            .unwrap();
        assert!(
            again.starts_with(&format!("URL: {base}/ (cached)")),
            "{again}"
        );
        let part = browser.browse(&http, None, None, Some(1)).await.unwrap();
        assert!(part.contains("(cached)"), "{part}");

        assert!(browser.browse(&http, None, Some(9), None).await.is_err());
        assert!(browser.browse(&http, None, None, None).await.is_err());
        assert!(
            browser
                .browse(&http, Some("file:///etc/passwd"), None, None)
                .await
                .is_err()
        );
        let err = browser
            .browse(&http, Some(&format!("{base}/image")), None, None)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("not a text page"), "{err}");
        assert!(
            Browser::default()
                .browse(&http, None, Some(1), None)
                .await
                .is_err()
        );
    }
}
