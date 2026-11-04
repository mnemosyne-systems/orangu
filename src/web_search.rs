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

//! Web search, local first.
//!
//! A SearXNG instance you run yourself is searched through its JSON API (its
//! `json` output format has to be enabled in `settings.yml`,
//! `search.formats`): that is what `orangu-server`'s `web_search` server tool
//! searches, and what the terminal's `web_search` tool tries first. Only when
//! no local instance answers does the terminal fall back to public engines,
//! DuckDuckGo, Bing and Google, by reading their plain result pages.
//!
//! Every request says it comes from orangu (`User-Agent: orangu/<version>`).
//! An engine that answers with a challenge or a page that needs JavaScript
//! instead of results is treated as unavailable and the next one is tried:
//! nothing here tries to get past an engine's bot checks.

use anyhow::{Context, Result, anyhow};
use serde_json::Value;
use std::time::Duration;

/// How orangu identifies itself to search engines and the pages it reads.
pub const USER_AGENT: &str = concat!("orangu/", env!("CARGO_PKG_VERSION"));

/// How long one engine or page may take to answer.
const TIMEOUT: Duration = Duration::from_secs(20);

/// The client every search and page read goes through: identified as
/// orangu, with a timeout, following redirects.
pub fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .timeout(TIMEOUT)
        .build()
        .unwrap_or_default()
}

/// The longest query sent.
pub const MAX_QUERY: usize = 500;

/// One search result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchResult {
    pub url: String,
    pub title: String,
    /// The engine's excerpt of the page.
    pub snippet: String,
}

/// The URL that asks the instance at `base` for `query` as JSON.
pub fn search_url(base: &str, query: &str) -> Result<reqwest::Url> {
    reqwest::Url::parse_with_params(
        &format!("{}/search", base.trim_end_matches('/')),
        [("q", query), ("format", "json")],
    )
    .with_context(|| format!("invalid web search URL '{base}'"))
}

/// The results in a SearXNG JSON reply, in its order; entries without a URL
/// are dropped.
pub fn parse_results(body: &Value) -> Vec<SearchResult> {
    body.get("results")
        .and_then(Value::as_array)
        .map(|results| {
            results
                .iter()
                .filter_map(|result| {
                    let text = |key: &str| {
                        result
                            .get(key)
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .trim()
                            .to_string()
                    };
                    let url = result.get("url")?.as_str()?.trim().to_string();
                    (!url.is_empty()).then(|| SearchResult {
                        url,
                        title: text("title"),
                        snippet: text("content"),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Searches the SearXNG instance at `base` for `query`.
pub async fn searxng(http: &reqwest::Client, base: &str, query: &str) -> Result<Vec<SearchResult>> {
    let query = query.trim();
    if query.is_empty() {
        return Err(anyhow!("empty search query"));
    }
    if query.len() > MAX_QUERY {
        return Err(anyhow!("search query longer than {MAX_QUERY} bytes"));
    }
    let body: Value = http
        .get(search_url(base, query)?)
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|err| anyhow!("web search failed: {err}"))?
        .json()
        .await
        .map_err(|err| {
            anyhow!("web search returned no JSON (is `json` in SearXNG's search.formats?): {err}")
        })?;
    Ok(parse_results(&body))
}

/// `results` as the text a model reads: numbered title, URL and excerpt.
pub fn format_results(query: &str, results: &[SearchResult]) -> String {
    let mut text = format!("Search results for \"{query}\":\n");
    if results.is_empty() {
        text.push_str("\nNo results.\n");
    }
    for (n, result) in results.iter().enumerate() {
        text.push_str(&format!(
            "\n[{}] {}\n{}\n{}\n",
            n + 1,
            result.title,
            result.url,
            result.snippet
        ));
    }
    text
}

/// Where a search goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Engine {
    /// A SearXNG instance, by base URL: local, so always tried first.
    Searxng(String),
    DuckDuckGo,
    Bing,
    Google,
}

impl Engine {
    pub fn name(&self) -> &str {
        match self {
            Engine::Searxng(base) => base,
            Engine::DuckDuckGo => "duckduckgo",
            Engine::Bing => "bing",
            Engine::Google => "google",
        }
    }
}

/// Written as it is configured: the SearXNG URL or the engine's name.
impl serde::Serialize for Engine {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.name())
    }
}

/// The public engines searched when nothing else is configured, in order.
pub const DEFAULT_ENGINES: &[Engine] = &[Engine::DuckDuckGo, Engine::Bing, Engine::Google];

/// `[orangu].web_search`: a comma-separated list of SearXNG base URLs and
/// engine names (`duckduckgo`, `bing`, `google`), tried in order but with
/// every SearXNG instance first, since it is local. Unset or empty means the
/// public engines; `off` means no web search.
pub fn parse_engines(value: Option<&str>) -> Result<Vec<Engine>> {
    let value = value.map(str::trim).unwrap_or_default();
    if value.is_empty() {
        return Ok(DEFAULT_ENGINES.to_vec());
    }
    if value.eq_ignore_ascii_case("off") {
        return Ok(Vec::new());
    }
    let mut engines = Vec::new();
    for entry in value.split(',').map(str::trim).filter(|e| !e.is_empty()) {
        let engine = match entry.to_ascii_lowercase().as_str() {
            "duckduckgo" | "ddg" => Engine::DuckDuckGo,
            "bing" => Engine::Bing,
            "google" => Engine::Google,
            _ if entry.starts_with("http://") || entry.starts_with("https://") => {
                reqwest::Url::parse(entry)
                    .map_err(|err| anyhow!("'{entry}' is not a URL: {err}"))?;
                Engine::Searxng(entry.trim_end_matches('/').to_string())
            }
            _ => {
                return Err(anyhow!(
                    "'{entry}' is neither a SearXNG URL nor duckduckgo, bing or google"
                ));
            }
        };
        if !engines.contains(&engine) {
            engines.push(engine);
        }
    }
    // Local first, whatever order the list was written in.
    engines.sort_by_key(|engine| !matches!(engine, Engine::Searxng(_)));
    Ok(engines)
}

/// Searches `engines` in order until one returns results: the engine that
/// answered and its results. Results that are empty everywhere come back as
/// no results; only when every engine failed is it an error, naming why
/// each did.
pub async fn search(
    http: &reqwest::Client,
    engines: &[Engine],
    query: &str,
) -> Result<(String, Vec<SearchResult>)> {
    let query = query.trim();
    if query.is_empty() {
        return Err(anyhow!("empty search query"));
    }
    if query.len() > MAX_QUERY {
        return Err(anyhow!("search query longer than {MAX_QUERY} bytes"));
    }
    if engines.is_empty() {
        return Err(anyhow!("web search is off ([orangu].web_search)"));
    }
    let mut failures = Vec::new();
    let mut answered_empty = None;
    for engine in engines {
        match search_engine(http, engine, query).await {
            Ok(results) if !results.is_empty() => return Ok((engine.name().to_string(), results)),
            Ok(_) => {
                answered_empty.get_or_insert_with(|| engine.name().to_string());
            }
            Err(err) => failures.push(format!("{}: {err:#}", engine.name())),
        }
    }
    match answered_empty {
        Some(engine) => Ok((engine, Vec::new())),
        None => Err(anyhow!(
            "every search engine failed: {}",
            failures.join("; ")
        )),
    }
}

async fn search_engine(
    http: &reqwest::Client,
    engine: &Engine,
    query: &str,
) -> Result<Vec<SearchResult>> {
    let page = |url: reqwest::Url| async move {
        let response = http
            .get(url)
            .header(reqwest::header::ACCEPT_LANGUAGE, "en")
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)?;
        // DuckDuckGo answers a challenge with 202 rather than an error.
        if response.status() != reqwest::StatusCode::OK {
            return Err(anyhow!("answered {} instead of results", response.status()));
        }
        Ok::<String, anyhow::Error>(response.text().await?)
    };
    match engine {
        Engine::Searxng(base) => searxng(http, base, query).await,
        Engine::DuckDuckGo => {
            let url = reqwest::Url::parse_with_params(
                "https://lite.duckduckgo.com/lite/",
                [("q", query)],
            )?;
            parse_duckduckgo(&page(url).await?)
        }
        Engine::Bing => {
            let url = reqwest::Url::parse_with_params(
                "https://www.bing.com/search",
                [("q", query), ("setlang", "en")],
            )?;
            parse_bing(&page(url).await?)
        }
        Engine::Google => {
            let url = reqwest::Url::parse_with_params(
                "https://www.google.com/search",
                [("q", query), ("hl", "en")],
            )?;
            parse_google(&page(url).await?)
        }
    }
}

/// `html` with its tags removed, entities decoded and whitespace collapsed.
pub fn html_text(html: &str) -> String {
    let mut text = String::with_capacity(html.len());
    let mut in_tag = false;
    for c in html.chars() {
        match c {
            '<' => in_tag = true,
            '>' if in_tag => in_tag = false,
            _ if !in_tag => text.push(c),
            _ => {}
        }
    }
    decode_entities(&text)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Decodes the named entities result pages use, and numeric ones.
fn decode_entities(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find('&') {
        out.push_str(&rest[..start]);
        rest = &rest[start..];
        let Some(end) = rest[..rest.len().min(12)].find(';') else {
            out.push('&');
            rest = &rest[1..];
            continue;
        };
        let entity = &rest[1..end];
        let decoded = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            "nbsp" => Some(' '),
            _ => entity
                .strip_prefix("#x")
                .or_else(|| entity.strip_prefix("#X"))
                .and_then(|hex| u32::from_str_radix(hex, 16).ok())
                .or_else(|| entity.strip_prefix('#').and_then(|dec| dec.parse().ok()))
                .and_then(char::from_u32),
        };
        match decoded {
            Some(c) => {
                out.push(c);
                rest = &rest[end + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// The value of attribute `name` in the start tag `tag`, entities decoded.
fn attribute(tag: &str, name: &str) -> Option<String> {
    let at = tag.find(&format!(" {name}="))? + name.len() + 2;
    let rest = &tag[at..];
    let quote = rest.chars().next().filter(|q| *q == '"' || *q == '\'')?;
    let end = rest[1..].find(quote)?;
    Some(decode_entities(&rest[1..1 + end]))
}

/// The value of query parameter `key` in `url`.
fn query_param(url: &str, key: &str) -> Option<String> {
    let url = if url.starts_with("//") {
        format!("https:{url}")
    } else {
        url.to_string()
    };
    let parsed = reqwest::Url::parse(&url)
        .or_else(|_| reqwest::Url::parse("https://example.invalid/")?.join(&url))
        .ok()?;
    parsed
        .query_pairs()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.into_owned())
}

/// The anchor tags in `html` whose start tag contains `marker`: each start
/// tag and the anchor's inner HTML.
fn anchors<'a>(html: &'a str, marker: &str) -> Vec<(&'a str, &'a str)> {
    let mut found = Vec::new();
    let mut rest = html;
    while let Some(start) = rest.find("<a ") {
        rest = &rest[start..];
        let Some(tag_end) = rest.find('>') else { break };
        let tag = &rest[..tag_end];
        let Some(close) = rest[tag_end..].find("</a>") else {
            break;
        };
        let inner = &rest[tag_end + 1..tag_end + close];
        if tag.contains(marker) {
            found.push((tag, inner));
        }
        rest = &rest[tag_end + close..];
    }
    found
}

fn is_web_url(url: &str) -> bool {
    reqwest::Url::parse(url).is_ok_and(|u| matches!(u.scheme(), "http" | "https"))
}

/// DuckDuckGo's lite result page. Its links go through
/// `duckduckgo.com/l/?uddg=<url>`; sponsored ones (`y.js`) are dropped.
pub fn parse_duckduckgo(html: &str) -> Result<Vec<SearchResult>> {
    if html.contains("anomaly") && !html.contains("result-link") {
        return Err(anyhow!("answered with a challenge instead of results"));
    }
    let mut results = Vec::new();
    let links = anchors(html, "result-link");
    for (index, (tag, inner)) in links.iter().enumerate() {
        let Some(href) = attribute(tag, "href") else {
            continue;
        };
        let url = query_param(&href, "uddg").unwrap_or(href);
        if !is_web_url(&url) || url.contains("duckduckgo.com/y.js") {
            continue;
        }
        // The snippet sits between this link and the next.
        let start = html.find(tag).unwrap_or(0);
        let end = links
            .get(index + 1)
            .and_then(|(next, _)| html[start + 1..].find(next).map(|at| start + 1 + at))
            .unwrap_or(html.len());
        let section = &html[start..end];
        let snippet = section
            .find("result-snippet")
            .and_then(|at| {
                let after = &section[at..];
                let open = after.find('>')? + 1;
                let close = after[open..].find("</td>")?;
                Some(html_text(&after[open..open + close]))
            })
            .unwrap_or_default();
        results.push(SearchResult {
            url,
            title: html_text(inner),
            snippet,
        });
    }
    Ok(results)
}

/// The target of one of Bing's `bing.com/ck/a?…&u=a1<base64url>` links.
fn bing_target(href: &str) -> Option<String> {
    use base64::Engine as _;
    if !href.contains("bing.com/ck/a") {
        return Some(href.to_string());
    }
    let encoded = query_param(href, "u")?;
    let encoded = encoded.strip_prefix("a1")?;
    let bytes = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(encoded.trim_end_matches('='))
        .ok()?;
    String::from_utf8(bytes).ok()
}

/// Bing's result page: one `<li class="b_algo">` per result, its title link
/// in the `<h2>` and its excerpt in the first paragraph.
pub fn parse_bing(html: &str) -> Result<Vec<SearchResult>> {
    let mut results = Vec::new();
    for block in html.split("<li class=\"b_algo\"").skip(1) {
        let block = block.split("</li>").next().unwrap_or(block);
        let Some(h2) = block.find("<h2") else {
            continue;
        };
        let Some((tag, inner)) = anchors(&block[h2..], "href=").into_iter().next() else {
            continue;
        };
        let Some(url) = attribute(tag, "href").and_then(|href| bing_target(&href)) else {
            continue;
        };
        if !is_web_url(&url) {
            continue;
        }
        let snippet = block[h2..]
            .find("<p")
            .and_then(|at| {
                let after = &block[h2 + at..];
                let open = after.find('>')? + 1;
                let close = after[open..].find("</p>")?;
                Some(html_text(&after[open..open + close]))
            })
            .unwrap_or_default();
        results.push(SearchResult {
            url,
            title: html_text(inner),
            snippet,
        });
    }
    Ok(results)
}

/// Google's result page, when it serves one without JavaScript: result
/// links go through `/url?q=<url>` and carry an `<h3>` title. A page that
/// needs JavaScript is an error, so the search moves on.
pub fn parse_google(html: &str) -> Result<Vec<SearchResult>> {
    if html.contains("enablejs") || html.contains("/httpservice/retry") {
        return Err(anyhow!("answered with a page that needs JavaScript"));
    }
    let mut results = Vec::new();
    for (tag, inner) in anchors(html, "href=\"/url?") {
        if !inner.contains("<h3") {
            continue;
        }
        let Some(url) = attribute(tag, "href").and_then(|href| query_param(&href, "q")) else {
            continue;
        };
        if is_web_url(&url) && !results.iter().any(|r: &SearchResult| r.url == url) {
            results.push(SearchResult {
                url,
                title: html_text(inner),
                snippet: String::new(),
            });
        }
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn search_url_asks_for_json() {
        let url = search_url("http://localhost:8888/", "rust async & tokio").unwrap();
        assert_eq!(
            url.as_str(),
            "http://localhost:8888/search?q=rust+async+%26+tokio&format=json"
        );
        assert!(search_url("not a url", "x").is_err());
    }

    #[test]
    fn parse_results_keeps_entries_with_urls_in_order() {
        let body = json!({"results": [
            {"url": "https://a.example/", "title": " A ", "content": "first"},
            {"title": "no url"},
            {"url": "https://b.example/"}
        ]});
        assert_eq!(
            parse_results(&body),
            [
                SearchResult {
                    url: "https://a.example/".into(),
                    title: "A".into(),
                    snippet: "first".into()
                },
                SearchResult {
                    url: "https://b.example/".into(),
                    title: String::new(),
                    snippet: String::new()
                },
            ]
        );
        assert!(parse_results(&json!({"error": "x"})).is_empty());
    }

    #[test]
    fn format_results_numbers_each_result() {
        let results = [SearchResult {
            url: "https://a.example/".into(),
            title: "A".into(),
            snippet: "first".into(),
        }];
        assert_eq!(
            format_results("q", &results),
            "Search results for \"q\":\n\n[1] A\nhttps://a.example/\nfirst\n"
        );
        assert!(format_results("q", &[]).contains("No results."));
    }

    #[test]
    fn parse_engines_defaults_to_public_engines_and_puts_local_first() {
        assert_eq!(parse_engines(None).unwrap(), DEFAULT_ENGINES);
        assert_eq!(parse_engines(Some("  ")).unwrap(), DEFAULT_ENGINES);
        assert!(parse_engines(Some("OFF")).unwrap().is_empty());
        assert_eq!(
            parse_engines(Some("bing, http://localhost:8888/, ddg, bing")).unwrap(),
            [
                Engine::Searxng("http://localhost:8888".into()),
                Engine::Bing,
                Engine::DuckDuckGo
            ]
        );
        assert_eq!(
            parse_engines(Some("http://localhost:8888")).unwrap(),
            [Engine::Searxng("http://localhost:8888".into())]
        );
        assert!(parse_engines(Some("yahoo")).is_err());
        assert!(parse_engines(Some("http://")).is_err());
    }

    #[test]
    fn html_text_strips_tags_and_decodes_entities() {
        assert_eq!(
            html_text("A <b>bold</b>\n  &amp; &lt;tag&gt; don&#x27;t&#39;s&nbsp;x &#169; &bogus;"),
            "A bold & <tag> don't's x © &bogus;"
        );
    }

    #[test]
    fn parses_duckduckgo_lite_results_and_skips_ads() {
        let html = r#"<table>
<tr><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.rs%2Fa&amp;rut=1" class='result-link'>A &amp; <b>B</b></a></td></tr>
<tr><td class='result-snippet'>
  The <b>first</b> snippet.
</td></tr>
<tr><td><a rel="nofollow" href="https://duckduckgo.com/y.js?ad=1" class='result-link'>Ad</a></td></tr>
<tr><td class='result-snippet'>Buy now</td></tr>
<tr><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fb&amp;rut=2" class='result-link'>Second</a></td></tr>
</table>"#;
        let results = parse_duckduckgo(html).unwrap();
        assert_eq!(
            results,
            [
                SearchResult {
                    url: "https://docs.rs/a".into(),
                    title: "A & B".into(),
                    snippet: "The first snippet.".into()
                },
                SearchResult {
                    url: "https://example.org/b".into(),
                    title: "Second".into(),
                    snippet: String::new()
                },
            ]
        );
        assert!(parse_duckduckgo("<form id='challenge'>anomaly</form>").is_err());
    }

    #[test]
    fn parses_bing_results_through_its_redirect_links() {
        // `a1` + base64url("https://docs.rs/tokio/").
        let html = r#"<ol><li class="b_algo" data-id><div class="b_tpcn"><a class="tilk" href="https://www.bing.com/ck/a?!&amp;&amp;p=x&amp;u=a1aHR0cHM6Ly9kb2NzLnJzL3Rva2lvLw&amp;ntb=1">icon</a></div><h2><a target="_blank" href="https://www.bing.com/ck/a?!&amp;&amp;p=x&amp;u=a1aHR0cHM6Ly9kb2NzLnJzL3Rva2lvLw&amp;ntb=1"><strong>Tokio</strong> docs</a></h2><div class="b_caption"><p class="b_lineclamp2">An async runtime &#8230;</p></div></li><li class="b_algo"><h2><a href="https://plain.example/">Plain</a></h2></li></ol>"#;
        assert_eq!(
            parse_bing(html).unwrap(),
            [
                SearchResult {
                    url: "https://docs.rs/tokio/".into(),
                    title: "Tokio docs".into(),
                    snippet: "An async runtime …".into()
                },
                SearchResult {
                    url: "https://plain.example/".into(),
                    title: "Plain".into(),
                    snippet: String::new()
                },
            ]
        );
    }

    #[test]
    fn parses_google_basic_results_and_refuses_its_javascript_page() {
        let html = r#"<a href="/url?q=https://g.example/a&amp;sa=U"><h3>Result A</h3></a><a href="/url?q=https://g.example/nav&amp;sa=U">Nav</a>"#;
        assert_eq!(
            parse_google(html).unwrap(),
            [SearchResult {
                url: "https://g.example/a".into(),
                title: "Result A".into(),
                snippet: String::new()
            }]
        );
        let err = parse_google(
            r#"<noscript><meta content="0;url=/httpservice/retry/enablejs"></noscript>"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("JavaScript"));
    }

    /// One-request SearXNG stand-in answering `body`; its base URL.
    async fn one_shot_searxng(body: &'static str) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = vec![0u8; 4096];
            let _ = socket.read(&mut request).await;
            let _ = socket
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await;
        });
        base
    }

    /// A base URL nothing listens on.
    async fn dead_base() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        base
    }

    #[tokio::test]
    async fn search_falls_back_until_an_engine_answers() {
        let http = client();
        let dead = Engine::Searxng(dead_base().await);
        let empty = Engine::Searxng(one_shot_searxng(r#"{"results":[]}"#).await);
        let full = Engine::Searxng(
            one_shot_searxng(r#"{"results":[{"url":"https://r.example/","title":"R"}]}"#).await,
        );
        let (engine, results) = search(&http, &[dead.clone(), empty, full.clone()], "q")
            .await
            .unwrap();
        assert_eq!(engine, full.name());
        assert_eq!(results.len(), 1);

        // Empty answers are no results; only all-failed is an error.
        let empty = Engine::Searxng(one_shot_searxng(r#"{"results":[]}"#).await);
        let (_, results) = search(&http, &[dead.clone(), empty], "q").await.unwrap();
        assert!(results.is_empty());
        let err = search(&http, std::slice::from_ref(&dead), "q")
            .await
            .unwrap_err();
        assert!(
            err.to_string().starts_with("every search engine failed:"),
            "{err}"
        );
        assert!(err.to_string().contains(dead.name()), "{err}");

        assert!(search(&http, &[], "q").await.is_err());
        assert!(search(&http, &[full], " ").await.is_err());
    }

    #[tokio::test]
    async fn searxng_queries_the_instance_and_reports_failures() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = vec![0u8; 4096];
            let n = socket.read(&mut request).await.unwrap();
            let request = String::from_utf8_lossy(&request[..n]).to_string();
            let body = r#"{"results":[{"url":"https://r.example/","title":"R","content":"c"}]}"#;
            socket
                .write_all(
                    format!(
                        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len()
                    )
                    .as_bytes(),
                )
                .await
                .unwrap();
            request
        });
        let http = reqwest::Client::new();
        let results = searxng(&http, &base, "orangu").await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].url, "https://r.example/");
        let request = server.await.unwrap();
        assert!(
            request.starts_with("GET /search?q=orangu&format=json "),
            "{request}"
        );

        assert!(searxng(&http, &base, "  ").await.is_err());
        assert!(
            searxng(&http, &base, &"x".repeat(MAX_QUERY + 1))
                .await
                .is_err()
        );
        // Nothing listens there any more.
        assert!(searxng(&http, &base, "again").await.is_err());
    }
}
