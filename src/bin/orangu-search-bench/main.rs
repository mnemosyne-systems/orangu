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

//! A small, committed retrieval benchmark for the hybrid-search PoC.
//!
//! Runs the same query fixture against lexical-only, semantic-only, and hybrid
//! ranking, reporting MRR, Recall@5, per-query latency, indexing time, and the
//! lexical sidecar's on-disk size. The fixture targets symbols that the current
//! Rust extractor actually emits; standalone constants that are not extracted
//! are marked `coverage_gap` and reported separately.
//!
//! ```text
//! cargo run --release --bin orangu-search-bench -- --workspace .
//! cargo run --release --bin orangu-search-bench -- --workspace . \
//!     --endpoint http://127.0.0.1:8080 --model embeddings
//! ```

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::time::{Duration, Instant};

use orangu::config::LlmConfiguration;
use orangu::embeddings::{EmbeddingClient, EmbeddingIndex, SearchHit, lexical::LexicalIndex};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    queries: Vec<QueryCase>,
}

#[derive(Deserialize)]
struct QueryCase {
    query: String,
    #[serde(default)]
    kind: String,
    expected: Vec<String>,
    #[serde(default)]
    coverage_gap: bool,
}

#[derive(Default)]
struct Scores {
    mrr: f64,
    recall: f64,
    latency: Duration,
    count: usize,
}

impl Scores {
    fn add(&mut self, mrr: f64, recall: f64, latency: Duration) {
        self.mrr += mrr;
        self.recall += recall;
        self.latency += latency;
        self.count += 1;
    }

    fn summary(&self) -> String {
        if self.count == 0 {
            return "-".to_string();
        }
        format!(
            "{:.3}/{:.3} {}ms",
            self.mrr / self.count as f64,
            self.recall / self.count as f64,
            self.latency.as_millis() / self.count as u128
        )
    }
}

fn main() -> anyhow::Result<()> {
    let options = Options::parse()?;
    let fixture: Fixture = serde_json::from_str(include_str!("queries.json"))?;
    let workspace = options.workspace.clone();
    let runtime = tokio::runtime::Runtime::new()?;

    let lexical_started = Instant::now();
    let lexical = LexicalIndex::build_or_update(&workspace, options.compile_workers)?;
    let lexical_build = lexical_started.elapsed();
    let sidecar_bytes = std::fs::metadata(LexicalIndex::cache_path(&workspace))
        .map(|meta| meta.len())
        .unwrap_or(0);

    println!("workspace: {}", workspace.display());
    println!(
        "lexical:   {} symbols, build {}ms, sidecar {:.1} KiB",
        lexical.len(),
        lexical_build.as_millis(),
        sidecar_bytes as f64 / 1024.0
    );

    let embedding = if let Some(endpoint) = &options.endpoint {
        let profile = LlmConfiguration {
            endpoint: endpoint.clone(),
            model: options.model.clone(),
            role: "embeddings".into(),
            api_key: options.api_key.clone(),
            request_timeout_seconds: 120,
            max_tool_rounds: 1,
            review_max_tokens: 0,
            code_max_tokens: 0,
            system_prompt: String::new(),
            model_verbosity: None,
            review_confidence_threshold: 0,
            security_confidence_threshold: 0,
            review_sanitize: Default::default(),
            review_sanitize_timeout: 1_800,
            review_sanitize_alloc_failures: 8,
        };
        let client = EmbeddingClient::from_profile(&profile)?;
        let started = Instant::now();
        let index = runtime.block_on(EmbeddingIndex::build_or_update(
            &workspace,
            &client,
            &AtomicBool::new(false),
            options.compile_workers,
            &AtomicU64::new(0),
            &AtomicU64::new(0),
        ))?;
        println!(
            "semantic:  {} symbols, build {}ms",
            index.len(),
            started.elapsed().as_millis()
        );
        Some((client, index))
    } else {
        println!("semantic:  skipped (pass --endpoint and --model to enable)");
        None
    };

    let mut lexical_scores = Scores::default();
    let mut semantic_scores = Scores::default();
    let mut hybrid_scores = Scores::default();
    let mut gaps = 0usize;

    println!();
    println!(
        "{:<48} {:<10} {:>18} {:>18} {:>18}",
        "query", "kind", "lexical", "semantic", "hybrid"
    );

    for case in fixture.queries.iter().filter(|case| !case.coverage_gap) {
        let started = Instant::now();
        let lexical_hits = lexical.search(
            &workspace,
            &case.query,
            None,
            options.top_k,
            options.budget_tokens,
        );
        let lexical_latency = started.elapsed();
        let (lexical_mrr, lexical_recall) = metrics(&lexical_hits, &case.expected, options.top_k);
        lexical_scores.add(lexical_mrr, lexical_recall, lexical_latency);

        let mut semantic_hits: Vec<SearchHit> = Vec::new();
        let mut hybrid_hits: Vec<SearchHit> = Vec::new();
        let (semantic_cell, hybrid_cell) = match &embedding {
            Some((client, index)) => {
                let started = Instant::now();
                let vector = runtime.block_on(client.embed_one(&case.query))?;
                let embed_latency = started.elapsed();

                let started = Instant::now();
                semantic_hits = index.search(&vector, None, options.top_k, options.budget_tokens);
                let semantic_latency = embed_latency + started.elapsed();
                let (semantic_mrr, semantic_recall) =
                    metrics(&semantic_hits, &case.expected, options.top_k);
                semantic_scores.add(semantic_mrr, semantic_recall, semantic_latency);

                let started = Instant::now();
                hybrid_hits = index.hybrid_search(
                    &workspace,
                    &case.query,
                    &vector,
                    &lexical,
                    None,
                    options.top_k,
                    options.budget_tokens,
                );
                let hybrid_latency = embed_latency + started.elapsed();
                let (hybrid_mrr, hybrid_recall) =
                    metrics(&hybrid_hits, &case.expected, options.top_k);
                hybrid_scores.add(hybrid_mrr, hybrid_recall, hybrid_latency);

                (
                    format!("{semantic_mrr:.3}/{semantic_recall:.3}"),
                    format!("{hybrid_mrr:.3}/{hybrid_recall:.3}"),
                )
            }
            None => ("-".to_string(), "-".to_string()),
        };

        println!(
            "{:<48} {:<10} {:>18} {:>18} {:>18}",
            truncate(&case.query, 48),
            truncate(&case.kind, 10),
            format!("{lexical_mrr:.3}/{lexical_recall:.3}"),
            semantic_cell,
            hybrid_cell
        );
        if options.verbose {
            for (label, hits) in [
                ("lexical", &lexical_hits),
                ("semantic", &semantic_hits),
                ("hybrid", &hybrid_hits),
            ] {
                if hits.is_empty() {
                    continue;
                }
                println!("  {label} hits:");
                for hit in hits.iter().take(options.top_k) {
                    println!(
                        "    {:<24} {:<32} {:.4}{}{}",
                        hit.symbol,
                        format!("{}:{}", hit.file, hit.start_line),
                        hit.score,
                        if hit.exact { " exact" } else { "" },
                        match &hit.expanded_from {
                            Some(from) => format!(" via {from}"),
                            None => String::new(),
                        }
                    );
                }
            }
        }
    }

    println!();
    println!(
        "{:<48} {:<10} {:>18} {:>18} {:>18}",
        "summary (MRR/Recall@5, mean latency)",
        "",
        lexical_scores.summary(),
        semantic_scores.summary(),
        hybrid_scores.summary()
    );

    for case in fixture.queries.iter().filter(|case| case.coverage_gap) {
        gaps += 1;
        println!(
            "coverage gap: {:<40} expected {} (not emitted by the current extractor)",
            truncate(&case.query, 40),
            case.expected.join(", ")
        );
    }
    if gaps > 0 {
        println!("{gaps} coverage-gap queries are excluded from ranking metrics.");
    }

    Ok(())
}

struct Options {
    workspace: PathBuf,
    endpoint: Option<String>,
    model: String,
    api_key: Option<String>,
    top_k: usize,
    budget_tokens: usize,
    compile_workers: usize,
    verbose: bool,
}

impl Options {
    fn parse() -> anyhow::Result<Self> {
        let mut options = Self {
            workspace: std::env::current_dir()?,
            endpoint: None,
            model: "embeddings".to_string(),
            api_key: std::env::var("ORANGU_EMBED_API_KEY").ok(),
            top_k: 5,
            budget_tokens: 16_384,
            compile_workers: 0,
            verbose: false,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--workspace" => {
                    options.workspace = PathBuf::from(
                        args.next()
                            .ok_or_else(|| anyhow::anyhow!("--workspace requires a path"))?,
                    );
                }
                "--endpoint" => {
                    options.endpoint = Some(
                        args.next()
                            .ok_or_else(|| anyhow::anyhow!("--endpoint requires a URL"))?,
                    );
                }
                "--model" => {
                    options.model = args
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("--model requires a name"))?;
                }
                "--api-key" => {
                    options.api_key = Some(
                        args.next()
                            .ok_or_else(|| anyhow::anyhow!("--api-key requires a value"))?,
                    );
                }
                "--top-k" => {
                    options.top_k = args
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("--top-k requires a number"))?
                        .parse()?;
                }
                "--budget-tokens" => {
                    options.budget_tokens = args
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("--budget-tokens requires a number"))?
                        .parse()?;
                }
                "--compile-workers" => {
                    options.compile_workers = args
                        .next()
                        .ok_or_else(|| anyhow::anyhow!("--compile-workers requires a number"))?
                        .parse()?;
                }
                "--verbose" => options.verbose = true,
                "--help" | "-h" => {
                    println!(
                        "usage: orangu-search-bench [--workspace PATH] [--endpoint URL] \
                         [--model NAME] [--api-key KEY] [--top-k N] [--budget-tokens N] \
                         [--compile-workers N]"
                    );
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown argument: {other}"),
            }
        }
        Ok(options)
    }
}

/// MRR and Recall@k for one query. A hit is relevant when its symbol, id, or
/// id suffix matches one of the expected names.
fn metrics(hits: &[SearchHit], expected: &[String], k: usize) -> (f64, f64) {
    let reciprocal_rank = hits
        .iter()
        .position(|hit| is_relevant(hit, expected))
        .map(|position| 1.0 / (position + 1) as f64)
        .unwrap_or(0.0);
    let recall = if hits.iter().take(k).any(|hit| is_relevant(hit, expected)) {
        1.0
    } else {
        0.0
    };
    (reciprocal_rank, recall)
}

fn is_relevant(hit: &SearchHit, expected: &[String]) -> bool {
    expected.iter().any(|name| {
        hit.symbol.eq_ignore_ascii_case(name)
            || hit.id.eq_ignore_ascii_case(name)
            || hit
                .id
                .to_ascii_lowercase()
                .ends_with(&format!("::{}", name.to_ascii_lowercase()))
    })
}

fn truncate(value: &str, width: usize) -> String {
    if value.chars().count() <= width {
        return value.to_string();
    }
    let mut truncated: String = value.chars().take(width.saturating_sub(1)).collect();
    truncated.push('…');
    truncated
}
