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

//! `--retrieval`: how well semantic `/search` finds the files a query is
//! about, scored by `orangu::embeddings::eval` over the same index and search
//! `/search` uses. Each query runs twice — cosine alone, then with the
//! knowledge graph's call-edge expansion — so a change to ranking is judged
//! against both.

use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64};

use orangu::embeddings::eval::{self, CaseScore, Summary};
use orangu::embeddings::{EmbeddingClient, EmbeddingIndex};

use crate::Args;

const TOP_K: usize = 10;

pub fn run_retrieval(args: &Args, cases_path: &Path) -> anyhow::Result<()> {
    let cases = eval::load_cases(cases_path)?;
    // Resolved as `orangu` resolves it, so the embedding cache is shared.
    let workspace =
        orangu::workspaces::resolve_workspace_root(Some(args.retrieval_workspace.clone()))?;
    // The index stores workspace-relative paths and search stats them as such.
    std::env::set_current_dir(&workspace)?;

    let client = EmbeddingClient::new(
        &args.url,
        args.model.as_deref().unwrap_or_default(),
        None,
        args.timeout,
    )?;
    let runtime = tokio::runtime::Runtime::new()?;
    let (cancel, progress, eta) = (AtomicBool::new(false), AtomicU64::new(0), AtomicU64::new(0));
    if !args.json {
        eprintln!("indexing {} ...", workspace.display());
    }
    let index = runtime.block_on(EmbeddingIndex::build_or_update(
        &workspace, &client, &cancel, 0, &progress, &eta,
    ))?;
    anyhow::ensure!(
        !index.is_empty(),
        "the index of {} is empty",
        workspace.display()
    );
    let graph = orangu::agents::hooks::run_session_start_hook(&workspace).store;

    let mut plain = Vec::new();
    let mut expanded = Vec::new();
    let mut rows = Vec::new();
    for case in &cases {
        let vector = runtime.block_on(client.embed_one(&case.query))?;
        let score = |g: Option<&orangu::graph::store::GraphStore>| {
            let hits = index.search(&vector, g, TOP_K, usize::MAX);
            eval::score_case(&eval::ranked_files(&hits), &case.expect)
        };
        let (p, e) = (score(None), score(Some(&graph)));
        rows.push((case.query.as_str(), p.clone(), e.clone()));
        plain.push(p);
        expanded.push(e);
    }
    let (plain_sum, expanded_sum) = (eval::summarize(&plain), eval::summarize(&expanded));

    if args.json {
        let cases: Vec<_> = rows
            .iter()
            .map(|(q, p, e)| serde_json::json!({ "query": q, "cosine": p, "graph": e }))
            .collect();
        println!(
            "{}",
            serde_json::json!({
                "indexed_symbols": index.len(),
                "top_k": TOP_K,
                "cosine": plain_sum,
                "graph": expanded_sum,
                "cases": cases,
            })
        );
        return Ok(());
    }

    println!("{:>6} {:>6}  query", "cosine", "graph");
    for (query, p, e) in &rows {
        println!("{:>6} {:>6}  {query}", rank(p), rank(e));
    }
    println!();
    println!(
        "{:>8} | {:>6} | {:>6} | {:>6} | {:>6}",
        "", "hit@1", "R@5", "R@10", "MRR"
    );
    println!("{}", "-".repeat(46));
    print_summary("cosine", &plain_sum);
    print_summary("graph", &expanded_sum);
    println!(
        "\n{} cases, {} symbols indexed, top {TOP_K} results per query",
        plain_sum.cases,
        index.len()
    );
    Ok(())
}

fn rank(score: &CaseScore) -> String {
    score
        .first_rank()
        .map_or("-".to_string(), |r| r.to_string())
}

fn print_summary(label: &str, s: &Summary) {
    println!(
        "{label:>8} | {:>6.3} | {:>6.3} | {:>6.3} | {:>6.3}",
        s.hit_at_1, s.recall_at_5, s.recall_at_10, s.mrr
    );
}
