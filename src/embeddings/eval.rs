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

//! Retrieval quality of semantic search: queries with the files that should
//! answer them, scored by where those files land in the ranking.
//!
//! Scoring is per file, not per chunk — a hit's chunks are collapsed to the
//! file's best rank — since a file is what a person or an agent opens next.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::SearchHit;

#[derive(Debug, Clone, Deserialize)]
pub struct EvalCase {
    pub query: String,
    /// Workspace-relative files; a path ending in `/` matches anything under it.
    pub expect: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct EvalFile {
    cases: Vec<EvalCase>,
}

pub fn load_cases(path: &Path) -> Result<Vec<EvalCase>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("could not read {}", path.display()))?;
    let file: EvalFile = serde_yaml::from_str(&text)
        .with_context(|| format!("could not parse {}", path.display()))?;
    anyhow::ensure!(!file.cases.is_empty(), "{} has no cases", path.display());
    for case in &file.cases {
        anyhow::ensure!(
            !case.query.trim().is_empty() && !case.expect.is_empty(),
            "every case needs a query and at least one expected file"
        );
    }
    Ok(file.cases)
}

/// The distinct files of `hits`, in ranking order.
pub fn ranked_files(hits: &[SearchHit]) -> Vec<String> {
    let mut files: Vec<String> = Vec::new();
    for hit in hits {
        if !files.contains(&hit.file) {
            files.push(hit.file.clone());
        }
    }
    files
}

fn normalize(path: &str) -> String {
    let path = path.replace('\\', "/");
    path.strip_prefix("./").unwrap_or(&path).to_string()
}

fn matches(expected: &str, file: &str) -> bool {
    let (expected, file) = (normalize(expected), normalize(file));
    if expected.ends_with('/') {
        file.starts_with(&expected)
    } else {
        file == expected
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CaseScore {
    /// 1-based rank of each expected file, `None` when it was not returned.
    pub ranks: Vec<Option<usize>>,
}

impl CaseScore {
    pub fn first_rank(&self) -> Option<usize> {
        self.ranks.iter().flatten().copied().min()
    }

    pub fn recall_at(&self, k: usize) -> f64 {
        let found = self
            .ranks
            .iter()
            .filter(|r| r.is_some_and(|r| r <= k))
            .count();
        found as f64 / self.ranks.len() as f64
    }
}

pub fn score_case(files: &[String], expect: &[String]) -> CaseScore {
    let ranks = expect
        .iter()
        .map(|e| files.iter().position(|f| matches(e, f)).map(|i| i + 1))
        .collect();
    CaseScore { ranks }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Summary {
    pub cases: usize,
    /// Share of cases whose top file is an expected one.
    pub hit_at_1: f64,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    /// Mean reciprocal rank of the first expected file; a miss counts as 0.
    pub mrr: f64,
}

pub fn summarize(scores: &[CaseScore]) -> Summary {
    let n = scores.len().max(1) as f64;
    let mean = |f: &dyn Fn(&CaseScore) -> f64| scores.iter().map(f).sum::<f64>() / n;
    Summary {
        cases: scores.len(),
        hit_at_1: mean(&|s| f64::from(u8::from(s.first_rank() == Some(1)))),
        recall_at_5: mean(&|s| s.recall_at(5)),
        recall_at_10: mean(&|s| s.recall_at(10)),
        mrr: mean(&|s| s.first_rank().map_or(0.0, |r| 1.0 / r as f64)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    fn hit(file: &str) -> SearchHit {
        SearchHit {
            id: String::new(),
            symbol: String::new(),
            file: file.to_string(),
            start_line: 1,
            end_line: 1,
            score: 0.0,
            expanded_from: None,
        }
    }

    #[test]
    fn ranked_files_collapses_chunks_of_one_file() {
        let hits = [hit("a.rs"), hit("b.rs"), hit("a.rs"), hit("c.rs")];
        assert_eq!(ranked_files(&hits), files(&["a.rs", "b.rs", "c.rs"]));
    }

    #[test]
    fn score_case_ranks_each_expected_file() {
        let ranked = files(&["x.rs", "src/a.rs", "y.rs"]);
        let score = score_case(&ranked, &files(&["src/a.rs", "missing.rs"]));
        assert_eq!(score.ranks, vec![Some(2), None]);
        assert_eq!(score.first_rank(), Some(2));
        assert_eq!(score.recall_at(1), 0.0);
        assert_eq!(score.recall_at(2), 0.5);
    }

    #[test]
    fn expected_paths_are_normalized_and_directories_match_by_prefix() {
        let ranked = files(&["src/pdf/tables.rs"]);
        assert_eq!(
            score_case(&ranked, &files(&["./src\\pdf\\tables.rs"])).ranks,
            vec![Some(1)]
        );
        assert_eq!(
            score_case(&ranked, &files(&["src/pdf/"])).ranks,
            vec![Some(1)]
        );
        assert_eq!(score_case(&ranked, &files(&["src/pd/"])).ranks, vec![None]);
    }

    #[test]
    fn summarize_averages_over_cases() {
        let scores = [
            CaseScore {
                ranks: vec![Some(1)],
            },
            CaseScore {
                ranks: vec![Some(4), Some(12)],
            },
            CaseScore { ranks: vec![None] },
        ];
        let s = summarize(&scores);
        assert_eq!(s.cases, 3);
        assert!((s.hit_at_1 - 1.0 / 3.0).abs() < 1e-9);
        assert!((s.recall_at_5 - 1.5 / 3.0).abs() < 1e-9);
        assert!((s.recall_at_10 - 1.5 / 3.0).abs() < 1e-9);
        assert!((s.mrr - 1.25 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn the_bundled_case_file_parses_and_names_real_files() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let cases = load_cases(&root.join("src/bin/orangu-bench/fixtures/retrieval.yaml")).unwrap();
        for case in &cases {
            for path in &case.expect {
                assert!(root.join(path).exists(), "{path} ({})", case.query);
            }
        }
    }
}
