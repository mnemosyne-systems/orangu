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

//! What the server found out about the machine it is on, and what it chose
//! because of it — one record, one line each in the log.
//!
//! Every machine is different, and the same binary has to be fast on each:
//! whether a prompt runs on the GPU or the cores, whether the NPU is worth
//! its hand-offs, which attention and weight kernels the cores can run, how
//! much memory there is for a faster copy of the weights. Each of those is
//! decided in its own layer — `prefill_backend`, `npu_tool`, `attention`,
//! `prompt_weights` — and most of them by *measuring* rather than assuming
//! (`doc/PERF-ALL.md`). This is where they write down what they measured
//! and what they chose, so an operator reads one set of lines
//!
//! ```text
//! orangu-server: [adapt] prompts: cpu — a 64-token 1536x6144 GEMM takes 3.1 ms on the cpu, 6.3 ms on the device
//! orangu-server: [adapt] prompt weights: int8 copies — 1.6 GB, a 256-token FFN GEMM 4.1 ms against 6.9 ms on the file's
//! ```
//!
//! rather than a scatter of formats, and `/props` can answer the same
//! question for a program.

use std::sync::Mutex;

/// One decision: which layer made it, what it chose, and why.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Decision {
    /// The layer that decided — `prompts`, `npu`, `attention`, …
    pub layer: &'static str,
    /// What it chose, in a few words.
    pub chose: String,
    /// What it detected or measured to choose it.
    pub because: String,
}

static DECISIONS: Mutex<Vec<Decision>> = Mutex::new(Vec::new());

/// Records a decision and logs it. A layer that decides again (a model
/// reloaded) replaces its earlier line rather than adding a second.
pub fn note(layer: &'static str, chose: impl Into<String>, because: impl Into<String>) {
    let decision = Decision {
        layer,
        chose: chose.into(),
        because: because.into(),
    };
    log::info!(
        "orangu-server: [adapt] {}: {} — {}",
        decision.layer,
        decision.chose,
        decision.because
    );
    if let Ok(mut all) = DECISIONS.lock() {
        all.retain(|d| d.layer != layer);
        all.push(decision);
    }
}

/// Every decision recorded so far, in the order they were made.
pub fn decisions() -> Vec<Decision> {
    DECISIONS.lock().map(|all| all.clone()).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_layer_that_decides_again_replaces_its_line() {
        note("test-layer", "a", "first");
        note("test-layer", "b", "second");
        let mine: Vec<_> = decisions()
            .into_iter()
            .filter(|d| d.layer == "test-layer")
            .collect();
        assert_eq!(mine.len(), 1);
        assert_eq!(mine[0].chose, "b");
    }
}
