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

//! Where a prompt runs: through the tree, part by part, every node its own
//! layers — or on one node that holds the whole model, alone, with every
//! other node then given the rows of its own layers
//! ([`super::protocol::FEATURE_ROWS`]). Decode goes on through the tree
//! either way.
//!
//! [`decide`] weighs the two by what was measured: each node's prompt speed
//! and fixed cost (`super::speed`), each link's bandwidth (its echo), and
//! what earlier prompts really took. A prompt goes to one node only when that
//! is clearly faster; a short one, or one nothing measured can price, stays
//! on the tree.

use crate::engine::arch::ModelForward;
use crate::engine::kv_cache::KvCache;
use anyhow::{Result, ensure};
use std::ops::Range;
use std::time::Duration;

/// Prompts shorter than this stay on the tree: a hand-over's round trips
/// and copies cost more than the few parts it would save.
pub const MIN_TOKENS: usize = 256;

/// How much faster one node has to be predicted: its time at most this
/// share of the tree's.
pub const CLEARLY: f64 = 0.75;

/// Bytes a second a node copies rows into its cache at, on each side of a
/// link: what taking the rows in costs beside sending them.
const INGEST_BYTES_PER_S: f64 = 1e9;

/// The most bytes of rows one message carries; a layer's rows are never
/// split, so a range is cut between layers to stay under it.
pub const GROUP_BYTES: u64 = 64 << 20;

/// The most bytes one layer's rows may take: a message holds at least one
/// layer, and a frame is at most [`super::protocol::MAX_FRAME`].
pub const LAYER_BYTES: u64 = (super::protocol::MAX_FRAME / 2) as u64;

/// `ORANGU_WORKERS_PREFILL`: where prompts run, for testing a tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mode {
    /// By [`decide`]'s measured prediction (unset, `auto`).
    Measured,
    /// Always through the tree (`0`, `off`, `tree`).
    Tree,
    /// Every prompt on one node that holds the whole model (`force`): the
    /// one predicted fastest, else the first worker of the plan that holds
    /// it, else this node; or the node named (any other value).
    Force(Option<String>),
}

impl Mode {
    pub fn parse(value: Option<&str>) -> Self {
        match value.map(str::trim) {
            None | Some("") => Self::Measured,
            Some(v) if v.eq_ignore_ascii_case("auto") => Self::Measured,
            Some(v)
                if ["0", "off", "no", "false", "tree"]
                    .iter()
                    .any(|off| v.eq_ignore_ascii_case(off)) =>
            {
                Self::Tree
            }
            Some(v) if v.eq_ignore_ascii_case("force") => Self::Force(None),
            Some(name) => Self::Force(Some(name.to_string())),
        }
    }

    /// The mode the environment sets, read once.
    pub fn from_env() -> Self {
        static MODE: std::sync::OnceLock<Mode> = std::sync::OnceLock::new();
        MODE.get_or_init(|| Self::parse(std::env::var("ORANGU_WORKERS_PREFILL").ok().as_deref()))
            .clone()
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::Measured => "measured",
            Self::Tree => "tree",
            Self::Force(_) => "force",
        }
    }
}

/// One node of the plan, with the layers it runs itself: what its part of a
/// prompt costs on the tree.
#[derive(Clone, Debug, Default)]
pub struct TreeNode {
    /// The bytes of its layers' weights.
    pub bytes: u64,
    /// Gigabytes of weights a second through a measured prompt chunk, and
    /// that chunk's cost whatever its layers; `0` when not measured.
    pub rate: f32,
    pub fixed_ms: f32,
    /// The link from this node's parent to it, when the top-level node timed
    /// it: round trip and bytes a second each way.
    pub link: Option<(Duration, f64)>,
}

/// A worker directly below the top-level node: who its rows go to.
#[derive(Clone, Debug, Default)]
pub struct Branch {
    /// Its address, as its stage is named.
    pub addr: String,
    /// The rows of its layers and the workers' below it, in bytes a position.
    pub row_bytes: u64,
    pub link: Option<(Duration, f64)>,
    /// It takes rows ([`super::protocol::FEATURE_ROWS`]).
    pub takes_rows: bool,
}

/// A node that holds the whole model and may run a prompt alone.
#[derive(Clone, Debug, Default)]
pub struct Candidate {
    /// `None` for the top-level node itself, else the address of the branch
    /// it is.
    pub addr: Option<String>,
    /// Its name, as plans print it.
    pub name: String,
    pub rate: f32,
    pub fixed_ms: f32,
    /// Tokens a second a hand-over to it took before — its compute, the
    /// rows and taking them in — when one did.
    pub observed: Option<f64>,
}

/// What [`decide`] weighs.
#[derive(Clone, Debug, Default)]
pub struct Inputs {
    /// The tokens the tree would run: the prompt past any reused prefix.
    pub tree_tokens: usize,
    /// The tokens one node would run: every position, from 0.
    pub all_tokens: usize,
    /// The width the tree's parts are cut to, and whether they overlap.
    pub part: usize,
    pub pipelined: bool,
    /// Bytes a position of the residual stream takes between two nodes.
    pub hidden_bytes: f64,
    /// Every layer's weights, in bytes.
    pub total_bytes: u64,
    /// The plan's nodes, in layer order.
    pub nodes: Vec<TreeNode>,
    /// The top-level node's own rows, in bytes a position.
    pub local_row_bytes: u64,
    pub branches: Vec<Branch>,
    pub candidates: Vec<Candidate>,
    /// The largest layer's rows, in bytes a position.
    pub widest_layer: u64,
    /// Tokens a second earlier prompts took through this tree, when one long
    /// enough ran.
    pub observed_tree: Option<f64>,
    /// Why no node can take a prompt at all, when none can: a model that
    /// keeps a recurrent state.
    pub refused: Option<String>,
}

/// Where a prompt runs, and why.
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    /// `None`: through the tree. `Some(None)`: on the top-level node.
    /// `Some(Some(addr))`: on the branch at `addr`.
    pub on: Option<Option<String>>,
    /// The node's name, as plans print it; empty for the tree.
    pub name: String,
    /// The predictions, in seconds, where there are any.
    pub tree_s: Option<f64>,
    pub node_s: Option<f64>,
    pub reason: String,
}

impl Choice {
    fn tree(reason: impl Into<String>, tree_s: Option<f64>, node_s: Option<f64>) -> Self {
        Self {
            on: None,
            name: String::new(),
            tree_s,
            node_s,
            reason: reason.into(),
        }
    }
}

/// The tree's predicted time for `inputs.tree_tokens`: what earlier prompts
/// took when they measured it, else each node's part — its layers at its
/// measured rate, its fixed cost, and the hidden state over its link — once
/// for every node to fill the pipeline and then once a part at the pace of
/// the slowest. `None` when a node with layers was never measured.
pub fn tree_seconds(inputs: &Inputs) -> Option<f64> {
    if let Some(rate) = inputs.observed_tree.filter(|r| *r > 0.0) {
        return Some(inputs.tree_tokens as f64 / rate);
    }
    let part = inputs.part.max(1);
    let scale = part as f64 / super::speed::PROMPT_TOKENS as f64;
    let mut times = Vec::new();
    for node in inputs.nodes.iter().filter(|n| n.bytes > 0) {
        if node.rate <= 0.0 {
            return None;
        }
        let compute = node.bytes as f64 / (node.rate as f64 * 1e9) * scale;
        let hop = node.link.map_or(0.0, |(rtt, bandwidth)| {
            part as f64 * inputs.hidden_bytes / bandwidth.max(1.0) + rtt.as_secs_f64() / 2.0
        });
        times.push(compute + node.fixed_ms as f64 / 1e3 + hop);
    }
    if times.is_empty() {
        return None;
    }
    let parts = inputs.tree_tokens.div_ceil(part) as f64;
    let sum: f64 = times.iter().sum();
    let slowest = times.iter().copied().fold(0.0, f64::max);
    Some(if inputs.pipelined {
        sum + (parts - 1.0).max(0.0) * slowest
    } else {
        parts * sum
    })
}

/// `candidate`'s predicted time for the whole prompt: what a hand-over to it
/// took before when one did, else every layer at its measured rate, part by
/// part, then every row it does not keep sent to the node that runs its layer
/// and taken in there. `None` when it was never measured or a link it needs
/// was never timed.
pub fn node_seconds(inputs: &Inputs, candidate: &Candidate) -> Option<f64> {
    if let Some(rate) = candidate.observed.filter(|r| *r > 0.0) {
        return Some(inputs.all_tokens as f64 / rate);
    }
    if candidate.rate <= 0.0 {
        return None;
    }
    let part = inputs.part.max(1);
    let scale = part as f64 / super::speed::PROMPT_TOKENS as f64;
    let parts = inputs.all_tokens.div_ceil(part) as f64;
    let compute = parts
        * (inputs.total_bytes as f64 / (candidate.rate as f64 * 1e9) * scale
            + candidate.fixed_ms as f64 / 1e3);
    let n = inputs.all_tokens as f64;
    let over = |link: Option<(Duration, f64)>, bytes: f64| -> Option<f64> {
        let (rtt, bandwidth) = link?;
        Some(bytes / bandwidth.max(1.0) + 2.0 * rtt.as_secs_f64())
    };
    let mut rows = 0.0;
    let mut moved = 0.0;
    match &candidate.addr {
        None => {
            for branch in &inputs.branches {
                let bytes = n * branch.row_bytes as f64;
                rows += over(branch.link, bytes)?;
                moved += bytes;
            }
            moved += n * inputs.local_row_bytes as f64;
        }
        Some(addr) => {
            let fast = inputs.branches.iter().find(|b| &b.addr == addr)?;
            let mut pulled = n * inputs.local_row_bytes as f64;
            for branch in inputs.branches.iter().filter(|b| &b.addr != addr) {
                let bytes = n * branch.row_bytes as f64;
                rows += over(branch.link, bytes)?;
                pulled += bytes;
            }
            rows += over(fast.link, pulled)?;
            moved += pulled + n * fast.row_bytes as f64;
        }
    }
    Some(compute + rows + 2.0 * moved / INGEST_BYTES_PER_S)
}

/// Where a prompt of `inputs` runs.
pub fn decide(inputs: &Inputs, mode: &Mode) -> Choice {
    if *mode == Mode::Tree {
        return Choice::tree(
            "ORANGU_WORKERS_PREFILL keeps prompts on the tree",
            None,
            None,
        );
    }
    if let Some(why) = &inputs.refused {
        return Choice::tree(why.clone(), None, None);
    }
    if inputs.branches.is_empty() {
        return Choice::tree("no worker", None, None);
    }
    if let Some(branch) = inputs.branches.iter().find(|b| !b.takes_rows) {
        return Choice::tree(format!("{} cannot take rows", branch.addr), None, None);
    }
    if inputs.widest_layer as f64 * inputs.all_tokens as f64 > LAYER_BYTES as f64 {
        return Choice::tree("a layer's rows are too large for one message", None, None);
    }
    if inputs.candidates.is_empty() {
        return Choice::tree("no node of the plan holds the whole model", None, None);
    }
    let tree_s = tree_seconds(inputs);
    let priced: Vec<(&Candidate, Option<f64>)> = inputs
        .candidates
        .iter()
        .map(|c| (c, node_seconds(inputs, c)))
        .collect();
    let fastest = priced
        .iter()
        .filter_map(|(c, s)| s.map(|s| (*c, s)))
        .min_by(|a, b| a.1.total_cmp(&b.1));
    let chosen = |candidate: &Candidate, node_s: Option<f64>, reason: String| Choice {
        on: Some(candidate.addr.clone()),
        name: candidate.name.clone(),
        tree_s,
        node_s,
        reason,
    };
    match mode {
        Mode::Force(Some(name)) => {
            return match priced
                .iter()
                .find(|(c, _)| c.name == *name || c.addr.as_deref() == Some(name.as_str()))
            {
                Some((c, s)) => chosen(c, *s, "ORANGU_WORKERS_PREFILL names it".to_string()),
                None => Choice::tree(
                    format!("ORANGU_WORKERS_PREFILL names {name}, which holds no whole model here"),
                    tree_s,
                    None,
                ),
            };
        }
        Mode::Force(None) => {
            let (candidate, node_s) = match fastest {
                Some((c, s)) => (c, Some(s)),
                None => {
                    let c = inputs
                        .candidates
                        .iter()
                        .find(|c| c.addr.is_some())
                        .unwrap_or(&inputs.candidates[0]);
                    (c, None)
                }
            };
            return chosen(
                candidate,
                node_s,
                "ORANGU_WORKERS_PREFILL=force".to_string(),
            );
        }
        _ => {}
    }
    if inputs.tree_tokens < MIN_TOKENS {
        return Choice::tree(
            format!("{} tokens are fewer than {MIN_TOKENS}", inputs.tree_tokens),
            None,
            None,
        );
    }
    let Some(tree) = tree_s else {
        return Choice::tree(
            "a node of the plan has no measured prompt speed",
            None,
            None,
        );
    };
    let Some((candidate, node_s)) = fastest else {
        return Choice::tree(
            "no node that holds the whole model has a measured prompt speed and timed links",
            Some(tree),
            None,
        );
    };
    if node_s <= CLEARLY * tree {
        chosen(
            candidate,
            Some(node_s),
            format!(
                "predicted {:.1} s there, {:.1} s through the tree",
                node_s, tree
            ),
        )
    } else {
        Choice::tree(
            format!(
                "predicted {:.1} s on {}, {:.1} s through the tree",
                node_s, candidate.name, tree
            ),
            Some(tree),
            Some(node_s),
        )
    }
}

/// Whether a model's caches can be handed between nodes: one row slot per
/// layer, and no recurrent state, which is of every position it saw and
/// would have to travel whole. `Err` says why not.
pub fn movable(model: &dyn ModelForward) -> Result<(), String> {
    let n_layer = model.config().n_layer;
    let template = model.new_kv_cache_for_layers(0..n_layer, 1);
    if !template.recurrent.is_empty() {
        return Err("the model keeps a recurrent state, which stays on the tree".to_string());
    }
    if template.layers.len() != n_layer {
        return Err("the model's cache is not one slot a layer".to_string());
    }
    Ok(())
}

/// Each layer's rows, in bytes a position: keys and values in `f32`.
pub fn row_bytes(model: &dyn ModelForward) -> Vec<u64> {
    let n_layer = model.config().n_layer;
    model
        .new_kv_cache_for_layers(0..n_layer, 1)
        .layers
        .iter()
        .map(|l| 8 * l.kv_dim() as u64)
        .collect()
}

/// The width a whole prompt is run in on one node: the parts the tree cuts
/// a prompt into, so its rows are computed in the same steps.
pub fn width() -> usize {
    match super::pipeline::sub_chunk() {
        0 => super::speed::PROMPT_TOKENS,
        width => width,
    }
}

/// Runs `tokens`, from position 0, through every layer of `model` in
/// `width`-token parts, as a tree's nodes run their layers
/// (`ModelForward::forward_layers`): a cache of every layer holding their
/// rows on the host, and the last position's logits.
pub fn run_whole(
    model: &dyn ModelForward,
    tokens: &[u32],
    width: usize,
) -> Result<(KvCache, Vec<f32>)> {
    ensure!(!tokens.is_empty(), "a prompt needs at least one token");
    movable(model).map_err(|why| anyhow::anyhow!(why))?;
    let n_layer = model.config().n_layer;
    let n_embd = model.config().n_embd;
    let mut cache = model.new_kv_cache_for_layers(0..n_layer, tokens.len());
    let mut last = Vec::new();
    for (i, part) in tokens.chunks(width.max(1)).enumerate() {
        let hidden = model.embed(part)?;
        let out = model.forward_layers(&mut cache, hidden, part, 0..n_layer, i * width.max(1))?;
        ensure!(
            out.len() == part.len() * n_embd,
            "{} values out of {} positions {n_embd} wide",
            out.len(),
            part.len()
        );
        last = out[out.len() - n_embd..].to_vec();
    }
    if let Some(vulkan) = model.vulkan_backend() {
        vulkan.fill_deferred_kv_rows(&mut cache);
        cache.pull_decoded_rows(vulkan);
        vulkan.prompt_prefilled();
    }
    let held = cache.host_committed_len();
    ensure!(
        held >= tokens.len(),
        "the prompt's rows are on the device: {held} of {} positions on the host",
        tokens.len()
    );
    let logits = model
        .head(&last, 1)?
        .pop()
        .ok_or_else(|| anyhow::anyhow!("no logits"))?;
    Ok((cache, logits))
}

/// The rows `cache` holds of `layers`, their first `len` positions, in the
/// form [`super::prefixes::encode_rows`] sends, for sessions of `n_ctx`
/// positions.
pub fn encode_layers(
    model: &dyn ModelForward,
    header_hash: [u8; 32],
    cache: &KvCache,
    layers: Range<usize>,
    n_ctx: usize,
    len: usize,
) -> Result<Vec<u8>> {
    let mut slice = model.new_kv_cache_for_layers(layers.clone(), len.max(1));
    slice.copy_layers_from(cache, len, layers.clone())?;
    Ok(super::prefixes::encode_rows(
        header_hash,
        layers,
        n_ctx,
        len,
        &slice,
    ))
}

/// What a node expects of rows of `layers` it takes in: its model's cache
/// shape for them, `n_ctx` and `len` positions.
pub fn expected_shape(model: &dyn ModelForward, layers: Range<usize>) -> Vec<u8> {
    model.new_kv_cache_for_layers(layers, 1).structure_tag()
}

/// `range` cut between layers so that no piece holds more than
/// [`GROUP_BYTES`] of rows — a layer whose rows weigh nothing (one that reads
/// another's) never starts a piece — given each layer's rows in bytes.
pub fn groups(range: Range<usize>, layer_bytes: impl Fn(usize) -> u64) -> Vec<Range<usize>> {
    let mut out: Vec<Range<usize>> = Vec::new();
    let mut start = range.start;
    let mut held = 0u64;
    for il in range.clone() {
        let bytes = layer_bytes(il);
        if bytes > 0 && held > 0 && held + bytes > GROUP_BYTES {
            out.push(start..il);
            start = il;
            held = 0;
        }
        held += bytes;
    }
    if start < range.end {
        out.push(start..range.end);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: Duration = Duration::from_millis(1);
    /// 100 MB/s.
    const LAN: f64 = 100e6;

    /// A lead and two workers, each running a third of 900 MB of layers at
    /// 0.5 GB/s, and the second worker holding the whole model at 20 GB/s.
    fn inputs(tokens: usize) -> Inputs {
        let node = |link| TreeNode {
            bytes: 300_000_000,
            rate: 0.5,
            fixed_ms: 5.0,
            link,
        };
        let branch = |addr: &str| Branch {
            addr: addr.to_string(),
            row_bytes: 8 * 1024 * 5,
            link: Some((MS, LAN)),
            takes_rows: true,
        };
        Inputs {
            tree_tokens: tokens,
            all_tokens: tokens,
            part: 128,
            pipelined: true,
            hidden_bytes: 2.0 * 2048.0,
            total_bytes: 900_000_000,
            nodes: vec![node(None), node(Some((MS, LAN))), node(Some((MS, LAN)))],
            local_row_bytes: 8 * 1024 * 5,
            branches: vec![branch("b:8400"), branch("c:8400")],
            candidates: vec![Candidate {
                addr: Some("c:8400".to_string()),
                name: "c:8400".to_string(),
                rate: 20.0,
                fixed_ms: 2.0,
                observed: None,
            }],
            widest_layer: 8 * 1024,
            observed_tree: None,
            refused: None,
        }
    }

    #[test]
    fn a_long_prompt_goes_to_the_fast_node() {
        let choice = decide(&inputs(2765), &Mode::Measured);
        assert_eq!(choice.on, Some(Some("c:8400".to_string())), "{choice:?}");
        assert_eq!(choice.name, "c:8400");
        let (tree, node) = (choice.tree_s.unwrap(), choice.node_s.unwrap());
        assert!(node <= CLEARLY * tree, "{node} against {tree}");
    }

    /// A node about as fast as the tree is not clearly faster.
    #[test]
    fn a_node_not_clearly_faster_leaves_the_prompt_on_the_tree() {
        let mut slow = inputs(2765);
        slow.candidates[0].rate = 1.2;
        let choice = decide(&slow, &Mode::Measured);
        assert_eq!(choice.on, None, "{choice:?}");
        assert!(choice.node_s.unwrap() > CLEARLY * choice.tree_s.unwrap());
    }

    /// A slow link costs the hand-over its rows: the same fast node, behind
    /// a link a thousand times slower, loses.
    #[test]
    fn the_rows_over_a_slow_link_count() {
        let mut far = inputs(2765);
        for branch in &mut far.branches {
            branch.link = Some((MS, LAN / 1000.0));
        }
        assert_eq!(decide(&far, &Mode::Measured).on, None);
    }

    #[test]
    fn a_short_prompt_stays_on_the_tree() {
        let choice = decide(&inputs(MIN_TOKENS - 1), &Mode::Measured);
        assert_eq!(choice.on, None);
        assert!(choice.reason.contains("fewer"), "{}", choice.reason);
    }

    /// Nothing to price a side by: the tree.
    #[test]
    fn unknown_measurements_keep_the_tree() {
        let mut unmeasured = inputs(4000);
        unmeasured.nodes[1].rate = 0.0;
        assert_eq!(decide(&unmeasured, &Mode::Measured).on, None);

        let mut untimed = inputs(4000);
        untimed.branches[0].link = None;
        let choice = decide(&untimed, &Mode::Measured);
        assert_eq!(
            (choice.on, choice.node_s),
            (None, None),
            "{}",
            choice.reason
        );

        let mut no_rate = inputs(4000);
        no_rate.candidates[0].rate = 0.0;
        assert_eq!(decide(&no_rate, &Mode::Measured).on, None);
    }

    /// What was observed outranks what was measured, both ways.
    #[test]
    fn observed_times_decide_over_measured_ones() {
        let mut seen = inputs(2765);
        seen.observed_tree = Some(10_000.0);
        assert_eq!(decide(&seen, &Mode::Measured).on, None);
        seen.candidates[0].observed = Some(100_000.0);
        assert_eq!(
            decide(&seen, &Mode::Measured).on,
            Some(Some("c:8400".to_string()))
        );
    }

    #[test]
    fn a_tree_that_cannot_take_rows_or_a_recurrent_model_keeps_the_tree() {
        let mut old = inputs(4000);
        old.branches[0].takes_rows = false;
        assert_eq!(decide(&old, &Mode::Force(None)).on, None);
        let mut recurrent = inputs(4000);
        recurrent.refused = Some("recurrent".to_string());
        assert_eq!(decide(&recurrent, &Mode::Force(None)).on, None);
        let mut nobody = inputs(4000);
        nobody.candidates.clear();
        assert_eq!(decide(&nobody, &Mode::Force(None)).on, None);
    }

    #[test]
    fn the_override_forces_either_way() {
        assert_eq!(decide(&inputs(100_000), &Mode::Tree).on, None);
        let mut short = inputs(3);
        short.candidates[0].rate = 0.0;
        short.candidates.insert(
            0,
            Candidate {
                addr: None,
                name: "a:8400".to_string(),
                ..Candidate::default()
            },
        );
        // Unmeasured: a worker before this node.
        assert_eq!(
            decide(&short, &Mode::Force(None)).on,
            Some(Some("c:8400".to_string()))
        );
        assert_eq!(
            decide(&short, &Mode::Force(Some("a:8400".to_string()))).on,
            Some(None)
        );
        assert_eq!(
            decide(&short, &Mode::Force(Some("x:1".to_string()))).on,
            None
        );
    }

    #[test]
    fn the_mode_reads_off_and_on() {
        assert_eq!(Mode::parse(None), Mode::Measured);
        assert_eq!(Mode::parse(Some("auto")), Mode::Measured);
        assert_eq!(Mode::parse(Some("0")), Mode::Tree);
        assert_eq!(Mode::parse(Some("off")), Mode::Tree);
        assert_eq!(Mode::parse(Some("force")), Mode::Force(None));
        assert_eq!(
            Mode::parse(Some("b:8400")),
            Mode::Force(Some("b:8400".to_string()))
        );
    }

    /// Pieces stay under the limit, cut between layers, and a layer of no
    /// rows rides with the one before it.
    #[test]
    fn a_range_is_cut_into_pieces_between_layers() {
        let mib = 1u64 << 20;
        let bytes = |il: usize| match il {
            4 | 5 => 0,
            _ => 30 * mib,
        };
        assert_eq!(groups(0..8, bytes), vec![0..2, 2..6, 6..8]);
        assert_eq!(groups(3..3, bytes), Vec::<Range<usize>>::new());
        assert_eq!(groups(4..6, bytes), vec![4..6]);
        assert_eq!(groups(0..1, |_| 1 << 40), vec![0..1]);
    }
}
