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

//! Whether a request runs on one node that holds the whole model, or is
//! divided over a tree of the channel's free nodes.
//!
//! A tree runs every layer once a token, one node after another, and every
//! hand-over adds a round trip, so it decodes faster than a node alone only
//! when its fast nodes cannot hold the model by themselves. [`decide`] weighs
//! the two by what was measured: each node's decode speed and fixed cost
//! (`super::speed`), as its probe answer carries them
//! ([`super::protocol::Pace`]), the links' round trips and bandwidth, and what
//! earlier requests found. A request is divided only when that is predicted
//! clearly faster, or when no node holds the model alone.

use super::plan;
use crate::config::LocalLayers;
use std::ops::Range;
use std::time::Duration;

/// How much faster a tree must be predicted than the best node alone, or a
/// tree that leads the model than a node alone, for a request to go there:
/// its time at most `1 − MARGIN` of the other's.
pub const MARGIN: f64 = 0.1;

/// The least share of a node's speed its load leaves it, as shares count it.
const MIN_SPARE: f64 = 0.05;

/// `ORANGU_WORKERS_DIVIDE`: whether a node decides from what it measured.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    /// By [`decide`] (unset, `auto`).
    #[default]
    Measured,
    /// Every model is divided over the free nodes, whatever its size
    /// (`always`): for testing a tree on nodes that each hold the model.
    Always,
}

impl Mode {
    pub fn parse(value: Option<&str>) -> Self {
        match value.map(str::trim) {
            Some(v) if v.eq_ignore_ascii_case("always") => Self::Always,
            _ => Self::Measured,
        }
    }

    /// The mode the environment sets.
    pub fn from_env() -> Self {
        Self::parse(std::env::var("ORANGU_WORKERS_DIVIDE").ok().as_deref())
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Measured => "measured",
            Self::Always => "always",
        }
    }
}

/// Milliseconds a decode step of `bytes` of layers takes at `rate` (GB a
/// second) with `fixed_ms` on top; `None` when the rate is not known.
pub fn step_ms(bytes: u64, rate: f64, fixed_ms: f64) -> Option<f64> {
    (rate > 0.0 && rate.is_finite()).then(|| bytes as f64 / (rate * 1e9) * 1e3 + fixed_ms)
}

/// A request's time a token on a node, weighed by how busy the node is:
/// `ms` over the share of it its load leaves, as shares count it.
pub fn score(ms: f64, load: f32) -> f64 {
    ms / (1.0 - f64::from(load)).max(MIN_SPARE)
}

/// A node of a tree as [`tree_ms`] prices it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Member {
    pub name: String,
    /// Gigabytes of its layers a second through a decode step, its load
    /// taken into account; `0` when not known.
    pub rate: f64,
    /// What a step costs it whatever its layers.
    pub fixed_ms: f64,
    /// The bytes of weights it can hold.
    pub budget: u64,
    /// The link from the top-level node to it: its round trip, and its
    /// bandwidth each way in bytes a second (`0` when not timed). `None`
    /// for the top-level node itself, and for a node whose rate was
    /// observed through the tree, which counts its link already.
    pub hop: Option<(Duration, f64)>,
}

/// What [`tree_ms`] needs besides the nodes: the model.
pub struct Model<'a> {
    pub layer_bytes: &'a [u64],
    /// The output head's bytes, read on the top-level node once a token.
    pub head_bytes: u64,
    /// Bytes of the hidden state that cross each link, each way, a token.
    pub hidden_bytes: f64,
    pub local_layers: LocalLayers,
    /// Where the model may be cut (`ModelForward::split_allowed`).
    pub allowed: &'a dyn Fn(usize) -> bool,
}

/// The predicted time of a decode step through a tree of `members`, the
/// top-level node first, in milliseconds: the layers cut as a plan cuts
/// them — by decode speed up to each node's budget, in the order that runs
/// soonest — then each node's layers at its rate and its fixed cost, every
/// worker's round trip and its hidden state each way, and the head on the
/// top-level node. `None` for fewer than two nodes, or when a node's rate or
/// budget is not known.
pub fn tree_ms(members: &[Member], model: &Model) -> Option<f64> {
    if members.len() < 2 || members.iter().any(|m| m.rate <= 0.0 || m.budget == 0) {
        return None;
    }
    let n = model.layer_bytes.len();
    let total: u64 = model.layer_bytes.iter().sum();
    let budgets: Vec<u64> = members.iter().map(|m| m.budget).collect();
    let rates: Vec<f64> = members.iter().map(|m| m.rate).collect();
    // Shares by speed; by memory when the budgets together cannot hold the
    // model, as a plan cuts it then.
    let weights = plan::by_speed(&budgets, &rates, total).unwrap_or(budgets);
    let top = &members[0];
    let head = step_ms(model.head_bytes, top.rate, 0.0)?;
    let cost = |order: &[usize]| -> f64 {
        let workers: Vec<u64> = order.iter().map(|i| weights[i + 1].max(1)).collect();
        let (local, shares) = plan::plan_shares(
            0..n,
            model.layer_bytes,
            model.local_layers,
            weights[0].max(1),
            &workers,
            model.allowed,
        );
        let stage = |member: &Member, layers: &Range<usize>| -> f64 {
            if layers.is_empty() {
                return 0.0;
            }
            let bytes = plan::bytes_of(layers, model.layer_bytes);
            let compute = step_ms(bytes, member.rate, member.fixed_ms).unwrap_or(f64::INFINITY);
            let hop = member.hop.map_or(0.0, |(rtt, bandwidth)| {
                let moving = if bandwidth > 0.0 {
                    2.0 * model.hidden_bytes / bandwidth * 1e3
                } else {
                    0.0
                };
                rtt.as_secs_f64() * 1e3 + moving
            });
            compute + hop
        };
        head + stage(top, &local)
            + shares
                .iter()
                .zip(order)
                .map(|(share, i)| stage(&members[i + 1], share))
                .sum::<f64>()
    };
    let order = plan::best_order(&rates[1..], &cost);
    Some(cost(&order))
}

/// A node that may serve a request: this one, another that holds the model
/// alone, or one that leads a tree of it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Server {
    /// Its API, `None` for this node.
    pub api: Option<String>,
    pub name: String,
    /// Its time a token for a request, when known.
    pub ms: Option<f64>,
    pub load: f32,
}

impl Server {
    fn score(&self) -> Option<f64> {
        self.ms.map(|ms| score(ms, self.load))
    }
}

/// Where a request goes.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// Served here, by this node alone.
    Here,
    /// Sent to the node whose API this is.
    To(String),
    /// Divided over the free nodes, this node leading.
    Divide,
}

/// A decision, with what it weighed.
#[derive(Clone, Debug, PartialEq)]
pub struct Decision {
    pub verdict: Verdict,
    /// The node it goes to by name, empty for this node.
    pub name: String,
    pub reason: String,
    /// The best node alone's time a token, weighed by its load, and the
    /// tree's, when they were priced.
    pub alone_ms: Option<f64>,
    pub tree_ms: Option<f64>,
}

impl Decision {
    fn new(verdict: Verdict, name: &str, reason: String) -> Self {
        Self {
            verdict,
            name: name.to_string(),
            reason,
            alone_ms: None,
            tree_ms: None,
        }
    }
}

/// Where a request goes.
///
/// - No node holds the model alone (`here` is `None` and `holders` is
///   empty): to the least busy node that leads a tree of it, else divided
///   over the free nodes when `can_divide`, else here.
/// - Otherwise, to the node that holds it alone and is fastest by its time a
///   token over what its load leaves free; this node keeps the request
///   unless another one's comes out at most `1 − route_margin` of its own.
///   When a node that holds it has no measured time, by load alone: this
///   node keeps it unless another is `route_margin` less busy, and a node
///   that does not hold it sends it to the least busy one that does.
/// - With every holder's time known, a tree that leads the model, or the
///   tree this node would divide it over (`tree`, priced at the nodes'
///   loads), takes the request instead only when its time comes out at most
///   `1 −` [`MARGIN`] of the best node alone. A tree that could not be priced
///   never does.
pub fn decide(
    here: Option<&Server>,
    holders: &[Server],
    leaders: &[Server],
    can_divide: bool,
    tree: Option<f64>,
    route_margin: f32,
) -> Decision {
    let least_busy = |servers: &[Server]| -> Option<Server> {
        servers
            .iter()
            .min_by(|a, b| a.load.total_cmp(&b.load))
            .cloned()
    };
    let to = |server: &Server, reason: String| -> Decision {
        match &server.api {
            None => Decision::new(Verdict::Here, "", reason),
            Some(api) => Decision::new(Verdict::To(api.clone()), &server.name, reason),
        }
    };
    if here.is_none() && holders.is_empty() {
        if let Some(leader) = least_busy(leaders) {
            return to(
                &leader,
                "no node holds the model alone; a tree of it is led there".to_string(),
            );
        }
        if can_divide {
            let mut divided = Decision::new(
                Verdict::Divide,
                "",
                "no node holds the model alone".to_string(),
            );
            divided.tree_ms = tree;
            return divided;
        }
        return Decision::new(
            Verdict::Here,
            "",
            "no node holds the model alone, and none is free".to_string(),
        );
    }
    let alone: Vec<&Server> = here.into_iter().chain(holders).collect();
    let measured = alone.iter().all(|s| s.ms.is_some());
    if !measured {
        // By load, as a node that cannot tell their speeds.
        let best = match here {
            Some(own) => holders
                .iter()
                .filter(|h| h.load + route_margin < own.load)
                .min_by(|a, b| a.load.total_cmp(&b.load))
                .unwrap_or(own),
            None => holders
                .iter()
                .min_by(|a, b| a.load.total_cmp(&b.load))
                .expect("a holder"),
        };
        let unmeasured: Vec<&str> = alone
            .iter()
            .filter(|s| s.ms.is_none())
            .map(|s| s.name.as_str())
            .collect();
        return to(
            best,
            format!(
                "{} measured no decode speed: served alone, by load",
                unmeasured.join(", ")
            ),
        );
    }
    let fastest = alone
        .iter()
        .min_by(|a, b| a.score().unwrap().total_cmp(&b.score().unwrap()))
        .copied()
        .expect("a holder");
    let best = match here {
        Some(own)
            if fastest.api.is_some()
                && fastest.score().unwrap()
                    > own.score().unwrap() * (1.0 - f64::from(route_margin)) =>
        {
            own
        }
        _ => fastest,
    };
    let best_ms = best.score().unwrap();
    let with = |mut decision: Decision| {
        decision.alone_ms = Some(best_ms);
        decision.tree_ms = tree;
        decision
    };
    let who = |s: &Server| {
        if s.api.is_none() {
            "this node".to_string()
        } else {
            s.name.clone()
        }
    };
    if let Some(leader) = leaders
        .iter()
        .filter(|l| l.ms.is_some())
        .min_by(|a, b| a.score().unwrap().total_cmp(&b.score().unwrap()))
        && leader.score().unwrap() <= best_ms * (1.0 - MARGIN)
    {
        let reason = format!(
            "the tree {} leads: {:.1} ms a token, {:.1} ms on {} alone",
            leader.name,
            leader.score().unwrap(),
            best_ms,
            who(best)
        );
        return with(to(leader, reason));
    }
    if can_divide
        && let Some(tree_ms) = tree
        && tree_ms <= best_ms * (1.0 - MARGIN)
    {
        let reason = format!(
            "predicted {tree_ms:.1} ms a token divided, {best_ms:.1} ms on {} alone",
            who(best)
        );
        return with(Decision::new(Verdict::Divide, "", reason));
    }
    let reason = match tree {
        Some(tree_ms) if can_divide => format!(
            "predicted {best_ms:.1} ms a token on {} alone, {tree_ms:.1} ms divided",
            who(best)
        ),
        None if can_divide => format!(
            "predicted {best_ms:.1} ms a token on {} alone; a tree could not be priced",
            who(best)
        ),
        _ => format!("predicted {best_ms:.1} ms a token on {} alone", who(best)),
    };
    with(to(best, reason))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn server(api: Option<&str>, name: &str, ms: Option<f64>, load: f32) -> Server {
        Server {
            api: api.map(str::to_string),
            name: name.to_string(),
            ms,
            load,
        }
    }

    /// Where a request goes, with a route margin of 0.1.
    fn to(
        here: Option<&Server>,
        holders: &[Server],
        leaders: &[Server],
        can_divide: bool,
        tree: Option<f64>,
    ) -> Verdict {
        decide(here, holders, leaders, can_divide, tree, 0.1).verdict
    }

    fn any(_: usize) -> bool {
        true
    }

    fn member(name: &str, rate: f64, fixed_ms: f64, budget: u64, rtt_ms: Option<u64>) -> Member {
        Member {
            name: name.to_string(),
            rate,
            fixed_ms,
            budget,
            hop: rtt_ms.map(|ms| (Duration::from_millis(ms), 100e6)),
        }
    }

    const GB: u64 = 1_000_000_000;

    /// Thirty layers of 0.1 GB: three gigabytes.
    fn model<'a>(layers: &'a [u64], allowed: &'a dyn Fn(usize) -> bool) -> Model<'a> {
        Model {
            layer_bytes: layers,
            head_bytes: 0,
            hidden_bytes: 4096.0,
            local_layers: LocalLayers::Auto,
            allowed,
        }
    }

    #[test]
    fn a_step_follows_its_bytes_and_a_busy_node_counts_slower() {
        assert_eq!(step_ms(GB, 10.0, 2.0), Some(102.0));
        assert_eq!(step_ms(GB, 0.0, 2.0), None);
        assert_eq!(score(10.0, 0.0), 10.0);
        assert_eq!(score(10.0, 0.5), 20.0);
        assert_eq!(score(10.0, 1.0), 200.0);
        assert_eq!(Mode::parse(None), Mode::Measured);
        assert_eq!(Mode::parse(Some(" Always ")), Mode::Always);
        assert_eq!(Mode::parse(Some("0")), Mode::Measured);
    }

    /// A laptop's GPU and three boards, each holding a third of a
    /// three-gigabyte model at a tenth of the laptop's speed: the tree's step
    /// is every node's part and every hop, slower than the laptop alone and
    /// faster than a board alone.
    #[test]
    fn a_tree_costs_every_part_and_every_hop() {
        let layers = vec![GB / 10; 30];
        let model = model(&layers, &any);
        let laptop = member("laptop", 100.0, 1.0, 8 * GB, None);
        let boards: Vec<Member> = (0..3)
            .map(|i| member(&format!("board{i}"), 10.0, 2.0, 8 * GB, Some(1)))
            .collect();
        let mut members = vec![laptop.clone()];
        members.extend(boards.iter().cloned());
        let tree = tree_ms(&members, &model).unwrap();
        let alone = step_ms(3 * GB, 100.0, 1.0).unwrap();
        assert!(tree > alone, "{tree} vs {alone}");
        let board_alone = step_ms(3 * GB, 10.0, 2.0).unwrap();
        assert!(tree < board_alone, "{tree} vs {board_alone}");
        // Hops and fixed costs alone, on top of the layers at the shares'
        // speed: every node over its share takes about as long.
        let layers_only = 3.0 * 1e3 / 130.0 * 4.0;
        assert!(tree >= layers_only - 1.0, "{tree} vs {layers_only}");
        // Unknown rates or budgets, or no worker: no price.
        assert_eq!(tree_ms(&members[..1], &model), None);
        let mut unknown = members.clone();
        unknown[2].rate = 0.0;
        assert_eq!(tree_ms(&unknown, &model), None);
        unknown[2].rate = 10.0;
        unknown[3].budget = 0;
        assert_eq!(tree_ms(&unknown, &model), None);
    }

    /// A slow board that holds the model and a fast GPU that holds only half
    /// of it: divided, the GPU takes what it holds and the tree beats the
    /// board alone.
    #[test]
    fn a_fast_node_that_cannot_hold_the_model_makes_a_tree_pay() {
        let layers = vec![GB / 10; 30];
        let model = model(&layers, &any);
        let board = member("board", 5.0, 2.0, 8 * GB, None);
        let gpu = member("gpu", 100.0, 1.0, 3 * GB / 2, Some(1));
        let tree = tree_ms(&[board, gpu], &model).unwrap();
        let alone = step_ms(3 * GB, 5.0, 2.0).unwrap();
        assert!(tree < alone * (1.0 - MARGIN), "{tree} vs {alone}");
    }

    #[test]
    fn a_fast_node_that_holds_the_model_serves_it_alone() {
        let here = server(None, "laptop", Some(13.0), 0.0);
        let decision = decide(Some(&here), &[], &[], true, Some(94.0), 0.1);
        assert_eq!(decision.verdict, Verdict::Here, "{decision:?}");
        assert_eq!(
            (decision.alone_ms, decision.tree_ms),
            (Some(13.0), Some(94.0))
        );
        assert!(decision.reason.contains("13.0 ms"), "{}", decision.reason);
    }

    #[test]
    fn a_faster_tree_divides() {
        let here = server(None, "board", Some(600.0), 0.0);
        let decision = decide(Some(&here), &[], &[], true, Some(120.0), 0.1);
        assert_eq!(decision.verdict, Verdict::Divide, "{decision:?}");
        // Faster, but not by the margin: alone.
        let decision = decide(Some(&here), &[], &[], true, Some(560.0), 0.1);
        assert_eq!(decision.verdict, Verdict::Here, "{decision:?}");
        // Nothing free to divide over.
        let decision = decide(Some(&here), &[], &[], false, Some(120.0), 0.1);
        assert_eq!(decision.verdict, Verdict::Here, "{decision:?}");
    }

    #[test]
    fn a_model_no_node_holds_is_divided() {
        let decision = decide(None, &[], &[], true, None, 0.1);
        assert_eq!(decision.verdict, Verdict::Divide);
        // A tree of it already led elsewhere takes the request.
        let leader = server(Some("http://l"), "l", None, 0.5);
        let decision = decide(None, &[], std::slice::from_ref(&leader), true, None, 0.1);
        assert_eq!(decision.verdict, Verdict::To("http://l".to_string()));
        assert_eq!(to(None, &[], &[], false, None), Verdict::Here);
    }

    /// A node with no measured time anywhere — its own, or a holder's — is
    /// served alone, by load, and never divided.
    #[test]
    fn an_unknown_speed_serves_alone_by_load() {
        let here = server(None, "a", None, 0.2);
        let idle = server(Some("http://b"), "b", Some(10.0), 0.0);
        assert_eq!(to(Some(&here), &[], &[], true, Some(1.0)), Verdict::Here);
        assert_eq!(
            to(
                Some(&here),
                std::slice::from_ref(&idle),
                &[],
                true,
                Some(1.0)
            ),
            Verdict::To("http://b".to_string())
        );
        let close = server(Some("http://b"), "b", Some(10.0), 0.15);
        assert_eq!(to(Some(&here), &[close], &[], true, None), Verdict::Here);
        // Not holding it, to the least busy that does.
        let busy = server(Some("http://c"), "c", None, 0.9);
        assert_eq!(
            to(None, &[busy, idle], &[], true, Some(1.0)),
            Verdict::To("http://b".to_string())
        );
    }

    /// A slow board with a request sends it to an idle laptop that is
    /// faster, and keeps it only when the laptop is so busy that it would be
    /// slower.
    #[test]
    fn a_request_goes_to_the_fastest_node_by_its_load() {
        let board = server(None, "board", Some(600.0), 0.0);
        let laptop = server(Some("http://laptop"), "laptop", Some(13.0), 0.0);
        let decision = decide(
            Some(&board),
            std::slice::from_ref(&laptop),
            &[],
            true,
            Some(94.0),
            0.1,
        );
        assert_eq!(decision.verdict, Verdict::To("http://laptop".to_string()));
        assert_eq!(decision.name, "laptop");
        // At full load the laptop's 13 ms count as 260: still faster than
        // 600, not than 200.
        let busy = server(Some("http://laptop"), "laptop", Some(13.0), 1.0);
        assert_eq!(
            to(Some(&board), &[busy], &[], false, None),
            Verdict::To("http://laptop".to_string())
        );
        let board = server(None, "board", Some(200.0), 0.0);
        let busy = server(Some("http://laptop"), "laptop", Some(13.0), 1.0);
        assert_eq!(to(Some(&board), &[busy], &[], false, None), Verdict::Here);
        // Equal speeds: kept unless the other is clearly less busy.
        let here = server(None, "a", Some(13.0), 0.05);
        let other = server(Some("http://b"), "b", Some(13.0), 0.0);
        assert_eq!(
            to(Some(&here), std::slice::from_ref(&other), &[], false, None),
            Verdict::Here
        );
        let here = server(None, "a", Some(13.0), 0.5);
        assert_eq!(
            to(Some(&here), &[other], &[], false, None),
            Verdict::To("http://b".to_string())
        );
    }

    /// A tree that leads the model takes a request from a node alone only
    /// when its time is known and clearly better.
    #[test]
    fn a_tree_led_elsewhere_takes_a_request_only_when_faster() {
        let board = server(None, "board", Some(600.0), 0.0);
        let fast = server(Some("http://t"), "t", Some(100.0), 0.0);
        assert_eq!(
            to(Some(&board), &[], std::slice::from_ref(&fast), false, None),
            Verdict::To("http://t".to_string())
        );
        let laptop = server(None, "laptop", Some(13.0), 0.0);
        assert_eq!(to(Some(&laptop), &[], &[fast], false, None), Verdict::Here);
        let unknown = server(Some("http://t"), "t", None, 0.0);
        assert_eq!(
            to(Some(&board), &[], &[unknown], false, None),
            Verdict::Here
        );
    }
}
