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

//! Moving this machine to the model its workers channel wants. With a
//! `[workers]` section, every [`CHECK_EVERY`] the running server's
//! `/v1/workers` `demand` is read, and [`decide`] says whether to swap to
//! another profile's model: one requests asked the channel for while no
//! free node served it, when this machine is free and not the last node
//! serving its own model. The swap is the one a request makes.

use crate::process::Coordinator;
use serde_json::Value;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// How often the channel's demand is read.
pub const CHECK_EVERY: Duration = Duration::from_secs(10);

/// How long after a move this coordinator does not move again.
pub const HOLD: Duration = Duration::from_secs(600);

/// The longest random wait before a move, after which the demand is read
/// again: two coordinators that see the same demand do not both move.
pub const JITTER: Duration = Duration::from_secs(15);

/// The server's load under which this machine counts as free.
pub const FREE_LOAD: f64 = 0.3;

/// How long reading `/v1/workers` may take.
const READ_TIMEOUT: Duration = Duration::from_secs(5);

/// One model of the channel's demand, as `/v1/workers` reports it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Row {
    pub model: String,
    /// Requests that wanted it with no free node serving it, channel-wide.
    pub wanted: u64,
    /// The nodes serving it, and how many of them are free.
    pub nodes: usize,
    pub free: usize,
}

/// What the running server says about itself and its channel.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChildView {
    pub load: f64,
    /// It works for a parent node.
    pub working_for_parent: bool,
    /// The workers it leads, `0` when it leads none.
    pub leading_workers: usize,
    /// Its own model's row.
    pub own: Row,
    /// Every model wanted on the channel.
    pub models: Vec<Row>,
}

/// One profile, as the mover weighs it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Profile {
    pub name: String,
    pub model: String,
    /// Paused after repeated crashes ([`crate::crashes`]).
    pub paused: bool,
}

/// What [`decide`] is given.
pub struct Inputs<'a> {
    pub child: &'a ChildView,
    /// Requests still being answered through this coordinator.
    pub in_flight: usize,
    /// A swap or start is under way.
    pub swapping: bool,
    /// Since this coordinator last moved; `None` when it never has.
    pub since_last_move: Option<Duration>,
    /// The profile the running server was started for.
    pub active_profile: &'a str,
    pub profiles: &'a [Profile],
}

/// What [`decide`] says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Decision {
    Stay(String),
    Move {
        profile: String,
        model: String,
        reason: String,
    },
}

/// Whether two model names are one model: equal ignoring case, or equal
/// once a `:QUANT` tag only one of them carries is left off.
pub fn same_model_name(a: &str, b: &str) -> bool {
    let (a, b) = (a.trim(), b.trim());
    if a.is_empty() || b.is_empty() {
        return false;
    }
    if a.eq_ignore_ascii_case(b) {
        return true;
    }
    let tagless = |s: &str| -> Option<String> {
        let (head, tag) = s.rsplit_once(':')?;
        (!tag.is_empty() && !tag.contains('/')).then(|| head.to_string())
    };
    match (tagless(a), tagless(b)) {
        (Some(head), None) => head.eq_ignore_ascii_case(b),
        (None, Some(head)) => head.eq_ignore_ascii_case(a),
        _ => false,
    }
}

/// Whether to move this machine to another profile's model, and why. It
/// moves only to a model some profile here serves, that the channel wants
/// with no node free to serve it, onto a profile not paused; and only when
/// nothing is in flight or swapping here, the server is free (no parent,
/// no workers of its own, load under [`FREE_LOAD`]), its own model has no
/// unserved demand and another node serves it too, and [`HOLD`] has passed
/// since the last move.
pub fn decide(inputs: &Inputs) -> Decision {
    let child = inputs.child;
    let mut wanted: Vec<&Row> = child
        .models
        .iter()
        .filter(|row| {
            row.wanted > 0 && row.free == 0 && !same_model_name(&row.model, &child.own.model)
        })
        .collect();
    wanted.sort_by(|a, b| b.wanted.cmp(&a.wanted).then_with(|| a.model.cmp(&b.model)));
    let mut target = None;
    let mut paused = None;
    for row in wanted {
        let serving: Vec<&Profile> = inputs
            .profiles
            .iter()
            .filter(|p| p.name != inputs.active_profile && same_model_name(&p.model, &row.model))
            .collect();
        match serving.iter().find(|p| !p.paused) {
            Some(profile) => {
                target = Some((row, *profile));
                break;
            }
            None => {
                if let Some(profile) = serving.first() {
                    paused.get_or_insert((row, *profile));
                }
            }
        }
    }
    let Some((row, profile)) = target else {
        return Decision::Stay(match paused {
            Some((row, profile)) => format!(
                "'{}' is wanted, but profile '{}' is paused after repeated crashes",
                row.model, profile.name
            ),
            None => "no model this coordinator serves is wanted without a free node".to_string(),
        });
    };
    let stay = |why: String| Decision::Stay(format!("'{}' is wanted, but {why}", row.model));
    if inputs.in_flight > 0 {
        return stay(format!(
            "{} request(s) are in flight here",
            inputs.in_flight
        ));
    }
    if inputs.swapping {
        return stay("a swap is under way here".to_string());
    }
    if child.working_for_parent {
        return stay("this node works for a parent".to_string());
    }
    if child.leading_workers > 0 {
        return stay(format!(
            "this node leads {} worker(s)",
            child.leading_workers
        ));
    }
    if child.load >= FREE_LOAD {
        return stay(format!(
            "this node's load is {:.2}, not under {FREE_LOAD}",
            child.load
        ));
    }
    if child.own.wanted > 0 {
        return stay(format!(
            "'{}' served here has unserved demand itself ({})",
            child.own.model, child.own.wanted
        ));
    }
    if child.own.nodes <= 1 {
        return stay(format!(
            "this is the last node serving '{}'",
            child.own.model
        ));
    }
    if let Some(since) = inputs.since_last_move
        && since < HOLD
    {
        return stay(format!(
            "this coordinator moved {}s ago and holds for {}s",
            since.as_secs(),
            HOLD.as_secs()
        ));
    }
    Decision::Move {
        profile: profile.name.clone(),
        model: row.model.clone(),
        reason: format!(
            "'{}' was wanted {} time(s) with no free node serving it, and '{}' is served by {} \
             other node(s)",
            row.model,
            row.wanted,
            child.own.model,
            child.own.nodes - 1
        ),
    }
}

/// A row of `/v1/workers`' `demand`.
fn row_of(v: &Value) -> Option<Row> {
    Some(Row {
        model: v.get("model")?.as_str()?.to_string(),
        wanted: v.get("wanted").and_then(Value::as_u64).unwrap_or(0),
        nodes: v.get("nodes").and_then(Value::as_array).map_or(0, Vec::len),
        free: v.get("free").and_then(Value::as_u64).unwrap_or(0) as usize,
    })
}

/// What a `/v1/workers` answer says; `None` from a server with no channel,
/// or one too old to report its demand.
pub fn child_view(v: &Value) -> Option<ChildView> {
    let demand = v.get("demand")?;
    v.get("channel")?.as_str()?;
    Some(ChildView {
        load: v.get("load").and_then(Value::as_f64).unwrap_or(1.0),
        working_for_parent: v.get("role").and_then(Value::as_str) == Some("worker"),
        leading_workers: if v.get("leading").and_then(Value::as_bool) == Some(true) {
            v.get("workers")
                .and_then(Value::as_array)
                .map_or(0, Vec::len)
        } else {
            0
        },
        own: row_of(demand.get("own")?)?,
        models: demand
            .get("models")
            .and_then(Value::as_array)
            .map(|rows| rows.iter().filter_map(row_of).collect())
            .unwrap_or_default(),
    })
}

/// What the mover last decided and did, for `GET /v1/coordinator`.
#[derive(Debug, Default)]
pub struct MoverState {
    last_move: Option<(Instant, String, String, String)>,
    last_decision: Option<(Instant, Decision)>,
}

impl MoverState {
    /// The `workers` object of `GET /v1/coordinator`.
    pub fn json(&self) -> Value {
        let decision = self.last_decision.as_ref().map(|(at, d)| match d {
            Decision::Stay(reason) => serde_json::json!({
                "seconds_ago": at.elapsed().as_secs(),
                "action": "stay",
                "reason": reason,
            }),
            Decision::Move {
                profile,
                model,
                reason,
            } => serde_json::json!({
                "seconds_ago": at.elapsed().as_secs(),
                "action": "move",
                "profile": profile,
                "model": model,
                "reason": reason,
            }),
        });
        let last_move = self.last_move.as_ref().map(|(at, profile, model, reason)| {
            serde_json::json!({
                "seconds_ago": at.elapsed().as_secs(),
                "profile": profile,
                "model": model,
                "reason": reason,
            })
        });
        serde_json::json!({
            "last_decision": decision,
            "last_move": last_move,
        })
    }
}

/// Reads the running server's demand and decides.
async fn consider(coordinator: &Coordinator) -> Decision {
    let swapping = coordinator.swapping();
    let Some((active_profile, origin)) = coordinator.current_server() else {
        return Decision::Stay(if swapping {
            "a swap is under way here".to_string()
        } else {
            "no server is running here".to_string()
        });
    };
    let answer = match coordinator
        .http_client()
        .get(format!("{origin}/v1/workers"))
        .timeout(READ_TIMEOUT)
        .send()
        .await
    {
        Ok(response) => response.json::<Value>().await.ok(),
        Err(_) => None,
    };
    let Some(answer) = answer else {
        return Decision::Stay(format!("'{active_profile}' did not answer /v1/workers"));
    };
    let Some(child) = child_view(&answer) else {
        return Decision::Stay(format!(
            "'{active_profile}' reports no channel demand in /v1/workers"
        ));
    };
    let since_last_move = coordinator
        .mover
        .lock()
        .unwrap()
        .last_move
        .as_ref()
        .map(|(at, ..)| at.elapsed());
    let profiles = coordinator.profiles();
    decide(&Inputs {
        child: &child,
        in_flight: coordinator.in_flight_total(),
        swapping,
        since_last_move,
        active_profile: &active_profile,
        profiles: &profiles,
    })
}

/// Records `decision` as the last one.
fn note(coordinator: &Coordinator, decision: &Decision) {
    coordinator.mover.lock().unwrap().last_decision = Some((Instant::now(), decision.clone()));
}

/// Every [`CHECK_EVERY`], while the configuration has a `[workers]`
/// section, weighs moving to the model the channel wants, and moves when
/// [`decide`] says so twice: now, and after a random wait of up to
/// [`JITTER`].
pub async fn run(coordinator: Arc<Coordinator>) {
    loop {
        tokio::time::sleep(CHECK_EVERY).await;
        if !coordinator.workers_enabled() {
            continue;
        }
        let decision = consider(&coordinator).await;
        let Decision::Move { profile, .. } = &decision else {
            note(&coordinator, &decision);
            continue;
        };
        let wait = JITTER.mul_f64(rand::random::<f64>());
        log::debug!(
            "workers: weighing a move to '{profile}'; reading the channel again in {:.1}s",
            wait.as_secs_f64()
        );
        tokio::time::sleep(wait).await;
        let again = consider(&coordinator).await;
        note(&coordinator, &again);
        let Decision::Move {
            profile,
            model,
            reason,
        } = again
        else {
            continue;
        };
        let Some(entry) = coordinator.profile(&profile) else {
            continue;
        };
        let from = coordinator
            .current_server()
            .map(|(name, _)| name)
            .unwrap_or_default();
        log::info!("workers: moving from '{from}' to '{profile}' ({model}): {reason}");
        coordinator.mover.lock().unwrap().last_move =
            Some((Instant::now(), profile.clone(), model, reason));
        if let Err(err) = coordinator.ensure_active(&entry).await {
            log::warn!("warning: workers: moving to '{profile}' failed: {err:#}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(model: &str, wanted: u64, nodes: usize, free: usize) -> Row {
        Row {
            model: model.to_string(),
            wanted,
            nodes,
            free,
        }
    }

    fn profile(name: &str, model: &str) -> Profile {
        Profile {
            name: name.to_string(),
            model: model.to_string(),
            paused: false,
        }
    }

    /// A free node whose model another node serves too, on a channel that
    /// wants `org/b` with no node free for it.
    fn free_child() -> ChildView {
        ChildView {
            load: 0.05,
            working_for_parent: false,
            leading_workers: 0,
            own: row("org/a:Q4_K_M", 0, 2, 2),
            models: vec![row("org/b:Q8_0", 4, 0, 0)],
        }
    }

    fn profiles() -> Vec<Profile> {
        vec![profile("a", "org/a:Q4_K_M"), profile("b", "org/b:Q8_0")]
    }

    fn inputs<'a>(child: &'a ChildView, profiles: &'a [Profile]) -> Inputs<'a> {
        Inputs {
            child,
            in_flight: 0,
            swapping: false,
            since_last_move: None,
            active_profile: "a",
            profiles,
        }
    }

    fn stays(decision: Decision, why: &str) {
        match decision {
            Decision::Stay(reason) => assert!(reason.contains(why), "{reason}"),
            other => panic!("moved: {other:?}"),
        }
    }

    #[test]
    fn a_free_node_moves_to_the_wanted_model_a_profile_serves() {
        let (child, profiles) = (free_child(), profiles());
        match decide(&inputs(&child, &profiles)) {
            Decision::Move { profile, model, .. } => {
                assert_eq!((profile.as_str(), model.as_str()), ("b", "org/b:Q8_0"));
            }
            other => panic!("stayed: {other:?}"),
        }
    }

    #[test]
    fn only_a_model_no_free_node_serves_is_moved_to() {
        let profiles = profiles();
        let mut child = free_child();
        child.models = vec![row("org/b:Q8_0", 4, 1, 1)];
        stays(decide(&inputs(&child, &profiles)), "no model");
        child.models = vec![row("org/b:Q8_0", 0, 0, 0)];
        stays(decide(&inputs(&child, &profiles)), "no model");
        // A name only the request used, with no tag, matches the profile.
        child.models = vec![row("ORG/B", 2, 0, 0)];
        assert!(matches!(
            decide(&inputs(&child, &profiles)),
            Decision::Move { .. }
        ));
        // A model no profile here serves.
        child.models = vec![row("gpt-4", 9, 0, 0)];
        stays(decide(&inputs(&child, &profiles)), "no model");
    }

    #[test]
    fn a_busy_node_stays() {
        let profiles = profiles();
        let child = free_child();
        let mut busy = inputs(&child, &profiles);
        busy.in_flight = 1;
        stays(decide(&busy), "in flight");
        let mut busy = inputs(&child, &profiles);
        busy.swapping = true;
        stays(decide(&busy), "swap");
        let mut child = free_child();
        child.load = FREE_LOAD;
        stays(decide(&inputs(&child, &profiles)), "load");
        let mut child = free_child();
        child.working_for_parent = true;
        stays(decide(&inputs(&child, &profiles)), "parent");
        let mut child = free_child();
        child.leading_workers = 2;
        stays(decide(&inputs(&child, &profiles)), "leads 2");
    }

    #[test]
    fn the_last_node_serving_a_model_stays() {
        let profiles = profiles();
        let mut child = free_child();
        child.own = row("org/a:Q4_K_M", 0, 1, 1);
        stays(decide(&inputs(&child, &profiles)), "last node");
        // Its own model wanted unserved: never moves, however many serve it.
        child.own = row("org/a:Q4_K_M", 3, 3, 0);
        stays(decide(&inputs(&child, &profiles)), "unserved demand itself");
    }

    #[test]
    fn a_move_holds_for_ten_minutes() {
        let (child, profiles) = (free_child(), profiles());
        let mut held = inputs(&child, &profiles);
        held.since_last_move = Some(HOLD - Duration::from_secs(1));
        stays(decide(&held), "holds");
        held.since_last_move = Some(HOLD);
        assert!(matches!(decide(&held), Decision::Move { .. }));
    }

    #[test]
    fn a_paused_profile_is_never_moved_to() {
        let child = free_child();
        let mut profiles = profiles();
        profiles[1].paused = true;
        stays(decide(&inputs(&child, &profiles)), "paused");
        // Another profile of the same model that is not paused is.
        profiles.push(profile("b2", "org/b:Q8_0"));
        match decide(&inputs(&child, &profiles)) {
            Decision::Move { profile, .. } => assert_eq!(profile, "b2"),
            other => panic!("stayed: {other:?}"),
        }
    }

    #[test]
    fn the_most_wanted_model_is_moved_to_first() {
        let mut profiles = profiles();
        profiles.push(profile("c", "org/c"));
        let mut child = free_child();
        child.models = vec![row("org/b:Q8_0", 2, 0, 0), row("org/c", 5, 0, 0)];
        match decide(&inputs(&child, &profiles)) {
            Decision::Move { profile, .. } => assert_eq!(profile, "c"),
            other => panic!("stayed: {other:?}"),
        }
    }

    #[test]
    fn model_names_match_with_or_without_a_tag() {
        assert!(same_model_name("org/a:Q4_K_M", "ORG/A:q4_k_m"));
        assert!(same_model_name("org/a:Q4_K_M", "org/a"));
        assert!(same_model_name("org/a", "org/a:Q4_K_M"));
        assert!(!same_model_name("org/a:Q4_K_M", "org/a:Q8_0"));
        assert!(!same_model_name("org/a", "org/b"));
        assert!(!same_model_name("", ""));
    }

    #[test]
    fn a_workers_answer_is_read() {
        let answer = serde_json::json!({
            "channel": "lab",
            "load": 0.1,
            "role": "top",
            "leading": true,
            "workers": [{}, {}],
            "demand": {
                "window_seconds": 300,
                "own": {"model": "org/a", "wanted": 0, "here": 0, "nodes": ["x", "y"], "free": 2},
                "models": [{"model": "org/b", "wanted": 3, "here": 1, "nodes": [], "free": 0}],
            },
        });
        let view = child_view(&answer).unwrap();
        assert_eq!(view.leading_workers, 2);
        assert!(!view.working_for_parent);
        assert_eq!(view.own, row("org/a", 0, 2, 2));
        assert_eq!(view.models, vec![row("org/b", 3, 0, 0)]);
        // A server without a channel, or too old to report demand.
        assert!(child_view(&serde_json::json!({"role": "none"})).is_none());
        assert!(child_view(&serde_json::json!({"channel": null, "demand": {}})).is_none());
    }
}
