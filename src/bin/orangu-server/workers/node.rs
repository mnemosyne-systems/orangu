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

//! One `orangu-server` process's place in a tree of workers.
//!
//! A [`Node`] is both halves at once, and which one it is right now follows
//! from what happens to it:
//!
//! - **Top-level.** With workers of its own and no parent, it dials them,
//!   plans the model's layers over itself and them, and serves its own API
//!   through a [`DelegatingModel`] on that plan.
//! - **Worker.** When a parent connects and assigns it a range of layers,
//!   it checks that it has the parent's model, plans that range over itself
//!   and its own workers the same way, and answers the parent's forwards
//!   from a [`SessionStore`]. Its own API is off meanwhile. When the parent
//!   goes away it becomes top-level again.
//!
//! A node listens for a parent whenever its config has a `[workers]`
//! section, and hands layers down whenever its `workers` list is not empty;
//! a middle node does both.

use super::auth;
use super::divide::Verdict;
use super::guard::{GuardedStage, TakeStandby};
use super::identity;
use super::link::{ChildInfo, ChildLink};
use super::metrics::{LinkSnapshot, NodeMetrics};
use super::pipeline::{DelegatingModel, Handoff, LayerPipeline, PipelineSource, Prefilled, Stage};
use super::plan;
use super::prefixes::{Manifest, PrefixFiles, SavedNode};
use super::protocol::{
    ActivationFormat, Assign, Capacity, DiscoveryRecord, ErrorCode, FEATURE_FETCH,
    FEATURE_PREFIXES, FEATURE_SWITCH, FEATURES, HelloAck, MAX_DEPTH, Message, NodeSetup,
    PROTOCOL_VERSION, PlanEntry, ProbeState, SessionTerms, WorkerError, check_path,
    parent_reads_discovery, read_frame, read_message, write_message,
};
use super::session::SessionStore;
use super::stage::{EncodedStage, MessageStage};
use super::transport::{self, Tls};
use crate::config::{HeadOn, LocalLayers, Shares};
use crate::engine::arch::ModelForward;
use crate::engine::loader::LoadedModel;
use crate::engine::prompt_weights::{self, Decision};
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::io::{BufReader, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, RwLock, Weak};
use std::time::{Duration, Instant};

/// How long a session may sit unused on a worker before it is reclaimed: a
/// parent that went away without releasing it leaves it behind.
const SESSION_IDLE: Duration = Duration::from_secs(600);

/// The sessions a node keeps for its parent's prefix pool, as its own
/// configuration has it: one a slot of its own, up to the pool's size.
fn pool_reserve(own_slots: usize) -> usize {
    own_slots.clamp(1, crate::engine::prefix_cache::ENTRIES)
}

/// How long a worker may take to answer an assignment: its layers may have
/// to be downloaded first (`[workers].download = range`).
const ASSIGN_TIMEOUT: Duration = Duration::from_secs(3600);

/// How long a parent may say nothing. It pings every 5 s while idle and more
/// often with a request out, so this is a dozen pings missed: a machine gone, or
/// a process stopped. Generous, as a parent paging heavily can stall for a while,
/// and letting it go costs a new plan and every session.
pub const PARENT_SILENCE: Duration = Duration::from_secs(60);

/// How long a node goes without a forward before it counts as between
/// requests, and may take a returning worker back. Sessions a parent keeps
/// for a slot's next request live far longer and do not count: a plan made
/// under them costs that request a rebuild.
const QUIET: Duration = Duration::from_secs(2);

/// How much less busy another node must be for a request this node could
/// serve to go there instead.
const ROUTE_MARGIN: f32 = 0.1;

/// The load above which a node is not claimed as a worker.
const CLAIM_CUTOFF: f32 = 0.8;

/// The least share of its speed a busy node is planned at.
const MIN_SPARE: f64 = 0.05;

/// How often the nodes a scan found are asked again how busy they are.
const REFRESH_EVERY: Duration = Duration::from_secs(2);

// How long a free node stays promised to a node about to claim it
// ([`Message::Lease`]): long enough for the claim's handshakes.
use super::channel::LEASE_FOR;

/// Probes in a row a parent seen on the channel may leave unanswered, with
/// its connection quiet for a quarter of [`NodeSettings::parent_silence`],
/// before this node takes it for gone without waiting out the silence.
const PARENT_MISSES: u32 = 3;

/// Decode steps a top-level node observes before it weighs planning again
/// by the speeds they showed (`Node::calibration_due`).
const CALIBRATE_TOKENS: usize = 32;

/// The share of a decode step, or of the slowest stage, that moving layers has
/// to save. A worker given other layers has to build and warm them, a minute on
/// a board with a large model, and every conversation is rebuilt.
const WORTH_MOVING: f64 = 0.05;

pub struct NodeSettings {
    /// How this node is named in plans and error paths: its `[workers]`
    /// address.
    pub name: String,
    /// The address to listen on for a parent; `None` listens nowhere.
    pub listen: Option<String>,
    /// Workers this node holds from the start, without a scan.
    pub workers: Vec<String>,
    /// Spare workers: never planned into a share, one of them takes a lost
    /// worker's layers (`super::guard`).
    pub standby: Vec<String>,
    /// `[workers].name`: the channel this node finds its workers on, and
    /// is found on.
    pub channel: Option<String>,
    /// Where a scan of the channel looks.
    pub scan: Scan,
    /// How often the channel is scanned.
    pub scan_every: Duration,
    /// `[workers].release`: how long a node that claimed workers keeps
    /// them with no request running.
    pub release: Duration,
    /// The port this node's API answers on, and whether with TLS: what a
    /// probe is told, for requests routed here.
    pub api_port: u16,
    pub api_tls: bool,
    pub secret: Option<String>,
    pub activations: ActivationFormat,
    pub timeout: Duration,
    pub connect_timeout: Duration,
    pub local_layers: LocalLayers,
    /// Positions a sequence may hold, as a top-level node.
    pub n_ctx: usize,
    /// Sequences at once, as a top-level node.
    pub slots: usize,
    /// The served model as `list` names it, and its quantization — for the
    /// model check's messages.
    pub label: String,
    pub quant: String,
    /// This node's own capacity; the `subtree_*` fields are filled in.
    pub capacity: Capacity,
    /// `[workers].shares`: what shares are sized by.
    pub shares: Shares,
    /// `[workers].head`: which node applies the output head.
    pub head: HeadOn,
    /// `ORANGU_WORKERS_DIVIDE`: whether a request is divided only when that
    /// is predicted faster than a node alone ([`super::divide`]).
    pub divide: super::divide::Mode,
    /// Whether this node serves its model through its tree at all: not a
    /// picture model, whose text encoder is never split. Without it the
    /// node plans nothing for its workers, and serves only as a worker.
    pub split: bool,
    /// How often idle workers are pinged.
    pub maintenance: Duration,
    /// How long a parent may say nothing before this node takes it for gone
    /// ([`PARENT_SILENCE`]).
    pub parent_silence: Duration,
    /// How often a top-level node missing a worker tries to take it back,
    /// between requests.
    pub readmit: Duration,
    /// TLS for the listener and for dialing workers, when configured.
    pub tls: Option<Tls>,
    /// Where this node's own metrics page lives, as `host` and `port`:
    /// the address it tells its parent about, so Prometheus can find it.
    /// `None` when the metrics listener is off then the node reports no
    /// address for itself.
    pub metrics_endpoint: Option<(String, u16)>,
    /// `prompt_weights`: how a top-level node decides the copies its tree
    /// runs prompts through (`engine::prompt_weights::decide`).
    pub prompt_weights: prompt_weights::PromptWeights,
    /// Where the prefix pool's entries are saved (`~/.orangu/server`), under
    /// the model and this node's name ([`super::prefixes`]); `None` saves
    /// nothing.
    pub prefix_root: Option<std::path::PathBuf>,
    /// The model file this node serves by range to the nodes of its channel
    /// that fetch it in part ([`super::share`]): the file it runs, when that
    /// is a whole model in a single file.
    pub serve_file: Option<std::path::PathBuf>,
}

/// Where a node looks for the other nodes of its channel.
/// Where a node looks for the rest of its channel.
#[derive(Clone, Debug, Default)]
pub struct Scan {
    /// The channel's port: where a node that probed this one, and is not
    /// known yet, is probed back.
    pub port: u16,
    /// Every address of these networks (`[workers].network`), or of this
    /// machine's own IPv4 networks when empty, on `port`; `None` scans no
    /// network (`ORANGU_WORKERS_PEERS`).
    pub networks: Option<Vec<(std::net::Ipv4Addr, u8)>>,
    /// `host:port` entries asked at every turn (`ORANGU_WORKERS_PEERS`).
    pub peers: Vec<String>,
}

/// Held while a request runs on this node ([`request_started`]).
pub struct RequestGuard(Arc<Node>);

impl Drop for RequestGuard {
    fn drop(&mut self) {
        *self.0.last_request.lock().unwrap() = Instant::now();
        self.0.active.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Where a top-level node's prompts ran ([`super::prefill`]), for
/// `/v1/workers` and the next decision.
#[derive(Default)]
struct PrefillSeen {
    on_node: u64,
    on_tree: u64,
    fell_back: u64,
    /// Tokens a second through the tree, on the plan of this generation.
    tree: Option<(f64, u64)>,
    /// Tokens a second a hand-over to each node took, by its name.
    nodes: HashMap<String, f64>,
    last: Option<serde_json::Value>,
}

/// A plan of a range over this node and its workers.
struct Planned {
    pipeline: Arc<LayerPipeline>,
    entries: Vec<PlanEntry>,
    /// The workers it hands layers to.
    used: Vec<String>,
    /// Standbys among them, and whom each fills in for.
    standing: Vec<(String, String)>,
}

/// What a parent's `Assign` asks of the nodes below.
#[derive(Clone, Copy)]
struct Terms {
    n_ctx: usize,
    slots: usize,
    activations: ActivationFormat,
    /// The top-level node's prompt-weight copies, for every node of the tree.
    weights: Decision,
}

struct Assignment {
    store: Arc<SessionStore>,
    layers: Range<usize>,
    /// What the parent asked for, and the path down to this node: what
    /// planning this range again needs.
    terms: Terms,
    path: Vec<String>,
}

struct Parent {
    connection: u64,
    node: String,
    stream: TcpStream,
    /// When its connection last carried a frame.
    heard: Instant,
    /// Its `[workers]` name and the addresses it answered the channel at,
    /// once a scan found it there: what [`Node::check_parent`] probes.
    seen: Option<(String, Vec<String>)>,
    /// Probes in a row it has not answered since it was seen.
    missed: u32,
    /// A probe went unanswered, or a request forwarded to it failed.
    suspect: bool,
}

pub struct Node {
    id: String,
    settings: NodeSettings,
    model: Arc<dyn ModelForward>,
    loaded: LoadedModel,
    /// The workers this node hands layers to: the nodes it claimed on its
    /// channel, in the order they were found.
    links: RwLock<Vec<Arc<ChildLink>>>,
    /// Spare workers: never planned into a share.
    standby_links: RwLock<Vec<Arc<ChildLink>>>,
    /// The nodes the last scan of the channel found, and when it ran.
    found: Mutex<(Vec<super::channel::Found>, Option<Instant>)>,
    /// The node id behind each address this node claimed: what keeps one
    /// node reachable at two addresses from being claimed twice.
    claimed: Mutex<HashMap<String, String>>,
    /// Who this node is promised to ([`Message::Lease`]) and until when:
    /// a node about to claim it, or this node itself while it claims.
    lease: Mutex<Option<(String, Instant)>>,
    /// Nodes of the channel that probed this one before any scan found
    /// them: probed back at the scan's next turn.
    probed_by: Mutex<Vec<SocketAddr>>,
    /// Set to have the channel asked again at once, not at the next refresh.
    refresh_now: AtomicBool,
    /// Set while this node claims workers on its channel or holds the ones
    /// it claimed: it then refuses to be claimed itself.
    leading: AtomicBool,
    /// Requests running on this node, and when the last one ended: what a
    /// leading node lets its workers go by.
    active: AtomicUsize,
    last_request: Mutex<Instant>,
    /// Forwards running here for a parent: what this node's load counts
    /// while it works as a worker.
    forwards: Arc<AtomicUsize>,
    /// Assignments this node waits on: a worker fetching the layers it was
    /// given holds up the requests that are running here, so its fetch is
    /// not paced for them.
    assigning: AtomicUsize,
    /// Whether this node holds the whole model within its memory: it can
    /// serve requests alone, so it routes rather than splits.
    fits_alone: bool,
    /// Samples of every layer's weights, hashed: what tells two releases of
    /// a model apart in a probe. Zeros on a node with part of the model.
    content_hash: [u8; 32],
    /// Standbys that took over a lost worker's layers since the last plan:
    /// `(lost worker, standby)`.
    standing_in: Mutex<Vec<(String, String)>>,
    /// This node, for the stages that call back into it for a standby.
    this: OnceLock<Weak<Node>>,
    /// A top-level node's prompt-weight decision, measured at its first plan.
    decision: OnceLock<Decision>,
    /// Switched to serve alone through `POST /props`: its workers
    /// let go, no plan made until switched back.
    alone: AtomicBool,
    /// Whether the node running the final layer applies the output head.
    head_on_last: AtomicBool,
    /// Each node's decode speed as requests found it, by its own name: the
    /// bytes it ran over its time a token (its subtree, its link and its
    /// fixed costs included), and over how many tokens.
    observed: Mutex<HashMap<String, (f64, usize)>>,
    /// Decode tokens observed since the last plan.
    observed_since_plan: AtomicUsize,
    /// The rates the plan in progress uses: the observed ones when every
    /// node has one, the measured ones otherwise (`None`).
    observed_now: Mutex<Option<HashMap<String, f64>>>,
    /// The rate each node of the last top-level plan was planned at.
    planned_rates: Mutex<HashMap<String, f64>>,
    /// How this process loads another model — its own image again, with
    /// that model — for a parent that assigns one. `None` where
    /// `[web].reexec` is off or there is no `execve`.
    handover: OnceLock<Arc<crate::reexec::Handover>>,
    /// Set once a switch to the parent's model is under way.
    switching: AtomicBool,
    /// The engine's way of serving alone, told what this node decides.
    serving: OnceLock<Arc<crate::engine::generate::ServingSwitch>>,
    /// What the last plan's shares followed: a speed, or memory.
    shares_by: Mutex<&'static str>,
    /// The layers and decision this node's prompt-weight copies were built
    /// for.
    copied: Mutex<Option<(Range<usize>, Decision)>>,
    /// The top-level pipeline and its generation.
    top: RwLock<(Arc<LayerPipeline>, u64)>,
    assignment: Mutex<Option<Assignment>>,
    parent: Mutex<Option<Parent>>,
    plan: Mutex<Vec<PlanEntry>>,
    /// The addresses of the workers the current plan hands layers to.
    used: Mutex<Vec<String>>,
    /// Held by whoever is planning; the time of the last plan.
    planning: Mutex<Instant>,
    delegating: OnceLock<Weak<DelegatingModel>>,
    /// Each layer's weights, in bytes — what a plan sizes shares by.
    layer_bytes: Vec<u64>,
    /// The weights outside every layer: the embedding, the final norm and
    /// the output head, which only the top-level node runs.
    non_layer_bytes: u64,
    /// The model this node serves, as its handshake tells a parent.
    identity: super::protocol::ModelIdentity,
    /// The file it serves by range, and how fast while it runs requests.
    served: Option<super::share::ServedFile>,
    fetch_pace: super::share::Pace,
    /// The output head's bytes: the matrix as wide as the vocabulary.
    head_bytes: u64,
    local_addr: Option<SocketAddr>,
    metrics: NodeMetrics,
    next_connection: AtomicU64,
    stop: AtomicBool,
    /// This node's saved prefixes ([`super::prefixes`]).
    prefix_files: Option<Arc<PrefixFiles>>,
    /// The saved split the next top-level plan keeps, while each node still
    /// holds its share: set by a claim, for the plan it makes.
    saved_plan: Mutex<Option<Manifest>>,
    /// Whether the last top-level plan kept [`Self::saved_plan`]'s split.
    saved_split_kept: AtomicBool,
    /// Each layer's rows, in bytes a position: what handing a prompt's rows
    /// over moves ([`super::prefill`]).
    row_bytes: Vec<u64>,
    /// Why this model's rows cannot be handed between nodes, if they cannot.
    unmovable: Option<String>,
    /// Where prompts ran, and how fast.
    prefills: Mutex<PrefillSeen>,
    /// A decode step's time through a tree as requests found it, in
    /// milliseconds: the names of its nodes (sorted), the time, and over how
    /// many tokens.
    observed_step: Mutex<Option<(Vec<String>, f64, usize)>>,
    /// The last decision between serving alone and dividing
    /// ([`super::divide::decide`]), for `/v1/workers`.
    last_decision: Mutex<Option<super::divide::Decision>>,
    /// Set while a sync benchmark holds a free node of the channel
    /// ([`Self::sync_bench`]): no claim meanwhile.
    benching: AtomicBool,
    /// The saved prefixes a claim planned for, and whether it kept their
    /// split: put back in the pool, or deleted, at the next request.
    pending_prefixes: Mutex<Option<(Manifest, bool)>>,
    /// Requests this node got for a model no free node of the channel
    /// served ([`note_request`]).
    demand: Mutex<super::demand::Window>,
}

static NODE: OnceLock<Arc<Node>> = OnceLock::new();

/// Makes `node` the one this process's API asks about.
pub fn set_global(node: Arc<Node>) {
    let _ = NODE.set(node);
}

/// Whether this process's API is off: working for a parent, or running a
/// model it has fetched only in part (`super::fetch`).
pub fn api_paused() -> bool {
    NODE.get().is_some_and(|node| node.is_assigned()) || super::fetch::incomplete()
}

/// Switches a top-level node between its tree and serving alone (the
/// engine's side is `engine::generate::ServingSwitch`). Refused on a node
/// that is not top-level with workers, and on one that has only part of the
/// model.
pub fn set_alone(alone: bool) -> Result<(), String> {
    match NODE.get() {
        Some(node) => node.set_alone(alone),
        None => Err("this server has no [workers] section".to_string()),
    }
}

/// A request is starting: a node on a channel claims the free nodes it
/// found, and plans over them, before the request picks how it is served.
/// The guard counts the request as running until it is dropped.
pub fn request_started() -> Option<RequestGuard> {
    Some(NODE.get()?.start_request())
}

/// Why this process's predecessor ended and when (seconds since the
/// epoch), as a `--daemon` watchdog that started this one again tells it
/// (`crate::daemon`). `null` on a first start.
fn last_crash() -> serde_json::Value {
    let Ok(reason) = std::env::var("ORANGU_LAST_CRASH") else {
        return serde_json::Value::Null;
    };
    let at = std::env::var("ORANGU_LAST_CRASH_AT")
        .ok()
        .and_then(|v| v.parse::<u64>().ok());
    serde_json::json!({ "reason": reason, "at": at })
}

/// How a node's bid to claim the free nodes of its channel ended
/// ([`Node::contend`]).
#[derive(Clone, Debug, PartialEq, Eq)]
enum Contest {
    Won,
    /// To this node, which claims them instead.
    Lost(String),
}

/// The node `me` loses its claim to, given what the free nodes answered:
/// one that already leads any of them, else the lowest id one of them is
/// promised to, if lower than `me`.
fn contest_winner(
    me: &str,
    answers: impl IntoIterator<Item = super::channel::LeaseAnswer>,
) -> Option<String> {
    use super::channel::LeaseAnswer;
    let mut lowest: Option<String> = None;
    for answer in answers {
        match answer {
            LeaseAnswer::Granted => {}
            LeaseAnswer::Committed(leader) if leader != me => return Some(leader),
            LeaseAnswer::Committed(_) => {}
            LeaseAnswer::Promised(holder) => {
                if holder.as_str() < me && lowest.as_ref().is_none_or(|l| holder < *l) {
                    lowest = Some(holder);
                }
            }
        }
    }
    lowest
}

/// Where a request this node's API got is served instead, as the base URL
/// of another node's API, or `None` to serve it here. `forwarded`: the
/// request was already routed once, and goes no further.
pub fn route(forwarded: bool) -> Option<String> {
    if forwarded {
        return None;
    }
    NODE.get()?.route()
}

/// [`route`], on a blocking thread: deciding can ask the channel's free
/// nodes for a lease ([`Node::contend`]).
pub async fn route_async(forwarded: bool) -> Option<String> {
    if forwarded {
        return None;
    }
    NODE.get()?;
    tokio::task::spawn_blocking(move || route(false))
        .await
        .ok()
        .flatten()
}

/// Whether a request's `model` field can send it to another node: this
/// node's channel has a node serving another model by name. Only then is a
/// request's body read before it is served.
pub fn routes_by_model() -> bool {
    NODE.get()
        .is_some_and(|node| !node.other_models().is_empty())
}

/// Where a request whose `model` field is `asked` is served instead, as the
/// base URL of another node's API: the request names a model this node does
/// not serve and another node of its channel does
/// ([`Node::route_by_model`]). `None` serves it as [`route`] decides.
/// `forwarded`: the request was already routed once, and goes no further.
pub fn route_by_model(asked: &str, forwarded: bool) -> Option<String> {
    if forwarded {
        return None;
    }
    NODE.get()?.route_by_model(asked)
}

/// Whether this process counts the models requests name
/// ([`note_request`]): it is on a channel. Only then, or when
/// [`routes_by_model`], is a request's body read before it is served.
pub fn counts_demand() -> bool {
    NODE.get()
        .is_some_and(|node| node.settings.channel.is_some())
}

/// A request that runs the model, and was not routed here by another node,
/// names `asked` in its `model` field: counted as unserved demand when no
/// free node of the channel serves that model ([`Node::note_request`]).
/// What happens to the request is not changed.
pub fn note_request(asked: &str) {
    if let Some(node) = NODE.get() {
        node.note_request(asked, Instant::now());
    }
}

/// The names of `orangu-server`'s roles: a request naming one names no
/// model, and counts as demand for none.
const ROLE_NAMES: [&str; 6] = [
    "all",
    "code",
    "review",
    "explorer",
    "embeddings",
    "embedding",
];

/// A node of the channel as demand sees it ([`Node::servers`]).
#[derive(Clone, Debug, PartialEq)]
struct Server {
    name: String,
    label: String,
    quant: String,
    /// Holds the whole model: a node with part of one serves no request
    /// alone.
    whole: bool,
    /// Serves its own clients and is under [`CLAIM_CUTOFF`]: a request for
    /// its model is served without waiting.
    free: bool,
}

impl Server {
    fn serves(&self, asked: &str) -> bool {
        self.whole && super::channel::names_model(asked, &self.label, &self.quant)
    }
}

/// What a request naming `asked` counts as, given the nodes of the
/// channel: `None` when a free node serves it; otherwise the model's name —
/// as the first node serving it names it, or as asked when none does. A
/// role's name, or an empty or overlong name, counts as nothing.
fn unserved_name(asked: &str, servers: &[Server]) -> Option<String> {
    let asked = asked.trim();
    if asked.is_empty()
        || asked.len() > super::demand::MAX_NAME_BYTES
        || ROLE_NAMES.iter().any(|r| asked.eq_ignore_ascii_case(r))
    {
        return None;
    }
    let mut serving = servers.iter().filter(|s| s.serves(asked)).peekable();
    let first = serving.peek().map(|s| model_id(&s.label, &s.quant));
    if serving.any(|s| s.free) {
        return None;
    }
    Some(first.unwrap_or_else(|| asked.to_string()))
}

/// One model of the channel's demand, for `/v1/workers`: how many requests
/// wanted it (this node's and the found nodes' together) and how many of
/// them this node got, the nodes serving it, and how many of those are free.
fn demand_row(model: &str, wanted: u32, here: u32, servers: &[Server]) -> serde_json::Value {
    let serving: Vec<&Server> = servers.iter().filter(|s| s.serves(model)).collect();
    serde_json::json!({
        "model": model,
        "wanted": wanted,
        "here": here,
        "nodes": serving.iter().map(|s| s.name.clone()).collect::<Vec<_>>(),
        "free": serving.iter().filter(|s| s.free).count(),
    })
}

/// Runs a sync benchmark against a free node of this node's channel
/// ([`Node::sync_bench`]): `peer` names it, and `tokens`, `layers` and
/// `hidden` default to 50 and this model's own sizes. An error carries the
/// HTTP status to answer with.
pub fn sync_bench(
    peer: String,
    tokens: Option<usize>,
    layers: Option<usize>,
    hidden: Option<usize>,
) -> Result<serde_json::Value, (u16, String)> {
    let node = NODE
        .get()
        .ok_or((404, "this server has no [workers] section".to_string()))?;
    let config = node.model.config();
    let ask = super::syncbench::Ask {
        peer,
        tokens: tokens.unwrap_or(50),
        layers: layers.unwrap_or(config.n_layer),
        hidden: hidden.unwrap_or(config.n_embd),
    };
    ask.check().map_err(|why| (400, why))?;
    node.sync_bench(&ask)
}

/// A model another node of the channel serves, for `/v1/models`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelModel {
    /// What a request names it by: its label, with its quantization's tag
    /// when the label does not carry it.
    pub id: String,
    pub label: String,
    pub quant: String,
    /// The `[workers]` names of the nodes that serve it.
    pub nodes: Vec<String>,
}

/// The models the other nodes of this node's channel serve, other than its
/// own ([`Node::channel_models`]); empty without a channel.
pub fn channel_models() -> Vec<ChannelModel> {
    NODE.get()
        .map_or_else(Vec::new, |node| node.channel_models())
}

/// A model's name with its quantization's tag: `label:quant`, or the label
/// alone when it carries the tag already or there is none.
fn model_id(label: &str, quant: &str) -> String {
    let tagged = label
        .len()
        .checked_sub(quant.len() + 1)
        .filter(|&cut| label.is_char_boundary(cut))
        .is_some_and(|cut| {
            label[cut..].starts_with(':') && label[cut + 1..].eq_ignore_ascii_case(quant)
        });
    if quant.is_empty() || tagged {
        label.to_string()
    } else {
        format!("{label}:{quant}")
    }
}

/// A request is starting: [`request_started`], on a blocking thread, since
/// claiming workers can take a while.
pub async fn request_started_async() -> Option<RequestGuard> {
    NODE.get()?;
    tokio::task::spawn_blocking(request_started)
        .await
        .ok()
        .flatten()
}

/// Hands the node the way this process loads another model.
pub fn attach_handover(handover: Arc<crate::reexec::Handover>) {
    if let Some(node) = NODE.get() {
        let _ = node.handover.set(handover);
    }
}

/// Hands the node the engine's way of serving alone, which it then keeps
/// told whether its plan uses any worker.
pub fn attach_serving(switch: Arc<crate::engine::generate::ServingSwitch>) {
    if let Some(node) = NODE.get() {
        switch.set_chosen(node.used.lock().unwrap().is_empty());
        let _ = node.serving.set(switch);
    }
}

/// The `workers` object of `GET /props`: whether the tree serves, and its
/// plan. `None` without a `[workers]` section.
pub fn props_json() -> Option<serde_json::Value> {
    let node = NODE.get()?;
    Some(serde_json::json!({
        "enabled": node.delegates() && !node.alone.load(Ordering::Acquire),
        "switchable": node.delegates() && !node.is_assigned() && !super::fetch::incomplete(),
        "plan": node.plan().iter().map(|e| serde_json::json!({
            "node": e.node,
            "layers": [e.layer_start, e.layer_end],
        })).collect::<Vec<_>>(),
    }))
}

/// Why this process should not be routed to, if a worker tree says so.
/// A request routed to `base` could not be forwarded there
/// ([`Node::forward_failed`]).
pub fn forward_failed(base: &str) {
    if let Some(node) = NODE.get() {
        node.forward_failed(base);
    }
}

/// Whether this process works for a parent that does not answer.
pub fn parent_suspect() -> bool {
    NODE.get().is_some_and(|node| node.parent_suspect())
}

pub fn readiness() -> Option<&'static str> {
    NODE.get().and_then(|node| node.unready())
}

/// The `orangu_server_worker*` metric families, when this process has a
/// `[workers]` section; empty otherwise.
pub fn render_metrics() -> String {
    NODE.get()
        .map(|node| node.render_metrics())
        .unwrap_or_default()
}

/// Nodes' processors and measured speeds, for `/v1/workers`.
fn setups_json(setups: &[NodeSetup]) -> serde_json::Value {
    setups
        .iter()
        .map(|s| {
            serde_json::json!({
                "name": s.name,
                "decode_gb_per_s": (s.decode_rate > 0.0).then_some(s.decode_rate),
                "prompt_gb_per_s": (s.prompt_rate > 0.0).then_some(s.prompt_rate),
                "decode_fixed_ms": (s.decode_fixed_ms > 0.0).then_some(s.decode_fixed_ms),
                "prompt_fixed_ms": (s.prompt_fixed_ms > 0.0).then_some(s.prompt_fixed_ms),
                "devices": s.devices.iter().map(|d| serde_json::json!({
                    "kind": d.kind,
                    "name": d.name,
                    "cores": (d.cores > 0).then_some(d.cores),
                    "memory_bytes": (d.memory_bytes > 0).then_some(d.memory_bytes),
                    "in_use": d.in_use,
                })).collect::<Vec<_>>(),
            })
        })
        .collect()
}

/// This process's place in its tree, for `GET /v1/workers`.
/// Saves the prefix pool's entries on every node of this process's tree, as
/// the process stops.
pub fn save_prefixes_on_exit() {
    if let Some(node) = NODE.get() {
        let _planning = node.planning.lock().unwrap();
        node.save_prefixes();
    }
}

pub fn status_json() -> serde_json::Value {
    match NODE.get() {
        Some(node) => node.status(),
        None => serde_json::json!({ "role": "none" }),
    }
}

/// The tree as Prometheus text, recounted at every scrape: one
/// `orangu_server_worker_node_info` series per node. A dropped node disappears
/// on the next scrape, it never reads zero, so alerts use `absent()` or
/// the parent's `worker_up`.
pub fn discovery_metrics() -> String {
    match NODE.get() {
        Some(node) => node.discovery_metrics(),
        None => String::new(),
    }
}

fn escape_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// One series. Label names are fixed here; only the values come from the tree.
fn node_info_line(record: &DiscoveryRecord, via: Option<&str>) -> String {
    let mut labels = format!("node=\"{}\"", escape_label(&record.node));
    // Spelled to match the parent's `worker_up{worker=...}` label, so the
    // two can be joined.
    if !record.worker.is_empty() {
        labels += &format!(",worker=\"{}\"", escape_label(&record.worker));
    }
    // Which branch it came through; the top never talks deeper than one hop.
    if let Some(via) = via {
        labels += &format!(",via=\"{}\"", escape_label(via));
    }
    labels += &format!(
        ",metrics_host=\"{}\",metrics_port=\"{}\"",
        escape_label(&record.metrics_host),
        record.metrics_port
    );
    format!("orangu_server_worker_node_info{{{labels}}} 1\n")
}

impl Node {
    /// Starts the node: binds its listener, plans a top-level tree when it
    /// has workers and the model can be split, and starts the thread that
    /// keeps an eye on its workers.
    pub fn start(
        mut settings: NodeSettings,
        model: Arc<dyn ModelForward>,
        loaded: LoadedModel,
    ) -> Result<Arc<Self>> {
        let n_layer = model.config().n_layer;
        let link_to = |addr: &String| {
            Arc::new(ChildLink::with_tls(
                addr.clone(),
                settings.timeout,
                settings.connect_timeout,
                settings.tls.clone(),
            ))
        };
        let links: Vec<Arc<ChildLink>> = settings.workers.iter().map(link_to).collect();
        let standby_links: Vec<Arc<ChildLink>> = settings.standby.iter().map(link_to).collect();
        let listener = match &settings.listen {
            Some(addr) => Some(
                TcpListener::bind(addr)
                    .with_context(|| format!("failed to bind the [workers] listener on {addr}"))?,
            ),
            None => None,
        };
        let local_addr = listener.as_ref().and_then(|l| l.local_addr().ok());
        let mut layer_bytes = vec![0u64; n_layer];
        let mut head_bytes = 0u64;
        let mut embd_bytes = 0u64;
        let mut non_layer_bytes = 0u64;
        for (name, bytes) in loaded.tensor_sizes() {
            match name {
                "output.weight" => head_bytes = bytes,
                "token_embd.weight" => embd_bytes = bytes,
                _ => {}
            }
            match crate::engine::loader::block_index(name).filter(|il| *il < n_layer) {
                Some(il) => layer_bytes[il] += bytes,
                None => non_layer_bytes += bytes,
            }
        }
        // A tied head is the embedding matrix read the other way.
        if head_bytes == 0 {
            head_bytes = embd_bytes;
        }
        if settings.capacity.setups.is_empty() {
            settings.capacity.setups.push(NodeSetup {
                name: settings.name.clone(),
                ..NodeSetup::default()
            });
        }
        // Measured before a parent can ask: not for a node sharing by
        // memory, nor for one that has only part of the model on disk and
        // does not know yet which layers it will have.
        if settings.shares != Shares::Memory
            && settings.capacity.setups[0].decode_rate == 0.0
            && model.supports_layer_split()
            && !super::fetch::incomplete()
        {
            match super::speed::measure(model.as_ref(), &layer_bytes) {
                Ok((speed, layers, took)) => {
                    log::info!(
                        "orangu-server: [workers] this node runs {:.2} GB of weights a second \
                         in a decode step, {:.1} in a {}-token prompt chunk, and each costs \
                         {:.1} and {:.1} ms whatever its layers (layers {}..{}, measured in \
                         {:.1} s)",
                        speed.decode,
                        speed.prompt,
                        super::speed::PROMPT_TOKENS,
                        speed.decode_fixed_ms,
                        speed.prompt_fixed_ms,
                        layers.start,
                        layers.end,
                        took.as_secs_f64()
                    );
                    settings.capacity.setups[0].decode_rate = speed.decode;
                    settings.capacity.setups[0].prompt_rate = speed.prompt;
                    settings.capacity.setups[0].decode_fixed_ms = speed.decode_fixed_ms;
                    settings.capacity.setups[0].prompt_fixed_ms = speed.prompt_fixed_ms;
                }
                Err(e) => log::warn!(
                    "orangu-server: [workers] could not time this node's layers, its share \
                     follows its memory: {e:#}"
                ),
            }
        }
        let local = if model.supports_layer_split() {
            LayerPipeline::new(model.clone(), 0..n_layer, Vec::new())?
        } else {
            // Never run: a model that cannot be split is never delegated,
            // and refuses every assignment.
            LayerPipeline::unchecked(model.clone(), 0..n_layer)
        };
        let total_bytes: u64 = layer_bytes.iter().sum();
        let fits_alone = model.supports_layer_split()
            && !super::fetch::incomplete()
            && settings.capacity.budget_bytes >= total_bytes + non_layer_bytes;
        // Samples of every layer, for nodes of a channel to compare before
        // they route to each other: a few KiB of each tensor.
        let content_hash = if settings.channel.is_some() && !super::fetch::incomplete() {
            identity::model_identity(&loaded, "", "", 0..n_layer).range_hash
        } else {
            [0; 32]
        };
        // The model this node serves, by its layout: what a parent compares.
        let own_identity =
            identity::model_identity(&loaded, &settings.label, &settings.quant, 0..0);
        // Only a file this node holds whole: a partial one has holes.
        let served = settings
            .serve_file
            .as_deref()
            .filter(|_| super::fetch::partial().is_none())
            .and_then(super::share::ServedFile::new);
        let prefix_files = settings.prefix_root.as_ref().map(|root| {
            Arc::new(PrefixFiles::under(
                root,
                own_identity.header_hash,
                &settings.name,
            ))
        });
        let row_bytes = if model.supports_layer_split() {
            super::prefill::row_bytes(model.as_ref())
        } else {
            Vec::new()
        };
        let unmovable = if model.supports_layer_split() {
            super::prefill::movable(model.as_ref()).err()
        } else {
            Some("the model cannot be split".to_string())
        };
        let node = Arc::new(Self {
            id: format!("{:016x}", rand::random::<u64>()),
            settings,
            model,
            loaded,
            links: RwLock::new(links),
            standby_links: RwLock::new(standby_links),
            found: Mutex::new((Vec::new(), None)),
            claimed: Mutex::new(HashMap::new()),
            lease: Mutex::new(None),
            probed_by: Mutex::new(Vec::new()),
            refresh_now: AtomicBool::new(false),
            leading: AtomicBool::new(false),
            active: AtomicUsize::new(0),
            last_request: Mutex::new(Instant::now()),
            forwards: Arc::new(AtomicUsize::new(0)),
            assigning: AtomicUsize::new(0),
            fits_alone,
            content_hash,
            standing_in: Mutex::new(Vec::new()),
            this: OnceLock::new(),
            decision: OnceLock::new(),
            shares_by: Mutex::new("memory"),
            alone: AtomicBool::new(false),
            head_on_last: AtomicBool::new(false),
            observed: Mutex::new(HashMap::new()),
            observed_since_plan: AtomicUsize::new(0),
            observed_now: Mutex::new(None),
            planned_rates: Mutex::new(HashMap::new()),
            handover: OnceLock::new(),
            switching: AtomicBool::new(false),
            serving: OnceLock::new(),
            copied: Mutex::new(None),
            top: RwLock::new((Arc::new(local), 0)),
            assignment: Mutex::new(None),
            parent: Mutex::new(None),
            plan: Mutex::new(Vec::new()),
            used: Mutex::new(Vec::new()),
            planning: Mutex::new(Instant::now()),
            delegating: OnceLock::new(),
            layer_bytes,
            non_layer_bytes,
            head_bytes,
            identity: own_identity,
            served,
            fetch_pace: super::share::Pace::default(),
            local_addr,
            metrics: NodeMetrics::default(),
            next_connection: AtomicU64::new(1),
            stop: AtomicBool::new(false),
            prefix_files,
            saved_plan: Mutex::new(None),
            saved_split_kept: AtomicBool::new(false),
            pending_prefixes: Mutex::new(None),
            row_bytes,
            unmovable,
            prefills: Mutex::new(PrefillSeen::default()),
            observed_step: Mutex::new(None),
            last_decision: Mutex::new(None),
            benching: AtomicBool::new(false),
            demand: Mutex::new(super::demand::Window::default()),
        });
        let _ = node.this.set(Arc::downgrade(&node));
        if let Some(listener) = listener {
            let node = node.clone();
            std::thread::Builder::new()
                .name("orangu-workers-listen".to_string())
                .spawn(move || node.accept(listener))?;
        }
        // Before the first plan, which can take minutes: a worker connected
        // early hears nothing else meanwhile.
        for link in node.links().iter().chain(&node.standby_links()) {
            let (node, link) = (node.clone(), link.clone());
            std::thread::Builder::new()
                .name("orangu-workers-ping".to_string())
                .spawn(move || node.ping_worker(&link))?;
        }
        if node.delegates() {
            node.plan_top();
        } else if !node.links().is_empty() || node.settings.channel.is_some() {
            log::warn!(
                "orangu-server: [workers] has workers, but the {} architecture cannot be split \
                 across them yet; serving it here alone",
                node.model.config().architecture
            );
        }
        {
            let node = node.clone();
            std::thread::Builder::new()
                .name("orangu-workers-watch".to_string())
                .spawn(move || node.watch())?;
        }
        if node.settings.channel.is_some() {
            // Tests run many nodes in one process: their loads are their
            // slots alone.
            #[cfg(not(test))]
            super::load::start_sampler();
            let node = node.clone();
            std::thread::Builder::new()
                .name("orangu-workers-scan".to_string())
                .spawn(move || node.scan_channel())?;
        }
        Ok(node)
    }

    /// Where the listener is bound, when there is one.
    #[cfg(test)]
    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.local_addr
    }

    #[cfg(test)]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Whether this node hands layers to workers when it is top-level.
    fn delegates(&self) -> bool {
        self.settings.split
            && self.model.supports_layer_split()
            && (!self.links().is_empty() || self.can_lead())
    }

    /// The workers this node hands layers to, as they stand.
    fn links(&self) -> Vec<Arc<ChildLink>> {
        self.links.read().unwrap().clone()
    }

    /// The spare workers, as they stand.
    fn standby_links(&self) -> Vec<Arc<ChildLink>> {
        self.standby_links.read().unwrap().clone()
    }

    /// Whether this node can claim workers on its channel and lead them:
    /// it is on one, and has the whole model to embed and sample with.
    fn can_lead(&self) -> bool {
        self.settings.channel.is_some()
            && self.settings.split
            && self.model.supports_layer_split()
            && !super::fetch::incomplete()
    }

    /// A request is starting here: [`request_started`].
    fn start_request(self: &Arc<Self>) -> RequestGuard {
        self.active.fetch_add(1, Ordering::AcqRel);
        self.claim();
        self.load_pending_prefixes();
        RequestGuard(self.clone())
    }

    /// How busy this node is, `0.0` to `1.0` ([`super::load`]).
    fn load(&self) -> f32 {
        self.load_besides(0)
    }

    /// How busy this node is besides `this` of the requests running on it:
    /// what a request that is already counted weighs the node by.
    fn load_besides(&self, this: usize) -> f32 {
        super::load::combine(
            self.active.load(Ordering::Acquire).saturating_sub(this)
                + self.forwards.load(Ordering::Acquire),
            self.settings.slots,
            super::load::busy(),
        )
    }

    /// Whether this node serves the model alone: it can lead, and holds the
    /// whole model.
    fn whole(&self) -> bool {
        self.can_lead() && self.fits_alone
    }

    /// Where a request this node got is served instead ([`route`]). A node
    /// working for a parent sends its clients to that parent; a node that
    /// leads keeps them. Otherwise [`Self::choose`] decides: here alone, on
    /// another node that holds the model or leads a tree of it, or divided
    /// over the free nodes of the channel — unless another node about to do
    /// the same wins them ([`Self::contend`]), and then it goes there.
    fn route(&self) -> Option<String> {
        self.settings.channel.as_ref()?;
        let found = self.found.lock().unwrap().0.clone();
        if self.is_assigned() {
            let parent = self.parent.lock().unwrap().as_ref()?.node.clone();
            return found.into_iter().find(|f| f.node == parent).map(|f| f.api);
        }
        if self.leading.load(Ordering::Acquire) {
            return None;
        }
        if self.settings.divide == super::divide::Mode::Always {
            return self.route_dividing(&found);
        }
        let api_of = |node: &str| found.iter().find(|f| f.node == node).map(|f| f.api.clone());
        let decision = self.choose(0);
        let routed = match &decision.verdict {
            Verdict::Here => None,
            Verdict::To(api) => Some(api.clone()),
            Verdict::Divide => match self.contend() {
                Contest::Won => None,
                Contest::Lost(winner) => api_of(&winner),
            },
        };
        self.note_decision(decision);
        routed
    }

    /// [`Self::route`] with `ORANGU_WORKERS_DIVIDE=always`: to a node that
    /// leads a tree of the model and is not fully busy; else this node
    /// divides the model over the free nodes, whatever its size, unless
    /// another node about to do the same wins them. With no one free, a
    /// node that holds the model keeps a request unless a node that can
    /// serve it too is [`ROUTE_MARGIN`] less busy; one that does not sends
    /// it to the least busy that can.
    fn route_dividing(&self, found: &[super::channel::Found]) -> Option<String> {
        let api_of = |node: &str| found.iter().find(|f| f.node == node).map(|f| f.api.clone());
        let leader = found
            .iter()
            .filter(|f| {
                f.state == ProbeState::Leading
                    && f.node != self.id
                    && f.load < 1.0
                    && self.same_model(f, true)
            })
            .min_by(|a, b| a.load.total_cmp(&b.load));
        if let Some(leader) = leader {
            return Some(leader.api.clone());
        }
        if self.may_claim() {
            return match self.contend() {
                Contest::Won => None,
                Contest::Lost(winner) => api_of(&winner),
            };
        }
        let best = found
            .iter()
            .filter(|f| {
                f.whole
                    && f.state == ProbeState::Idle
                    && f.node != self.id
                    && self.same_model(f, true)
            })
            .min_by(|a, b| a.load.total_cmp(&b.load))?;
        (!self.whole() || best.load + ROUTE_MARGIN < self.load()).then(|| best.api.clone())
    }

    /// Whether `asked`, a request's `model` field, names the model this node
    /// serves ([`super::channel::names_model`]).
    fn serves_named(&self, asked: &str) -> bool {
        super::channel::names_model(asked, &self.settings.label, &self.settings.quant)
    }

    /// The nodes of the channel that serve another model under a name, as
    /// the last scan found them: each holds the whole of it and serves its
    /// own clients. A node working for another serves nobody's requests,
    /// and one with part of a model serves none alone; a node that does not
    /// say what it serves is never chosen by name. A node with this node's
    /// own model is left to [`Self::route`], whatever it calls it.
    fn other_models(&self) -> Vec<super::channel::Found> {
        if self.settings.channel.is_none() {
            return Vec::new();
        }
        self.found
            .lock()
            .unwrap()
            .0
            .iter()
            .filter(|f| {
                f.node != self.id
                    && !f.label.is_empty()
                    && f.content != [0; 32]
                    && matches!(f.state, ProbeState::Idle | ProbeState::Leading)
                    && !self.same_model(f, true)
            })
            .cloned()
            .collect()
    }

    /// Where a request whose `model` field is `asked` is served instead
    /// ([`route_by_model`]), when it names a model this node does not serve
    /// and another node of the channel does: the least busy of those that
    /// lead a tree of it and are not fully busy, else the least busy free
    /// one, else the least busy of all. `None` when it names this node's
    /// model, or one no node of the channel serves: the request is served
    /// as any other is.
    fn route_by_model(&self, asked: &str) -> Option<String> {
        if self.serves_named(asked) {
            return None;
        }
        let preference = |f: &super::channel::Found| match f.state {
            ProbeState::Leading if f.load < 1.0 => 0,
            ProbeState::Idle => 1,
            _ => 2,
        };
        self.other_models()
            .into_iter()
            .filter(|f| f.serves_named(asked))
            .min_by(|a, b| {
                preference(a)
                    .cmp(&preference(b))
                    .then_with(|| a.load.total_cmp(&b.load))
            })
            .map(|f| f.api)
    }

    /// The other models the channel serves ([`channel_models`]), one entry
    /// per name and quantization, in the order the scan found them. A name
    /// a request would find this node's own model by is left out: such a
    /// request is served here.
    fn channel_models(&self) -> Vec<ChannelModel> {
        let mut out: Vec<ChannelModel> = Vec::new();
        for f in self.other_models() {
            let id = model_id(&f.label, &f.quant);
            if self.serves_named(&id) {
                continue;
            }
            match out.iter_mut().find(|m| m.id.eq_ignore_ascii_case(&id)) {
                Some(model) => model.nodes.push(f.name),
                None => out.push(ChannelModel {
                    id,
                    label: f.label,
                    quant: f.quant,
                    nodes: vec![f.name],
                }),
            }
        }
        out
    }

    /// This node and the others of the channel as the last scan found them,
    /// for counting demand: each with what it serves by name, whether it
    /// holds it whole (the samples of its weights a probe answer carries are
    /// zeros otherwise), and whether it is free.
    fn servers(&self, found: &[super::channel::Found]) -> Vec<Server> {
        let mut out = vec![Server {
            name: self.settings.name.clone(),
            label: self.settings.label.clone(),
            quant: self.settings.quant.clone(),
            whole: self.content_hash != [0; 32],
            free: matches!(self.probe_state(), ProbeState::Idle | ProbeState::Leading)
                && self.load() < CLAIM_CUTOFF,
        }];
        out.extend(found.iter().filter(|f| f.node != self.id).map(|f| Server {
            name: f.name.clone(),
            label: f.label.clone(),
            quant: f.quant.clone(),
            whole: f.content != [0; 32],
            free: matches!(f.state, ProbeState::Idle | ProbeState::Leading)
                && f.load < CLAIM_CUTOFF,
        }));
        out
    }

    /// Counts a request naming `asked` as unserved demand ([`note_request`])
    /// when no free node of the channel serves it ([`unserved_name`]).
    fn note_request(&self, asked: &str, now: Instant) {
        if self.settings.channel.is_none() {
            return;
        }
        let found = self.found.lock().unwrap().0.clone();
        if let Some(name) = unserved_name(asked, &self.servers(&found)) {
            self.demand.lock().unwrap().note(&name, now);
        }
    }

    /// What this node's probe answer says was wanted here.
    fn shared_demand(&self) -> Vec<super::protocol::Wanted> {
        self.demand.lock().unwrap().shared(Instant::now())
    }

    /// The channel's unserved demand, for `/v1/workers`: this node's own
    /// model, and every model wanted anywhere on the channel, most wanted
    /// first.
    fn demand_status(&self) -> serde_json::Value {
        let here = self.demand.lock().unwrap().counts(Instant::now());
        let found = self.found.lock().unwrap().0.clone();
        let mut wanted = here.clone();
        for f in found.iter().filter(|f| f.node != self.id) {
            super::demand::merge(&mut wanted, &f.demand);
        }
        wanted.sort_by(|a, b| b.count.cmp(&a.count).then_with(|| a.model.cmp(&b.model)));
        let servers = self.servers(&found);
        let here_of = |model: &str| {
            here.iter()
                .find(|w| w.model.eq_ignore_ascii_case(model))
                .map_or(0, |w| w.count)
        };
        let own_id = model_id(&self.settings.label, &self.settings.quant);
        let names_own = |model: &str| self.serves_named(model);
        let own_wanted = wanted
            .iter()
            .filter(|w| names_own(&w.model))
            .fold(0u32, |sum, w| sum.saturating_add(w.count));
        let own_here = here
            .iter()
            .filter(|w| names_own(&w.model))
            .fold(0u32, |sum, w| sum.saturating_add(w.count));
        serde_json::json!({
            "window_seconds": super::demand::WINDOW.as_secs(),
            "own": demand_row(&own_id, own_wanted, own_here, &servers),
            "models": wanted
                .iter()
                .map(|w| demand_row(&w.model, w.count, here_of(&w.model), &servers))
                .collect::<Vec<_>>(),
        })
    }

    /// How many workers this node holds while it leads, its standby
    /// included, `0` otherwise.
    fn workers_held(&self) -> u32 {
        if self.leading.load(Ordering::Acquire) {
            (self.links().len() + self.standby_links().len()) as u32
        } else {
            0
        }
    }

    /// Who this node is promised to now, if anyone.
    fn lease_holder(&self) -> Option<String> {
        let lease = self.lease.lock().unwrap();
        lease
            .as_ref()
            .filter(|(_, until)| Instant::now() < *until)
            .map(|(holder, _)| holder.clone())
    }

    /// This node's answer to `candidate` asking for it ([`Message::Lease`]).
    /// A node already working for a parent, or leading, is that parent's;
    /// otherwise it is promised to the lowest id that asked within
    /// [`LEASE_FOR`] — itself included, while it claims — so two nodes that
    /// claim at once end with the same winner on every node.
    fn answer_lease(&self, candidate: &str) -> Message {
        let refused = |holder: String, committed: bool| Message::LeaseAck {
            granted: false,
            holder,
            committed,
        };
        if let Some(parent) = self.parent.lock().unwrap().as_ref() {
            return refused(parent.node.clone(), true);
        }
        if self.leading.load(Ordering::Acquire) {
            return refused(self.id.clone(), true);
        }
        let mut lease = self.lease.lock().unwrap();
        let now = Instant::now();
        match super::channel::promised_against(lease.as_ref(), candidate, now) {
            Some(holder) => refused(holder, false),
            None => {
                *lease = Some((candidate.to_string(), now + LEASE_FOR));
                Message::LeaseAck {
                    granted: true,
                    holder: candidate.to_string(),
                    committed: false,
                }
            }
        }
    }

    /// Before this node claims the free nodes it found, asks each of them
    /// to promise itself to it ([`Message::Lease`]), having promised itself
    /// first. It loses to a node that already leads one of them, or that
    /// has a lower id and asked too — on any node, this one included —
    /// and the request then goes to that node, which claims the rest.
    fn contend(&self) -> Contest {
        let Some(channel) = self.settings.channel.as_deref() else {
            return Contest::Won;
        };
        {
            let mut lease = self.lease.lock().unwrap();
            let now = Instant::now();
            match lease.as_ref() {
                Some((holder, until))
                    if now < *until && holder != &self.id && *holder < self.id =>
                {
                    return Contest::Lost(holder.clone());
                }
                _ => *lease = Some((self.id.clone(), now + LEASE_FOR)),
            }
        }
        let hash = super::channel::channel_hash(channel, self.settings.secret.as_deref());
        let tls = self.settings.tls.clone();
        let free = self.unclaimed();
        let answers: Vec<Option<super::channel::LeaseAnswer>> = std::thread::scope(|scope| {
            let asks: Vec<_> = free
                .iter()
                .map(|found| {
                    let (hash, tls, id) = (&hash, tls.as_ref(), &self.id);
                    scope.spawn(move || {
                        let addr = found.addr.parse().ok()?;
                        super::channel::lease(addr, hash, id, tls)
                    })
                })
                .collect();
            asks.into_iter()
                .map(|ask| ask.join().ok().flatten())
                .collect()
        });
        if let Some(winner) = contest_winner(&self.id, answers.into_iter().flatten()) {
            log::info!(
                "orangu-server: workers: another node claims channel '{channel}' at the same \
                 time; this request goes to it ({winner})"
            );
            return Contest::Lost(winner);
        }
        // A lower id may have asked this node while it asked the others.
        match self.lease_holder() {
            Some(holder) if holder != self.id => Contest::Lost(holder),
            _ => Contest::Won,
        }
    }

    /// Whether this node leads a smaller tree of the same model than
    /// another node does — fewer workers, or as many and a higher id —
    /// and should let its workers go as soon as it is idle, so that one
    /// channel forms one tree.
    fn yields_to_larger(&self) -> bool {
        if !self.leading.load(Ordering::Acquire) || self.active.load(Ordering::Acquire) > 0 {
            return false;
        }
        let mine = self.workers_held();
        self.found.lock().unwrap().0.iter().any(|f| {
            f.state == ProbeState::Leading
                && f.node != self.id
                && self.same_model(f, true)
                && (f.workers > mine || (f.workers == mine && f.node < self.id))
        })
    }

    /// Whether a request may claim workers now: [`Self::may_claim_any`],
    /// and dividing pays ([`Self::divides`]).
    fn may_claim(&self) -> bool {
        self.may_claim_with(0)
    }

    /// [`Self::may_claim`], for a request already counted as running here
    /// when `this` is `1`.
    fn may_claim_with(&self, this: usize) -> bool {
        self.may_claim_any() && self.divides(this)
    }

    /// Whether this node could claim workers now: it can lead, serves its
    /// own clients, runs no sync benchmark, and its last scan found someone
    /// free that it does not hold yet.
    fn may_claim_any(&self) -> bool {
        self.can_lead()
            && !self.is_assigned()
            && !self.alone.load(Ordering::Acquire)
            && !self.benching.load(Ordering::Acquire)
            && !self.unclaimed().is_empty()
    }

    /// Whether claiming the free nodes pays, by [`NodeSettings::divide`]:
    /// always with `ORANGU_WORKERS_DIVIDE=always`. Otherwise, for a node not
    /// leading yet, when [`Self::choose`] divides; for one that leads, when
    /// its workers lack the room for the model, when a node's speed is not
    /// known, or when the free nodes are predicted to make its tree's decode
    /// step at least [`super::divide::MARGIN`] faster.
    fn divides(&self, this: usize) -> bool {
        if self.settings.divide == super::divide::Mode::Always {
            return true;
        }
        if !self.leading.load(Ordering::Acquire) {
            return self.choose(this).verdict == Verdict::Divide;
        }
        let room: u64 = self.own_budget(true)
            + self
                .links()
                .iter()
                .filter_map(|link| link.info())
                .map(|info| plan::subtree_budget(&info.capacity))
                .sum::<u64>();
        if room < self.layer_bytes.iter().sum::<u64>() {
            return true;
        }
        match (
            self.predicted_tree_ms(&[]),
            self.predicted_tree_ms(&self.unclaimed()),
        ) {
            (Some(now), Some(with)) => with <= now * (1.0 - super::divide::MARGIN),
            _ => true,
        }
    }

    /// Where a request starting now is served ([`super::divide::decide`]),
    /// `this` being `1` for a request already counted as running here: this
    /// node alone when it holds the model; each free node that holds it
    /// and the nodes that lead a tree of it, at the time a token their probes
    /// said; and, when this node could claim workers, the tree over it and
    /// the free nodes, as predicted ([`Self::predicted_tree_ms`]).
    fn choose(&self, this: usize) -> super::divide::Decision {
        use super::divide::Server;
        let found = self.found.lock().unwrap().0.clone();
        let here = self.whole().then(|| Server {
            api: None,
            name: self.settings.name.clone(),
            ms: self.alone_ms(),
            load: self.load_besides(this),
        });
        let server = |f: &super::channel::Found| Server {
            api: Some(f.api.clone()),
            name: f.name.clone(),
            ms: (f.pace.step_ms > 0.0 && f.pace.step_ms.is_finite())
                .then_some(f64::from(f.pace.step_ms)),
            load: f.load,
        };
        let holders: Vec<Server> = found
            .iter()
            .filter(|f| {
                f.whole
                    && f.state == ProbeState::Idle
                    && f.node != self.id
                    && self.same_model(f, true)
            })
            .map(server)
            .collect();
        let leaders: Vec<Server> = found
            .iter()
            .filter(|f| {
                f.state == ProbeState::Leading
                    && f.node != self.id
                    && f.load < 1.0
                    && self.same_model(f, true)
            })
            .map(server)
            .collect();
        let can_divide = self.may_claim_any();
        let tree = if can_divide {
            self.predicted_tree_ms(&self.unclaimed())
        } else {
            None
        };
        super::divide::decide(
            here.as_ref(),
            &holders,
            &leaders,
            can_divide,
            tree,
            ROUTE_MARGIN,
        )
    }

    /// Keeps `decision` for `/v1/workers`, and logs it when it goes another
    /// way than the last one.
    fn note_decision(&self, decision: super::divide::Decision) {
        let mut last = self.last_decision.lock().unwrap();
        let changed = last.as_ref().is_none_or(|l| l.verdict != decision.verdict);
        if changed {
            let to = match &decision.verdict {
                Verdict::Here => "served here alone".to_string(),
                Verdict::To(_) => format!("sent to {}", decision.name),
                Verdict::Divide => "divided over the free nodes".to_string(),
            };
            log::info!(
                "orangu-server: workers: requests are {to}: {}",
                decision.reason
            );
        }
        *last = Some(decision);
    }

    /// Times what two nodes splitting every layer would spend summing their
    /// halves ([`super::syncbench`]): connects to `ask.peer`, a free node of
    /// the channel, as a parent does — over TLS when the channel uses it, the
    /// secret proved — and echoes a hidden-size payload off it, `f32` and then
    /// `f16`, `2 × layers` times a simulated token. The peer works for nobody
    /// meanwhile and is let go after. Refused while this node leads a tree or
    /// works for one, so a running tree is never disturbed, and for a peer
    /// that is not free. The result sets the sync time beside this node's
    /// predicted decode alone (`predicted`).
    fn sync_bench(&self, ask: &super::syncbench::Ask) -> Result<serde_json::Value, (u16, String)> {
        use super::syncbench::{Prediction, Timing, exchange};
        let refused = |why: String| Err((409, why));
        if self.settings.channel.is_none() {
            return refused("this node is on no channel ([workers].name)".to_string());
        }
        if self.parent.lock().unwrap().is_some() {
            return refused("this node works for a parent".to_string());
        }
        if self.benching.swap(true, Ordering::AcqRel) {
            return refused("a sync benchmark is running here already".to_string());
        }
        struct Done<'a>(&'a AtomicBool);
        impl Drop for Done<'_> {
            fn drop(&mut self) {
                self.0.store(false, Ordering::Release);
            }
        }
        let _done = Done(&self.benching);
        // Checked with the benchmark held: no claim starts from here on.
        let _planning = self.planning.lock().unwrap();
        if self.leading.load(Ordering::Acquire) {
            return refused(
                "this node leads a tree: benchmark between requests, once its workers are let go"
                    .to_string(),
            );
        }
        let found = self.found.lock().unwrap().0.clone();
        let wanted = ask.peer.trim();
        let Some(peer) = found
            .iter()
            .find(|f| {
                f.name == wanted
                    || f.addr == wanted
                    || f.node == wanted
                    || f.alternates.iter().any(|a| a == wanted)
            })
            .cloned()
        else {
            return Err((
                404,
                format!(
                    "the channel has no node {wanted}; it has {}",
                    found
                        .iter()
                        .map(|f| f.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            ));
        };
        if !peer.state.claimable() {
            return refused(format!(
                "{} is {}: only a free node is benchmarked",
                peer.name,
                peer.state.as_str()
            ));
        }
        drop(_planning);
        let link = ChildLink::with_tls(
            peer.addr.clone(),
            self.settings.timeout,
            self.settings.connect_timeout,
            self.settings.tls.clone(),
        );
        let info = link
            .connect(
                self.settings.secret.as_deref(),
                std::slice::from_ref(&self.id),
            )
            .map_err(|e| (502, format!("{}: {e}", peer.name)))?;
        if info.features & super::protocol::FEATURE_ECHO == 0 {
            link.disconnect();
            return Err((
                409,
                format!("{} does not echo: run the same release on both", peer.name),
            ));
        }
        log::info!(
            "orangu-server: workers: sync benchmark with {} ({} tokens, {} layers, {} wide)",
            peer.name,
            ask.tokens,
            ask.layers,
            ask.hidden
        );
        let started = Instant::now();
        let mut timings = Vec::new();
        for width in [4usize, 2] {
            let bytes = ask.hidden * width;
            match exchange(&link, ask.tokens, ask.round_trips(), bytes) {
                Ok(times) => timings.push(Timing::of(&times, bytes)),
                Err(error) => {
                    link.disconnect();
                    return Err((502, format!("{}: {error}", peer.name)));
                }
            }
        }
        link.disconnect();
        let (f32_timing, f16_timing) = (&timings[0], &timings[1]);
        let (sync_from, sync) = match self.settings.activations {
            ActivationFormat::F32 => ("f32", f32_timing),
            _ => ("f16", f16_timing),
        };
        let own = &self.settings.capacity.setups[0];
        let peer_rate = info
            .capacity
            .setups
            .first()
            .map(|s| f64::from(s.decode_rate))
            .filter(|r| *r > 0.0);
        let predicted = Prediction::new(
            self.layer_bytes.iter().sum::<u64>() + self.head_bytes,
            f64::from(own.decode_rate),
            f64::from(own.decode_fixed_ms),
            peer_rate,
            sync.token_median_ms,
        );
        log::info!(
            "orangu-server: workers: sync with {}: {:.2} ms a token in f16, {:.2} in f32 \
             ({:.3} ms a round trip)",
            peer.name,
            f16_timing.token_median_ms,
            f32_timing.token_median_ms,
            f16_timing.round_trip_median_ms
        );
        Ok(serde_json::json!({
            "peer": {
                "name": peer.name,
                "address": peer.addr,
                "tls": link.tls(),
                "same_model": self.same_model(&peer, false),
                "decode_gb_per_s": peer_rate,
            },
            "link": info.link.map(|(rtt, bandwidth)| serde_json::json!({
                "rtt_ms": rtt.as_secs_f64() * 1e3,
                "mb_per_s": bandwidth / 1e6,
            })),
            "tokens": ask.tokens,
            "layers": ask.layers,
            "hidden": ask.hidden,
            "round_trips_per_token": ask.round_trips(),
            "f32": f32_timing.json(),
            "f16": f16_timing.json(),
            "seconds": started.elapsed().as_secs_f64(),
            "predicted": predicted.map(|p| {
                let mut json = p.json();
                json["sync_from"] = serde_json::json!(sync_from);
                json["peer_rate_known"] = serde_json::json!(peer_rate.is_some());
                json
            }),
        }))
    }

    /// The bytes of the hidden state that cross a link each way, a position.
    fn hidden_bytes(&self) -> f64 {
        let n_embd = self.model.config().n_embd as f64;
        match self.settings.activations {
            ActivationFormat::F32 => 4.0 * n_embd,
            ActivationFormat::F16 => 2.0 * n_embd,
            ActivationFormat::Q8_0 => 34.0 / 32.0 * n_embd,
        }
    }

    /// This node's time a token serving its model alone, in milliseconds:
    /// every layer and the output head at its measured decode speed, and the
    /// cost of a step whatever its layers. `None` when it does not hold the
    /// model or did not measure.
    fn alone_ms(&self) -> Option<f64> {
        if !self.fits_alone || super::fetch::incomplete() {
            return None;
        }
        let own = &self.settings.capacity.setups[0];
        super::divide::step_ms(
            self.layer_bytes.iter().sum::<u64>() + self.head_bytes,
            f64::from(own.decode_rate),
            f64::from(own.decode_fixed_ms),
        )
    }

    /// What a request sent here takes a token, as a probe says
    /// ([`super::protocol::Pace::step_ms`]): through the tree this node
    /// leads, or its model alone; `0` when it cannot tell.
    fn step_ms(&self) -> f32 {
        let ms = if self.leading.load(Ordering::Acquire) {
            self.predicted_tree_ms(&[])
        } else if self.whole() {
            self.alone_ms()
        } else {
            None
        };
        ms.map_or(0.0, |ms| ms as f32)
    }

    /// One node of a tree for [`super::divide::tree_ms`]: its rate as
    /// requests found it through a tree, else as it measured it (`measured`,
    /// gigabytes a second), at what its load leaves free.
    fn member(
        &self,
        name: &str,
        measured: f64,
        fixed_ms: f32,
        budget: u64,
        hop: Option<(Duration, f64)>,
    ) -> super::divide::Member {
        let observed = self.observed.lock().unwrap().get(name).map(|(r, _)| *r);
        let spare = self.spare(name);
        match observed {
            // Its subtree, its link and its fixed costs are in what it showed.
            Some(rate) => super::divide::Member {
                name: name.to_string(),
                rate: rate * spare,
                fixed_ms: 0.0,
                budget,
                hop: None,
            },
            None => super::divide::Member {
                name: name.to_string(),
                rate: measured * spare,
                fixed_ms: f64::from(fixed_ms),
                budget,
                hop,
            },
        }
    }

    /// The nodes of a tree over this node, the workers it holds that answer,
    /// and `extra`, free nodes as the channel found them: a free node's link
    /// is its probe's time, halved — a dial and a question — and its
    /// bandwidth is not known.
    fn tree_members(&self, extra: &[super::channel::Found]) -> Vec<super::divide::Member> {
        let own = &self.settings.capacity.setups[0];
        let mut members = vec![self.member(
            &self.settings.name,
            f64::from(own.decode_rate),
            own.decode_fixed_ms,
            self.own_budget(true),
            None,
        )];
        for link in &self.links() {
            let Some(info) = link.info() else { continue };
            let setups = &info.capacity.setups;
            let name = setups
                .first()
                .map_or_else(|| link.addr.clone(), |s| s.name.clone());
            let measured = if setups.is_empty() || setups.iter().any(|s| s.decode_rate <= 0.0) {
                0.0
            } else {
                setups.iter().map(|s| f64::from(s.decode_rate)).sum()
            };
            let fixed = setups.first().map_or(0.0, |s| s.decode_fixed_ms);
            members.push(self.member(
                &name,
                measured,
                fixed,
                plan::subtree_budget(&info.capacity),
                info.link,
            ));
        }
        for found in extra {
            members.push(self.member(
                &found.name,
                f64::from(found.pace.decode_rate),
                found.pace.decode_fixed_ms,
                found.pace.budget_bytes,
                Some((found.rtt / 2, 0.0)),
            ));
        }
        members
    }

    /// The names of [`Self::tree_members`], sorted: what a tree's observed
    /// step is kept under.
    fn member_names(&self, extra: &[super::channel::Found]) -> Vec<String> {
        let mut names: Vec<String> = self
            .tree_members(extra)
            .into_iter()
            .map(|m| m.name)
            .collect();
        names.sort();
        names
    }

    /// A decode step's time through a tree over [`Self::tree_members`], in
    /// milliseconds: what requests found through a tree of the same nodes,
    /// after [`CALIBRATE_TOKENS`] tokens, else as predicted
    /// ([`super::divide::tree_ms`]). `None` for no worker, or a node whose
    /// speed or budget is not known.
    fn predicted_tree_ms(&self, extra: &[super::channel::Found]) -> Option<f64> {
        let members = self.tree_members(extra);
        if members.len() < 2 {
            return None;
        }
        let mut names: Vec<String> = members.iter().map(|m| m.name.clone()).collect();
        names.sort();
        if let Some((seen, ms, tokens)) = self.observed_step.lock().unwrap().as_ref()
            && *seen == names
            && *tokens >= CALIBRATE_TOKENS
        {
            return Some(*ms);
        }
        let model = self.model.clone();
        let allowed = move |at: usize| model.split_allowed(at);
        super::divide::tree_ms(
            &members,
            &super::divide::Model {
                layer_bytes: &self.layer_bytes,
                head_bytes: self.head_bytes,
                hidden_bytes: self.hidden_bytes(),
                local_layers: self.settings.local_layers,
                allowed: &allowed,
            },
        )
    }

    /// Why a node that leads a tree of a model it holds alone serves it
    /// alone again: requests found the tree's decode step slower than this
    /// node alone is predicted to take, over [`CALIBRATE_TOKENS`] tokens of
    /// the tree it leads now. Only between requests, and only by measurement.
    fn tree_is_slower_than_alone(&self) -> Option<String> {
        if self.settings.divide == super::divide::Mode::Always
            || !self.leading.load(Ordering::Acquire)
            || self.active.load(Ordering::Acquire) > 0
        {
            return None;
        }
        let alone = self.alone_ms()?;
        let names = self.member_names(&[]);
        let (seen, ms, tokens) = self.observed_step.lock().unwrap().clone()?;
        (seen == names && tokens >= CALIBRATE_TOKENS && ms > alone).then(|| {
            format!(
                "the tree decoded at {ms:.1} ms a token, this node alone at a predicted                  {alone:.1} ms"
            )
        })
    }

    /// The free nodes the last scan found that this node does not hold.
    ///
    /// A node busier than [`CLAIM_CUTOFF`] is left alone, except, for a node
    /// not leading yet, one its saved prefixes' split names: claimed whatever
    /// its load, so a busy moment does not cost the saved entries.
    fn unclaimed(&self) -> Vec<super::channel::Found> {
        let busy = self
            .found
            .lock()
            .unwrap()
            .0
            .iter()
            .any(|found| found.load > CLAIM_CUTOFF);
        let saved: Vec<String> = if busy && !self.leading.load(Ordering::Acquire) {
            self.prefix_manifest()
                .map(|m| m.workers.into_iter().map(|w| w.name).collect())
                .unwrap_or_default()
        } else {
            Vec::new()
        };
        let links: Vec<Arc<ChildLink>> = self
            .links()
            .into_iter()
            .chain(self.standby_links())
            .collect();
        let held: Vec<String> = links.iter().map(|link| link.addr.clone()).collect();
        // A node held at one address is not claimed again at another; one
        // whose link went down is, where it answers now.
        let claimed = self.claimed.lock().unwrap();
        let held_nodes: Vec<&String> = links
            .iter()
            .filter(|link| link.is_connected())
            .filter_map(|link| claimed.get(&link.addr))
            .collect();
        self.found
            .lock()
            .unwrap()
            .0
            .iter()
            .filter(|found| {
                found.state.claimable()
                    && self.same_model(found, false)
                    && (found.load <= CLAIM_CUTOFF || saved.contains(&found.name))
                    && found.node != self.id
                    && !held.contains(&found.addr)
                    && !held_nodes.contains(&&found.node)
            })
            .cloned()
            .collect()
    }

    /// Whether `found` serves this node's model: the same layout, and —
    /// with `weights`, for a node to route to — the same samples of every
    /// layer's weights. A worker's weights are compared layer by layer when
    /// it is claimed, so a node with part of the model can be claimed.
    fn same_model(&self, found: &super::channel::Found, weights: bool) -> bool {
        found.model == self.identity.header_hash
            && (!weights || (found.content != [0; 32] && found.content == self.content_hash))
    }

    /// A node of this channel probed this one from `tcp`'s far end: if it
    /// is not known yet, it is probed back on the channel's port at the
    /// next turn of the scan, rather than at the next full scan — a node
    /// that starts after this one is found within seconds.
    fn note_prober(&self, tcp: &TcpStream) {
        let port = &self.settings.scan.port;
        if *port == 0 {
            return;
        }
        let Ok(peer) = tcp.peer_addr() else {
            return;
        };
        let addr = SocketAddr::new(peer.ip(), *port);
        let shown = addr.to_string();
        let known = self
            .found
            .lock()
            .unwrap()
            .0
            .iter()
            .any(|f| f.addr == shown || f.alternates.contains(&shown));
        let mut pending = self.probed_by.lock().unwrap();
        if !known && !pending.contains(&addr) && Some(addr) != self.local_addr {
            pending.push(addr);
        }
    }

    /// What a probe of this node is told it is doing.
    fn probe_state(&self) -> ProbeState {
        if self.parent.lock().unwrap().is_some() {
            ProbeState::Serving
        } else if self.leading.load(Ordering::Acquire) {
            ProbeState::Leading
        } else if self.can_lead() {
            ProbeState::Idle
        } else {
            ProbeState::WorkerOnly
        }
    }

    /// Claims the free nodes the last scan found, and plans over them with
    /// the ones already held. A node another node claimed first refuses
    /// the handshake and is let go again.
    fn claim(self: &Arc<Self>) {
        // The request that asks is counted as running here already.
        if !self.may_claim_with(1) {
            return;
        }
        let mut last = self.planning.lock().unwrap();
        // Claimed by a parent, or another request claimed them, or a sync
        // benchmark started, while this one waited.
        let fresh = self.unclaimed();
        if fresh.is_empty() || self.is_assigned() || self.benching.load(Ordering::Acquire) {
            return;
        }
        let was_leading = self.leading.swap(true, Ordering::AcqRel);
        // A tree formed again: planned on the split its saved prefixes were
        // made on, when every node still holds its share.
        let offered = if was_leading {
            None
        } else {
            self.prefix_manifest()
        };
        *self.saved_plan.lock().unwrap() = offered.clone();
        self.saved_split_kept.store(false, Ordering::Release);
        log::info!(
            "orangu-server: workers: claiming {} on channel '{}'",
            fresh
                .iter()
                .map(|f| format!("{} ({})", f.name, f.addr))
                .collect::<Vec<_>>()
                .join(", "),
            self.settings.channel.as_deref().unwrap_or_default()
        );
        for found in fresh {
            // The same node at an address that stopped answering: that
            // link goes, this one takes its place.
            let stale: Vec<String> = {
                let claimed = self.claimed.lock().unwrap();
                claimed
                    .iter()
                    .filter(|(addr, node)| **node == found.node && **addr != found.addr)
                    .map(|(addr, _)| addr.clone())
                    .collect()
            };
            if !stale.is_empty() {
                self.links
                    .write()
                    .unwrap()
                    .retain(|link| !stale.contains(&link.addr));
                let mut claimed = self.claimed.lock().unwrap();
                for addr in &stale {
                    claimed.remove(addr);
                }
            }
            self.claimed
                .lock()
                .unwrap()
                .insert(found.addr.clone(), found.node.clone());
            let link = Arc::new(ChildLink::with_tls(
                found.addr.clone(),
                self.settings.timeout,
                self.settings.connect_timeout,
                self.settings.tls.clone(),
            ));
            self.links.write().unwrap().push(link.clone());
            let node = self.clone();
            let _ = std::thread::Builder::new()
                .name("orangu-workers-ping".to_string())
                .spawn(move || node.ping_worker(&link));
        }
        self.plan_top_locked();
        // Nobody answered the handshake: another node has them, or they
        // went away. They are dropped, and found again by a later scan.
        let lost: Vec<String> = self
            .links()
            .iter()
            .filter(|link| !link.is_connected())
            .map(|link| link.addr.clone())
            .collect();
        if !lost.is_empty() {
            self.links
                .write()
                .unwrap()
                .retain(|link| !lost.contains(&link.addr));
            self.claimed
                .lock()
                .unwrap()
                .retain(|addr, _| !lost.contains(addr));
            self.found
                .lock()
                .unwrap()
                .0
                .retain(|found| !lost.contains(&found.addr));
            if self.links().is_empty() {
                self.leading.store(false, Ordering::Release);
            }
            self.plan_top_locked();
        }
        *self.saved_plan.lock().unwrap() = None;
        if let Some(manifest) = offered {
            let kept = self.saved_split_kept.load(Ordering::Acquire);
            *self.pending_prefixes.lock().unwrap() = Some((manifest, kept));
        }
        *last = Instant::now();
    }

    /// The manifest of this node's saved prefixes, when it serves through a
    /// tree with a prefix pool.
    fn prefix_manifest(&self) -> Option<Manifest> {
        let model = self.delegating.get().and_then(Weak::upgrade)?;
        model.prefix_cache()?;
        self.prefix_files.as_ref()?.manifest()
    }

    /// The node name of the worker behind the stage named `stage`, as its
    /// handshake said, and what it said.
    fn worker_named(&self, stage: &str) -> Option<(String, super::link::ChildInfo)> {
        let link = self
            .links()
            .into_iter()
            .chain(self.standby_links())
            .find(|link| link.addr == stage)?;
        let info = link.info()?;
        let name = info.capacity.setups.first()?.name.clone();
        Some((name, info))
    }

    /// `manifest`'s split of `range` over this node and `candidates` — the
    /// candidates in its order — when it still stands: every worker it names
    /// is a candidate and no other, the context is the same, the cuts are
    /// still allowed, and every node holds its share: within the budget it
    /// reports now, or with no less a budget than when the entries were saved
    /// (a share the plan gave past its budget then, paged in as it runs).
    /// Every node is different, and a node's memory or load can change
    /// between claims; a split one of them cannot hold is cut afresh. Says
    /// why it does not stand.
    #[allow(clippy::type_complexity)]
    fn saved_cut(
        &self,
        manifest: &Manifest,
        range: &Range<usize>,
        own_budget: u64,
        candidates: &[(Arc<ChildLink>, ChildInfo)],
    ) -> Result<
        (
            Vec<(Arc<ChildLink>, ChildInfo)>,
            Range<usize>,
            Vec<Range<usize>>,
        ),
        String,
    > {
        if manifest.n_ctx != self.settings.n_ctx {
            return Err(format!(
                "they were saved for a context of {}, not {}",
                manifest.n_ctx, self.settings.n_ctx
            ));
        }
        if candidates.len() != manifest.workers.len() {
            return Err(format!(
                "they were saved on {} workers, and {} are here",
                manifest.workers.len(),
                candidates.len()
            ));
        }
        let name_of = |info: &ChildInfo| info.capacity.setups.first().map(|s| s.name.clone());
        let mut ordered = Vec::new();
        for worker in &manifest.workers {
            let found = candidates
                .iter()
                .find(|(_, info)| name_of(info).as_deref() == Some(worker.name.as_str()))
                .ok_or_else(|| format!("{} is not here", worker.name))?;
            ordered.push(found.clone());
        }
        let local = manifest.local();
        let shares: Vec<Range<usize>> = manifest.workers.iter().map(SavedNode::layers).collect();
        let n_layer = self.model.config().n_layer;
        let mut end = local.start;
        for part in std::iter::once(&local).chain(&shares) {
            let allowed = |at: usize| at == 0 || at == n_layer || self.model.split_allowed(at);
            if part.start != end || part.start > part.end || !allowed(part.end) {
                return Err(format!("layers {part:?} do not follow on from {end}"));
            }
            end = part.end;
        }
        if local.start != range.start || end != range.end {
            return Err(format!("they cover {}..{end}, not {range:?}", local.start));
        }
        let holds = |share: &Range<usize>, budget: u64, then: u64| {
            plan::bytes_of(share, &self.layer_bytes) <= budget || budget >= then
        };
        if !holds(&local, own_budget, manifest.local_budget) {
            return Err(format!("this node cannot hold layers {local:?}"));
        }
        for (((_, info), share), worker) in ordered.iter().zip(&shares).zip(&manifest.workers) {
            if !holds(share, plan::subtree_budget(&info.capacity), worker.budget) {
                return Err(format!("{} cannot hold layers {share:?}", worker.name));
            }
        }
        Ok((ordered, local, shares))
    }

    /// The prefixes a claim planned for, at the request that made it: put
    /// back in the pool when the claim kept their split — each entry only
    /// when every worker listed it in its handshake, for the layers it runs
    /// now — and deleted on every node otherwise.
    fn load_pending_prefixes(&self) {
        let Some((manifest, kept)) = self.pending_prefixes.lock().unwrap().take() else {
            return;
        };
        let _planning = self.planning.lock().unwrap();
        let (Some(model), Some(files)) = (
            self.delegating.get().and_then(Weak::upgrade),
            self.prefix_files.as_ref(),
        ) else {
            return;
        };
        let pipeline = self.top.read().unwrap().0.clone();
        if !kept {
            for entry in &manifest.entries {
                pipeline.forget(&entry.key);
            }
            files.clear_lead();
            model.load_prefixes(&manifest, &[]);
            log::info!(
                "orangu-server: workers: {} saved prefix{} dropped on every node: the tree is                  split differently from when they were saved",
                manifest.entries.len(),
                if manifest.entries.len() == 1 {
                    ""
                } else {
                    "es"
                }
            );
            return;
        }
        let plan = self.plan();
        let listed = |stage: &Arc<dyn Stage>, key: &str| -> bool {
            let Some((name, info)) = self.worker_named(&stage.name()) else {
                return false;
            };
            let Some(now) = plan.iter().find(|e| e.node == name) else {
                return false;
            };
            let then = manifest.plan.iter().find(|e| e.name == name);
            stage.saves_prefixes()
                && info.features & FEATURE_PREFIXES != 0
                && then.is_some_and(|then| {
                    then.layers == (now.layer_start as usize, now.layer_end as usize)
                })
                && info.saved.iter().any(|saved| {
                    saved.key == key
                        && saved.layer_start == now.layer_start
                        && saved.layer_end == now.layer_end
                })
        };
        let mut keys = Vec::new();
        for entry in &manifest.entries {
            if pipeline
                .stages()
                .iter()
                .all(|stage| listed(stage, &entry.key))
            {
                keys.push(entry.key.clone());
            } else {
                log::info!(
                    "orangu-server: workers: saved prefix {} dropped on every node: a worker did                      not list it for its layers",
                    entry.key
                );
                pipeline.forget(&entry.key);
                files.forget_lead(&entry.key);
            }
        }
        model.load_prefixes(&manifest, &keys);
    }

    /// Saves the prefix pool's entries on every node of the current plan,
    /// and empties the pool ([`DelegatingModel::save_prefixes`]), with the
    /// plan's split as the manifest: what a top-level node does as it lets
    /// its workers go and as it stops. Called with the planning lock held.
    fn save_prefixes(&self) {
        if self.is_assigned() || self.prefix_files.is_none() {
            return;
        }
        let Some(model) = self.delegating.get().and_then(Weak::upgrade) else {
            return;
        };
        if model.prefix_cache().is_none() {
            return;
        }
        let pipeline = self.top.read().unwrap().0.clone();
        let local = pipeline.local_layers();
        let workers = pipeline
            .stages()
            .iter()
            .map(|stage| {
                let named = self.worker_named(&stage.name());
                SavedNode {
                    budget: named
                        .as_ref()
                        .map_or(0, |(_, info)| plan::subtree_budget(&info.capacity)),
                    name: named.map_or_else(|| stage.name(), |(name, _)| name),
                    layers: (stage.layers().start, stage.layers().end),
                }
            })
            .collect();
        let plan = self
            .plan()
            .into_iter()
            .map(|e| SavedNode {
                name: e.node,
                layers: (e.layer_start as usize, e.layer_end as usize),
                budget: 0,
            })
            .collect();
        model.save_prefixes(Manifest {
            n_ctx: self.settings.n_ctx,
            local: (local.start, local.end),
            local_budget: self.own_budget(true),
            workers,
            plan,
            entries: Vec::new(),
        });
    }

    /// Claims a free node of the channel as a standby, for a worker that is
    /// missing when no standby is held: it takes the worker's layers rather
    /// than the nodes left sharing them. `None` when the channel has none.
    fn claim_spare(&self) -> Option<Arc<ChildLink>> {
        self.settings.channel.as_ref()?;
        let found = self.unclaimed().into_iter().next()?;
        let link = Arc::new(ChildLink::with_tls(
            found.addr.clone(),
            self.settings.timeout,
            self.settings.connect_timeout,
            self.settings.tls.clone(),
        ));
        self.claimed
            .lock()
            .unwrap()
            .insert(found.addr.clone(), found.node.clone());
        self.standby_links.write().unwrap().push(link.clone());
        if let Some(node) = self.this.get().and_then(Weak::upgrade) {
            let held = link.clone();
            let _ = std::thread::Builder::new()
                .name("orangu-workers-ping".to_string())
                .spawn(move || node.ping_worker(&held));
        }
        log::info!(
            "orangu-server: workers: claiming {} ({}) on channel '{}' to stand in for a lost worker",
            found.name,
            found.addr,
            self.settings.channel.as_deref().unwrap_or_default()
        );
        Some(link)
    }

    /// Lets go of every worker this node claimed on its channel: each
    /// drops its sessions and serves its own clients again, free for the
    /// next node with a request to claim.
    fn release_workers(&self, why: &str) {
        let mut last = self.planning.lock().unwrap();
        if !self.leading.load(Ordering::Acquire) || self.active.load(Ordering::Acquire) > 0 {
            return;
        }
        self.save_prefixes();
        let links: Vec<Arc<ChildLink>> = std::mem::take(&mut *self.links.write().unwrap());
        let spares: Vec<Arc<ChildLink>> = std::mem::take(&mut *self.standby_links.write().unwrap());
        self.claimed.lock().unwrap().clear();
        for link in links.iter().chain(&spares) {
            link.disconnect();
        }
        self.leading.store(false, Ordering::Release);
        log::info!(
            "orangu-server: workers: {why}; {} let go",
            links
                .iter()
                .map(|link| link.addr.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        self.plan_top_locked();
        *last = Instant::now();
    }

    /// Whether a leading node has had no request for `[workers].release`.
    fn release_due(&self) -> bool {
        self.leading.load(Ordering::Acquire)
            && self.active.load(Ordering::Acquire) == 0
            && self.last_request.lock().unwrap().elapsed() >= self.settings.release
    }

    /// The addresses the parent answered the channel at, once seen there.
    fn parent_addrs(&self) -> Vec<SocketAddr> {
        self.parent
            .lock()
            .unwrap()
            .as_ref()
            .and_then(|p| p.seen.as_ref())
            .map(|(_, addrs)| addrs.iter().filter_map(|a| a.parse().ok()).collect())
            .unwrap_or_default()
    }

    /// Weighs what a turn of the channel scan found against the parent. A
    /// parent that answers is in good health. One seen on the channel that
    /// stops answering is suspect, and once it leaves [`PARENT_MISSES`]
    /// probes in a row unanswered with its connection quiet too, it is
    /// taken for gone: its connection is closed, which lets go of it as a
    /// closed connection does. A parent never seen on the channel is left
    /// to [`NodeSettings::parent_silence`].
    fn check_parent(&self, found: &[super::channel::Found]) {
        let mut slot = self.parent.lock().unwrap();
        let Some(parent) = slot.as_mut() else {
            return;
        };
        if let Some(f) = found.iter().find(|f| f.node == parent.node) {
            let addrs = std::iter::once(&f.addr)
                .chain(&f.alternates)
                .cloned()
                .collect();
            parent.seen = Some((f.name.clone(), addrs));
            parent.missed = 0;
            parent.suspect = false;
            return;
        }
        let Some((name, _)) = parent.seen.as_ref() else {
            return;
        };
        parent.missed += 1;
        parent.suspect = true;
        let quiet = parent.heard.elapsed();
        if parent.missed >= PARENT_MISSES && quiet >= self.settings.parent_silence / 4 {
            log::warn!(
                "orangu-server: the parent {name} has answered neither the channel ({} probes) \
                 nor its connection ({} s); it is taken for gone",
                parent.missed,
                quiet.as_secs()
            );
            let _ = parent.stream.shutdown(std::net::Shutdown::Both);
        }
    }

    /// A request forwarded to `base` failed. When that is the parent's API,
    /// the parent is suspect and the channel is asked about it at once.
    fn forward_failed(&self, base: &str) {
        let api = {
            let found = self.found.lock().unwrap();
            let parent = self.parent.lock().unwrap();
            let Some(parent) = parent.as_ref() else {
                return;
            };
            found
                .0
                .iter()
                .find(|f| f.node == parent.node)
                .map(|f| f.api.clone())
        };
        if api.as_deref() != Some(base) {
            return;
        }
        if let Some(parent) = self.parent.lock().unwrap().as_mut() {
            parent.suspect = true;
        }
        self.refresh_now.store(true, Ordering::Release);
    }

    /// Whether the parent this node works for is suspect.
    fn parent_suspect(&self) -> bool {
        self.parent
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|p| p.suspect)
    }

    /// Scans the channel every `scan_every`, and asks the nodes it found
    /// how busy they are every [`REFRESH_EVERY`] in between, for as long as
    /// the node runs.
    fn scan_channel(self: Arc<Self>) {
        let Some(channel) = self.settings.channel.clone() else {
            return;
        };
        let hash = super::channel::channel_hash(&channel, self.settings.secret.as_deref());
        let tls = self.settings.tls.clone();
        let scan = self.settings.scan.clone();
        let mut scanned: Option<Instant> = None;
        let mut peers: Vec<SocketAddr> = Vec::new();
        while !self.stop.load(Ordering::Relaxed) {
            let full = scanned.is_none_or(|at| at.elapsed() >= self.settings.scan_every);
            // Names are looked up again at each full scan, not every turn.
            if full {
                peers = super::channel::resolve_peers(&scan.peers);
            }
            let probed_by: Vec<SocketAddr> = std::mem::take(&mut *self.probed_by.lock().unwrap());
            // Asked at every turn: the peers, the nodes found and their
            // alternates (one whose address stopped answering moves to
            // another within a turn), nodes that probed this one, and the
            // parent.
            let mut addrs: Vec<SocketAddr> = peers.clone();
            addrs.extend(
                self.found
                    .lock()
                    .unwrap()
                    .0
                    .iter()
                    .flat_map(|f| std::iter::once(&f.addr).chain(&f.alternates))
                    .filter_map(|addr| addr.parse::<SocketAddr>().ok()),
            );
            addrs.extend(probed_by);
            addrs.extend(self.parent_addrs());
            let mut found = match (&scan.networks, full) {
                (Some(networks), true) => super::channel::scan(
                    &super::channel::scan_hosts(networks),
                    scan.port,
                    &hash,
                    tls.as_ref(),
                ),
                _ => Vec::new(),
            };
            if full {
                scanned = Some(Instant::now());
            }
            let mut rest: Vec<SocketAddr> = Vec::new();
            for addr in addrs {
                if !rest.contains(&addr) && !found.iter().any(|f| f.addr == addr.to_string()) {
                    rest.push(addr);
                }
            }
            found.extend(super::channel::probe_all(&rest, &hash, tls.as_ref()));
            // A gguf node runs no layers: it is never this node's worker.
            found.retain(|f| f.node != self.id && f.kind == super::protocol::NodeKind::Server);
            self.check_parent(&found);
            {
                let mut slot = self.found.lock().unwrap();
                let found = super::channel::one_per_node(found, &slot.0);
                let shape = |list: &[super::channel::Found]| {
                    list.iter()
                        .map(|f| (f.addr.clone(), f.state))
                        .collect::<Vec<_>>()
                };
                if shape(&slot.0) != shape(&found) {
                    log::info!(
                        "orangu-server: workers: channel '{channel}' has {}",
                        if found.is_empty() {
                            "no other node".to_string()
                        } else {
                            found
                                .iter()
                                .map(|f| {
                                    format!(
                                        "{} ({}, {}{})",
                                        f.name,
                                        f.addr,
                                        f.state.as_str(),
                                        if self.same_model(f, false) {
                                            ""
                                        } else {
                                            ", another model: not used"
                                        }
                                    )
                                })
                                .collect::<Vec<_>>()
                                .join(", ")
                        }
                    );
                }
                *slot = (found, Some(Instant::now()));
            }
            let started = Instant::now();
            let wait = REFRESH_EVERY.min(self.settings.scan_every);
            while started.elapsed() < wait
                && !self.stop.load(Ordering::Relaxed)
                && self.probed_by.lock().unwrap().is_empty()
                && !self.refresh_now.swap(false, Ordering::AcqRel)
            {
                std::thread::sleep(self.settings.maintenance.min(wait));
            }
        }
    }

    /// The model this node's own API serves: a [`DelegatingModel`] over
    /// its current plan when it has workers to delegate to, `None` when it
    /// runs the model alone.
    pub fn delegating_model(self: &Arc<Self>) -> Option<Arc<DelegatingModel>> {
        if !self.delegates() {
            return None;
        }
        let source: Arc<dyn PipelineSource> = self.clone();
        let model = Arc::new(DelegatingModel::with_source(self.model.clone(), source));
        if let Some(files) = &self.prefix_files {
            model.attach_prefix_files(files.clone());
        }
        let _ = self.delegating.set(Arc::downgrade(&model));
        Some(model)
    }

    pub fn is_assigned(&self) -> bool {
        self.assignment.lock().unwrap().is_some()
    }

    fn unready(&self) -> Option<&'static str> {
        if self.is_assigned() {
            return Some(if self.parent_suspect() {
                "serving a parent orangu-server that does not answer"
            } else {
                "serving a parent orangu-server"
            });
        }
        if super::fetch::incomplete() {
            return Some("the model is only partly downloaded; this node serves only as a worker");
        }
        let used = self.used.lock().unwrap();
        let lost = self
            .links()
            .iter()
            .any(|link| !link.is_connected() && used.contains(&link.addr));
        lost.then_some("a worker was lost")
    }

    /// The plan as it stands, one line per node.
    pub fn plan(&self) -> Vec<PlanEntry> {
        self.plan.lock().unwrap().clone()
    }

    fn render_metrics(&self) -> String {
        let held = self.links();
        let links: Vec<LinkSnapshot<'_>> = held
            .iter()
            .map(|link| LinkSnapshot {
                addr: &link.addr,
                up: link.is_connected(),
                metrics: &link.metrics,
            })
            .collect();
        let mut out = super::metrics::render(&self.metrics, self.is_assigned(), &links);
        if let Some(pool) = self.prefix_cache() {
            out.push_str(&super::metrics::render_prefix_cache(&pool.stats()));
        }
        out
    }

    /// The store for a parent's sequences: two sessions a parent's slot —
    /// the request running and the conversation kept for its next turn — and
    /// what this node keeps for the parent's prefix pool.
    fn session_store(&self, pipeline: Arc<LayerPipeline>, terms: Terms) -> SessionStore {
        let reserve = pool_reserve(self.settings.slots);
        let own = SessionTerms {
            pool_reserve: reserve as u32,
            ..SessionTerms::default()
        };
        SessionStore::new(pipeline, terms.n_ctx, own.session_limit(terms.slots))
            .with_pool_reserve(reserve)
            .with_prefixes(self.prefix_files.clone())
            .with_whole_model(self.fits_alone && !super::fetch::incomplete())
            .with_model_hash(self.identity.header_hash)
            .named(&self.settings.name)
    }

    /// How this node is configured to hold sequences, for its parent's
    /// handshake: its own slots and context, and the sessions it — and every
    /// worker below it — keeps for the parent's prefix pool.
    fn session_terms(&self) -> SessionTerms {
        let reserve = self
            .links()
            .iter()
            .filter_map(|link| link.info())
            .map(|info| info.pool_reserve())
            .fold(pool_reserve(self.settings.slots), usize::min);
        SessionTerms {
            slots: self.settings.slots as u32,
            n_ctx: self.settings.n_ctx as u32,
            pool_reserve: reserve as u32,
        }
    }

    /// The prefix pool of the model this node serves through its tree.
    fn prefix_cache(&self) -> Option<Arc<crate::engine::prefix_cache::PrefixCache>> {
        self.delegating
            .get()
            .and_then(Weak::upgrade)
            .and_then(|model| model.prefix_cache())
    }

    fn status(&self) -> serde_json::Value {
        // The parent this node works for, and whether it still answers.
        let parent = self.parent.lock().unwrap().as_ref().map(|p| {
            serde_json::json!({
                "node": p.node,
                "name": p.seen.as_ref().map(|(name, _)| name.clone()),
                "suspect": p.suspect,
                "missed_probes": p.missed,
                "last_heard_seconds": p.heard.elapsed().as_secs(),
            })
        });
        let assignment = self.assignment.lock().unwrap();
        let plan: Vec<_> = self
            .plan()
            .into_iter()
            .map(|e| {
                serde_json::json!({
                    "node": e.node,
                    "layers": [e.layer_start, e.layer_end],
                })
            })
            .collect();
        // Sessions each worker of the plan holds for the prefix pool: one an
        // entry, on every worker the plan uses.
        let pool_sessions = self
            .delegating
            .get()
            .and_then(Weak::upgrade)
            .map_or(0, |model| model.pool_sessions());
        let used = self.used.lock().unwrap().clone();
        let workers: Vec<_> = self
            .links()
            .iter()
            .map(|link| {
                let info = link.info();
                // How it is configured to hold sequences, as its handshake
                // said, and what this node's pool holds there.
                let sessions = info.as_ref().and_then(|i| {
                    i.sessions.map(|terms| {
                        serde_json::json!({
                            "slots": terms.slots,
                            "n_ctx": terms.n_ctx,
                            "session_limit": terms.session_limit(self.settings.slots),
                            "pool_reserve": i.pool_reserve(),
                            "retained_sessions": if used.contains(&link.addr) { pool_sessions } else { 0 },
                        })
                    })
                });
                // The child's own record is always first (see
                // anything after it is a grandchild. An empty
                // address means its metrics listener is disabled.
                let metrics = info
                    .as_ref()
                    .and_then(|i| i.discovery.first())
                    .filter(|r| r.metrics_port != 0)
                    .map(|r| {
                        serde_json::json!({
                            "host": r.metrics_host,
                            "port": r.metrics_port,
                        })
                    });
                serde_json::json!({
                    "address": link.addr,
                    "connected": info.is_some(),
                    // Its failures in a row, and how long they leave it out.
                    "failures_in_a_row": link.failures_in_a_row(),
                    "left_out_seconds": link.out_for().map(|left| left.as_secs()),
                    "node": info.as_ref().map(|i| i.node.clone()),
                    // Its name as plans print it, so a reader can match the
                    // worker to the channel's nodes and the plan's entries.
                    "name": info.as_ref().and_then(|i| i.capacity.setups.first()).map(|s| s.name.clone()),
                    "metrics": metrics,
                    "subtree_nodes": info.as_ref().map(|i| i.capacity.subtree_nodes),
                    "features": info.as_ref().map(|i| super::protocol::feature_names(i.features)),
                    "sessions": sessions,
                    "setups": info.as_ref().map(|i| setups_json(&i.capacity.setups)),
                    "link": info.as_ref().and_then(|i| i.link).map(|(rtt, bandwidth)| serde_json::json!({
                        "rtt_ms": rtt.as_secs_f64() * 1e3,
                        "mb_per_s": bandwidth / 1e6,
                    })),
                    "tls": link.tls(),
                })
            })
            .collect();
        // This node's own metrics address; not shown when the metrics
        // listener is off.
        let record = self.self_record();
        let metrics = (record.metrics_port != 0).then_some(serde_json::json!({
            "host": record.metrics_host,
            "port": record.metrics_port,
        }));
        let (found, scanned) = self.found.lock().unwrap().clone();
        // What the prefix pool holds and served, on a node serving through
        // its tree; the sessions a worker keeps for its parent's pool.
        let prefix_cache = self.prefix_cache().map(|pool| {
            let stats = pool.stats();
            let (saved, loaded) = self
                .delegating
                .get()
                .and_then(Weak::upgrade)
                .map_or((0, 0), |model| model.prefixes_saved());
            serde_json::json!({
                "entries": stats.entries,
                "hits": stats.hits,
                "misses": stats.misses,
                "tokens_held": stats.tokens_held,
                // Entries saved on every node's disk, and entries put back in
                // the pool from there at the last claim.
                "saved": saved,
                "loaded": loaded,
            })
        });
        // What this node holds on its disk for a parent's prefix pool.
        let saved_prefixes = self.prefix_files.as_ref().map(|files| {
            let (entries, bytes) = files.usage();
            serde_json::json!({ "entries": entries, "bytes": bytes })
        });
        let retained = assignment.as_ref().map(|a| {
            let (sessions, bytes) = a.store.retained();
            serde_json::json!({ "sessions": sessions, "bytes": bytes })
        });
        serde_json::json!({
            "node": self.id,
            "name": self.settings.name,
            // The channel, how busy this node is, and the other nodes of the
            // channel as the last scan found them.
            "channel": self.settings.channel,
            // Why the server before this one ended, when a `--daemon`
            // watchdog started this one after it.
            "last_crash": last_crash(),
            "load": self.load(),
            "holds_model": self.whole(),
            "leading": self.leading.load(Ordering::Acquire),
            "scanned_seconds_ago": scanned.map(|at| at.elapsed().as_secs()),
            "found": found
                .iter()
                .map(|f| {
                    serde_json::json!({
                        "name": f.name,
                        "address": f.addr,
                        "alternates": f.alternates,
                        "probe_ms": f.rtt.as_secs_f64() * 1e3,
                        "state": f.state.as_str(),
                        "load": f.load,
                        "holds_model": f.whole,
                        "same_model": self.same_model(f, false),
                        // The model it serves by name, as a request names
                        // it; empty from a node that does not say.
                        "model_name": f.label,
                        "model_quant": f.quant,
                        "api": f.api,
                        // How fast it decodes, and its time a token for a
                        // request sent to it, as its probe said; `0` unknown.
                        "decode_gb_per_s": f.pace.decode_rate,
                        "step_ms": f.pace.step_ms,
                    })
                })
                .collect::<Vec<_>>(),
            "role": if assignment.is_some() { "worker" } else { "top" },
            "parent": parent,
            "alone": self.alone.load(Ordering::Acquire),
            "layers": assignment.as_ref().map(|a| [a.layers.start, a.layers.end]),
            "listen": self.local_addr.map(|a| a.to_string()),
            "model": self.settings.label,
            "quant": self.settings.quant,
            "plan": plan,
            "workers": workers,
            // Memory this node has promised prompts on their way, and the
            // prompts its tree refused for lack of it.
            "kv_promised_bytes": self.top.read().unwrap().0.room().promised_bytes(),
            "kv_refusals": super::room::refused(),
            "prefix_cache": prefix_cache,
            "retained_sessions": retained,
            "saved_prefixes": saved_prefixes,
            // This node's processors and speed, and what the last
            // plan's shares followed.
            "setup": setups_json(&self.settings.capacity.setups[..1])[0],
            "shares": *self.shares_by.lock().unwrap(),
            // Each node's decode speed as requests found it (gigabytes of its
            // layers a second, and over how many tokens), and the speed the
            // plan went by.
            "observed_gb_per_s": self.observed.lock().unwrap().iter().map(|(n, (rate, tokens))| (n.clone(), serde_json::json!({"rate": rate, "tokens": tokens}))).collect::<serde_json::Map<_, _>>(),
            "planned_gb_per_s": self.planned_rates.lock().unwrap().clone(),
            "head": if self.head_on_last.load(Ordering::Acquire) { "on the node with the final layer" } else { "here" },
            // Where prompts ran: on one node that holds the whole model, or
            // through the tree, and the last one's numbers.
            "prefill": self.prefill_status(),
            // Whether requests are served alone or divided, and why.
            "divide": self.divide_status(),
            // Requests for models no free node of the channel served.
            "demand": self.demand_status(),
            // Spare workers, and whom each stands in for since the last
            // plan, if anyone.
            "standby": self
                .standby_links()
                .iter()
                .map(|link| {
                    let standing = self.standing_in.lock().unwrap();
                    serde_json::json!({
                        "address": link.addr,
                        "connected": link.is_connected(),
                        "standing_in_for": standing
                            .iter()
                            .find(|(_, standby)| *standby == link.addr)
                            .map(|(lost, _)| lost.clone()),
                    })
                })
                .collect::<Vec<_>>(),
            // Workers the current plan hands layers to that are not
            // connected: what a repair — the next request, or the worker
            // coming back — is waiting for.
            "lost": self
                .used
                .lock()
                .unwrap()
                .iter()
                .filter(|addr| {
                    self.links()
                        .iter()
                        .any(|link| &link.addr == *addr && !link.is_connected())
                })
                .cloned()
                .collect::<Vec<_>>(),
            // This node's own metrics address, for Prometheus's tree list.
            "metrics": metrics,
        })
    }

    /// Whether requests are served alone or divided, for `/v1/workers`: the
    /// mode, this node's time a token alone and for a request sent to it,
    /// what requests found through its last tree, and the last decision.
    fn divide_status(&self) -> serde_json::Value {
        let observed = self.observed_step.lock().unwrap().clone().map(
            |(nodes, ms, tokens)| serde_json::json!({ "nodes": nodes, "ms": ms, "tokens": tokens }),
        );
        let last = self.last_decision.lock().unwrap().clone().map(|d| {
            serde_json::json!({
                "verdict": match d.verdict {
                    Verdict::Here => "here",
                    Verdict::To(_) => "routed",
                    Verdict::Divide => "divided",
                },
                "to": d.name,
                "reason": d.reason,
                "alone_ms": d.alone_ms,
                "tree_ms": d.tree_ms,
            })
        });
        serde_json::json!({
            "mode": self.settings.divide.label(),
            "alone_ms": self.alone_ms(),
            "step_ms": self.step_ms(),
            "observed_tree": observed,
            "last": last,
        })
    }

    /// Where prompts ran, for `/v1/workers`.
    fn prefill_status(&self) -> serde_json::Value {
        let seen = self.prefills.lock().unwrap();
        let mut observed: serde_json::Map<String, serde_json::Value> = seen
            .nodes
            .iter()
            .map(|(name, rate)| (name.clone(), serde_json::json!(rate)))
            .collect();
        if let Some((rate, _)) = seen.tree {
            observed.insert("tree".to_string(), serde_json::json!(rate));
        }
        serde_json::json!({
            "mode": super::prefill::Mode::from_env().label(),
            "on_node": seen.on_node,
            "on_tree": seen.on_tree,
            "fell_back": seen.fell_back,
            "observed_tokens_per_s": observed,
            "last": seen.last,
        })
    }

    /// What [`super::prefill::decide`] weighs for a prompt on `pipeline`:
    /// every node's measured prompt speed, the links to the workers, the rows
    /// each worker's layers hold, and what earlier prompts took.
    fn prefill_inputs(
        &self,
        pipeline: &LayerPipeline,
        tree_tokens: usize,
        all_tokens: usize,
    ) -> super::prefill::Inputs {
        use super::prefill::{Branch, Candidate, TreeNode};
        let rows_of = |range: Range<usize>| -> u64 {
            range
                .map(|il| self.row_bytes.get(il).copied().unwrap_or(0))
                .sum()
        };
        let links: Vec<Arc<ChildLink>> = self
            .links()
            .into_iter()
            .chain(self.standby_links())
            .collect();
        let found = self.found.lock().unwrap().0.clone();
        let seen = self.prefills.lock().unwrap();
        let generation = self.top.read().unwrap().1;
        let own = self.settings.capacity.setups[0].clone();
        let mut setups = vec![own.clone()];
        let mut candidates = Vec::new();
        if self.fits_alone && !super::fetch::incomplete() {
            candidates.push(Candidate {
                addr: None,
                name: self.settings.name.clone(),
                rate: own.prompt_rate,
                fixed_ms: own.prompt_fixed_ms,
                observed: seen.nodes.get(&self.settings.name).copied(),
            });
        }
        // The plan's direct workers: their own names and links.
        let mut direct: Vec<(String, Option<(Duration, f64)>)> = Vec::new();
        let mut branches = Vec::new();
        for stage in pipeline.stages() {
            let addr = stage.name();
            let info = links.iter().find(|l| l.addr == addr).and_then(|l| l.info());
            let link = info.as_ref().and_then(|i| i.link);
            branches.push(Branch {
                addr: addr.clone(),
                row_bytes: rows_of(stage.layers()),
                link,
                takes_rows: stage.takes_rows(),
            });
            let Some(info) = info else { continue };
            let theirs = info.capacity.setups.first().cloned().unwrap_or_default();
            let name = if theirs.name.is_empty() {
                addr.clone()
            } else {
                theirs.name.clone()
            };
            direct.push((name.clone(), link));
            setups.extend(info.capacity.setups.iter().cloned());
            let whole = found.iter().any(|f| f.node == info.node && f.whole);
            if whole && stage.takes_rows() {
                candidates.push(Candidate {
                    addr: Some(addr),
                    observed: seen.nodes.get(&name).copied(),
                    name,
                    rate: theirs.prompt_rate,
                    fixed_ms: theirs.prompt_fixed_ms,
                });
            }
        }
        let nodes = self
            .plan()
            .iter()
            .filter(|e| e.layer_start < e.layer_end)
            .map(|e| {
                let setup = setups.iter().find(|s| s.name == e.node);
                TreeNode {
                    bytes: plan::bytes_of(
                        &(e.layer_start as usize..e.layer_end as usize),
                        &self.layer_bytes,
                    ),
                    rate: setup.map_or(0.0, |s| s.prompt_rate),
                    fixed_ms: setup.map_or(0.0, |s| s.prompt_fixed_ms),
                    link: direct
                        .iter()
                        .find(|(name, _)| *name == e.node)
                        .and_then(|(_, link)| *link),
                }
            })
            .collect();
        let hidden_bytes = self.hidden_bytes();
        let width = super::pipeline::sub_chunk();
        super::prefill::Inputs {
            tree_tokens,
            all_tokens,
            part: super::prefill::width(),
            pipelined: width > 0 && super::pipeline::pipelining(),
            hidden_bytes,
            total_bytes: self.layer_bytes.iter().sum(),
            nodes,
            local_row_bytes: rows_of(pipeline.local_layers()),
            branches,
            candidates,
            widest_layer: self.row_bytes.iter().copied().max().unwrap_or(0),
            observed_tree: seen
                .tree
                .filter(|(_, at)| *at == generation)
                .map(|(rate, _)| rate),
            refused: self.unmovable.clone(),
        }
    }

    /// This node and every node below it that it is connected to.
    fn subtree_ids(&self) -> Vec<String> {
        let mut ids = vec![self.id.clone()];
        for link in &self.links() {
            if let Some(info) = link.info() {
                ids.extend(info.subtree);
            }
        }
        ids
    }

    /// One entry per node with a metrics address: me first, then my
    /// children's cards, each with the branch (`via`) it came through;
    /// built fresh, so a lost child vanishes.
    fn discovery_entries(&self) -> Vec<(DiscoveryRecord, Option<String>)> {
        let mut out = Vec::new();
        let record = self.self_record();
        if record.metrics_port != 0 {
            out.push((record, None));
        }
        for link in &self.links() {
            let Some(info) = link.info() else {
                continue;
            };
            let mut records = info.discovery.iter();
            // First record = the child itself; rewrite its `worker` with the
            // address I dialed, matching `worker_up{worker=...}` exactly.
            if let Some(record) = records.next()
                && record.metrics_port != 0
            {
                let mut first = record.clone();
                first.worker = link.addr.clone();
                out.push((first, Some(link.addr.clone())));
            }
            for record in records {
                // No metrics listener behind it: Prometheus could never
                // reach it.
                if record.metrics_port == 0 {
                    continue;
                }
                out.push((record.clone(), Some(link.addr.clone())));
            }
        }
        out
    }

    fn discovery_metrics(&self) -> String {
        let mut out = String::from(
            "# HELP orangu_server_worker_node_info A node in this worker tree and where its metrics page lives.\n\
                # TYPE orangu_server_worker_node_info gauge\n",
        );
        for (record, via) in self.discovery_entries() {
            out += &node_info_line(&record, via.as_deref());
        }
        out
    }

    /// This node's own card, always present (address may be empty) its
    /// top spot in the list marks it as this parent's direct child.
    fn self_record(&self) -> DiscoveryRecord {
        let (metrics_host, metrics_port) =
            self.settings.metrics_endpoint.clone().unwrap_or_default();
        DiscoveryRecord {
            node: self.settings.name.clone(),
            // The address the system actually bound, not the settings'
            // `:0`/`0.0.0.0`, which nobody can dial.
            worker: self
                .local_addr
                .map(|addr| addr.to_string())
                .unwrap_or_default(),
            metrics_host,
            metrics_port,
        }
    }

    /// What this handshake sends up: my card first, then the connected
    /// children's, with each child's `worker` rewritten to the dialed
    /// address; empty when the parent does not read the list.
    fn discovery_for_parent(&self, parent_features: u64) -> Vec<DiscoveryRecord> {
        if !parent_reads_discovery(parent_features) {
            return Vec::new();
        }
        let mut out = vec![self.self_record()];
        for link in &self.links() {
            let Some(info) = link.info() else {
                continue;
            };
            let mut records = info.discovery.iter();
            if let Some(record) = records.next() {
                let mut first = record.clone();
                first.worker = link.addr.clone();
                out.push(first);
            }
            out.extend(records.cloned());
        }
        out
    }

    fn subtree_capacity(&self) -> Capacity {
        let mut capacity = self.settings.capacity.clone();
        capacity.subtree_nodes = 1;
        capacity.subtree_device_bytes = capacity.device_bytes;
        capacity.subtree_host_bytes = capacity.host_bytes;
        capacity.subtree_budget_bytes = capacity.budget_bytes;
        for info in self.links().iter().filter_map(|link| link.info()) {
            capacity.subtree_nodes += info.capacity.subtree_nodes;
            capacity.subtree_device_bytes += info.capacity.subtree_device_bytes;
            capacity.subtree_host_bytes += info.capacity.subtree_host_bytes;
            capacity.subtree_budget_bytes += info.capacity.subtree_budget_bytes;
            capacity.setups.extend(info.capacity.setups);
        }
        capacity
    }

    /// This node's layers and each worker's share of `range`, in the order of
    /// `candidates`: by speed when every node below has measured it, each
    /// share up to what its memory holds; by memory otherwise. Also says
    /// which it followed.
    fn cut(
        &self,
        range: &Range<usize>,
        own_budget: u64,
        candidates: &[(Arc<ChildLink>, ChildInfo)],
    ) -> (Range<usize>, Vec<Range<usize>>, &'static str) {
        let budgets: Vec<u64> = candidates
            .iter()
            .map(|(_, info)| plan::subtree_budget(&info.capacity))
            .collect();
        let mut weights = vec![own_budget];
        weights.extend(&budgets);
        let mut rates = vec![self.rate(&self.settings.capacity.setups[..1])];
        rates.extend(
            candidates
                .iter()
                .map(|(_, info)| self.rate(&info.capacity.setups)),
        );
        let by_speed = if self.settings.shares == Shares::Memory {
            None
        } else {
            plan::by_speed(&weights, &rates, plan::bytes_of(range, &self.layer_bytes))
        };
        let by = match (&by_speed, self.settings.shares) {
            (Some(_), Shares::Prompt) => "prompt speed",
            (Some(_), _) => "decode speed",
            (None, _) => "memory",
        };
        if let Some(by_speed) = by_speed {
            weights = by_speed;
        }
        let (local, shares) = plan::plan_shares(
            range.clone(),
            &self.layer_bytes,
            self.settings.local_layers,
            // A budget of nothing still takes a share when every budget
            // is nothing — a tree of nodes that did not say.
            weights[0].max(1),
            &weights[1..].iter().map(|b| (*b).max(1)).collect::<Vec<_>>(),
            &|at| self.model.split_allowed(at),
        );
        (local, shares, by)
    }

    /// `candidates` in the order whose plan runs a decode step soonest, each
    /// node's layers at its measured speed. A model may not let its layers
    /// be cut anywhere (Gemma 4's last layers read an earlier layer's KV
    /// cache), and what cannot be divided goes to whichever worker's share
    /// starts at the last cut — so which worker that is depends on the
    /// order. Every order of up to six workers is tried, of more the
    /// configured one and both by speed. The configured order when a worker
    /// did not measure its speed, or shares follow memory. An order saving
    /// less than [`WORTH_MOVING`] loses to the plan's, or at the first plan to
    /// the configured one, so that a restart does not move layers either.
    fn best_order(
        &self,
        range: &Range<usize>,
        own_budget: u64,
        candidates: Vec<(Arc<ChildLink>, ChildInfo)>,
    ) -> Vec<(Arc<ChildLink>, ChildInfo)> {
        let own = self.rate(&self.settings.capacity.setups[..1]);
        let rates: Vec<f64> = candidates
            .iter()
            .map(|(_, info)| self.rate(&info.capacity.setups))
            .collect();
        if self.settings.shares == Shares::Memory
            || candidates.len() < 2
            || own <= 0.0
            || rates.iter().any(|r| *r <= 0.0)
        {
            return candidates;
        }
        let bytes = |layers: &Range<usize>| plan::bytes_of(layers, &self.layer_bytes) as f64;
        let cost = |order: &[usize]| {
            let ordered: Vec<_> = order.iter().map(|i| candidates[*i].clone()).collect();
            let (local, shares, _) = self.cut(range, own_budget, &ordered);
            bytes(&local) / own
                + shares
                    .iter()
                    .zip(order)
                    .map(|(share, i)| bytes(share) / rates[*i])
                    .sum::<f64>()
        };
        let best = plan::best_order(&rates, &cost);
        // The order the plan has, unless another saves enough to pay for moving
        // layers: two workers of about the same speed stay where they are.
        let used = self.used.lock().unwrap();
        let position = |i: &usize| {
            used.iter()
                .position(|u| *u == candidates[*i].0.addr)
                .unwrap_or(usize::MAX)
        };
        let mut current: Vec<usize> = (0..candidates.len()).collect();
        current.sort_by_key(position);
        drop(used);
        let order = if cost(&best) < cost(&current) * (1.0 - WORTH_MOVING) {
            best
        } else {
            current
        };
        order.into_iter().map(|i| candidates[i].clone()).collect()
    }

    /// The speed a share over `setups` — a node, or a worker and every node
    /// below it — runs at, by `[workers].shares`: their rates added up, as
    /// each takes a part in proportion. `0` when one of them did not measure.
    fn rate(&self, setups: &[NodeSetup]) -> f64 {
        let spare = setups.first().map_or(1.0, |first| self.spare(&first.name));
        spare * self.busy_rate(setups)
    }

    /// The share of a node's speed that is free, by its load: `1 − load`,
    /// never under [`MIN_SPARE`], so a busy node keeps a small share rather
    /// than none. `1.0` off a channel.
    fn spare(&self, name: &str) -> f64 {
        if self.settings.channel.is_none() {
            return 1.0;
        }
        let load = if *name == self.settings.name {
            super::load::busy()
        } else {
            match self.found.lock().unwrap().0.iter().find(|f| f.name == name) {
                Some(found) => found.load,
                None => 0.0,
            }
        };
        (1.0 - f64::from(load)).max(MIN_SPARE)
    }

    /// A node's speed at what shares follow, its load aside.
    fn busy_rate(&self, setups: &[NodeSetup]) -> f64 {
        if self.settings.shares == Shares::Decode
            && let Some(first) = setups.first()
            && let Some(observed) = self
                .observed_now
                .lock()
                .unwrap()
                .as_ref()
                .and_then(|rates| rates.get(&first.name).copied())
        {
            return observed;
        }
        let rate = |s: &NodeSetup| match self.settings.shares {
            Shares::Decode => s.decode_rate as f64,
            Shares::Prompt => s.prompt_rate as f64,
            Shares::Memory => 0.0,
        };
        if setups.is_empty() || setups.iter().any(|s| rate(s) <= 0.0) {
            return 0.0;
        }
        setups.iter().map(rate).sum()
    }

    /// What this node has for layers. The top-level node also holds the
    /// embedding and the output head, which no layer range counts: its budget
    /// for layers is what they leave.
    fn own_budget(&self, top: bool) -> u64 {
        if top {
            self.settings
                .capacity
                .budget_bytes
                .saturating_sub(self.non_layer_bytes)
        } else {
            self.settings.capacity.budget_bytes
        }
    }

    /// A top-level plan goes by the speeds requests found once every node has
    /// one — a measurement at start can misjudge a model whose steps read
    /// different weights each token (a mixture's experts) — and remembers what
    /// it went by.
    fn note_rates(&self, candidates: &[(Arc<ChildLink>, ChildInfo)]) {
        let observed = self.observed.lock().unwrap();
        let mut names = vec![self.settings.name.clone()];
        names.extend(
            candidates
                .iter()
                .filter_map(|(_, info)| info.capacity.setups.first().map(|s| s.name.clone())),
        );
        *self.observed_now.lock().unwrap() = names
            .iter()
            .all(|name| observed.contains_key(name))
            .then(|| names.iter().map(|n| (n.clone(), observed[n].0)).collect());
        drop(observed);
        let mut planned = HashMap::new();
        planned.insert(
            self.settings.name.clone(),
            self.rate(&self.settings.capacity.setups[..1]),
        );
        for (_, info) in candidates {
            if let Some(first) = info.capacity.setups.first() {
                planned.insert(first.name.clone(), self.rate(&info.capacity.setups));
            }
        }
        *self.planned_rates.lock().unwrap() = planned;
        self.observed_since_plan.store(0, Ordering::Relaxed);
    }

    /// Whether the plan stands though the speeds requests found differ from
    /// what it was made by: the plan they call for saves too little of a decode
    /// step and of the slowest stage to pay for moving layers. Those speeds are
    /// then the ones it goes by. Plans again whenever a worker it does not use
    /// is there to take a share, or a standby stands in.
    fn plan_is_good_enough(&self) -> bool {
        let _planning = self.planning.lock().unwrap();
        let used = self.used.lock().unwrap().clone();
        if used.is_empty() || !self.standing_in.lock().unwrap().is_empty() {
            return false;
        }
        let mut candidates = Vec::new();
        for link in &self.links() {
            match link.info() {
                Some(_) if !used.contains(&link.addr) => return false,
                Some(info) => candidates.push((link.clone(), info)),
                None if used.contains(&link.addr) => return false,
                None => {}
            }
        }
        if candidates.len() != used.len() {
            return false;
        }
        candidates.sort_by_key(|(link, _)| used.iter().position(|u| *u == link.addr));
        let entries = self.plan();
        let range = 0..self.model.config().n_layer;
        let own_budget = self.own_budget(true);
        self.note_rates(&candidates);

        // What a plan costs: a decode step through every stage, and its slowest.
        let own = self.rate(&self.settings.capacity.setups[..1]);
        let costs = |local: &Range<usize>,
                     shares: &[Range<usize>],
                     nodes: &[(Arc<ChildLink>, ChildInfo)]| {
            let mut times = vec![plan::bytes_of(local, &self.layer_bytes) as f64 / own];
            for (share, (_, info)) in shares.iter().zip(nodes) {
                times.push(
                    plan::bytes_of(share, &self.layer_bytes) as f64
                        / self.rate(&info.capacity.setups),
                );
            }
            (
                times.iter().sum::<f64>(),
                times.iter().fold(0.0f64, |a, b| a.max(*b)),
            )
        };
        if own <= 0.0
            || candidates
                .iter()
                .any(|(_, info)| self.rate(&info.capacity.setups) <= 0.0)
        {
            return false;
        }
        let local_now = entries
            .iter()
            .find(|e| e.node == self.settings.name)
            .map_or(0..0, |e| e.layer_start as usize..e.layer_end as usize);
        let mut shares_now = Vec::new();
        for (_, info) in &candidates {
            let below = |e: &&PlanEntry| info.capacity.setups.iter().any(|s| s.name == e.node);
            let start = entries.iter().filter(below).map(|e| e.layer_start).min();
            let end = entries.iter().filter(below).map(|e| e.layer_end).max();
            let (Some(start), Some(end)) = (start, end) else {
                return false;
            };
            shares_now.push(start as usize..end as usize);
        }
        let now = costs(&local_now, &shares_now, &candidates);

        let proposed = self.best_order(&range, own_budget, candidates);
        let (local, shares, _) = self.cut(&range, own_budget, &proposed);
        let then = costs(&local, &shares, &proposed);
        let saves = |now: f64, then: f64| then < now * (1.0 - WORTH_MOVING);
        !saves(now.0, then.0) && !saves(now.1, then.1)
    }

    /// Plans `range` over this node and its workers, reached with `path` (the
    /// ids from the root down to this node), assigning each its share. A
    /// worker that cannot be reached, refuses, or does not have the model is
    /// left out, and the rest planned again without it.
    fn plan_subtree(&self, range: Range<usize>, path: &[String], terms: Terms) -> Result<Planned> {
        let secret = self.settings.secret.as_deref();
        let mut candidates: Vec<(Arc<ChildLink>, ChildInfo)> = Vec::new();
        // Standbys filling in for workers that do not answer:
        // `(missing worker, standby)`.
        let mut standing: Vec<(String, String)> = Vec::new();
        if path.len() < MAX_DEPTH {
            let mut missing = Vec::new();
            for link in &self.links() {
                if let Some(left) = link.out_for() {
                    log::debug!(
                        "orangu-server: worker {} left out for {} s more",
                        link.addr,
                        left.as_secs()
                    );
                    missing.push(link.addr.clone());
                    continue;
                }
                match link.ensure_connected(secret, path) {
                    Ok(info) => candidates.push((link.clone(), info)),
                    // Said once per outage: a worker that stays away is
                    // tried again every `readmit`, and the log should not
                    // fill with it.
                    Err(error) if link.note_unreachable() => {
                        log::warn!("orangu-server: worker left out: {error}");
                        missing.push(link.addr.clone());
                    }
                    Err(error) => {
                        log::debug!("orangu-server: worker still out: {error}");
                        missing.push(link.addr.clone());
                    }
                }
            }
            let spare_links = self.standby_links();
            let mut spares = spare_links.iter();
            for lost in missing {
                let mut filled = false;
                for link in spares.by_ref() {
                    if link.out_for().is_some() {
                        continue;
                    }
                    match link.ensure_connected(secret, path) {
                        Ok(info) => {
                            candidates.push((link.clone(), info));
                            standing.push((lost.clone(), link.addr.clone()));
                            filled = true;
                            break;
                        }
                        Err(error) => log::warn!("orangu-server: standby left out: {error}"),
                    }
                }
                // No standby held: a free node of the channel, if any.
                if !filled
                    && path.len() == 1
                    && let Some(link) = self.claim_spare()
                {
                    match link.ensure_connected(secret, path) {
                        Ok(info) => {
                            candidates.push((link.clone(), info));
                            standing.push((lost, link.addr.clone()));
                        }
                        Err(error) => log::warn!("orangu-server: standby left out: {error}"),
                    }
                }
            }
        }
        // A worker with another model is left out before anything is cut —
        // unless it switches to this node's model when assigned it. Said once
        // for as long as the reason stands; the link stays up, so a worker
        // that changes its model is seen when it reconnects.
        candidates.retain(|(link, info)| {
            let Some(model) = &info.model else {
                return true;
            };
            if model.header_hash == self.identity.header_hash || info.features & FEATURE_SWITCH != 0
            {
                return true;
            }
            let reason = format!(
                "{} serves {}, not {}, and cannot switch to it ([web].reexec is off there)",
                link.addr,
                named(model),
                named(&self.identity)
            );
            if link.note_left_out(&reason) {
                log::warn!("orangu-server: worker left out: {reason}");
            }
            false
        });
        let top = path.len() == 1;
        let own_budget = self.own_budget(top);
        if top {
            self.note_rates(&candidates);
        }
        candidates = self.best_order(&range, own_budget, candidates);
        // The split a claim's saved prefixes were made on ([`Self::claim`]).
        let offered = if top {
            self.saved_plan.lock().unwrap().clone()
        } else {
            None
        };
        let mut said = false;
        loop {
            let saved = offered.as_ref().and_then(|manifest| {
                match self.saved_cut(manifest, &range, own_budget, &candidates) {
                    Ok(cut) => Some(cut),
                    Err(why) => {
                        if !std::mem::replace(&mut said, true) {
                            log::info!(
                                "orangu-server: workers: the split the saved prefixes were made                                  on does not stand ({why}); cutting afresh"
                            );
                        }
                        None
                    }
                }
            });
            let kept = saved.is_some();
            let (local, shares, by) = match saved {
                Some((ordered, local, shares)) => {
                    candidates = ordered;
                    (local, shares, "the saved split")
                }
                None => self.cut(&range, own_budget, &candidates),
            };
            *self.shares_by.lock().unwrap() = by;
            let mut entries = vec![PlanEntry {
                node: self.settings.name.clone(),
                layer_start: local.start as u32,
                layer_end: local.end as u32,
            }];
            let mut stages: Vec<Box<dyn Stage>> = Vec::new();
            let mut used = Vec::new();
            let mut failed = None;
            for (i, ((link, _), share)) in candidates.iter().zip(&shares).enumerate() {
                if share.is_empty() {
                    continue;
                }
                match self.assign(link, share.clone(), terms) {
                    Ok(plan) => {
                        link.clear_left_out();
                        entries.extend(plan);
                        used.push(link.addr.clone());
                        stages.push(self.stage_for(link, share.clone(), path, terms));
                    }
                    Err(error) => {
                        // Once for as long as the reason stands: a worker
                        // that refuses is asked again at every plan.
                        if link.note_left_out(&error.to_string()) {
                            log::warn!("orangu-server: worker left out: {error}");
                        } else {
                            log::debug!("orangu-server: worker still left out: {error}");
                        }
                        failed = Some(i);
                        break;
                    }
                }
            }
            match failed {
                Some(i) => {
                    let (link, _) = candidates.remove(i);
                    link.disconnect();
                }
                None => {
                    let budgets: Vec<u64> = candidates
                        .iter()
                        .map(|(_, info)| plan::subtree_budget(&info.capacity))
                        .collect();
                    self.report_fit(&local, own_budget, &candidates, &shares, &budgets);
                    let pipeline = LayerPipeline::new(self.model.clone(), local, stages)?
                        .named(&self.settings.name);
                    standing.retain(|(_, standby)| used.contains(standby));
                    if top {
                        self.saved_split_kept.store(kept, Ordering::Release);
                    }
                    if !standing.is_empty() {
                        log::warn!(
                            "orangu-server: standing in for workers that do not answer: {}",
                            standing
                                .iter()
                                .map(|(lost, standby)| format!("{standby} for {lost}"))
                                .collect::<Vec<_>>()
                                .join(", ")
                        );
                    }
                    return Ok(Planned {
                        pipeline: Arc::new(pipeline),
                        entries,
                        used,
                        standing,
                    });
                }
            }
        }
    }

    /// The stage that sends `layers` to the worker behind `link`: guarded
    /// by a standby when this node has any (`super::guard`).
    fn stage_for(
        &self,
        link: &Arc<ChildLink>,
        layers: Range<usize>,
        path: &[String],
        terms: Terms,
    ) -> Box<dyn Stage> {
        let stage = MessageStage::over(link.clone(), layers.clone(), terms.activations);
        if self.standby_links().is_empty() {
            return Box::new(stage);
        }
        let node = self.this.get().cloned().unwrap_or_default();
        let path = path.to_vec();
        let take: TakeStandby = Box::new(move |layers, lost| {
            let node = node
                .upgrade()
                .ok_or_else(|| anyhow::anyhow!("the node has stopped"))?;
            node.take_standby(layers, &path, terms, lost)
        });
        let width = self.model.config().n_embd;
        match GuardedStage::new(Arc::new(stage), width, take) {
            Some(guarded) => Box::new(guarded),
            None => {
                log::warn!(
                    "orangu-server: [workers].standby needs {}-wide rows to be whole q8_0 \
                     blocks; {} is not guarded",
                    width,
                    link.addr
                );
                Box::new(MessageStage::over(link.clone(), layers, terms.activations))
            }
        }
    }

    /// Assigns `layers` to the first free standby that answers and holds
    /// the same model, in place of the lost worker `lost`, and records it in
    /// the plan. The stage returned sends to it.
    fn take_standby(
        &self,
        layers: Range<usize>,
        path: &[String],
        terms: Terms,
        lost: &str,
    ) -> Result<Arc<dyn EncodedStage>> {
        let secret = self.settings.secret.as_deref();
        let mut standing = self.standing_in.lock().unwrap();
        for link in &self.standby_links() {
            if standing.iter().any(|(_, standby)| *standby == link.addr) || link.out_for().is_some()
            {
                continue;
            }
            if let Err(error) = link.ensure_connected(secret, path) {
                log::warn!("orangu-server: standby left out: {error}");
                continue;
            }
            let plan = match self.assign(link, layers.clone(), terms) {
                Ok(plan) => plan,
                Err(error) => {
                    log::warn!("orangu-server: standby left out: {error}");
                    link.disconnect();
                    continue;
                }
            };
            standing.push((lost.to_string(), link.addr.clone()));
            self.metrics.takeovers.fetch_add(1, Ordering::Relaxed);
            for used in self.used.lock().unwrap().iter_mut() {
                if used == lost {
                    *used = link.addr.clone();
                }
            }
            let mut entries = self.plan.lock().unwrap();
            entries.retain(|e| {
                e.node == self.settings.name
                    || (e.layer_end as usize) <= layers.start
                    || (e.layer_start as usize) >= layers.end
            });
            entries.extend(plan);
            entries.sort_by_key(|e| e.layer_start);
            log::info!("orangu-server: workers plan: {}", describe(&entries));
            return Ok(Arc::new(MessageStage::over(
                link.clone(),
                layers,
                terms.activations,
            )));
        }
        anyhow::bail!(
            "no standby is free to take layers {}..{}",
            layers.start,
            layers.end
        )
    }

    /// Records the standbys a new plan uses; any other that stood in is
    /// let go, and serves its own API again.
    fn set_standing(&self, standing: Vec<(String, String)>) {
        let old = std::mem::replace(&mut *self.standing_in.lock().unwrap(), standing.clone());
        for (_, standby) in old {
            if standing.iter().any(|(_, kept)| *kept == standby) {
                continue;
            }
            if let Some(link) = self.standby_links().iter().find(|l| l.addr == standby) {
                link.disconnect();
            }
        }
    }

    /// Warns about every share of a plan that is more than its node (or
    /// worker subtree) said it can hold. The plan stands — a share too
    /// large for memory still runs, paged in and out — but the operator
    /// should know that the tree is short of room, and where.
    fn report_fit(
        &self,
        local: &Range<usize>,
        own_budget: u64,
        candidates: &[(Arc<ChildLink>, ChildInfo)],
        shares: &[Range<usize>],
        budgets: &[u64],
    ) {
        let mut over = Vec::new();
        let local_bytes = plan::bytes_of(local, &self.layer_bytes);
        if local_bytes > own_budget {
            over.push(format!(
                "{} (this node) gets {} for {}",
                self.settings.name,
                orangu::format::format_bytes(local_bytes),
                orangu::format::format_bytes(own_budget)
            ));
        }
        for (((link, _), share), budget) in candidates.iter().zip(shares).zip(budgets) {
            let bytes = plan::bytes_of(share, &self.layer_bytes);
            if bytes > *budget {
                over.push(format!(
                    "{} gets {} for {}",
                    link.addr,
                    orangu::format::format_bytes(bytes),
                    orangu::format::format_bytes(*budget)
                ));
            }
        }
        if !over.is_empty() {
            log::warn!(
                "orangu-server: the workers are short of memory for this model — {}; the \
                 layers over budget will be paged in from disk as they run",
                over.join(", ")
            );
        }
    }

    /// Assigns `layers` to the worker behind `link`, and checks that what
    /// comes back is the same model.
    fn assign(
        &self,
        link: &ChildLink,
        layers: Range<usize>,
        terms: Terms,
    ) -> Result<Vec<PlanEntry>, WorkerError> {
        let expected = identity::model_identity(
            &self.loaded,
            &self.settings.label,
            &self.settings.quant,
            layers.clone(),
        );
        // An assignment may fetch its layers first (`super::fetch`): gigabytes,
        // at whatever the worker's link to this node, another node or the Hub
        // gives.
        self.assigning.fetch_add(1, Ordering::AcqRel);
        let reply = link.request_within(
            &Message::Assign(Assign {
                model: self.settings.label.clone(),
                identity: expected.clone(),
                layer_start: layers.start as u32,
                layer_end: layers.end as u32,
                n_ctx: terms.n_ctx as u32,
                kv_type: String::new(),
                slots: terms.slots as u32,
                activations: terms.activations,
                prompt_weights: Some(terms.weights.bits()),
            }),
            ASSIGN_TIMEOUT.max(self.settings.timeout),
        );
        self.assigning.fetch_sub(1, Ordering::AcqRel);
        let reply = reply?;
        let with_path = |mut error: WorkerError| {
            if error.path.first() != Some(&link.addr) {
                error.path.insert(0, link.addr.clone());
            }
            error
        };
        match reply {
            Message::AssignAck { identity, plan } => {
                // Checked here as well as there: a parent does not take a
                // worker's word for which model it has.
                identity::check(&expected, &identity, &layers).map_err(with_path)?;
                Ok(plan)
            }
            Message::Error(error) => Err(with_path(error)),
            other => Err(with_path(WorkerError::new(
                ErrorCode::Internal,
                format!("unexpected answer to an assignment: {other:?}"),
            ))),
        }
    }

    /// Plans this node as top-level: every layer, over itself and its
    /// workers.
    pub fn plan_top(&self) {
        let mut last = self.planning.lock().unwrap();
        self.plan_top_locked();
        *last = Instant::now();
    }

    fn plan_top_locked(&self) {
        if !self.delegates() || self.is_assigned() || self.alone.load(Ordering::Acquire) {
            return;
        }
        let n_layer = self.model.config().n_layer;
        let terms = Terms {
            n_ctx: self.settings.n_ctx,
            slots: self.settings.slots,
            activations: self.settings.activations,
            weights: *self
                .decision
                .get_or_init(|| prompt_weights::decide(&self.loaded, self.settings.prompt_weights)),
        };
        self.metrics.plans.fetch_add(1, Ordering::Relaxed);
        let planned = self.plan_subtree(0..n_layer, std::slice::from_ref(&self.id), terms);
        let (pipeline, entries, used, standing) = match planned {
            Ok(p) => (p.pipeline, p.entries, p.used, p.standing),
            Err(error) => {
                log::warn!("orangu-server: planning the workers failed, serving alone: {error:#}");
                let local = LayerPipeline::new(self.model.clone(), 0..n_layer, Vec::new())
                    .expect("a splittable model runs every layer alone");
                let entry = PlanEntry {
                    node: self.settings.name.clone(),
                    layer_start: 0,
                    layer_end: n_layer as u32,
                };
                (Arc::new(local), vec![entry], Vec::new(), Vec::new())
            }
        };
        if let Some(switch) = self.serving.get() {
            // No worker in the plan: the model's own paths serve.
            switch.set_chosen(used.is_empty());
        }
        if *self.plan.lock().unwrap() != entries {
            log::info!("orangu-server: workers plan: {}", describe(&entries));
        }
        let head_on_last = self.decides_head_on_last(&entries, &used);
        if head_on_last != self.head_on_last.swap(head_on_last, Ordering::AcqRel) {
            log::info!(
                "orangu-server: workers: the output head runs on {}",
                if head_on_last {
                    "the node with the final layer"
                } else {
                    "this node"
                }
            );
        }
        self.release_unused(&pipeline.local_layers(), true);
        self.copy_prompt_weights(&pipeline.local_layers(), terms.weights);
        let mut top = self.top.write().unwrap();
        *top = (pipeline, top.1 + 1);
        drop(top);
        self.set_standing(standing);
        *self.plan.lock().unwrap() = entries;
        *self.used.lock().unwrap() = used;
    }

    /// The slowest link to the workers `used`: the longest round trip and
    /// the least bandwidth among them. `None` when none was timed.
    fn slowest_link(&self, used: &[String]) -> Option<(Duration, f64)> {
        self.links()
            .iter()
            .chain(&self.standby_links())
            .filter(|link| used.contains(&link.addr))
            .filter_map(|link| link.info().and_then(|info| info.link))
            .reduce(|a, b| (a.0.max(b.0), a.1.min(b.1)))
    }

    /// Whether the node running the model's final layer applies the output
    /// head, by `[workers].head`: only when that node is a worker of
    /// this plan and every worker used can send logits back — and, with
    /// `auto`, when reading the head there at its measured speed and sending
    /// the logits back over the slowest link takes less than reading it
    /// here: the head is a matrix as wide as the vocabulary, read once a
    /// token.
    fn decides_head_on_last(&self, entries: &[PlanEntry], used: &[String]) -> bool {
        let n_layer = self.model.config().n_layer as u32;
        let Some(last) = entries
            .iter()
            .find(|e| e.layer_start < e.layer_end && e.layer_end == n_layer)
        else {
            return false;
        };
        if self.settings.head == HeadOn::Top || last.node == self.settings.name {
            return false;
        }
        let head = self
            .links()
            .iter()
            .chain(&self.standby_links())
            .all(|link| {
                !used.contains(&link.addr)
                    || link
                        .info()
                        .is_some_and(|info| info.features & super::protocol::FEATURE_HEAD != 0)
            });
        if !head {
            return false;
        }
        if self.settings.head == HeadOn::Last {
            return true;
        }
        let mut setups = self.settings.capacity.setups.clone();
        for link in &self.links() {
            if let Some(info) = link.info() {
                setups.extend(info.capacity.setups);
            }
        }
        let rate = |name: &str| {
            setups
                .iter()
                .find(|s| s.name == name)
                .map(|s| s.decode_rate)
                .filter(|r| *r > 0.0)
        };
        let (Some(there), Some(here)) = (rate(&last.node), rate(&self.settings.name)) else {
            return false;
        };
        // Nanoseconds a token: the head read at each node's rate, and from
        // there its logits back over the slowest link, `f32` — a wide
        // vocabulary's megabyte a token matters on a slow link.
        let head = self.head_bytes as f64;
        let logits = 4.0 * self.model.config().n_vocab as f64;
        let back = self
            .slowest_link(used)
            .map_or(0.0, |(_, bandwidth)| logits / bandwidth * 1e9);
        head / (there as f64) + back < head / (here as f64)
    }

    fn set_alone(&self, alone: bool) -> Result<(), String> {
        if self.is_assigned() {
            return Err("this node is working for a parent".to_string());
        }
        if !self.delegates() {
            return Err(
                "this node has no workers, or its model cannot be split across them".to_string(),
            );
        }
        if super::fetch::incomplete() {
            return Err("this node has only part of the model".to_string());
        }
        if !alone {
            self.alone.store(false, Ordering::Release);
            self.plan_top();
            log::info!(
                "orangu-server: [workers] serving through the tree again: {}",
                describe(&self.plan())
            );
            return Ok(());
        }
        let mut last = self.planning.lock().unwrap();
        self.alone.store(true, Ordering::Release);
        let n_layer = self.model.config().n_layer;
        let local = LayerPipeline::new(self.model.clone(), 0..n_layer, Vec::new())
            .expect("a splittable model runs every layer alone");
        let mut top = self.top.write().unwrap();
        *top = (Arc::new(local), top.1 + 1);
        drop(top);
        // Let go: each worker drops its sessions and serves its own API.
        for link in self.links().iter().chain(&self.standby_links()) {
            link.disconnect();
        }
        self.set_standing(Vec::new());
        *self.plan.lock().unwrap() = vec![PlanEntry {
            node: self.settings.name.clone(),
            layer_start: 0,
            layer_end: n_layer as u32,
        }];
        self.used.lock().unwrap().clear();
        let weights = *self
            .decision
            .get_or_init(|| prompt_weights::decide(&self.loaded, self.settings.prompt_weights));
        self.copy_prompt_weights(&(0..n_layer), weights);
        *last = Instant::now();
        log::info!("orangu-server: [workers] serving alone; the workers are let go");
        Ok(())
    }

    /// Lets go of the weights this node's plan no longer runs: every layer
    /// outside `local`, and — unless `keep_head`: the top-level node, which
    /// embeds and samples, or the node with the final layer — the
    /// embedding and output head. The kernel faults them
    /// back if a later plan wants them.
    fn release_unused(&self, local: &Range<usize>, keep_head: bool) {
        let n_layer = self.model.config().n_layer;
        let released =
            self.loaded.release_tensors(|name| {
                match crate::engine::loader::block_index(name).filter(|il| *il < n_layer) {
                    Some(il) => local.contains(&il),
                    None => keep_head,
                }
            });
        log::debug!(
            "orangu-server: workers: released {} of weights outside layers {}..{}",
            orangu::format::format_bytes(released),
            local.start,
            local.end
        );
    }

    /// Builds the prompt-weight copies of the layers this node runs, by the
    /// top-level node's decision: a node copies only its own layers,
    /// and every node rounds as the top does — then runs the layers once
    /// ([`super::speed::warm`]). Kept while a plan leaves the layers and the
    /// decision as they were.
    fn copy_prompt_weights(&self, local: &Range<usize>, weights: Decision) {
        let mut copied = self.copied.lock().unwrap();
        if copied.as_ref() == Some(&(local.clone(), weights)) {
            return;
        }
        prompt_weights::build_for_layers(
            &self.loaded,
            weights,
            local.clone(),
            self.model.config().n_layer,
            "as the top-level node decided",
        );
        // The layers are new to this node: run them once now, so that no
        // request pays for reading them in and preparing the device.
        if !local.is_empty() && self.model.supports_layer_split() {
            match super::speed::warm(self.model.as_ref(), local.clone()) {
                Ok(took) => log::info!(
                    "orangu-server: [workers] layers {}..{} warmed in {:.1} s",
                    local.start,
                    local.end,
                    took.as_secs_f64()
                ),
                Err(e) => log::warn!(
                    "orangu-server: [workers] could not warm layers {}..{}: {e:#}",
                    local.start,
                    local.end
                ),
            }
        }
        *copied = Some((local.clone(), weights));
    }

    /// A worker runs its parent's model: assigned another model than
    /// the one loaded, it answers that it is switching, and replaces itself
    /// with this binary serving `model` — found under `models` or fetched,
    /// whole or in range, the way a start resolves it — once the answer is
    /// on its way; the parent takes it back at a later plan. `None`, and the
    /// assignment is refused as before, when the parent names the model this
    /// node already serves (another file of it: switching would change
    /// nothing), when a switch to `model` already failed here, or when this
    /// process cannot load another model.
    fn switch_to(&self, model: &str) -> Option<Message> {
        if model.is_empty() || model == self.settings.label {
            return None;
        }
        if crate::reexec::refused_model().as_deref() == Some(model) {
            log::warn!(
                "orangu-server: the parent serves {model}, which this node could not load; \
                 serving {} and left out",
                self.settings.label
            );
            return None;
        }
        let Some(handover) = self.handover.get().cloned() else {
            log::warn!(
                "orangu-server: the parent serves {model}; this node cannot switch to it \
                 ([web].reexec is off)"
            );
            return None;
        };
        if !self.switching.swap(true, Ordering::AcqRel) {
            log::info!(
                "orangu-server: the parent serves {model}; switching to it from {}",
                self.settings.label
            );
            let target = model.to_string();
            let _ = std::thread::Builder::new()
                .name("orangu-workers-switch".to_string())
                .spawn(move || {
                    // The answer below reaches the parent first.
                    std::thread::sleep(Duration::from_millis(500));
                    let previous = handover.current_model().to_string();
                    let error = handover.exec(&target, handover.role(), Some(&previous));
                    log::error!("orangu-server: could not switch to the parent's model: {error:#}");
                });
        }
        Some(error(
            ErrorCode::Busy,
            format!(
                "switching to {model}, the parent's model, from {}; ask again once it is loaded",
                self.settings.label
            ),
        ))
    }

    /// Answers a parent's `Assign`.
    fn handle_assign(&self, assign: Assign, parent_path: &[String]) -> Message {
        let n_layer = self.model.config().n_layer;
        let layers = assign.layer_start as usize..assign.layer_end as usize;
        // Another model altogether: switch to it, before anything is
        // asked of its layers here.
        if !identity::same_layout(&self.loaded, &assign.identity)
            && let Some(switching) = self.switch_to(&assign.model)
        {
            return switching;
        }
        if !self.model.supports_layer_split() {
            return error(
                ErrorCode::Unsupported,
                format!(
                    "the {} architecture cannot be split across workers yet",
                    self.model.config().architecture
                ),
            );
        }
        if layers.is_empty() || layers.end > n_layer {
            return error(
                ErrorCode::Unsupported,
                format!("layers {layers:?} of a model with {n_layer}"),
            );
        }
        for at in [layers.start, layers.end] {
            if at != 0 && at != n_layer && !self.model.split_allowed(at) {
                return error(
                    ErrorCode::Unsupported,
                    format!(
                        "layers {layers:?}: this model cannot be cut before layer {at}, where a \
                         later layer reads an earlier one's KV cache"
                    ),
                );
            }
        }
        // A partly downloaded model fetches the layers first: the check
        // below reads them. The parent first, when it serves the file.
        if super::fetch::partial().is_some() {
            let parent = self.parent.lock().unwrap().as_ref().map(|p| p.node.clone());
            super::fetch::prefer(
                &self.found.lock().unwrap().0,
                parent.as_deref(),
                self.identity.header_hash,
                self.settings.secret.as_deref(),
                self.settings.tls.as_ref(),
            );
        }
        if let Err(e) = super::fetch::layers(&self.loaded, &layers) {
            log::warn!("orangu-server: could not fetch layers {layers:?}: {e:#}");
            return error(
                ErrorCode::Internal,
                format!("fetching layers {}..{}: {e:#}", layers.start, layers.end),
            );
        }
        let mut own = identity::model_identity(
            &self.loaded,
            &self.settings.label,
            &self.settings.quant,
            layers.clone(),
        );
        // Weights fetched in part that do not match a parent of the same
        // layout came wrong from their source: fetched again, from the next.
        if super::fetch::partial().is_some() && own.header_hash == assign.identity.header_hash {
            for _ in 0..super::fetch::REFETCHES {
                if identity::check(&assign.identity, &own, &layers).is_ok() {
                    break;
                }
                if let Err(e) = super::fetch::refetch(&self.loaded, &layers) {
                    log::warn!("orangu-server: could not fetch layers {layers:?} again: {e:#}");
                    break;
                }
                own = identity::model_identity(
                    &self.loaded,
                    &self.settings.label,
                    &self.settings.quant,
                    layers.clone(),
                );
            }
        }
        if let Err(error) = identity::check(&assign.identity, &own, &layers) {
            if let Some(switch) = self.switch_to(&assign.model) {
                return switch;
            }
            log::warn!("orangu-server: refused an assignment: {error}");
            return Message::Error(error);
        }
        let mut last = self.planning.lock().unwrap();
        let mut path = parent_path.to_vec();
        path.push(self.id.clone());
        let terms = Terms {
            n_ctx: assign.n_ctx as usize,
            slots: assign.slots.max(1) as usize,
            activations: assign.activations,
            // A parent that does not say: the file's weights, rather than a
            // measurement of layers this node may not have fetched.
            weights: assign
                .prompt_weights
                .map(Decision::from_bits)
                .unwrap_or_default(),
        };
        self.metrics.plans.fetch_add(1, Ordering::Relaxed);
        let (pipeline, entries, used, standing) =
            match self.plan_subtree(layers.clone(), &path, terms) {
                Ok(p) => (p.pipeline, p.entries, p.used, p.standing),
                Err(e) => return error(ErrorCode::Internal, format!("{e:#}")),
            };
        self.set_standing(standing);
        // A node with the final layer keeps the output head, which it may be
        // asked to apply.
        let local = pipeline.local_layers();
        self.release_unused(&local, !local.is_empty() && local.end == n_layer);
        self.copy_prompt_weights(&pipeline.local_layers(), terms.weights);
        let store = Arc::new(self.session_store(pipeline, terms));
        *self.assignment.lock().unwrap() = Some(Assignment {
            store,
            layers: layers.clone(),
            terms,
            path: path.clone(),
        });
        // The top-level plan's sessions are gone with it; bumped so any
        // that come back rebuild on a fresh one.
        let mut top = self.top.write().unwrap();
        top.1 += 1;
        drop(top);
        log::info!(
            "orangu-server: working for a parent on layers {}..{} (this API is off until it \
             lets go): {}",
            layers.start,
            layers.end,
            describe(&entries)
        );
        *self.plan.lock().unwrap() = entries.clone();
        *self.used.lock().unwrap() = used;
        *last = Instant::now();
        Message::AssignAck {
            identity: own,
            plan: entries,
        }
    }

    /// The parent went away: drop its sessions, and be top-level again.
    fn unassign(self: &Arc<Self>) {
        let released = self.assignment.lock().unwrap().take();
        if released.is_none() {
            return;
        }
        drop(released);
        log::info!("orangu-server: the parent let go; serving this node's own API again");
        self.plan.lock().unwrap().clear();
        self.used.lock().unwrap().clear();
        if self.delegates() {
            let node = self.clone();
            let _ = std::thread::Builder::new()
                .name("orangu-workers-plan".to_string())
                .spawn(move || node.plan_top());
        }
    }

    fn accept(self: Arc<Self>, listener: TcpListener) {
        for stream in listener.incoming() {
            if self.stop.load(Ordering::Relaxed) {
                break;
            }
            let Ok(stream) = stream else { continue };
            let node = self.clone();
            let _ = std::thread::Builder::new()
                .name("orangu-workers-parent".to_string())
                .spawn(move || node.serve_parent(stream));
        }
    }

    /// One connection from a would-be parent: the handshake, then its
    /// requests until it goes away.
    fn serve_parent(self: Arc<Self>, stream: TcpStream) {
        let connection = self.next_connection.fetch_add(1, Ordering::Relaxed);
        let _ = stream.set_nodelay(true);
        let _ = stream.set_read_timeout(Some(self.settings.timeout));
        let Ok(parent_handle) = stream.try_clone() else {
            return;
        };
        let split = match transport::server(self.settings.tls.as_ref(), stream) {
            Ok(split) => split,
            Err(error) => {
                log::warn!("orangu-server: refused a parent: TLS: {error:#}");
                return;
            }
        };
        let mut reader = BufReader::new(split.reader);
        let mut writer = split.writer;
        let Some(hello) = self.handshake(&mut reader, &mut *writer, connection, parent_handle)
        else {
            return;
        };
        // A parent that lost its power closes nothing, and a worker held by
        // one keeps its layers, its API off and every other parent away. A
        // live one pings ([`Self::ping_worker`]). Limited per read and per
        // write, so a large frame on a slow link is not cut short.
        let silence = self.settings.parent_silence;
        let _ = split.tcp.set_read_timeout(Some(silence));
        let _ = split.tcp.set_write_timeout(Some(silence));
        let tcp = split.tcp.clone();
        let writer = Arc::new(Mutex::new(writer));
        let answer = |writer: &Mutex<Box<dyn Write + Send>>, id: u64, reply: &Message| {
            let mut writer = writer.lock().unwrap();
            write_message(&mut **writer, id, reply).is_ok()
        };
        loop {
            let (id, message) = match read_frame(&mut reader) {
                Ok((id, message, _)) => {
                    if let Some(parent) = self.parent.lock().unwrap().as_mut()
                        && parent.connection == connection
                    {
                        parent.heard = Instant::now();
                    }
                    (id, message)
                }
                Err(error) => {
                    if timed_out(&error) {
                        log::warn!(
                            "orangu-server: nothing heard from the parent for {} s; it is taken \
                             for gone",
                            silence.as_secs()
                        );
                    }
                    break;
                }
            };
            if self.stop.load(Ordering::Relaxed) {
                break;
            }
            let reply = match message {
                // Planning is one thing at a time, and so are pings — which
                // is what keeps them answered while forwards run.
                Message::Assign(assign) => self.handle_assign(assign, &hello),
                Message::Ping { nonce } => Message::Pong { nonce },
                Message::Echo { data } => Message::Echo { data },
                message => {
                    let store = self
                        .assignment
                        .lock()
                        .unwrap()
                        .as_ref()
                        .map(|a| a.store.clone());
                    let Some(store) = store else {
                        if !answer(
                            &writer,
                            id,
                            &error(ErrorCode::UnknownSession, "no layers assigned here"),
                        ) {
                            break;
                        }
                        continue;
                    };
                    // Every forward on a thread of its own, answered when it
                    // is done: sequences overlap here as they do at the
                    // parent. One sequence's forwards never overlap each
                    // other — its parent waits for each before the next.
                    let writer = writer.clone();
                    let tcp = tcp.clone();
                    let forwards = self.forwards.clone();
                    forwards.fetch_add(1, Ordering::AcqRel);
                    let spawned = std::thread::Builder::new()
                        .name("orangu-workers-forward".to_string())
                        .spawn(move || {
                            struct Running(Arc<AtomicUsize>);
                            impl Drop for Running {
                                fn drop(&mut self) {
                                    self.0.fetch_sub(1, Ordering::AcqRel);
                                }
                            }
                            let _running = Running(forwards);
                            super::clock::hold();
                            // In the request pool, as a request at the top
                            // runs: parallel regions split from a pool
                            // worker's own deque rather than being handed
                            // in from outside (`engine::cpu_pools`).
                            let reply =
                                crate::engine::cpu_pools::for_request(|| store.handle(message));
                            super::clock::hold();
                            let mut writer = writer.lock().unwrap();
                            // Part of a frame may have gone: nothing after it
                            // could be read.
                            if write_message(&mut **writer, id, &reply).is_err() {
                                let _ = tcp.shutdown(std::net::Shutdown::Both);
                            }
                        });
                    if spawned.is_err() {
                        self.forwards.fetch_sub(1, Ordering::AcqRel);
                        break;
                    }
                    continue;
                }
            };
            if !answer(&writer, id, &reply) {
                break;
            }
        }
        let _ = split.tcp.shutdown(std::net::Shutdown::Both);
        let mut parent = self.parent.lock().unwrap();
        if parent.as_ref().is_some_and(|p| p.connection == connection) {
            *parent = None;
            drop(parent);
            self.unassign();
        }
    }

    /// The worker's half of the handshake. Returns the parent's path on
    /// success, having claimed this node for it.
    fn handshake(
        &self,
        reader: &mut BufReader<Box<dyn Read + Send>>,
        writer: &mut dyn Write,
        connection: u64,
        tcp: TcpStream,
    ) -> Option<Vec<String>> {
        let refuse = |writer: &mut dyn Write, error: WorkerError| {
            log::warn!("orangu-server: refused a parent: {error}");
            let _ = write_message(writer, 0, &Message::Error(error));
            None
        };
        let hello = match read_message(reader) {
            Ok(Message::Hello(hello)) => hello,
            // A node scanning for its channel: answered only on this one.
            Ok(Message::Probe { channel }) => {
                if let Some(own) = self.settings.channel.as_deref()
                    && super::channel::channel_hash(own, self.settings.secret.as_deref()) == channel
                {
                    let _ = write_message(
                        writer,
                        0,
                        &Message::ProbeAck {
                            node: self.id.clone(),
                            name: self.settings.name.clone(),
                            state: self.probe_state(),
                            load: self.load(),
                            whole: self.whole(),
                            api_port: self.settings.api_port,
                            api_tls: self.settings.api_tls,
                            model: self.identity.header_hash,
                            content: self.content_hash,
                            workers: self.workers_held(),
                            label: self.settings.label.clone(),
                            quant: self.settings.quant.clone(),
                            features: if self.served.is_some() {
                                FEATURE_FETCH
                            } else {
                                0
                            },
                            pace: super::protocol::Pace {
                                decode_rate: self.settings.capacity.setups[0].decode_rate,
                                decode_fixed_ms: self.settings.capacity.setups[0].decode_fixed_ms,
                                budget_bytes: self.own_budget(false),
                                step_ms: self.step_ms(),
                            },
                            demand: self.shared_demand(),
                            kind: Default::default(),
                            capabilities: Vec::new(),
                        },
                    );
                    self.note_prober(&tcp);
                }
                return None;
            }
            // A node fetching the model in part: served bytes of the file,
            // once it proves the secret, whoever claimed this node.
            Ok(Message::FetchHello { nonce }) => {
                let offer = self.served.as_ref().map(|file| super::share::Offer {
                    file,
                    model: self.identity.header_hash,
                    secret: self.settings.secret.as_deref(),
                });
                super::share::serve(reader, writer, nonce, offer, &self.fetch_pace, &|| {
                    self.assigning.load(Ordering::Acquire) == 0
                        && self.active.load(Ordering::Acquire)
                            + self.forwards.load(Ordering::Acquire)
                            > 0
                });
                return None;
            }
            // A node about to claim workers on this channel.
            Ok(Message::Lease { channel, node }) => {
                if let Some(own) = self.settings.channel.as_deref()
                    && super::channel::channel_hash(own, self.settings.secret.as_deref()) == channel
                {
                    let _ = write_message(writer, 0, &self.answer_lease(&node));
                }
                return None;
            }
            _ => return None,
        };
        if hello.version != PROTOCOL_VERSION {
            return refuse(
                writer,
                WorkerError::new(
                    ErrorCode::Unsupported,
                    format!(
                        "this node speaks worker protocol {PROTOCOL_VERSION}, the parent {}: run \
                         the same orangu-server release on every node",
                        hello.version
                    ),
                ),
            );
        }
        let parent_node = hello.path.last().cloned().unwrap_or_default();
        // Leading a tree of its own: a tree on a channel is one level deep.
        if self.leading.load(Ordering::Acquire) {
            return refuse(
                writer,
                WorkerError::new(ErrorCode::Busy, "leading workers of its own"),
            );
        }
        if let Some(parent) = self.parent.lock().unwrap().as_ref()
            && parent.node != parent_node
        {
            return refuse(
                writer,
                WorkerError::new(ErrorCode::Busy, "already working for another parent"),
            );
        }
        if let Err(error) = check_path(&hello.path, &self.subtree_ids()) {
            return refuse(writer, error);
        }
        let secret = self.settings.secret.as_deref();
        let nonce = auth::nonce();
        // Whether it would switch to a parent's other model (`switch_to`).
        let switches = self.handover.get().is_some();
        let ack = Message::HelloAck(Box::new(HelloAck {
            version: PROTOCOL_VERSION,
            features: FEATURES | if switches { FEATURE_SWITCH } else { 0 },
            nonce,
            proof: auth::worker_proof(secret, &hello.nonce, &nonce),
            node: self.id.clone(),
            subtree: self.subtree_ids(),
            capacity: self.subtree_capacity(),
            model: Some(self.identity.clone()),
            discovery: self.discovery_for_parent(hello.features),
            sessions: Some(self.session_terms()),
            // What this node holds on its disk, for a parent to know before
            // it asks for any of it.
            saved: match &self.prefix_files {
                Some(files) if hello.features & FEATURE_PREFIXES != 0 => files.list(),
                _ => Vec::new(),
            },
        }));
        write_message(writer, 0, &ack).ok()?;
        let proof = match read_message(reader) {
            Ok(Message::Auth { proof }) => proof,
            _ => return None,
        };
        if !auth::worker_accepts(secret, &hello.nonce, &nonce, &proof) {
            return refuse(
                writer,
                WorkerError::new(
                    ErrorCode::Unauthorized,
                    "the parent could not prove [workers].secret",
                ),
            );
        }
        // Promised to another node about to claim it.
        if let Some(holder) = self.lease_holder()
            && holder != parent_node
        {
            return refuse(
                writer,
                WorkerError::new(ErrorCode::Busy, format!("promised to {holder}")),
            );
        }
        {
            let mut parent = self.parent.lock().unwrap();
            // The same parent coming back — its old connection is dead —
            // takes over; anyone else is turned away.
            match parent.as_ref() {
                Some(p) if p.node != parent_node => {
                    drop(parent);
                    return refuse(
                        writer,
                        WorkerError::new(ErrorCode::Busy, "already working for another parent"),
                    );
                }
                Some(p) => {
                    let _ = p.stream.shutdown(std::net::Shutdown::Both);
                }
                None => {}
            }
            *parent = Some(Parent {
                connection,
                node: parent_node,
                stream: tcp,
                heard: Instant::now(),
                seen: None,
                missed: 0,
                suspect: false,
            });
        }
        *self.lease.lock().unwrap() = None;
        write_message(writer, 0, &Message::Ready).ok()?;
        Some(hello.path)
    }

    /// Pings a worker or standby while it is idle, which tells it this node is
    /// alive ([`NodeSettings::parent_silence`]) and this node that it is. On a
    /// thread per worker, away from [`Self::watch`], which plans: a ping waits
    /// for its answer, and a hung worker or a long plan would silence this node
    /// to the others.
    fn ping_worker(&self, link: &ChildLink) {
        while !self.stop.load(Ordering::Relaxed) {
            std::thread::sleep(self.settings.maintenance);
            // Let go of: the link is the next scan's to find again.
            let held = self
                .links()
                .iter()
                .chain(&self.standby_links())
                .any(|held| std::ptr::eq(Arc::as_ptr(held), link));
            if !held {
                break;
            }
            link.ping_if_idle();
        }
    }

    /// Reclaims forgotten sessions and — as top-level, between requests — takes
    /// back workers that were lost.
    fn watch(self: Arc<Self>) {
        while !self.stop.load(Ordering::Relaxed) {
            std::thread::sleep(self.settings.maintenance);
            let store = self
                .assignment
                .lock()
                .unwrap()
                .as_ref()
                .map(|a| a.store.clone());
            if let Some(store) = store {
                store.evict_idle(SESSION_IDLE);
                if self.delegates() && store.quiet(QUIET) && self.missing_and_due() {
                    self.replan_assignment();
                }
                continue;
            }
            if self.release_due() {
                self.release_workers(&format!(
                    "no request for {} s",
                    self.settings.release.as_secs()
                ));
                continue;
            }
            if self.yields_to_larger() {
                self.release_workers(
                    "another node leads a larger tree of this model; this one gives way",
                );
                continue;
            }
            if let Some(why) = self.tree_is_slower_than_alone() {
                self.release_workers(&format!("{why}; this node serves its model alone"));
                continue;
            }
            if !self.delegates() || self.alone.load(Ordering::Acquire) {
                continue;
            }
            let idle = self
                .delegating
                .get()
                .and_then(Weak::upgrade)
                .is_none_or(|model| model.quiet(QUIET));
            if idle && self.missing_and_due() {
                self.plan_top();
            } else if idle && let Some(why) = self.calibration_due() {
                log::info!(
                    "orangu-server: workers: planning again by the speeds requests found: {why}"
                );
                self.plan_top();
            }
        }
    }

    /// Whether the speeds requests found tell the plan apart from the speeds
    /// it was made by — some node more than 20% off, set against this
    /// node's own — after enough decode steps to tell ([`CALIBRATE_TOKENS`]).
    /// Says which, for the log. Only for shares by decode speed, and only
    /// when every node the plan uses has been seen.
    fn calibration_due(&self) -> Option<String> {
        if self.settings.shares != Shares::Decode
            || self.observed_since_plan.load(Ordering::Relaxed) < CALIBRATE_TOKENS
        {
            return None;
        }
        let in_plan: Vec<String> = self
            .plan()
            .into_iter()
            .filter(|e| e.layer_start < e.layer_end)
            .map(|e| e.node)
            .collect();
        let planned = self.planned_rates.lock().unwrap().clone();
        let observed = self.observed.lock().unwrap();
        let own = &self.settings.name;
        let (&own_planned, &(own_observed, _)) = (planned.get(own)?, observed.get(own)?);
        let mut off = Vec::new();
        for (name, planned_rate) in &planned {
            if name == own || !in_plan.contains(name) {
                continue;
            }
            let &(rate, _) = observed.get(name)?;
            let ratio = (rate / own_observed) / (planned_rate / own_planned);
            if !(1.0 / 1.2..=1.2).contains(&ratio) {
                off.push(format!(
                    "{name} runs at {ratio:.2} of what it was planned at"
                ));
            }
        }
        drop(observed);
        if off.is_empty() {
            // The plan holds; look again after as many tokens more.
            self.observed_since_plan.store(0, Ordering::Relaxed);
            return None;
        }
        if self.plan_is_good_enough() {
            log::info!(
                "orangu-server: workers: keeping the plan, another would save too little: {}",
                off.join(", ")
            );
            return None;
        }
        Some(off.join(", "))
    }

    /// Whether a plan is due: a configured worker is not connected, and either
    /// it answers now or the plan still has it. A worker that stays gone is
    /// tried every `readmit` and costs nothing more: a plan would drop every
    /// conversation here for the same plan, and a standby standing in for it.
    ///
    /// A worker left out for its failures is not tried, only planned out if
    /// the plan still has it.
    fn missing_and_due(&self) -> bool {
        if self.planning.lock().unwrap().elapsed() < self.settings.readmit {
            return false;
        }
        let path = self
            .assignment
            .lock()
            .unwrap()
            .as_ref()
            .map(|a| a.path.clone())
            .unwrap_or_else(|| vec![self.id.clone()]);
        let secret = self.settings.secret.as_deref();
        let used = self.used.lock().unwrap().clone();
        let mut tried = false;
        let due = self.links().iter().any(|link| {
            if link.is_connected() {
                return false;
            }
            if used.contains(&link.addr) {
                return true;
            }
            if link.out_for().is_some() {
                return false;
            }
            tried = true;
            link.ensure_connected(secret, &path).is_ok()
        });
        if tried && !due {
            *self.planning.lock().unwrap() = Instant::now();
        }
        due
    }

    /// Plans the assigned range again over the workers there are now — the
    /// same range, so the parent is none the wiser — once the node is
    /// quiet ([`QUIET`]). A worker lost below this node, and back, is taken
    /// back here: the parent only sees its own workers. Sessions still held
    /// go with the old plan; the parent rebuilds any it takes up again.
    fn replan_assignment(&self) {
        let mut last = self.planning.lock().unwrap();
        let current = self.assignment.lock().unwrap().as_ref().map(|a| {
            (
                a.layers.clone(),
                a.terms,
                a.path.clone(),
                a.store.quiet(QUIET),
            )
        });
        let Some((layers, terms, path, true)) = current else {
            return;
        };
        self.metrics.plans.fetch_add(1, Ordering::Relaxed);
        match self.plan_subtree(layers.clone(), &path, terms) {
            Ok(Planned {
                pipeline,
                entries,
                used,
                standing,
            }) => {
                let local = pipeline.local_layers();
                let n_layer = self.model.config().n_layer;
                self.release_unused(&local, !local.is_empty() && local.end == n_layer);
                self.copy_prompt_weights(&pipeline.local_layers(), terms.weights);
                let store = Arc::new(self.session_store(pipeline, terms));
                if let Some(assignment) = self.assignment.lock().unwrap().as_mut() {
                    assignment.store = store;
                }
                if *self.plan.lock().unwrap() != entries {
                    log::info!(
                        "orangu-server: layers {}..{} planned again: {}",
                        layers.start,
                        layers.end,
                        describe(&entries)
                    );
                }
                *self.plan.lock().unwrap() = entries;
                *self.used.lock().unwrap() = used;
                self.set_standing(standing);
            }
            Err(error) => log::warn!("orangu-server: planning the workers failed: {error:#}"),
        }
        *last = Instant::now();
    }

    /// Stops listening, lets go of the parent and the workers. For tests,
    /// which run several nodes in one process.
    #[cfg(test)]
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(addr) = self.local_addr {
            let _ = TcpStream::connect_timeout(&addr, Duration::from_secs(1));
        }
        if let Some(parent) = self.parent.lock().unwrap().take() {
            let _ = parent.stream.shutdown(std::net::Shutdown::Both);
        }
        for link in &self.links() {
            link.disconnect();
        }
    }
}

impl PipelineSource for Node {
    fn current(&self) -> (Arc<LayerPipeline>, u64) {
        self.top.read().unwrap().clone()
    }

    fn recover(&self, generation: u64) -> bool {
        let mut last = self.planning.lock().unwrap();
        if self.top.read().unwrap().1 != generation {
            return true;
        }
        if self.is_assigned() {
            return false;
        }
        self.metrics.recoveries.fetch_add(1, Ordering::Relaxed);
        self.plan_top_locked();
        *last = Instant::now();
        true
    }

    fn head_on_last(&self) -> bool {
        self.head_on_last.load(Ordering::Acquire)
    }

    fn served_whole(&self) {
        let used = self.used.lock().unwrap();
        for link in self.links().iter().chain(&self.standby_links()) {
            if link.is_connected() && used.contains(&link.addr) {
                link.forgive();
            }
        }
    }

    fn prefill_choice(
        &self,
        pipeline: &LayerPipeline,
        tree_tokens: usize,
        all_tokens: usize,
    ) -> Option<Handoff> {
        let inputs = self.prefill_inputs(pipeline, tree_tokens, all_tokens);
        let choice = super::prefill::decide(&inputs, &super::prefill::Mode::from_env());
        if choice.on.is_none() {
            log::debug!(
                "orangu-server: workers: a prompt of {tree_tokens} tokens runs through the tree: {}",
                choice.reason
            );
        }
        Some(Handoff {
            choice,
            header_hash: self.identity.header_hash,
            n_ctx: self.settings.n_ctx,
        })
    }

    fn prefilled(&self, ran: &Prefilled) {
        let generation = self.top.read().unwrap().1;
        let secs = ran.took.as_secs_f64();
        let mut seen = self.prefills.lock().unwrap();
        // Halfway to the newest: one odd prompt moves it, two confirm it.
        let blend = |old: Option<f64>, new: f64| old.map_or(new, |old| (old + new) / 2.0);
        if ran.fell_back.is_some() {
            seen.fell_back += 1;
        } else if ran.on.is_empty() {
            seen.on_tree += 1;
            if ran.tokens >= super::prefill::MIN_TOKENS && secs > 0.0 {
                let old = seen
                    .tree
                    .filter(|(_, at)| *at == generation)
                    .map(|(r, _)| r);
                seen.tree = Some((blend(old, ran.tokens as f64 / secs), generation));
            }
        } else {
            seen.on_node += 1;
            if secs > 0.0 {
                let rate = blend(seen.nodes.get(&ran.on).copied(), ran.tokens as f64 / secs);
                seen.nodes.insert(ran.on.clone(), rate);
            }
        }
        let ms = |s: Option<f64>| s.map(|s| s * 1e3);
        seen.last = Some(serde_json::json!({
            "on": if ran.on.is_empty() { "tree" } else { ran.on.as_str() },
            "tokens": ran.tokens,
            "ms": secs * 1e3,
            "rows_ms": ran.rows.as_secs_f64() * 1e3,
            "reason": ran.choice.reason,
            "predicted_tree_ms": ms(ran.choice.tree_s),
            "predicted_node_ms": ms(ran.choice.node_s),
            "fell_back": ran.fell_back,
        }));
    }

    fn observed_decode(&self, parts: &[(String, Duration)], tokens: usize) {
        if tokens == 0 {
            return;
        }
        let plan = self.plan();
        let bytes_of_nodes = |names: &[String]| -> u64 {
            plan.iter()
                .filter(|e| names.contains(&e.node))
                .map(|e| {
                    plan::bytes_of(
                        &(e.layer_start as usize..e.layer_end as usize),
                        &self.layer_bytes,
                    )
                })
                .sum()
        };
        let mut observed = self.observed.lock().unwrap();
        for (part, elapsed) in parts {
            // This node's own layers, or a worker's subtree by its address.
            let names: Vec<String> = if *part == self.settings.name {
                vec![part.clone()]
            } else {
                match self
                    .links()
                    .iter()
                    .find(|l| l.addr == *part)
                    .and_then(|l| l.info())
                {
                    Some(info) => info
                        .capacity
                        .setups
                        .iter()
                        .map(|s| s.name.clone())
                        .collect(),
                    None => continue,
                }
            };
            let bytes = bytes_of_nodes(&names);
            let per_token = elapsed.as_nanos() as f64 / tokens as f64;
            let (Some(own), true) = (names.first(), bytes > 0 && per_token > 0.0) else {
                continue;
            };
            // Bytes over nanoseconds: gigabytes a second, as measured rates
            // are. Weighted by tokens, the old capped so a change shows.
            let rate = bytes as f64 / per_token;
            let entry = observed.entry(own.clone()).or_insert((rate, 0));
            let old = entry.1.min(256) as f64;
            entry.0 = (entry.0 * old + rate * tokens as f64) / (old + tokens as f64);
            entry.1 += tokens;
        }
        drop(observed);
        // The step as a whole: every part, one after another.
        let step: f64 = parts
            .iter()
            .map(|(_, elapsed)| elapsed.as_secs_f64() * 1e3)
            .sum::<f64>()
            / tokens as f64;
        if step > 0.0 {
            let names = self.member_names(&[]);
            let mut seen = self.observed_step.lock().unwrap();
            *seen = Some(match seen.take() {
                Some((was, ms, count)) if was == names => {
                    let old = count.min(256) as f64;
                    let blended = (ms * old + step * tokens as f64) / (old + tokens as f64);
                    (names, blended, count + tokens)
                }
                _ => (names, step, tokens),
            });
        }
        self.observed_since_plan
            .fetch_add(tokens, Ordering::Relaxed);
    }
}

fn error(code: ErrorCode, message: impl Into<String>) -> Message {
    Message::Error(WorkerError::new(code, message))
}

/// A model as a log line names it: its label, and its quantization when
/// the label does not end with it.
fn named(model: &super::protocol::ModelIdentity) -> String {
    if model.quant.is_empty() || model.label.ends_with(&model.quant) {
        model.label.clone()
    } else {
        format!("{} ({})", model.label, model.quant)
    }
}

/// `a 0..8, b 8..20, c 20..32`.
pub fn describe(entries: &[PlanEntry]) -> String {
    entries
        .iter()
        .filter(|e| e.layer_start < e.layer_end)
        .map(|e| format!("{} {}..{}", e.node, e.layer_start, e.layer_end))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Whether `error` is a read running into its limit, rather than a connection
/// ending.
fn timed_out(error: &anyhow::Error) -> bool {
    error
        .chain()
        .filter_map(|cause| cause.downcast_ref::<std::io::Error>())
        .any(|e| {
            matches!(
                e.kind(),
                std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workers::pipeline::fixture::{self, N_CTX, N_LAYER, Variant};

    fn settings(name: &str, workers: Vec<String>, secret: Option<&str>) -> NodeSettings {
        NodeSettings {
            name: name.to_string(),
            listen: Some("127.0.0.1:0".to_string()),
            workers,
            standby: Vec::new(),
            channel: None,
            scan: Scan::default(),
            scan_every: Duration::from_millis(100),
            release: Duration::from_secs(300),
            api_port: 0,
            api_tls: false,
            secret: secret.map(str::to_string),
            activations: ActivationFormat::F32,
            timeout: Duration::from_secs(10),
            connect_timeout: Duration::from_secs(2),
            local_layers: LocalLayers::Auto,
            n_ctx: N_CTX,
            slots: 2,
            label: "fixture".to_string(),
            quant: "F32".to_string(),
            capacity: Capacity {
                backend: "cpu".to_string(),
                host_bytes: 1 << 30,
                budget_bytes: plan::budget(0, 1 << 30),
                ..Capacity::default()
            },
            shares: Shares::Memory,
            head: HeadOn::Top,
            divide: super::super::divide::Mode::Measured,
            split: true,
            maintenance: Duration::from_millis(50),
            parent_silence: Duration::from_secs(30),
            readmit: Duration::from_millis(200),
            tls: None,
            metrics_endpoint: None,
            prompt_weights: prompt_weights::PromptWeights::File,
            prefix_root: None,
            serve_file: None,
        }
    }

    fn node(name: &str, workers: &[&Arc<Node>], variant: Variant) -> Arc<Node> {
        node_with(
            settings(
                name,
                workers
                    .iter()
                    .map(|w| w.local_addr().unwrap().to_string())
                    .collect(),
                None,
            ),
            variant,
        )
    }

    fn node_with(settings: NodeSettings, variant: Variant) -> Arc<Node> {
        Node::start(
            settings,
            fixture::model_for(variant),
            fixture::loaded(variant),
        )
        .unwrap()
    }

    fn run(model: &dyn ModelForward, tokens: &[u32], prompt: usize) -> Vec<Vec<f32>> {
        let mut cache = model.new_kv_cache(N_CTX);
        let mut out = vec![model.forward(&mut cache, &tokens[..prompt], 0, 0).unwrap()];
        for (pos, token) in tokens.iter().enumerate().skip(prompt) {
            out.push(model.forward(&mut cache, &[*token], pos, 0).unwrap());
        }
        out
    }

    /// A node on `channel` that probes only `peers`, holding the model alone
    /// when `fits`.
    fn on_channel(name: &str, channel: &str, peers: &[&Arc<Node>], fits: bool) -> NodeSettings {
        let mut settings = settings(name, Vec::new(), None);
        settings.channel = Some(channel.to_string());
        settings.scan.peers = peers
            .iter()
            .map(|p| p.local_addr().unwrap().to_string())
            .collect();
        settings.release = Duration::from_millis(300);
        if !fits {
            settings.capacity.budget_bytes = 1;
        }
        settings
    }

    /// Waits until `node`'s scans have found `n` nodes.
    fn found(node: &Node, n: usize) -> Vec<super::super::channel::Found> {
        for _ in 0..100 {
            let found = node.found.lock().unwrap().0.clone();
            if found.len() == n {
                return found;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        panic!(
            "{} found {:?}",
            node.settings.name,
            node.found.lock().unwrap().0
        );
    }

    fn probe_of(node: &Node, channel: &str) -> Option<super::super::channel::Found> {
        super::super::channel::probe(
            node.local_addr().unwrap(),
            &super::super::channel::channel_hash(channel, None),
            None,
        )
    }

    /// A probe is answered on the node's own channel only, with what the
    /// node is doing and whether it holds the model.
    #[test]
    fn a_probe_is_answered_on_its_channel_only() {
        let beta = node_with(on_channel("beta", "lab", &[], true), Variant::default());
        let found = probe_of(&beta, "lab").expect("on the channel");
        assert_eq!(found.name, "beta");
        assert_eq!(found.state, ProbeState::Idle);
        assert!(found.whole);
        assert_eq!(found.load, 0.0);
        assert!(probe_of(&beta, "other").is_none());
        let off = node("off", &[], Variant::default());
        assert!(probe_of(&off, "lab").is_none());
    }

    /// A node holding its model in one file serves its bytes to any node
    /// that proves the secret, says so when probed, and refuses a range
    /// outside the file; a node with no file to serve says so too, and is
    /// never named a source by its probe.
    #[test]
    fn a_node_serves_its_model_file_by_range() {
        use crate::workers::share::PeerSource;
        use orangu::model_download::RangeSource;
        let dir = tempfile::tempdir().unwrap();
        let snapshot = dir.path().join("models--u--m/snapshots/feed");
        std::fs::create_dir_all(&snapshot).unwrap();
        let path = snapshot.join("m-F32.gguf");
        let body: Vec<u8> = (0..50_000u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&path, &body).unwrap();
        let mut s = on_channel("beta", "lab", &[], true);
        s.secret = Some("s1".to_string());
        s.serve_file = Some(path);
        let beta = node_with(s, Variant::default());
        let addr = beta.local_addr().unwrap();
        let found = super::super::channel::probe(
            addr,
            &super::super::channel::channel_hash("lab", Some("s1")),
            None,
        )
        .expect("on the channel");
        assert_eq!(found.features & FEATURE_FETCH, FEATURE_FETCH);
        assert_eq!(found.model, beta.identity.header_hash);

        let peer = PeerSource::new("beta".into(), addr.to_string(), Some("s1".into()), None);
        let info = peer.info().unwrap();
        assert_eq!(
            (info.size, info.file.as_str(), info.commit.as_str()),
            (body.len() as u64, "m-F32.gguf", "feed")
        );
        assert_eq!(info.model, beta.identity.header_hash);
        assert_eq!(peer.read(1000, 20_000).unwrap(), body[1000..21_000]);
        assert!(peer.read(49_990, 11).is_err());
        assert_eq!(peer.read(49_990, 10).unwrap(), body[49_990..]);
        // Fetching claims nothing: the node is still free.
        assert!(beta.parent.lock().unwrap().is_none());
        assert_eq!(probe_of_secret(&beta, "lab", "s1").state, ProbeState::Idle);

        let stranger = PeerSource::new("beta".into(), addr.to_string(), Some("s2".into()), None);
        assert!(stranger.read(0, 10).is_err());
        let anonymous = PeerSource::new("beta".into(), addr.to_string(), None, None);
        assert!(anonymous.read(0, 10).is_err());

        let gamma = node_with(on_channel("gamma", "lab", &[], true), Variant::default());
        let found = probe_of(&gamma, "lab").expect("on the channel");
        assert_eq!(found.features & FEATURE_FETCH, 0);
        let refused = PeerSource::new(
            "gamma".into(),
            gamma.local_addr().unwrap().to_string(),
            None,
            None,
        );
        assert!(refused.info().is_err());
        beta.stop();
        gamma.stop();
    }

    fn probe_of_secret(node: &Node, channel: &str, secret: &str) -> super::super::channel::Found {
        super::super::channel::probe(
            node.local_addr().unwrap(),
            &super::super::channel::channel_hash(channel, Some(secret)),
            None,
        )
        .expect("on the channel")
    }

    /// With a secret, a probe is answered only by a node that shares it:
    /// one that knows the channel's name alone finds nothing, and two
    /// nodes with another secret each never see each other.
    #[test]
    fn a_secret_keys_the_channel() {
        let mut s = on_channel("beta", "lab", &[], true);
        s.secret = Some("s1".to_string());
        let beta = node_with(s, Variant::default());
        let addr = beta.local_addr().unwrap();
        let ask = |secret: Option<&str>| {
            super::super::channel::probe(
                addr,
                &super::super::channel::channel_hash("lab", secret),
                None,
            )
        };
        assert!(ask(Some("s1")).is_some());
        assert!(ask(None).is_none());
        assert!(ask(Some("s2")).is_none());
        let mut s = on_channel("gamma", "lab", &[&beta], true);
        s.secret = Some("s2".to_string());
        let gamma = node_with(s, Variant::default());
        let mut s = on_channel("delta", "lab", &[&beta], true);
        s.secret = Some("s1".to_string());
        let delta = node_with(s, Variant::default());
        assert_eq!(found(&delta, 1)[0].name, "beta");
        std::thread::sleep(Duration::from_millis(300));
        assert!(gamma.found.lock().unwrap().0.is_empty());
        for n in [&beta, &gamma, &delta] {
            n.stop();
        }
    }

    /// A model no node holds alone is divided: the node a request reaches
    /// claims the free nodes of its channel when the request starts, plans
    /// over them, and lets them go once it has had no request for `release`.
    #[test]
    fn a_node_that_cannot_hold_the_model_claims_its_channel_and_lets_go() {
        let beta = node_with(on_channel("beta", "lab", &[], false), Variant::default());
        let gamma = node_with(on_channel("gamma", "lab", &[], false), Variant::default());
        let alpha = node_with(
            on_channel("alpha", "lab", &[&beta, &gamma], false),
            Variant::default(),
        );
        assert!(alpha.delegates());
        assert_eq!(found(&alpha, 2).len(), 2);
        // Nothing is claimed before a request.
        assert!(!beta.is_assigned() && !gamma.is_assigned());

        let request = alpha.start_request();
        assert!(alpha.leading.load(Ordering::Acquire));
        assert_eq!(covered(&alpha.plan()), N_LAYER);
        let used = alpha.used.lock().unwrap().clone();
        assert!(!used.is_empty(), "{:?}", alpha.plan());
        assert!(beta.is_assigned() || gamma.is_assigned());
        // The tree is one level deep: a leading node is not claimed.
        let refused = ChildLink::new(
            alpha.local_addr().unwrap().to_string(),
            Duration::from_secs(5),
            Duration::from_secs(2),
        )
        .connect(None, &["someone".to_string()])
        .unwrap_err();
        assert_eq!(refused.code, ErrorCode::Busy, "{refused}");
        assert_eq!(probe_of(&alpha, "lab").unwrap().state, ProbeState::Leading);

        // Kept while the request runs, let go after it.
        std::thread::sleep(Duration::from_millis(500));
        assert!(alpha.leading.load(Ordering::Acquire));
        drop(request);
        for _ in 0..100 {
            if !alpha.leading.load(Ordering::Acquire) && !beta.is_assigned() && !gamma.is_assigned()
            {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(!alpha.leading.load(Ordering::Acquire));
        assert!(alpha.links().is_empty());
        assert!(!beta.is_assigned() && !gamma.is_assigned());
    }

    /// With `ORANGU_WORKERS_DIVIDE=always`, a model the node with the
    /// request could hold alone is divided all the same: every free node of
    /// the channel takes a share, none is held back, and the answer is the
    /// unsplit model's.
    #[test]
    fn every_free_node_takes_a_share_of_a_model_that_fits_alone() {
        let beta = node_with(on_channel("beta", "lab", &[], true), Variant::default());
        let gamma = node_with(on_channel("gamma", "lab", &[], true), Variant::default());
        let mut s = on_channel("alpha", "lab", &[&beta, &gamma], true);
        s.divide = super::super::divide::Mode::Always;
        let alpha = node_with(s, Variant::default());
        assert!(alpha.whole());
        found(&alpha, 2);
        assert_eq!(alpha.route(), None, "the node with the request leads");
        let request = alpha.start_request();
        assert_eq!(alpha.links().len(), 2);
        assert!(alpha.standby_links().is_empty());
        assert!(beta.is_assigned() && gamma.is_assigned());
        let plan = alpha.plan();
        assert_eq!(plan.len(), 3, "{plan:?}");
        assert!(plan.iter().all(|e| e.layer_end > e.layer_start), "{plan:?}");

        let tokens = fixture::tokens();
        let whole = run(fixture::model().as_ref(), &tokens, 5);
        let split = alpha.delegating_model().unwrap();
        let mut cache = split.new_kv_cache(N_CTX);
        split.forward(&mut cache, &tokens[..5], 0, 0).unwrap();
        let mut worst = 0f32;
        for (pos, token) in tokens.iter().enumerate().skip(5) {
            let got = split.forward(&mut cache, &[*token], pos, 0).unwrap();
            for (a, b) in got.iter().zip(&whole[pos - 4]) {
                worst = worst.max((a - b).abs());
            }
        }
        assert!(worst < 1e-3, "{worst}");
        drop(cache);
        drop(request);
        for n in [&alpha, &beta, &gamma] {
            n.stop();
        }
    }

    /// With no standby held, a worker lost mid-sequence is replaced by a
    /// free node of the channel rather than its layers spread over the
    /// nodes left.
    #[test]
    fn a_lost_worker_is_replaced_by_a_free_node_of_the_channel() {
        let beta = node_with(on_channel("beta", "lab", &[], false), Variant::default());
        let later = TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let mut s = on_channel("alpha", "lab", &[&beta], false);
        s.scan.peers.push(later.to_string());
        let alpha = node_with(s, Variant::default());
        found(&alpha, 1);
        let request = alpha.start_request();
        assert!(alpha.standby_links().is_empty());
        assert!(beta.is_assigned());
        let mut s = on_channel("gamma", "lab", &[], false);
        s.listen = Some(later.to_string());
        let gamma = node_with(s, Variant::default());
        found(&alpha, 2);

        let tokens = fixture::tokens();
        let split = alpha.delegating_model().unwrap();
        let mut cache = split.new_kv_cache(N_CTX);
        split.forward(&mut cache, &tokens[..5], 0, 0).unwrap();
        beta.stop();
        for (pos, token) in tokens.iter().enumerate().skip(5) {
            split.forward(&mut cache, &[*token], pos, 0).unwrap();
        }
        assert!(gamma.is_assigned());
        assert!(
            alpha.plan().iter().any(|e| e.node == "gamma"),
            "{}",
            describe(&alpha.plan())
        );
        assert_eq!(
            alpha.status()["standby"][0]["standing_in_for"],
            beta.local_addr().unwrap().to_string()
        );
        drop(cache);
        drop(request);
        for n in [&alpha, &gamma] {
            n.stop();
        }
    }

    /// A node that answers at two addresses is found once, and claimed once.
    #[test]
    #[cfg_attr(
        target_os = "macos",
        ignore = "macOS answers on 127.0.0.1 alone, not the rest of 127.0.0.0/8"
    )]
    fn a_node_at_two_addresses_is_claimed_once() {
        let mut s = on_channel("beta", "lab", &[], false);
        s.listen = Some("0.0.0.0:0".to_string());
        let beta = node_with(s, Variant::default());
        let port = beta.local_addr().unwrap().port();
        let mut s = on_channel("alpha", "lab", &[], false);
        s.scan.peers = vec![format!("127.0.0.1:{port}"), format!("127.0.0.2:{port}")];
        let alpha = node_with(s, Variant::default());
        let found = found(&alpha, 1);
        assert_eq!(found[0].alternates.len(), 1, "{found:?}");
        let _request = alpha.start_request();
        assert_eq!(alpha.links().len(), 1);
        assert!(beta.is_assigned());
        // Claimed already: not a candidate at either address.
        assert!(alpha.unclaimed().is_empty());
    }

    /// A claimer loses to a node that already leads one of the free nodes,
    /// whatever its id, else to the lowest lower id one is promised to.
    #[test]
    fn a_contest_goes_to_a_leader_then_to_the_lowest_id() {
        use super::super::channel::LeaseAnswer::{Committed, Granted, Promised};
        assert_eq!(contest_winner("m", [Granted, Granted]), None);
        assert_eq!(contest_winner("m", [Promised("z".into())]), None);
        assert_eq!(
            contest_winner("m", [Promised("k".into()), Granted, Promised("c".into())]).as_deref(),
            Some("c")
        );
        assert_eq!(
            contest_winner("m", [Promised("c".into()), Committed("z".into())]).as_deref(),
            Some("z")
        );
        assert_eq!(contest_winner("m", [Committed("m".into())]), None);
    }

    /// A free node is promised to the lowest id that asks within the
    /// lease, turns away the handshake of anyone else, and answers a
    /// node that leads it with that leader.
    #[test]
    fn a_free_node_is_promised_to_the_lowest_id() {
        let beta = node_with(on_channel("beta", "lab", &[], true), Variant::default());
        let granted = |message: Message| matches!(message, Message::LeaseAck { granted: true, .. });
        assert!(granted(beta.answer_lease("m")));
        assert!(granted(beta.answer_lease("c")));
        assert_eq!(
            beta.answer_lease("k"),
            Message::LeaseAck {
                granted: false,
                holder: "c".to_string(),
                committed: false,
            }
        );
        // Asked over the channel's port, as a claimer does.
        let addr = beta.local_addr().unwrap();
        let hash = super::super::channel::channel_hash("lab", None);
        assert_eq!(
            super::super::channel::lease(addr, &hash, "x", None),
            Some(super::super::channel::LeaseAnswer::Promised(
                "c".to_string()
            ))
        );
        assert_eq!(
            super::super::channel::lease(
                addr,
                &super::super::channel::channel_hash("no", None),
                "a",
                None
            ),
            None
        );
        // Promised to "c": another parent's handshake is refused.
        let link = ChildLink::new(
            addr.to_string(),
            Duration::from_secs(5),
            Duration::from_secs(2),
        );
        let refused = link.connect(None, &["k".to_string()]).unwrap_err();
        assert_eq!(refused.code, ErrorCode::Busy, "{refused}");
        assert!(refused.to_string().contains("promised to c"), "{refused}");
        // "c" itself is taken, and from then on beta answers as its worker.
        link.connect(None, &["c".to_string()]).unwrap();
        assert!(beta.lease_holder().is_none());
        assert_eq!(
            beta.answer_lease("a"),
            Message::LeaseAck {
                granted: false,
                holder: "c".to_string(),
                committed: true,
            }
        );
    }

    /// Two nodes that claim at the same moment a model no node holds alone:
    /// whichever asks first, every free node ends promised to the lower id,
    /// and the other node loses to it.
    #[test]
    fn two_claimers_end_with_one_winner() {
        let beta = node_with(on_channel("beta", "lab", &[], false), Variant::default());
        let gamma = node_with(on_channel("gamma", "lab", &[], false), Variant::default());
        let one = node_with(
            on_channel("one", "lab", &[&beta, &gamma], false),
            Variant::default(),
        );
        let two = node_with(
            on_channel("two", "lab", &[&beta, &gamma], false),
            Variant::default(),
        );
        found(&one, 2);
        found(&two, 2);
        let (low, high) = if one.id < two.id {
            (&one, &two)
        } else {
            (&two, &one)
        };
        // The higher id asks first, and gets them, until the lower asks.
        assert_eq!(high.contend(), Contest::Won);
        assert_eq!(low.contend(), Contest::Won);
        for free in [&beta, &gamma] {
            assert_eq!(free.lease_holder().as_deref(), Some(low.id.as_str()));
        }
        assert_eq!(high.contend(), Contest::Lost(low.id.clone()));
        // The winner holds both as workers; the loser's handshakes would be
        // refused.
        let _request = low.start_request();
        assert_eq!(low.workers_held(), 2);
        assert!(beta.is_assigned() && gamma.is_assigned());
        for free in [&beta, &gamma] {
            assert_eq!(probe_of(free, "lab").unwrap().state, ProbeState::Serving);
        }
    }

    /// A node that cannot hold the model, with no free node that can,
    /// sends a request to a node that leads a tree of the model.
    #[test]
    fn a_request_goes_to_a_node_leading_the_model() {
        let beta = node_with(on_channel("beta", "lab", &[], false), Variant::default());
        let mut s = on_channel("alpha", "lab", &[&beta], false);
        s.api_port = 1111;
        s.release = Duration::from_secs(60);
        let alpha = node_with(s, Variant::default());
        found(&alpha, 1);
        let _request = alpha.start_request();
        assert!(alpha.leading.load(Ordering::Acquire));
        assert_eq!(probe_of(&alpha, "lab").unwrap().workers, 1);

        let delta = node_with(
            on_channel("delta", "lab", &[&alpha, &beta], false),
            Variant::default(),
        );
        let found = found(&delta, 2);
        assert!(
            found.iter().any(|f| f.state == ProbeState::Leading),
            "{found:?}"
        );
        assert_eq!(delta.route().as_deref(), Some("http://127.0.0.1:1111"));
    }

    /// Of two nodes that lead trees of the same model, the one with fewer
    /// workers gives way once it is idle. (A node that finds a tree of its
    /// model sends its requests there; `delta` divides whatever it finds, as
    /// two trees that form at once would.)
    #[test]
    fn the_smaller_of_two_trees_gives_way() {
        let beta = node_with(on_channel("beta", "lab", &[], false), Variant::default());
        let gamma = node_with(on_channel("gamma", "lab", &[], false), Variant::default());
        let epsilon = node_with(on_channel("epsilon", "lab", &[], false), Variant::default());
        let mut s = on_channel("alpha", "lab", &[&beta, &gamma], false);
        s.release = Duration::from_secs(60);
        let alpha = node_with(s, Variant::default());
        found(&alpha, 2);
        let _larger = alpha.start_request();
        assert_eq!(alpha.workers_held(), 2);

        let mut s = on_channel("delta", "lab", &[&alpha, &epsilon], false);
        s.release = Duration::from_secs(60);
        s.divide = super::super::divide::Mode::Always;
        let delta = node_with(s, Variant::default());
        found(&delta, 2);
        let smaller = delta.start_request();
        assert_eq!(delta.links().len(), 1);
        // Not while its own request runs.
        assert!(!delta.yields_to_larger());
        assert!(!alpha.yields_to_larger());
        drop(smaller);
        for _ in 0..100 {
            if !delta.leading.load(Ordering::Acquire) {
                break;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(!delta.leading.load(Ordering::Acquire));
        assert!(!epsilon.is_assigned());
        assert!(alpha.leading.load(Ordering::Acquire));
    }

    /// A node that cannot divide the model (`split` off) sends a request
    /// to the least busy free node that holds it.
    #[test]
    fn a_node_that_cannot_divide_routes_to_one_that_holds_the_model() {
        let gamma = node_with(on_channel("gamma", "lab", &[], true), Variant::default());
        let mut s = on_channel("alpha", "lab", &[&gamma], true);
        s.split = false;
        let alpha = node_with(s, Variant::default());
        found(&alpha, 1);
        assert!(!alpha.may_claim());
        assert_eq!(alpha.route().as_deref(), Some("http://127.0.0.1:0"));
        // A request already routed once stays.
        assert!(route(true).is_none());
    }

    /// A node that holds the model, with no measured speeds, never claims
    /// workers: a tree that cannot be priced is not divided over.
    #[test]
    fn a_node_that_holds_the_model_does_not_claim_unmeasured() {
        let beta = node_with(on_channel("beta", "lab", &[], true), Variant::default());
        let alpha = node_with(
            on_channel("alpha", "lab", &[&beta], true),
            Variant::default(),
        );
        found(&alpha, 1);
        assert!(alpha.may_claim_any());
        assert!(!alpha.may_claim());
        assert_eq!(alpha.route(), None);
        let _request = alpha.start_request();
        assert!(!beta.is_assigned());
        assert!(alpha.links().is_empty());
        let status = alpha.status();
        assert_eq!(status["divide"]["last"]["verdict"], "here", "{status}");
    }

    /// With no measured speeds, a busy node that holds the model sends a
    /// request to a less busy one that holds it too; a node that does not
    /// hold the model sends it to whichever free node does.
    #[test]
    fn a_request_goes_to_the_least_busy_node_that_holds_the_model() {
        let gamma = node_with(on_channel("gamma", "lab", &[], true), Variant::default());
        let beta = node_with(
            on_channel("beta", "lab", &[&gamma], true),
            Variant::default(),
        );
        found(&beta, 1);
        // Equally idle: served here.
        assert_eq!(beta.route(), None);
        // Busy: both slots in use.
        beta.active.store(2, Ordering::Release);
        assert_eq!(beta.route().as_deref(), Some("http://127.0.0.1:0"));
        beta.active.store(0, Ordering::Release);

        let alpha = node_with(
            on_channel("alpha", "lab", &[&gamma], false),
            Variant::default(),
        );
        found(&alpha, 1);
        assert_eq!(alpha.route().as_deref(), Some("http://127.0.0.1:0"));
        let _request = alpha.start_request();
        assert!(!gamma.is_assigned());
    }

    /// The bytes of the fixture's layers.
    fn layer_total() -> u64 {
        let loaded = fixture::loaded(Variant::default());
        loaded
            .tensor_sizes()
            .filter(|(name, _)| {
                crate::engine::loader::block_index(name).is_some_and(|il| il < N_LAYER)
            })
            .map(|(_, bytes)| bytes)
            .sum()
    }

    /// A node on `channel` probing `peers` that measured its decode at
    /// `ms_for_all` milliseconds for every layer of the fixture, plus
    /// `fixed_ms` a step, with `budget` bytes for weights.
    fn paced(
        name: &str,
        peers: &[&Arc<Node>],
        ms_for_all: f64,
        fixed_ms: f32,
        budget: u64,
        api_port: u16,
    ) -> Arc<Node> {
        node_with(
            paced_settings(name, peers, ms_for_all, fixed_ms, budget, api_port),
            Variant::default(),
        )
    }

    fn paced_settings(
        name: &str,
        peers: &[&Arc<Node>],
        ms_for_all: f64,
        fixed_ms: f32,
        budget: u64,
        api_port: u16,
    ) -> NodeSettings {
        let mut s = on_channel(name, "lab", peers, true);
        s.capacity.budget_bytes = budget;
        s.capacity.setups = vec![NodeSetup {
            name: name.to_string(),
            decode_rate: (layer_total() as f64 / 1e9 / (ms_for_all / 1e3)) as f32,
            decode_fixed_ms: fixed_ms,
            ..NodeSetup::default()
        }];
        s.api_port = api_port;
        s
    }

    /// Room for the whole fixture, and for about half its layers.
    fn room() -> (u64, u64) {
        let all: u64 = fixture::loaded(Variant::default())
            .tensor_sizes()
            .map(|(_, bytes)| bytes)
            .sum();
        (all * 4, layer_total() * 6 / 10)
    }

    /// A node that holds the model and decodes it faster than a tree over
    /// the free nodes would serves it alone: it claims nobody, and says so.
    #[test]
    fn a_fast_node_that_holds_the_model_serves_it_alone() {
        let (whole, _) = room();
        let beta = paced("beta", &[], 400.0, 20.0, whole, 0);
        let gamma = paced("gamma", &[], 400.0, 20.0, whole, 0);
        let alpha = paced("alpha", &[&beta, &gamma], 10.0, 1.0, whole, 0);
        found(&alpha, 2);
        let decision = alpha.choose(0);
        assert_eq!(decision.verdict, Verdict::Here, "{decision:?}");
        assert!(
            decision.tree_ms.unwrap() > decision.alone_ms.unwrap(),
            "{decision:?}"
        );
        let _request = alpha.start_request();
        assert!(alpha.links().is_empty());
        assert!(!beta.is_assigned() && !gamma.is_assigned());
        // The probe tells the others what a request takes here.
        let alone = alpha.alone_ms().unwrap();
        assert!(alone >= 11.0, "{alone}");
        let probe = probe_of(&alpha, "lab").unwrap();
        assert!(
            (f64::from(probe.pace.step_ms) - alone).abs() < 0.01,
            "{probe:?}"
        );
        assert!(probe.pace.decode_rate > 0.0 && probe.pace.budget_bytes == whole);
    }

    /// A slow node that holds the model sends a request to an idle node that
    /// holds it too and decodes it faster.
    #[test]
    fn a_slow_node_sends_a_request_to_a_faster_one() {
        let (whole, _) = room();
        let laptop = paced("laptop", &[], 10.0, 1.0, whole, 4444);
        let board = paced("board", &[&laptop], 400.0, 20.0, whole, 0);
        found(&board, 1);
        assert_eq!(board.route().as_deref(), Some("http://127.0.0.1:4444"));
        let status = board.status();
        assert_eq!(status["divide"]["last"]["verdict"], "routed", "{status}");
        assert_eq!(status["divide"]["last"]["to"], "laptop", "{status}");
        // The laptop keeps its own.
        found(&laptop, 0);
        assert_eq!(laptop.route(), None);
    }

    /// A slow node that holds the model, beside a fast one that holds only
    /// about half of it: the tree over both is predicted faster, and the
    /// model is divided.
    #[test]
    fn a_tree_predicted_faster_divides_a_model_that_fits_alone() {
        let (whole, half) = room();
        let fast = paced("fast", &[], 10.0, 1.0, half, 0);
        let slow = paced("slow", &[&fast], 2000.0, 1.0, whole, 0);
        found(&slow, 1);
        let decision = slow.choose(0);
        assert_eq!(decision.verdict, Verdict::Divide, "{decision:?}");
        assert_eq!(slow.route(), None);
        let _request = slow.start_request();
        assert!(fast.is_assigned());
        assert_eq!(slow.links().len(), 1);
        // Leading, it reports the tree's time a token.
        let probe = probe_of(&slow, "lab").unwrap();
        assert_eq!(probe.state, ProbeState::Leading);
        assert!(
            f64::from(probe.pace.step_ms) < decision.alone_ms.unwrap(),
            "{probe:?} {decision:?}"
        );
    }

    /// A tree whose decode step requests found slower than the node alone is
    /// let go between requests, and is not formed again.
    #[test]
    fn a_tree_found_slower_than_alone_is_let_go() {
        let (whole, half) = room();
        let fast = paced("fast", &[], 10.0, 1.0, half, 0);
        let mut s = paced_settings("slow", &[&fast], 2000.0, 1.0, whole, 0);
        s.release = Duration::from_secs(60);
        let slow = node_with(s, Variant::default());
        found(&slow, 1);
        let request = slow.start_request();
        assert!(slow.leading.load(Ordering::Acquire));
        // Requests found every step far slower than predicted.
        let names = slow.member_names(&[]);
        *slow.observed_step.lock().unwrap() = Some((names, 1e6, CALIBRATE_TOKENS));
        assert_eq!(slow.tree_is_slower_than_alone(), None, "not mid-request");
        drop(request);
        assert!(slow.tree_is_slower_than_alone().is_some());
        until("the tree is let go", || {
            !slow.leading.load(Ordering::Acquire) && !fast.is_assigned()
        });
        until("fast is free again", || {
            slow.found
                .lock()
                .unwrap()
                .0
                .iter()
                .any(|f| f.name == "fast" && f.state == ProbeState::Idle)
        });
        assert_eq!(slow.choose(0).verdict, Verdict::Here);
    }

    /// A sync benchmark against a free node of the channel: every round
    /// trip echoed over the worker transport, a token's time the sum of its
    /// round trips, the prediction beside it, and the peer let go after.
    /// Refused for a node that leads, a peer that is not free, and one the
    /// channel does not have.
    #[test]
    fn a_sync_benchmark_times_the_sums_against_a_free_peer() {
        use super::super::syncbench::Ask;
        let (whole, _) = room();
        let beta = paced("beta", &[], 25.0, 1.0, whole, 0);
        let alpha = paced("alpha", &[&beta], 50.0, 1.0, whole, 0);
        found(&alpha, 1);
        let ask = Ask {
            peer: "beta".to_string(),
            tokens: 5,
            layers: 3,
            hidden: 64,
        };
        let result = alpha.sync_bench(&ask).unwrap();
        assert_eq!(result["round_trips_per_token"], 6, "{result}");
        assert_eq!(result["f32"]["payload_bytes"], 256);
        assert_eq!(result["f16"]["payload_bytes"], 128);
        assert_eq!(result["peer"]["name"], "beta");
        assert_eq!(result["peer"]["tls"], false);
        assert!(result["link"]["rtt_ms"].as_f64().is_some(), "{result}");
        for format in ["f32", "f16"] {
            let token = result[format]["per_token_ms"]["median"].as_f64().unwrap();
            let trip = result[format]["round_trip_ms"]["median"].as_f64().unwrap();
            assert!(token > 0.0 && token >= trip, "{result}");
            assert!(result[format]["mb_per_s"].as_f64().unwrap() > 0.0);
        }
        // Hidden states travel in f32 here: the sync is the f32 one, and
        // the halves run at the slower node's rate — this one's.
        let predicted = &result["predicted"];
        let alone = alpha.alone_ms().unwrap();
        assert!((predicted["alone_ms"].as_f64().unwrap() - alone).abs() < 1e-6);
        assert_eq!(predicted["sync_from"], "f32");
        assert_eq!(predicted["peer_rate_known"], true);
        assert_eq!(
            predicted["sync_ms"],
            result["f32"]["per_token_ms"]["median"]
        );
        let compute = predicted["tp2_compute_ms"].as_f64().unwrap();
        assert!(compute < alone && compute > 1.0, "{predicted}");
        let total = predicted["tp2_total_ms"].as_f64().unwrap();
        assert!(
            (total - compute - predicted["sync_ms"].as_f64().unwrap()).abs() < 1e-9,
            "{predicted}"
        );
        until("beta is let go", || beta.parent.lock().unwrap().is_none());
        assert!(!alpha.benching.load(Ordering::Acquire));

        let missing = Ask {
            peer: "nobody".to_string(),
            ..ask.clone()
        };
        assert_eq!(alpha.sync_bench(&missing).unwrap_err().0, 404);
        alpha.leading.store(true, Ordering::Release);
        let (status, why) = alpha.sync_bench(&ask).unwrap_err();
        assert_eq!(status, 409, "{why}");
        assert!(why.contains("leads a tree"), "{why}");
        alpha.leading.store(false, Ordering::Release);
        // No more scans: beta is made busy by hand.
        alpha.stop.store(true, Ordering::Release);
        std::thread::sleep(Duration::from_millis(300));
        alpha.found.lock().unwrap().0[0].state = ProbeState::Serving;
        let (status, why) = alpha.sync_bench(&ask).unwrap_err();
        assert_eq!(status, 409, "{why}");
        assert!(why.contains("only a free node"), "{why}");
        beta.stop();
    }

    /// Nodes cooperate only on the same model: one with another
    /// quantization is neither routed to nor claimed, and one with other
    /// weights in the same layout is not routed to.
    #[test]
    fn a_node_with_another_model_is_not_used() {
        let other = node_with(
            on_channel("other", "lab", &[], true),
            Variant {
                f16_ffn_down: true,
                ..Variant::default()
            },
        );
        let nudged = node_with(
            on_channel("nudged", "lab", &[], true),
            Variant {
                nudge_layer: Some(1),
                ..Variant::default()
            },
        );
        let alpha = node_with(
            on_channel("alpha", "lab", &[&other, &nudged], false),
            Variant::default(),
        );
        let found = found(&alpha, 2);
        let by_name = |name: &str| found.iter().find(|f| f.name == name).unwrap().clone();
        assert!(!alpha.same_model(&by_name("other"), false));
        assert!(alpha.same_model(&by_name("nudged"), false));
        assert!(!alpha.same_model(&by_name("nudged"), true));
        // Neither serves a request routed here.
        assert_eq!(alpha.route(), None);
        // Only the one with the same layout is a candidate worker; its
        // weights are compared when it is claimed.
        let unclaimed: Vec<String> = alpha.unclaimed().into_iter().map(|f| f.name).collect();
        assert_eq!(unclaimed, ["nudged"]);
        assert_eq!(
            alpha.status()["found"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|f| f["same_model"] == false)
                .count(),
            1
        );
    }

    /// One channel, two models: a node divides its own model over the free
    /// nodes with that model only, and a request naming the other model goes
    /// to a node that serves it, from either side.
    #[test]
    fn a_channel_with_two_models_routes_each_by_name() {
        let second = |name: &str, peers: &[&Arc<Node>], api_port: u16| {
            let mut s = on_channel(name, "lab", peers, true);
            s.label = "second-model".to_string();
            s.quant = "F16".to_string();
            s.api_port = api_port;
            node_with(
                s,
                Variant {
                    f16_ffn_down: true,
                    ..Variant::default()
                },
            )
        };
        let beta = node_with(on_channel("beta", "lab", &[], true), Variant::default());
        let other = second("other", &[], 2222);
        let mut s = on_channel("alpha", "lab", &[&beta, &other], true);
        s.api_port = 1111;
        s.release = Duration::from_secs(60);
        s.divide = super::super::divide::Mode::Always;
        let alpha = node_with(s, Variant::default());
        let seen = found(&alpha, 2);
        let named = seen.iter().find(|f| f.name == "other").unwrap();
        assert_eq!(
            (named.label.as_str(), named.quant.as_str()),
            ("second-model", "F16")
        );
        let status = alpha.status();
        let shown = status["found"]
            .as_array()
            .unwrap()
            .iter()
            .find(|f| f["name"] == "other")
            .unwrap();
        assert_eq!(shown["model_name"], "second-model");
        assert_eq!(shown["model_quant"], "F16");

        // A request naming the other model, by any of its names, goes to the
        // node that serves it; one naming this node's model, or a model
        // nobody serves, is left to the ordinary routing.
        for asked in ["second-model", "SECOND-MODEL", "second-model:f16"] {
            assert_eq!(
                alpha.route_by_model(asked).as_deref(),
                Some("http://127.0.0.1:2222"),
                "{asked}"
            );
        }
        for asked in ["fixture", "FIXTURE:f32", "gpt-4", "second-model:Q8_0", ""] {
            assert_eq!(alpha.route_by_model(asked), None, "{asked}");
        }
        assert_eq!(
            alpha.channel_models(),
            [ChannelModel {
                id: "second-model:F16".to_string(),
                label: "second-model".to_string(),
                quant: "F16".to_string(),
                nodes: vec!["other".to_string()],
            }]
        );

        // Only the node with the same model is claimed.
        let unclaimed: Vec<String> = alpha.unclaimed().into_iter().map(|f| f.name).collect();
        assert_eq!(unclaimed, ["beta"]);
        let _request = alpha.start_request();
        assert_eq!(alpha.links().len(), 1);
        assert!(beta.is_assigned());
        assert!(!other.is_assigned());
        assert_eq!(probe_of(&other, "lab").unwrap().state, ProbeState::Idle);
        assert_eq!(other.route_by_model("second-model"), None);

        // From a node with the second model: a request for the first goes to
        // the node leading a tree of it, never to that tree's worker; the
        // node claims only the free node with its own model.
        let delta = second("delta", &[&alpha, &beta, &other], 3333);
        found(&delta, 3);
        until("delta sees alpha leading", || {
            delta
                .found
                .lock()
                .unwrap()
                .0
                .iter()
                .any(|f| f.name == "alpha" && f.state == ProbeState::Leading)
        });
        for asked in ["fixture", "fixture:F32"] {
            assert_eq!(
                delta.route_by_model(asked).as_deref(),
                Some("http://127.0.0.1:1111"),
                "{asked}"
            );
        }
        assert_eq!(delta.route_by_model("second-model"), None);
        let unclaimed: Vec<String> = delta.unclaimed().into_iter().map(|f| f.name).collect();
        assert_eq!(unclaimed, ["other"]);
        assert_eq!(
            delta
                .channel_models()
                .into_iter()
                .map(|m| (m.id, m.nodes))
                .collect::<Vec<_>>(),
            [("fixture:F32".to_string(), vec!["alpha".to_string()])]
        );
    }

    /// A model's name for `/v1/models` carries its quantization once.
    #[test]
    fn a_model_id_carries_its_tag_once() {
        assert_eq!(model_id("m", "Q4_K_M"), "m:Q4_K_M");
        assert_eq!(model_id("m:Q4_K_M", "Q4_K_M"), "m:Q4_K_M");
        assert_eq!(model_id("m:q4_k_m", "Q4_K_M"), "m:q4_k_m");
        assert_eq!(model_id("m", ""), "m");
    }

    /// A busy node is not claimed, and a less busy one is planned a larger
    /// share for its speed.
    #[test]
    fn load_sizes_what_a_node_is_given() {
        let beta = node_with(on_channel("beta", "lab", &[], true), Variant::default());
        let alpha = node_with(
            on_channel("alpha", "lab", &[&beta], false),
            Variant::default(),
        );
        found(&alpha, 1);
        // No more scans: the loads below are set by hand.
        alpha.stop.store(true, Ordering::Release);
        std::thread::sleep(Duration::from_millis(300));
        {
            let mut found = alpha.found.lock().unwrap();
            found.0[0].load = 0.9;
        }
        assert!(alpha.unclaimed().is_empty());
        assert!(!alpha.may_claim());
        {
            let mut found = alpha.found.lock().unwrap();
            found.0[0].load = 0.75;
        }
        assert!((alpha.spare("beta") - 0.25).abs() < 1e-6);
        assert_eq!(alpha.spare("nobody"), 1.0);
        {
            let mut found = alpha.found.lock().unwrap();
            found.0[0].load = 1.0;
        }
        assert_eq!(alpha.spare("beta"), MIN_SPARE);
    }

    fn covered(plan: &[PlanEntry]) -> usize {
        plan.iter()
            .map(|e| (e.layer_end - e.layer_start) as usize)
            .sum()
    }

    /// A pyramid over TCP — a top-level node, a middle node, and a leaf
    /// below it — answers bit for bit what the model does alone, and the
    /// nodes below turn their own APIs off.
    #[test]
    fn a_pyramid_over_tcp_is_exact() {
        let leaf = node("leaf", &[], Variant::default());
        let middle = node("middle", &[&leaf], Variant::default());
        let top = node("top", &[&middle], Variant::default());
        let plan = top.plan();
        assert_eq!(covered(&plan), N_LAYER, "{}", describe(&plan));
        assert!(
            plan.iter()
                .any(|e| e.node == "leaf" && e.layer_start < e.layer_end)
        );
        assert!(middle.is_assigned() && leaf.is_assigned() && !top.is_assigned());

        let tokens = fixture::tokens();
        let whole = run(fixture::model().as_ref(), &tokens, 5);
        let split = top.delegating_model().unwrap();
        assert_eq!(run(split.as_ref(), &tokens, 5), whole);
        assert_eq!(split.active_sessions(), 0);
        for n in [&top, &middle, &leaf] {
            n.stop();
        }
    }

    /// A worker configured with fewer slots than its parent says so in its
    /// handshake, and the parent's prefix pool keeps no more sessions there
    /// than it said — whatever the parent's own pool size. `/v1/workers`
    /// shows what the worker reported and what the pool holds.
    #[test]
    fn a_worker_with_fewer_slots_gets_no_more_pool_sessions_than_it_reported() {
        use crate::engine::prefix_cache::{ENTRIES, PrefixCache};
        use crate::engine::slot_store::SlotStore;
        let mut small = settings("small", vec![], None);
        small.slots = 1;
        let worker = node_with(small, Variant::default());
        let mut big = settings("top", vec![worker.local_addr().unwrap().to_string()], None);
        big.slots = 4;
        let top = node_with(big, Variant::default());
        assert!(worker.is_assigned());

        let info = top.links()[0].info().unwrap();
        let terms = info
            .sessions
            .expect("the worker said how it holds sequences");
        assert_eq!((terms.slots, terms.n_ctx as usize), (1, N_CTX));
        assert_eq!(info.pool_reserve(), 1);

        let split = top.delegating_model().unwrap();
        let pool = Arc::new(PrefixCache::new(ENTRIES));
        split.attach_prefix_cache(&pool);
        let slots = SlotStore::at(std::path::PathBuf::new(), String::new(), 4);
        let retained = || {
            worker
                .assignment
                .lock()
                .unwrap()
                .as_ref()
                .map_or((0, 0), |a| a.store.retained())
                .0
        };
        for (slot, start) in [(0usize, 1u32), (1, 20), (2, 40)] {
            let history: Vec<u32> = (start..start + 6).collect();
            let mut cache = split.new_kv_cache(N_CTX);
            split.forward(&mut cache, &history, 0, 0).unwrap();
            crate::engine::generate::keep_finished(
                Some(&pool),
                Some(&slots),
                slot,
                false,
                history,
                cache,
            );
            assert_eq!(retained(), 1, "after the conversation on slot {slot}");
        }
        assert_eq!(pool.stats().entries, 1);
        assert_eq!(split.pool_sessions(), 1);

        let status = top.status();
        let entry = &status["workers"][0]["sessions"];
        assert_eq!(entry["slots"], 1);
        assert_eq!(entry["pool_reserve"], 1);
        assert_eq!(entry["session_limit"], 4 * 2 + 1);
        assert_eq!(entry["retained_sessions"], 1);
        assert_eq!(status["prefix_cache"]["entries"], 1);
        assert_eq!(worker.status()["retained_sessions"]["sessions"], 1);

        drop((pool, slots));
        for n in [&top, &worker] {
            n.stop();
        }
    }

    /// A node of channel `channel` named `name` saving prefixes under
    /// `root`, with `budget` bytes for layers.
    fn saving(
        name: &str,
        channel: &str,
        peers: &[&Arc<Node>],
        root: &std::path::Path,
        budget: u64,
    ) -> Arc<Node> {
        let mut settings = on_channel(name, channel, peers, true);
        settings.capacity.budget_bytes = budget;
        settings.prefix_root = Some(root.to_path_buf());
        node_with(settings, Variant::default())
    }

    /// Two workers, neither of which holds the model alone, and a node that
    /// holds none of it and claims them both: every node saves prefixes
    /// under `root`.
    fn saving_tree(tag: &str) -> (Arc<Node>, Arc<Node>, Arc<Node>, std::path::PathBuf) {
        let root =
            std::env::temp_dir().join(format!("orangu-node-prefixes-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let need: u64 = fixture::loaded(Variant::default())
            .tensor_sizes()
            .map(|(_, bytes)| bytes)
            .sum();
        let beta = saving("beta", tag, &[], &root, need * 6 / 10);
        let gamma = saving("gamma", tag, &[], &root, need * 6 / 10);
        let alpha = saving("alpha", tag, &[&beta, &gamma], &root, 1);
        assert_eq!(found(&alpha, 2).len(), 2);
        (alpha, beta, gamma, root)
    }

    /// A first conversation on `alpha`'s tree, kept in `pool`; the tree is
    /// let go of after it.
    fn converse_and_let_go(
        alpha: &Arc<Node>,
        workers: &[&Arc<Node>],
        split: &DelegatingModel,
        pool: &crate::engine::prefix_cache::PrefixCache,
        history: &[u32],
    ) {
        use crate::engine::slot_store::SlotStore;
        let slots = SlotStore::at(std::path::PathBuf::new(), String::new(), 2);
        let request = alpha.start_request();
        assert_eq!(alpha.used.lock().unwrap().len(), 2, "{:?}", alpha.plan());
        let mut cache = split.new_kv_cache(N_CTX);
        split.forward(&mut cache, history, 0, 0).unwrap();
        crate::engine::generate::keep_finished(
            Some(pool),
            Some(&slots),
            0,
            false,
            history.to_vec(),
            cache,
        );
        assert_eq!(pool.stats().entries, 1);
        slots.clear();
        drop(request);
        until("the workers are let go", || {
            !alpha.leading.load(Ordering::Acquire) && workers.iter().all(|w| !w.is_assigned())
        });
        until("the channel finds them free", || {
            alpha.unclaimed().len() == workers.len()
        });
    }

    fn files_of(node: &Node) -> Vec<super::super::protocol::SavedPrefix> {
        node.prefix_files.as_ref().unwrap().list()
    }

    /// The pool is saved on every node as the tree is let go; the next claim
    /// of the same nodes keeps the split it was saved on, puts the entry back
    /// in the pool, and a prompt sharing its prefix runs only what is new —
    /// with the answer the model gives alone.
    #[test]
    fn a_saved_prefix_survives_a_release_and_a_new_claim() {
        use crate::engine::prefix_cache::{ENTRIES, PrefixCache};
        let (alpha, beta, gamma, root) = saving_tree("survives");
        let split = alpha.delegating_model().unwrap();
        let pool = Arc::new(PrefixCache::new(ENTRIES));
        split.attach_prefix_cache(&pool);
        let history = fixture::tokens()[..7].to_vec();
        converse_and_let_go(&alpha, &[&beta, &gamma], &split, &pool, &history);
        let manifest = alpha.prefix_files.as_ref().unwrap().manifest().unwrap();
        assert_eq!(manifest.entries.len(), 1);
        assert_eq!(manifest.workers.len(), 2);
        for worker in [&beta, &gamma] {
            let saved = files_of(worker);
            assert_eq!(saved.len(), 1, "{}", worker.settings.name);
            assert_eq!(saved[0].key, manifest.entries[0].key);
        }
        assert_eq!(beta.status()["saved_prefixes"]["entries"], 1);
        assert_eq!(pool.stats().entries, 0);

        let request = alpha.start_request();
        assert_eq!(*alpha.shares_by.lock().unwrap(), "the saved split");
        let status = alpha.status();
        assert_eq!(status["prefix_cache"]["loaded"], 1, "{status}");
        assert_eq!(status["prefix_cache"]["saved"], 1, "{status}");
        let mut prompt = history.clone();
        prompt.extend([11, 12, 13]);
        let mut next = split.new_kv_cache(N_CTX);
        assert_eq!(
            crate::engine::generate::reuse_from_pool(&pool, &prompt, &mut next),
            7
        );
        let logits = split.forward(&mut next, &prompt[7..], 7, 0).unwrap();
        let alone = fixture::model();
        let mut reference = alone.new_kv_cache(N_CTX);
        alone.forward(&mut reference, &history, 0, 0).unwrap();
        let expected = alone.forward(&mut reference, &prompt[7..], 7, 0).unwrap();
        assert_eq!(logits, expected);
        assert_eq!(split.rebuilds(), 0, "the prefix was not replayed");
        drop((next, request));
        for n in [&alpha, &beta, &gamma] {
            n.stop();
        }
        let _ = std::fs::remove_dir_all(root);
    }

    /// A node of the saved split gone: the tree is cut afresh, and the saved
    /// prefixes are deleted on every node of it and on the top-level node.
    #[test]
    fn a_tree_split_differently_drops_its_saved_prefixes() {
        use crate::engine::prefix_cache::{ENTRIES, PrefixCache};
        let (alpha, beta, gamma, root) = saving_tree("resplit");
        let split = alpha.delegating_model().unwrap();
        let pool = Arc::new(PrefixCache::new(ENTRIES));
        split.attach_prefix_cache(&pool);
        let history = fixture::tokens()[..7].to_vec();
        converse_and_let_go(&alpha, &[&beta, &gamma], &split, &pool, &history);
        assert_eq!(files_of(&beta).len(), 1);
        gamma.stop();
        until("gamma is gone from the channel", || {
            alpha.found.lock().unwrap().0.len() == 1 && alpha.unclaimed().len() == 1
        });

        let request = alpha.start_request();
        assert_ne!(*alpha.shares_by.lock().unwrap(), "the saved split");
        assert_eq!(pool.stats().entries, 0);
        assert_eq!(split.prefixes_saved(), (0, 0));
        assert!(files_of(&beta).is_empty(), "beta was told to forget");
        assert_eq!(alpha.prefix_files.as_ref().unwrap().manifest(), None);
        drop(request);
        for n in [&alpha, &beta] {
            n.stop();
        }
        let _ = std::fs::remove_dir_all(root);
    }

    /// A tree of three nodes exposes every node through the top node's
    /// Prometheus metrics, each tagged with the branch it came through.
    #[test]
    fn a_pyramids_discovery_reaches_the_top() {
        fn endpointed(name: &str, workers: Vec<String>, port: u16) -> NodeSettings {
            let mut settings = settings(name, workers, None);
            settings.metrics_endpoint = Some(("127.0.0.1".to_string(), port));
            settings
        }

        let leaf = node_with(endpointed("leaf", vec![], 9171), Variant::default());

        let middle = node_with(
            endpointed("middle", vec![leaf.local_addr().unwrap().to_string()], 9172),
            Variant::default(),
        );

        let top = node_with(
            endpointed("top", vec![middle.local_addr().unwrap().to_string()], 9173),
            Variant::default(),
        );

        // Planning connects the links; the middle hands its subtree's
        // records up with its own handshake.
        let _ = top.plan();

        let mid_addr = middle.local_addr().unwrap().to_string();

        let mut entries = Vec::new();

        for _ in 0..50 {
            entries = top.discovery_entries();

            if entries.len() == 3 {
                break;
            }

            std::thread::sleep(Duration::from_millis(100));
        }

        assert_eq!(entries.len(), 3, "{entries:?}");

        // Render the discovered tree as Prometheus metrics.
        let text = top.discovery_metrics();

        assert!(
            text.contains("# TYPE orangu_server_worker_node_info gauge"),
            "{text}"
        );

        for (name, port) in [("top", 9173), ("middle", 9172), ("leaf", 9171)] {
            let line = text
                .lines()
                .find(|line| line.contains(&format!("node=\"{name}\"")))
                .unwrap_or_else(|| panic!("no series for {name}:\n{text}"));

            assert!(
                line.contains(&format!(
                    "metrics_host=\"127.0.0.1\",metrics_port=\"{port}\""
                )),
                "{line}"
            );

            assert!(line.ends_with(" 1"), "{line}");

            if name == "top" {
                assert!(!line.contains("via="), "{line}");
            } else {
                // Both descendants sit under the top's one direct child.
                assert!(line.contains(&format!("via=\"{mid_addr}\"")), "{line}");
            }
        }

        // The middle's series carries the address the top dialed it at,
        // matching the parent's worker_up{worker=...} label.
        let middle_line = text
            .lines()
            .find(|line| line.contains("node=\"middle\""))
            .unwrap();

        assert!(
            middle_line.contains(&format!("worker=\"{mid_addr}\"")),
            "{middle_line}"
        );

        for n in [&top, &middle, &leaf] {
            n.stop();
        }
    }
    /// Shares follow memory: a worker with three times the top's budget
    /// gets three times the layers.
    #[test]
    fn shares_follow_memory() {
        let mut big = settings("big", vec![], None);
        big.capacity.budget_bytes = 3 << 30;
        let big = node_with(big, Variant::default());
        let mut top = settings("top", vec![big.local_addr().unwrap().to_string()], None);
        top.capacity.budget_bytes = 1 << 30;
        let top = node_with(top, Variant::default());
        assert_eq!(describe(&top.plan()), format!("top 0..1, big 1..{N_LAYER}"));
        big.stop();
        top.stop();
    }

    /// Several sequences at once through one pyramid — their forwards in
    /// flight together on every link, answered in whatever order they
    /// finish — each come out exactly as they would alone.
    #[test]
    fn concurrent_sequences_through_one_tree_are_each_exact() {
        let leaf = node("leaf", &[], Variant::default());
        let middle = node("middle", &[&leaf], Variant::default());
        // Room for all six: a worker holds what its parent's slots need.
        let mut settings = settings("top", vec![middle.local_addr().unwrap().to_string()], None);
        settings.slots = 6;
        let top = node_with(settings, Variant::default());
        let split = top.delegating_model().unwrap();
        let model = fixture::model();
        let sequences: Vec<Vec<u32>> = (0..6u32)
            .map(|i| (0..8u32).map(|t| (i * 7 + t * 3) % 60 + 1).collect())
            .collect();
        let want: Vec<_> = sequences
            .iter()
            .map(|s| run(model.as_ref(), s, 4))
            .collect();
        let got: Vec<_> = std::thread::scope(|scope| {
            let handles: Vec<_> = sequences
                .iter()
                .map(|s| {
                    let split = split.clone();
                    scope.spawn(move || run(split.as_ref(), s, 4))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert_eq!(got, want);
        for n in [&top, &middle, &leaf] {
            n.stop();
        }
    }

    /// A worker with another quantization is left out, and the top serves
    /// without it.
    #[test]
    fn a_worker_with_another_model_is_left_out() {
        let other = node(
            "other",
            &[],
            Variant {
                f16_ffn_down: true,
                ..Variant::default()
            },
        );
        let top = node("top", &[&other], Variant::default());
        assert!(!other.is_assigned());
        assert_eq!(describe(&top.plan()), format!("top 0..{N_LAYER}"));
        // Found by the handshake, not by an assignment it refused: the link
        // stays up, and planning again leaves it out again, quietly.
        assert!(top.links()[0].is_connected());
        top.plan_top();
        assert!(!other.is_assigned());
        assert!(top.links()[0].is_connected());
        other.stop();
        top.stop();
    }

    /// A node whose model is not split across its tree — a picture model's
    /// text encoder — plans nothing for its workers.
    #[test]
    fn a_node_that_does_not_split_assigns_nothing() {
        let worker = node("worker", &[], Variant::default());
        let mut s = settings("top", vec![worker.local_addr().unwrap().to_string()], None);
        s.split = false;
        let top = node_with(s, Variant::default());
        assert!(top.delegating_model().is_none());
        std::thread::sleep(Duration::from_millis(300));
        assert!(!worker.is_assigned());
        assert!(top.plan().is_empty());
        worker.stop();
        top.stop();
    }

    #[test]
    fn a_worker_with_another_secret_is_left_out() {
        let worker = node_with(settings("worker", vec![], Some("one")), Variant::default());
        let top = node_with(
            settings(
                "top",
                vec![worker.local_addr().unwrap().to_string()],
                Some("two"),
            ),
            Variant::default(),
        );
        assert!(!worker.is_assigned());
        assert_eq!(describe(&top.plan()), format!("top 0..{N_LAYER}"));
        let with_secret = node_with(
            settings(
                "top2",
                vec![worker.local_addr().unwrap().to_string()],
                Some("one"),
            ),
            Variant::default(),
        );
        assert!(worker.is_assigned());
        assert_eq!(covered(&with_secret.plan()), N_LAYER);
        for n in [&worker, &top, &with_secret] {
            n.stop();
        }
    }

    fn tls(
        pair: Option<&(std::path::PathBuf, std::path::PathBuf)>,
        ca: Option<&std::path::Path>,
    ) -> Option<Tls> {
        Tls::from_paths(pair.map(|(c, k)| (c.as_path(), k.as_path())), ca).unwrap()
    }

    /// A tree sharing one certificate runs over TLS with nothing else set,
    /// and answers exactly as it does in the clear.
    #[test]
    fn a_tree_runs_over_tls() {
        let dir = tempfile::tempdir().unwrap();
        let shared = crate::workers::fixture::certificate(dir.path(), "shared");
        let with_tls = |name: &str, workers: Vec<String>| {
            let mut s = settings(name, workers, Some("s"));
            s.tls = tls(Some(&shared), None);
            node_with(s, Variant::default())
        };
        let leaf = with_tls("leaf", vec![]);
        let middle = with_tls("middle", vec![leaf.local_addr().unwrap().to_string()]);
        let top = with_tls("top", vec![middle.local_addr().unwrap().to_string()]);
        assert_eq!(covered(&top.plan()), N_LAYER);
        assert!(leaf.is_assigned() && middle.is_assigned());
        let tokens = fixture::tokens();
        let whole = run(fixture::model().as_ref(), &tokens, 5);
        assert_eq!(
            run(top.delegating_model().unwrap().as_ref(), &tokens, 5),
            whole
        );
        for n in [&top, &middle, &leaf] {
            n.stop();
        }
    }

    /// A frame far past what a TLS session buffers — a prompt chunk's
    /// activations, a layer's rows — crosses a TLS link whole, and the
    /// link is timed.
    #[test]
    fn a_large_frame_crosses_tls() {
        let dir = tempfile::tempdir().unwrap();
        let shared = crate::workers::fixture::certificate(dir.path(), "shared");
        let mut s = settings("leaf", vec![], Some("s"));
        s.tls = tls(Some(&shared), None);
        let leaf = node_with(s, Variant::default());
        let link = ChildLink::with_tls(
            leaf.local_addr().unwrap().to_string(),
            Duration::from_secs(10),
            Duration::from_secs(2),
            tls(Some(&shared), None),
        );
        let info = link.connect(Some("s"), &["x".to_string()]).unwrap();
        assert!(info.link.is_some_and(|(_, bandwidth)| bandwidth > 0.0));
        assert!(link.tls());
        let data: Vec<u8> = (0..4usize << 20).map(|i| i as u8).collect();
        match link.request(&Message::Echo { data: data.clone() }) {
            Ok(Message::Echo { data: back }) => assert!(back == data),
            other => panic!("{other:?}"),
        }
        leaf.stop();
    }

    /// A worker behind TLS is left out by a parent that dials in the clear,
    /// or that trusts another certificate.
    #[test]
    fn a_tls_mismatch_leaves_the_worker_out() {
        let dir = tempfile::tempdir().unwrap();
        let worker_cert = crate::workers::fixture::certificate(dir.path(), "worker");
        let other = crate::workers::fixture::certificate(dir.path(), "other");
        let mut s = settings("worker", vec![], None);
        s.tls = tls(Some(&worker_cert), None);
        let worker = node_with(s, Variant::default());
        let addr = worker.local_addr().unwrap().to_string();

        let clear = node_with(
            settings("clear", vec![addr.clone()], None),
            Variant::default(),
        );
        assert_eq!(describe(&clear.plan()), format!("clear 0..{N_LAYER}"));
        let mut s = settings("wrong", vec![addr.clone()], None);
        s.tls = tls(None, Some(&other.0));
        let wrong = node_with(s, Variant::default());
        assert_eq!(describe(&wrong.plan()), format!("wrong 0..{N_LAYER}"));
        assert!(!worker.is_assigned());

        let mut s = settings("right", vec![addr], None);
        s.tls = tls(None, Some(&worker_cert.0));
        let right = node_with(s, Variant::default());
        assert!(worker.is_assigned());
        assert_eq!(covered(&right.plan()), N_LAYER);
        for n in [&worker, &clear, &wrong, &right] {
            n.stop();
        }
    }

    /// A second parent is turned away while the first holds the worker.
    #[test]
    fn a_worker_serves_one_parent() {
        let worker = node("worker", &[], Variant::default());
        let first = node("first", &[&worker], Variant::default());
        let second = node("second", &[&worker], Variant::default());
        assert!(first.plan().iter().any(|e| e.node == "worker"));
        assert_eq!(describe(&second.plan()), format!("second 0..{N_LAYER}"));
        for n in [&worker, &first, &second] {
            n.stop();
        }
    }

    /// Losing the leaf in the middle of a sequence costs a re-prefill, not
    /// the request: the next forward plans again without it, rebuilds the
    /// sequence, and continues with the same answers.
    #[test]
    fn a_lost_worker_is_recovered_mid_sequence() {
        let leaf = node("leaf", &[], Variant::default());
        let middle = node("middle", &[&leaf], Variant::default());
        let top = node("top", &[&middle], Variant::default());
        let tokens = fixture::tokens();
        let whole = run(fixture::model().as_ref(), &tokens, 5);
        let split = top.delegating_model().unwrap();

        let mut cache = split.new_kv_cache(N_CTX);
        let first = split.forward(&mut cache, &tokens[..5], 0, 0).unwrap();
        assert_eq!(first, whole[0]);
        leaf.stop();
        let mut worst = 0f32;
        for (pos, token) in tokens.iter().enumerate().skip(5) {
            let got = split.forward(&mut cache, &[*token], pos, 0).unwrap();
            for (a, b) in got.iter().zip(&whole[pos - 4]) {
                worst = worst.max((a - b).abs());
            }
        }
        assert!(worst < 1e-3, "{worst}");
        let plan = top.plan();
        assert_eq!(covered(&plan), N_LAYER);
        assert!(
            !plan
                .iter()
                .any(|e| e.node == "leaf" && e.layer_start < e.layer_end)
        );
        drop(cache);
        top.stop();
        middle.stop();
    }

    /// A worker lost mid-sequence, with a standby configured: the standby
    /// takes its layers and the sequence is rebuilt there, with no new plan
    /// and nothing reaching the request. When the worker comes back it is
    /// taken back, and the standby let go.
    #[test]
    fn a_standby_stands_in_for_a_lost_worker() {
        let worker = node("worker", &[], Variant::default());
        let addr = worker.local_addr().unwrap();
        let standby = node("standby", &[], Variant::default());
        let mut top_settings = settings("top", vec![addr.to_string()], None);
        top_settings.standby = vec![standby.local_addr().unwrap().to_string()];
        let top = node_with(top_settings, Variant::default());
        let tokens = fixture::tokens();
        let whole = run(fixture::model().as_ref(), &tokens, 5);
        let split = top.delegating_model().unwrap();
        assert!(top.plan().iter().any(|e| e.node == "worker"));
        assert!(!standby.is_assigned(), "a standby waits");

        let mut cache = split.new_kv_cache(N_CTX);
        split.forward(&mut cache, &tokens[..5], 0, 0).unwrap();
        worker.stop();
        let mut worst = 0f32;
        for (pos, token) in tokens.iter().enumerate().skip(5) {
            let got = split.forward(&mut cache, &[*token], pos, 0).unwrap();
            for (a, b) in got.iter().zip(&whole[pos - 4]) {
                worst = worst.max((a - b).abs());
            }
        }
        assert!(worst < 1e-3, "{worst}");
        assert!(standby.is_assigned());
        assert_eq!(top.metrics.takeovers.load(Ordering::Relaxed), 1);
        assert_eq!(top.metrics.recoveries.load(Ordering::Relaxed), 0);
        let status = top.status();
        assert_eq!(
            status["standby"][0]["standing_in_for"],
            addr.to_string(),
            "{status}"
        );
        assert!(top.plan().iter().any(|e| e.node == "standby"));
        assert!(top.unready().is_none(), "repaired, not lost");
        drop(cache);

        // Back: taken back once the node is quiet, and the standby let go.
        let mut again = settings("worker", vec![], None);
        again.listen = Some(addr.to_string());
        let back = node_with(again, Variant::default());
        let deadline = Instant::now() + Duration::from_secs(20);
        while !top.plan().iter().any(|e| e.node == "worker") {
            assert!(Instant::now() < deadline, "{}", describe(&top.plan()));
            std::thread::sleep(Duration::from_millis(50));
        }
        while standby.is_assigned() {
            assert!(Instant::now() < deadline, "the standby was not let go");
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(top.status()["standby"][0]["standing_in_for"].is_null());
        top.stop();
        back.stop();
        standby.stop();
    }

    /// A worker that does not answer when the node plans is filled in for
    /// by a standby, which takes the share the worker would have had.
    /// Switched alone, a node lets its workers go and plans nothing
    /// while it stays so, even with them there to take back; switched back,
    /// it plans them in again.
    #[test]
    fn a_node_switched_alone_lets_its_workers_go_until_switched_back() {
        let worker = node("worker", &[], Variant::default());
        let top = node("top", &[&worker], Variant::default());
        let tokens = fixture::tokens();
        let split = top.delegating_model().unwrap();
        assert!(top.plan().iter().any(|e| e.node == "worker"));
        top.set_alone(true).unwrap();
        assert_eq!(describe(&top.plan()), format!("top 0..{N_LAYER}"));
        assert_eq!(top.status()["alone"], true);
        let deadline = Instant::now() + Duration::from_secs(10);
        while worker.is_assigned() {
            assert!(Instant::now() < deadline, "the worker was not let go");
            std::thread::sleep(Duration::from_millis(50));
        }
        // Past `readmit` (200 ms here): still alone.
        std::thread::sleep(Duration::from_millis(600));
        assert_eq!(describe(&top.plan()), format!("top 0..{N_LAYER}"));
        assert!(!worker.is_assigned());
        top.set_alone(false).unwrap();
        assert!(
            top.plan().iter().any(|e| e.node == "worker"),
            "{}",
            describe(&top.plan())
        );
        assert_eq!(
            run(split.as_ref(), &tokens, 5),
            run(fixture::model().as_ref(), &tokens, 5)
        );
        // A worker cannot switch: it serves its parent's tree.
        assert!(worker.set_alone(true).is_err());
        top.stop();
        worker.stop();
    }

    /// A worker that runs slower than it measured — a mixture's experts read
    /// from disk, say — is given fewer layers once requests show it: the
    /// plan goes by the speeds they found.
    #[test]
    fn a_plan_follows_the_speeds_requests_find() {
        let with_rate = |name: &str, workers: Vec<String>| {
            let mut s = settings(name, workers, None);
            s.shares = Shares::Decode;
            s.capacity.setups = vec![NodeSetup {
                name: name.to_string(),
                decode_rate: 1.0,
                prompt_rate: 1.0,
                ..NodeSetup::default()
            }];
            node_with(s, Variant::default())
        };
        let a = with_rate("a", vec![]);
        let addr = a.local_addr().unwrap().to_string();
        let top = with_rate("top", vec![addr.clone()]);
        let _model = top.delegating_model().unwrap();
        let share = |n: &Node| -> usize {
            n.plan()
                .iter()
                .filter(|e| e.node == "a")
                .map(|e| (e.layer_end - e.layer_start) as usize)
                .sum()
        };
        let before = share(&top);
        assert!(before >= 2, "{}", describe(&top.plan()));
        assert!(top.calibration_due().is_none(), "nothing observed yet");
        // Per byte, `a` took four times as long as this node.
        let top_layers = N_LAYER - before;
        top.observed_decode(
            &[
                (
                    "top".to_string(),
                    Duration::from_millis(10 * top_layers as u64),
                ),
                (addr, Duration::from_millis(40 * before as u64)),
            ],
            40,
        );
        let why = top.calibration_due().expect("a runs slower than planned");
        assert!(why.contains("a runs at"), "{why}");
        top.plan_top();
        assert!(share(&top) < before, "{}", describe(&top.plan()));
        assert!(top.calibration_due().is_none(), "planned by what was found");
        top.stop();
        a.stop();
    }

    /// Two workers about as fast as each other, one a little ahead: moving the
    /// layers over would save almost nothing, and costs a minute on a board.
    /// The speeds are off from the plan, but it stays and nothing is assigned.
    #[test]
    fn a_plan_stays_when_another_would_save_too_little() {
        let with_rate = |name: &str, workers: Vec<String>, rate: f32| {
            let mut s = settings(name, workers, None);
            s.shares = Shares::Decode;
            s.capacity.setups = vec![NodeSetup {
                name: name.to_string(),
                decode_rate: rate,
                prompt_rate: rate,
                ..NodeSetup::default()
            }];
            node_with(s, Variant::default())
        };
        let a = with_rate("a", vec![], 2.0);
        let b = with_rate("b", vec![], 1.0);
        let (at_a, at_b) = (
            a.local_addr().unwrap().to_string(),
            b.local_addr().unwrap().to_string(),
        );
        let top = with_rate("top", vec![at_a.clone(), at_b.clone()], 1.0);
        let _model = top.delegating_model().unwrap();
        let plan = describe(&top.plan());
        let layers = |n: &str| -> u64 {
            top.plan()
                .iter()
                .filter(|e| e.node == n)
                .map(|e| u64::from(e.layer_end - e.layer_start))
                .sum()
        };
        assert_eq!((layers("a"), layers("b")), (2, 1), "{plan}");
        let plans = top.metrics.plans.load(Ordering::Relaxed);
        let plans_at_a = a.metrics.plans.load(Ordering::Relaxed);
        let generation = top.top.read().unwrap().1;
        // Per layer, `b` now runs just ahead of `a`: over twice what it was
        // planned at, enough for a look, and a hair better placed with more.
        top.observed_decode(
            &[
                (
                    "top".to_string(),
                    Duration::from_micros(10_000 * layers("top")),
                ),
                (at_a, Duration::from_micros(5_000 * layers("a"))),
                (at_b, Duration::from_micros(4_900 * layers("b"))),
            ],
            40,
        );
        assert_eq!(top.calibration_due(), None, "{plan}");
        assert_eq!(describe(&top.plan()), plan);
        assert_eq!(top.metrics.plans.load(Ordering::Relaxed), plans);
        assert_eq!(a.metrics.plans.load(Ordering::Relaxed), plans_at_a);
        assert_eq!(top.top.read().unwrap().1, generation, "sessions are kept");
        // It went by what was found, so it does not look again for the same.
        top.observed_since_plan
            .store(CALIBRATE_TOKENS, Ordering::Relaxed);
        assert_eq!(top.calibration_due(), None);
        // A plan for another reason leaves them in the same order, though `b`
        // is ahead now and the shares follow the speeds.
        top.plan_top();
        let entries = top.plan();
        let start = |n: &str| entries.iter().find(|e| e.node == n).unwrap().layer_start;
        assert!(start("a") < start("b"), "{}", describe(&entries));
        top.stop();
        a.stop();
        b.stop();
    }

    /// Workers as fast as each other stay in the order the plan has them, not
    /// the configured one: going back to it would move every layer they hold.
    #[test]
    fn a_new_plan_leaves_equal_workers_where_they_are() {
        let with_rate = |name: &str, workers: Vec<String>| {
            let mut s = settings(name, workers, None);
            s.shares = Shares::Decode;
            s.capacity.setups = vec![NodeSetup {
                name: name.to_string(),
                decode_rate: 1.0,
                prompt_rate: 1.0,
                ..NodeSetup::default()
            }];
            node_with(s, Variant::default())
        };
        let a = with_rate("a", vec![]);
        let b = with_rate("b", vec![]);
        let (at_a, at_b) = (
            a.local_addr().unwrap().to_string(),
            b.local_addr().unwrap().to_string(),
        );
        let top = with_rate("top", vec![at_a.clone(), at_b.clone()]);
        let start = |n: &str| {
            let plan = top.plan();
            plan.iter().find(|e| e.node == n).unwrap().layer_start
        };
        assert!(start("a") < start("b"), "{}", describe(&top.plan()));
        *top.used.lock().unwrap() = vec![at_b, at_a];
        top.plan_top();
        assert!(start("b") < start("a"), "{}", describe(&top.plan()));
        top.stop();
        a.stop();
        b.stop();
    }

    /// Shares follow the nodes' speeds when they differ, and each
    /// node's setup reaches `/v1/workers`.
    #[test]
    fn a_faster_node_takes_more_layers() {
        let with_rate = |name: &str, workers: Vec<String>, rate: f32| {
            let mut s = settings(name, workers, None);
            s.shares = Shares::Decode;
            s.capacity.setups = vec![NodeSetup {
                name: name.to_string(),
                decode_rate: rate,
                prompt_rate: rate * 10.0,
                ..NodeSetup::default()
            }];
            node_with(s, Variant::default())
        };
        let a = with_rate("a", vec![], 1.0);
        let b = with_rate("b", vec![], 1.0);
        let addrs = [&a, &b]
            .map(|n| n.local_addr().unwrap().to_string())
            .to_vec();
        let top = with_rate("top", addrs, 2.0);
        let tokens = fixture::tokens();
        let split = top.delegating_model().unwrap();
        assert_eq!(describe(&top.plan()), "top 0..2, a 2..3, b 3..4");
        let status = top.status();
        assert_eq!(status["shares"], "decode speed");
        assert_eq!(status["setup"]["decode_gb_per_s"], 2.0);
        assert_eq!(status["workers"][1]["setups"][0]["name"], "b");
        // What the console's topology view draws an edge from: the worker's
        // name, its link's timing and whether the link is encrypted.
        assert_eq!(status["workers"][1]["name"], "b");
        assert_eq!(status["workers"][1]["tls"], false);
        assert!(status["workers"][1]["link"]["rtt_ms"].is_number());
        assert!(status["workers"][1]["link"]["mb_per_s"].is_number());
        assert_eq!(
            run(split.as_ref(), &tokens, 5),
            run(fixture::model().as_ref(), &tokens, 5)
        );
        // One that does not say how fast it is measures itself.
        let mut s = settings("measured", vec![], None);
        s.shares = Shares::Decode;
        let measured = node_with(s, Variant::default());
        assert!(measured.settings.capacity.setups[0].decode_rate > 0.0);
        for n in [&top, &a, &b, &measured] {
            n.stop();
        }
    }

    #[test]
    fn a_standby_fills_in_at_plan_time() {
        let worker = node("worker", &[], Variant::default());
        let addr = worker.local_addr().unwrap().to_string();
        worker.stop();
        let standby = node("standby", &[], Variant::default());
        let mut top_settings = settings("top", vec![addr.clone()], None);
        top_settings.standby = vec![standby.local_addr().unwrap().to_string()];
        let top = node_with(top_settings, Variant::default());
        let tokens = fixture::tokens();
        let split = top.delegating_model().unwrap();
        assert!(
            top.plan().iter().any(|e| e.node == "standby"),
            "{}",
            describe(&top.plan())
        );
        assert_eq!(top.status()["standby"][0]["standing_in_for"], addr);
        assert_eq!(
            run(split.as_ref(), &tokens, 5),
            run(fixture::model().as_ref(), &tokens, 5)
        );
        top.stop();
        standby.stop();
    }

    /// Every node of a tree copies the prompt weights of its own layers only,
    /// by the top-level node's decision.
    #[test]
    fn each_node_copies_its_own_layers_by_the_top_s_decision() {
        let a = node("a", &[], Variant::default());
        let b = node("b", &[], Variant::default());
        let top = node("top", &[&a, &b], Variant::default());
        let _model = top.delegating_model().unwrap();
        let decided = *top.decision.get().unwrap();
        let plan = top.plan();
        assert_eq!(plan.len(), 3, "{}", describe(&plan));
        for (name, n) in [("top", &top), ("a", &a), ("b", &b)] {
            let entry = plan.iter().find(|e| e.node == name).unwrap();
            let range = entry.layer_start as usize..entry.layer_end as usize;
            assert_eq!(
                *n.copied.lock().unwrap(),
                Some((range, decided)),
                "{name} in {}",
                describe(&plan)
            );
        }
        top.stop();
        a.stop();
        b.stop();
    }

    /// A worker that goes away between requests is taken back once it
    /// returns — at the next plan, never in the middle of a sequence.
    #[test]
    fn a_returning_worker_is_taken_back() {
        let worker = node("worker", &[], Variant::default());
        let addr = worker.local_addr().unwrap();
        let top = node("top", &[&worker], Variant::default());
        let _model = top.delegating_model().unwrap();
        assert!(top.plan().iter().any(|e| e.node == "worker"));
        worker.stop();
        // Gone: the next ping notices, and the plan falls back to the top.
        let deadline = Instant::now() + Duration::from_secs(10);
        while !describe(&top.plan()).eq(&format!("top 0..{N_LAYER}")) {
            assert!(Instant::now() < deadline, "{}", describe(&top.plan()));
            std::thread::sleep(Duration::from_millis(50));
        }
        let mut again = settings("worker", vec![], None);
        again.listen = Some(addr.to_string());
        let back = node_with(again, Variant::default());
        while !top.plan().iter().any(|e| e.node == "worker") {
            assert!(Instant::now() < deadline, "{}", describe(&top.plan()));
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(back.is_assigned());
        top.stop();
        back.stop();
    }

    /// A worker lost below the top — the leaf under a middle node — is
    /// taken back by the node above it, once that node holds no session,
    /// without the top having to plan again.
    #[test]
    fn a_returning_worker_is_taken_back_below_the_top() {
        let leaf = node("leaf", &[], Variant::default());
        let addr = leaf.local_addr().unwrap();
        let middle = node("middle", &[&leaf], Variant::default());
        let top = node("top", &[&middle], Variant::default());
        let split = top.delegating_model().unwrap();
        leaf.stop();
        // The next sequence recovers without the leaf.
        let tokens = fixture::tokens();
        let whole = run(fixture::model().as_ref(), &tokens, 5);
        let got = run(split.as_ref(), &tokens, 5);
        assert_eq!(got.len(), whole.len());
        assert!(!middle.plan().iter().any(|e| e.node == "leaf"));

        let mut again = settings("leaf", vec![], None);
        again.listen = Some(addr.to_string());
        let back = node_with(again, Variant::default());
        let deadline = Instant::now() + Duration::from_secs(10);
        while !middle.plan().iter().any(|e| e.node == "leaf") {
            assert!(Instant::now() < deadline, "{}", describe(&middle.plan()));
            std::thread::sleep(Duration::from_millis(50));
        }
        assert!(back.is_assigned());
        assert_eq!(run(split.as_ref(), &tokens, 5), whole);
        for n in [&top, &middle, &back] {
            n.stop();
        }
    }

    /// A worker that stays gone costs nothing once it is planned out: the node
    /// above it does not plan again until it answers, as a plan would drop every
    /// conversation that node holds, on every node above it too.
    #[test]
    fn a_worker_that_stays_gone_is_not_planned_for_again() {
        let leaf = node("leaf", &[], Variant::default());
        let middle = node("middle", &[&leaf], Variant::default());
        let top = node("top", &[&middle], Variant::default());
        let split = top.delegating_model().unwrap();
        leaf.stop();
        let tokens = fixture::tokens();
        run(split.as_ref(), &tokens, 5);
        assert!(!middle.plan().iter().any(|e| e.node == "leaf"));
        // A conversation kept across several `readmit`s (200 ms here).
        let mut cache = split.new_kv_cache(N_CTX);
        split.forward(&mut cache, &tokens[..5], 0, 0).unwrap();
        let plans = (
            middle.metrics.plans.load(Ordering::Relaxed),
            top.metrics.plans.load(Ordering::Relaxed),
        );
        let recoveries = top.metrics.recoveries.load(Ordering::Relaxed);
        // Past `QUIET`, which a node waits for before it plans.
        std::thread::sleep(Duration::from_secs(4));
        split.forward(&mut cache, &tokens[5..6], 5, 0).unwrap();
        assert_eq!(
            plans,
            (
                middle.metrics.plans.load(Ordering::Relaxed),
                top.metrics.plans.load(Ordering::Relaxed),
            )
        );
        assert_eq!(top.metrics.recoveries.load(Ordering::Relaxed), recoveries);
        top.stop();
        middle.stop();
    }

    /// The same at the top, whose plans start every conversation over.
    #[test]
    fn a_worker_that_stays_gone_is_not_planned_for_again_by_the_top() {
        let a = node("a", &[], Variant::default());
        let b = node("b", &[], Variant::default());
        let top = node("top", &[&a, &b], Variant::default());
        let split = top.delegating_model().unwrap();
        b.stop();
        let tokens = fixture::tokens();
        run(split.as_ref(), &tokens, 5);
        assert!(!top.plan().iter().any(|e| e.node == "b"));
        std::thread::sleep(Duration::from_millis(500));
        let plans = top.metrics.plans.load(Ordering::Relaxed);
        std::thread::sleep(Duration::from_secs(2));
        assert_eq!(plans, top.metrics.plans.load(Ordering::Relaxed));
        top.stop();
        a.stop();
    }

    /// The terms `node` plans under, and it plans `layers` as a plan in progress
    /// does: its workers have their new layers, and it is not yet using them.
    fn half_planned(node: &Node, layers: Range<usize>, path: &[String], terms: Terms) {
        let planned = node.plan_subtree(layers, path, terms).unwrap();
        drop(planned);
    }

    /// A request that meets a plan in progress — a worker on its new layers, and
    /// this node's pipeline on the old ones — goes on over the new plan, and the
    /// worker is not blamed for it.
    #[test]
    fn a_request_during_a_plan_goes_on_over_the_new_one() {
        let worker = node("worker", &[], Variant::default());
        let top = node("top", &[&worker], Variant::default());
        let tokens = fixture::tokens();
        let whole = run(fixture::model().as_ref(), &tokens, 5);
        let split = top.delegating_model().unwrap();
        let mut cache = split.new_kv_cache(N_CTX);
        split.forward(&mut cache, &tokens[..5], 0, 0).unwrap();

        let terms = Terms {
            n_ctx: top.settings.n_ctx,
            slots: top.settings.slots,
            activations: top.settings.activations,
            weights: *top.decision.get().unwrap(),
        };
        half_planned(&top, 0..N_LAYER - 1, std::slice::from_ref(&top.id), terms);

        let mut worst = 0f32;
        for (pos, token) in tokens.iter().enumerate().skip(5) {
            let got = split.forward(&mut cache, &[*token], pos, 0).unwrap();
            for (a, b) in got.iter().zip(&whole[pos - 4]) {
                worst = worst.max((a - b).abs());
            }
        }
        assert!(worst < 1e-3, "{worst}");
        assert_eq!(covered(&top.plan()), N_LAYER);
        assert_eq!(top.status()["workers"][0]["failures_in_a_row"], 0);
        top.stop();
        worker.stop();
    }

    /// The same below the top: the middle node's plan is in progress, it cannot
    /// plan for its parent, and the top gets past it.
    #[test]
    fn a_request_during_a_plan_below_the_top_goes_on() {
        let leaf = node("leaf", &[], Variant::default());
        let middle = node("middle", &[&leaf], Variant::default());
        let top = node("top", &[&middle], Variant::default());
        let tokens = fixture::tokens();
        let whole = run(fixture::model().as_ref(), &tokens, 5);
        let split = top.delegating_model().unwrap();
        let mut cache = split.new_kv_cache(N_CTX);
        split.forward(&mut cache, &tokens[..5], 0, 0).unwrap();

        let (layers, terms, path) = {
            let assignment = middle.assignment.lock().unwrap();
            let assignment = assignment.as_ref().unwrap();
            (
                assignment.layers.clone(),
                assignment.terms,
                assignment.path.clone(),
            )
        };
        half_planned(&middle, layers.start..layers.end - 1, &path, terms);

        let mut worst = 0f32;
        for (pos, token) in tokens.iter().enumerate().skip(5) {
            let got = split.forward(&mut cache, &[*token], pos, 0).unwrap();
            for (a, b) in got.iter().zip(&whole[pos - 4]) {
                worst = worst.max((a - b).abs());
            }
        }
        assert!(worst < 1e-3, "{worst}");
        assert_eq!(covered(&top.plan()), N_LAYER);
        for n in [&top, &middle] {
            assert_eq!(
                n.status()["workers"][0]["failures_in_a_row"],
                0,
                "{}",
                n.settings.name
            );
            n.stop();
        }
        leaf.stop();
    }

    /// A batched decode step that loses a worker declines, with every row
    /// rolled back, and the rows then recover one by one.
    #[test]
    fn a_batched_step_that_loses_a_worker_falls_back() {
        use crate::engine::arch::DecodeRow;
        let leaf = node("leaf", &[], Variant::default());
        let middle = node("middle", &[&leaf], Variant::default());
        let top = node("top", &[&middle], Variant::default());
        let split = top.delegating_model().unwrap();
        let prompts: [&[u32]; 2] = [&[1, 2, 3], &[9, 8]];
        let mut caches: Vec<_> = prompts
            .iter()
            .map(|p| {
                let mut cache = split.new_kv_cache(N_CTX);
                split.forward(&mut cache, p, 0, 0).unwrap();
                cache
            })
            .collect();
        leaf.stop();
        let mut rows: Vec<DecodeRow<'_>> = caches
            .iter_mut()
            .zip(&prompts)
            .map(|(cache, p)| DecodeRow {
                cache,
                pos: p.len(),
                slot: 0,
            })
            .collect();
        assert!(
            split
                .forward_decode_batch(&mut rows, &[5, 6])
                .unwrap()
                .is_none()
        );
        drop(rows);
        let model = fixture::model();
        for ((cache, prompt), token) in caches.iter_mut().zip(&prompts).zip([5u32, 6]) {
            assert_eq!(cache.committed_len(), prompt.len());
            let got = split.forward(cache, &[token], prompt.len(), 0).unwrap();
            let mut alone = model.new_kv_cache(N_CTX);
            model.forward(&mut alone, prompt, 0, 0).unwrap();
            let want = model
                .forward(&mut alone, &[token], prompt.len(), 0)
                .unwrap();
            let worst = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            assert!(worst < 1e-3, "{worst}");
        }
        drop(caches);
        top.stop();
        middle.stop();
    }

    fn worst(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    /// Losing the middle node takes its whole subtree with it; the top runs
    /// every layer itself from then on, and the sequence carries on.
    #[test]
    fn a_lost_subtree_is_recovered_mid_sequence() {
        let leaf = node("leaf", &[], Variant::default());
        let middle = node("middle", &[&leaf], Variant::default());
        let top = node("top", &[&middle], Variant::default());
        let tokens = fixture::tokens();
        let whole = run(fixture::model().as_ref(), &tokens, 5);
        let split = top.delegating_model().unwrap();
        let mut cache = split.new_kv_cache(N_CTX);
        assert_eq!(
            split.forward(&mut cache, &tokens[..5], 0, 0).unwrap(),
            whole[0]
        );
        middle.stop();
        for (pos, token) in tokens.iter().enumerate().skip(5) {
            let got = split.forward(&mut cache, &[*token], pos, 0).unwrap();
            assert!(worst(&got, &whole[pos - 4]) < 1e-3);
        }
        assert_eq!(describe(&top.plan()), format!("top 0..{N_LAYER}"));
        // The leaf, whose parent went away, serves its own API again.
        let deadline = Instant::now() + Duration::from_secs(10);
        while leaf.is_assigned() {
            assert!(Instant::now() < deadline);
            std::thread::sleep(Duration::from_millis(20));
        }
        drop(cache);
        top.stop();
        leaf.stop();
    }

    /// A worker lost between two chunks of a prompt: the second chunk
    /// rebuilds the first on the new plan, and the prompt's answer is the
    /// same.
    #[test]
    fn a_worker_lost_mid_prefill_is_recovered() {
        let leaf = node("leaf", &[], Variant::default());
        let middle = node("middle", &[&leaf], Variant::default());
        let top = node("top", &[&middle], Variant::default());
        let tokens = fixture::tokens();
        let model = fixture::model();
        let mut alone = model.new_kv_cache(N_CTX);
        model
            .forward_no_logits(&mut alone, &tokens[..3], 0, 0)
            .unwrap();
        let want = model.forward(&mut alone, &tokens[3..6], 3, 0).unwrap();

        let split = top.delegating_model().unwrap();
        let mut cache = split.new_kv_cache(N_CTX);
        split
            .forward_no_logits(&mut cache, &tokens[..3], 0, 0)
            .unwrap();
        leaf.stop();
        let got = split.forward(&mut cache, &tokens[3..6], 3, 0).unwrap();
        assert!(worst(&got, &want) < 1e-3);
        assert_eq!(cache.committed_len(), 6);
        drop(cache);
        top.stop();
        middle.stop();
    }

    /// A two-level tree of a real model over TCP — top, middle, leaf,
    /// each node with its own copy of the model (any architecture that
    /// splits; the prompt is Llama token ids, nonsense to another
    /// vocabulary, where `f16` and `q8_0` then drift apart on a chaotic
    /// continuation; `ORANGU_TEST_N_CTX` sets the context a worker is
    /// assigned) —
    /// against the model alone, greedy. With `f32` on the wire every token
    /// and every logit is the same; `f16` and `q8_0` keep the tokens and
    /// stay within a small distance of the logits. Run with
    /// `ORANGU_TEST_LLAMA_MODEL=/path/to/model.gguf cargo test --release
    /// --bin orangu-server a_real_tree_is_equivalent -- --ignored
    /// --nocapture`.
    #[test]
    #[ignore]
    fn a_real_tree_is_equivalent() {
        let path = std::path::PathBuf::from(
            std::env::var("ORANGU_TEST_LLAMA_MODEL").expect("set ORANGU_TEST_LLAMA_MODEL"),
        );
        // Each node its own model and mapping, as separate processes have:
        // any architecture that splits.
        let build = |settings: NodeSettings| {
            let loaded = LoadedModel::open(&path).unwrap();
            let backend: Arc<dyn crate::engine::backend::Backend> =
                Arc::new(crate::engine::backend::CpuBackend);
            let model = crate::build_model(&loaded, &backend).unwrap();
            (Node::start(settings, model.clone(), loaded).unwrap(), model)
        };
        let greedy = |m: &dyn ModelForward| {
            let mut cache = m.new_kv_cache(N_CTX);
            let prompt = [1u32, 785, 6722, 315, 9625, 374, 12095, 13];
            let mut logits = m.forward(&mut cache, &prompt, 0, 0).unwrap();
            let (mut tokens, mut steps) = (Vec::new(), vec![logits.clone()]);
            for pos in prompt.len()..prompt.len() + 24 {
                let next = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0 as u32;
                tokens.push(next);
                logits = m.forward(&mut cache, &[next], pos, 0).unwrap();
                steps.push(logits.clone());
            }
            (tokens, steps)
        };
        let mut want = None;
        for (format, tolerance) in [
            (ActivationFormat::F32, 0.0),
            (ActivationFormat::F16, 0.5),
            (ActivationFormat::Q8_0, 1.5),
        ] {
            let with = |name: &str, workers: Vec<String>| {
                let mut s = settings(name, workers, Some("s"));
                s.activations = format;
                if let Some(n_ctx) = std::env::var("ORANGU_TEST_N_CTX")
                    .ok()
                    .and_then(|v| v.parse().ok())
                {
                    s.n_ctx = n_ctx;
                }
                s.label = "model".to_string();
                s.quant = "any".to_string();
                s
            };
            let (leaf, _) = build(with("leaf", vec![]));
            let (middle, _) = build(with("middle", vec![leaf.local_addr().unwrap().to_string()]));
            let (top, model) = build(with("top", vec![middle.local_addr().unwrap().to_string()]));
            let (want_tokens, want_steps) =
                want.get_or_insert_with(|| greedy(model.as_ref())).clone();
            let (tokens, steps) = greedy(top.delegating_model().unwrap().as_ref());
            let worst = steps
                .iter()
                .zip(&want_steps)
                .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x - y).abs()))
                .fold(0f32, f32::max);
            println!(
                "{format:?}: plan {}; worst logit difference {worst}",
                describe(&top.plan())
            );
            assert_eq!(tokens, want_tokens, "{format:?}");
            assert!(worst <= tolerance, "{format:?}: {worst} > {tolerance}");
            for n in [&top, &middle, &leaf] {
                n.stop();
            }
        }
    }

    /// Two nodes that list each other: whichever plans second finds itself
    /// below itself and refuses, so neither ends up its own ancestor.
    #[test]
    fn nodes_that_list_each_other_do_not_loop() {
        let a = node_with(settings("a", vec![], None), Variant::default());
        let mut b_settings = settings("b", vec![a.local_addr().unwrap().to_string()], None);
        b_settings.listen = Some("127.0.0.1:0".to_string());
        let b = node_with(b_settings, Variant::default());
        assert!(a.is_assigned());
        // `a` now tries to take `b`, which holds `a` below it.
        let link = ChildLink::new(
            b.local_addr().unwrap().to_string(),
            Duration::from_secs(5),
            Duration::from_secs(2),
        );
        let err = link.connect(None, &[a.id().to_string()]).unwrap_err();
        assert!(err.message.contains("loop"), "{err}");
        a.stop();
        b.stop();
    }

    /// A proxy in front of a worker that cuts the connection at the next
    /// `cuts` forwards: the worker's side first, so it has let its parent go
    /// by the time the parent finds out.
    struct Flaky {
        addr: String,
        cuts: Arc<AtomicUsize>,
        dials: Arc<AtomicUsize>,
        /// Reservations it has passed on.
        asked: Arc<AtomicUsize>,
        /// Set: the parent's frames go no further, and nothing closes. What the
        /// worker sees of a parent that lost its power.
        mute: Arc<AtomicBool>,
    }

    fn flaky(worker: &Arc<Node>, cuts: usize) -> Flaky {
        let target = worker.local_addr().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let cuts = Arc::new(AtomicUsize::new(cuts));
        let dials = Arc::new(AtomicUsize::new(0));
        let asked = Arc::new(AtomicUsize::new(0));
        let mute = Arc::new(AtomicBool::new(false));
        let muted = mute.clone();
        let (left, dialled, reservations) = (cuts.clone(), dials.clone(), asked.clone());
        std::thread::spawn(move || {
            for parent in listener.incoming() {
                let Ok(parent) = parent else { return };
                dialled.fetch_add(1, Ordering::Relaxed);
                let left = left.clone();
                let reservations = reservations.clone();
                let muted = muted.clone();
                std::thread::spawn(move || {
                    let mut to_worker = TcpStream::connect(target).unwrap();
                    let mut from_worker = to_worker.try_clone().unwrap();
                    let mut to_parent = parent.try_clone().unwrap();
                    std::thread::spawn(move || {
                        let _ = std::io::copy(&mut from_worker, &mut to_parent);
                    });
                    let mut from_parent = BufReader::new(parent.try_clone().unwrap());
                    while let Ok((id, message, _)) = read_frame(&mut from_parent) {
                        if muted.load(Ordering::Relaxed) {
                            continue;
                        }
                        if matches!(message, Message::Reserve { .. }) {
                            reservations.fetch_add(1, Ordering::Relaxed);
                        }
                        if matches!(message, Message::Forward(_) | Message::ForwardBatch(_))
                            && left
                                .try_update(Ordering::Relaxed, Ordering::Relaxed, |n| {
                                    n.checked_sub(1)
                                })
                                .is_ok()
                        {
                            break;
                        }
                        if write_message(&mut to_worker, id, &message).is_err() {
                            break;
                        }
                    }
                    let _ = to_worker.shutdown(std::net::Shutdown::Both);
                    std::thread::sleep(Duration::from_millis(100));
                    let _ = parent.shutdown(std::net::Shutdown::Both);
                });
            }
        });
        Flaky {
            addr,
            cuts,
            dials,
            asked,
            mute,
        }
    }

    fn top_over(flaky: &Flaky) -> Arc<Node> {
        node_with(
            settings("top", vec![flaky.addr.clone()], None),
            Variant::default(),
        )
    }

    /// Long enough for the watch to try a missing worker: it waits for the
    /// node to be quiet, and `readmit` is 200 ms here.
    const PAST_QUIET: Duration = Duration::from_millis(QUIET.as_millis() as u64 + 800);

    /// A worker that takes every assignment and loses every forward: the
    /// request is answered without it, and it is neither dialled nor planned
    /// for again.
    #[test]
    fn a_worker_failing_every_request_is_left_out() {
        let worker = node("worker", &[], Variant::default());
        let proxy = flaky(&worker, usize::MAX);
        let top = top_over(&proxy);
        let tokens = fixture::tokens();
        let split = top.delegating_model().unwrap();
        assert!(top.plan().iter().any(|e| e.node == "worker"));

        assert_eq!(
            run(split.as_ref(), &tokens, 5),
            run(fixture::model().as_ref(), &tokens, 5),
        );
        assert_eq!(describe(&top.plan()), format!("top 0..{N_LAYER}"));
        let dials = proxy.dials.load(Ordering::Relaxed);
        let plans = top.metrics.plans.load(Ordering::Relaxed);
        std::thread::sleep(PAST_QUIET);
        assert_eq!(proxy.dials.load(Ordering::Relaxed), dials, "dialled again");
        assert_eq!(
            top.metrics.plans.load(Ordering::Relaxed),
            plans,
            "planned again"
        );
        assert_eq!(describe(&top.plan()), format!("top 0..{N_LAYER}"));
        top.stop();
        worker.stop();
    }

    fn worker_status(node: &Node) -> serde_json::Value {
        node.status()["workers"][0].clone()
    }

    /// Runs a request and reports it finished, as the engine does.
    fn request(model: &Arc<DelegatingModel>, tokens: &[u32]) {
        let mut cache = model.new_kv_cache(N_CTX);
        model.forward(&mut cache, &tokens[..5], 0, 0).unwrap();
        for (pos, token) in tokens.iter().enumerate().skip(5) {
            model.forward(&mut cache, &[*token], pos, 0).unwrap();
        }
        cache.request_finished();
    }

    #[test]
    fn the_status_says_who_is_left_out() {
        let worker = node("worker", &[], Variant::default());
        let proxy = flaky(&worker, usize::MAX);
        let top = top_over(&proxy);
        let split = top.delegating_model().unwrap();
        assert_eq!(worker_status(&top)["failures_in_a_row"], 0);
        assert!(worker_status(&top)["left_out_seconds"].is_null());
        request(&split, &fixture::tokens());
        assert_eq!(worker_status(&top)["failures_in_a_row"], 2);
        let left = worker_status(&top)["left_out_seconds"].as_u64().unwrap();
        assert!((60..=120).contains(&left), "{left}");
        assert!(top.unready().is_none(), "serving without it");
        top.stop();
        worker.stop();
    }

    /// Dies once a request and comes back: the replay finishes each request,
    /// and is no reason to forgive it.
    #[test]
    fn a_request_finished_by_a_replay_does_not_forgive() {
        let worker = node("worker", &[], Variant::default());
        let proxy = flaky(&worker, 1);
        let top = top_over(&proxy);
        let tokens = fixture::tokens();
        let split = top.delegating_model().unwrap();
        request(&split, &tokens);
        assert_eq!(worker_status(&top)["failures_in_a_row"], 1);
        assert!(top.plan().iter().any(|e| e.node == "worker"));

        proxy.cuts.store(1, Ordering::Relaxed);
        request(&split, &tokens);
        assert_eq!(worker_status(&top)["failures_in_a_row"], 2);
        assert!(worker_status(&top)["left_out_seconds"].is_u64());
        assert_eq!(describe(&top.plan()), format!("top 0..{N_LAYER}"));
        top.stop();
        worker.stop();
    }

    #[test]
    fn a_request_served_whole_forgives() {
        let worker = node("worker", &[], Variant::default());
        let proxy = flaky(&worker, 1);
        let top = top_over(&proxy);
        let tokens = fixture::tokens();
        let split = top.delegating_model().unwrap();
        request(&split, &tokens);
        assert_eq!(worker_status(&top)["failures_in_a_row"], 1);
        request(&split, &tokens);
        assert_eq!(worker_status(&top)["failures_in_a_row"], 0);

        proxy.cuts.store(1, Ordering::Relaxed);
        request(&split, &tokens);
        assert_eq!(worker_status(&top)["failures_in_a_row"], 1, "starts over");
        assert!(worker_status(&top)["left_out_seconds"].is_null());
        top.stop();
        worker.stop();
    }

    /// Its wait over, a worker is taken back between requests; failing again
    /// it is left out for longer, its failures still counted.
    #[test]
    fn a_worker_is_taken_back_when_its_wait_ends() {
        let worker = node("worker", &[], Variant::default());
        let proxy = flaky(&worker, usize::MAX);
        let top = top_over(&proxy);
        let tokens = fixture::tokens();
        let split = top.delegating_model().unwrap();
        request(&split, &tokens);
        assert!(worker_status(&top)["left_out_seconds"].is_u64());

        top.links()[0].end_wait();
        let deadline = Instant::now() + Duration::from_secs(10);
        while !top.plan().iter().any(|e| e.node == "worker") {
            assert!(Instant::now() < deadline, "not taken back");
            std::thread::sleep(Duration::from_millis(50));
        }
        request(&split, &tokens);
        assert_eq!(worker_status(&top)["failures_in_a_row"], 3);
        let left = worker_status(&top)["left_out_seconds"].as_u64().unwrap();
        assert!((120..=240).contains(&left), "{left}");
        assert_eq!(describe(&top.plan()), format!("top 0..{N_LAYER}"));

        // Healthy by the next try: it serves, and that forgives it.
        proxy.cuts.store(0, Ordering::Relaxed);
        top.links()[0].end_wait();
        let deadline = Instant::now() + Duration::from_secs(10);
        while !top.plan().iter().any(|e| e.node == "worker") {
            assert!(Instant::now() < deadline, "not taken back");
            std::thread::sleep(Duration::from_millis(50));
        }
        request(&split, &tokens);
        assert_eq!(worker_status(&top)["failures_in_a_row"], 0);
        top.stop();
        worker.stop();
    }

    /// A worker left out with no request running still leaves the plan: the
    /// node would say it is short of a worker for as long as the wait lasts.
    #[test]
    fn a_worker_left_out_while_idle_leaves_the_plan() {
        let worker = node("worker", &[], Variant::default());
        let top = node("top", &[&worker], Variant::default());
        assert!(top.plan().iter().any(|e| e.node == "worker"));
        top.links()[0].count_failure();
        top.links()[0].count_failure();
        top.links()[0].disconnect();
        assert!(top.unready().is_some());
        let deadline = Instant::now() + Duration::from_secs(10);
        while top.unready().is_some() {
            assert!(Instant::now() < deadline, "still in the plan");
            std::thread::sleep(Duration::from_millis(50));
        }
        assert_eq!(describe(&top.plan()), format!("top 0..{N_LAYER}"));
        top.stop();
        worker.stop();
    }

    /// A worker that says it is busy is working, and comes back when free.
    #[test]
    fn a_refusal_is_no_failure() {
        let worker = node("worker", &[], Variant::default());
        let first = node("first", &[&worker], Variant::default());
        let second = node("second", &[&worker], Variant::default());
        assert!(first.plan().iter().any(|e| e.node == "worker"));
        assert_eq!(describe(&second.plan()), format!("second 0..{N_LAYER}"));
        std::thread::sleep(Duration::from_millis(800));
        assert_eq!(worker_status(&second)["failures_in_a_row"], 0);
        assert!(worker_status(&second)["left_out_seconds"].is_null());
        for n in [&second, &first, &worker] {
            n.stop();
        }
    }

    /// A worker failing below the top is left out by the node above it. That
    /// node stays in the plan: its connection never failed.
    #[test]
    fn a_failing_worker_below_the_top_is_left_out_there() {
        let leaf = node("leaf", &[], Variant::default());
        let proxy = flaky(&leaf, usize::MAX);
        let middle = node_with(
            settings("middle", vec![proxy.addr.clone()], None),
            Variant::default(),
        );
        let top = node("top", &[&middle], Variant::default());
        let tokens = fixture::tokens();
        let split = top.delegating_model().unwrap();
        assert!(top.plan().iter().any(|e| e.node == "leaf"));
        assert_eq!(
            run(split.as_ref(), &tokens, 5),
            run(fixture::model().as_ref(), &tokens, 5),
        );
        assert!(worker_status(&middle)["left_out_seconds"].is_u64());
        assert_eq!(worker_status(&top)["failures_in_a_row"], 0);
        let plan = top.plan();
        assert!(plan.iter().any(|e| e.node == "middle"), "{plan:?}");
        assert!(!plan.iter().any(|e| e.node == "leaf"), "{plan:?}");
        let dials = proxy.dials.load(Ordering::Relaxed);
        std::thread::sleep(PAST_QUIET);
        assert_eq!(proxy.dials.load(Ordering::Relaxed), dials, "dialled again");
        top.stop();
        middle.stop();
        leaf.stop();
    }

    /// A worker refusing a prompt for memory, over a real connection and on
    /// this machine's real memory: nobody has failed, the connection stays,
    /// the plan stands, and the next request runs.
    #[test]
    fn a_prompt_refused_for_memory_costs_the_worker_nothing() {
        let worker = node("worker", &[], Variant::default());
        let mut with = settings("top", vec![worker.local_addr().unwrap().to_string()], None);
        with.local_layers = LocalLayers::Count(0);
        let top = node_with(with, Variant::default());
        let split = top.delegating_model().unwrap();
        let tokens = fixture::tokens();
        let plan = describe(&top.plan());
        let plans = top.metrics.plans.load(Ordering::Relaxed);

        let cache = split.new_kv_cache(N_CTX);
        let error = split.admit_prompt(&cache, u32::MAX as usize).unwrap_err();
        assert!(error.is::<crate::engine::generate::NoRoomForPrompt>());
        assert!(error.to_string().contains("OutOfMemory"), "{error}");
        drop(cache);

        assert_eq!(worker_status(&top)["failures_in_a_row"], 0);
        assert_eq!(worker_status(&top)["connected"], true);
        assert_eq!(describe(&top.plan()), plan);
        assert_eq!(top.metrics.plans.load(Ordering::Relaxed), plans);
        assert_eq!(top.metrics.recoveries.load(Ordering::Relaxed), 0);
        assert!(top.status()["kv_refusals"].as_u64().unwrap() > 0);
        request(&split, &tokens);
        top.stop();
        worker.stop();
    }

    /// A worker lost after the promise: the rebuild on the new plan asks
    /// again, for the same length.
    #[test]
    fn a_rebuilt_sequence_asks_for_its_room_again() {
        let worker = node("worker", &[], Variant::default());
        let proxy = flaky(&worker, 1);
        let top = top_over(&proxy);
        let tokens = fixture::tokens();
        let split = top.delegating_model().unwrap();
        let mut cache = split.new_kv_cache(N_CTX);
        split.admit_prompt(&cache, 6).unwrap();
        assert_eq!(proxy.asked.load(Ordering::Relaxed), 1);
        split.forward(&mut cache, &tokens[..5], 0, 0).unwrap();
        assert_eq!(proxy.asked.load(Ordering::Relaxed), 2);
        top.stop();
        worker.stop();
    }

    const QUICK_SILENCE: Duration = Duration::from_millis(600);

    /// A node that gives a silent parent [`QUICK_SILENCE`]. The others get
    /// longer, as a test on a busy machine can stall that long.
    fn quick(name: &str, workers: &[&Arc<Node>]) -> Arc<Node> {
        let mut with = settings(
            name,
            workers
                .iter()
                .map(|w| w.local_addr().unwrap().to_string())
                .collect(),
            None,
        );
        with.parent_silence = QUICK_SILENCE;
        node_with(with, Variant::default())
    }

    fn until(what: &str, done: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !done() {
            assert!(Instant::now() < deadline, "{what}");
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn parent_link(worker: &Arc<Node>) -> ChildLink {
        ChildLink::with_tls(
            worker.local_addr().unwrap().to_string(),
            Duration::from_secs(30),
            Duration::from_secs(2),
            None,
        )
    }

    /// The parent did its handshake and has said nothing since: the worker
    /// stops waiting, and another parent can have it.
    #[test]
    fn a_parent_gone_silent_is_let_go() {
        let worker = quick("worker", &[]);
        let first = parent_link(&worker);
        first.connect(None, &["first".to_string()]).unwrap();
        let second = parent_link(&worker);
        let refused = second.connect(None, &["second".to_string()]).unwrap_err();
        assert_eq!(refused.code, ErrorCode::Busy);
        until("still held by the first", || {
            second.connect(None, &["second".to_string()]).is_ok()
        });
        worker.stop();
    }

    /// Pings alone keep a parent, however long it asks for nothing else.
    #[test]
    fn a_parent_that_only_pings_is_kept() {
        let worker = quick("worker", &[]);
        let first = parent_link(&worker);
        first.connect(None, &["first".to_string()]).unwrap();
        let until_then = Instant::now() + QUICK_SILENCE * 3;
        while Instant::now() < until_then {
            assert_eq!(first.ping_if_idle(), Some(true));
            std::thread::sleep(Duration::from_millis(50));
        }
        let second = parent_link(&worker);
        let refused = second.connect(None, &["second".to_string()]).unwrap_err();
        assert_eq!(refused.code, ErrorCode::Busy);
        worker.stop();
    }

    /// `node` as a channel probe finds it at `addr`.
    fn answered(node: &str, addr: &str) -> super::super::channel::Found {
        super::super::channel::Found {
            addr: addr.to_string(),
            node: node.to_string(),
            name: node.to_string(),
            state: ProbeState::Leading,
            load: 0.0,
            whole: false,
            api: format!("http://{addr}"),
            model: [0; 32],
            content: [0; 32],
            workers: 1,
            label: String::new(),
            quant: String::new(),
            features: 0,
            pace: Default::default(),
            demand: Vec::new(),
            kind: Default::default(),
            capabilities: Vec::new(),
            rtt: Duration::from_millis(1),
            alternates: Vec::new(),
        }
    }

    /// Makes the parent's connection look quiet for `for_`.
    fn quiet_parent(worker: &Node, for_: Duration) {
        let mut parent = worker.parent.lock().unwrap();
        let parent = parent.as_mut().unwrap();
        parent.heard = Instant::now().checked_sub(for_).unwrap();
    }

    fn held_by(worker: &Node) -> Option<String> {
        worker
            .parent
            .lock()
            .unwrap()
            .as_ref()
            .map(|p| p.node.clone())
    }

    /// A parent seen on the channel that stops answering probes, with its
    /// connection quiet too, is let go after [`PARENT_MISSES`] probes rather
    /// than after the whole silence.
    #[test]
    fn a_parent_the_channel_stops_finding_is_let_go_early() {
        let worker = node("worker", &[], Variant::default());
        let first = parent_link(&worker);
        first.connect(None, &["first".to_string()]).unwrap();
        worker.check_parent(&[answered("first", "10.0.0.1:8400")]);
        assert_eq!(
            worker.parent_addrs(),
            vec!["10.0.0.1:8400".parse().unwrap()]
        );
        quiet_parent(&worker, Duration::from_secs(29));
        worker.check_parent(&[]);
        let status = worker.status();
        assert_eq!(status["parent"]["suspect"], true);
        assert_eq!(status["parent"]["missed_probes"], 1);
        assert_eq!(status["parent"]["name"], "first");
        for _ in 1..PARENT_MISSES {
            assert_eq!(held_by(&worker).as_deref(), Some("first"));
            worker.check_parent(&[]);
        }
        until("still held by the first", || held_by(&worker).is_none());
        let second = parent_link(&worker);
        second.connect(None, &["second".to_string()]).unwrap();
        worker.stop();
    }

    /// Unanswered probes alone keep a parent that still talks on its
    /// connection, and one answer clears the suspicion.
    #[test]
    fn a_parent_heard_on_its_connection_is_kept() {
        let worker = node("worker", &[], Variant::default());
        let first = parent_link(&worker);
        first.connect(None, &["first".to_string()]).unwrap();
        worker.check_parent(&[answered("first", "10.0.0.1:8400")]);
        for _ in 0..PARENT_MISSES * 2 {
            worker.check_parent(&[]);
        }
        assert!(worker.parent_suspect());
        assert_eq!(first.ping_if_idle(), Some(true));
        assert_eq!(held_by(&worker).as_deref(), Some("first"));
        worker.check_parent(&[answered("first", "10.0.0.1:8400")]);
        assert!(!worker.parent_suspect());
        worker.stop();
    }

    /// A parent never seen on the channel is left to the silence, however
    /// many scans miss it.
    #[test]
    fn a_parent_off_the_channel_is_left_to_the_silence() {
        let worker = node("worker", &[], Variant::default());
        let first = parent_link(&worker);
        first.connect(None, &["first".to_string()]).unwrap();
        quiet_parent(&worker, Duration::from_secs(29));
        for _ in 0..PARENT_MISSES * 2 {
            worker.check_parent(&[answered("other", "10.0.0.2:8400")]);
        }
        assert!(!worker.parent_suspect());
        assert_eq!(held_by(&worker).as_deref(), Some("first"));
        worker.stop();
    }

    /// A request that could not be forwarded to the parent's API makes it
    /// suspect and asks the channel again at once; one sent elsewhere does
    /// not.
    #[test]
    fn a_failed_forward_to_the_parent_makes_it_suspect() {
        let worker = node("worker", &[], Variant::default());
        let first = parent_link(&worker);
        first.connect(None, &["first".to_string()]).unwrap();
        worker.found.lock().unwrap().0 = vec![answered("first", "10.0.0.1:8400")];
        worker.forward_failed("http://10.0.0.9:8400");
        assert!(!worker.parent_suspect());
        worker.forward_failed("http://10.0.0.1:8400");
        assert!(worker.parent_suspect());
        assert!(worker.refresh_now.load(Ordering::Acquire));
        worker.stop();
    }

    /// Nothing closes when a parent loses power. The worker gives up its
    /// layers and serves its own API again.
    #[test]
    fn a_worker_lets_go_of_its_layers_when_its_parent_goes_silent() {
        let worker = quick("worker", &[]);
        let proxy = flaky(&worker, 0);
        let top = top_over(&proxy);
        assert!(top.plan().iter().any(|e| e.node == "worker"));
        assert!(worker.is_assigned());
        std::thread::sleep(QUICK_SILENCE * 2);
        assert!(worker.is_assigned(), "the parent pings, through the proxy");
        proxy.mute.store(true, Ordering::Relaxed);
        until("still assigned", || !worker.is_assigned());
        assert!(worker.unready().is_none(), "{:?}", worker.unready());
        top.stop();
        worker.stop();
    }

    /// A node below the top whose parent goes silent is top-level again, over
    /// the workers it has, as it is when a parent closes.
    #[test]
    fn a_middle_node_goes_on_with_its_workers_when_its_parent_goes_silent() {
        let leaf = node("leaf", &[], Variant::default());
        let middle = quick("middle", &[&leaf]);
        let proxy = flaky(&middle, 0);
        let top = top_over(&proxy);
        assert!(top.plan().iter().any(|e| e.node == "leaf"));
        proxy.mute.store(true, Ordering::Relaxed);
        until("still assigned", || !middle.is_assigned());
        until("not planned again", || covered(&middle.plan()) == N_LAYER);
        assert!(middle.plan().iter().any(|e| e.node == "leaf"));
        top.stop();
        middle.stop();
        leaf.stop();
    }

    /// One worker takes seconds over its assignment, as one fetching its
    /// layers does: the workers waiting while that happens go on hearing from
    /// their parent, which has not even started yet.
    #[test]
    fn a_long_plan_does_not_silence_the_parent() {
        let patient = quick("patient", &[]);
        assert_eq!(patient.next_connection.load(Ordering::Relaxed), 1);
        // Does the handshake, sits on the assignment, and fails it.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let slow = listener.local_addr().unwrap().to_string();
        std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let mut writer = stream.try_clone().unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            read_message(&mut reader).unwrap();
            let budget = 1u64 << 30;
            let ack = Message::HelloAck(Box::new(HelloAck {
                version: PROTOCOL_VERSION,
                features: 0,
                nonce: [0; 32],
                proof: [0; 32],
                node: "slow".to_string(),
                subtree: vec!["slow".to_string()],
                capacity: Capacity {
                    budget_bytes: budget,
                    subtree_budget_bytes: budget,
                    ..Capacity::default()
                },
                model: None,
                discovery: Vec::new(),
                sessions: None,
                saved: Vec::new(),
            }));
            write_message(&mut writer, 0, &ack).unwrap();
            read_message(&mut reader).unwrap();
            write_message(&mut writer, 0, &Message::Ready).unwrap();
            read_message(&mut reader).unwrap();
            std::thread::sleep(QUICK_SILENCE * 3);
            drop(stream);
        });
        let started = Instant::now();
        let top = node_with(
            settings(
                "top",
                vec![patient.local_addr().unwrap().to_string(), slow],
                None,
            ),
            Variant::default(),
        );
        assert!(started.elapsed() >= QUICK_SILENCE * 3, "not held up");
        assert_eq!(
            patient.next_connection.load(Ordering::Relaxed),
            2,
            "it went back for the worker it had lost"
        );
        assert!(top.plan().iter().any(|e| e.node == "patient"));
        top.stop();
        patient.stop();
    }

    /// A standby standing in is a worker like any other, and hears the same.
    #[test]
    fn a_standby_standing_in_goes_on_hearing_from_its_parent() {
        let worker = node("worker", &[], Variant::default());
        let standby = quick("standby", &[]);
        let mut with = settings("top", vec![worker.local_addr().unwrap().to_string()], None);
        with.standby = vec![standby.local_addr().unwrap().to_string()];
        let top = node_with(with, Variant::default());
        let tokens = fixture::tokens();
        let split = top.delegating_model().unwrap();
        let mut cache = split.new_kv_cache(N_CTX);
        split.forward(&mut cache, &tokens[..5], 0, 0).unwrap();
        worker.stop();
        split.forward(&mut cache, &tokens[5..6], 5, 0).unwrap();
        assert!(standby.is_assigned());
        std::thread::sleep(QUICK_SILENCE * 3);
        assert!(standby.is_assigned(), "let go for silence");
        top.stop();
        standby.stop();
    }

    fn server(name: &str, label: &str, quant: &str, free: bool) -> Server {
        Server {
            name: name.to_string(),
            label: label.to_string(),
            quant: quant.to_string(),
            whole: true,
            free,
        }
    }

    /// A request counts as demand only when no free node serves what it
    /// names: under the serving node's name for the model, or as asked
    /// when no node serves it. Roles and blank names count as nothing.
    #[test]
    fn a_request_counts_as_demand_only_when_no_free_node_serves_it() {
        let a = "org/a-GGUF:Q4_K_M";
        let busy_a = server("alpha:8400", a, "Q4_K_M", false);
        let free_a = server("beta:8400", a, "Q4_K_M", true);
        let other = server("gamma:8400", "org/b-GGUF", "Q8_0", true);
        assert_eq!(
            unserved_name("org/a-gguf", &[busy_a.clone(), other.clone()]),
            Some(a.to_string())
        );
        assert_eq!(unserved_name("org/a-GGUF", &[busy_a.clone(), free_a]), None);
        assert_eq!(
            unserved_name(" gpt-4 ", &[busy_a.clone(), other.clone()]),
            Some("gpt-4".to_string())
        );
        // A node with only part of the model serves nothing alone.
        let partial = Server {
            whole: false,
            ..server("delta:8400", "org/b-GGUF", "Q8_0", true)
        };
        assert_eq!(
            unserved_name("org/b-GGUF", &[partial]),
            Some("org/b-GGUF".to_string())
        );
        assert_eq!(unserved_name("org/b-GGUF:Q8_0", &[other]), None);
        for nothing in ["", "  ", "explorer", "ALL", "embeddings"] {
            assert_eq!(
                unserved_name(nothing, std::slice::from_ref(&busy_a)),
                None,
                "{nothing:?}"
            );
        }
    }

    /// A model's row lists every node serving it and how many are free.
    #[test]
    fn a_demand_row_counts_the_free_nodes_serving_it() {
        let servers = [
            server("alpha:8400", "org/a-GGUF", "Q4_K_M", false),
            server("beta:8400", "org/a-GGUF", "Q4_K_M", true),
            server("gamma:8400", "org/b-GGUF", "Q8_0", true),
        ];
        let row = demand_row("org/a-GGUF:Q4_K_M", 7, 2, &servers);
        assert_eq!(row["wanted"], 7);
        assert_eq!(row["here"], 2);
        assert_eq!(row["nodes"], serde_json::json!(["alpha:8400", "beta:8400"]));
        assert_eq!(row["free"], 1);
        let row = demand_row("gpt-4", 1, 0, &servers);
        assert_eq!(row["nodes"], serde_json::json!([]));
        assert_eq!(row["free"], 0);
    }
}
