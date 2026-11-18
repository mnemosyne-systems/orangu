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

//! The wire protocol between a node and its workers.
//!
//! One persistent connection per parent/child pair carries frames:
//!
//! ```text
//! [len: u32 LE][type: u8][id: u64 LE][payload: len - 9 bytes]
//! ```
//!
//! `id` pairs an answer with its request: a parent has several requests in
//! flight on one connection — one per sequence it is running — and a
//! worker answers each as soon as it is done, in whatever order that is.
//! The handshake, before any of that, uses `0`.
//!
//! Integers are little-endian; a string or byte blob is a `u32` length and
//! its bytes; a list is a `u32` count and its items. A decoder reads the
//! fields it knows and **ignores anything after them**, so a newer peer can
//! append fields to a message without breaking an older one — gated, on the
//! sending side, by the `features` bits the peer advertised in the handshake.
//! An unknown message *type* is an error the receiver answers with
//! [`ErrorCode::Unsupported`].

use anyhow::{Result, anyhow, bail, ensure};
use std::io::{Read, Write};

/// Bumped on any change an older peer cannot ignore. Peers with different
/// versions refuse each other in the handshake.
pub const PROTOCOL_VERSION: u16 = 2;

/// Optional extensions this build understands, one bit each: a parent
/// sends a child a message of an extension only when the child's
/// `HelloAck` advertised its bit, so peers without it still talk.
pub const FEATURES: u64 = FEATURE_FORK
    | FEATURE_HEAD
    | FEATURE_ECHO
    | FEATURE_ROOM
    | FEATURE_METRICS
    | FEATURE_RETAIN
    | FEATURE_PREFIXES
    | FEATURE_ROWS;

/// [`Message::Fork`]: a worker copies one session's rows into another.
pub const FEATURE_FORK: u64 = 1;

// Bits 2 and 32, and message tags 18, 19 and 21, are unused: never reuse them.

/// [`Rows::LastLogits`] and [`Rows::AllLogits`]: the node running the final
/// layer sends back logits.
pub const FEATURE_HEAD: u64 = 4;

/// [`Message::Echo`]: a worker sends back what it was sent, for its parent
/// to time the link.
pub const FEATURE_ECHO: u64 = 8;

/// This worker switches to its parent's model when assigned another one; a
/// worker without it, and another model, is left out before any layers are
/// cut. Advertised per node, not in [`FEATURES`].
pub const FEATURE_SWITCH: u64 = 16;

/// It answers [`Message::Reserve`]. Without it a prompt goes down unasked.
pub const FEATURE_ROOM: u64 = 64;

/// A `HelloAck` that ends with a list of [`DiscoveryRecord`]s: each node
/// in the tree reports where Prometheus can scrape it. A child sends the
/// list only when the parent's `Hello` advertised this bit. The list is
/// written at the very end of the message, so older versions of
/// orangu-server — which stop reading when the message runs out — lose
/// nothing and never see it. A parent that advertises [`FEATURE_ROOM`] but
/// not this bit is sent the list too: some builds advertised both
/// extensions under bit 64.
pub const FEATURE_METRICS: u64 = 128;

/// [`Message::Retain`]: a worker keeps a session its parent's prefix pool
/// holds until the parent releases it, past its own idle and shortfall
/// eviction.
pub const FEATURE_RETAIN: u64 = 256;

/// [`Message::Save`], [`Message::Load`] and [`Message::Forget`]: a worker
/// writes a session's rows to its own disk under a key its parent chose,
/// opens a session from them again, and deletes them. A `HelloAck` sent to a
/// parent that advertised this bit ends with the [`SavedPrefix`]es the node
/// holds.
pub const FEATURE_PREFIXES: u64 = 512;

/// This node serves byte ranges of its model file ([`Message::FetchHello`],
/// [`Message::FetchRange`]) to a node of `[workers].download = range`: it
/// holds the whole file, a single one. Advertised per node, in a
/// [`Message::ProbeAck`] and a [`Message::FetchAck`], not in [`FEATURES`]; a
/// node that did not advertise it is never asked for bytes.
pub const FEATURE_FETCH: u64 = 1024;

/// [`Message::Prefill`], [`Message::RowsOf`], [`Message::Put`] and
/// [`Message::Open`]: a node takes rows computed elsewhere into its layers
/// and sends the slices of the workers below on, and — when it holds the
/// whole model — runs a prompt through every layer alone and hands out the
/// rows. A parent sends none of them to a child without it.
pub const FEATURE_ROWS: u64 = 2048;

/// The most bytes one [`Message::FetchRange`] may ask for: a larger range
/// is asked for in pieces.
pub const MAX_FETCH_BYTES: u64 = 8 << 20;

/// Every extension bit, each a distinct power of two.
const ALL_FEATURES: [u64; 10] = [
    FEATURE_FORK,
    FEATURE_HEAD,
    FEATURE_ECHO,
    FEATURE_SWITCH,
    FEATURE_ROOM,
    FEATURE_METRICS,
    FEATURE_RETAIN,
    FEATURE_PREFIXES,
    FEATURE_FETCH,
    FEATURE_ROWS,
];

const _: () = {
    let mut seen = 0u64;
    let mut i = 0;
    while i < ALL_FEATURES.len() {
        let bit = ALL_FEATURES[i];
        assert!(bit.is_power_of_two(), "a feature is one bit");
        assert!(seen & bit == 0, "two features share a bit");
        seen |= bit;
        i += 1;
    }
};

/// Whether a parent that advertised `features` reads the discovery list
/// at the end of a `HelloAck`.
pub fn parent_reads_discovery(features: u64) -> bool {
    features & (FEATURE_METRICS | FEATURE_ROOM) != 0
}

/// The extensions `features` advertises, by name, for `/v1/workers`.
pub fn feature_names(features: u64) -> Vec<&'static str> {
    [
        (FEATURE_FORK, "fork"),
        (FEATURE_HEAD, "head"),
        (FEATURE_ECHO, "echo"),
        (FEATURE_SWITCH, "switch"),
        (FEATURE_ROOM, "room"),
        (FEATURE_METRICS, "metrics"),
        (FEATURE_RETAIN, "retain"),
        (FEATURE_PREFIXES, "prefixes"),
        (FEATURE_FETCH, "fetch"),
        (FEATURE_ROWS, "rows"),
    ]
    .into_iter()
    .filter(|(bit, _)| features & bit != 0)
    .map(|(_, name)| name)
    .collect()
}

/// The largest frame either side accepts: a prefill chunk of 8192 tokens
/// of an 8192-wide model in `f32` is 256 MiB, so this leaves room without
/// letting a corrupt length prefix allocate without bound.
pub const MAX_FRAME: usize = 1 << 30;

/// The largest frame of a handshake, which is all a peer that has proved
/// nothing can send. Well above what a large tree's reply takes.
pub const MAX_HANDSHAKE_FRAME: usize = 1 << 20;

/// The room made for a frame before any of it has arrived. Beyond this it
/// grows with what is received, so a length alone costs nothing.
const UP_FRONT: usize = 1 << 20;

/// How deep a tree of workers may be. A loop the path check cannot see —
/// the same node reached under two spellings of its address — still ends
/// here rather than recursing until something runs out.
pub const MAX_DEPTH: usize = 16;

/// How hidden states travel: `[workers].activations`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ActivationFormat {
    /// Exact: a split model computes bit for bit what an unsplit one does.
    F32,
    /// Half the bytes of `f32`, and the default.
    #[default]
    F16,
    /// Blocks of 32 values sharing an `f16` scale: a quarter of `f32`, at
    /// a small, measurable cost in accuracy.
    Q8_0,
}

impl ActivationFormat {
    pub const NAMES: [&str; 3] = ["f32", "f16", "q8_0"];

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "f32" => Some(Self::F32),
            "f16" => Some(Self::F16),
            "q8_0" => Some(Self::Q8_0),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Q8_0 => "q8_0",
        }
    }

    fn tag(self) -> u8 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q8_0 => 2,
        }
    }

    fn from_tag(tag: u8) -> Result<Self> {
        match tag {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q8_0),
            other => bail!("unknown activation format {other}"),
        }
    }
}

const Q8_BLOCK: usize = 32;
/// An `f16` scale and 32 signed bytes.
const Q8_BLOCK_BYTES: usize = 2 + Q8_BLOCK;

/// `rows × width` values of the residual stream, encoded.
#[derive(Clone, Debug, PartialEq)]
pub struct Activations {
    pub format: ActivationFormat,
    pub rows: u32,
    pub width: u32,
    pub data: Vec<u8>,
}

impl Activations {
    /// The bytes one row of `width` values takes in `format`, when rows
    /// can be cut apart: always for `f32` and `f16`, and for `q8_0` when a
    /// row is whole blocks. `None` when a `q8_0` block straddles two rows.
    pub fn bytes_per_row(format: ActivationFormat, width: usize) -> Option<usize> {
        match format {
            ActivationFormat::F32 => Some(width * 4),
            ActivationFormat::F16 => Some(width * 2),
            ActivationFormat::Q8_0 => width
                .is_multiple_of(Q8_BLOCK)
                .then_some(width / Q8_BLOCK * Q8_BLOCK_BYTES),
        }
    }

    pub fn encode(values: &[f32], rows: usize, width: usize, format: ActivationFormat) -> Self {
        assert_eq!(values.len(), rows * width, "activations are rows × width");
        let data = match format {
            ActivationFormat::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ActivationFormat::F16 => values
                .iter()
                .flat_map(|v| half::f16::from_f32(*v).to_bits().to_le_bytes())
                .collect(),
            ActivationFormat::Q8_0 => {
                let mut data = Vec::with_capacity(values.len().div_ceil(Q8_BLOCK) * Q8_BLOCK_BYTES);
                for block in values.chunks(Q8_BLOCK) {
                    let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let scale = half::f16::from_f32(amax / 127.0);
                    let inverse = if scale.to_f32() > 0.0 {
                        1.0 / scale.to_f32()
                    } else {
                        0.0
                    };
                    data.extend_from_slice(&scale.to_bits().to_le_bytes());
                    for i in 0..Q8_BLOCK {
                        let q = block
                            .get(i)
                            .map_or(0, |v| (v * inverse).round().clamp(-127.0, 127.0) as i8);
                        data.push(q as u8);
                    }
                }
                data
            }
        };
        Self {
            format,
            rows: rows as u32,
            width: width as u32,
            data,
        }
    }

    pub fn decode(&self) -> Result<Vec<f32>> {
        // A peer's sizes: the product can be too large to count.
        let sizes = (self.rows as usize)
            .checked_mul(self.width as usize)
            .and_then(|n| {
                let bytes = match self.format {
                    ActivationFormat::F32 => n.checked_mul(4),
                    ActivationFormat::F16 => n.checked_mul(2),
                    ActivationFormat::Q8_0 => n.div_ceil(Q8_BLOCK).checked_mul(Q8_BLOCK_BYTES),
                };
                Some((n, bytes?))
            });
        let Some((n, expected)) = sizes else {
            bail!(
                "{} rows × {} are too many activations",
                self.rows,
                self.width
            );
        };
        ensure!(
            self.data.len() == expected,
            "{} activations of {} rows × {} hold {} bytes, expected {expected}",
            self.format.label(),
            self.rows,
            self.width,
            self.data.len()
        );
        Ok(match self.format {
            ActivationFormat::F32 => self
                .data
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            ActivationFormat::F16 => self
                .data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half::f16::from_bits(u16::from_le_bytes(*b)).to_f32())
                .collect(),
            ActivationFormat::Q8_0 => {
                let mut values = Vec::with_capacity(n);
                for block in self.data.as_chunks::<Q8_BLOCK_BYTES>().0 {
                    let scale =
                        half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
                    values.extend(block[2..].iter().map(|q| *q as i8 as f32 * scale));
                }
                values.truncate(n);
                values
            }
        })
    }
}

/// What a node can hold and how fast it runs, reported upward in the
/// handshake. The `subtree_*` fields add up the node and every worker below
/// it, which is all a parent needs to size the range it hands down.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Capacity {
    pub backend: String,
    pub device_bytes: u64,
    pub host_bytes: u64,
    pub subtree_nodes: u32,
    pub subtree_device_bytes: u64,
    pub subtree_host_bytes: u64,
    /// The bytes of weights this node can hold: what its layers live in —
    /// its device's memory when it has one, its RAM otherwise — less room
    /// for everything else (`super::plan::budget`).
    pub budget_bytes: u64,
    /// The same, added up over the node and every node below it: what a
    /// parent sizes this worker's share by.
    pub subtree_budget_bytes: u64,
    /// The processors and measured speed of the node and of every node
    /// below it, the node's own first: what a parent sizes a
    /// worker's share by when its nodes differ, and what `/v1/workers`
    /// shows.
    pub setups: Vec<NodeSetup>,
}

/// One node's processors, and how fast its layers ran when it measured
/// them (`super::speed`).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct NodeSetup {
    /// The node's `[workers]` address, as plans name it.
    pub name: String,
    pub devices: Vec<Device>,
    /// Gigabytes of layer weights a second through one decode step, and
    /// through a 128-token prompt chunk; `0` when not measured.
    pub decode_rate: f32,
    pub prompt_rate: f32,
    /// What a decode step and a prompt chunk cost on it whatever their
    /// layers, in milliseconds; `0` when not measured.
    pub decode_fixed_ms: f32,
    pub prompt_fixed_ms: f32,
}

/// A CPU, GPU or NPU of a node.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Device {
    /// `cpu`, `gpu` or `npu`.
    pub kind: String,
    pub name: String,
    /// Cores, where the device says: a CPU's logical cores, an NPU's.
    pub cores: u32,
    /// Its memory: a GPU's own (or what it shares), a CPU's RAM.
    pub memory_bytes: u64,
    /// Whether this node's layers run on it.
    pub in_use: bool,
}

/// Which model, exactly — down to its quantization. Two nodes whose files
/// differ must not compute together, whatever the files are called. See
/// `super::identity` for what each hash covers.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ModelIdentity {
    /// The model as its node names it (`<user>/<model>:<quant>` or a file
    /// name), for messages only: two copies of one file may be named
    /// differently and are still the same model.
    pub label: String,
    /// The quantization, as `list` shows it, for messages only.
    pub quant: String,
    /// SHA-256 over the hyperparameters and the whole tensor directory —
    /// every tensor's name, ggml type and shape. Two quantizations of one
    /// model never share it.
    pub header_hash: [u8; 32],
    /// SHA-256 over samples of the tensor data of the layers in question:
    /// two files with the same layout and different weights (another
    /// release, another importance matrix) differ here.
    pub range_hash: [u8; 32],
}

/// Which rows of the result a forward wants back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rows {
    /// Every row: what the next stage needs, and a multi-position verify.
    All,
    /// The last row only: a prefill chunk or decode step whose stream ends
    /// at this stage and is read for its next-token logits alone.
    Last,
    /// No rows at all: a prompt chunk before the last, whose keys and
    /// values are the point and whose output nobody reads. A worker answers
    /// it as soon as its own layers are done, and passes it on to its own
    /// workers in the background — which is what pipelines a long prompt
    /// through the tree.
    None,
    /// The last row's next-token logits: the node that runs the model's
    /// final layer applies the output head to it. Only sent to a
    /// child that advertised [`FEATURE_HEAD`].
    LastLogits,
    /// Every row's logits, the same way: a multi-position verify.
    AllLogits,
}

impl Rows {
    /// How many rows come back for a forward of `n_tokens`.
    pub fn count(self, n_tokens: usize) -> usize {
        match self {
            Rows::All | Rows::AllLogits => n_tokens,
            Rows::Last | Rows::LastLogits => 1,
            Rows::None => 0,
        }
    }

    /// Whether they come back as logits rather than the residual stream.
    pub fn logits(self) -> bool {
        matches!(self, Rows::LastLogits | Rows::AllLogits)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Hello {
    pub version: u16,
    pub features: u64,
    pub nonce: [u8; 32],
    /// The id of every node from the root down to the sender, so a node
    /// that finds itself — or anything below itself — on the path refuses
    /// the connection. Ids, not addresses: one machine has many spellings
    /// of its address, and one node id.
    pub path: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HelloAck {
    pub version: u16,
    pub features: u64,
    pub nonce: [u8; 32],
    /// The worker's proof that it knows the secret — see `super::auth`.
    pub proof: [u8; 32],
    /// The worker's node id.
    pub node: String,
    /// The ids of the worker and of every node currently below it, so a
    /// parent that finds itself there refuses a loop from its side too.
    pub subtree: Vec<String>,
    pub capacity: Capacity,
    /// The model this node serves (its layout: `range_hash` is left
    /// empty), so a parent knows before planning whether it runs the same.
    /// Appended; `None` from a node that does not say.
    pub model: Option<ModelIdentity>,
    /// Where every node of this tree can be scraped by Prometheus this
    /// node's own address first, then every node below it, as each of
    /// them reported it. A parent collects the whole tree this way, one
    /// hop at a time. A node with no metrics listener sends an empty
    /// list. Written at the very end of the message, so a reader that
    /// does not know it just ignores it.
    pub discovery: Vec<DiscoveryRecord>,
    /// How this node is configured to hold sequences. Appended after
    /// `discovery` (written, empty, when there are no records), so a reader
    /// that does not know it ignores it; `None` from a node that does not
    /// say. Sent only with a `model`.
    pub sessions: Option<SessionTerms>,
    /// The prefixes this node holds on its disk ([`FEATURE_PREFIXES`]):
    /// appended after `sessions`, and sent only with them, to a parent that
    /// advertised the bit. Empty from a node that does not say.
    pub saved: Vec<SavedPrefix>,
}

/// One prefix a node holds on its disk ([`Message::Save`]): its key, the
/// layers its rows are of, and the positions they hold.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SavedPrefix {
    pub key: String,
    pub layer_start: u32,
    pub layer_end: u32,
    pub len: u32,
}

/// What a node is configured to hold, as its own settings decide: what its
/// parent needs to know before sending it a session to keep.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct SessionTerms {
    /// Its own `slots`.
    pub slots: u32,
    /// Its own context length.
    pub n_ctx: u32,
    /// Sessions it keeps for its parent's prefix pool ([`Message::Retain`]),
    /// beside the two per slot of the parent's: the fewest of this node and
    /// every node below it, since a pool entry holds a session on each.
    pub pool_reserve: u32,
}

impl SessionTerms {
    /// The sessions this node holds at most for a parent with
    /// `parent_slots` slots.
    pub fn session_limit(&self, parent_slots: usize) -> usize {
        parent_slots * 2 + self.pool_reserve as usize
    }
}

/// One entry in the tree's phone book: a node's name and the address
/// where Prometheus finds its metrics page.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct DiscoveryRecord {
    /// The node's name from its config, as plans print it (for example
    /// `ad:8402`). This never changes across restarts. The handshake also
    /// carries an `id`, but that one is random every start and exists only
    /// to catch a node dialing itself a label needs a name that stays.
    pub node: String,
    /// The address others used to reach this node as a worker. Nobody
    /// connects with it — it only tags metrics, spelled exactly like the
    /// `worker` label on the parent's `worker_up` series, so data from two
    /// places can be joined by name. It is always a real address: when a
    /// parent passes its child's card up, it replaces the child's
    /// `0.0.0.0` bind (`0.0.0.0` means "all network cards" and is not an
    /// address you can visit). An address recorded two or more hops up may
    /// only make sense on the machine it was dialed from.
    pub worker: String,
    /// Host and port of this node's metrics listener.
    pub metrics_host: String,
    pub metrics_port: u16,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Assign {
    /// The model spec, resolved against the worker's own `models`
    /// directory: a path, an `NR`/label, or a Hugging Face reference.
    pub model: String,
    pub identity: ModelIdentity,
    pub layer_start: u32,
    pub layer_end: u32,
    pub n_ctx: u32,
    pub kv_type: String,
    pub slots: u32,
    pub activations: ActivationFormat,
    /// The top-level node's prompt-weight decision
    /// (`engine::prompt_weights::Decision::bits`); appended, so `None` from
    /// a parent that does not send it.
    pub prompt_weights: Option<u8>,
}

/// One entry of the plan a worker reports back: which node runs which
/// layers, for the banner and `/v1/workers`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanEntry {
    pub node: String,
    pub layer_start: u32,
    pub layer_end: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Forward {
    pub session: u64,
    pub step: u64,
    /// The absolute position of the first row. A node refuses a forward
    /// whose `start_pos` is not the length of its cache for the session.
    pub start_pos: u32,
    pub rows: Rows,
    /// Roll the session back to this many positions before running.
    pub kv_truncate_to: Option<u32>,
    pub tokens: Vec<u32>,
    pub hidden: Activations,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ForwardResult {
    pub session: u64,
    pub step: u64,
    /// The layers this result has been through, echoed so a late or
    /// misrouted reply cannot answer the wrong step.
    pub layer_start: u32,
    pub layer_end: u32,
    pub hidden: Activations,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorCode {
    PositionMismatch,
    UnknownSession,
    ModelMismatch,
    OutOfMemory,
    ChildLost,
    Busy,
    Unauthorized,
    Unsupported,
    ContextFull,
    Internal,
    /// A reply over other layers than this node planned: a worker already on
    /// the plan in progress. A node that does not know this code reads
    /// [`Self::Internal`].
    LayersChanged,
}

impl ErrorCode {
    fn tag(self) -> u8 {
        match self {
            Self::PositionMismatch => 1,
            Self::UnknownSession => 2,
            Self::ModelMismatch => 3,
            Self::OutOfMemory => 4,
            Self::ChildLost => 5,
            Self::Busy => 6,
            Self::Unauthorized => 7,
            Self::Unsupported => 8,
            Self::ContextFull => 9,
            Self::Internal => 10,
            Self::LayersChanged => 11,
        }
    }

    /// An unknown code reads as [`Self::Internal`]: a newer peer's more
    /// specific failure is still a failure.
    fn from_tag(tag: u8) -> Self {
        match tag {
            1 => Self::PositionMismatch,
            2 => Self::UnknownSession,
            3 => Self::ModelMismatch,
            4 => Self::OutOfMemory,
            5 => Self::ChildLost,
            6 => Self::Busy,
            7 => Self::Unauthorized,
            8 => Self::Unsupported,
            9 => Self::ContextFull,
            11 => Self::LayersChanged,
            _ => Self::Internal,
        }
    }
}

/// A typed refusal. `path` names the node it started at, root first, so a
/// failure deep in the tree says where.
#[derive(Clone, Debug, PartialEq)]
pub struct WorkerError {
    pub code: ErrorCode,
    pub message: String,
    pub path: Vec<String>,
}

impl WorkerError {
    /// The longest message sent to a peer. Longer text is cut, so an
    /// error cannot carry a whole prompt or a stack of context upward.
    pub const MAX_MESSAGE: usize = 256;

    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: sanitize(&message.into()),
            path: Vec::new(),
        }
    }
}

impl std::fmt::Display for WorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)?;
        if !self.path.is_empty() {
            write!(f, " (at {})", self.path.join(" > "))?;
        }
        Ok(())
    }
}

impl std::error::Error for WorkerError {}

/// The first line of `message`, cut to [`WorkerError::MAX_MESSAGE`] bytes on
/// a character boundary.
fn sanitize(message: &str) -> String {
    let line = message.lines().next().unwrap_or("").trim();
    let mut end = line.len().min(WorkerError::MAX_MESSAGE);
    while !line.is_char_boundary(end) {
        end -= 1;
    }
    line[..end].to_string()
}

/// Whether a node whose `subtree` (itself and every node below it) is to
/// hang below `path` (the root down to its new parent) would close a loop —
/// the configs list each other — or make the tree deeper than
/// [`MAX_DEPTH`]. Checked by the worker on `Hello` and by the parent on
/// `HelloAck`, each with what it knows.
pub fn check_path(path: &[String], subtree: &[String]) -> Result<(), WorkerError> {
    if let Some(node) = path.iter().find(|node| subtree.contains(node)) {
        return Err(WorkerError::new(
            ErrorCode::Unsupported,
            format!(
                "node {node} would be its own ancestor (path {}): the [workers] lists form a loop",
                path.join(" > ")
            ),
        ));
    }
    if path.len() >= MAX_DEPTH {
        return Err(WorkerError::new(
            ErrorCode::Unsupported,
            format!("the tree is deeper than {MAX_DEPTH} levels"),
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub enum Message {
    Hello(Hello),
    HelloAck(Box<HelloAck>),
    /// The parent's proof that it knows the secret, answering the worker's
    /// nonce — see `super::auth`.
    Auth {
        proof: [u8; 32],
    },
    Assign(Assign),
    /// The worker's answer to `Assign` once its whole subtree is ready:
    /// its own identity for the assigned range, which the parent checks
    /// against its own, and who runs what.
    AssignAck {
        identity: ModelIdentity,
        plan: Vec<PlanEntry>,
    },
    Forward(Forward),
    ForwardResult(ForwardResult),
    /// Several sessions' forwards at once: one decode step of every slot.
    ForwardBatch(Vec<Forward>),
    ForwardBatchResult(Vec<Result<ForwardResult, WorkerError>>),
    Truncate {
        session: u64,
        len: u32,
    },
    Release {
        session: u64,
    },
    /// Promise `session` memory for `positions`, on this worker and every
    /// worker below it, before a prompt that long starts. Answered with the
    /// same message, or [`ErrorCode::OutOfMemory`] having promised nothing
    /// anywhere; `0` takes a promise back. Only sent to a child that advertised
    /// [`FEATURE_ROOM`].
    Reserve {
        session: u64,
        positions: u32,
    },
    /// Start session `to` with the first `len` positions of session `from`,
    /// copied, on this worker and every worker below it. Answered with the
    /// same message, or an error when `from` is unknown or does not hold
    /// `len` positions here. Only sent to a child that advertised
    /// [`FEATURE_FORK`].
    Fork {
        from: u64,
        to: u64,
        len: u32,
    },
    /// `held`: keep `session` on this worker and every worker below it until
    /// a [`Message::Release`] for it, exempt from idle and shortfall
    /// eviction — a session the parent's prefix pool holds. `!held` returns
    /// it to ordinary eviction. Answered with the same message. Only sent to
    /// a child that advertised [`FEATURE_RETAIN`].
    Retain {
        session: u64,
        held: bool,
    },
    /// Write `session`'s rows on this worker's own layers to its disk under
    /// `key` (32 hex digits), and have every worker below do the same.
    /// Answered with the same message. Only sent to a child that advertised
    /// [`FEATURE_PREFIXES`].
    Save {
        session: u64,
        key: String,
    },
    /// Open `session`, held as by [`Message::Retain`], from the rows saved
    /// under `key`, its first `len` positions, here and on every worker
    /// below. Answered with the same message, or an error — the session
    /// opened nowhere — when a node has no such rows, or rows that are
    /// damaged or do not match its model, layers, shape or context. Only sent
    /// to a child that listed `key` in its handshake.
    Load {
        session: u64,
        key: String,
        len: u32,
    },
    /// Delete the rows saved under `key` here and on every worker below.
    /// Answered with the same message. Only sent to a child that advertised
    /// [`FEATURE_PREFIXES`].
    Forget {
        key: String,
    },
    /// Run `tokens`, from position 0, through every layer of the model on
    /// this node alone, and answer with the last position's logits
    /// ([`Message::Prefilled`]). The rows of the layers this node and the
    /// workers below it run open `session` there, as [`Message::Open`] does;
    /// the rest are kept for the parent to take ([`Message::RowsOf`]). Only
    /// sent to a child that advertised [`FEATURE_ROWS`], and refused by one
    /// that does not hold the whole model.
    Prefill {
        session: u64,
        tokens: Vec<u32>,
    },
    /// The answer to a [`Message::Prefill`]: `f32` logits, one row.
    Prefilled {
        session: u64,
        logits: Activations,
    },
    /// The rows a [`Message::Prefill`] kept of layers
    /// `layer_start..layer_end`, answered with [`Message::RowsData`];
    /// `last` lets go of what is left of them.
    RowsOf {
        session: u64,
        layer_start: u32,
        layer_end: u32,
        last: bool,
    },
    /// Rows of a range of layers in the self-describing form of
    /// `super::prefixes::encode_rows`.
    RowsData {
        session: u64,
        data: Vec<u8>,
    },
    /// Rows for `session` on a range of the layers this node and the workers
    /// below it run (`data`, as in [`Message::RowsData`]): its own kept until
    /// [`Message::Open`], the workers' slices sent on. Answered with an empty
    /// `data`. Only sent to a child that advertised [`FEATURE_ROWS`].
    Put {
        session: u64,
        data: Vec<u8>,
    },
    /// Open `session` from the rows put here, its first `len` positions, here
    /// and on every worker below: refused, with nothing opened, when a layer
    /// has none or they hold another length. Answered with the same message.
    Open {
        session: u64,
        len: u32,
    },
    /// Answered with the same bytes: what a parent times a link's bandwidth
    /// by. Only sent to a child that advertised [`FEATURE_ECHO`].
    Echo {
        data: Vec<u8>,
    },
    Cancel {
        session: u64,
        step: u64,
    },
    Ping {
        nonce: u64,
    },
    Pong {
        nonce: u64,
    },
    Error(WorkerError),
    /// The worker's answer to a proof it accepts: the handshake is done.
    Ready,
    /// Sent instead of `Hello` by a node scanning its networks: whether the
    /// node listening here is on the channel that hashes to `channel`
    /// ([`super::channel::channel_hash`]). Answered with [`Message::ProbeAck`]
    /// by a node on that channel; any other node closes the connection.
    Probe {
        channel: [u8; 32],
    },
    ProbeAck {
        /// The node's id for this run of the process: what a node finds
        /// itself by.
        node: String,
        /// Its `[workers]` name, `host:port`.
        name: String,
        state: ProbeState,
        /// How busy it is, `0.0` (idle) to `1.0` (fully busy)
        /// (`load` (orangu-server)).
        load: f32,
        /// Whether it can serve the model alone: it has the whole model
        /// and memory for it.
        whole: bool,
        /// The port its API answers on, and whether with TLS: where a
        /// request routed to it goes.
        api_port: u16,
        api_tls: bool,
        /// The model it serves: [`ModelIdentity::header_hash`], and a hash
        /// of samples of every layer's weights — zeros on a node that has
        /// only part of the model. Nodes cooperate only on the same model.
        model: [u8; 32],
        content: [u8; 32],
        /// How many workers it holds while it leads, `0` otherwise: when
        /// two nodes lead the same model, the smaller tree gives way.
        /// Appended; `0` from a node that does not send it.
        workers: u32,
        /// The model it serves by name, as its `/v1/models` lists it, and its
        /// quantization: what a request naming a model is routed by.
        /// Appended; empty from a node that does not send them, which no
        /// request names.
        label: String,
        quant: String,
        /// The extensions it serves to any node of the channel, not only
        /// its parent: [`FEATURE_FETCH`]. Appended; `0` from a node that
        /// does not send it.
        features: u64,
        /// How fast it decodes, and a request sent to it would: what a node
        /// weighs serving a request alone against dividing it
        /// (`divide` (orangu-server)). Appended; unknown (zeros) from a node that
        /// does not send it.
        pace: Pace,
        /// The models requests asked this node for that no free node of the
        /// channel served, with how many in the last few minutes
        /// (`demand` (orangu-server)); at most [`MAX_DEMAND`], most wanted first.
        /// Appended; empty from a node that does not send it.
        demand: Vec<Wanted>,
        /// What kind of node it is. Appended; [`NodeKind::Server`] from a
        /// node that does not send it.
        kind: NodeKind,
        /// The jobs it takes, by name ([`NodeKind::Gguf`]'s: `quantize`,
        /// `gradients`). Appended; empty from a node that does not send them.
        capabilities: Vec<String>,
    },
    /// Sent instead of `Hello` by a node that fetches a model in part
    /// (`[workers].download = range`) from a node holding the whole file.
    /// Authenticated as a parent is ([`super::auth`]): answered with
    /// [`Message::FetchAck`] and its proof, then the fetcher's
    /// [`Message::Auth`] is answered with [`Message::Ready`], after which
    /// the connection carries [`Message::FetchRange`]s. A node that serves
    /// no file answers with an error, and an older one closes the
    /// connection.
    FetchHello {
        nonce: [u8; 32],
    },
    FetchAck {
        /// The serving node's [`FEATURE_FETCH`], and anything else it
        /// serves a fetcher.
        features: u64,
        nonce: [u8; 32],
        proof: [u8; 32],
        /// [`ModelIdentity::header_hash`] of the file it serves, which its
        /// probe answer names too.
        model: [u8; 32],
        /// The file: its size, its name, and the snapshot (Hub commit) its
        /// directory is named after, empty when it is in none.
        size: u64,
        file: String,
        commit: String,
    },
    /// Bytes `offset..offset + len` of the served file, `len` at most
    /// [`MAX_FETCH_BYTES`]. Answered with [`Message::FetchData`], or an
    /// error for a range outside the file.
    FetchRange {
        offset: u64,
        len: u64,
    },
    FetchData {
        offset: u64,
        data: Vec<u8>,
    },
    /// Sent instead of `Hello` by a node about to claim workers on the
    /// channel that hashes to `channel`: asks the listener to promise
    /// itself to candidate `node` for a few seconds, so two nodes that get
    /// a request at the same moment do not split the free nodes between
    /// them. Answered with [`Message::LeaseAck`]; a node that does not
    /// know the message closes the connection.
    Lease {
        channel: [u8; 32],
        node: String,
    },
    LeaseAck {
        /// Whether the listener is now promised to the candidate.
        granted: bool,
        /// Who it is promised to, or works for, when not granted.
        holder: String,
        /// `holder` already leads it (or is it, leading a tree of its
        /// own), rather than holding a promise.
        committed: bool,
    },
    /// Sent instead of `Hello` by an `orangu-gguf` top claiming a
    /// [`NodeKind::Gguf`] node: `node` is the top's id. Answered with
    /// [`Message::ClaimAck`] and the node's proof, or [`ErrorCode::Busy`]
    /// when it works for another top; the top's [`Message::Auth`] is then
    /// answered with [`Message::Ready`], and the node works for that top
    /// until the connection closes. A server node closes
    /// the connection, as an older node of either kind does.
    Claim {
        nonce: [u8; 32],
        node: String,
    },
    ClaimAck {
        nonce: [u8; 32],
        proof: [u8; 32],
        /// The claimed node's id.
        node: String,
    },
    /// A share of a build sent to a claimed gguf node: what to do, by
    /// name, and its input, encoded by the job. Answered with
    /// [`Message::JobDone`], or an error.
    Job {
        name: String,
        data: Vec<u8>,
    },
    JobDone {
        data: Vec<u8>,
    },
}

/// How fast a node of a channel decodes, as its [`Message::ProbeAck`] says.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pace {
    /// Gigabytes of layer weights a second through a decode step, and what
    /// a step costs it whatever its layers (`super::speed`); `0` when not
    /// measured.
    pub decode_rate: f32,
    pub decode_fixed_ms: f32,
    /// The bytes of weights it can hold ([`Capacity::budget_bytes`]).
    pub budget_bytes: u64,
    /// Milliseconds a token for a request sent to it: its model alone, or
    /// through the tree it leads; `0` when it cannot tell.
    pub step_ms: f32,
}

/// The most [`Wanted`] entries a [`Message::ProbeAck`] carries.
pub const MAX_DEMAND: usize = 8;

/// A model requests asked a node for while no free node of the channel
/// served it, as its [`Message::ProbeAck`] says.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Wanted {
    /// The name the requests used, or the name of the model they matched.
    pub model: String,
    /// How many such requests the node got in its window.
    pub count: u32,
}

/// The most capabilities a [`Message::ProbeAck`] carries.
pub const MAX_CAPABILITIES: usize = 16;

/// Which program a node of a channel is: what it can be asked to do.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum NodeKind {
    /// `orangu-server`: serves a model, and runs layers for another node.
    #[default]
    Server,
    /// `orangu-gguf --worker`: does a share of building a model file.
    Gguf,
}

impl NodeKind {
    fn to_u8(self) -> u8 {
        match self {
            Self::Server => 0,
            Self::Gguf => 1,
        }
    }

    /// A kind this build does not know reads as a server, the kind that
    /// asks nothing of a gguf top.
    fn from_u8(value: u8) -> Self {
        match value {
            1 => Self::Gguf,
            _ => Self::Server,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Server => "server",
            Self::Gguf => "gguf",
        }
    }
}

/// What a node on a channel is doing, as a [`Message::ProbeAck`] says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProbeState {
    /// Serving its own clients, and free to be claimed.
    Idle,
    /// Top-level, with workers it claimed.
    Leading,
    /// A worker of another node.
    Serving,
    /// Free, but it never leads: it has only part of the model.
    WorkerOnly,
}

impl ProbeState {
    fn to_u8(self) -> u8 {
        match self {
            Self::Idle => 0,
            Self::Leading => 1,
            Self::Serving => 2,
            Self::WorkerOnly => 3,
        }
    }

    fn from_u8(value: u8) -> Self {
        match value {
            0 => Self::Idle,
            1 => Self::Leading,
            3 => Self::WorkerOnly,
            _ => Self::Serving,
        }
    }

    /// Whether a node in this state can be claimed as a worker.
    pub fn claimable(self) -> bool {
        matches!(self, Self::Idle | Self::WorkerOnly)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Leading => "leading",
            Self::Serving => "serving",
            Self::WorkerOnly => "worker_only",
        }
    }
}

mod tag {
    pub const HELLO: u8 = 1;
    pub const HELLO_ACK: u8 = 2;
    pub const AUTH: u8 = 3;
    pub const ASSIGN: u8 = 4;
    pub const ASSIGN_ACK: u8 = 5;
    pub const FORWARD: u8 = 6;
    pub const FORWARD_RESULT: u8 = 7;
    pub const FORWARD_BATCH: u8 = 8;
    pub const FORWARD_BATCH_RESULT: u8 = 9;
    pub const TRUNCATE: u8 = 10;
    pub const RELEASE: u8 = 11;
    pub const CANCEL: u8 = 12;
    pub const PING: u8 = 13;
    pub const PONG: u8 = 14;
    pub const ERROR: u8 = 15;
    pub const READY: u8 = 16;
    pub const FORK: u8 = 17;
    pub const ECHO: u8 = 20;
    pub const RESERVE: u8 = 22;
    pub const PROBE: u8 = 23;
    pub const PROBE_ACK: u8 = 24;
    pub const LEASE: u8 = 25;
    pub const LEASE_ACK: u8 = 26;
    pub const RETAIN: u8 = 27;
    pub const SAVE: u8 = 28;
    pub const LOAD: u8 = 29;
    pub const FORGET: u8 = 30;
    pub const FETCH_HELLO: u8 = 31;
    pub const FETCH_ACK: u8 = 32;
    pub const FETCH_RANGE: u8 = 33;
    pub const FETCH_DATA: u8 = 34;
    pub const PREFILL: u8 = 35;
    pub const PREFILLED: u8 = 36;
    pub const ROWS_OF: u8 = 37;
    pub const ROWS_DATA: u8 = 38;
    pub const PUT: u8 = 39;
    pub const OPEN: u8 = 40;
    pub const CLAIM: u8 = 41;
    pub const CLAIM_ACK: u8 = 42;
    pub const JOB: u8 = 43;
    pub const JOB_DONE: u8 = 44;
}

struct Writer(Vec<u8>);

impl Writer {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn fixed(&mut self, v: &[u8; 32]) {
        self.0.extend_from_slice(v);
    }
    fn bytes(&mut self, v: &[u8]) {
        self.u32(v.len() as u32);
        self.0.extend_from_slice(v);
    }
    fn string(&mut self, v: &str) {
        self.bytes(v.as_bytes());
    }
    fn strings(&mut self, v: &[String]) {
        self.u32(v.len() as u32);
        for s in v {
            self.string(s);
        }
    }
    fn activations(&mut self, a: &Activations) {
        self.u8(a.format.tag());
        self.u32(a.rows);
        self.u32(a.width);
        self.bytes(&a.data);
    }
    fn forward(&mut self, f: &Forward) {
        self.u64(f.session);
        self.u64(f.step);
        self.u32(f.start_pos);
        self.u8(match f.rows {
            Rows::All => 0,
            Rows::Last => 1,
            Rows::None => 2,
            Rows::LastLogits => 3,
            Rows::AllLogits => 4,
        });
        match f.kv_truncate_to {
            Some(len) => {
                self.u8(1);
                self.u32(len);
            }
            None => self.u8(0),
        }
        self.u32(f.tokens.len() as u32);
        for t in &f.tokens {
            self.u32(*t);
        }
        self.activations(&f.hidden);
    }
    fn forward_result(&mut self, r: &ForwardResult) {
        self.u64(r.session);
        self.u64(r.step);
        self.u32(r.layer_start);
        self.u32(r.layer_end);
        self.activations(&r.hidden);
    }
    fn identity(&mut self, i: &ModelIdentity) {
        self.string(&i.label);
        self.string(&i.quant);
        self.fixed(&i.header_hash);
        self.fixed(&i.range_hash);
    }
    fn discovery(&mut self, records: &[DiscoveryRecord]) {
        self.u32(records.len() as u32);
        for r in records {
            self.string(&r.node);
            self.string(&r.worker);
            self.string(&r.metrics_host);
            self.u16(r.metrics_port);
        }
    }
    fn error(&mut self, e: &WorkerError) {
        self.u8(e.code.tag());
        self.string(&e.message);
        self.strings(&e.path);
    }
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        ensure!(
            n <= self.0.len(),
            "message ends early: wanted {n} more bytes, {} left",
            self.0.len()
        );
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into()?))
    }
    fn fixed(&mut self) -> Result<[u8; 32]> {
        Ok(self.take(32)?.try_into()?)
    }
    fn bytes(&mut self) -> Result<Vec<u8>> {
        let n = self.u32()? as usize;
        Ok(self.take(n)?.to_vec())
    }
    fn string(&mut self) -> Result<String> {
        String::from_utf8(self.bytes()?).map_err(|_| anyhow!("string is not UTF-8"))
    }
    /// A count about to drive `n` reads of at least `min_item` bytes each,
    /// checked against what is left so a corrupt count cannot reserve
    /// gigabytes before failing.
    fn count(&mut self, min_item: usize) -> Result<usize> {
        let n = self.u32()? as usize;
        ensure!(
            n.saturating_mul(min_item) <= self.0.len(),
            "list of {n} items cannot fit in the {} bytes left",
            self.0.len()
        );
        Ok(n)
    }
    fn strings(&mut self) -> Result<Vec<String>> {
        let n = self.count(4)?;
        (0..n).map(|_| self.string()).collect()
    }
    fn setups(&mut self) -> Result<Vec<NodeSetup>> {
        let n = self.count(24)?;
        (0..n)
            .map(|_| {
                let name = self.string()?;
                let decode_rate = self.f32()?;
                let prompt_rate = self.f32()?;
                let decode_fixed_ms = self.f32()?;
                let prompt_fixed_ms = self.f32()?;
                let n = self.count(21)?;
                let devices = (0..n)
                    .map(|_| {
                        Ok(Device {
                            kind: self.string()?,
                            name: self.string()?,
                            cores: self.u32()?,
                            memory_bytes: self.u64()?,
                            in_use: self.u8()? != 0,
                        })
                    })
                    .collect::<Result<_>>()?;
                Ok(NodeSetup {
                    name,
                    devices,
                    decode_rate,
                    prompt_rate,
                    decode_fixed_ms,
                    prompt_fixed_ms,
                })
            })
            .collect()
    }
    fn activations(&mut self) -> Result<Activations> {
        Ok(Activations {
            format: ActivationFormat::from_tag(self.u8()?)?,
            rows: self.u32()?,
            width: self.u32()?,
            data: self.bytes()?,
        })
    }
    fn forward(&mut self) -> Result<Forward> {
        let session = self.u64()?;
        let step = self.u64()?;
        let start_pos = self.u32()?;
        let rows = match self.u8()? {
            0 => Rows::All,
            1 => Rows::Last,
            2 => Rows::None,
            3 => Rows::LastLogits,
            4 => Rows::AllLogits,
            other => bail!("unknown rows selector {other}"),
        };
        let kv_truncate_to = match self.u8()? {
            0 => None,
            _ => Some(self.u32()?),
        };
        let n = self.count(4)?;
        let tokens = (0..n).map(|_| self.u32()).collect::<Result<_>>()?;
        Ok(Forward {
            session,
            step,
            start_pos,
            rows,
            kv_truncate_to,
            tokens,
            hidden: self.activations()?,
        })
    }
    fn forward_result(&mut self) -> Result<ForwardResult> {
        Ok(ForwardResult {
            session: self.u64()?,
            step: self.u64()?,
            layer_start: self.u32()?,
            layer_end: self.u32()?,
            hidden: self.activations()?,
        })
    }
    fn identity(&mut self) -> Result<ModelIdentity> {
        Ok(ModelIdentity {
            label: self.string()?,
            quant: self.string()?,
            header_hash: self.fixed()?,
            range_hash: self.fixed()?,
        })
    }
    fn discovery(&mut self) -> Result<Vec<DiscoveryRecord>> {
        // Each record holds four texts and one port. A text always costs
        // at least 4 bytes (its length marker), so the count is checked
        // against the bytes actually present — a made-up count fails
        // instead of being believed.
        let n = self.count(14)?;
        (0..n)
            .map(|_| {
                Ok(DiscoveryRecord {
                    node: self.string()?,
                    worker: self.string()?,
                    metrics_host: self.string()?,
                    metrics_port: self.u16()?,
                })
            })
            .collect()
    }
    fn error(&mut self) -> Result<WorkerError> {
        Ok(WorkerError {
            code: ErrorCode::from_tag(self.u8()?),
            message: self.string()?,
            path: self.strings()?,
        })
    }
}

impl Message {
    /// The frame's type byte and payload.
    pub fn encode(&self) -> (u8, Vec<u8>) {
        let mut w = Writer(Vec::new());
        let tag = match self {
            Self::Hello(h) => {
                w.u16(h.version);
                w.u64(h.features);
                w.fixed(&h.nonce);
                w.strings(&h.path);
                tag::HELLO
            }
            Self::HelloAck(h) => {
                w.u16(h.version);
                w.u64(h.features);
                w.fixed(&h.nonce);
                w.fixed(&h.proof);
                w.string(&h.node);
                w.strings(&h.subtree);
                let c = &h.capacity;
                w.string(&c.backend);
                w.u64(c.device_bytes);
                w.u64(c.host_bytes);
                w.u32(c.subtree_nodes);
                w.u64(c.subtree_device_bytes);
                w.u64(c.subtree_host_bytes);
                w.u64(c.budget_bytes);
                w.u64(c.subtree_budget_bytes);
                w.u32(c.setups.len() as u32);
                for setup in &c.setups {
                    w.string(&setup.name);
                    w.f32(setup.decode_rate);
                    w.f32(setup.prompt_rate);
                    w.f32(setup.decode_fixed_ms);
                    w.f32(setup.prompt_fixed_ms);
                    w.u32(setup.devices.len() as u32);
                    for d in &setup.devices {
                        w.string(&d.kind);
                        w.string(&d.name);
                        w.u32(d.cores);
                        w.u64(d.memory_bytes);
                        w.u8(u8::from(d.in_use));
                    }
                }
                if let Some(model) = &h.model {
                    w.identity(model);
                }
                let sessions = h.sessions.filter(|_| h.model.is_some());
                if !h.discovery.is_empty() || sessions.is_some() {
                    w.discovery(&h.discovery);
                }
                if let Some(terms) = sessions {
                    w.u32(terms.slots);
                    w.u32(terms.n_ctx);
                    w.u32(terms.pool_reserve);
                    if !h.saved.is_empty() {
                        w.u32(h.saved.len() as u32);
                        for saved in &h.saved {
                            w.string(&saved.key);
                            w.u32(saved.layer_start);
                            w.u32(saved.layer_end);
                            w.u32(saved.len);
                        }
                    }
                }
                tag::HELLO_ACK
            }
            Self::Auth { proof } => {
                w.fixed(proof);
                tag::AUTH
            }
            Self::Assign(a) => {
                w.string(&a.model);
                w.identity(&a.identity);
                w.u32(a.layer_start);
                w.u32(a.layer_end);
                w.u32(a.n_ctx);
                w.string(&a.kv_type);
                w.u32(a.slots);
                w.u8(a.activations.tag());
                if let Some(bits) = a.prompt_weights {
                    w.u8(bits);
                }
                tag::ASSIGN
            }
            Self::AssignAck { identity, plan } => {
                w.identity(identity);
                w.u32(plan.len() as u32);
                for entry in plan {
                    w.string(&entry.node);
                    w.u32(entry.layer_start);
                    w.u32(entry.layer_end);
                }
                tag::ASSIGN_ACK
            }
            Self::Forward(f) => {
                w.forward(f);
                tag::FORWARD
            }
            Self::ForwardResult(r) => {
                w.forward_result(r);
                tag::FORWARD_RESULT
            }
            Self::ForwardBatch(items) => {
                w.u32(items.len() as u32);
                for f in items {
                    w.forward(f);
                }
                tag::FORWARD_BATCH
            }
            Self::ForwardBatchResult(items) => {
                w.u32(items.len() as u32);
                for item in items {
                    match item {
                        Ok(r) => {
                            w.u8(0);
                            w.forward_result(r);
                        }
                        Err(e) => {
                            w.u8(1);
                            w.error(e);
                        }
                    }
                }
                tag::FORWARD_BATCH_RESULT
            }
            Self::Truncate { session, len } => {
                w.u64(*session);
                w.u32(*len);
                tag::TRUNCATE
            }
            Self::Release { session } => {
                w.u64(*session);
                tag::RELEASE
            }
            Self::Reserve { session, positions } => {
                w.u64(*session);
                w.u32(*positions);
                tag::RESERVE
            }
            Self::Fork { from, to, len } => {
                w.u64(*from);
                w.u64(*to);
                w.u32(*len);
                tag::FORK
            }
            Self::Retain { session, held } => {
                w.u64(*session);
                w.u8(u8::from(*held));
                tag::RETAIN
            }
            Self::Save { session, key } => {
                w.u64(*session);
                w.string(key);
                tag::SAVE
            }
            Self::Load { session, key, len } => {
                w.u64(*session);
                w.string(key);
                w.u32(*len);
                tag::LOAD
            }
            Self::Forget { key } => {
                w.string(key);
                tag::FORGET
            }
            Self::Prefill { session, tokens } => {
                w.u64(*session);
                w.u32(tokens.len() as u32);
                for t in tokens {
                    w.u32(*t);
                }
                tag::PREFILL
            }
            Self::Prefilled { session, logits } => {
                w.u64(*session);
                w.activations(logits);
                tag::PREFILLED
            }
            Self::RowsOf {
                session,
                layer_start,
                layer_end,
                last,
            } => {
                w.u64(*session);
                w.u32(*layer_start);
                w.u32(*layer_end);
                w.u8(u8::from(*last));
                tag::ROWS_OF
            }
            Self::RowsData { session, data } => {
                w.u64(*session);
                w.bytes(data);
                tag::ROWS_DATA
            }
            Self::Put { session, data } => {
                w.u64(*session);
                w.bytes(data);
                tag::PUT
            }
            Self::Open { session, len } => {
                w.u64(*session);
                w.u32(*len);
                tag::OPEN
            }
            Self::Echo { data } => {
                w.bytes(data);
                tag::ECHO
            }
            Self::Cancel { session, step } => {
                w.u64(*session);
                w.u64(*step);
                tag::CANCEL
            }
            Self::Ping { nonce } => {
                w.u64(*nonce);
                tag::PING
            }
            Self::Pong { nonce } => {
                w.u64(*nonce);
                tag::PONG
            }
            Self::Error(e) => {
                w.error(e);
                tag::ERROR
            }
            Self::Ready => tag::READY,
            Self::Probe { channel } => {
                w.fixed(channel);
                tag::PROBE
            }
            Self::ProbeAck {
                node,
                name,
                state,
                load,
                whole,
                api_port,
                api_tls,
                model,
                content,
                workers,
                label,
                quant,
                features,
                pace,
                demand,
                kind,
                capabilities,
            } => {
                w.string(node);
                w.string(name);
                w.u8(state.to_u8());
                w.f32(*load);
                w.u8(u8::from(*whole));
                w.u16(*api_port);
                w.u8(u8::from(*api_tls));
                w.fixed(model);
                w.fixed(content);
                w.u32(*workers);
                w.string(label);
                w.string(quant);
                w.u64(*features);
                w.f32(pace.decode_rate);
                w.f32(pace.decode_fixed_ms);
                w.u64(pace.budget_bytes);
                w.f32(pace.step_ms);
                let shared = &demand[..demand.len().min(MAX_DEMAND)];
                w.u8(shared.len() as u8);
                for wanted in shared {
                    w.string(&wanted.model);
                    w.u32(wanted.count);
                }
                w.u8(kind.to_u8());
                let listed = &capabilities[..capabilities.len().min(MAX_CAPABILITIES)];
                w.u8(listed.len() as u8);
                for capability in listed {
                    w.string(capability);
                }
                tag::PROBE_ACK
            }
            Self::FetchHello { nonce } => {
                w.fixed(nonce);
                tag::FETCH_HELLO
            }
            Self::FetchAck {
                features,
                nonce,
                proof,
                model,
                size,
                file,
                commit,
            } => {
                w.u64(*features);
                w.fixed(nonce);
                w.fixed(proof);
                w.fixed(model);
                w.u64(*size);
                w.string(file);
                w.string(commit);
                tag::FETCH_ACK
            }
            Self::FetchRange { offset, len } => {
                w.u64(*offset);
                w.u64(*len);
                tag::FETCH_RANGE
            }
            Self::FetchData { offset, data } => {
                w.u64(*offset);
                w.bytes(data);
                tag::FETCH_DATA
            }
            Self::Lease { channel, node } => {
                w.fixed(channel);
                w.string(node);
                tag::LEASE
            }
            Self::LeaseAck {
                granted,
                holder,
                committed,
            } => {
                w.u8(u8::from(*granted));
                w.string(holder);
                w.u8(u8::from(*committed));
                tag::LEASE_ACK
            }
            Self::Claim { nonce, node } => {
                w.fixed(nonce);
                w.string(node);
                tag::CLAIM
            }
            Self::ClaimAck { nonce, proof, node } => {
                w.fixed(nonce);
                w.fixed(proof);
                w.string(node);
                tag::CLAIM_ACK
            }
            Self::Job { name, data } => {
                w.string(name);
                w.bytes(data);
                tag::JOB
            }
            Self::JobDone { data } => {
                w.bytes(data);
                tag::JOB_DONE
            }
        };
        (tag, w.0)
    }

    /// Reads a message of type `tag` from `payload`, ignoring any bytes
    /// after the fields this build knows.
    pub fn decode(tag: u8, payload: &[u8]) -> Result<Self> {
        let mut r = Reader(payload);
        Ok(match tag {
            tag::HELLO => Self::Hello(Hello {
                version: r.u16()?,
                features: r.u64()?,
                nonce: r.fixed()?,
                path: r.strings()?,
            }),
            tag::HELLO_ACK => Self::HelloAck(Box::new(HelloAck {
                version: r.u16()?,
                features: r.u64()?,
                nonce: r.fixed()?,
                proof: r.fixed()?,
                node: r.string()?,
                subtree: r.strings()?,
                capacity: Capacity {
                    backend: r.string()?,
                    device_bytes: r.u64()?,
                    host_bytes: r.u64()?,
                    subtree_nodes: r.u32()?,
                    subtree_device_bytes: r.u64()?,
                    subtree_host_bytes: r.u64()?,
                    budget_bytes: r.u64()?,
                    subtree_budget_bytes: r.u64()?,
                    setups: r.setups()?,
                },
                model: if r.0.is_empty() {
                    None
                } else {
                    Some(r.identity()?)
                },
                discovery: if r.0.is_empty() {
                    Vec::new()
                } else {
                    r.discovery()?
                },
                sessions: if r.0.is_empty() {
                    None
                } else {
                    Some(SessionTerms {
                        slots: r.u32()?,
                        n_ctx: r.u32()?,
                        pool_reserve: r.u32()?,
                    })
                },
                saved: if r.0.is_empty() {
                    Vec::new()
                } else {
                    let n = r.count(16)?;
                    (0..n)
                        .map(|_| {
                            Ok(SavedPrefix {
                                key: r.string()?,
                                layer_start: r.u32()?,
                                layer_end: r.u32()?,
                                len: r.u32()?,
                            })
                        })
                        .collect::<Result<_>>()?
                },
            })),
            tag::AUTH => Self::Auth { proof: r.fixed()? },
            tag::ASSIGN => Self::Assign(Assign {
                model: r.string()?,
                identity: r.identity()?,
                layer_start: r.u32()?,
                layer_end: r.u32()?,
                n_ctx: r.u32()?,
                kv_type: r.string()?,
                slots: r.u32()?,
                activations: ActivationFormat::from_tag(r.u8()?)?,
                prompt_weights: if r.0.is_empty() { None } else { Some(r.u8()?) },
            }),
            tag::ASSIGN_ACK => {
                let identity = r.identity()?;
                let n = r.count(12)?;
                let plan = (0..n)
                    .map(|_| {
                        Ok(PlanEntry {
                            node: r.string()?,
                            layer_start: r.u32()?,
                            layer_end: r.u32()?,
                        })
                    })
                    .collect::<Result<_>>()?;
                Self::AssignAck { identity, plan }
            }
            tag::FORWARD => Self::Forward(r.forward()?),
            tag::FORWARD_RESULT => Self::ForwardResult(r.forward_result()?),
            tag::FORWARD_BATCH => {
                let n = r.count(1)?;
                Self::ForwardBatch((0..n).map(|_| r.forward()).collect::<Result<_>>()?)
            }
            tag::FORWARD_BATCH_RESULT => {
                let n = r.count(1)?;
                let items = (0..n)
                    .map(|_| {
                        Ok(match r.u8()? {
                            0 => Ok(r.forward_result()?),
                            _ => Err(r.error()?),
                        })
                    })
                    .collect::<Result<_>>()?;
                Self::ForwardBatchResult(items)
            }
            tag::TRUNCATE => Self::Truncate {
                session: r.u64()?,
                len: r.u32()?,
            },
            tag::RELEASE => Self::Release { session: r.u64()? },
            tag::RESERVE => Self::Reserve {
                session: r.u64()?,
                positions: r.u32()?,
            },
            tag::FORK => Self::Fork {
                from: r.u64()?,
                to: r.u64()?,
                len: r.u32()?,
            },
            tag::RETAIN => Self::Retain {
                session: r.u64()?,
                held: r.u8()? != 0,
            },
            tag::SAVE => Self::Save {
                session: r.u64()?,
                key: r.string()?,
            },
            tag::LOAD => Self::Load {
                session: r.u64()?,
                key: r.string()?,
                len: r.u32()?,
            },
            tag::FORGET => Self::Forget { key: r.string()? },
            tag::PREFILL => {
                let session = r.u64()?;
                let n = r.count(4)?;
                let tokens = (0..n).map(|_| r.u32()).collect::<Result<_>>()?;
                Self::Prefill { session, tokens }
            }
            tag::PREFILLED => Self::Prefilled {
                session: r.u64()?,
                logits: r.activations()?,
            },
            tag::ROWS_OF => Self::RowsOf {
                session: r.u64()?,
                layer_start: r.u32()?,
                layer_end: r.u32()?,
                last: r.u8()? != 0,
            },
            tag::ROWS_DATA => Self::RowsData {
                session: r.u64()?,
                data: r.bytes()?,
            },
            tag::PUT => Self::Put {
                session: r.u64()?,
                data: r.bytes()?,
            },
            tag::OPEN => Self::Open {
                session: r.u64()?,
                len: r.u32()?,
            },
            tag::ECHO => Self::Echo { data: r.bytes()? },
            tag::CANCEL => Self::Cancel {
                session: r.u64()?,
                step: r.u64()?,
            },
            tag::PING => Self::Ping { nonce: r.u64()? },
            tag::PONG => Self::Pong { nonce: r.u64()? },
            tag::ERROR => Self::Error(r.error()?),
            tag::READY => Self::Ready,
            tag::PROBE => Self::Probe {
                channel: r.fixed()?,
            },
            tag::PROBE_ACK => Self::ProbeAck {
                node: r.string()?,
                name: r.string()?,
                state: ProbeState::from_u8(r.u8()?),
                load: r.f32()?,
                whole: r.u8()? != 0,
                api_port: r.u16()?,
                api_tls: r.u8()? != 0,
                model: r.fixed()?,
                content: r.fixed()?,
                workers: if r.0.is_empty() { 0 } else { r.u32()? },
                label: if r.0.is_empty() {
                    String::new()
                } else {
                    r.string()?
                },
                quant: if r.0.is_empty() {
                    String::new()
                } else {
                    r.string()?
                },
                features: if r.0.is_empty() { 0 } else { r.u64()? },
                pace: if r.0.is_empty() {
                    Pace::default()
                } else {
                    Pace {
                        decode_rate: r.f32()?,
                        decode_fixed_ms: r.f32()?,
                        budget_bytes: r.u64()?,
                        step_ms: r.f32()?,
                    }
                },
                demand: if r.0.is_empty() {
                    Vec::new()
                } else {
                    let n = r.u8()? as usize;
                    let mut demand = (0..n)
                        .map(|_| {
                            Ok(Wanted {
                                model: r.string()?,
                                count: r.u32()?,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    demand.truncate(MAX_DEMAND);
                    demand
                },
                kind: if r.0.is_empty() {
                    NodeKind::Server
                } else {
                    NodeKind::from_u8(r.u8()?)
                },
                capabilities: if r.0.is_empty() {
                    Vec::new()
                } else {
                    let n = r.u8()? as usize;
                    (0..n.min(MAX_CAPABILITIES))
                        .map(|_| r.string())
                        .collect::<Result<Vec<_>>>()?
                },
            },
            tag::FETCH_HELLO => Self::FetchHello { nonce: r.fixed()? },
            tag::FETCH_ACK => Self::FetchAck {
                features: r.u64()?,
                nonce: r.fixed()?,
                proof: r.fixed()?,
                model: r.fixed()?,
                size: r.u64()?,
                file: r.string()?,
                commit: r.string()?,
            },
            tag::FETCH_RANGE => Self::FetchRange {
                offset: r.u64()?,
                len: r.u64()?,
            },
            tag::FETCH_DATA => Self::FetchData {
                offset: r.u64()?,
                data: r.bytes()?,
            },
            tag::LEASE => Self::Lease {
                channel: r.fixed()?,
                node: r.string()?,
            },
            tag::LEASE_ACK => Self::LeaseAck {
                granted: r.u8()? != 0,
                holder: r.string()?,
                committed: r.u8()? != 0,
            },
            tag::CLAIM => Self::Claim {
                nonce: r.fixed()?,
                node: r.string()?,
            },
            tag::CLAIM_ACK => Self::ClaimAck {
                nonce: r.fixed()?,
                proof: r.fixed()?,
                node: r.string()?,
            },
            tag::JOB => Self::Job {
                name: r.string()?,
                data: r.bytes()?,
            },
            tag::JOB_DONE => Self::JobDone { data: r.bytes()? },
            other => bail!("unknown message type {other}"),
        })
    }

    /// The whole frame: length, type, request id, payload.
    pub fn to_frame(&self, id: u64) -> Vec<u8> {
        let (tag, payload) = self.encode();
        let mut frame = Vec::with_capacity(13 + payload.len());
        frame.extend_from_slice(&((payload.len() + 9) as u32).to_le_bytes());
        frame.push(tag);
        frame.extend_from_slice(&id.to_le_bytes());
        frame.extend_from_slice(&payload);
        frame
    }
}

/// Writes `message` as one frame of request `id`; returns the frame's size
/// in bytes.
pub fn write_message<W: Write + ?Sized>(out: &mut W, id: u64, message: &Message) -> Result<usize> {
    let frame = message.to_frame(id);
    ensure!(
        frame.len() - 4 <= MAX_FRAME,
        "a {} byte message is larger than the {MAX_FRAME} byte limit",
        frame.len() - 4
    );
    out.write_all(&frame)?;
    out.flush()?;
    Ok(frame.len())
}

/// Reads one frame of a handshake, up to [`MAX_HANDSHAKE_FRAME`], dropping its
/// request id: there is only ever one request at a time.
pub fn read_message<R: Read + ?Sized>(input: &mut R) -> Result<Message> {
    read_up_to(input, MAX_HANDSHAKE_FRAME).map(|(_, message, _)| message)
}

/// Reads one frame: its request id, its message, and its size in bytes.
pub fn read_frame<R: Read + ?Sized>(input: &mut R) -> Result<(u64, Message, usize)> {
    read_up_to(input, MAX_FRAME)
}

fn read_up_to<R: Read + ?Sized>(input: &mut R, limit: usize) -> Result<(u64, Message, usize)> {
    let mut len = [0u8; 4];
    input.read_exact(&mut len)?;
    let len = u32::from_le_bytes(len) as usize;
    ensure!(
        len >= 9,
        "a {len} byte frame is too short to hold a type and an id"
    );
    ensure!(
        len <= limit,
        "a {len} byte frame is larger than the {limit} byte limit"
    );
    // Doubled as it arrives, so the room never passes twice what was sent.
    let mut body = vec![0u8; len.min(UP_FRONT)];
    input.read_exact(&mut body)?;
    while body.len() < len {
        let filled = body.len();
        body.resize(len.min(filled * 2), 0);
        input.read_exact(&mut body[filled..])?;
    }
    let id = u64::from_le_bytes(body[1..9].try_into()?);
    Ok((id, Message::decode(body[0], &body[9..])?, len + 4))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(message: Message) {
        let frame = message.to_frame(42);
        let (id, back, size) = read_frame(&mut frame.as_slice()).expect("decodes");
        assert_eq!((id, &back, size), (42, &message, frame.len()));
    }

    fn activations() -> Activations {
        Activations::encode(&[1.0, -2.0, 0.5, 4.0], 2, 2, ActivationFormat::F32)
    }

    fn forward() -> Forward {
        Forward {
            session: 7,
            step: 3,
            start_pos: 12,
            rows: Rows::Last,
            kv_truncate_to: Some(10),
            tokens: vec![5, 6],
            hidden: activations(),
        }
    }

    /// Clearing one extension leaves every other one advertised.
    #[test]
    fn each_feature_has_its_own_bit() {
        for bit in ALL_FEATURES {
            let names = feature_names(ALL_FEATURES.iter().fold(0, |all, b| all | b) & !bit);
            assert_eq!(names.len(), ALL_FEATURES.len() - 1, "{names:?}");
        }
        let without_room = feature_names(FEATURES & !FEATURE_ROOM);
        assert!(!without_room.contains(&"room"));
        assert!(without_room.contains(&"metrics"));
        assert_eq!(FEATURE_RETAIN, 256);
        assert!(ALL_FEATURES.contains(&FEATURE_RETAIN));
        assert!(feature_names(FEATURES).contains(&"retain"));
        assert!(!feature_names(FEATURES & !FEATURE_RETAIN).contains(&"retain"));
        assert_eq!(FEATURE_PREFIXES, 512);
        assert!(ALL_FEATURES.contains(&FEATURE_PREFIXES));
        assert!(feature_names(FEATURES).contains(&"prefixes"));
        assert!(!feature_names(FEATURES & !FEATURE_PREFIXES).contains(&"prefixes"));
        assert_eq!(FEATURE_FETCH, 1024);
        assert!(ALL_FEATURES.contains(&FEATURE_FETCH));
        assert!(feature_names(FEATURE_FETCH).contains(&"fetch"));
        // Advertised per node, by a node that holds its whole file.
        assert_eq!(FEATURES & FEATURE_FETCH, 0);
        assert_eq!(FEATURE_ROWS, 2048);
        assert!(ALL_FEATURES.contains(&FEATURE_ROWS));
        assert!(feature_names(FEATURES).contains(&"rows"));
        assert!(!feature_names(FEATURES & !FEATURE_ROWS).contains(&"rows"));
    }

    /// The messages that move a prompt's rows, each through a frame and
    /// back; their tags follow the last one in use, and none is a retired tag.
    #[test]
    fn the_rows_messages_survive_a_round_trip() {
        let messages = [
            Message::Prefill {
                session: 9,
                tokens: vec![1, 2, 3],
            },
            Message::Prefilled {
                session: 9,
                logits: activations(),
            },
            Message::RowsOf {
                session: 9,
                layer_start: 2,
                layer_end: 5,
                last: true,
            },
            Message::RowsData {
                session: 9,
                data: vec![7; 33],
            },
            Message::Put {
                session: 9,
                data: vec![1, 2],
            },
            Message::Open {
                session: 9,
                len: 300,
            },
        ];
        for (message, expected) in messages.into_iter().zip(35u8..) {
            assert_eq!(message.encode().0, expected);
            assert!(![18, 19, 21].contains(&expected));
            round_trip(message);
        }
    }

    /// A probe answer from a node that sends no kind reads as a server with
    /// no capabilities.
    #[test]
    fn a_probe_ack_without_a_kind_is_a_server() {
        let (tag, payload) = ack_wanting(Vec::new()).encode();
        let Message::ProbeAck {
            kind, capabilities, ..
        } = Message::decode(tag, &payload[..payload.len() - 2]).unwrap()
        else {
            panic!("not a probe answer");
        };
        assert_eq!(kind, NodeKind::Server);
        assert!(capabilities.is_empty());
    }

    /// A probe answer from a node that does not send its features reads as
    /// a node that serves no bytes.
    #[test]
    fn a_probe_ack_without_features_serves_nothing() {
        let (tag, payload) = Message::ProbeAck {
            node: "n".to_string(),
            name: "n:8400".to_string(),
            state: ProbeState::Idle,
            load: 0.0,
            whole: true,
            api_port: 8100,
            api_tls: false,
            model: [1; 32],
            content: [2; 32],
            workers: 0,
            label: "m".to_string(),
            quant: "Q8_0".to_string(),
            features: FEATURE_FETCH,
            pace: Pace::default(),
            demand: Vec::new(),
            kind: Default::default(),
            capabilities: Vec::new(),
        }
        .encode();
        // The kind and the capabilities' count take a byte each at the end,
        // the demand's count one before them, the pace twenty before it, the
        // features eight before that.
        let older = &payload[..payload.len() - 31];
        let Message::ProbeAck {
            features, label, ..
        } = Message::decode(tag, older).unwrap()
        else {
            panic!("not a probe answer");
        };
        assert_eq!((features, label.as_str()), (0, "m"));
    }

    fn ack_wanting(demand: Vec<Wanted>) -> Message {
        Message::ProbeAck {
            node: "n".to_string(),
            name: "n:8400".to_string(),
            state: ProbeState::Idle,
            load: 0.0,
            whole: true,
            api_port: 8100,
            api_tls: false,
            model: [1; 32],
            content: [2; 32],
            workers: 0,
            label: "m".to_string(),
            quant: "Q8_0".to_string(),
            features: 0,
            pace: Pace {
                decode_rate: 1.0,
                decode_fixed_ms: 2.0,
                budget_bytes: 3,
                step_ms: 4.0,
            },
            demand,
            kind: Default::default(),
            capabilities: Vec::new(),
        }
    }

    /// A probe answer from a node that does not send its demand reads as
    /// a node no request went unserved at; the pace before it survives.
    #[test]
    fn a_probe_ack_without_demand_decodes_to_none() {
        let (tag, payload) = ack_wanting(vec![Wanted {
            model: "gpt-4".to_string(),
            count: 3,
        }])
        .encode();
        // The demand: a count, then the name (four bytes and five) and its
        // count of requests; then the kind and the capabilities' count.
        let older = &payload[..payload.len() - (1 + 4 + 5 + 4 + 2)];
        let Message::ProbeAck { demand, pace, .. } = Message::decode(tag, older).unwrap() else {
            panic!("not a probe answer");
        };
        assert!(demand.is_empty());
        assert_eq!(pace.step_ms, 4.0);
    }

    /// A node sends at most [`MAX_DEMAND`] models it was asked for, and a
    /// longer list read from a peer is cut to as many.
    #[test]
    fn a_probe_ack_carries_at_most_max_demand_models() {
        let many: Vec<Wanted> = (0..MAX_DEMAND as u32 + 3)
            .map(|i| Wanted {
                model: format!("m{i}"),
                count: 100 - i,
            })
            .collect();
        let (tag, payload) = ack_wanting(many.clone()).encode();
        let Message::ProbeAck { demand, .. } = Message::decode(tag, &payload).unwrap() else {
            panic!("not a probe answer");
        };
        assert_eq!(demand, many[..MAX_DEMAND].to_vec());
    }

    /// A parent that advertised both extensions under bit 64 still gets
    /// the discovery list; one with neither does not.
    #[test]
    fn a_parent_on_the_shared_bit_reads_discovery() {
        assert!(parent_reads_discovery(FEATURE_ROOM));
        assert!(parent_reads_discovery(FEATURE_METRICS));
        assert!(parent_reads_discovery(FEATURES));
        assert!(!parent_reads_discovery(
            FEATURES & !FEATURE_ROOM & !FEATURE_METRICS
        ));
    }

    fn result() -> ForwardResult {
        ForwardResult {
            session: 7,
            step: 3,
            layer_start: 4,
            layer_end: 9,
            hidden: activations(),
        }
    }

    fn identity() -> ModelIdentity {
        ModelIdentity {
            label: "org/model".to_string(),
            quant: "Q4_K_M".to_string(),
            header_hash: [2; 32],
            range_hash: [8; 32],
        }
    }

    fn error() -> WorkerError {
        WorkerError {
            code: ErrorCode::PositionMismatch,
            message: "expected 12".to_string(),
            path: vec!["a:1".to_string(), "b:2".to_string()],
        }
    }

    #[test]
    fn every_message_survives_a_round_trip() {
        round_trip(Message::Hello(Hello {
            version: PROTOCOL_VERSION,
            features: FEATURES,
            nonce: [3; 32],
            path: vec!["root:8400".to_string()],
        }));
        round_trip(Message::HelloAck(Box::new(HelloAck {
            version: PROTOCOL_VERSION,
            features: 5,
            nonce: [4; 32],
            proof: [9; 32],
            node: "n1".to_string(),
            subtree: vec!["n1".to_string(), "n2".to_string()],
            capacity: Capacity {
                backend: "vulkan".to_string(),
                device_bytes: 1 << 34,
                host_bytes: 1 << 35,
                subtree_nodes: 3,
                subtree_device_bytes: 3 << 34,
                subtree_host_bytes: 3 << 35,
                budget_bytes: 1 << 33,
                subtree_budget_bytes: 3 << 33,
                setups: vec![NodeSetup {
                    name: "n1:8400".to_string(),
                    devices: vec![
                        Device {
                            kind: "cpu".to_string(),
                            name: "Cortex-A720".to_string(),
                            cores: 12,
                            memory_bytes: 1 << 35,
                            in_use: false,
                        },
                        Device {
                            kind: "gpu".to_string(),
                            name: "Mali-G720".to_string(),
                            cores: 0,
                            memory_bytes: 1 << 34,
                            in_use: true,
                        },
                    ],
                    decode_rate: 21.5,
                    prompt_rate: 310.0,
                    decode_fixed_ms: 1.5,
                    prompt_fixed_ms: 12.0,
                }],
            },
            model: Some(identity()),
            discovery: vec![
                DiscoveryRecord {
                    node: "n1".to_string(),
                    worker: "n1:8400".to_string(),
                    metrics_host: "10.0.0.1".to_string(),
                    metrics_port: 8102,
                },
                DiscoveryRecord {
                    node: "n2".to_string(),
                    worker: "127.0.0.1:8401".to_string(),
                    metrics_host: "10.0.0.2".to_string(),
                    metrics_port: 8102,
                },
            ],
            sessions: Some(SessionTerms {
                slots: 2,
                n_ctx: 4096,
                pool_reserve: 2,
            }),
            saved: vec![
                SavedPrefix {
                    key: "0123456789abcdef0123456789abcdef".to_string(),
                    layer_start: 4,
                    layer_end: 9,
                    len: 300,
                },
                SavedPrefix {
                    key: "fedcba9876543210fedcba9876543210".to_string(),
                    layer_start: 4,
                    layer_end: 9,
                    len: 12,
                },
            ],
        })));
        round_trip(Message::Auth { proof: [1; 32] });
        round_trip(Message::Assign(Assign {
            model: "org/model:Q4_K_M".to_string(),
            identity: identity(),
            layer_start: 10,
            layer_end: 30,
            n_ctx: 8192,
            kv_type: "f16".to_string(),
            slots: 4,
            activations: ActivationFormat::Q8_0,
            prompt_weights: Some(3),
        }));
        round_trip(Message::AssignAck {
            identity: identity(),
            plan: vec![PlanEntry {
                node: "a:8400".to_string(),
                layer_start: 10,
                layer_end: 18,
            }],
        });
        round_trip(Message::Forward(forward()));
        round_trip(Message::Forward(Forward {
            kv_truncate_to: None,
            rows: Rows::All,
            ..forward()
        }));
        round_trip(Message::ForwardResult(result()));
        round_trip(Message::ForwardBatch(vec![forward(), forward()]));
        round_trip(Message::ForwardBatchResult(vec![
            Ok(result()),
            Err(error()),
        ]));
        round_trip(Message::Truncate { session: 1, len: 2 });
        round_trip(Message::Echo {
            data: vec![1, 2, 3],
        });
        round_trip(Message::Reserve {
            session: 1,
            positions: 4096,
        });
        round_trip(Message::Fork {
            from: 3,
            to: 4,
            len: 5,
        });
        round_trip(Message::Release { session: 1 });
        round_trip(Message::Retain {
            session: 6,
            held: true,
        });
        round_trip(Message::Retain {
            session: 6,
            held: false,
        });
        round_trip(Message::Save {
            session: 6,
            key: "0123456789abcdef0123456789abcdef".to_string(),
        });
        round_trip(Message::Load {
            session: 7,
            key: "0123456789abcdef0123456789abcdef".to_string(),
            len: 300,
        });
        round_trip(Message::Forget {
            key: "0123456789abcdef0123456789abcdef".to_string(),
        });
        round_trip(Message::Cancel {
            session: 1,
            step: 2,
        });
        round_trip(Message::Ping { nonce: 42 });
        round_trip(Message::Pong { nonce: 42 });
        round_trip(Message::Probe { channel: [7; 32] });
        for state in [
            ProbeState::Idle,
            ProbeState::Leading,
            ProbeState::Serving,
            ProbeState::WorkerOnly,
        ] {
            round_trip(Message::ProbeAck {
                node: "00ff".to_string(),
                name: "beta:8400".to_string(),
                state,
                load: 0.25,
                whole: true,
                api_port: 8100,
                api_tls: false,
                model: [1; 32],
                content: [2; 32],
                workers: 3,
                label: "unsloth/model-GGUF".to_string(),
                quant: "Q4_K_M".to_string(),
                features: FEATURE_FETCH,
                pace: Pace {
                    decode_rate: 12.5,
                    decode_fixed_ms: 1.25,
                    budget_bytes: 7 << 30,
                    step_ms: 13.0,
                },
                demand: vec![
                    Wanted {
                        model: "unsloth/other-GGUF:Q8_0".to_string(),
                        count: 5,
                    },
                    Wanted {
                        model: "gpt-4".to_string(),
                        count: 1,
                    },
                ],
                kind: NodeKind::Gguf,
                capabilities: vec!["quantize".to_string(), "gradients".to_string()],
            });
        }
        round_trip(Message::FetchHello { nonce: [3; 32] });
        round_trip(Message::Claim {
            nonce: [4; 32],
            node: "top".to_string(),
        });
        round_trip(Message::ClaimAck {
            nonce: [5; 32],
            proof: [6; 32],
            node: "worker".to_string(),
        });
        round_trip(Message::Job {
            name: "imatrix".to_string(),
            data: vec![1, 2, 3],
        });
        round_trip(Message::JobDone { data: vec![4; 9] });
        round_trip(Message::FetchAck {
            features: FEATURE_FETCH,
            nonce: [4; 32],
            proof: [5; 32],
            model: [6; 32],
            size: 1 << 33,
            file: "model-Q4_K_M.gguf".to_string(),
            commit: "0123abcd".to_string(),
        });
        round_trip(Message::FetchRange {
            offset: 1 << 32,
            len: MAX_FETCH_BYTES,
        });
        round_trip(Message::FetchData {
            offset: 7,
            data: vec![1, 2, 3],
        });
        round_trip(Message::Lease {
            channel: [5; 32],
            node: "00aa".to_string(),
        });
        for (granted, committed) in [(true, false), (false, false), (false, true)] {
            round_trip(Message::LeaseAck {
                granted,
                holder: "0011".to_string(),
                committed,
            });
        }
        round_trip(Message::Error(error()));
        round_trip(Message::Ready);
    }

    /// A newer peer may append fields; this build reads what it knows.
    #[test]
    fn an_assign_without_prompt_weights_decodes() {
        let mut assign = Assign {
            model: "m".to_string(),
            identity: identity(),
            layer_start: 0,
            layer_end: 4,
            n_ctx: 64,
            kv_type: String::new(),
            slots: 1,
            activations: ActivationFormat::F16,
            prompt_weights: None,
        };
        round_trip(Message::Assign(assign.clone()));
        assign.prompt_weights = Some(1);
        let (tag, payload) = Message::Assign(assign.clone()).encode();
        let short = Message::decode(tag, &payload[..payload.len() - 1]).unwrap();
        assign.prompt_weights = None;
        assert_eq!(short, Message::Assign(assign));
    }

    #[test]
    fn every_rows_selector_survives_a_round_trip() {
        for rows in [
            Rows::All,
            Rows::Last,
            Rows::None,
            Rows::LastLogits,
            Rows::AllLogits,
        ] {
            let mut f = forward();
            f.rows = rows;
            round_trip(Message::Forward(f));
        }
        assert_eq!(Rows::AllLogits.count(5), 5);
        assert_eq!(Rows::LastLogits.count(5), 1);
        assert!(Rows::LastLogits.logits() && !Rows::Last.logits());
    }

    /// A probe answer from a node that does not send `workers` reads as
    /// a node that leads nobody.
    #[test]
    fn a_probe_ack_without_workers_decodes_to_none() {
        let (tag, payload) = Message::ProbeAck {
            node: "n".to_string(),
            name: "n:8400".to_string(),
            state: ProbeState::Leading,
            load: 0.5,
            whole: false,
            api_port: 8100,
            api_tls: false,
            model: [1; 32],
            content: [2; 32],
            workers: 7,
            label: String::new(),
            quant: String::new(),
            features: 0,
            pace: Pace::default(),
            demand: Vec::new(),
            kind: Default::default(),
            capabilities: Vec::new(),
        }
        .encode();
        // The two empty names take eight bytes, the workers count four
        // more, the features eight, the pace twenty, the demand's count one,
        // the kind one and the capabilities' count one.
        let older = &payload[..payload.len() - 43];
        let Message::ProbeAck { workers, state, .. } = Message::decode(tag, older).unwrap() else {
            panic!("not a probe answer");
        };
        assert_eq!((workers, state), (0, ProbeState::Leading));
    }

    /// A probe answer from a node that sends `workers` but not its model's
    /// name reads as a node no request names.
    #[test]
    fn a_probe_ack_without_a_model_name_decodes_to_empty() {
        let (tag, payload) = Message::ProbeAck {
            node: "n".to_string(),
            name: "n:8400".to_string(),
            state: ProbeState::Idle,
            load: 0.0,
            whole: true,
            api_port: 8100,
            api_tls: false,
            model: [1; 32],
            content: [2; 32],
            workers: 2,
            label: "m".to_string(),
            quant: "Q8_0".to_string(),
            features: FEATURE_FETCH,
            pace: Pace::default(),
            demand: Vec::new(),
            kind: Default::default(),
            capabilities: Vec::new(),
        }
        .encode();
        let label_bytes = 4 + 1 + 4 + 4 + 8 + 20 + 1 + 2;
        let older = &payload[..payload.len() - label_bytes];
        let Message::ProbeAck {
            workers,
            label,
            quant,
            ..
        } = Message::decode(tag, older).unwrap()
        else {
            panic!("not a probe answer");
        };
        assert_eq!((workers, label.as_str(), quant.as_str()), (2, "", ""));
    }

    #[test]
    fn trailing_bytes_are_ignored() {
        let (tag, mut payload) = Message::Release { session: 9 }.encode();
        payload.extend_from_slice(&[1, 2, 3]);
        assert_eq!(
            Message::decode(tag, &payload).unwrap(),
            Message::Release { session: 9 }
        );
    }

    /// A message with nothing added at the end — the shape an older
    /// version of orangu-server sends — comes back with no model and no
    /// discovery records, instead of failing.
    #[test]
    fn an_ack_without_appended_fields_decodes_to_empty() {
        let ack = HelloAck {
            version: PROTOCOL_VERSION,
            features: 5,
            nonce: [4; 32],
            proof: [9; 32],
            node: "n1".to_string(),
            subtree: vec!["n1".to_string()],
            capacity: Capacity::default(),
            model: None,
            discovery: Vec::new(),
            sessions: None,
            saved: Vec::new(),
        };
        round_trip(Message::HelloAck(Box::new(ack)));
    }

    /// The session terms ride after an empty discovery list, and a reader
    /// that stops before them — an older one — still reads the rest.
    #[test]
    fn session_terms_follow_an_empty_discovery_list() {
        let ack = HelloAck {
            version: PROTOCOL_VERSION,
            features: FEATURES,
            nonce: [4; 32],
            proof: [9; 32],
            node: "n1".to_string(),
            subtree: vec!["n1".to_string()],
            capacity: Capacity::default(),
            model: Some(identity()),
            discovery: Vec::new(),
            sessions: Some(SessionTerms {
                slots: 1,
                n_ctx: 512,
                pool_reserve: 1,
            }),
            saved: Vec::new(),
        };
        round_trip(Message::HelloAck(Box::new(ack.clone())));
        assert_eq!(ack.sessions.unwrap().session_limit(4), 9);
    }

    #[test]
    fn a_short_payload_is_refused() {
        let (tag, payload) = Message::Forward(forward()).encode();
        for cut in [0, 1, 20, payload.len() - 1] {
            assert!(Message::decode(tag, &payload[..cut]).is_err(), "cut {cut}");
        }
    }

    #[test]
    fn an_unknown_type_is_refused() {
        let err = Message::decode(200, &[]).unwrap_err();
        assert!(err.to_string().contains("unknown message type"), "{err:#}");
    }

    #[test]
    fn an_oversized_or_empty_frame_is_refused_before_reading_it() {
        let huge = ((MAX_FRAME + 1) as u32).to_le_bytes();
        let err = read_message(&mut huge.as_slice()).unwrap_err();
        assert!(err.to_string().contains("larger than"), "{err:#}");
        let empty = 0u32.to_le_bytes();
        assert!(read_message(&mut empty.as_slice()).is_err());
    }

    #[test]
    fn a_frame_too_large_for_a_handshake_is_refused_before_reading_it() {
        let mut frame = ((MAX_HANDSHAKE_FRAME + 1) as u32).to_le_bytes().to_vec();
        frame.extend_from_slice(&[0; 64]);
        let err = read_message(&mut frame.as_slice()).unwrap_err();
        assert!(err.to_string().contains("larger than"), "{err:#}");
    }

    /// What a parent of a large tree hears has to fit in what it accepts.
    #[test]
    fn the_handshake_of_a_large_tree_fits_its_limit() {
        let name = "a-rather-long-host-name.example.org:8400".to_string();
        let device = Device {
            kind: "vulkan".to_string(),
            name: name.clone(),
            cores: 12,
            memory_bytes: 1 << 35,
            in_use: true,
        };
        let setup = NodeSetup {
            name: name.clone(),
            devices: vec![device; 8],
            decode_rate: 1.0,
            prompt_rate: 1.0,
            decode_fixed_ms: 1.0,
            prompt_fixed_ms: 1.0,
        };
        let discovery = vec![
            DiscoveryRecord {
                node: name.clone(),
                worker: name.clone(),
                metrics_host: name.clone(),
                metrics_port: 8400,
            };
            256
        ];
        let ack = Message::HelloAck(Box::new(HelloAck {
            version: PROTOCOL_VERSION,
            features: FEATURES,
            nonce: [1; 32],
            proof: [2; 32],
            node: name.clone(),
            subtree: vec![name; 256],
            capacity: Capacity {
                setups: vec![setup; 256],
                ..Capacity::default()
            },
            model: Some(identity()),
            discovery,
            sessions: Some(SessionTerms {
                slots: 1,
                n_ctx: 2048,
                pool_reserve: 1,
            }),
            saved: Vec::new(),
        }));
        let frame = ack.to_frame(0);
        assert!(frame.len() < MAX_HANDSHAKE_FRAME, "{} bytes", frame.len());
        assert_eq!(read_message(&mut frame.as_slice()).unwrap(), ack);
    }

    /// Hands over `left` bytes and then ends, noting the widest buffer it was
    /// given to fill: what the reader had to have in memory at that point.
    struct Stingy {
        head: Vec<u8>,
        left: usize,
        widest: usize,
    }

    impl Read for Stingy {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if !self.head.is_empty() {
                let n = self.head.len().min(buf.len());
                buf[..n].copy_from_slice(&self.head[..n]);
                self.head.drain(..n);
                return Ok(n);
            }
            self.widest = self.widest.max(buf.len());
            let n = self.left.min(buf.len());
            buf[..n].fill(0);
            self.left -= n;
            Ok(n)
        }
    }

    #[test]
    fn a_claimed_length_costs_only_what_is_sent() {
        for sent in [0, 1000, UP_FRONT, 3 * UP_FRONT] {
            let mut peer = Stingy {
                head: (MAX_FRAME as u32).to_le_bytes().to_vec(),
                left: sent,
                widest: 0,
            };
            let err = read_frame(&mut peer).unwrap_err();
            let io = err.downcast_ref::<std::io::Error>().expect("an I/O error");
            assert_eq!(io.kind(), std::io::ErrorKind::UnexpectedEof);
            assert!(
                peer.widest <= sent.max(UP_FRONT),
                "{sent} bytes sent, room made for {}",
                peer.widest
            );
        }
    }

    #[test]
    fn a_frame_arrives_whole_at_any_size() {
        for len in [
            0,
            1,
            UP_FRONT - 10,
            UP_FRONT - 9,
            UP_FRONT - 8,
            UP_FRONT,
            2 * UP_FRONT - 9,
            2 * UP_FRONT,
            5 * UP_FRONT + 3,
        ] {
            let data = (0..len).map(|i| (i % 251) as u8).collect();
            round_trip(Message::Echo { data });
        }
    }

    /// Sizes past `wraps_at` values whose bytes, in `bytes_for`, wrap around to
    /// a small number that a frame could really hold.
    fn wrapping_sizes(wraps_at: u64, bytes_for: impl Fn(u64) -> u64) -> (u32, u32, usize) {
        for rows in wraps_at.isqrt()..=u64::from(u32::MAX) {
            let width = wraps_at.div_ceil(rows);
            let wrapped = bytes_for(rows.wrapping_mul(width));
            if width <= u64::from(u32::MAX) && wrapped < 1 << 16 {
                return (rows as u32, width as u32, wrapped as usize);
            }
        }
        panic!("no such sizes");
    }

    #[test]
    fn sizes_that_wrap_around_are_refused() {
        type Format = (ActivationFormat, u64, fn(u64) -> u64);
        let formats: [Format; 3] = [
            (ActivationFormat::F32, 1 << 62, |n| n.wrapping_mul(4)),
            (ActivationFormat::F16, 1 << 63, |n| n.wrapping_mul(2)),
            (
                ActivationFormat::Q8_0,
                u64::MAX / Q8_BLOCK_BYTES as u64 * Q8_BLOCK as u64,
                |n| {
                    n.div_ceil(Q8_BLOCK as u64)
                        .wrapping_mul(Q8_BLOCK_BYTES as u64)
                },
            ),
        ];
        for (format, wraps_at, bytes_for) in formats {
            let (rows, width, len) = wrapping_sizes(wraps_at, bytes_for);
            let a = Activations {
                format,
                rows,
                width,
                data: vec![0; len],
            };
            let err = a.decode().unwrap_err();
            assert!(err.to_string().contains("too many"), "{err:#}");
        }
    }

    /// Whatever arrives is an error or a message, never a panic.
    #[test]
    fn a_damaged_frame_does_not_panic() {
        let messages = [
            Message::Hello(Hello {
                version: PROTOCOL_VERSION,
                features: FEATURES,
                nonce: [3; 32],
                path: vec!["root:8400".to_string()],
            }),
            Message::Auth { proof: [1; 32] },
            Message::Forward(forward()),
            Message::ForwardBatch(vec![forward(), forward()]),
            Message::ForwardResult(result()),
            Message::ForwardBatchResult(vec![Ok(result()), Err(error())]),
            Message::AssignAck {
                identity: identity(),
                plan: vec![PlanEntry {
                    node: "a".to_string(),
                    layer_start: 0,
                    layer_end: 4,
                }],
            },
            Message::Error(error()),
        ];
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as usize
        };
        for message in messages {
            let (tag, payload) = message.encode();
            for _ in 0..2_000 {
                let mut damaged = payload.clone();
                for _ in 0..1 + next() % 4 {
                    let at = next() % damaged.len();
                    damaged[at] = [0, 0xff, next() as u8][next() % 3];
                }
                damaged.truncate(next() % (damaged.len() + 1));
                if let Ok(Message::Forward(f)) = Message::decode(tag, &damaged) {
                    let _ = f.hidden.decode();
                }
                let _ = Message::decode(next() as u8, &damaged);
            }
        }
    }

    /// A corrupt count must fail on the bytes present, not allocate for
    /// the count first.
    #[test]
    fn a_corrupt_list_count_is_refused() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&PROTOCOL_VERSION.to_le_bytes());
        payload.extend_from_slice(&0u64.to_le_bytes());
        payload.extend_from_slice(&[0; 32]);
        payload.extend_from_slice(&u32::MAX.to_le_bytes());
        let err = Message::decode(tag::HELLO, &payload).unwrap_err();
        assert!(err.to_string().contains("cannot fit"), "{err:#}");
    }

    #[test]
    fn a_loop_or_a_tree_too_deep_is_refused() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        let path = s(&["root", "a"]);
        assert!(check_path(&path, &s(&["b", "c"])).is_ok());
        // The worker is on the path already.
        let err = check_path(&path, &s(&["a"])).unwrap_err();
        assert!(err.message.contains("loop"), "{err}");
        // Something below the worker is: the root lists a node that lists
        // the root.
        assert!(check_path(&path, &s(&["b", "root"])).is_err());
        let deep: Vec<String> = (0..MAX_DEPTH).map(|i| format!("n{i}")).collect();
        assert!(check_path(&deep, &s(&["x"])).is_err());
    }

    #[test]
    fn an_unknown_error_code_reads_as_internal() {
        let (tag, mut payload) = Message::Error(error()).encode();
        payload[0] = 250;
        match Message::decode(tag, &payload).unwrap() {
            Message::Error(e) => assert_eq!(e.code, ErrorCode::Internal),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn error_text_is_one_short_line() {
        let long = format!("first line {}\nsecond line", "x".repeat(1000));
        let e = WorkerError::new(ErrorCode::Internal, long);
        assert!(e.message.len() <= WorkerError::MAX_MESSAGE);
        assert!(!e.message.contains('\n'));
        assert!(e.message.starts_with("first line"));
        // Cutting never splits a character.
        let e = WorkerError::new(ErrorCode::Internal, "é".repeat(300));
        assert!(e.message.len() <= WorkerError::MAX_MESSAGE);
    }

    fn sample(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0)
            .collect()
    }

    #[test]
    fn f32_activations_are_exact() {
        let values = sample(70);
        let a = Activations::encode(&values, 7, 10, ActivationFormat::F32);
        assert_eq!(a.data.len(), 280);
        assert_eq!(a.decode().unwrap(), values);
    }

    #[test]
    fn f16_activations_round_to_half_precision() {
        let values = sample(70);
        let a = Activations::encode(&values, 7, 10, ActivationFormat::F16);
        assert_eq!(a.data.len(), 140);
        for (v, back) in values.iter().zip(a.decode().unwrap()) {
            assert!((v - back).abs() <= v.abs() / 1024.0, "{v} -> {back}");
        }
    }

    /// Q8_0 is within half a quantization step of every value, including a
    /// last block shorter than 32 and a block of zeros.
    #[test]
    fn q8_0_activations_are_within_half_a_step() {
        let mut values = sample(70);
        values[32..64].iter_mut().for_each(|v| *v = 0.0);
        let a = Activations::encode(&values, 7, 10, ActivationFormat::Q8_0);
        assert_eq!(a.data.len(), 3 * 34);
        let back = a.decode().unwrap();
        assert_eq!(back.len(), 70);
        for (block, decoded) in values.chunks(32).zip(back.chunks(32)) {
            let amax = block.iter().fold(0f32, |m, v| m.max(v.abs()));
            let step = amax / 127.0;
            for (v, d) in block.iter().zip(decoded) {
                assert!((v - d).abs() <= step * 0.51 + 1e-6, "{v} -> {d}");
            }
        }
    }

    #[test]
    fn activations_of_the_wrong_size_are_refused() {
        let mut a = Activations::encode(&sample(8), 2, 4, ActivationFormat::F16);
        a.data.pop();
        assert!(a.decode().is_err());
    }

    #[test]
    fn activation_formats_parse_their_own_names() {
        for name in ActivationFormat::NAMES {
            assert_eq!(ActivationFormat::parse(name).unwrap().label(), name);
        }
        assert_eq!(
            ActivationFormat::parse(" Q8_0 "),
            Some(ActivationFormat::Q8_0)
        );
        assert_eq!(ActivationFormat::parse("bf16"), None);
        assert_eq!(ActivationFormat::default(), ActivationFormat::F16);
    }
}
