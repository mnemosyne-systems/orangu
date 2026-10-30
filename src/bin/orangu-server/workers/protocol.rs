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
pub const FEATURES: u64 =
    FEATURE_FORK | FEATURE_ROWS | FEATURE_HEAD | FEATURE_ECHO | FEATURE_STATE | FEATURE_METRICS;

/// [`Message::Fork`]: a worker copies one session's rows into another.
pub const FEATURE_FORK: u64 = 1;

/// [`Message::Rows`]: a worker sends back one layer's keys and values for a
/// session, so the top-level node can decode alone.
pub const FEATURE_ROWS: u64 = 2;

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

/// [`Message::LayerState`]: asked for a recurrent layer's rows, a worker
/// sends back its state, so a hybrid model's top-level node can decode
/// alone too.
pub const FEATURE_STATE: u64 = 32;

/// A `HelloAck` that ends with a list of [`DiscoveryRecord`]s: each node
/// in the tree reports where Prometheus can scrape it. A child sends the
/// list only when the parent's `Hello` advertised this bit. The list is
/// written at the very end of the message, so older versions of
/// orangu-server — which stop reading when the message runs out — lose
/// nothing and never see it.
pub const FEATURE_METRICS: u64 = 64;

/// The extensions `features` advertises, by name, for `/v1/workers`.
pub fn feature_names(features: u64) -> Vec<&'static str> {
    [
        (FEATURE_FORK, "fork"),
        (FEATURE_ROWS, "rows"),
        (FEATURE_HEAD, "head"),
        (FEATURE_ECHO, "echo"),
        (FEATURE_SWITCH, "switch"),
        (FEATURE_STATE, "state"),
        (FEATURE_METRICS, "metrics"),
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

    pub fn len(&self) -> usize {
        self.rows as usize * self.width as usize
    }

    pub fn decode(&self) -> Result<Vec<f32>> {
        let n = self.len();
        let expected = match self.format {
            ActivationFormat::F32 => n * 4,
            ActivationFormat::F16 => n * 2,
            ActivationFormat::Q8_0 => n.div_ceil(Q8_BLOCK) * Q8_BLOCK_BYTES,
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
    HelloAck(HelloAck),
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
    /// Send back layer `layer`'s first `len` positions of `session`, from
    /// this worker or whichever node below it runs that layer. Answered with
    /// [`Message::LayerRows`] — or, for a recurrent layer, by
    /// [`Message::LayerState`]. Only sent to a child that advertised
    /// [`FEATURE_ROWS`].
    Rows {
        session: u64,
        layer: u32,
        len: u32,
    },
    /// Answered with the same bytes: what a parent times a link's bandwidth
    /// by. Only sent to a child that advertised [`FEATURE_ECHO`].
    Echo {
        data: Vec<u8>,
    },
    /// One layer's rows: `len` rows of `kv_dim` keys, then as many values,
    /// `f32` little-endian in `data`.
    LayerRows {
        kv_dim: u32,
        len: u32,
        data: Vec<u8>,
    },
    /// One recurrent layer's state: its conv window, then its state
    /// matrices, each `f32` little-endian.
    LayerState {
        conv: Vec<u8>,
        state: Vec<u8>,
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
    pub const ROWS: u8 = 18;
    pub const LAYER_ROWS: u8 = 19;
    pub const ECHO: u8 = 20;
    pub const LAYER_STATE: u8 = 21;
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
                if !h.discovery.is_empty() {
                    w.discovery(&h.discovery);
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
            Self::Fork { from, to, len } => {
                w.u64(*from);
                w.u64(*to);
                w.u32(*len);
                tag::FORK
            }
            Self::Rows {
                session,
                layer,
                len,
            } => {
                w.u64(*session);
                w.u32(*layer);
                w.u32(*len);
                tag::ROWS
            }
            Self::Echo { data } => {
                w.bytes(data);
                tag::ECHO
            }
            Self::LayerRows { kv_dim, len, data } => {
                w.u32(*kv_dim);
                w.u32(*len);
                w.bytes(data);
                tag::LAYER_ROWS
            }
            Self::LayerState { conv, state } => {
                w.bytes(conv);
                w.bytes(state);
                tag::LAYER_STATE
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
            tag::HELLO_ACK => Self::HelloAck(HelloAck {
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
            }),
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
            tag::FORK => Self::Fork {
                from: r.u64()?,
                to: r.u64()?,
                len: r.u32()?,
            },
            tag::ROWS => Self::Rows {
                session: r.u64()?,
                layer: r.u32()?,
                len: r.u32()?,
            },
            tag::ECHO => Self::Echo { data: r.bytes()? },
            tag::LAYER_ROWS => Self::LayerRows {
                kv_dim: r.u32()?,
                len: r.u32()?,
                data: r.bytes()?,
            },
            tag::LAYER_STATE => Self::LayerState {
                conv: r.bytes()?,
                state: r.bytes()?,
            },
            tag::CANCEL => Self::Cancel {
                session: r.u64()?,
                step: r.u64()?,
            },
            tag::PING => Self::Ping { nonce: r.u64()? },
            tag::PONG => Self::Pong { nonce: r.u64()? },
            tag::ERROR => Self::Error(r.error()?),
            tag::READY => Self::Ready,
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

/// Reads one frame, dropping its request id: for the handshake, where
/// there is only ever one request at a time.
pub fn read_message<R: Read + ?Sized>(input: &mut R) -> Result<Message> {
    read_frame(input).map(|(_, message, _)| message)
}

/// Reads one frame: its request id, its message, and its size in bytes.
pub fn read_frame<R: Read + ?Sized>(input: &mut R) -> Result<(u64, Message, usize)> {
    let mut len = [0u8; 4];
    input.read_exact(&mut len)?;
    let len = u32::from_le_bytes(len) as usize;
    ensure!(
        len >= 9,
        "a {len} byte frame is too short to hold a type and an id"
    );
    ensure!(
        len <= MAX_FRAME,
        "a {len} byte frame is larger than the {MAX_FRAME} byte limit"
    );
    let mut body = vec![0u8; len];
    input.read_exact(&mut body)?;
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
        round_trip(Message::HelloAck(HelloAck {
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
        }));
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
        round_trip(Message::Rows {
            session: 7,
            layer: 12,
            len: 440,
        });
        round_trip(Message::Echo {
            data: vec![1, 2, 3],
        });
        round_trip(Message::LayerState {
            conv: vec![1, 2, 3, 4],
            state: vec![5, 6, 7, 8, 9, 10, 11, 12],
        });
        round_trip(Message::LayerRows {
            kv_dim: 2,
            len: 1,
            data: vec![0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64],
        });
        round_trip(Message::Fork {
            from: 3,
            to: 4,
            len: 5,
        });
        round_trip(Message::Release { session: 1 });
        round_trip(Message::Cancel {
            session: 1,
            step: 2,
        });
        round_trip(Message::Ping { nonce: 42 });
        round_trip(Message::Pong { nonce: 42 });
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
        };
        round_trip(Message::HelloAck(ack));
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
