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

//! The parent side of a forward: a [`MessageStage`] turns the [`Stage`]
//! calls a [`super::pipeline::LayerPipeline`] makes into protocol messages,
//! and checks the answers, over any [`Exchange`] — a TCP connection to a
//! worker (`super::link`), or [`Loopback`], a [`SessionStore`] in this
//! process reached through the same encoding.

use super::pipeline::{BatchItem, Stage};
#[cfg(test)]
use super::protocol::read_frame;
use super::protocol::{
    ActivationFormat, Activations, ErrorCode, Forward, ForwardResult, Message, Rows, WorkerError,
};
#[cfg(test)]
use super::session::SessionStore;
use crate::engine::kv_cache::LayerHold;
use anyhow::{Result, anyhow, bail, ensure};
use std::collections::HashMap;
use std::ops::Range;
#[cfg(test)]
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// One request, one answer.
pub trait Exchange: Send + Sync {
    fn exchange(&self, message: &Message) -> Result<Message>;

    /// Who is at the other end, for traces.
    fn name(&self) -> String;

    /// The extensions the other end advertised (`protocol::FEATURES`).
    fn features(&self) -> u64 {
        0
    }
}

/// A [`SessionStore`] in this process, every message encoded to a frame and
/// decoded again each way, so it runs the protocol as a connection would.
#[cfg(test)]
pub struct Loopback(pub Arc<SessionStore>);

#[cfg(test)]
impl Exchange for Loopback {
    fn exchange(&self, message: &Message) -> Result<Message> {
        let (_, request, _) = read_frame(&mut message.to_frame(0).as_slice())?;
        let reply = self.0.handle(request);
        read_frame(&mut reply.to_frame(0).as_slice()).map(|(_, reply, _)| reply)
    }

    fn name(&self) -> String {
        self.0.name().to_string()
    }

    fn features(&self) -> u64 {
        super::protocol::FEATURES
    }
}

pub struct MessageStage<E: Exchange> {
    exchange: E,
    layers: Range<usize>,
    format: ActivationFormat,
    next_step: AtomicU64,
    /// Rollbacks not yet sent: each rides on its session's next forward.
    pending_truncate: Mutex<HashMap<u64, usize>>,
}

#[cfg(test)]
pub type LoopbackStage = MessageStage<Loopback>;

#[cfg(test)]
impl LoopbackStage {
    pub fn new(store: Arc<SessionStore>, format: ActivationFormat) -> Self {
        let layers = store.layers();
        MessageStage::over(Loopback(store), layers, format)
    }
}

impl<E: Exchange> MessageStage<E> {
    /// A stage for `layers`, the range the worker at the other end of
    /// `exchange` was assigned.
    pub fn over(exchange: E, layers: Range<usize>, format: ActivationFormat) -> Self {
        Self {
            exchange,
            layers,
            format,
            next_step: AtomicU64::new(1),
            pending_truncate: Mutex::new(HashMap::new()),
        }
    }
}

impl<E: Exchange> MessageStage<E> {
    /// Checks that `result` answers `session`'s `step` over this stage's
    /// layers with the rows asked for, and decodes it.
    fn accept(
        &self,
        result: ForwardResult,
        session: u64,
        step: u64,
        n_tokens: usize,
        rows: Rows,
    ) -> Result<Vec<f32>> {
        let layers = &self.layers;
        ensure!(
            result.session == session && result.step == step,
            "a reply for session {} step {} answered session {session} step {step}",
            result.session,
            result.step
        );
        if result.layer_start as usize != layers.start || result.layer_end as usize != layers.end {
            return Err(anyhow!(WorkerError::new(
                ErrorCode::LayersChanged,
                format!(
                    "a reply for layers {}..{} answered layers {layers:?}",
                    result.layer_start, result.layer_end
                ),
            )));
        }
        let wanted = rows.count(n_tokens);
        ensure!(
            result.hidden.rows as usize == wanted,
            "{} rows came back, {wanted} were asked for",
            result.hidden.rows
        );
        result.hidden.decode()
    }
}

/// A stage that can be sent activations already encoded: what a
/// [`super::guard::GuardedStage`] keeps and replays, byte for byte.
pub trait EncodedStage: Stage {
    /// [`Stage::forward`] of `hidden` as encoded.
    fn forward_encoded(
        &self,
        session: u64,
        hidden: Activations,
        tokens: &[u32],
        start_pos: usize,
        rows: Rows,
    ) -> Result<Vec<f32>>;

    /// How this stage encodes what it sends.
    fn format(&self) -> ActivationFormat;
}

impl<E: Exchange> EncodedStage for MessageStage<E> {
    fn forward_encoded(
        &self,
        session: u64,
        hidden: Activations,
        tokens: &[u32],
        start_pos: usize,
        rows: Rows,
    ) -> Result<Vec<f32>> {
        let step = self.next_step.fetch_add(1, Ordering::Relaxed);
        let kv_truncate_to = self
            .pending_truncate
            .lock()
            .unwrap()
            .remove(&session)
            .map(|len| len as u32);
        let reply = self.exchange.exchange(&Message::Forward(Forward {
            session,
            step,
            start_pos: start_pos as u32,
            rows,
            kv_truncate_to,
            tokens: tokens.to_vec(),
            hidden,
        }))?;
        match reply {
            Message::ForwardResult(result) => {
                self.accept(result, session, step, tokens.len(), rows)
            }
            Message::Error(error) => Err(anyhow!(error)),
            other => bail!("unexpected reply to a forward: {other:?}"),
        }
    }

    fn format(&self) -> ActivationFormat {
        self.format
    }
}

impl<E: Exchange> Stage for MessageStage<E> {
    fn layers(&self) -> Range<usize> {
        self.layers.clone()
    }

    fn name(&self) -> String {
        self.exchange.name()
    }

    fn forward(
        &self,
        session: u64,
        hidden: Vec<f32>,
        tokens: &[u32],
        start_pos: usize,
        rows: Rows,
    ) -> Result<Vec<f32>> {
        let width = hidden.len() / tokens.len().max(1);
        let hidden = Activations::encode(&hidden, tokens.len(), width, self.format);
        self.forward_encoded(session, hidden, tokens, start_pos, rows)
    }

    fn forward_batch(&self, items: Vec<BatchItem>) -> Result<Vec<Result<Vec<f32>, WorkerError>>> {
        let first = self
            .next_step
            .fetch_add(items.len() as u64, Ordering::Relaxed);
        let expected: Vec<(u64, u64, usize, Rows)> = items
            .iter()
            .enumerate()
            .map(|(i, item)| (item.session, first + i as u64, item.tokens.len(), item.rows))
            .collect();
        let forwards = {
            let mut pending = self.pending_truncate.lock().unwrap();
            items
                .into_iter()
                .enumerate()
                .map(|(i, item)| {
                    let width = item.hidden.len() / item.tokens.len().max(1);
                    Forward {
                        session: item.session,
                        step: first + i as u64,
                        start_pos: item.start_pos as u32,
                        rows: item.rows,
                        kv_truncate_to: pending.remove(&item.session).map(|len| len as u32),
                        hidden: Activations::encode(
                            &item.hidden,
                            item.tokens.len(),
                            width,
                            self.format,
                        ),
                        tokens: item.tokens,
                    }
                })
                .collect()
        };
        match self.exchange.exchange(&Message::ForwardBatch(forwards))? {
            Message::ForwardBatchResult(results) => {
                ensure!(
                    results.len() == expected.len(),
                    "{} answers to {} forwards",
                    results.len(),
                    expected.len()
                );
                Ok(results
                    .into_iter()
                    .zip(expected)
                    .map(|(result, (session, step, n, rows))| {
                        let result = result?;
                        self.accept(result, session, step, n, rows).map_err(|e| {
                            e.downcast::<WorkerError>().unwrap_or_else(|e| {
                                WorkerError::new(ErrorCode::Internal, format!("{e:#}"))
                            })
                        })
                    })
                    .collect())
            }
            Message::Error(error) => Err(anyhow!(error)),
            other => bail!("unexpected reply to a batch of forwards: {other:?}"),
        }
    }

    fn truncate(&self, session: u64, len: usize) {
        let mut pending = self.pending_truncate.lock().unwrap();
        let entry = pending.entry(session).or_insert(len);
        *entry = (*entry).min(len);
    }

    fn release(&self, session: u64) {
        self.pending_truncate.lock().unwrap().remove(&session);
        // Nothing to do with a failure: the worker's idle eviction reclaims
        // a session whose release was lost.
        let _ = self.exchange.exchange(&Message::Release { session });
    }

    fn reserve(&self, session: u64, positions: usize) -> Result<()> {
        if self.exchange.features() & super::protocol::FEATURE_ROOM == 0 {
            return Ok(());
        }
        let positions = u32::try_from(positions)?;
        match self
            .exchange
            .exchange(&Message::Reserve { session, positions })?
        {
            Message::Reserve { .. } => Ok(()),
            Message::Error(error) => Err(anyhow!(error)),
            other => bail!("unexpected reply to a reservation: {other:?}"),
        }
    }

    fn fork(&self, from: u64, to: u64, len: usize) -> Result<()> {
        ensure!(
            self.exchange.features() & super::protocol::FEATURE_FORK != 0,
            "{} cannot copy a session",
            self.exchange.name()
        );
        let len = u32::try_from(len)?;
        match self.exchange.exchange(&Message::Fork { from, to, len })? {
            Message::Fork { .. } => Ok(()),
            Message::Error(error) => Err(anyhow!(error)),
            other => bail!("unexpected reply to a fork: {other:?}"),
        }
    }

    fn layer_rows(&self, session: u64, layer: usize, len: usize) -> Result<LayerHold> {
        ensure!(
            self.exchange.features() & super::protocol::FEATURE_ROWS != 0,
            "{} cannot send rows back",
            self.exchange.name()
        );
        let message = Message::Rows {
            session,
            layer: u32::try_from(layer)?,
            len: u32::try_from(len)?,
        };
        match self.exchange.exchange(&message)? {
            Message::LayerRows {
                kv_dim,
                len: n,
                data,
            } => {
                let (kv_dim, n) = (kv_dim as usize, n as usize);
                ensure!(
                    n == len && data.len() == 2 * 4 * n * kv_dim,
                    "{} sent {} bytes for {n} of {len} rows of {kv_dim}",
                    self.exchange.name(),
                    data.len()
                );
                let floats = floats_of(&data);
                let (k, v) = floats.split_at(n * kv_dim);
                Ok(LayerHold::Rows {
                    kv_dim,
                    k: k.to_vec(),
                    v: v.to_vec(),
                })
            }
            Message::LayerState { conv, state } => {
                ensure!(
                    conv.len() % 4 == 0 && state.len() % 4 == 0,
                    "{} sent a state of {} and {} bytes",
                    self.exchange.name(),
                    conv.len(),
                    state.len()
                );
                Ok(LayerHold::State {
                    conv: floats_of(&conv),
                    state: floats_of(&state),
                })
            }
            Message::Error(error) => Err(anyhow!(error)),
            other => bail!("unexpected reply to a rows request: {other:?}"),
        }
    }
}

/// `f32` little-endian bytes as the values they are.
fn floats_of(data: &[u8]) -> Vec<f32> {
    data.as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}
