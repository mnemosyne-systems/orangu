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

//! A worker's stage with a standby behind it.
//!
//! A lost worker otherwise costs the whole tree a new plan and every
//! sequence a replay of all its tokens from the top. With `[workers].standby`
//! configured, the node directly above the lost worker repairs its own
//! stage instead: a standby takes the same layers, and each sequence is
//! rebuilt there from what this node sent into those layers — kept, byte
//! for byte as it went on the wire, so the rebuilt rows are the rows that
//! were lost. The stages after it are untouched: their inputs did not
//! change. The forward that met the loss is sent again, and nothing above
//! this node notices.
//!
//! What is kept is a sequence's input rows for this stage, in the stage's
//! activation format: `n_embd` values a position, 6 KiB in `f16` for a
//! 3072-wide model. Only rows a forward got through with count; a rollback
//! trims them. A sequence whose rows this stage did not all see — one
//! copied from another without them, say — is not rebuilt here and goes
//! the old way, from the top.
//!
//! The rows go to the standby in [`REPLAY_ROWS`]-position forwards, not in
//! the forwards they first came in: the same bytes, but a matrix product's
//! rounding depends on how many rows it multiplies at once, so the rebuilt
//! keys and values agree to rounding rather than bit for bit — the
//! tolerance a replay from the top has always had. Replaying forward by
//! forward would be exact, at a round trip per generated token.

use super::pipeline::{BatchItem, Stage};
use super::protocol::{ActivationFormat, Activations, ErrorCode, Rows, WorkerError};
use super::stage::EncodedStage;
use crate::engine::kv_cache::LayerHold;
use anyhow::Result;
use std::collections::HashMap;
use std::ops::Range;
use std::sync::{Arc, Mutex, RwLock};

/// Positions a rebuilt sequence is sent in, per forward.
const REPLAY_ROWS: usize = 256;

/// Finds a standby for `layers` and assigns them to it, given the name of
/// the worker that was lost.
pub type TakeStandby =
    Box<dyn Fn(Range<usize>, &str) -> Result<Arc<dyn EncodedStage>> + Send + Sync>;

/// One sequence's input rows for this stage, as sent.
#[derive(Default)]
struct Kept {
    tokens: Vec<u32>,
    data: Vec<u8>,
    /// Rows went by that this stage did not see; it cannot rebuild the
    /// sequence, and keeps nothing more of it.
    gap: bool,
}

pub struct GuardedStage {
    layers: Range<usize>,
    format: ActivationFormat,
    width: usize,
    row_bytes: usize,
    /// The worker's stage and a number that changes when it is replaced.
    current: RwLock<(Arc<dyn EncodedStage>, u64)>,
    kept: Mutex<HashMap<u64, Kept>>,
    take_standby: TakeStandby,
    /// One repair at a time; the forwards that met the same loss wait for
    /// it and go on to the new stage.
    repairing: Mutex<()>,
}

impl GuardedStage {
    /// Guards `inner`, a stage of `width`-wide rows. `None` when its rows
    /// cannot be kept apart — a `q8_0` block straddling two rows.
    pub fn new(
        inner: Arc<dyn EncodedStage>,
        width: usize,
        take_standby: TakeStandby,
    ) -> Option<Self> {
        let format = inner.format();
        let row_bytes = Activations::bytes_per_row(format, width)?;
        Some(Self {
            layers: inner.layers(),
            format,
            width,
            row_bytes,
            current: RwLock::new((inner, 0)),
            kept: Mutex::new(HashMap::new()),
            take_standby,
            repairing: Mutex::new(()),
        })
    }

    fn current(&self) -> (Arc<dyn EncodedStage>, u64) {
        let current = self.current.read().unwrap();
        (current.0.clone(), current.1)
    }

    /// Whether `error` is this stage's own worker gone — not a worker
    /// further down that the worker reports, which it repairs itself or
    /// passes on.
    fn lost(error: &anyhow::Error, name: &str) -> bool {
        error
            .chain()
            .find_map(|cause| cause.downcast_ref::<WorkerError>())
            .is_some_and(|e| e.code == ErrorCode::ChildLost && e.path == [name])
    }

    /// Records rows `start_pos..` of `session`, which a forward got through.
    fn keep(&self, session: u64, tokens: &[u32], start_pos: usize, data: &[u8]) {
        let mut kept = self.kept.lock().unwrap();
        let entry = kept.entry(session).or_default();
        if entry.gap {
            return;
        }
        let held = entry.tokens.len();
        if start_pos < held {
            entry.tokens.truncate(start_pos);
            entry.data.truncate(start_pos * self.row_bytes);
        } else if start_pos > held {
            *entry = Kept {
                gap: true,
                ..Kept::default()
            };
            return;
        }
        entry.tokens.extend_from_slice(tokens);
        entry.data.extend_from_slice(data);
    }

    /// Puts a standby in place of the stage of generation `generation`,
    /// unless another forward already did, and rebuilds every sequence
    /// kept here on it.
    fn repair(&self, generation: u64) -> Result<()> {
        let _one = self.repairing.lock().unwrap();
        let (lost, current) = self.current();
        if current != generation {
            return Ok(());
        }
        let name = lost.name();
        let standby = (self.take_standby)(self.layers.clone(), &name)?;
        {
            let kept = self.kept.lock().unwrap();
            for (&session, entry) in kept.iter().filter(|(_, e)| !e.gap && !e.tokens.is_empty()) {
                for (i, tokens) in entry.tokens.chunks(REPLAY_ROWS).enumerate() {
                    let start = i * REPLAY_ROWS;
                    let data = entry.data
                        [start * self.row_bytes..(start + tokens.len()) * self.row_bytes]
                        .to_vec();
                    let hidden = Activations {
                        format: self.format,
                        rows: tokens.len() as u32,
                        width: self.width as u32,
                        data,
                    };
                    standby.forward_encoded(session, hidden, tokens, start, Rows::None)?;
                }
            }
            log::warn!(
                "orangu-server: {name} was lost; {} took over layers {}..{} and {} sequence{} \
                 rebuilt from what was sent to it",
                standby.name(),
                self.layers.start,
                self.layers.end,
                kept.len(),
                if kept.len() == 1 { " was" } else { "s were" }
            );
        }
        *self.current.write().unwrap() = (standby, generation + 1);
        Ok(())
    }
}

impl Stage for GuardedStage {
    fn layers(&self) -> Range<usize> {
        self.layers.clone()
    }

    fn name(&self) -> String {
        self.current().0.name()
    }

    fn forward(
        &self,
        session: u64,
        hidden: Vec<f32>,
        tokens: &[u32],
        start_pos: usize,
        rows: Rows,
    ) -> Result<Vec<f32>> {
        let encoded = Activations::encode(&hidden, tokens.len(), self.width, self.format);
        let mut repaired = false;
        loop {
            let (stage, generation) = self.current();
            match stage.forward_encoded(session, encoded.clone(), tokens, start_pos, rows) {
                Ok(out) => {
                    self.keep(session, tokens, start_pos, &encoded.data);
                    return Ok(out);
                }
                Err(error) if !repaired && Self::lost(&error, &stage.name()) => {
                    repaired = true;
                    if let Err(failed) = self.repair(generation) {
                        log::warn!("orangu-server: no standby took over: {failed:#}");
                        return Err(error);
                    }
                }
                Err(error) => return Err(error),
            }
        }
    }

    /// Kept rows are recorded for each item that got through. A failure of
    /// the whole batch is answered as it is: the caller then steps each
    /// sequence on its own, through [`Self::forward`], which repairs.
    fn forward_batch(&self, items: Vec<BatchItem>) -> Result<Vec<Result<Vec<f32>, WorkerError>>> {
        let sent: Vec<(u64, Vec<u32>, usize, Vec<u8>)> = items
            .iter()
            .map(|item| {
                let encoded =
                    Activations::encode(&item.hidden, item.tokens.len(), self.width, self.format);
                (
                    item.session,
                    item.tokens.clone(),
                    item.start_pos,
                    encoded.data,
                )
            })
            .collect();
        let results = self.current().0.forward_batch(items)?;
        for ((session, tokens, start_pos, data), result) in sent.iter().zip(&results) {
            if result.is_ok() {
                self.keep(*session, tokens, *start_pos, data);
            }
        }
        Ok(results)
    }

    fn truncate(&self, session: u64, len: usize) {
        if let Some(entry) = self.kept.lock().unwrap().get_mut(&session)
            && entry.tokens.len() > len
        {
            entry.tokens.truncate(len);
            entry.data.truncate(len * self.row_bytes);
        }
        self.current().0.truncate(session, len);
    }

    fn release(&self, session: u64) {
        self.kept.lock().unwrap().remove(&session);
        self.current().0.release(session);
    }

    fn layer_rows(&self, session: u64, layer: usize, len: usize) -> Result<LayerHold> {
        self.current().0.layer_rows(session, layer, len)
    }

    fn reserve(&self, session: u64, positions: usize) -> Result<()> {
        self.current().0.reserve(session, positions)
    }

    fn fork(&self, from: u64, to: u64, len: usize) -> Result<()> {
        self.current().0.fork(from, to, len)?;
        let mut kept = self.kept.lock().unwrap();
        let copy = match kept.get(&from) {
            Some(entry) if !entry.gap && entry.tokens.len() >= len => Kept {
                tokens: entry.tokens[..len].to_vec(),
                data: entry.data[..len * self.row_bytes].to_vec(),
                gap: false,
            },
            _ => Kept {
                gap: true,
                ..Kept::default()
            },
        };
        kept.insert(to, copy);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::arch::ModelForward;
    use crate::workers::pipeline::fixture::{self, N_CTX, N_EMBD};
    use crate::workers::pipeline::{DelegatingModel, LayerPipeline};
    use crate::workers::protocol::{FEATURES, Message};
    use crate::workers::session::SessionStore;
    use crate::workers::stage::{Exchange, Loopback, MessageStage};
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    /// A worker in this process that can be killed: from then on it answers
    /// as a dead link does, `ChildLost` naming it — or, with `relayed`, as a
    /// worker whose own worker died.
    struct Killable {
        inner: Loopback,
        killed: Arc<AtomicBool>,
        relayed: bool,
    }

    impl Exchange for Killable {
        fn exchange(&self, message: &Message) -> Result<Message> {
            if self.killed.load(Ordering::Relaxed) {
                let mut error = WorkerError::new(ErrorCode::ChildLost, "gone");
                error.path = if self.relayed {
                    vec!["a:8400".to_string(), "b:8400".to_string()]
                } else {
                    vec!["a:8400".to_string()]
                };
                return Err(anyhow::Error::new(error));
            }
            self.inner.exchange(message)
        }

        fn name(&self) -> String {
            "a:8400".to_string()
        }

        fn features(&self) -> u64 {
            FEATURES
        }
    }

    struct Tree {
        split: DelegatingModel,
        worker: Arc<SessionStore>,
        standby: Arc<SessionStore>,
        killed: Arc<AtomicBool>,
        taken: Arc<AtomicUsize>,
    }

    /// Layers 0..2 here, 2..4 on a killable worker, and a standby for them
    /// (none when `spare` is false).
    fn tree(spare: bool, relayed: bool) -> Tree {
        let model = fixture::model();
        let store = |layers: Range<usize>| {
            Arc::new(SessionStore::new(
                Arc::new(LayerPipeline::new(model.clone(), layers, vec![]).unwrap()),
                N_CTX,
                8,
            ))
        };
        let (worker, standby) = (store(2..4), store(2..4));
        let killed = Arc::new(AtomicBool::new(false));
        let taken = Arc::new(AtomicUsize::new(0));
        let stage = MessageStage::over(
            Killable {
                inner: Loopback(worker.clone()),
                killed: killed.clone(),
                relayed,
            },
            2..4,
            ActivationFormat::F32,
        );
        let (to, count) = (standby.clone(), taken.clone());
        let take: TakeStandby = Box::new(move |layers, lost| {
            assert_eq!(lost, "a:8400");
            anyhow::ensure!(spare, "no standby is free");
            count.fetch_add(1, Ordering::Relaxed);
            Ok(Arc::new(MessageStage::over(
                Loopback(to.clone()),
                layers,
                ActivationFormat::F32,
            )))
        });
        let guarded = GuardedStage::new(Arc::new(stage), N_EMBD, take).unwrap();
        let split = DelegatingModel::new(Arc::new(
            LayerPipeline::new(model, 0..2, vec![Box::new(guarded)]).unwrap(),
        ))
        .unwrap();
        Tree {
            split,
            worker,
            standby,
            killed,
            taken,
        }
    }

    /// A prompt of five and two decode steps.
    fn first_steps(model: &dyn ModelForward, cache: &mut crate::engine::kv_cache::KvCache) {
        let tokens = fixture::tokens();
        model.forward(cache, &tokens[..5], 0, 0).unwrap();
        for pos in 5..7 {
            model.forward(cache, &tokens[pos..pos + 1], pos, 0).unwrap();
        }
    }

    /// The worker dies between two decode steps: the standby takes its
    /// layers, the sequence is rebuilt there from what was sent, and the
    /// step answers what the same work does by hand — layers 0..2 as they
    /// ran, 2..4 over the seven kept rows at once.
    #[test]
    fn a_standby_takes_over_a_lost_worker_s_layers() {
        let t = tree(true, false);
        let tokens = fixture::tokens();
        let mut cache = t.split.new_kv_cache(N_CTX);
        first_steps(&t.split, &mut cache);
        assert_eq!((t.worker.len(), t.standby.len()), (1, 0));
        t.killed.store(true, Ordering::Relaxed);
        let got = t.split.forward(&mut cache, &tokens[7..8], 7, 0).unwrap();
        assert_eq!(t.taken.load(Ordering::Relaxed), 1);
        assert_eq!(t.standby.len(), 1);

        let model = fixture::model();
        let mut first = model.new_kv_cache_for_layers(0..2, N_CTX);
        let mut second = model.new_kv_cache_for_layers(2..4, N_CTX);
        let mut sent = Vec::new();
        for (at, len) in [(0, 5), (5, 1), (6, 1)] {
            let chunk = &tokens[at..at + len];
            let x = model.embed(chunk).unwrap();
            sent.extend(
                model
                    .forward_layers(&mut first, x, chunk, 0..2, at)
                    .unwrap(),
            );
        }
        model
            .forward_layers(&mut second, sent, &tokens[..7], 2..4, 0)
            .unwrap();
        let x = model.embed(&tokens[7..8]).unwrap();
        let x = model
            .forward_layers(&mut first, x, &tokens[7..8], 0..2, 7)
            .unwrap();
        let x = model
            .forward_layers(&mut second, x, &tokens[7..8], 2..4, 7)
            .unwrap();
        let want = model.head(&x, 1).unwrap().pop().unwrap();
        assert_eq!(got, want);
        drop(cache);
        assert_eq!(t.standby.len(), 0, "released there");
    }

    /// With no standby free, the loss is answered as before: an error the
    /// top-level node recovers from by planning again.
    #[test]
    fn without_a_free_standby_the_loss_goes_up() {
        let t = tree(false, false);
        let mut cache = t.split.new_kv_cache(N_CTX);
        first_steps(&t.split, &mut cache);
        t.killed.store(true, Ordering::Relaxed);
        let error = t.split.forward(&mut cache, &[3], 7, 0).unwrap_err();
        let error = error.downcast::<WorkerError>().unwrap();
        assert_eq!(error.code, ErrorCode::ChildLost);
    }

    /// A worker that reports its own worker lost is still there: this node
    /// leaves it to repair itself, and takes no standby.
    #[test]
    fn a_loss_further_down_is_not_this_node_s_to_repair() {
        let t = tree(true, true);
        let mut cache = t.split.new_kv_cache(N_CTX);
        first_steps(&t.split, &mut cache);
        t.killed.store(true, Ordering::Relaxed);
        assert!(t.split.forward(&mut cache, &[3], 7, 0).is_err());
        assert_eq!(t.taken.load(Ordering::Relaxed), 0);
    }
}
