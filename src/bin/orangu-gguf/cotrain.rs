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

//! A training step's sequences spread over the channel's gguf nodes.
//!
//! Every step, each claimed node is sent the model's parameters, then takes
//! sequences one at a time from the step's list, while this node takes its
//! share of what is left, up to as many as it runs side by side: a faster
//! node takes more of them.
//! Once the list is empty this node also runs the sequences the others are
//! still at; a node it beats to one, or that ran none, is released, so a
//! node that costs more in transfer than it gives in compute does not stay
//! in the run. Each sequence's gradient is computed
//! from zero and comes back whole; the step's gradient is their sum in
//! sequence order, exactly as on one machine, so the weights a run
//! produces do not depend on how many nodes took part. A node that is lost
//! leaves its sequence to this one.

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Result, anyhow, bail};
use rayon::prelude::*;

use crate::aligned::Aligned;
use crate::jobs::{Bytes, Reader, call};
use crate::node::Claimed;
use crate::train::Network;

/// Floats in one message: parameters and gradients travel in pieces of at
/// most this many.
const PIECE: usize = 32 << 20;

/// What a node holds for training: a model, and the gradient of the last
/// sequence it ran.
pub trait Trainee: Send {
    fn kind(&self) -> u32;
    fn config(&self) -> Vec<u8>;
    /// Parameters, as many as the gradient has.
    fn len(&self) -> usize;
    /// Writes `values` into the parameters from `offset`.
    fn load(&mut self, offset: usize, values: &[f32]) -> Result<()>;
    /// Runs one window, keeping its gradient; returns its loss.
    fn run(&mut self, window: &[u32], sequence: usize) -> f32;
    /// Floats `offset..offset + count` of the kept gradient.
    fn gradient(&self, offset: usize, count: usize) -> Result<&[f32]>;
}

struct Held<N: Network> {
    model: N,
    grads: Aligned,
}

impl<N: Network + Send> Trainee for Held<N> {
    fn kind(&self) -> u32 {
        N::KIND
    }
    fn config(&self) -> Vec<u8> {
        let mut out = Vec::new();
        N::write_config(&mut out, self.model.config()).expect("a Vec takes every write");
        out
    }
    fn len(&self) -> usize {
        self.grads.len()
    }
    fn load(&mut self, offset: usize, values: &[f32]) -> Result<()> {
        let (params, _) = self.model.parts();
        let end = offset
            .checked_add(values.len())
            .filter(|&end| end <= params.len())
            .ok_or_else(|| anyhow!("parameters {offset}+{} do not fit", values.len()))?;
        params[offset..end].copy_from_slice(values);
        Ok(())
    }
    fn run(&mut self, window: &[u32], sequence: usize) -> f32 {
        self.grads.fill(0.0);
        self.model
            .pass(&window[..sequence], &window[1..], Some(&mut self.grads))
    }
    fn gradient(&self, offset: usize, count: usize) -> Result<&[f32]> {
        offset
            .checked_add(count)
            .filter(|&end| end <= self.grads.len())
            .map(|end| &self.grads[offset..end])
            .ok_or_else(|| anyhow!("gradient {offset}+{count} is out of range"))
    }
}

/// A model of `kind` and `config`, its parameters zero until loaded.
pub fn trainee(kind: u32, config: &[u8]) -> Result<Box<dyn Trainee>> {
    fn of<N: Network + Send + 'static>(config: &[u8]) -> Result<Box<dyn Trainee>> {
        let cfg = N::read_config(&mut &config[..])?;
        let len = N::parameters(&cfg);
        Ok(Box::new(Held {
            model: N::assemble(cfg, Aligned::zeros(len)),
            grads: Aligned::zeros(len),
        }))
    }
    match kind {
        k if k == crate::model::Model::KIND => of::<crate::model::Model>(config),
        k if k == crate::model4::Model::KIND => of::<crate::model4::Model>(config),
        other => bail!("a gguf node does not train models of kind {other}"),
    }
}

/// When the step this node is in last started: the parameters' first
/// piece, or the end of a pass.
static SINCE: std::sync::LazyLock<Mutex<Instant>> =
    std::sync::LazyLock::new(|| Mutex::new(Instant::now()));

/// A node's answer to a training job: `open` (kind and config), `weights`
/// (an offset and floats), `pass` (a window and the sequence length; the
/// loss back), `gradient` (an offset and a count; the floats back).
pub fn serve(
    held: &mut Option<Box<dyn Trainee>>,
    name: &str,
    r: &mut Reader<'_>,
) -> Result<Vec<u8>> {
    match name {
        "open" => {
            let kind = u32::try_from(r.u64()?)?;
            let config = r.blob()?;
            if !held
                .as_ref()
                .is_some_and(|t| t.kind() == kind && t.config() == config)
            {
                *held = Some(trainee(kind, config)?);
                eprintln!(
                    "train: a model of kind {kind} opened, {} parameters",
                    held.as_ref().map_or(0, |t| t.len())
                );
            }
            Ok(Vec::new())
        }
        "weights" => {
            let t = held
                .as_mut()
                .ok_or_else(|| anyhow!("no model is open to train"))?;
            let offset = r.u64()? as usize;
            let values = r.floats()?;
            if offset == 0 {
                *SINCE.lock().unwrap() = Instant::now();
            }
            t.load(offset, &values)?;
            if offset + values.len() == t.len() {
                eprintln!(
                    "train: parameters received, {:.0} MB in {:.1} s after the first piece",
                    t.len() as f64 * 4.0 / 1e6,
                    SINCE.lock().unwrap().elapsed().as_secs_f64()
                );
            }
            Ok(Vec::new())
        }
        "pass" => {
            let t = held
                .as_mut()
                .ok_or_else(|| anyhow!("no model is open to train"))?;
            let window = r.tokens()?;
            let sequence = r.u64()? as usize;
            if sequence == 0 || window.len() != sequence + 1 {
                bail!(
                    "a window of {} tokens is not a sequence of {sequence}",
                    window.len()
                );
            }
            #[cfg(test)]
            std::thread::sleep(std::time::Duration::from_millis(
                tests::REMOTE_PAUSE_MS.load(std::sync::atomic::Ordering::Relaxed),
            ));
            let began = Instant::now();
            let loss = t.run(&window, sequence);
            eprintln!(
                "train: a sequence of {sequence} tokens ran in {:.1} s, loss {loss:.4}",
                began.elapsed().as_secs_f64()
            );
            *SINCE.lock().unwrap() = Instant::now();
            Ok(loss.to_le_bytes().to_vec())
        }
        "gradient" => {
            let t = held
                .as_ref()
                .ok_or_else(|| anyhow!("no model is open to train"))?;
            let offset = r.u64()? as usize;
            let count = r.u64()? as usize;
            let piece = bytemuck::cast_slice(t.gradient(offset, count)?).to_vec();
            if offset + count == t.len() {
                eprintln!(
                    "train: gradient sent, {:.0} MB in {:.1} s after the pass",
                    t.len() as f64 * 4.0 / 1e6,
                    SINCE.lock().unwrap().elapsed().as_secs_f64()
                );
            }
            Ok(piece)
        }
        other => bail!("a gguf node does not take train-{other}"),
    }
}

/// Sends `model`'s kind, config and parameters to `link`.
fn send_model<N: Network>(link: &mut Claimed, model: &N) -> Result<()> {
    let mut open = Bytes::default();
    open.u64(u64::from(N::KIND));
    let mut config = Vec::new();
    N::write_config(&mut config, model.config())?;
    open.blob(&config);
    call(link, "train-open", open.0)?;
    let params = model.params();
    for (i, piece) in params.chunks(PIECE).enumerate() {
        let mut w = Bytes::default();
        w.u64((i * PIECE) as u64);
        w.floats(piece);
        call(link, "train-weights", w.0)?;
    }
    Ok(())
}

/// Runs `window` on `link` and brings its gradient back in `grads`, with
/// the seconds the pass and the gradient's transfer took.
fn remote_pass(
    link: &mut Claimed,
    window: &[u32],
    sequence: usize,
    mut grads: Aligned,
) -> Result<(f32, Aligned, f64, f64)> {
    let mut w = Bytes::default();
    w.tokens(window);
    w.u64(sequence as u64);
    let began = Instant::now();
    let loss = call(link, "train-pass", w.0)?;
    let passed = began.elapsed().as_secs_f64();
    let began = Instant::now();
    let loss = f32::from_le_bytes(
        loss.as_slice()
            .try_into()
            .map_err(|_| anyhow!("a loss came back as {} bytes", loss.len()))?,
    );
    for (i, piece) in grads.chunks_mut(PIECE).enumerate() {
        let mut w = Bytes::default();
        w.u64((i * PIECE) as u64);
        w.u64(piece.len() as u64);
        let data = call(link, "train-gradient", w.0)?;
        if data.len() != piece.len() * 4 {
            bail!("a gradient piece came back as {} bytes", data.len());
        }
        for (g, b) in piece.iter_mut().zip(data.as_chunks::<4>().0) {
            *g = f32::from_le_bytes(*b);
        }
    }
    Ok((loss, grads, passed, began.elapsed().as_secs_f64()))
}

struct Shared {
    next: usize,
    retry: Vec<usize>,
    /// Windows a node is at, with the node's index in the links.
    running: Vec<(usize, usize, Instant)>,
    /// Windows this node also runs because their node is still at them.
    again: Vec<usize>,
    pending: BTreeMap<usize, (f32, Aligned)>,
    losses: Vec<f32>,
    /// Gradient buffers free for the next window.
    pool: Vec<Aligned>,
}

impl Shared {
    /// Whether window `i`'s result is in.
    fn done(&self, i: usize) -> bool {
        i < self.losses.len() || self.pending.contains_key(&i)
    }
}

/// One step's gradient into `grads` (zero on the way in), each of `windows`
/// run here or on one of `links` and the results added in window order.
/// This node runs its share of the windows still to start, up to `at_once`
/// side by side, in buffers from `spare`; once none is left to start, it
/// runs the windows the nodes are still at, and a node it finishes one
/// before (its connection cut, its pass not waited for), or that ran none,
/// is released. `counted` is called after each window. Lost and released
/// links are removed. Returns each window's loss, in order, and how many
/// windows each node ran.
#[allow(clippy::too_many_arguments)]
pub fn step<N: Network>(
    model: &N,
    links: &mut Vec<Claimed>,
    windows: &[Vec<u32>],
    sequence: usize,
    grads: &mut [f32],
    spare: &mut Vec<Aligned>,
    at_once: usize,
    counted: &(dyn Fn() + Sync),
) -> Result<(Vec<f32>, Vec<usize>)> {
    let len = grads.len();
    let keep = spare.len();
    let shared = Mutex::new(Shared {
        next: 0,
        retry: Vec::new(),
        running: Vec::new(),
        again: Vec::new(),
        pending: BTreeMap::new(),
        losses: Vec::with_capacity(windows.len()),
        pool: std::mem::take(spare),
    });
    let buffer = || {
        shared
            .lock()
            .unwrap()
            .pool
            .pop()
            .unwrap_or_else(|| Aligned::zeros(len))
    };
    let grads = Mutex::new(grads);
    // The next window to start, for `link` or (`None`) for this node.
    let pick = |link: Option<usize>| -> Option<usize> {
        let mut s = shared.lock().unwrap();
        let i = loop {
            match s.retry.pop() {
                Some(i) if s.done(i) || s.again.contains(&i) => continue,
                Some(i) => break Some(i),
                None => {
                    break (s.next < windows.len()).then(|| {
                        s.next += 1;
                        s.next - 1
                    });
                }
            }
        }?;
        if let Some(l) = link {
            s.running.push((i, l, Instant::now()));
        }
        Some(i)
    };
    // Adds every result that is next in order; a window's gradient is taken
    // as it is when it is the first. A window already in is dropped
    // (`None`); otherwise the links still at it are returned, with when
    // they started it.
    let deliver = |i: usize, loss: f32, part: Aligned| -> Option<Vec<(usize, Instant)>> {
        let mut s = shared.lock().unwrap();
        let overtaken: Vec<(usize, Instant)> = s
            .running
            .iter()
            .filter(|&&(w, _, _)| w == i)
            .map(|&(_, l, at)| (l, at))
            .collect();
        s.running.retain(|&(w, _, _)| w != i);
        if s.done(i) {
            // Back to the pool zero, as every buffer in it is.
            let mut part = part;
            part.par_chunks_mut(1 << 16).for_each(|c| c.fill(0.0));
            s.pool.push(part);
            return None;
        }
        counted();
        s.pending.insert(i, (loss, part));
        let mut sum = grads.lock().unwrap();
        while let Some((loss, mut part)) = {
            let at = s.losses.len();
            s.pending.remove(&at)
        } {
            let first = s.losses.is_empty();
            // Added in, and left zero for the pool.
            sum.par_chunks_mut(1 << 16)
                .zip(part.par_chunks_mut(1 << 16))
                .for_each(|(sum, part)| {
                    if first {
                        sum.copy_from_slice(part);
                    } else {
                        for (s, p) in sum.iter_mut().zip(part.iter()) {
                            *s += p;
                        }
                    }
                    part.fill(0.0);
                });
            s.losses.push(loss);
            s.pool.push(part);
        }
        Some(overtaken)
    };
    let names: Vec<String> = links.iter().map(|l| l.found.name.clone()).collect();
    // Each link's socket, to cut a link this node has overtaken.
    let sockets: Vec<_> = links.iter().map(|l| l.link.tcp.clone()).collect();
    let cut: Vec<std::sync::atomic::AtomicBool> = (0..links.len())
        .map(|_| std::sync::atomic::AtomicBool::new(false))
        .collect();
    let mb = |floats: usize| floats as f64 * 4.0 / 1e6;
    let mut ran = vec![0usize; links.len() + 1];
    let lost = Mutex::new(vec![false; links.len()]);
    // Per link beaten to a window: how long it had been at it, and how long
    // it took here.
    let mut slow: Vec<Option<(f64, f64)>> = vec![None; links.len()];
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (l, link) in links.iter_mut().enumerate() {
            let (pick, deliver, buffer, shared, lost, cut) =
                (&pick, &deliver, &buffer, &shared, &lost, &cut);
            handles.push(scope.spawn(move || {
                let mut count = 0;
                let name = link.found.name.clone();
                let began = Instant::now();
                if let Err(e) = send_model(link, model) {
                    eprintln!("train: {name} left out: {e:#}");
                    lost.lock().unwrap()[l] = true;
                    return count;
                }
                let secs = began.elapsed().as_secs_f64();
                eprintln!(
                    "train: {name}: parameters sent, {:.0} MB in {secs:.1} s ({:.0} MB/s)",
                    mb(len),
                    mb(len) / secs.max(1e-9)
                );
                while let Some(i) = pick(Some(l)) {
                    match remote_pass(link, &windows[i], sequence, buffer()) {
                        Ok((loss, part, passed, fetched)) => {
                            let used = deliver(i, loss, part).is_some();
                            eprintln!(
                                "train: {name}: sequence {i} ran in {passed:.1} s there, \
                                 gradient {:.0} MB back in {fetched:.1} s ({:.0} MB/s){}",
                                mb(len),
                                mb(len) / fetched.max(1e-9),
                                if used { "" } else { "; already done here" }
                            );
                            if used {
                                count += 1;
                            }
                        }
                        Err(_) if cut[l].load(std::sync::atomic::Ordering::Relaxed) => {
                            shared.lock().unwrap().running.retain(|&(w, _, _)| w != i);
                            return count;
                        }
                        Err(e) => {
                            eprintln!(
                                "train: {} was lost ({e:#}); its sequence runs here",
                                link.found.name
                            );
                            let mut s = shared.lock().unwrap();
                            s.running.retain(|&(w, _, _)| w != i);
                            if !s.done(i) {
                                s.retry.push(i);
                            }
                            lost.lock().unwrap()[l] = true;
                            return count;
                        }
                    }
                }
                count
            }));
        }
        let mut local = 0;
        loop {
            // This node's share of what is still to start, so the nodes
            // still in the step have some left when they are ready.
            let live = handles.iter().filter(|h| !h.is_finished()).count();
            let unstarted = {
                let s = shared.lock().unwrap();
                windows.len() - s.next + s.retry.len()
            };
            let share = unstarted.div_ceil(live + 1).clamp(1, at_once.max(1));
            let mut group: Vec<usize> = std::iter::from_fn(|| pick(None)).take(share).collect();
            if group.is_empty() {
                let mut s = shared.lock().unwrap();
                let ahead: Vec<usize> = s
                    .running
                    .iter()
                    .map(|&(w, _, _)| w)
                    .filter(|&w| !s.again.contains(&w) && !s.done(w))
                    .take(at_once.max(1))
                    .collect();
                for &w in &ahead {
                    if let Some(&(_, l, at)) = s.running.iter().find(|r| r.0 == w) {
                        eprintln!(
                            "train: sequence {w} runs here too; {} has been at it {:.1} s",
                            names[l],
                            at.elapsed().as_secs_f64()
                        );
                    }
                }
                s.again.extend(&ahead);
                group = ahead;
            }
            if group.is_empty() {
                // A link lost after the list ran out leaves its window here.
                if handles.iter().all(|h| h.is_finished())
                    && shared.lock().unwrap().retry.is_empty()
                {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(2));
                continue;
            }
            #[cfg(test)]
            std::thread::sleep(std::time::Duration::from_millis(
                tests::LOCAL_PAUSE_MS.load(std::sync::atomic::Ordering::Relaxed),
            ));
            let began = Instant::now();
            let mut parts: Vec<Aligned> = group.iter().map(|_| buffer()).collect();
            let losses: Vec<f32> = parts
                .par_iter_mut()
                .zip(group.par_iter())
                .map(|(part, &i)| {
                    // Zero: every buffer the pool holds is.
                    model.pass(
                        &windows[i][..sequence],
                        &windows[i][1..],
                        Some(&mut part[..]),
                    )
                })
                .collect();
            eprintln!(
                "train: this node ran sequences {group:?} in {:.1} s",
                began.elapsed().as_secs_f64()
            );
            for ((&i, loss), part) in group.iter().zip(losses).zip(parts) {
                if let Some(overtaken) = deliver(i, loss, part) {
                    local += 1;
                    let here = began.elapsed().as_secs_f64();
                    for (l, at) in overtaken {
                        slow[l] = Some((at.elapsed().as_secs_f64(), here));
                        // Its pass is not waited for: the link is cut, and
                        // its thread ends when its read fails.
                        cut[l].store(true, std::sync::atomic::Ordering::Relaxed);
                        let _ = sockets[l].shutdown(std::net::Shutdown::Both);
                    }
                }
            }
        }
        ran[0] = local;
        for (l, h) in handles.into_iter().enumerate() {
            ran[l + 1] = h.join().unwrap_or(0);
        }
    });
    let lost = lost.into_inner().unwrap();
    let mut at = 0;
    links.retain(|link| {
        at += 1;
        let idle = ran[at] == 0;
        let reason = if lost[at - 1] {
            None
        } else if let Some((there, here)) = slow[at - 1] {
            Some(format!(
                "a sequence it had been at for {there:.1} s took {here:.1} s here"
            ))
        } else if idle {
            Some("the step's sequences were all taken before it was ready".to_string())
        } else {
            None
        };
        if let Some(reason) = reason {
            eprintln!("train: {} released: {reason}", link.found.name);
        }
        !lost[at - 1] && slow[at - 1].is_none() && !idle
    });
    let mut s = shared.into_inner().unwrap();
    s.pool.truncate(keep);
    *spare = s.pool;
    if s.losses.len() != windows.len() {
        bail!(
            "{} of {} sequences came back",
            s.losses.len(),
            windows.len()
        );
    }
    Ok((s.losses, ran))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::model4;
    use crate::node::{Settings, Worker, claim};
    use std::sync::atomic::AtomicU64;

    /// How long this node waits before each of its own windows, so the
    /// others take some.
    pub static LOCAL_PAUSE_MS: AtomicU64 = AtomicU64::new(0);

    /// How long a node waits before each window it is sent, so this one
    /// overtakes it.
    pub static REMOTE_PAUSE_MS: AtomicU64 = AtomicU64::new(0);

    fn settings() -> Settings {
        Settings {
            name: "cotrain".to_string(),
            secret: Some("s".to_string()),
            host: "127.0.0.1".to_string(),
            port: 0,
            server_port: 0,
            networks: Vec::new(),
            peers: Vec::new(),
            scan: false,
            tls: None,
        }
    }

    /// A step's gradient and losses are the same, to the bit, on one, two
    /// and three nodes, and the same as the single-machine loop's; a node
    /// this one beats to its sequence is released, and the step is still
    /// the same.
    #[test]
    fn a_step_is_the_same_on_one_two_and_three_nodes() {
        let model = model4::Model::new(model4::tests::tiny(), 7);
        let vocab = model.cfg.vocab;
        let sequence = 12;
        let windows: Vec<Vec<u32>> = (0..12)
            .map(|w| {
                (0..=sequence)
                    .map(|i| ((w * 5 + i * 3 + 1) % vocab) as u32)
                    .collect()
            })
            .collect();
        let len = model.params().len();
        let mut alone = Aligned::zeros(len);
        let mut spare = vec![Aligned::zeros(len)];
        let losses = crate::train::batch_gradient(
            &model,
            &windows,
            sequence,
            &mut alone,
            &mut spare,
            2,
            &|| {},
        );
        LOCAL_PAUSE_MS.store(40, std::sync::atomic::Ordering::Relaxed);
        for nodes in [1, 2, 3] {
            let workers: Vec<_> = (0..nodes - 1)
                .map(|_| Worker::start(settings()).unwrap().1)
                .collect();
            let found: Vec<_> = workers
                .iter()
                .map(|addr| {
                    orangu::workers::channel::probe(
                        *addr,
                        &orangu::workers::channel::channel_hash("cotrain", Some("s")),
                        None,
                    )
                    .unwrap()
                })
                .collect();
            let mut links = claim(&settings(), &format!("top{nodes}"), &found, "train");
            assert_eq!(links.len(), nodes - 1);
            let mut grads = Aligned::zeros(len);
            let mut spare = vec![Aligned::zeros(len)];
            let (got, ran) = step(
                &model,
                &mut links,
                &windows,
                sequence,
                &mut grads,
                &mut spare,
                2,
                &|| {},
            )
            .unwrap();
            assert_eq!(links.len(), nodes - 1, "a node was lost");
            assert_eq!(ran.iter().sum::<usize>(), windows.len());
            assert!(ran.iter().all(|&r| r > 0), "{nodes} nodes ran {ran:?}");
            assert_eq!(
                got.iter().map(|l| l.to_bits()).collect::<Vec<_>>(),
                losses.iter().map(|l| l.to_bits()).collect::<Vec<_>>(),
                "{nodes} nodes"
            );
            assert!(
                grads
                    .iter()
                    .zip(alone.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "{nodes} nodes: the gradient differs"
            );
        }
        // The slow node's pass is several times what the whole step takes
        // here even on a slow machine, so finishing before it is a property
        // of the step and not of the machine's speed.
        REMOTE_PAUSE_MS.store(6000, std::sync::atomic::Ordering::Relaxed);
        let addr = Worker::start(settings()).unwrap().1;
        let found = orangu::workers::channel::probe(
            addr,
            &orangu::workers::channel::channel_hash("cotrain", Some("s")),
            None,
        )
        .unwrap();
        let mut links = claim(&settings(), "slow", &[found], "train");
        assert_eq!(links.len(), 1);
        let mut grads = Aligned::zeros(len);
        let mut spare = vec![Aligned::zeros(len)];
        let began = std::time::Instant::now();
        let (got, ran) = step(
            &model,
            &mut links,
            &windows,
            sequence,
            &mut grads,
            &mut spare,
            2,
            &|| {},
        )
        .unwrap();
        assert!(
            began.elapsed() < std::time::Duration::from_millis(4500),
            "the step waited for the slow node's pass"
        );
        REMOTE_PAUSE_MS.store(0, std::sync::atomic::Ordering::Relaxed);
        LOCAL_PAUSE_MS.store(0, std::sync::atomic::Ordering::Relaxed);
        assert!(links.is_empty(), "the slow node was kept");
        assert_eq!(ran, vec![windows.len(), 0]);
        assert_eq!(
            got.iter().map(|l| l.to_bits()).collect::<Vec<_>>(),
            losses.iter().map(|l| l.to_bits()).collect::<Vec<_>>()
        );
        assert!(
            grads
                .iter()
                .zip(alone.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "the gradient differs"
        );
    }
}
