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

//! The training loop: AdamW with decoupled weight decay, a warmup and
//! cosine schedule, global gradient clipping, and a checkpoint that a later
//! run can pick up from.
//!
//! Three details are worth stating because getting any of them wrong
//! produces a run that looks fine and learns badly:
//!
//! - **Weight decay is applied to matrices only.** Decaying a norm weight
//!   pulls it toward zero, and a norm weight at zero deletes the signal it
//!   scales.
//! - **The gradient is clipped by its global norm**, across every tensor at
//!   once, not per tensor. Per-tensor clipping changes the *direction* of
//!   the step, not just its length.
//! - **The last slice of the token stream is never trained on.** It is the
//!   validation set, and a validation loss measured on trained tokens
//!   measures nothing.

use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use crate::aligned::Aligned;
use crate::model::{Config, Layout, Model, Rng};
use crate::pack::Tokens;

/// The share of the token stream held back for validation.
const VALIDATION_SHARE: f64 = 0.01;
/// Sequences a validation pass measures. Enough to be stable, few enough
/// that it is not the run's main cost.
const VALIDATION_SEQUENCES: usize = 8;

#[derive(Debug, Clone)]
pub struct Options {
    pub steps: u64,
    /// Sequences per optimizer step.
    pub batch: usize,
    pub sequence: usize,
    pub peak_lr: f32,
    pub warmup: u64,
    pub weight_decay: f32,
    pub grad_clip: f32,
    pub seed: u64,
    /// Seconds between progress lines; `0` prints only the last step.
    pub log_every: u64,
    pub eval_every: u64,
    pub checkpoint_every: u64,
}

/// AdamW's per-parameter moments, and the step count they are corrected by.
pub struct Optimizer {
    first: Aligned,
    second: Aligned,
    pub step: u64,
    beta1: f32,
    beta2: f32,
    eps: f32,
}

impl Optimizer {
    pub fn new(parameters: usize) -> Self {
        Optimizer {
            first: Aligned::zeros(parameters),
            second: Aligned::zeros(parameters),
            step: 0,
            beta1: 0.9,
            // 0.95 rather than 0.999: a pretraining run's gradient
            // statistics move fast enough that a slower second moment lags
            // behind them.
            beta2: 0.95,
            eps: 1e-8,
        }
    }

    /// The global L2 norm of the gradient.
    /// The global gradient norm, summed in a fixed order.
    ///
    /// `par_iter().sum()` folds in whatever order the work happened to
    /// split, which is a property of how threads stole from each other and
    /// not of the input — so two runs of the same seed disagree in the last
    /// bits. That would be harmless if it stayed there, but this number
    /// scales *every* gradient through the clip, so the difference is in
    /// the weights on the next step and compounding by the one after. A
    /// run with a fixed seed should be a run with a fixed answer.
    pub fn gradient_norm(grads: &[f32]) -> f32 {
        // Not covered by `the_same_seed_gives_the_same_weights_at_any_thread_count`:
        // a test-sized model's gradients fit in one chunk, so there is
        // nothing for the split to reorder. The check that covers it is
        // training a real model and hashing the file.
        let partials: Vec<f64> = grads
            .par_chunks(1 << 16)
            .map(|chunk| chunk.iter().map(|g| (*g as f64) * (*g as f64)).sum::<f64>())
            .collect();
        partials.iter().sum::<f64>().sqrt() as f32
    }

    /// One AdamW step over every parameter, decaying only the matrices.
    pub fn apply(
        &mut self,
        params: &mut [f32],
        grads: &[f32],
        layout: &Layout,
        lr: f32,
        weight_decay: f32,
        scale: f32,
    ) {
        self.step += 1;
        let correction1 = 1.0 - self.beta1.powi(self.step as i32);
        let correction2 = 1.0 - self.beta2.powi(self.step as i32);
        let (beta1, beta2, eps) = (self.beta1, self.beta2, self.eps);

        for spec in &layout.specs {
            let range = spec.offset..spec.offset + spec.len();
            let decay = if spec.is_matrix() { weight_decay } else { 0.0 };
            params[range.clone()]
                .par_iter_mut()
                .zip(grads[range.clone()].par_iter())
                .zip(self.first[range.clone()].par_iter_mut())
                .zip(self.second[range].par_iter_mut())
                .for_each(|(((p, &g), m), v)| {
                    let g = g * scale;
                    *m = beta1 * *m + (1.0 - beta1) * g;
                    *v = beta2 * *v + (1.0 - beta2) * g * g;
                    let m_hat = *m / correction1;
                    let v_hat = *v / correction2;
                    *p -= lr * (m_hat / (v_hat.sqrt() + eps) + decay * *p);
                });
        }
    }
}

/// The learning rate at `step`: linear warmup, then a cosine decay to a
/// tenth of the peak. The floor matters — a rate that reaches zero stops
/// the run learning before it stops running.
pub fn learning_rate(step: u64, options: &Options) -> f32 {
    let peak = options.peak_lr;
    if step < options.warmup {
        return peak * (step + 1) as f32 / options.warmup.max(1) as f32;
    }
    let progress = (step - options.warmup) as f32
        / (options.steps.saturating_sub(options.warmup)).max(1) as f32;
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress.min(1.0)).cos());
    peak * (0.1 + 0.9 * cosine)
}

/// What a progress line says, kept where a reporter running beside the
/// training loop can read it.
struct Status {
    /// Steps finished. The step in flight is this plus one.
    done: u64,
    /// The last finished step's numbers.
    loss: f32,
    lr: f32,
    norm: f32,
    /// Tokens since the last line, and when that line was printed.
    tokens: u64,
    window: Instant,
    /// Tokens over the whole run, for the closing line — which follows the
    /// last validation pass, and so has an empty window behind it.
    total: u64,
    /// Tokens this run has to get through in all. The estimate is against
    /// this, not against a count of steps.
    target: u64,
}

impl Status {
    /// One progress line, from whatever is known when it is asked for.
    ///
    /// `showing` is the step the line is about — the one in flight for a
    /// line printed while it runs, and the last one for the closing line.
    ///
    /// The rate and the estimate need at least one finished step to exist,
    /// so before that there is neither, which is itself the useful thing to
    /// say about a step that has been running for four minutes.
    fn line(&self, started: Instant, steps: u64, start_step: u64, showing: u64) -> String {
        let elapsed = started.elapsed();
        let mut out = format!("step {showing}/{steps}");
        if self.done > start_step {
            out += &format!(
                "  loss {:.4}  lr {:.2e}  |g| {:.3}",
                self.loss, self.lr, self.norm
            );
        }
        if self.total > 0 {
            // The window's rate while a window has something in it, and the
            // run's own when it does not. Never a zero that only means "no
            // sequence finished in the last minute".
            let rate = if self.tokens > 0 {
                self.tokens as f64 / self.window.elapsed().as_secs_f64()
            } else {
                self.total as f64 / elapsed.as_secs_f64()
            };
            out += &format!("  {}", format_rate(rate));
        }
        out += &format!("  elapsed {}", format_duration(elapsed.as_secs_f64()));
        if self.total > 0 {
            // Estimated from tokens, not from finished steps. Tokens are
            // counted as each sequence lands, so the estimate moves *during*
            // a step as well as at the end of one — which is the difference
            // between an estimate and a number that stands still for four
            // minutes and then jumps. It is against the run's own average
            // rate, so a slow patch shows up as the estimate lengthening
            // rather than as a rate the next line contradicts.
            let overall = self.total as f64 / elapsed.as_secs_f64();
            let left = self.target.saturating_sub(self.total) as f64;
            out += &format!("  eta {}", format_duration(left / overall));
        }
        out
    }
}

/// Prints a progress line every `interval` seconds until it is dropped.
///
/// A thread rather than a check at the top of the loop, because a step is
/// not a unit of time: at `2b` one takes minutes, and a loop that reports
/// between steps reports whenever it feels like it. What somebody watching
/// a run that lasts days wants to know is that it is still moving, on a
/// schedule they can predict.
struct Reporter {
    stop: Arc<AtomicBool>,
    thread: Option<thread::JoinHandle<()>>,
}

impl Reporter {
    fn start(
        status: Arc<Mutex<Status>>,
        interval: u64,
        started: Instant,
        steps: u64,
        start_step: u64,
    ) -> Option<Reporter> {
        if interval == 0 {
            return None;
        }
        let stop = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&stop);
        let thread = thread::spawn(move || {
            while !flag.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(250));
                let mut status = status.lock().unwrap_or_else(|e| e.into_inner());
                if status.window.elapsed().as_secs() < interval {
                    continue;
                }
                let running = (status.done + 1).min(steps);
                println!("{}", status.line(started, steps, start_step, running));
                status.tokens = 0;
                status.window = Instant::now();
            }
        });
        Some(Reporter {
            stop,
            thread: Some(thread),
        })
    }
}

impl Drop for Reporter {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Runs training to completion, checkpointing as it goes.
pub fn run(
    model: &mut Model,
    optimizer: &mut Optimizer,
    tokens: &Tokens,
    options: &Options,
    checkpoint: &Path,
    start_step: u64,
) -> Result<()> {
    let total = tokens.len();
    let need = options.sequence + 1;
    if total < need * 4 {
        bail!(
            "the corpus is {total} tokens — too few to train a sequence of {} on",
            options.sequence
        );
    }
    let validation_start = total - ((total as f64 * VALIDATION_SHARE) as usize).max(need);
    let train_limit = validation_start - need;

    let mut grads = Aligned::zeros(model.layout.total);
    let mut rng = Rng::new(options.seed ^ 0x5EED_0000 ^ start_step);
    let started = Instant::now();
    let status = Arc::new(Mutex::new(Status {
        done: start_step,
        loss: 0.0,
        lr: 0.0,
        norm: 0.0,
        tokens: 0,
        window: started,
        total: 0,
        target: options.steps.saturating_sub(start_step)
            * (options.batch * options.sequence) as u64,
    }));
    let reporter = Reporter::start(
        Arc::clone(&status),
        options.log_every,
        started,
        options.steps,
        start_step,
    );

    for step in start_step..options.steps {
        grads.fill(0.0);
        let mut loss = 0.0f32;
        for _ in 0..options.batch {
            let at = (rng.next_u64() % train_limit as u64) as usize;
            let window = tokens.window(at, need);
            loss +=
                model.forward_backward(&window[..options.sequence], &window[1..], Some(&mut grads));
            // Counted per sequence rather than per step: at the sizes where
            // a step takes minutes, a step is too coarse a thing to measure
            // a rate with.
            let mut status = status.lock().unwrap_or_else(|e| e.into_inner());
            status.tokens += options.sequence as u64;
            status.total += options.sequence as u64;
        }
        loss /= options.batch as f32;

        // One scale carries both the batch average and the clip, so the
        // gradient is only ever multiplied through once.
        let norm = Optimizer::gradient_norm(&grads) / options.batch as f32;
        let clip = if norm > options.grad_clip && norm > 0.0 {
            options.grad_clip / norm
        } else {
            1.0
        };
        let lr = learning_rate(step, options);
        optimizer.apply(
            &mut model.params,
            &grads,
            &model.layout,
            lr,
            options.weight_decay,
            clip / options.batch as f32,
        );

        let done = step + 1;
        {
            let mut status = status.lock().unwrap_or_else(|e| e.into_inner());
            status.done = done;
            status.loss = loss;
            status.lr = lr;
            status.norm = norm;
        }
        if options.eval_every > 0 && (done % options.eval_every == 0 || done == options.steps) {
            let validation = validate(model, tokens, validation_start, options);
            println!("  validation loss {validation:.4}");
            // The validation pass is not training, so it does not count
            // against the rate the next line reports.
            let mut status = status.lock().unwrap_or_else(|e| e.into_inner());
            status.tokens = 0;
            status.window = Instant::now();
        }
        if options.checkpoint_every > 0
            && (done % options.checkpoint_every == 0 || done == options.steps)
        {
            save(checkpoint, model, optimizer, done)?;
        }
    }

    // The last line is printed here rather than by the reporter, so a run
    // never ends on a line that is up to an interval out of date.
    drop(reporter);
    let mut status = status.lock().unwrap_or_else(|e| e.into_inner());
    // The closing line is about the run, not about the last window — which
    // the final validation pass has usually just emptied.
    status.tokens = 0;
    println!(
        "{}",
        status.line(started, options.steps, start_step, status.done)
    );
    Ok(())
}

/// Mean loss over held-out windows. Fixed offsets, so two runs measure the
/// same thing and the number is comparable across steps.
pub fn validate(model: &Model, tokens: &Tokens, start: usize, options: &Options) -> f32 {
    let need = options.sequence + 1;
    let span = tokens.len() - start;
    let count = VALIDATION_SEQUENCES.min(span / need).max(1);
    let stride = (span - need) / count;
    let mut total = 0.0;
    for i in 0..count {
        let at = start + i * stride;
        let window = tokens.window(at, need);
        total += model.forward_backward(&window[..options.sequence], &window[1..], None);
    }
    total / count as f32
}

const CHECKPOINT_MAGIC: &[u8; 8] = b"ORANGUCK";
const CHECKPOINT_VERSION: u32 = 1;

/// Writes the checkpoint to a temporary file and renames it into place, so
/// an interrupted write cannot leave a half-written checkpoint where a
/// complete one used to be.
pub fn save(path: &Path, model: &Model, optimizer: &Optimizer, step: u64) -> Result<()> {
    let temporary: PathBuf = path.with_extension("tmp");
    {
        let file = File::create(&temporary)
            .with_context(|| format!("creating {}", temporary.display()))?;
        let mut out = BufWriter::with_capacity(4 << 20, file);
        out.write_all(CHECKPOINT_MAGIC)?;
        out.write_all(&CHECKPOINT_VERSION.to_le_bytes())?;
        out.write_all(&step.to_le_bytes())?;
        write_config(&mut out, &model.cfg)?;
        write_floats(&mut out, &model.params)?;
        write_floats(&mut out, &optimizer.first)?;
        write_floats(&mut out, &optimizer.second)?;
        out.flush()?;
    }
    fs::rename(&temporary, path)
        .with_context(|| format!("renaming {} into place", temporary.display()))?;
    Ok(())
}

/// The step and the config, without the weights behind them. A caller
/// deciding whether a checkpoint is in the way should not have to read the
/// whole thing to find out.
pub fn peek(path: &Path) -> Result<(u64, Config)> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut input = BufReader::with_capacity(1 << 12, file);
    read_header(&mut input, path)
}

/// Magic, version and step, then the config. Shared so `peek` and `load`
/// cannot drift apart on the layout.
fn read_header(input: &mut impl Read, path: &Path) -> Result<(u64, Config)> {
    let mut magic = [0u8; 8];
    input.read_exact(&mut magic)?;
    if &magic != CHECKPOINT_MAGIC {
        bail!("{} is not a checkpoint", path.display());
    }
    let version = read_u32(input)?;
    if version != CHECKPOINT_VERSION {
        bail!(
            "{} is a version {version} checkpoint, not {CHECKPOINT_VERSION}",
            path.display()
        );
    }
    let step = read_u64(input)?;
    let cfg = read_config(input)?;
    Ok((step, cfg))
}

/// Reads a checkpoint back, refusing one that was written for a different
/// shape of model — resuming into a mismatched architecture would produce
/// weights that are numerically fine and completely meaningless.
pub fn load(path: &Path, expected: &Config) -> Result<(Model, Optimizer, u64)> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut input = BufReader::with_capacity(4 << 20, file);

    let (step, cfg) = read_header(&mut input, path)?;
    if cfg != *expected {
        bail!(
            "{} holds a different model than this run asks for — delete it or change the options back\n  checkpoint: {cfg:?}\n  requested:  {expected:?}",
            path.display()
        );
    }
    let params = read_floats(&mut input)?;
    let first = read_floats(&mut input)?;
    let second = read_floats(&mut input)?;

    let layout = Layout::new(&cfg);
    if params.len() != layout.total || first.len() != layout.total || second.len() != layout.total {
        bail!("{} is truncated", path.display());
    }
    let model = Model {
        cfg,
        layout,
        params: Aligned::from_slice(&params),
    };
    let optimizer = Optimizer {
        first: Aligned::from_slice(&first),
        second: Aligned::from_slice(&second),
        step,
        ..Optimizer::new(0)
    };
    Ok((model, optimizer, step))
}

fn write_config(out: &mut impl Write, cfg: &Config) -> Result<()> {
    for value in [
        cfg.vocab,
        cfg.hidden,
        cfg.ffn,
        cfg.layers,
        cfg.heads,
        cfg.kv_heads,
        cfg.head_dim,
        cfg.context,
    ] {
        out.write_all(&(value as u64).to_le_bytes())?;
    }
    out.write_all(&cfg.rope_base.to_le_bytes())?;
    out.write_all(&cfg.eps.to_le_bytes())?;
    Ok(())
}

fn read_config(input: &mut impl Read) -> Result<Config> {
    let mut values = [0usize; 8];
    for value in values.iter_mut() {
        *value = read_u64(input)? as usize;
    }
    Ok(Config {
        vocab: values[0],
        hidden: values[1],
        ffn: values[2],
        layers: values[3],
        heads: values[4],
        kv_heads: values[5],
        head_dim: values[6],
        context: values[7],
        rope_base: f32::from_le_bytes(read_bytes(input)?),
        eps: f32::from_le_bytes(read_bytes(input)?),
    })
}

fn write_floats(out: &mut impl Write, values: &[f32]) -> Result<()> {
    out.write_all(&(values.len() as u64).to_le_bytes())?;
    // Chunked rather than one allocation the size of the model.
    let mut buffer = Vec::with_capacity(1 << 20);
    for chunk in values.chunks(1 << 18) {
        buffer.clear();
        for value in chunk {
            buffer.extend_from_slice(&value.to_le_bytes());
        }
        out.write_all(&buffer)?;
    }
    Ok(())
}

fn read_floats(input: &mut impl Read) -> Result<Vec<f32>> {
    let count = read_u64(input)? as usize;
    let mut values = Vec::with_capacity(count);
    let mut buffer = vec![0u8; 1 << 20];
    let mut left = count;
    while left > 0 {
        let take = left.min(buffer.len() / 4);
        input.read_exact(&mut buffer[..take * 4])?;
        values.extend(
            buffer[..take * 4]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])),
        );
        left -= take;
    }
    Ok(values)
}

fn read_bytes<const N: usize>(input: &mut impl Read) -> Result<[u8; N]> {
    let mut buffer = [0u8; N];
    input.read_exact(&mut buffer)?;
    Ok(buffer)
}

fn read_u32(input: &mut impl Read) -> Result<u32> {
    Ok(u32::from_le_bytes(read_bytes(input)?))
}

fn read_u64(input: &mut impl Read) -> Result<u64> {
    Ok(u64::from_le_bytes(read_bytes(input)?))
}

/// `3d:1h:2m:3s`. Every unit is always present, so the fields line up between
/// one progress line and the next, and each one names itself — a run measured
/// in days should not depend on the reader counting colons.
pub fn format_duration(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "--d:--h:--m:--s".to_string();
    }
    let total = seconds as u64;
    format!(
        "{}d:{}h:{}m:{}s",
        total / 86_400,
        (total % 86_400) / 3600,
        (total % 3600) / 60,
        total % 60
    )
}

pub fn format_rate(tokens_per_second: f64) -> String {
    if tokens_per_second >= 1000.0 {
        format!("{:.1}k tok/s", tokens_per_second / 1000.0)
    } else {
        format!("{tokens_per_second:.0} tok/s")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(steps: u64) -> Options {
        Options {
            steps,
            batch: 1,
            sequence: 16,
            peak_lr: 3e-3,
            warmup: 5,
            weight_decay: 0.1,
            grad_clip: 1.0,
            seed: 1,
            log_every: 0,
            eval_every: 0,
            checkpoint_every: 0,
        }
    }

    fn tiny_config(vocab: usize) -> Config {
        Config {
            vocab,
            hidden: 32,
            ffn: 64,
            layers: 2,
            heads: 4,
            kv_heads: 2,
            head_dim: 8,
            context: 64,
            rope_base: 10000.0,
            eps: 1e-5,
        }
    }

    #[test]
    fn the_schedule_warms_up_then_decays_to_a_floor() {
        let options = options(1000);
        assert!(learning_rate(0, &options) < options.peak_lr);
        assert!((learning_rate(4, &options) - options.peak_lr).abs() < 1e-9);
        assert!(learning_rate(500, &options) < options.peak_lr);
        let end = learning_rate(999, &options);
        assert!(end > 0.0, "the rate must not reach zero");
        assert!(end < options.peak_lr * 0.15);
    }

    /// The check that the whole pipeline exists for: with real gradients
    /// and a real optimizer, the loss on a fixed sequence has to fall from
    /// the uniform-guess loss to near nothing.
    /// The same seed has to give the same weights — on any number of
    /// threads, and not just to four decimals.
    ///
    /// Comparing one width against another is what makes this catch
    /// anything: a reduction that sums in whatever order the work happened
    /// to split is *stable* while a pool is idle, so running it twice at
    /// the same width can agree by luck. Changing the width changes the
    /// split, and anything that sums in split order says so immediately.
    /// The difference lands in the weights, where every later step
    /// amplifies it.
    #[test]
    fn the_same_seed_gives_the_same_weights_at_any_thread_count() {
        fn train_a_little() -> Vec<f32> {
            let cfg = tiny_config(23);
            let mut model = Model::new(cfg, 5);
            let mut optimizer = Optimizer::new(model.layout.total);
            let options = options(12);
            let tokens: Vec<u32> = (0..257).map(|i| (i * 7 % 23) as u32).collect();
            let mut grads = Aligned::zeros(model.layout.total);
            for step in 0..options.steps {
                grads.fill(0.0);
                model.forward_backward(&tokens[..256], &tokens[1..], Some(&mut grads));
                let norm = Optimizer::gradient_norm(&grads);
                let clip = if norm > 1.0 { 1.0 / norm } else { 1.0 };
                optimizer.apply(
                    &mut model.params,
                    &grads,
                    &model.layout,
                    learning_rate(step, &options),
                    options.weight_decay,
                    clip,
                );
            }
            model.params.to_vec()
        }

        let at = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(train_a_little)
        };
        let first = at(1);
        let second = at(8);
        let differing = first
            .iter()
            .zip(second.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            differing,
            0,
            "{differing} of {} weights differ",
            first.len()
        );
    }

    #[test]
    fn training_drives_the_loss_down() {
        let cfg = tiny_config(23);
        let mut model = Model::new(cfg, 5);
        let mut optimizer = Optimizer::new(model.layout.total);
        let options = options(120);

        let tokens: Vec<u32> = (0..33).map(|i| (i * 5 % 23) as u32).collect();
        let inputs = &tokens[..32];
        let targets = &tokens[1..];

        let mut grads = Aligned::zeros(model.layout.total);
        let start = model.forward_backward(inputs, targets, None);
        for step in 0..options.steps {
            grads.fill(0.0);
            model.forward_backward(inputs, targets, Some(&mut grads));
            let norm = Optimizer::gradient_norm(&grads);
            let clip = if norm > 1.0 { 1.0 / norm } else { 1.0 };
            optimizer.apply(
                &mut model.params,
                &grads,
                &model.layout,
                learning_rate(step, &options),
                options.weight_decay,
                clip,
            );
        }
        let end = model.forward_backward(inputs, targets, None);
        assert!(
            start > 2.5 && end < 0.5,
            "loss went from {start} to {end}; it should start near ln(23) = 3.1 and end near zero"
        );
    }

    #[test]
    fn a_checkpoint_round_trips_and_refuses_a_different_shape() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("checkpoint.bin");
        let cfg = tiny_config(23);
        let mut model = Model::new(cfg.clone(), 2);
        let mut optimizer = Optimizer::new(model.layout.total);

        // Move the moments off zero so the round trip has something to
        // prove.
        let grads = vec![0.01f32; model.layout.total];
        optimizer.apply(&mut model.params, &grads, &model.layout, 1e-3, 0.1, 1.0);
        save(&path, &model, &optimizer, 7).unwrap();

        let (restored, restored_optimizer, step) = load(&path, &cfg).unwrap();
        assert_eq!(step, 7);
        assert_eq!(restored_optimizer.step, 7);
        assert_eq!(&restored.params[..], &model.params[..]);
        assert_eq!(&restored_optimizer.first[..], &optimizer.first[..]);
        assert_eq!(&restored_optimizer.second[..], &optimizer.second[..]);

        let mut other = cfg.clone();
        other.layers += 1;
        let err = match load(&path, &other) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("a checkpoint for another shape of model was accepted"),
        };
        assert!(err.contains("different model"), "{err}");
    }

    /// Weight decay must leave the norms alone.
    #[test]
    fn decay_touches_matrices_only() {
        let cfg = tiny_config(23);
        let mut model = Model::new(cfg, 4);
        let mut optimizer = Optimizer::new(model.layout.total);
        let norm_offset = model.layout.layers[0].attn_norm;
        let matrix_offset = model.layout.layers[0].wq;
        model.params[norm_offset] = 1.0;
        model.params[matrix_offset] = 1.0;

        let grads = vec![0.0f32; model.layout.total];
        optimizer.apply(&mut model.params, &grads, &model.layout, 0.1, 0.5, 1.0);
        assert_eq!(model.params[norm_offset], 1.0);
        assert!(model.params[matrix_offset] < 1.0);
    }

    #[test]
    fn peeking_reads_the_step_without_the_weights() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("checkpoint.bin");
        let cfg = tiny_config(23);
        let model = Model::new(cfg.clone(), 3);
        let optimizer = Optimizer::new(model.layout.total);
        save(&path, &model, &optimizer, 7).unwrap();

        let (step, seen) = peek(&path).unwrap();
        assert_eq!(step, 7);
        assert_eq!(seen, cfg);
    }

    #[test]
    fn durations_carry_days_once_a_run_passes_one() {
        assert_eq!(format_duration(74.0), "0d:0h:1m:14s");
        assert_eq!(format_duration(86_399.0), "0d:23h:59m:59s");
        assert_eq!(format_duration(86_400.0), "1d:0h:0m:0s");
        assert_eq!(format_duration(3.0 * 86_400.0 + 3_723.0), "3d:1h:2m:3s");
        assert_eq!(format_duration(-1.0), "--d:--h:--m:--s");
        assert_eq!(format_duration(f64::INFINITY), "--d:--h:--m:--s");
    }
}
