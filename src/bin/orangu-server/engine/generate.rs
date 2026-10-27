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

//! Wires the model, tokenizer, and sampler into the one operation the HTTP
//! layer actually needs: take a prompt (already tokenized), stream back
//! generated tokens. Each call acquires a slot from the `SlotPool` (waiting
//! if every slot is busy), runs prefill+decode on its own blocking-pool
//! thread against its own KV cache, and reports throughput the same way
//! llama-server's own console log does — a completion line per request
//! through `log`, and, on a console only, a progress line in place while
//! the request runs.

use anyhow::Result;
use std::collections::VecDeque;
use std::io::Write;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::{self, UnboundedReceiver};

use super::arch::{ForwardOutcome, GreedySampleParams, ModelForward, MtpHead, MtpStep};
use super::kv_cache::KvCache;
use super::prefix_cache::PrefixCache;
use super::sampling::{Sampler, SamplingParams};
use super::scheduler::SlotPool;
use super::tokenizer::Tokenizer;
use super::tool_calls;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Clone, Debug)]
pub struct GenerateStats {
    pub prompt_tokens: usize,
    /// How many of `prompt_tokens` came from a cache (the cross-slot prefix
    /// pool or this slot's own retained cache) and so never went through a
    /// forward pass. The difference between this and `prompt_tokens` is what
    /// `prompt_time` was actually spent on, which is what makes a prefill
    /// measurement interpretable — a fast prompt that was 99% cached says
    /// nothing about prefill speed.
    pub cached_tokens: usize,
    pub prompt_time: Duration,
    pub generated_tokens: usize,
    pub generate_time: Duration,
    /// Where the request's forwards spent their time in a tree of workers
    /// (`crate::workers`): this node's own layers and each worker's round
    /// trip, prefill and decode together, in pipeline order. Empty when the
    /// model runs here alone, and until the request ends.
    pub workers: Vec<(String, Duration)>,
}

impl GenerateStats {
    pub fn prompt_tokens_per_second(&self) -> f64 {
        self.prompt_tokens as f64 / self.prompt_time.as_secs_f64().max(1e-9)
    }

    pub fn generate_tokens_per_second(&self) -> f64 {
        self.generated_tokens as f64 / self.generate_time.as_secs_f64().max(1e-9)
    }

    /// Prompt tokens that actually needed a forward pass this request.
    pub fn prefilled_tokens(&self) -> usize {
        self.prompt_tokens.saturating_sub(self.cached_tokens)
    }

    /// The line printed to stdout per completed request — llama-server's
    /// own console log carries the same two figures.
    pub fn log_line(&self) -> String {
        format!(
            "prompt {} tokens in {:.2}s ({:.2} tok/s), generated {} tokens in {:.2}s ({:.2} tok/s)",
            self.prompt_tokens,
            self.prompt_time.as_secs_f64(),
            self.prompt_tokens_per_second(),
            self.generated_tokens,
            self.generate_time.as_secs_f64(),
            self.generate_tokens_per_second(),
        )
    }
}

pub struct GenerateRequest {
    pub prompt_tokens: Vec<u32>,
    pub sampling: SamplingParams,
    /// Constrain the output to valid JSON — OpenAI's
    /// `response_format: {"type": "json_object"}`.
    ///
    /// Not a sampler parameter, because it is not a knob on the distribution:
    /// it decides which tokens *exist* at each step rather than how likely
    /// they are, and it needs the tokenizer, which `SamplingParams` has no
    /// business knowing about.
    pub json_output: bool,
    /// A JSON Schema the output is held to — a stricter `json_output`, which
    /// it takes the place of when both are set.
    pub json_schema: Option<Arc<crate::engine::json_schema::Schema>>,
    pub max_tokens: usize,
    pub stop_token_ids: Vec<u32>,
    /// Whether this request may reuse an already-computed KV cache for
    /// whatever prefix of its prompt one is available for (llama.cpp's field
    /// of the same name, and its default of `true`). Setting it `false` forces
    /// every prompt token through a real forward pass, which is what makes a
    /// prefill measurement mean anything — a cached prompt "prefills" in
    /// microseconds and measures only the lookup. It does not stop this
    /// request's own cache from being *stored* for later requests.
    pub cache_prompt: bool,
    /// Pin this request to one specific slot (llama.cpp's field of the same
    /// name), waiting for it rather than taking whichever slot is free.
    ///
    /// What it buys is cache affinity, not fairness: a slot retains the
    /// `(tokens, KvCache)` of the last request that ran on it
    /// (`engine::slot_store`), so a conversation that returns to its own slot
    /// continues from a warm prefix. Landing on a neighbour instead finds
    /// another conversation's cache and reprefills the whole prompt — which is
    /// exactly what happened for as long as this field did not exist and the
    /// client's `id_slot` was parsed away.
    ///
    /// `None` keeps the old behaviour: any free slot.
    pub id_slot: Option<usize>,
    /// Emit a [`StreamEvent::Timings`] after every generated token
    /// (llama.cpp's `timings_per_token`). Off by default: it costs one extra
    /// channel message per token, which is nothing next to a forward pass but
    /// is pure waste for a caller that never reads it.
    pub timings_per_token: bool,
    /// Serve this request under a role other than the process's own — the
    /// `x-orangu-role` an `orangu-coordinator` sets when one `orangu-server`
    /// answers for several profiles that name the same model.
    ///
    /// It reaches the engine because the role decides one thing the HTTP layer
    /// cannot apply on its own: whether a reasoning message is **hidden or
    /// marked** on the way out (`MessageHeader`). Rendering the prompt for a
    /// suppressing role while filtering the output for a permissive one is not
    /// half-right — it is the failure the whole role exists to prevent, since
    /// what the caller sees is exactly the thinking it asked not to get.
    ///
    /// `None` means "this process's role", which is every request that arrives
    /// without a coordinator in front of it.
    pub role: Option<crate::config::Role>,
    /// Where in `prompt_tokens` the conversation's last user message ends —
    /// the point every next turn's prompt repeats verbatim, whatever the
    /// template does with this turn's reply. A model with recurrent state
    /// keeps a checkpoint there (see `run`), since its state at the end of
    /// the prompt has run tokens the next turn will not repeat. `None` when
    /// unknown, or for a model without recurrent state.
    pub cache_point: Option<usize>,
    /// Work the server does for itself — the prefix warm-up after a restart
    /// (`engine::warm_prefixes`) — rather than for a client. Its cache fills
    /// a slot that retains nothing and never replaces what a client's request
    /// left there: a slot keeps one conversation, and the one a client is in
    /// the middle of is worth more than a guess at an earlier one.
    pub background: bool,
    /// The most tokens the model may reason for: once it has, the token
    /// that closes its reasoning is the next one it writes. `None` leaves
    /// the reasoning as long as the model makes it.
    pub reasoning_budget: Option<usize>,
}

impl Default for GenerateRequest {
    fn default() -> Self {
        Self {
            prompt_tokens: Vec::new(),
            sampling: SamplingParams::default(),
            json_output: false,
            json_schema: None,
            max_tokens: 0,
            stop_token_ids: Vec::new(),
            cache_prompt: true,
            id_slot: None,
            timings_per_token: false,
            role: None,
            cache_point: None,
            background: false,
            reasoning_budget: None,
        }
    }
}

pub enum StreamEvent {
    Token(String),
    /// Text the model addressed to itself: a chain-of-thought body, in one
    /// of the three ways a format *names* one — a `<think>` block, an
    /// `inkling` `<|content_thinking|>` body, a `muse-glimmer` message
    /// addressed `to=self` (see [`MessageHeader`]).
    ///
    /// Its own event rather than a flag on `Token` so that a caller which
    /// has no use for reasoning cannot accidentally treat it as the answer
    /// — which is what every caller did while this did not exist, because
    /// the markers around a `<think>` block are hidden and nothing else
    /// said where the thinking ended. A reasoning-suppressing role
    /// (`--review`) emits none of these at all; the body is dropped before
    /// it gets here.
    ///
    /// A gemma-4 `<|channel>` body is deliberately **not** one of these.
    /// That format names the channel rather than its kind, and `tool_code`
    /// is one of the names — routing it away from the answer would take a
    /// tool call out of the content `engine::tool_calls` reads.
    Reasoning(String),
    /// The admission queue was full, so this request was refused before it
    /// ever reached a slot.
    ///
    /// Its own event rather than an `Error`, because the two mean different
    /// things to a caller: an error is a request that cannot be served, and
    /// this is one that could be served later. It becomes a `503` with
    /// `Retry-After`, which a client can act on, where a `500` invites a
    /// bug report.
    Overloaded,
    /// Emitted after each prefill chunk, while the prompt is still being
    /// processed. Until this existed the only progress report was the `Done`
    /// event at the very end, so a client had nothing to show during the part
    /// of a turn that takes the longest — a 30-second prefill looked
    /// identical to a hung server.
    PromptProgress {
        total: usize,
        cached: usize,
        processed: usize,
        elapsed: Duration,
    },
    /// Emitted after each generated token when the request asked for
    /// per-token timings. Carries the server's own measurement, so a client
    /// never has to estimate a decode rate from its own wall clock — or, as
    /// this project's own client did, by re-tokenising the whole accumulated
    /// answer on every redraw.
    Timings(GenerateStats),
    Done {
        stats: GenerateStats,
        finish_reason: FinishReason,
    },
    Error(String),
}

/// A second, smaller model used to draft tokens the served model then
/// verifies — `[orangu-server].draft_model`.
///
/// Held whole rather than as a bare `Arc<dyn ModelForward>` because the two
/// things a speculative step needs beyond the forward pass — how many tokens
/// to draft, and what to call the pair in a log line — have nowhere else to
/// live that both the request path and the startup banner can reach.
pub struct DraftModel {
    pub model: Arc<dyn ModelForward>,
    /// How many tokens to draft per verification. See
    /// `[orangu-server].draft_tokens`.
    pub tokens: usize,
    /// The draft's own model label, for the banner and the acceptance log.
    pub label: String,
}

/// The served model's own multi-token-prediction draft head, when one was
/// found beside its weights — `engine::arch::qwen4exp_mtp`.
///
/// Held next to [`Engine::draft`] rather than inside it because the two are
/// alternatives, not variants of one thing: a draft model is a second model
/// and this is a block of the first. When both are configured the draft
/// model wins, since it was asked for explicitly and this was not.
pub struct MtpDraft {
    pub head: Arc<dyn MtpHead>,
    /// How many tokens to draft per verification. See
    /// `[orangu-server].draft_tokens`.
    pub tokens: usize,
    /// The head file's own label, for the banner and the acceptance log.
    pub label: String,
}

pub struct Engine {
    pub model: Arc<dyn ModelForward>,
    /// The draft half of a speculative pair, when one is configured.
    ///
    /// `None` — the default — leaves decoding exactly as it was.
    ///
    /// **Both** models must implement `ModelForward::forward_all_logits`, not
    /// only the target. The target obviously needs it — verification is one
    /// multi-position forward — but so does the draft, for a less obvious
    /// reason: it is the only entry point that keeps the KV rows on the host,
    /// and a draft cache is rolled back and re-read on every single step. A
    /// draft running through the single-token `forward` would take the fused
    /// GPU decode path instead, whose rows exist only on the device. See
    /// [`draft_forward`].
    pub draft: Option<Arc<DraftModel>>,
    /// The served model's own MTP head, when one shipped with it. Unlike
    /// [`Self::draft`] this needs no second vocabulary check and no second
    /// backend: it is a block of the model already loaded.
    pub mtp: Option<Arc<MtpDraft>>,
    pub tokenizer: Arc<Tokenizer>,
    pub chat_template_source: Option<String>,
    /// [`Self::chat_template_source`], compiled once for every request.
    pub chat_template: Option<Arc<crate::engine::chat_template::ChatTemplate>>,
    /// Whether the model's cache carries recurrent state — what decides
    /// whether a chat request computes a [`GenerateRequest::cache_point`].
    pub recurrent: bool,
    pub slots: Arc<SlotPool>,
    /// Cross-request KV-cache prefix reuse (`engine::prefix_cache`) —
    /// `None` disables it entirely (same as `Some(PrefixCache::new(0))`,
    /// just without even the pool's own mutex/lookup cost). See that
    /// module's own doc comment for what it does and doesn't cover.
    pub prefix_cache: Option<Arc<PrefixCache>>,
    /// The rendezvous that batches concurrent slots' decode steps — see
    /// `engine::decode_batch`.
    pub decode_batcher: Arc<super::decode_batch::DecodeBatcher>,
    /// Durable per-slot KV-cache persistence (`engine::slot_store`) — `Some`
    /// by default (unless `ORANGU_NO_SLOT_SAVE` is set or the home directory
    /// can't be resolved). Backs the `POST /slots/{id}?action=save|restore` endpoints
    /// and, while live, lets a slot's own retained cache serve as a prefix-
    /// reuse source for the next request on that slot (independent of the
    /// cross-slot `prefix_cache` pool). See that module's own doc comment.
    pub slot_store: Option<Arc<super::slot_store::SlotStore>>,
    /// The paged KV pool and its prefix index (`ORANGU_PAGED_KV=1`), or `None`
    /// for the per-request contiguous caches this engine has always used.
    ///
    /// Both together or neither. A pool without an index pages without
    /// sharing, which pays paging's indirection for none of its benefit; an
    /// index without a pool has nothing to point at.
    pub paged_kv: Option<(
        Arc<super::kv_pool::KvPool>,
        Arc<super::prefix_index::PrefixIndex>,
    )>,
    /// Which of `--all`/`--code`/`--review`/`--explorer`/`--embedding` this
    /// deployment was started with — read by the HTTP layer for default
    /// sampling parameters, generation-endpoint gating, and (`Review`
    /// only) reasoning suppression. See `config::Role`'s own doc comment.
    pub role: crate::config::Role,
    /// `[orangu-server].reasoning_effort`, passed into the chat template by
    /// every endpoint that renders one. Held here beside [`Self::role`]
    /// because the two answer the same question — how much thinking this
    /// deployment wants — and the three HTTP paths that render a prompt
    /// already reach for the role.
    pub reasoning_effort: Option<String>,
    /// Latency distributions and totals for `/metrics` — see
    /// `engine::metrics`. Always present: an unscraped deployment pays a
    /// handful of relaxed atomic adds per request and one per token, which is
    /// nothing beside a forward pass, and a metric that only exists when
    /// something is configured is one nobody can ask for after the fact.
    pub metrics: Arc<super::metrics::ServerMetrics>,
    /// Whether a request in flight rewrites a progress line on stdout once a
    /// second (`\r`, no newline) until its completion line replaces it.
    ///
    /// Only for a console someone can watch: under `[orangu-server].log_type
    /// = file` the progress is skipped and the file gets the completion line
    /// alone — a file has no cursor to move back, and a line rewritten every
    /// second would land in it once per second. Off under `--daemon` too,
    /// whose stdout is `/dev/null`. The `log` facade carries the completion
    /// line either way; this is the one piece of output that is not a log
    /// record, because it is not a line.
    pub live_stats: bool,
    /// The Qwen-Image pipeline, when the served model is a `qwen_image`
    /// diffusion transformer. Then [`Self::model`] and [`Self::tokenizer`]
    /// are its **text encoder** — a real `qwen2vl` language model, loaded
    /// through the ordinary path and shared with the pipeline — and the
    /// generation endpoints answer with pictures rather than tokens. `None`
    /// for every language model, which is every other architecture.
    pub image: Option<Arc<super::image::Pipeline>>,
    /// The LTX-2.5 pipeline, when the served model is an `ltxv` video
    /// diffusion transformer. Then [`Self::model`] and [`Self::tokenizer`]
    /// are its text encoder (LTX's Gemma 4), and the generation endpoints
    /// answer with clips. `None` for every other architecture.
    pub video: Option<Arc<super::video::Pipeline>>,
    /// The served model's vision tower, when one was found beside it: a
    /// chat turn's pictures become prompt tokens through it. `None` for a
    /// model that reads text alone.
    pub vision: Option<Arc<super::vision::Vision>>,
    /// A `[workers]` top-level node's two ways of serving: through
    /// its tree, or alone on the model's own paths, switched by `POST
    /// /props`. [`Self::model`], [`Self::mtp`], [`Self::prefix_cache`] and
    /// [`Self::paged_kv`] are the tree's then. `None` for every other
    /// server, which serves by those fields and nothing else.
    pub switch: Option<Arc<ServingSwitch>>,
}

/// What a request is served by: the model and what goes with it.
#[derive(Clone)]
pub struct Serving {
    pub model: Arc<dyn ModelForward>,
    pub mtp: Option<Arc<MtpDraft>>,
    pub prefix_cache: Option<Arc<PrefixCache>>,
    pub paged_kv: Option<(
        Arc<super::kv_pool::KvPool>,
        Arc<super::prefix_index::PrefixIndex>,
    )>,
}

/// A top-level `[workers]` node's model alone — its own fused decode,
/// resident prefill, prefix pool, paged KV and MTP head, as a server
/// without `[workers]` has them — beside the tree the engine's own fields
/// hold, and which of the two serves.
pub struct ServingSwitch {
    pub alone: Serving,
    /// Switched off through `POST /props`.
    off: std::sync::atomic::AtomicBool,
    /// The node judged its workers not worth it.
    chosen: std::sync::atomic::AtomicBool,
    /// Which way serves changed since a request last looked: the slots'
    /// kept conversations were made the other way, and cannot serve.
    changed: std::sync::atomic::AtomicBool,
}

impl ServingSwitch {
    pub fn new(alone: Serving) -> Self {
        Self {
            alone,
            off: std::sync::atomic::AtomicBool::new(false),
            chosen: std::sync::atomic::AtomicBool::new(false),
            changed: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Whether requests are served alone: switched off, or chosen.
    pub fn is_alone(&self) -> bool {
        self.off.load(std::sync::atomic::Ordering::Acquire)
            || self.chosen.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Whether the workers were switched off through `POST /props`.
    pub fn is_off(&self) -> bool {
        self.off.load(std::sync::atomic::Ordering::Acquire)
    }

    pub fn set_off(&self, off: bool) {
        let before = self.is_alone();
        self.off.store(off, std::sync::atomic::Ordering::Release);
        self.note_change(before);
    }

    /// The node's own judgement of its workers.
    pub fn set_chosen(&self, alone: bool) {
        let before = self.is_alone();
        self.chosen
            .store(alone, std::sync::atomic::Ordering::Release);
        self.note_change(before);
    }

    fn note_change(&self, before: bool) {
        if before != self.is_alone() {
            self.changed
                .store(true, std::sync::atomic::Ordering::Release);
        }
    }

    /// Whether which way serves changed since the last call.
    pub fn take_changed(&self) -> bool {
        self.changed
            .swap(false, std::sync::atomic::Ordering::AcqRel)
    }
}

/// What a caught generation panic is reported to the caller as: the panic's
/// own captured `detail` (message, location, backtrace — see
/// `crate::panic_capture`), except when the GPU device has been lost.
///
/// A lost device is the one panic whose detail tells the caller nothing:
/// the backtrace names whichever `wgpu` readback happened to be in flight
/// when the driver reset the device, which is about the driver, not about
/// their request — and the process is already on its way out
/// (`crate::device_lost`), so what they actually need to know is that
/// retrying in a moment will work. The full detail is not lost: it is in
/// this server's own log, written by `device_lost::fail` before the unwind
/// even started.
fn panic_report(detail: String, device_lost: bool) -> String {
    if device_lost {
        crate::device_lost::CLIENT_MESSAGE.to_string()
    } else {
        detail
    }
}

impl Engine {
    /// What to ask the chat template for: whether to think, and how hard.
    ///
    /// The effort falls back to [`DEFAULT_REASONING_EFFORT`] rather than
    /// being left unset, because "unset" is not neutral — it hands the
    /// choice to the template, and a template's own default can be the most
    /// expensive setting it has. Qwen3.x's asks for
    /// `reasoning_effort|default('xhigh')`. Measured on `Qwen3.8-27B`, same
    /// prompt ("implement a doubly linked list in C") and same machine, the
    /// difference is not a matter of degree: at `low` it closed its
    /// `<think>` block after **184** tokens and wrote the answer; left to
    /// the template it was still thinking at **8192**, having drafted the
    /// program three times, and the answer never arrived at all.
    ///
    /// `medium` rather than `low`: the point is to stop asking for the
    /// maximum, not to ask for the minimum. A deployment that wants either
    /// end says so in `[orangu-server].reasoning_effort`, and only that
    /// stated value is held against the template if it rejects it — see
    /// `chat_template::Reasoning::effort_is_default`.
    pub fn reasoning(&self) -> crate::engine::chat_template::Reasoning<'_> {
        self.reasoning_as(self.role)
    }

    /// The same, for a request being served under `role` rather than under
    /// this process's own — see `http::openai::request_role`.
    ///
    /// `enable_thinking` is the *exact* mechanism a suppressing role has:
    /// the template is told not to open a reasoning block at all, where
    /// `append_reasoning_suppression` only talks the model out of one after
    /// the fact. Reading the process role here while the rest of the request
    /// read the header's was the difference between `--review` and something
    /// that looked like it: the template kept rendering with thinking on, and
    /// an `/auto_review` served by a process started for another role spent
    /// its response cap on reasoning.
    pub fn reasoning_as(
        &self,
        role: crate::config::Role,
    ) -> crate::engine::chat_template::Reasoning<'_> {
        crate::engine::chat_template::Reasoning {
            enabled: role.enable_thinking(),
            effort: Some(
                self.reasoning_effort
                    .as_deref()
                    .unwrap_or(DEFAULT_REASONING_EFFORT),
            ),
            effort_is_default: self.reasoning_effort.is_none(),
        }
    }
}

/// The `reasoning_effort` this server asks for when the deployment names
/// none — see [`Engine::reasoning`].
pub const DEFAULT_REASONING_EFFORT: &str = "medium";

impl Engine {
    /// What the next request is served by: the engine's own model and pools,
    /// or — on a `[workers]` node switched to serve alone — the model's.
    pub fn serving(&self) -> Serving {
        match &self.switch {
            Some(switch) if switch.is_alone() => switch.alone.clone(),
            _ => Serving {
                model: self.model.clone(),
                mtp: self.mtp.clone(),
                prefix_cache: self.prefix_cache.clone(),
                paged_kv: self.paged_kv.clone(),
            },
        }
    }
}

impl Engine {
    /// [`super::teacher::teacher_forced`] in one of the request slots, so it
    /// waits its turn beside generation instead of racing it for the device.
    pub async fn teacher_forced(
        &self,
        tokens: Vec<u32>,
        rows: usize,
        top: usize,
    ) -> Result<Vec<super::teacher::TopLogprobs>, String> {
        let guard = self
            .slots
            .try_acquire()
            .await
            .ok_or("every slot is busy and the queue is full")?;
        let model = self.serving().model;
        tokio::task::spawn_blocking(move || {
            super::cpu_pools::for_request(move || {
                super::teacher::teacher_forced(model.as_ref(), &tokens, rows, top, guard.id())
            })
        })
        .await
        .map_err(|err| err.to_string())?
        .map_err(|err| format!("{err:#}"))
    }

    /// Starts generating in the background (on tokio's blocking pool) and
    /// returns a channel of [`StreamEvent`]s — waits for a free slot first
    /// if every one is already busy.
    pub async fn generate(&self, req: GenerateRequest) -> UnboundedReceiver<StreamEvent> {
        let (tx, rx) = mpsc::unbounded_channel();
        // A request that arrives in the short window between the GPU going
        // away and this process exiting (`device_lost`) is answered with
        // the same one sentence the request that hit the loss got, rather
        // than being started, run into the same dead device, and answered
        // with a second panic's worth of detail.
        if crate::device_lost::is_lost() {
            let _ = tx.send(StreamEvent::Error(
                crate::device_lost::CLIENT_MESSAGE.to_string(),
            ));
            return rx;
        }
        // Served the other way than the slots' kept conversations were made
        // (a `[workers]` node switching): they cannot serve.
        if let Some(switch) = &self.switch
            && switch.take_changed()
            && let Some(store) = &self.slot_store
        {
            store.clear();
        }
        let Serving {
            model,
            mtp,
            prefix_cache,
            paged_kv,
        } = self.serving();
        let decode_batcher = self.decode_batcher.clone();
        let tokenizer = self.tokenizer.clone();
        let slots = self.slots.clone();
        let draft = self.draft.clone();
        let slot_store = self.slot_store.clone();
        // Whether a reasoning message reaches the client is the *serving*
        // role's call — this process's, unless a coordinator named another one
        // for this request (`GenerateRequest::role`). It has to be the same
        // role the prompt was rendered for: a prompt built to suppress
        // reasoning and a stream filtered to show it disagree in the one
        // direction that matters, and the caller gets the thinking anyway.
        let process_role = self.role;
        let role = req.role.unwrap_or(process_role);
        let metrics = self.metrics.clone();
        let live_stats = self.live_stats;
        // Before the spawn, not inside it: what an operator means by "how long
        // did this request take" starts when the request arrives, and a task
        // that has not been scheduled yet is already waiting.
        let arrived = Instant::now();

        let id_slot = req.id_slot;
        tokio::spawn(async move {
            let guard = match id_slot {
                // A pinned request is waiting for one specific slot's warm
                // cache and is not competing for admission, so the queue limit
                // does not apply to it — the same reason it bypasses the
                // ticket queue entirely.
                Some(index) => slots.acquire_slot(index).await,
                None => match slots.try_acquire().await {
                    Some(guard) => guard,
                    None => {
                        // Refused, not queued. The caller gets an answer in
                        // microseconds instead of joining a pile — see
                        // `SlotPool::with_queue_limit`.
                        metrics.observe_refusal();
                        let _ = tx.send(StreamEvent::Overloaded);
                        return;
                    }
                },
            };
            // Only for a request that actually queued. A pinned request waits
            // for one named slot's warm cache rather than for capacity, so
            // folding its wait in here would make cache affinity read as
            // server overload.
            if id_slot.is_none() {
                metrics.observe_queue_wait(arrived.elapsed());
            }
            let task_tx = tx.clone();
            let result = tokio::task::spawn_blocking(move || {
                // The whole request — its prompt and its answer, and the
                // panic capture below, which must stay on one thread — in
                // the CPU pool its phase measured faster in
                // (`engine::cpu_pools`).
                // Waits here, on this blocking thread, for the pool to have
                // room for the prompt where others hold it (see
                // `wait_for_prompt_pages`), and for a peer building the
                // same new prompt prefix (`PrefixIndex::wait_or_claim`) — not
                // inside the CPU pool below. A pool worker that blocks can be
                // the one holding a stolen piece of the builder's own
                // parallel work, and then the builder never finishes.
                let prompt_pages = paged_kv.as_ref().and_then(|(pool, _)| {
                    wait_for_prompt_pages(
                        pool,
                        req.prompt_tokens.len(),
                        slot_store.as_deref(),
                        || task_tx.is_closed(),
                    )
                });
                let prefix_claim = paged_kv
                    .as_ref()
                    .filter(|_| {
                        req.cache_prompt && prefix_wait() && !model.new_kv_cache(1).has_recurrent()
                    })
                    .map(|(pool, index)| {
                        // The pages `run` will promise this request first:
                        // its prompt and the first answer token's row (see
                        // `KvCache::try_into_paged_promising`).
                        let tokens =
                            (req.prompt_tokens.len() + 1).min(model.config().n_ctx_train.max(1));
                        let pages = pool.pages_for(tokens);
                        // This request's own admission promise is traded
                        // for that one.
                        let held = prompt_pages.as_ref().map_or(0, |p| p.pages);
                        index.wait_or_claim(
                            &req.prompt_tokens,
                            || task_tx.is_closed(),
                            // Pages another request holds are shared, not
                            // promised again (`KvPool::try_reserve_sharing`).
                            |shared| pool.available() + held + shared >= pages,
                        )
                    });
                super::cpu_pools::for_request(move || {
                    // `catch_unwind` here (not left to `spawn_blocking`'s own
                    // panic-to-`JoinError` conversion below) so a panic's real
                    // detail can be recovered at all: this closure runs to
                    // completion on the *same* blocking-pool thread the panic
                    // hook (`crate::panic_capture`) just stashed its message/
                    // backtrace on, so `take_last_panic_detail` can only read
                    // it back correctly from right here — by the time this
                    // propagated out as a `JoinError` on a different
                    // (async-runtime) thread, there would be no way to
                    // associate that stash with this specific panic at all.
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        run(
                            model.as_ref(),
                            tokenizer.as_ref(),
                            draft.as_deref(),
                            mtp.as_deref(),
                            prefix_cache.as_deref(),
                            slot_store.as_deref(),
                            paged_kv.as_ref(),
                            &decode_batcher,
                            &guard,
                            req,
                            role,
                            process_role,
                            live_stats,
                            &metrics,
                            arrived,
                            task_tx.clone(),
                            prefix_claim,
                            prompt_pages,
                        )
                    }));
                    if let Err(_panic) = result {
                        let detail =
                            crate::panic_capture::take_last_panic_detail().unwrap_or_else(|| {
                                "generation task panicked (no detail captured)".to_string()
                            });
                        let message = panic_report(detail, crate::device_lost::is_lost());
                        let _ = task_tx.send(StreamEvent::Error(message));
                    }
                })
            })
            .await;
            if let Err(join_err) = result {
                // `spawn_blocking` itself failed *without* the closure
                // above panicking (e.g. the task was cancelled) — the
                // panic case is already handled and reported from inside
                // the closure, so this is only ever the non-panic
                // fallback now.
                let _ = tx.send(StreamEvent::Error(format!(
                    "generation task failed: {join_err}"
                )));
            }
        });

        rx
    }
}

/// Pages of the pool promised to a request's prompt from its admission
/// until its sequence's own promise takes them in
/// (`KvCache::try_into_paged_promising`) — see [`wait_for_prompt_pages`].
/// Given back when dropped.
struct PromptPages {
    pool: Arc<super::kv_pool::KvPool>,
    pages: usize,
}

impl PromptPages {
    /// Ends the promise without giving its pages back: the sequence's own
    /// promise took them in (`KvCache::try_into_paged_promising`).
    fn traded(mut self) {
        self.pages = 0;
    }
}

impl Drop for PromptPages {
    fn drop(&mut self) {
        self.pool.release_reserved(self.pages);
    }
}

/// Admits a request against the pool by its prompt: promises the pages the
/// prompt needs and returns that promise, waiting while the pool cannot
/// make it and other requests hold the pages it would need, until one of
/// them gives them back (or the client goes). Without the wait it is
/// admitted anyway, on a contiguous cache of its own beside the pool
/// (`run`'s fallback when `try_into_paged_promising` is refused): several such caches
/// at once outgrow the device's room, and every request on the card slows
/// with them, the ones already running too, and answers stop early when the
/// room runs out. Waiting turns the same requests into ones that each run
/// on the pool, one after the other.
///
/// Only the prompt is weighed — the answer is promised step by step as it
/// grows ([`room_for_step`]), so a long answer cap reserves nothing. The promise
/// is what stops several requests arriving together from all seeing the
/// same free pages. A prompt larger than the whole pool, or one arriving at
/// an empty pool that still cannot promise it, takes no promise and does
/// not wait: there is nothing to wait for. The slots' retained
/// conversations go before anyone waits (`SlotStore::release_paged`): an
/// idle cache gives nobody its pages back by itself.
/// `ORANGU_KV_ADMIT_WAIT=0` turns the admission off.
fn wait_for_prompt_pages(
    pool: &Arc<super::kv_pool::KvPool>,
    prompt_tokens: usize,
    slot_store: Option<&super::slot_store::SlotStore>,
    closed: impl Fn() -> bool,
) -> Option<PromptPages> {
    static ON: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        crate::engine::env::flag_on_unless_disabled("ORANGU_KV_ADMIT_WAIT")
    });
    let pages = pool.pages_for(prompt_tokens + 1);
    if !*ON || pages > pool.num_pages() {
        return None;
    }
    let mut started: Option<Instant> = None;
    loop {
        if pool.try_reserve(pages) {
            if let Some(started) = started {
                log::info!(
                    "orangu-server: [kv] the prompt went ahead after {:.1}s",
                    started.elapsed().as_secs_f64()
                );
            }
            return Some(PromptPages {
                pool: pool.clone(),
                pages,
            });
        }
        if slot_store.is_some_and(super::slot_store::SlotStore::release_paged) {
            continue;
        }
        // Nothing to wait for: no one else holds or has been promised a
        // page, or the client has gone.
        if pool.available() == pool.num_pages() || closed() {
            return None;
        }
        if started.is_none() {
            log::info!(
                "orangu-server: [kv] a {prompt_tokens}-token prompt waits for {pages} pages of \
                 the pool ({} free)",
                pool.available()
            );
            started = Some(Instant::now());
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// Records exactly one outcome for a request, however it leaves [`run`].
///
/// A `Drop` rather than a call at each exit, because `run` has five places
/// that report an error and return, one that returns on a client disconnect,
/// and a `catch_unwind` above it that turns a panic into a reply — and an
/// outcome counter with a path that forgets to increment it is worse than no
/// counter, since the total silently stops matching the request count.
/// Defaulting to `Error` and being told otherwise means a new failure path is
/// counted correctly without anyone noticing it needed to be.
struct OutcomeGuard<'a> {
    metrics: &'a super::metrics::ServerMetrics,
    arrived: Instant,
    /// `None` until something says how this ended.
    finished: Option<(super::metrics::Outcome, usize, usize, usize)>,
}

impl OutcomeGuard<'_> {
    fn finish(
        &mut self,
        outcome: super::metrics::Outcome,
        prompt: usize,
        cached: usize,
        generated: usize,
    ) {
        self.finished = Some((outcome, prompt, cached, generated));
    }
}

impl Drop for OutcomeGuard<'_> {
    fn drop(&mut self) {
        // A request that failed contributes to the error counter but not to
        // the token totals: those count work *delivered*, and a rate taken
        // over them should not move because something broke.
        let (outcome, prompt, cached, generated) =
            self.finished
                .take()
                .unwrap_or((super::metrics::Outcome::Error, 0, 0, 0));
        self.metrics
            .observe_request(self.arrived.elapsed(), outcome, prompt, cached, generated);
    }
}

/// What [`room_for_step`] found.
enum StepRoom {
    /// The step's pages are promised.
    Ready,
    /// An older sequence waits for pages, or the pool cannot promise this
    /// one's and nothing younger would give any back: this sequence gives
    /// its own back ([`rebuild_on_pool`]).
    Yield,
    /// The client went while this waited.
    Gone,
}

/// Promises the pages a step needs — the first `tokens` positions of
/// `cache` — before the step runs, while the sequence can still wait.
/// Short of them, the sequence waits for younger ones to give theirs back
/// (`KvPool`'s [`Occupant`] order), leaving the decode batch while it does
/// (`leave_batch`: the batch waits for every member, and a younger one
/// gives its pages back only at its own next step). With nothing younger to
/// wait for, the slots' retained conversations go first
/// (`SlotStore::release_paged`), then this sequence's own pages.
///
/// [`Occupant`]: super::kv_pool::Occupant
fn room_for_step(
    cache: &KvCache,
    occupant: &super::kv_pool::Occupant,
    tokens: usize,
    slot_store: Option<&super::slot_store::SlotStore>,
    closed: &dyn Fn() -> bool,
    leave_batch: &mut dyn FnMut(),
) -> StepRoom {
    loop {
        if occupant.yield_due() {
            return StepRoom::Yield;
        }
        if cache.ensure_room(tokens, Some(occupant.age())) {
            occupant.wait(false);
            return StepRoom::Ready;
        }
        occupant.wait(true);
        if !occupant.has_younger() {
            if slot_store.is_some_and(super::slot_store::SlotStore::release_paged) {
                continue;
            }
            return StepRoom::Yield;
        }
        if closed() {
            occupant.wait(false);
            return StepRoom::Gone;
        }
        leave_batch();
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Publishes the pages `cache` sealed of `tokens`, the positions it holds,
/// so a later sequence with the same tokens shares them instead of
/// recomputing them — the next turn of a conversation, whose prompt is this
/// one's prompt and reply, and this sequence itself when it rebuilds after
/// giving its pages back. The pages decode built were taken before their
/// tokens were known, so they are named here. A reply the client re-renders
/// differently hashes differently and is simply not found.
///
/// Recorded once the sequence is done with the pages rather than as each
/// seals: an aborted request leaves pages the pool reclaims, and an index
/// entry for one of those promises content that is gone. Only what the pool
/// actually holds is recorded — an index entry for a page the pool never
/// sealed is worse than no entry: `resolve` promises it, the adoption then
/// fails, and the request pays a full prefill having been told it would
/// not have to.
fn publish_pages(
    model: &dyn ModelForward,
    pool: &super::kv_pool::KvPool,
    index: &super::prefix_index::PrefixIndex,
    cache: &KvCache,
    tokens: &[u32],
) {
    let page_tokens = pool.page_tokens();
    let page_tags = super::prefix_index::page_tags(tokens, page_tokens);
    if let Some(vulkan) = model.vulkan_backend() {
        cache.fill_decoded_pages(vulkan, page_tokens);
    }
    cache.name_sealed_pages(&page_tags);
    for (i, &tag) in page_tags.iter().enumerate() {
        if pool.holds(tag) {
            index.remember(tag, &tokens[i * page_tokens..(i + 1) * page_tokens]);
        }
    }
}

/// Gives a sequence's pages back to the pool and rebuilds its cache once
/// the pool can promise it again: `built`, the positions `cache` holds, are
/// published first ([`publish_pages`]), so whatever of them is still
/// resident when the sequence returns is adopted rather than recomputed,
/// and the rest is prefilled. Waits in the pool's age order with nothing
/// held. `None` when the client goes first.
#[allow(clippy::too_many_arguments)]
fn rebuild_on_pool(
    model: &dyn ModelForward,
    pool: &Arc<super::kv_pool::KvPool>,
    index: &super::prefix_index::PrefixIndex,
    occupant: &super::kv_pool::Occupant,
    cache: KvCache,
    built: &[u32],
    tokens: usize,
    capacity: usize,
    slot_id: usize,
    cache_prompt: bool,
    closed: &dyn Fn() -> bool,
) -> Result<Option<KvCache>> {
    occupant.yielded();
    occupant.wait(true);
    publish_pages(model, pool, index, &cache, built);
    drop(cache);
    occupant.park(true);
    let pages = pool.pages_for(tokens);
    let held = loop {
        if closed() {
            occupant.park(false);
            occupant.wait(false);
            return Ok(None);
        }
        if occupant.try_reserve(pages, 0) {
            break PromptPages {
                pool: pool.clone(),
                pages,
            };
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    occupant.park(false);
    let resolved = cache_prompt.then(|| index.resolve(built, true));
    let shared: &[u64] = resolved.as_ref().map_or(&[], |r| &r.shared);
    let age = Some(occupant.age());
    let Ok(mut cache) = model
        .new_kv_cache(capacity)
        .try_into_paged_sharing_promising(pool.clone(), shared, held.pages, tokens, age)
        .or_else(|cache| cache.try_into_paged_promising(pool.clone(), held.pages, tokens, age))
    else {
        occupant.wait(false);
        anyhow::bail!("the KV pool could not take back a sequence it had promised room");
    };
    held.traded();
    let mut reused = 0;
    if !shared.is_empty()
        && let Some(tokens) = cache.adopt_shared_pages(shared, pool.page_tokens())
    {
        reused = tokens;
    }
    cache.set_page_tags(&super::prefix_index::page_tags(built, pool.page_tokens()));
    let filled = prefill(
        model,
        &mut cache,
        &built[reused..],
        reused,
        slot_id,
        None,
        &mut |_| {
            if closed() {
                Err(anyhow::Error::new(PrefillCancelled))
            } else {
                Ok(())
            }
        },
    );
    occupant.wait(false);
    match filled {
        Ok(_) => {}
        Err(err) if err.is::<PrefillCancelled>() => return Ok(None),
        Err(err) => return Err(err),
    }
    if let Some(vulkan) = model.vulkan_backend() {
        vulkan.fill_deferred_kv_rows(&mut cache);
    }
    cache.commit_pages();
    Ok(Some(cache))
}

#[allow(clippy::too_many_arguments)]
fn run(
    model: &dyn ModelForward,
    tokenizer: &Tokenizer,
    draft: Option<&DraftModel>,
    mtp: Option<&MtpDraft>,
    prefix_cache: Option<&PrefixCache>,
    slot_store: Option<&super::slot_store::SlotStore>,
    paged_kv: Option<&(
        Arc<super::kv_pool::KvPool>,
        Arc<super::prefix_index::PrefixIndex>,
    )>,
    decode_batcher: &super::decode_batch::DecodeBatcher,
    guard: &super::scheduler::SlotGuard,
    req: GenerateRequest,
    role: crate::config::Role,
    // This process's own role, for the one line that reports when a request
    // was served under another — see the completion line below.
    process_role: crate::config::Role,
    // See `Engine::live_stats`.
    live_stats: bool,
    metrics: &super::metrics::ServerMetrics,
    arrived: Instant,
    tx: mpsc::UnboundedSender<StreamEvent>,
    // This request's claim on the prompt pages it is about to build — see
    // `Engine::generate`, which takes it before the request enters the CPU
    // pool. Ended once they are published, after the prefill.
    prefix_claim: Option<super::prefix_index::Claim>,
    // The pages the prompt was admitted against — see
    // `wait_for_prompt_pages` — traded for the request's own promise for
    // its whole length, or given back where it takes none.
    prompt_pages: Option<PromptPages>,
) -> Result<()> {
    let mut outcome = OutcomeGuard {
        metrics,
        arrived,
        finished: None,
    };
    let config = model.config();
    let capacity = (req.prompt_tokens.len() + req.max_tokens).min(config.n_ctx_train.max(1));
    if req.prompt_tokens.len() > capacity {
        let _ = tx.send(StreamEvent::Error(format!(
            "prompt ({} tokens) exceeds the model's context length ({})",
            req.prompt_tokens.len(),
            config.n_ctx_train
        )));
        return Ok(());
    }
    // The device's room bounds the capacity the same way the model's context
    // length just did: `max_tokens` is a cap on the answer, and a cap that
    // reaches past the room is shortened to it, not refused. The web console
    // asks for 32768 on every turn and the API's own default is 8192 —
    // neither knows the card, and a device whose room is 16384 tokens turned
    // the console away on every prompt. Only a prompt that leaves no room to
    // answer is refused — over the ceiling a contiguous cache grows its
    // device mirror until the driver resets the card (see
    // [`KV_DEVICE_TOKENS`]), and the paged pool cannot hold it either.
    let capacity = match kv_device_token_ceiling() {
        Some(ceiling) if !prompt_fits_device(req.prompt_tokens.len(), ceiling) => {
            let _ = tx.send(StreamEvent::Error(format!(
                "prompt ({} tokens) plus one generated token needs {} tokens of KV cache, \
                 more than the {ceiling} this device has room for — send a shorter prompt, \
                 or run the model on a device with more memory",
                req.prompt_tokens.len(),
                req.prompt_tokens.len().saturating_add(1),
            )));
            return Ok(());
        }
        Some(ceiling) => capacity.min(ceiling),
        None => capacity,
    };

    guard.set_prompt_tokens(req.prompt_tokens.len());
    // Reuse a previous request's already-computed KV cache for however
    // much of this prompt matches one — see `engine::prefix_cache`'s own
    // doc comment. Always allocate this request's own cache fresh, at its
    // own `capacity` (never reused directly: two requests' capacities can
    // differ), then copy the matched prefix into it — `reused_len` tokens'
    // worth of the prompt never need a forward pass at all. Left at 1
    // fewer than the full matched length whenever it would otherwise equal
    // this prompt's own length, so there's always at least one real
    // forward call to produce fresh logits for the first sampled token
    // from (this only matters for the degenerate case of re-sending a
    // prompt identical to one already fully cached).
    //
    // `cache_prompt: false` skips both reuse paths below, so the prompt is
    // prefilled in full. Storing this request's own cache afterwards is
    // unaffected — the flag governs what this request may *read*.
    // The paged path, when a pool exists. It replaces both reuse mechanisms
    // below rather than layering on them: those hand a whole cache to one
    // request at a time, and the point of the pool is that a prefix has as many
    // holders as want it.
    let mut reused_len = 0usize;
    // The pool can only promise room it has. A request longer than the whole
    // pool — a review prompt carrying a few thousand lines of source can be
    // several times it — keeps the contiguous cache instead, which grows on
    // demand and is what every request used before the pool existed. What that
    // costs
    // is page sharing for this one request. What asking anyway used to cost
    // was the server: the sequence discovered the pool was full a page at a
    // time, deep inside a forward pass with nowhere to report it.
    //
    // `try_into_paged_promising` hands the cache back on refusal, so the fallback is the
    // very cache that was already built rather than a second one.
    //
    // The prefix is resolved first, so a request sharing pages another one
    // holds is promised only the rest of its prompt
    // (`KvCache::try_into_paged_sharing_promising`), and adopts exactly what it was
    // promised against. `keep_last`: a fully matched prompt still needs one
    // page of real work to produce fresh logits from — the same reason the
    // contiguous path clamps a full match by one token.
    let resolved = paged_kv
        .filter(|_| req.cache_prompt)
        .map(|(_, index)| index.resolve(&req.prompt_tokens, true));
    // The admission's promise is traded for the sequence's in one step, so
    // nobody else can take its pages in between.
    let held = prompt_pages.as_ref().map_or(0, |p| p.pages);
    // Promised its prompt and the first answer token's row, as admitted;
    // every step after that is promised its own rows as it comes
    // (`room_for_step`), so an answer cap far beyond what the reply will use
    // keeps nobody else off the pool.
    let first_room = req.prompt_tokens.len() + 1;
    let mut occupant = None;
    let (mut new_cache, paged_kv) = match paged_kv {
        Some((pool, index)) => {
            let shared: &[u64] = resolved.as_ref().map_or(&[], |r| &r.shared);
            let entered = pool.enter();
            let age = Some(entered.age());
            match model
                .new_kv_cache(capacity)
                .try_into_paged_sharing_promising(pool.clone(), shared, held, first_room, age)
                .or_else(|cache| {
                    cache.try_into_paged_promising(pool.clone(), held, first_room, age)
                }) {
                Ok(cache) => {
                    // Its pages are the sequence's now.
                    if let Some(p) = prompt_pages {
                        p.traded();
                    }
                    occupant = Some(entered);
                    (cache, Some((pool, index)))
                }
                Err(cache) => {
                    log::info!(
                        "orangu-server: [kv] slot {} runs on a cache of its own: the pool \
                         could not promise its {}-token prompt",
                        guard.id(),
                        req.prompt_tokens.len()
                    );
                    (cache, None)
                }
            }
        }
        None => (model.new_kv_cache(capacity), None),
    };
    if let Some((pool, index)) = paged_kv {
        // The architecture builds its own cache — recurrent state included —
        // and only its positional layers move into the pool.
        if let Some(resolved) = resolved.filter(|_| req.cache_prompt) {
            if crate::engine::env::flag_on("ORANGU_KV_TRACE") {
                eprintln!(
                    "[kv-trace] resolve: {} prompt tokens, {} pages shared of {} known",
                    req.prompt_tokens.len(),
                    resolved.shared.len(),
                    index.len()
                );
            }
            if !resolved.shared.is_empty()
                && let Some(tokens) =
                    new_cache.adopt_shared_pages(&resolved.shared, pool.page_tokens())
            {
                reused_len = tokens;
            }
            // Nothing shared: this slot's own conversation, retained at its
            // last request's end or loaded by a `restore`, may still hold the
            // prefix — after a restart the pool and index start empty.
            if reused_len == 0
                && let Some(store) = slot_store
            {
                reused_len = store.reuse_into_paged(guard.id(), &req.prompt_tokens, &mut new_cache);
            }
            if crate::engine::env::flag_on("ORANGU_KV_TRACE") {
                let stats = pool.stats();
                eprintln!(
                    "[kv-trace] slot {} reuses {reused_len} of {} prompt tokens; pool {} live, {} \
                     cached, {} never used, {} available",
                    guard.id(),
                    req.prompt_tokens.len(),
                    stats.live_pages,
                    stats.cached_pages,
                    stats.free_pages,
                    pool.available()
                );
            }
        }
        // Pages this request seals become findable under these identities. Set
        // even when nothing was adopted: this request is the one that makes the
        // prefix available to the next.
        new_cache.set_page_tags(&super::prefix_index::page_tags(
            &req.prompt_tokens,
            pool.page_tokens(),
        ));
    }
    if paged_kv.is_none()
        && req.cache_prompt
        && let Some(pool) = prefix_cache
        && let Some((matched, entry)) = pool.take_best_match(&req.prompt_tokens)
    {
        let mut entry = entry;
        let matched = matched.min(req.prompt_tokens.len().saturating_sub(1));
        if matched > 0 && new_cache.take_remote_prefix(&mut entry.cache, matched) {
            // Moved, not copied: `take_best_match` removed this entry from the
            // pool, so nothing else can still be reading it and its buffers
            // can become this request's own. The slot store below cannot do
            // the same — it keeps its snapshot for the slot's next request.
            new_cache.adopt_prefix(entry.cache, matched);
            reused_len = matched;
        }
    }
    // The cross-slot pool (above) has first claim; only if it found nothing
    // does this slot's own durably-retained cache (`engine::slot_store`, the
    // source a `restore` populated) get consulted — it applies the same
    // leave-one-token and recurrent-state rules internally.
    if paged_kv.is_none()
        && req.cache_prompt
        && reused_len == 0
        && let Some(store) = slot_store
    {
        reused_len = store.reuse_into(guard.id(), &req.prompt_tokens, &mut new_cache);
    }
    // `Option` (not a plain `KvCache`) so the decode loop can *move* it
    // that call crosses to a different thread (whichever one ends up
    // leading this batch), which needs ownership, not a borrow. `.take()`/
    // reassignment stands in for a borrow everywhere else, at zero real
    // cost (this is never actually `None` except mid-swap).
    let mut cache = Some(new_cache);
    let mut sampler = Sampler::new(req.sampling);
    if let Some(schema) = req.json_schema.clone() {
        sampler = sampler.with_constraint(crate::engine::sampling::Constraint::schema(
            schema,
            tokenizer.token_bytes_shared(),
            req.stop_token_ids.clone(),
        ));
    } else if req.json_output {
        // The byte table is built on the tokenizer's first constrained
        // request and shared from then on, so an unconstrained deployment
        // never pays for it.
        sampler = sampler.with_constraint(crate::engine::sampling::Constraint::json(
            tokenizer.token_bytes_shared(),
            req.stop_token_ids.clone(),
        ));
    }
    let mut history = req.prompt_tokens.clone();

    // Speculative decoding, greedy or sampled, and not while fused batching
    // is running the decode step. Sampled is exact because the sampler's draw
    // for a position depends on the seed and the position alone
    // (`sampling::position_draw`), so the verifier picks at each drafted
    // position what plain decoding would pick there. `spec_buf` (below) holds
    // tokens a speculative step already verified and committed to the KV
    // cache, waiting to be emitted before the next forward.
    //
    // Under a constraint too: verification picks each position with the
    // request's own sampler, in order, so the constraint has accepted exactly
    // the tokens before a position when that position is picked. A drafted
    // token it forbids differs from the pick and ends the accepted run there.
    //
    // A target with recurrent layers verifies from a checkpoint of its
    // state and returns to it when a draft is rejected
    // (`KvCache::rollback`), then runs the accepted tokens again, or keeps
    // the state the verify left behind its last accepted token where the
    // device kept one (`KvCache::keep_verified`) — by default only where it
    // verifies on its device (`spec_recurrent`): the host's prompt loop costs
    // more than the steps it saves. One whose layers other nodes hold
    // cannot roll back, and does not speculate. A draft model's cache
    // follows the target's with `KvCache::truncate` alone, so a recurrent
    // draft model is not built, and neither is an MTP head on a recurrent
    // target, whose states the rollback would have to replay too.
    let target_recurrent = cache.as_ref().is_some_and(KvCache::has_recurrent);
    let may_speculate = (!target_recurrent
        || (spec_recurrent(model) && cache.as_ref().is_some_and(KvCache::can_roll_back)))
        && !draft.is_some_and(|d| d.model.new_kv_cache(1).has_recurrent());
    let mtp = mtp.filter(|_| !target_recurrent);
    // A configured draft model wins over the model's own MTP head, which
    // wins over prompt-lookup. All three guess at the same tokens, so
    // running more than one only turns the loser's misses into a second,
    // wasted verification forward; the order is explicit-over-implicit, then
    // trained-over-free.
    //
    // **Built before prefill, not after it.** An MTP head reads the served
    // model's hidden state at the position before each token, and prefill is
    // where the prompt's states exist — a head that first appears afterwards
    // has no way back to them and would enter generation with an empty
    // cache.
    let mut drafter = match (
        may_speculate,
        draft,
        mtp.filter(|_| model.mtp_state_width().is_some()),
        speculative_config(),
    ) {
        (false, ..) => None,
        (true, Some(draft), _, _) => Some(Drafter::Model {
            model: draft.model.as_ref(),
            tokens: draft.tokens,
            cache: draft.model.new_kv_cache(capacity),
            committed: 0,
        }),
        (true, None, Some(mtp), _) if mtp.head.reads_served_cache() => Some(Drafter::Assistant {
            head: mtp.head.as_ref(),
            tokens: mtp.tokens,
            states: Vec::new(),
            states_start: 0,
            width: mtp.head.state_width(),
        }),
        (true, None, Some(mtp), lookup) => Some(Drafter::Mtp {
            head: mtp.head.as_ref(),
            tokens: mtp.tokens,
            cache: mtp.head.new_kv_cache(capacity),
            committed: 0,
            first_pos: None,
            states: Vec::new(),
            states_start: 0,
            width: mtp.head.state_width(),
            echo: mtp
                .head
                .block_size()
                .and(lookup)
                .map(|(ngram, max_draft)| EchoLookup {
                    ngram,
                    max_draft,
                    echo_draft: spec_echo_draft(model.vulkan_backend().is_some()),
                }),
        }),
        (true, None, None, Some((ngram, max_draft))) => Some(Drafter::PromptLookup {
            ngram,
            max_draft,
            echo_draft: spec_echo_draft(model.vulkan_backend().is_some()),
        }),
        (true, None, None, None) => None,
    };

    let prompt_start = Instant::now();
    let total_prompt = req.prompt_tokens.len();
    let progress_tx = tx.clone();
    let mut on_chunk = |processed: usize| {
        // A client that hung up mid-prompt stops the prefill at the next
        // chunk, as one that hangs up mid-answer stops the decode at the
        // next token: a long prompt otherwise holds the slot, and in a tree
        // every worker, for the rest of its prefill.
        if progress_tx.is_closed() {
            return Err(anyhow::Error::new(PrefillCancelled));
        }
        // Cached tokens never went through a forward pass, so they are
        // already "processed" as far as a progress bar is concerned —
        // otherwise a mostly-cached prompt appears to start at zero and jump.
        let _ = progress_tx.send(StreamEvent::PromptProgress {
            total: total_prompt,
            cached: reused_len,
            processed: reused_len + processed,
            elapsed: prompt_start.elapsed(),
        });
        Ok(())
    };
    // Committed after the whole prefill, not inside it. Prefill is
    // **layer-major** — `arch::gemma` pushes every token of layer 0, then every
    // token of layer 1 — so during it a page can be complete for one layer and
    // untouched by the rest. The only moment a span of positions is known to
    // have been through every layer is when the pass returns, which is why this
    // is the caller's call to make and not something the cache can infer.
    // A recurrent model's checkpoint at the conversation's cache point, for
    // the slot store to keep in place of the end state — see
    // `GenerateRequest::cache_point`. Only where something will keep it.
    let checkpoint_at = req.cache_point.filter(|&at| {
        slot_store.is_some()
            && paged_kv.is_none()
            && at > reused_len
            && at < req.prompt_tokens.len()
            && cache.as_ref().is_some_and(KvCache::has_recurrent)
    });
    let mut checkpoint: Option<(usize, Vec<crate::engine::kv_cache::RecurrentLayerState>)> = None;
    let prefilled = match checkpoint_at {
        Some(at) => {
            let cache = cache.as_mut().expect("cache is always Some here");
            let head = &req.prompt_tokens[reused_len..at];
            prefill(
                model,
                cache,
                head,
                reused_len,
                guard.id(),
                None,
                &mut on_chunk,
            )
            .and_then(|_| {
                checkpoint = Some((at, cache.recurrent_checkpoint()));
                let done = head.len();
                prefill(
                    model,
                    cache,
                    &req.prompt_tokens[at..],
                    at,
                    guard.id(),
                    None,
                    &mut |processed| on_chunk(done + processed),
                )
            })
        }
        None => prefill(
            model,
            cache.as_mut().expect("cache is always Some here"),
            &req.prompt_tokens[reused_len..],
            reused_len,
            guard.id(),
            drafter.as_mut(),
            &mut on_chunk,
        ),
    };
    let logits = match prefilled {
        Ok(l) => l,
        Err(err) if err.is::<PrefillCancelled>() => {
            outcome.finish(
                super::metrics::Outcome::Cancelled,
                req.prompt_tokens.len(),
                reused_len,
                0,
            );
            return Ok(());
        }
        Err(err) => {
            // `{err:?}` (anyhow's chain-plus-backtrace Debug format, not
            // `{err}`'s bare top-level message) — `main`'s own unconditional
            // `RUST_BACKTRACE=1` means this always includes a captured
            // backtrace, not just whatever `.context()` calls happened to
            // add, matching the detail a panic's own captured backtrace
            // (`panic_capture`) gives for a debug report worth saving.
            let _ = tx.send(StreamEvent::Error(format!("{err:?}")));
            return Ok(());
        }
    };
    if let Some((pool, index)) = paged_kv
        && let Some(cache) = cache.as_mut()
    {
        // The last chunk's parked rows first: a page is sealed only once
        // every layer holds all of it.
        if let Some(vulkan) = model.vulkan_backend() {
            vulkan.fill_deferred_kv_rows(cache);
        }
        cache.commit_pages();
        // The prompt's whole pages are final once the prefill returns, so
        // they are published now rather than at the request's end: a peer
        // waiting on this prompt (`prefix_claim`) adopts them and builds only
        // its own suffix while this one decodes.
        if prefix_claim.is_some() {
            let page_tokens = pool.page_tokens();
            let tags = super::prefix_index::page_tags(&req.prompt_tokens, page_tokens);
            for (i, &tag) in tags.iter().enumerate() {
                if pool.holds(tag) {
                    index.remember(
                        tag,
                        &req.prompt_tokens[i * page_tokens..(i + 1) * page_tokens],
                    );
                }
            }
        }
    }
    drop(prefix_claim);
    let prompt_time = prompt_start.elapsed();
    // Prefill is never decode-shaped (`n_tokens > 1`), so it never takes a
    // GPU-fused sampling fast path either way — this first sample always
    // runs the plain CPU chain.
    let mut next = sampler.sample(&logits, &history, history.len());

    let generate_start = Instant::now();
    let mut generated = 0usize;
    let finish_reason;
    let mut last_report = Instant::now();
    let mut reported = false;
    let mut spec_buf: VecDeque<u32> = VecDeque::new();
    let mut draft_control = super::draft_control::DraftControl::new();
    let mut spec_accepted = 0usize;
    let mut spec_steps = 0usize;
    let mut header = MessageHeader::for_prompt(tokenizer, &req.prompt_tokens, role);
    let (mut reasoning_tokens, mut budget_spent) = (0usize, false);
    // One decoder for the whole stream, not one call per token: a character
    // wider than a byte is routinely several tokens, and decoding each of them
    // alone replaces both halves. See `tokenizer::StreamDecoder`.
    let mut decoder = crate::engine::tokenizer::StreamDecoder::default();
    // When the previous token was produced, so the gap to the next one can be
    // observed. `None` until there is a previous one to measure from.
    let mut last_token_at: Option<Instant> = None;
    // This slot decodes with the others from here on — one forward for all
    // of them per step (`engine::decode_batch`). Only the plain path: a
    // draft or a multi-token head steps on its own schedule.
    let batches = draft.is_none() && mtp.is_none() && model.batches_decode();
    let mut batch_registration = batches.then(|| decode_batcher.register()).flatten();
    // Another request's prefill narrows its chunks while this one decodes —
    // see `decode_yield_width`.
    let decoding = Decoding::enter();
    loop {
        if generated >= req.max_tokens {
            finish_reason = FinishReason::Length;
            break;
        }
        if req.stop_token_ids.contains(&next) {
            finish_reason = FinishReason::Stop;
            break;
        }
        history.push(next);
        generated += 1;
        guard.set_generated_tokens(generated);
        // Time to first token counts from arrival, so it carries the queue
        // wait and the prefill together — which is what an interactive caller
        // actually waits through. Every later token contributes a gap.
        //
        // The first token *produced*, not the first the client sees: a chat
        // format's structural prefix is filtered out of the stream
        // (`MessageHeader`), and charging the template's shape to the
        // server's latency would make two models with the same speed report
        // different numbers.
        let now = Instant::now();
        match last_token_at {
            None => metrics.observe_first_token(now.duration_since(arrived)),
            Some(previous) => metrics.observe_inter_token(now.duration_since(previous)),
        }
        last_token_at = Some(now);
        // Structural tokens (a chat format's turn/channel/tool markers, and
        // any stray BOS/EOS) still go into `history` so the KV cache and any
        // continued generation stay correct, but are not rendered to the
        // user — otherwise a gemma-4 reply spills literal `<turn|>`/
        // `<channel|>` tokens into the stream (`skip_special_tokens`).
        let emitted = if tokenizer.is_special(next) {
            // A message marker opens or closes a header; it is hidden
            // either way, like every other structural token, but ending one
            // can call for a separator between two visible messages.
            match header.observe_marker(next) {
                Some(separator) => Some(separator.to_string()),
                // Almost every special token is structural and stays hidden.
                // The exception is a tool-call marker: the model writes its
                // call *between* special tokens, so suppressing them would
                // leave the arguments as loose prose with nothing to say
                // they were a call. Rendered back to literal text here and
                // read by `engine::tool_calls`; the HTTP layer removes the
                // whole span from what the user sees, so this never reaches
                // a chat client as content.
                None => tokenizer
                    .token_text(next)
                    .and_then(tool_calls::marker_text)
                    .map(str::to_string),
            }
        } else {
            // Suppressed while this is a message *header* (the recipient the
            // model wrote for the format's benefit, not the reader's), and
            // while it is a message body the role hides.
            header.observe_text(decoder.push(tokenizer, next))
        };
        if let Some(text) = emitted {
            let _ = tx.send(if header.in_reasoning() {
                StreamEvent::Reasoning(text)
            } else {
                StreamEvent::Token(text)
            });
        }
        // A reasoning budget spent: the next token closes the reasoning.
        if let Some(budget) = req.reasoning_budget
            && header.in_reasoning()
        {
            reasoning_tokens += 1;
            if reasoning_tokens >= budget
                && !budget_spent
                && let Some(close) = header.reasoning_close()
            {
                sampler.force(close);
                budget_spent = true;
            }
        }
        // Whether anyone is still listening, asked **every token** rather than
        // only when there was text to send.
        //
        // The check used to ride on the send, which meant a token rendering to
        // nothing never made it. That is not a rare case: a chat format's
        // structural markers are suppressed, and a whole reasoning body is
        // suppressed under `--review`, so a request could generate to
        // `max_tokens` — holding the slot every other request is queued for —
        // for a client that hung up at the first token. Measured on a model
        // whose output was almost entirely suppressed: a disconnected 2000-token
        // request ran to completion; on one with visible text the same request
        // stopped within a token.
        //
        // Its own outcome, not an error: nothing went wrong, and the tokens
        // generated so far were real work a rate should see.
        if tx.is_closed() {
            outcome.finish(
                super::metrics::Outcome::Cancelled,
                req.prompt_tokens.len(),
                reused_len,
                generated,
            );
            return Ok(());
        }
        // Not after the first token: it was sampled from the prefill's own
        // logits, so `generate_time` at that point is a few microseconds of
        // bookkeeping and the "rate" comes out in the tens of thousands of
        // tokens per second. A rate needs an interval between two tokens
        // before it means anything.
        if req.timings_per_token && generated >= 2 {
            let _ = tx.send(StreamEvent::Timings(GenerateStats {
                prompt_tokens: req.prompt_tokens.len(),
                cached_tokens: reused_len,
                prompt_time,
                generated_tokens: generated,
                generate_time: generate_start.elapsed(),
                workers: Vec::new(),
            }));
        }
        if live_stats && last_report.elapsed() >= Duration::from_secs(1) {
            let partial = GenerateStats {
                prompt_tokens: req.prompt_tokens.len(),
                cached_tokens: reused_len,
                prompt_time,
                generated_tokens: generated,
                generate_time: generate_start.elapsed(),
                workers: Vec::new(),
            };
            // \x1b[K ("erase to end of line") clears any leftover tail from
            // a longer previous update before the cursor returns to the
            // start of the line — plain \r alone can't shrink a line, only
            // overwrite its prefix.
            print!(
                "\rorangu-server: [slot {}] {}\x1b[K",
                guard.id(),
                partial.log_line()
            );
            std::io::stdout().flush().ok();
            last_report = Instant::now();
            reported = true;
        }
        if history.len() >= capacity {
            finish_reason = FinishReason::Length;
            break;
        }
        // When the sampler is greedy, let the model pick the
        // next token itself (a GPU-fused argmax, for backends that have
        // one) instead of always reading back the full `[n_vocab]` logits
        // vector just to immediately re-derive the same argmax on the
        // CPU. `recent_tokens` is trimmed to `repeat_last_n` here, not
        // inside the callee, matching `engine::sampling::
        // apply_repeat_penalty`'s own trim exactly.
        let repeat_last_n = sampler.repeat_last_n();
        let recent_start = history.len().saturating_sub(repeat_last_n);
        let start_pos = history.len() - 1;

        // The rows this step writes — its token's, and a draft's behind it
        // when it verifies — promised before it runs, while the sequence can
        // still wait for them or give its pages back: inside the forward
        // pass a page that is not there has nowhere to be reported.
        if spec_buf.is_empty()
            && let Some(occupant) = occupant.as_ref()
            && let Some((pool, index)) = paged_kv
        {
            let rows = 1 + drafter.as_ref().map_or(0, Drafter::max_tokens);
            let room = (start_pos + rows).min(capacity);
            let closed = || tx.is_closed();
            let found = room_for_step(
                cache
                    .as_ref()
                    .expect("cache is always Some between iterations"),
                occupant,
                room,
                slot_store,
                &closed,
                &mut || drop(batch_registration.take()),
            );
            match found {
                StepRoom::Ready => {}
                StepRoom::Gone => {
                    outcome.finish(
                        super::metrics::Outcome::Cancelled,
                        req.prompt_tokens.len(),
                        reused_len,
                        generated,
                    );
                    return Ok(());
                }
                StepRoom::Yield => {
                    drop(batch_registration.take());
                    log::info!(
                        "orangu-server: [kv] slot {} gives its {start_pos} positions back to \
                         the pool for an older request, and rebuilds them when there is room",
                        guard.id()
                    );
                    let rebuilt = rebuild_on_pool(
                        model,
                        pool,
                        index,
                        occupant,
                        cache
                            .take()
                            .expect("cache is always Some between iterations"),
                        &history[..start_pos],
                        room,
                        capacity,
                        guard.id(),
                        req.cache_prompt,
                        &closed,
                    );
                    match rebuilt {
                        Ok(Some(rebuilt)) => cache = Some(rebuilt),
                        Ok(None) => {
                            outcome.finish(
                                super::metrics::Outcome::Cancelled,
                                req.prompt_tokens.len(),
                                reused_len,
                                generated,
                            );
                            return Ok(());
                        }
                        Err(err) => {
                            let _ = tx.send(StreamEvent::Error(format!("{err:?}")));
                            return Ok(());
                        }
                    }
                    // A head that reads the served model's states from its
                    // last forward has none for the rebuilt positions.
                    if drafter.as_ref().is_some_and(Drafter::verifies_with_states) {
                        drafter = None;
                    }
                }
            }
            if batches && batch_registration.is_none() {
                batch_registration = decode_batcher.register();
            }
        }

        // Draft a continuation and verify it in one multi-position forward,
        // when there is nothing already verified to emit and verifying has
        // been paying (`engine::draft_control`); the model's own next token,
        // with any further accepted tokens queued in `spec_buf`.
        // An echo in progress is verified whatever the gate says, at the
        // width that has been yielding more tokens a millisecond.
        let echo = spec_buf.is_empty() && drafter.as_ref().is_some_and(|d| d.echoes(&history));
        let wide = echo
            && drafter.as_ref().is_some_and(Drafter::has_echo_draft)
            && draft_control.echo_wide();
        let speculated = if spec_buf.is_empty()
            && let Some(drafter) = drafter.as_mut()
            && (drafter.verifies_with_states()
                || draft_control.should_verify(
                    if batch_registration.is_some() {
                        decode_batcher.rows()
                    } else {
                        1
                    },
                    batch_registration.is_some() && model.batches_verify(),
                )
                || echo)
        {
            let started = Instant::now();
            match speculative_next(
                model,
                cache
                    .as_mut()
                    .expect("cache is always Some between iterations"),
                &mut sampler,
                &history,
                next,
                start_pos,
                capacity,
                drafter,
                guard.id(),
                drafter.draft_len(wide),
                &mut spec_buf,
                &mut spec_accepted,
                &mut spec_steps,
                batch_registration
                    .as_ref()
                    .filter(|_| model.batches_verify())
                    .map(|registration| (decode_batcher, registration)),
            ) {
                Ok(Some((t, yielded))) => {
                    if !drafter.verifies_with_states() || echo {
                        draft_control.verified(yielded, started.elapsed(), echo.then_some(wide));
                    }
                    Some(t)
                }
                Ok(None) => None,
                Err(err) => {
                    let _ = tx.send(StreamEvent::Error(format!("{err:?}")));
                    return Ok(());
                }
            }
        } else {
            None
        };

        next = if let Some(tok) = speculated {
            tok
        } else if let Some(tok) = spec_buf.pop_front() {
            // A token an earlier speculative step already verified and
            // committed to the KV cache — emit it without another forward.
            tok
        } else {
            let step_started = drafter.is_some().then(Instant::now);
            // The device-side argmax picks a token without ever consulting
            // the grammar, so a constrained request has to come back to the
            // CPU sampler for every step. It costs the fast path; the
            // alternative is a constraint that silently does not apply.
            // A sampled step whose sampler only looks at its top `k` can have
            // the device find those `k` (`top_k` below) and read back a few
            // dozen pairs instead of the whole vocabulary.
            let device_top_k = sampler.device_top_k();
            let greedy_sample = (!sampler.is_constrained()
                && (sampler.is_greedy() || device_top_k > 0))
                .then(|| GreedySampleParams {
                    recent_tokens: &history[recent_start..],
                    repeat_penalty: sampler.repeat_penalty(),
                    top_k: if sampler.is_greedy() { 0 } else { device_top_k },
                });
            // GPU submissions for this one decode step. `gemma.rs` had this
            // instrumentation privately; it belongs here, because the number it
            // reports is *the* difference between the two decode paths and
            // cannot be compared across architectures from inside one of them.
            // A decode step that is one submission lets concurrent requests
            // interleave on the GPU; one that is two hundred does not, which
            // was measured as a 2x aggregate-throughput ceiling. Costs a cached
            // env read and one atomic load when the flag is on, nothing when it is
            // off.
            let submissions_before = gpu_trace()
                .then(|| model.vulkan_backend())
                .flatten()
                .map(|v| v.submission_count());
            // Timed as one whole pass, here rather than inside any
            // architecture: this is the single call every model's decode step
            // goes through, so `forward` — the denominator the per-stage
            // breakdown is read against — is measured for all of them. See
            // `engine::decode_stages`.
            let outcome = crate::engine::decode_stages::pass(|| {
                decode_batcher.step(
                    batch_registration.as_ref(),
                    model,
                    cache
                        .as_mut()
                        .expect("cache is always Some between iterations"),
                    next,
                    start_pos,
                    greedy_sample,
                    guard.id(),
                )
            });
            if let Some(started) = step_started {
                draft_control.stepped(started.elapsed());
            }
            if let Some(before) = submissions_before
                && let Some(v) = model.vulkan_backend()
            {
                eprintln!(
                    "orangu-server: [gpu-trace] {} GPU submissions for this decode step (pos {start_pos})",
                    v.submission_count() - before
                );
            }
            // Every layer has run over this position now, so whichever page it
            // completed may be published. A no-op on all but one step in
            // `page_tokens`, and a lock plus a scan of the fill counts when it
            // is not.
            if paged_kv.is_some()
                && let Some(cache) = cache.as_mut()
            {
                cache.commit_pages();
            }
            match outcome {
                Ok(ForwardOutcome::Token(t)) => t,
                Ok(ForwardOutcome::Candidates(c)) => {
                    sampler.sample_from_candidates(c, history.len())
                }
                Ok(ForwardOutcome::Logits(l)) => {
                    let t0 = std::time::Instant::now();
                    let t = sampler.sample(&l, &history, history.len());
                    if std::env::var_os("ORANGU_CPU_TIMESTAMPS").is_some() {
                        eprintln!(
                            "orangu-server: [cpu-trace] pos {start_pos}: host sampler {:.3}ms over {} logits",
                            t0.elapsed().as_secs_f64() * 1000.0,
                            l.len()
                        );
                    }
                    t
                }
                Err(err) => {
                    let _ = tx.send(StreamEvent::Error(format!("{err:?}")));
                    return Ok(());
                }
            }
        };
    }
    drop(decoding);
    // Generation can stop part-way through a character — a token limit, a stop
    // token — and those bytes are still bytes the model produced. Sent through
    // the same filter every other token's text goes through, so a stream that
    // ends mid-header stays withheld.
    if let Some(text) = header.observe_text(decoder.flush())
        && !text.is_empty()
    {
        let _ = tx.send(if header.in_reasoning() {
            StreamEvent::Reasoning(text)
        } else {
            StreamEvent::Token(text)
        });
    }
    let generate_time = generate_start.elapsed();
    if spec_steps > 0 {
        // Draft acceptance: `spec_accepted` drafted tokens confirmed across
        // `spec_steps` forwards, i.e. this many extra tokens produced beyond the
        // one each forward always yields — the whole payoff of speculation.
        // The drafter is named because the two have completely different cost
        // profiles: prompt-lookup's misses are free, a draft model's are a
        // forward pass each, so the same acceptance figure is a win for one
        // and a loss for the other.
        eprintln!(
            "orangu-server: [speculative/{}] {spec_accepted} drafted tokens accepted over \
             {spec_steps} steps ({:.2} extra tokens/forward)",
            drafter.as_ref().map_or("none", Drafter::label),
            spec_accepted as f64 / spec_steps as f64
        );
    }

    // Offer this request's own final (full token sequence, resulting KV
    // cache) to the pool for a later request to reuse — win or not this
    // time, it's a candidate prefix for whatever comes next (most
    // obviously the same conversation's following turn, whose prompt will
    // be exactly `history` plus a short new suffix). The same completed
    // cache is also retained as this slot's durable snapshot (`slot_store`)
    // so a later `save` can persist it; when both features are on, the pool
    // gets a `duplicate()` and the slot keeps the original, since each needs
    // to own its copy.
    let mut workers = Vec::new();
    // `ORANGU_TRACE_TOKENS=1`: the exact token ids a request ran and emitted,
    // so a reply can be teacher-forced through another path without
    // re-tokenizing its text.
    if trace_tokens() {
        let (prompt, reply) = history.split_at(req.prompt_tokens.len().min(history.len()));
        log::info!(
            "orangu-server: [slot {}] tokens prompt={prompt:?} reply={reply:?}",
            guard.id()
        );
    }
    if let Some(mut final_cache) = cache.take() {
        workers = final_cache.request_finished();
        let history = std::mem::take(&mut history);
        if let Some((pool, index)) = paged_kv {
            // Every whole page of the sequence, the reply's too.
            publish_pages(model, pool, index, &final_cache, &history);
            // A request that asked for a slot keeps its conversation there for
            // a `save`, as a contiguous cache does. Only then: a kept paged
            // cache holds its pages, which the pool cannot reclaim until the
            // slot's next request replaces it.
            match slot_store {
                Some(store) if req.id_slot.is_some() && req.background => {
                    store.fill_if_empty(guard.id(), history, final_cache)
                }
                Some(store) if req.id_slot.is_some() => {
                    store.retain(guard.id(), history, final_cache)
                }
                _ => drop(final_cache),
            }
        } else {
            // A recurrent model keeps its checkpoint, not its end: the next
            // turn repeats the prompt up to the cache point and not what the
            // state ran after it.
            let mut history = history;
            if let Some((at, state)) = checkpoint.take()
                && final_cache.rewind_to_checkpoint(at, state)
            {
                history.truncate(at);
            } else if (prefix_cache.is_some() || slot_store.is_some())
                && let Some(vulkan) = model.vulkan_backend()
            {
                // What outlives the request is reused only as far as its host
                // rows reach; the reply's are on the device.
                final_cache.pull_decoded_rows(vulkan);
            }
            match (prefix_cache, slot_store) {
                (Some(pool), Some(store)) => {
                    if req.background {
                        store.fill_if_empty(guard.id(), history.clone(), final_cache.duplicate());
                    } else {
                        store.retain(guard.id(), history.clone(), final_cache.duplicate());
                    };
                    pool.store(history, final_cache);
                }
                (Some(pool), None) => pool.store(history, final_cache),
                (None, Some(store)) if req.background => {
                    store.fill_if_empty(guard.id(), history, final_cache)
                }
                (None, Some(store)) => store.retain(guard.id(), history, final_cache),
                (None, None) => {}
            }
        }
    }

    let stats = GenerateStats {
        prompt_tokens: req.prompt_tokens.len(),
        cached_tokens: reused_len,
        prompt_time,
        generated_tokens: generated,
        generate_time,
        workers,
    };
    // A prefix another request left in the cache, reused here: worth
    // prefilling again after a restart (`engine::warm_prefixes`).
    if reused_len >= super::warm_prefixes::MIN_TOKENS {
        super::warm_prefixes::record(&req.prompt_tokens[..reused_len]);
    }
    // The `\r` and the trailing `\x1b[K` (erase to end of line) only matter
    // if a live update above already moved the cursor onto this line, and
    // are left out otherwise — a log file gets a plain line, not a cursor
    // movement it can't act on.
    let (prefix, suffix) = if reported { ("\r", "\x1b[K") } else { ("", "") };
    // The serving role, named only when it is not the one the banner already
    // reported — a coordinator can hand this process a request belonging to
    // another of its profiles, and "did my `/auto_review` actually run in
    // review mode" is otherwise a question the log cannot answer.
    let served_as = match req.role {
        Some(request_role) if request_role != process_role => {
            format!(" as {}", request_role.label())
        }
        _ => String::new(),
    };
    log::info!(
        "{prefix}orangu-server: [slot {}{served_as}] {}{suffix}",
        guard.id(),
        stats.log_line()
    );
    outcome.finish(
        match finish_reason {
            FinishReason::Stop => super::metrics::Outcome::Stop,
            FinishReason::Length => super::metrics::Outcome::Length,
        },
        stats.prompt_tokens,
        stats.cached_tokens,
        stats.generated_tokens,
    );
    let _ = tx.send(StreamEvent::Done {
        stats,
        finish_reason,
    });
    Ok(())
}

/// The recipient text that marks a message as the model's own reasoning
/// rather than something addressed to the caller.
///
/// `muse-glimmer` writes `<|start|>assistant to=self<|message|>` for a
/// chain-of-thought message and ` to=user` for the answer; the gpt-oss
/// family spells the same distinction with a channel name. Matching on the
/// text the model writes, the way `Tokenizer::message_framing` matches on
/// the vocabulary's own token names — a format that doesn't use it simply
/// never produces a header containing it.
const REASONING_RECIPIENT: &str = "to=self";

/// How much text may be withheld as a message header before
/// [`MessageHeader`] concludes it is not looking at one. A recipient is a
/// role plus a name (` to=self`, ` to=weather.get_current`); this is far
/// above any of them and far below a reply worth losing.
const MAX_HEADER_LEN: usize = 128;

/// Splits a `<|start|>…<|message|>`-framed reply into its parts as it
/// streams: the message *headers*, which are framing and never shown, and
/// the message *bodies*, which are shown or not depending on who the model
/// addressed them to.
///
/// The markers themselves are CONTROL tokens and were always hidden. Two
/// things are new:
///
/// - **The header text is hidden.** `muse-glimmer` stops its generation
///   prompt at `<|start|>assistant` so the model can pick its own
///   recipient, and the ` to=self` / ` to=user` it writes there is framing,
///   not prose. Left visible, every reply from that model began
///   `" to=self"`.
/// - **A reasoning message is hidden when the role says so.** An assistant
///   turn in this format is *several* messages — reasoning addressed
///   `to=self`, then the answer addressed `to=user` — and which of them the
///   caller sees is the same question `Role::enable_thinking` already
///   answers for models that mark reasoning with `<think>`. A role that
///   suppresses reasoning gets the answer alone; every other role gets
///   both, separated by a blank line (they are different messages, and
///   running them together produced `…Final.Three primes larger than…`).
///
/// A third format frames the same distinction with a *channel*: gemma-4
/// writes `<|channel>thought\n`, its chain of thought, then `<channel|>`,
/// and may open a `tool_code` channel the same way. Both markers are already
/// hidden, so what was reaching the reader was the channel name and its body
/// run together with the answer — replies that began `"thought"` and ended
/// `"tool_code\nprint(create_directory(path='x'))"`. The name is framing and
/// is always dropped; the body is a reasoning message and follows the same
/// rule as the other two formats' — hidden when the role suppresses
/// reasoning, and otherwise separated from the answer rather than glued to
/// it.
///
/// A second format asks the same question with tokens instead of text, and
/// is answered here too: `inkling` opens each body with a marker naming its
/// *kind* (`<|content_thinking|>` for reasoning, `<|content_text|>` for the
/// answer) and writes no header at all. Same rule — a reasoning body is
/// hidden when the role says so, and two visible bodies are separated by a
/// blank line — read off `Tokenizer::content_kinds` rather than off a
/// recipient string.
///
/// A fourth format is the one nearly every reasoning model uses, and the
/// last to be handled here: a `<think>` … `</think>` pair (Qwen3.x,
/// DeepSeek-R1, QwQ, GLM). Both tokens are special and so were already
/// hidden, which is precisely what made this format the quiet one — with
/// the tags gone and nothing marking the span, a whole chain of thought
/// arrived as the answer, and a reply to "implement a doubly linked list"
/// was 8192 tokens of the model talking to itself, code drafts and all,
/// with the answer never reached. Same rule as the other three: hidden when
/// the role suppresses reasoning, and otherwise marked, so a caller can
/// present the thinking as thinking. One wrinkle the others don't have —
/// the *template* writes the opening tag as the tail of the generation
/// prompt, so the model emits only the close; see
/// [`MessageHeader::prompt_ends_in_think`].
///
/// Inert (`framing`, `kinds`, `channel` and `think` all `None`) for every
/// vocabulary with none of them, which is every other model this server
/// serves — those keep byte-for-byte the behavior they had. That is also
/// what `Default` gives, which is how the tests build one framing at a
/// time.
/// The name of the channel gemma-4 reasons in.
const REASONING_CHANNEL: &str = "thought";

#[derive(Default)]
struct MessageHeader {
    /// `(<|start|>, <|message|>)`, or `None` when this vocabulary has no
    /// such framing and nothing here applies.
    framing: Option<(u32, u32)>,
    /// The body-kind markers, for a vocabulary that types its bodies
    /// instead of naming a recipient. `None` for every other.
    kinds: Option<crate::engine::tokenizer::ContentKinds>,
    /// `(<|channel>, <channel|>)`, for a vocabulary that frames the model's
    /// side channels with them. `None` for every other.
    channel: Option<(u32, u32)>,
    /// `(<think>, </think>)`, for a vocabulary that marks its chain of
    /// thought with them. `None` for every other.
    think: Option<(u32, u32)>,
    /// Whether the stream is currently inside a channel *name* — between
    /// `<|channel>` and the newline that ends it.
    naming_channel: bool,
    /// A separator owed to the next visible text because a message boundary
    /// was crossed without a marker to hang it on. A channel closes with
    /// `<channel|>` and the answer simply resumes, so unlike the other two
    /// formats there is no opening marker later to emit it at; holding it
    /// until there is text to put it in front of also keeps it off the end
    /// of a reply that stops right after the channel.
    owed_separator: bool,
    /// Whether the stream is currently inside a header.
    inside: bool,
    /// The current header's text so far, accumulated to be read once at
    /// `<|message|>` and then discarded. Bounded by the header's own length
    /// (a recipient name), not by the reply's.
    recipient: String,
    /// The current channel's name so far, accumulated only so that a run of
    /// text long past any name's length can be released as content instead
    /// of swallowed. Bounded the same way [`Self::recipient`] is.
    channel_name: String,
    /// Whether the message body now streaming is suppressed.
    hidden_body: bool,
    /// Whether the body now streaming is the model's reasoning — what
    /// decides whether its text leaves as `StreamEvent::Reasoning` or as
    /// `StreamEvent::Token`. Distinct from [`Self::hidden_body`]: a
    /// suppressing role hides reasoning, every other role marks it.
    reasoning_body: bool,
    /// Whether the body just opened is still owed a trim: leading
    /// whitespace to drop before its first real character.
    ///
    /// A think-tagging template writes its markers with the blank lines
    /// that used to separate them from the prose — Qwen3.x's generation
    /// prompt ends `<think>\n` and its replies close with `</think>\n\n`.
    /// The markers are hidden and the newlines are not, so before this
    /// every such answer arrived starting with a blank line and every
    /// chain of thought with a newline. Harmless in a markdown renderer,
    /// which ignores leading blank lines, and not harmless at all in
    /// `content` / `reasoning_content`, which callers compare, hash and
    /// concatenate.
    ///
    /// Only the think framing sets it. The other three already emit their
    /// own separators and have no stray whitespace to lose.
    trim_leading: bool,
    /// Whether any *visible* body has already been emitted this turn —
    /// what decides if the next one needs a separator in front of it.
    emitted_body: bool,
    suppress_reasoning: bool,
}

impl MessageHeader {
    fn for_prompt(tokenizer: &Tokenizer, prompt_tokens: &[u32], role: crate::config::Role) -> Self {
        Self {
            framing: tokenizer.message_framing(),
            kinds: tokenizer.content_kinds(),
            channel: tokenizer.channel_framing(),
            think: tokenizer.think_framing(),
            naming_channel: false,
            owed_separator: false,
            inside: Self::prompt_ends_in_header(tokenizer, prompt_tokens),
            recipient: String::new(),
            channel_name: String::new(),
            hidden_body: Self::prompt_ends_in_think(tokenizer, prompt_tokens)
                && role.suppresses_reasoning(),
            reasoning_body: Self::prompt_ends_in_think(tokenizer, prompt_tokens),
            trim_leading: Self::prompt_ends_in_think(tokenizer, prompt_tokens),
            emitted_body: false,
            suppress_reasoning: role.suppresses_reasoning(),
        }
    }

    /// Whether the body now streaming is the model's reasoning.
    fn in_reasoning(&self) -> bool {
        self.reasoning_body
    }

    /// The token that ends reasoning in this format: `</think>`, or
    /// `<channel|>` for a channel-framed one. `None` for a format whose
    /// reasoning ends some other way.
    fn reasoning_close(&self) -> Option<u32> {
        self.think
            .map(|(_, close)| close)
            .or(self.channel.map(|(_, close)| close))
    }

    /// Whether generation resumes *inside* a `<think>` block.
    ///
    /// Read off the prompt for the same reason [`Self::
    /// prompt_ends_in_header`] is, and needed for the same reason: a
    /// `<think>`-marking template writes the opening tag itself as the last
    /// thing in the generation prompt (Qwen3.x ends `<|im_start|>assistant\n
    /// <think>\n`), so the model emits only the *closing* one. Seeded
    /// `false`, every such reply would be taken for an answer up to the
    /// `</think>` and for reasoning after it — exactly backwards.
    ///
    /// A prompt whose last think marker is the opening one resumes inside
    /// reasoning. One that closed its last block, and one with no think
    /// markers at all — which includes every raw-completion prompt and the
    /// empty block `--review` prefills — resumes in an answer.
    fn prompt_ends_in_think(tokenizer: &Tokenizer, prompt_tokens: &[u32]) -> bool {
        tokenizer.think_framing().is_some_and(|(open, close)| {
            prompt_tokens
                .iter()
                .rev()
                .find_map(|&t| (t == open || t == close).then_some(t == open))
                .unwrap_or(false)
        })
    }

    /// Whether generation resumes *inside* a header.
    ///
    /// Read off the prompt rather than taken as a flag from the caller: the
    /// prompt is the ground truth, every caller renders one, and a flag is
    /// one more thing three HTTP paths would have to keep consistent.
    ///
    /// A prompt whose last marker is `<|start|>` (the ordinary generation
    /// prompt, `…<|eot|><|start|>assistant`) resumes inside a header. One
    /// whose last marker is `<|message|>` — a prefilled partial reply —
    /// resumes inside a body. One with neither, which includes every
    /// raw-completion prompt, likewise resumes in a body.
    fn prompt_ends_in_header(tokenizer: &Tokenizer, prompt_tokens: &[u32]) -> bool {
        tokenizer.message_framing().is_some_and(|(start, message)| {
            prompt_tokens
                .iter()
                .rev()
                .find_map(|&t| (t == start || t == message).then_some(t == start))
                .unwrap_or(false)
        })
    }

    /// Feed every *special* token the model generates through this.
    /// Returns text to emit in the marker's place — a separator between two
    /// visible messages, and otherwise nothing.
    fn observe_marker(&mut self, id: u32) -> Option<&'static str> {
        // A channel opens and closes messages the way the other two formats'
        // markers do, so it is settled here and returns rather than falling
        // through to them — a vocabulary has at most one of the three.
        if let Some((open, close)) = self.channel {
            // A channel is hidden from a role that suppresses reasoning.
            // Whether its body *is* reasoning — gemma-4's `thought` — is
            // known once its name has been read (see `observe_text`).
            if id == open {
                self.naming_channel = true;
                self.channel_name.clear();
                self.hidden_body = self.suppress_reasoning;
                return None;
            }
            if id == close {
                // Whatever follows is the answer again: a new message, and
                // one no marker will announce. Shown, reasoning leaves as
                // events of its own and nothing joins it to the answer; any
                // other body, or one hidden, would run into what follows.
                self.owed_separator |=
                    self.emitted_body && (self.suppress_reasoning || !self.reasoning_body);
                self.naming_channel = false;
                self.hidden_body = false;
                self.reasoning_body = false;
                return None;
            }
        }
        // A `<think>` block is the same message boundary a channel is, and
        // is settled the same way. The one difference is that a template
        // opens the block itself, so the opening marker is often never
        // generated at all — `prompt_ends_in_think` has already put this
        // stream inside one by then, and only the close arrives here.
        if let Some((open, close)) = self.think {
            // Neither edge owes a separator, unlike a channel's. A
            // separator exists to keep two visible bodies from running
            // together in one stream; these two leave as *different*
            // events, so nothing downstream concatenates them, and the
            // "\n\n" would show up as a blank first line in whichever pane
            // it landed in.
            if id == open {
                self.reasoning_body = true;
                self.hidden_body = self.suppress_reasoning;
                self.trim_leading = true;
                return None;
            }
            if id == close {
                // Whatever follows is the answer: a new message, and one no
                // marker will announce.
                self.reasoning_body = false;
                self.hidden_body = false;
                self.trim_leading = true;
                return None;
            }
        }
        // A body-kind marker settles the same question a header's recipient
        // does, and settles it on its own — a format that has these writes
        // no header, so this is checked first and returns rather than
        // falling through to the header machinery.
        if let Some(kinds) = &self.kinds
            && (id == kinds.reasoning || kinds.other.contains(&id))
        {
            self.reasoning_body = id == kinds.reasoning;
            self.hidden_body = self.suppress_reasoning && id == kinds.reasoning;
            let separator = (!self.hidden_body && self.emitted_body).then_some("\n\n");
            self.emitted_body |= !self.hidden_body;
            return separator;
        }
        let (start, message) = self.framing?;
        if id == start {
            self.inside = true;
            self.recipient.clear();
            return None;
        }
        if id != message {
            return None;
        }
        // The header just ended: its text says who this message is for.
        self.inside = false;
        self.reasoning_body = self
            .recipient
            .replace(' ', "")
            .contains(REASONING_RECIPIENT);
        self.hidden_body = self.suppress_reasoning && self.reasoning_body;
        let separator = (!self.hidden_body && self.emitted_body).then_some("\n\n");
        self.emitted_body |= !self.hidden_body;
        separator
    }

    /// Feed every *ordinary* (non-special) token's text through this.
    /// Returns what to emit: nothing while inside a header, and nothing
    /// inside a body the role suppresses.
    fn observe_text(&mut self, mut text: String) -> Option<String> {
        if self.naming_channel {
            // The name runs to the first newline. One token can carry both
            // it and the start of the body, so the tail is kept rather than
            // dropped with the name.
            match text.find('\n') {
                Some(at) => {
                    self.naming_channel = false;
                    self.channel_name.push_str(&text[..at]);
                    self.reasoning_body = self.channel_name.trim() == REASONING_CHANNEL;
                    // Another body is a message beside the answer, kept
                    // apart from it.
                    self.owed_separator |=
                        !self.reasoning_body && !self.hidden_body && self.emitted_body;
                    text = text.split_off(at + 1);
                }
                None => {
                    self.channel_name.push_str(&text);
                    // Same bound, and for the same reason, as a recipient's:
                    // withholding is what has to stay bounded. A name this
                    // long is not a name, so it is released as content.
                    if self.channel_name.len() <= MAX_HEADER_LEN {
                        return None;
                    }
                    self.naming_channel = false;
                    self.hidden_body = false;
                    self.reasoning_body = false;
                    text = std::mem::take(&mut self.channel_name);
                }
            }
        }
        if !self.inside {
            if self.hidden_body || text.is_empty() {
                return None;
            }
            if self.trim_leading {
                // One token can be all whitespace and the next carry the
                // first real character, so the flag stays set until
                // something survives the trim.
                text = text.trim_start().to_string();
                if text.is_empty() {
                    return None;
                }
                self.trim_leading = false;
            }
            self.emitted_body = true;
            if std::mem::take(&mut self.owed_separator) {
                text.insert_str(0, "\n\n");
            }
            return Some(text);
        }
        self.recipient.push_str(&text);
        if self.recipient.len() <= MAX_HEADER_LEN {
            return None;
        }
        // Long past any recipient's length and still no `<|message|>`: this
        // is not a header, so whatever was withheld is content and is
        // released now. Withholding is the one failure worth bounding —
        // a stray recipient in the reply is cosmetic, a reply swallowed
        // whole is not, and only the second can happen without a bound.
        self.inside = false;
        self.hidden_body = false;
        self.reasoning_body = false;
        self.emitted_body = true;
        Some(std::mem::take(&mut self.recipient))
    }
}

/// Prompt-lookup speculative-decode settings, read once at the start of a
/// request. `None` (the default) leaves decoding exactly as it was. Setting
/// `ORANGU_SPECULATIVE` turns it on; `ORANGU_SPEC_NGRAM` (default 2) is how
/// many trailing tokens must match a earlier spot in the context to trigger a
/// draft, and `ORANGU_SPEC_DRAFT` (default 4) how many tokens to draft from
/// there. See `Self::speculative_next`.
/// How many prompt tokens go into one forward pass — `ORANGU_PREFILL_BATCH`,
/// default [`PREFILL_BATCH_DEFAULT`]. `0` means no limit: the whole prompt in
/// one pass, which is what this code did unconditionally before.
fn prefill_batch() -> usize {
    prefill_batch_override().unwrap_or_else(|| PREFILL_BATCH_DEFAULT.max(chunk_ceiling()))
}

/// The widest chunk this model wants, recorded once at startup by
/// [`set_chunk_ceiling`] — `0` unless the model has a reason to be prefilled
/// wider than [`PREFILL_BATCH_DEFAULT`].
///
/// A mixture whose routed experts run on the card has one. Every pass
/// streams the layer's whole expert stack across the bus, which is a cost
/// per *pass* and not per token, so a prompt taken in five passes pays it
/// five times — the same shape as a streamed model's re-reads, one level
/// down the memory hierarchy. It is answered from the file, before the
/// layers are placed, because the KV estimate and the sliding-window ring
/// are both sized from this width.
///
/// The larger of its callers wins, as with [`CHUNK_FLOOR`], and an
/// `ORANGU_PREFILL_BATCH` written by hand overrules it entirely.
static CHUNK_CEILING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// See [`CHUNK_CEILING`].
pub fn set_chunk_ceiling(ceiling: usize) {
    CHUNK_CEILING.fetch_max(ceiling, std::sync::atomic::Ordering::Relaxed);
}

fn chunk_ceiling() -> usize {
    CHUNK_CEILING.load(std::sync::atomic::Ordering::Relaxed)
}

/// The widest prefill chunk the sizer will choose, as configured — what a
/// cache sized for "the window plus one chunk" (`LayerCache::set_mirror_ring`)
/// is sized against. A wider chunk asked for by another route is bounded
/// back to the cache's own limit (`KvCache::max_positions_per_call`).
pub(crate) fn prefill_chunk_ceiling() -> usize {
    prefill_batch()
}

/// `ORANGU_PREFILL_BATCH` if it was set, `None` if it was not — the
/// distinction [`flat_width`] needs and [`prefill_batch`] throws away. An
/// operator who wrote `512` gets 512 everywhere; one who wrote nothing gets a
/// width chosen for the regime.
fn prefill_batch_override() -> Option<usize> {
    static BATCH: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *BATCH.get_or_init(|| {
        std::env::var("ORANGU_PREFILL_BATCH")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
    })
}

/// How much of the model must be in RAM before an extra prefill pass is
/// cheap.
///
/// Not a tuning knob so much as a statement about which of two measured
/// regimes a run is in. Above it an extra pass costs locality; below it an
/// extra pass costs a re-read of everything missing, which is the whole model
/// when residency is poor.
const RESIDENT_ENOUGH: f64 = 0.95;

/// The width one flat-policy prefill should use.
///
/// **The two regimes disagree, and both were measured.** With the model
/// resident, a narrow chunk is faster *and* smaller — a smaller working set
/// pages less — and the ordering is monotone in width. With the model
/// streamed from disk, every extra pass re-reads what is not resident, so the
/// ordering reverses and one pass wins by a factor rather than a percentage.
/// No single constant is right for both, so this asks which regime it is in.
///
/// Two things keep the question cheap. A prompt that fits the narrow default
/// is one pass at any width, so the answer cannot matter and is not asked
/// for — and that is every short prompt, while the probe is `mincore` over
/// the whole model. And residency that cannot be established keeps **today's**
/// width, on the same principle as [`Backend::has_submission_timeout`]'s
/// default: a component that cannot answer must not be read as having
/// answered.
///
/// [`Backend::has_submission_timeout`]: crate::engine::backend::Backend::has_submission_timeout
fn flat_width(prompt_tokens: usize) -> usize {
    let configured = prefill_batch_override();
    // Short-circuited before the probe, not inside it: `resident_fraction` is
    // `mincore` over every shard, and a prompt this size does not care.
    let resident = (configured.is_none() && prompt_tokens > PREFILL_BATCH_DEFAULT)
        .then(resident_fraction)
        .flatten();
    flat_width_for(configured, prompt_tokens, resident)
}

/// The mapping alone, so each direction can be tested without a process-wide
/// `OnceLock` a test cannot then undo — the same shape as [`policy_for`], and
/// for the same reason: getting this backwards is a factor, not a percentage.
fn flat_width_for(
    configured: Option<usize>,
    prompt_tokens: usize,
    resident_fraction: Option<f64>,
) -> usize {
    match (configured, resident_fraction) {
        // An operator who named a width gets it, in either regime.
        (Some(width), _) => width,
        // One pass at any width: the regime cannot change the pass count.
        _ if prompt_tokens <= PREFILL_BATCH_DEFAULT => PREFILL_BATCH_DEFAULT,
        // Streamed: every extra pass re-reads what is not resident.
        (None, Some(fraction)) if fraction < RESIDENT_ENOUGH => 0,
        // Resident, or residency unknowable — keep today's width.
        (None, _) => PREFILL_BATCH_DEFAULT,
    }
}

/// What fraction of the model's bytes are in RAM, or `None` if that cannot be
/// established for every shard — a partial answer would understate residency
/// and read as a colder run than happened.
fn resident_fraction() -> Option<f64> {
    let shards = crate::engine::page_cache::residency();
    let (bytes, resident) = crate::engine::page_cache::residency_totals(&shards);
    if bytes == 0 {
        return None;
    }
    resident.map(|r| r as f64 / bytes as f64)
}

/// The default prompt-chunk size.
///
/// A forward pass's work and scratch both scale with the tokens in it, and
/// handing the model a whole long prompt at once therefore sizes them by
/// however much text the caller happened to send. On a GPU that is already
/// short of memory, that is a promise nothing can keep — and the way it
/// breaks is not a clean allocation failure but a **device reset**: under
/// paging pressure a single large submission stops finishing inside
/// `amdgpu`'s ~10s ring timeout, the kernel resets the ring and names this
/// process as the guilty context (`ring gfx_0.0.0 timeout … Ring gfx_0.0.0
/// reset succeeded`), and every buffer on the lost device dies with it (see
/// `crate::device_lost`).
///
/// Measured on a 4 GiB RX 5500M holding 2.5 GiB of weights, with a
/// 17.5k-token prompt — the size an `orangu` TUI turn reaches once it
/// carries a workspace diff:
///
/// | chunk | outcome |
/// | :-- | :-- |
/// | whole prompt at once | device lost after 21s |
/// | 2048 | device lost after 3m54s |
/// | 512 | **completed in 4m13s** |
///
/// Peak VRAM was the same (3.67 GiB of 4.08) in every case, which is the
/// point: what changes is how long any *one* submission holds the ring
/// while the driver pages, not how much is resident. So the chunk is a
/// timeout budget, not a memory budget, and 512 is the value that kept
/// every submission comfortably inside it.
///
/// It is not a throughput tax either — at an 8k-token prompt on the same
/// card, 512 measured *faster* than no chunking at all (115.4 vs 105.5
/// tok/s prefill), since a smaller working set pages less. The matmuls were
/// already striped at
/// `engine::backend::vulkan`'s own per-phase submission cap regardless of
/// this, so what a bigger chunk buys is fewer attention and PLE calls, not
/// bigger GEMMs.
///
/// Prompts shorter than this are unaffected: one chunk *is* the old path.
/// `ORANGU_PREFILL_BATCH=0` restores it outright, and a machine with VRAM
/// to spare can raise it — nothing here is specific to one card beyond the
/// numbers that set it.
const PREFILL_BATCH_DEFAULT: usize = 512;

/// Runs `tokens` through the model as consecutive [`prefill_batch`]-sized
/// forward passes, returning the last one's logits — the only ones a caller
/// wants, since sampling continues from the end of the prompt.
///
/// Each chunk is fed at the position the previous chunk ended at, so the KV
/// cache accumulates exactly as it would from one pass over the whole
/// prompt: this is the same `(cache, tokens, start_pos)` shape the prefix
/// cache already uses to resume a prompt it has partly seen, not a new
/// contract with the architectures.
/// How a prefill splits a prompt across forward passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChunkPolicy {
    /// Size each chunk from the last one's measured rate, aiming at
    /// [`prefill_chunk_budget`]. For a backend whose driver resets a device
    /// that takes too long over one submission — see
    /// [`Backend::has_submission_timeout`].
    ///
    /// [`Backend::has_submission_timeout`]: crate::engine::backend::Backend::has_submission_timeout
    Adaptive,
    /// Every chunk the configured width. For a backend with no submission
    /// timeout, where the wall clock carries no information the sizer can
    /// use: a streamed model's per-pass cost is fixed rather than per-token,
    /// so adapting to it shrinks the chunk and multiplies the reads. The
    /// width still bounds a pass's scratch; nothing infers a rate.
    Flat,
}

/// How one prefill splits: how wide a chunk may be, and whether that width is
/// a starting point or the whole rule.
///
/// The two travel together because neither means anything alone — a width with
/// no policy does not say whether it is a ceiling or a probe, and a policy with
/// no width has nothing to bound a pass's scratch by.
#[derive(Clone, Copy, Debug)]
struct Chunking {
    /// Tokens per forward pass. `0` opts out of chunking entirely — see
    /// [`prefill_batch`].
    width: usize,
    policy: ChunkPolicy,
}

impl Chunking {
    /// How this process should split a prompt of `prompt_tokens`.
    ///
    /// The width is regime-chosen only under [`ChunkPolicy::Flat`]. Under
    /// `Adaptive` it stays a ceiling the sizer works below, and widening it
    /// would hand the driver exactly the long submission the chunker exists
    /// to prevent.
    fn for_prompt(prompt_tokens: usize) -> Self {
        let policy = chunk_policy();
        let width = match policy {
            ChunkPolicy::Adaptive => prefill_batch(),
            ChunkPolicy::Flat => flat_width(prompt_tokens),
        };
        Self { width, policy }
    }
}

/// The longest KV context one **contiguous** request may hold on this device,
/// recorded once at startup by [`set_kv_device_token_ceiling`]. `None` where
/// there is no device to run out of, or where its capacity is unknown — which
/// is not the same as zero, and must not be read as a refusal.
///
/// A request over this does not fail cleanly if it is allowed to run: the
/// mirror grows through prefill until a submission stops finishing inside the
/// driver's reset timeout, the driver resets the device and names this process
/// as the guilty context, and every buffer on it dies — `exit 75`, the whole
/// server, for one oversized request. An `/auto_review` prompt carrying a few
/// thousand lines of source reaches that size wherever headroom is modest,
/// which is why this is enforced rather than documented.
///
/// So the size is refused where a caller can still act on the answer. Paged
/// requests are not subject to it: their rows live in the pool's pre-allocated
/// device pages, which is a fixed footprint no request can grow.
static KV_DEVICE_TOKENS: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();

/// Records the ceiling. Called once, from `main`, after the device, the model
/// and the paged pool are all known — the pool's own device pages come out of
/// the same headroom, so this cannot be answered before it is built.
pub fn set_kv_device_token_ceiling(tokens: Option<usize>) {
    let _ = KV_DEVICE_TOKENS.set(tokens);
}

pub(crate) fn kv_device_token_ceiling() -> Option<usize> {
    *KV_DEVICE_TOKENS.get().unwrap_or(&None)
}

/// Whether a prompt of this many tokens leaves room for at least one generated
/// token under `ceiling`. The one test both the HTTP layer and the engine ask,
/// so a request the one lets through is never one the other refuses.
pub(crate) fn prompt_fits_device(prompt_tokens: usize, ceiling: usize) -> bool {
    prompt_tokens < ceiling
}

#[cfg(test)]
mod device_ceiling_tests {
    use super::prompt_fits_device;

    /// A prompt needs the ceiling to hold it *and* one generated token; what
    /// the client's `max_tokens` asked for beyond that is a cap, not a
    /// requirement, and never a reason to refuse.
    #[test]
    fn the_prompt_alone_decides_whether_a_request_fits() {
        assert!(prompt_fits_device(0, 1));
        assert!(prompt_fits_device(16383, 16384));
        assert!(!prompt_fits_device(16384, 16384));
        assert!(!prompt_fits_device(usize::MAX, usize::MAX));
    }
}

/// Whether the backend this process selected can lose its device to a
/// submission timeout, recorded once at startup by [`set_chunk_policy`].
///
/// A process-wide fact rather than a parameter because the prefill call site
/// holds a `&dyn ModelForward` and no backend handle, and threading one down
/// would touch every architecture — the thing this whole design is trying not
/// to do. Unset means [`ChunkPolicy::Adaptive`]: a test, a benchmark harness
/// or any caller that never registered gets today's behaviour.
static CHUNK_POLICY: std::sync::OnceLock<ChunkPolicy> = std::sync::OnceLock::new();

/// The narrowest chunk the sizer may choose, recorded once at startup by
/// [`set_chunk_floor`] — `0` (the constants below) unless the model has a
/// reason to never be prefilled narrower.
///
/// A model split onto the host has one: a host layer's projections stream
/// to the card from `ORANGU_HOST_STREAM_TOKENS` (32) tokens up and run on
/// the CPU below, ten times slower per token. The sizer prices a chunk by
/// the last one's rate, so a 16-token probe measured on the CPU says a
/// 512-token chunk would take minutes, the next chunk stays under 32, and
/// every chunk after it too — measured, a 1120-token prompt on a 31B model
/// went in 47 chunks of ~24 tokens at 5.6 tok/s where 512-token chunks
/// run at 17. With the floor at the streaming width the probe itself is
/// measured on the fast path.
///
/// A mixture whose routed experts run on the card has the same shape: below
/// the grouped GEMM's threshold the chunk takes the host's per-expert path,
/// which is slower, which sizes the next chunk narrower still.
///
/// Two callers can each have a reason, so the larger of them wins — both are
/// "never narrower than", and honouring only the first one recorded would
/// leave a split mixture prefilled below one of its two floors.
static CHUNK_FLOOR: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// See [`CHUNK_FLOOR`].
pub fn set_chunk_floor(floor: usize) {
    CHUNK_FLOOR.fetch_max(floor, std::sync::atomic::Ordering::Relaxed);
}

fn chunk_floor() -> usize {
    chunk_floor_override().unwrap_or_else(|| CHUNK_FLOOR.load(std::sync::atomic::Ordering::Relaxed))
}

/// `ORANGU_PREFILL_CHUNK_FLOOR`, the override that puts both settings of the
/// floor on one binary — `0` is "no floor", which is what this did before a
/// model could ask for one.
///
/// A floor is chosen by what the model is, so without this the two arms of
/// the A/B would have to be two builds, and two builds differ by more than
/// the thing under test.
fn chunk_floor_override() -> Option<usize> {
    static FLOOR: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *FLOOR.get_or_init(|| {
        std::env::var("ORANGU_PREFILL_CHUNK_FLOOR")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
    })
}

/// Records what the selected backend answered for
/// [`Backend::has_submission_timeout`]. Called once, from `main`, right after
/// the backend is chosen.
///
/// [`Backend::has_submission_timeout`]: crate::engine::backend::Backend::has_submission_timeout
pub fn set_chunk_policy(has_submission_timeout: bool) {
    let _ =
        CHUNK_POLICY.set(env_chunk_policy().unwrap_or_else(|| policy_for(has_submission_timeout)));
}

/// `ORANGU_CHUNK_POLICY=adaptive|flat`, the override that lets a sweep put
/// both policies on one backend and compare them.
///
/// Without it the policy is a function of the hardware, so the two arms of the
/// A/B would have to be two machines — and a cross-machine A/B is not a
/// measurement. Anything unrecognised, including an empty value, is ignored
/// rather than treated as one of the two: a typo in a harness must not
/// silently select an arm.
fn env_chunk_policy() -> Option<ChunkPolicy> {
    match std::env::var("ORANGU_CHUNK_POLICY")
        .ok()?
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "adaptive" => Some(ChunkPolicy::Adaptive),
        "flat" => Some(ChunkPolicy::Flat),
        _ => None,
    }
}

/// The mapping alone, so the direction can be tested without writing the
/// process-wide `OnceLock` a test cannot then undo. Getting it backwards
/// costs a device.
fn policy_for(has_submission_timeout: bool) -> ChunkPolicy {
    if has_submission_timeout {
        ChunkPolicy::Adaptive
    } else {
        ChunkPolicy::Flat
    }
}

/// The registered policy, defaulting to the safe one.
fn chunk_policy() -> ChunkPolicy {
    *CHUNK_POLICY.get().unwrap_or(&ChunkPolicy::Adaptive)
}

#[allow(clippy::too_many_arguments)]
/// A prefill stopped between chunks because its client went away.
#[derive(Debug)]
struct PrefillCancelled;

impl std::fmt::Display for PrefillCancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the client went away during the prompt")
    }
}

impl std::error::Error for PrefillCancelled {}

fn prefill(
    model: &dyn ModelForward,
    cache: &mut KvCache,
    tokens: &[u32],
    start_pos: usize,
    slot_id: usize,
    drafter: Option<&mut Drafter<'_>>,
    on_chunk: &mut dyn FnMut(usize) -> Result<()>,
) -> Result<Vec<f32>> {
    // Priced once and carried forward: see [`CHUNK_COST`].
    let mut cost = load_chunk_cost();
    let out = prefill_in_chunks(
        &mut cost,
        model,
        cache,
        tokens,
        start_pos,
        slot_id,
        Chunking::for_prompt(tokens.len()),
        drafter,
        on_chunk,
    );
    // Written back on the error path too: a chunk that ran is a chunk that
    // was measured, and what it cost is true whether or not a later one
    // failed.
    store_chunk_cost(cost);
    // What the prefill borrowed from the card goes back before decode.
    if let Some(vulkan) = model.vulkan_backend() {
        vulkan.prompt_prefilled();
    }
    out
}

/// What prefill has learned about the cost of one submission here, carried
/// across requests.
///
/// The curve belongs to this machine and this model, not to one request, so
/// re-deriving it per request throws away an answer already paid for — and
/// that is what made every prompt open with a probe and climb its way back to
/// a workable width. One `orangu-server` process serves one model, so one
/// estimate describes every prefill it will ever run.
static CHUNK_COST: Mutex<ChunkCost> = Mutex::new(ChunkCost::new());

/// The carried estimate, or a fresh one if another thread poisoned the lock.
///
/// A poisoned lock costs a ramp, not a wrong answer: the worst an unusable
/// estimate can do is start the next prompt at the probe, which is where every
/// prompt started before this existed.
fn load_chunk_cost() -> ChunkCost {
    if chunk_cost_fit_disabled() {
        return ChunkCost::proportional_only();
    }
    *CHUNK_COST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// Whether `ORANGU_NO_CHUNK_COST_FIT` asked for the single-point sizer back.
///
/// The control arm for the two-point fit, so the change can be measured
/// against what it replaced in one `orangu-bench --sweep` rather than across
/// two sessions, where this machine's drift is larger than the effect.
fn chunk_cost_fit_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| crate::engine::env::flag_on("ORANGU_NO_CHUNK_COST_FIT"))
}

fn store_chunk_cost(cost: ChunkCost) {
    *CHUNK_COST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = cost;
}

/// [`prefill`] with the chunk size passed in rather than read from the
/// environment, so the splitting itself is testable — the property that
/// matters (every token fed exactly once, in order, at the right position)
/// is the one that silently corrupts a KV cache when it's wrong.
#[allow(clippy::too_many_arguments)]
fn prefill_in_chunks(
    cost: &mut ChunkCost,
    model: &dyn ModelForward,
    cache: &mut KvCache,
    tokens: &[u32],
    start_pos: usize,
    slot_id: usize,
    chunking: Chunking,
    mut drafter: Option<&mut Drafter<'_>>,
    on_chunk: &mut dyn FnMut(usize) -> Result<()>,
) -> Result<Vec<f32>> {
    // One extra decoder block over the prompt, and only for the drafter that
    // reads it. Everything else takes the forward it always took: exporting
    // a state row per position costs a `hc * n_embd` buffer per chunk, and
    // there is no reason to build one nobody will read.
    let mut chunk = |model: &dyn ModelForward,
                     cache: &mut KvCache,
                     part: &[u32],
                     pos: usize,
                     last: bool|
     -> Result<Vec<f32>> {
        let Some(drafter) = drafter.as_deref_mut().filter(|d| d.wants_states()) else {
            // A chunk before the last produces logits nobody reads; the
            // architecture is told so and may skip the work that makes them.
            if !last {
                crate::engine::decode_stages::pass_of(part.len(), || {
                    model.forward_no_logits(cache, part, pos, slot_id)
                })?;
                return Ok(Vec::new());
            }
            return crate::engine::decode_stages::pass_of(part.len(), || {
                model.forward(cache, part, pos, slot_id)
            });
        };
        let (logits, states) = crate::engine::decode_stages::pass_of(part.len(), || {
            model.forward_with_states(cache, part, pos, slot_id)
        })?;
        drafter.observe_states(pos, &states);
        // The head runs over the chunk that was just committed, so the
        // states it needs never have to outlive one chunk.
        drafter.catch_up(part, pos)?;
        Ok(logits)
    };
    run_in_chunks(
        cost,
        cache,
        tokens,
        start_pos,
        chunking,
        &mut |cache: &mut KvCache, part: &[u32], pos: usize, last: bool| {
            chunk(model, cache, part, pos, last)
        },
        on_chunk,
    )
}

/// Every token's final hidden state for an embeddings request, the input
/// run the way a prompt runs: in chunks [`prefill`]'s sizer picks, each at
/// the position the one before it ended, with the cost estimate it carries
/// across requests. That keeps a long input's submissions under the
/// driver's timeout. A model whose tokens must all see the whole input
/// (bidirectional attention) cannot be split this way and runs in one pass.
pub fn embedding_hidden_states(model: &dyn ModelForward, tokens: &[u32]) -> Result<Vec<f32>> {
    if !model.hidden_states_are_causal() {
        return model.forward_hidden_states(tokens);
    }
    let mut cache = model.new_kv_cache(tokens.len().max(1));
    let mut hidden = Vec::with_capacity(tokens.len() * model.config().n_embd);
    let mut cost = load_chunk_cost();
    let out = run_in_chunks(
        &mut cost,
        &mut cache,
        tokens,
        0,
        Chunking::for_prompt(tokens.len()),
        &mut |cache: &mut KvCache, part: &[u32], pos: usize, _last: bool| {
            hidden.extend(crate::engine::decode_stages::pass_of(part.len(), || {
                model.forward_hidden_states_at(cache, part, pos)
            })?);
            Ok(Vec::new())
        },
        &mut |_| Ok(()),
    );
    store_chunk_cost(cost);
    if let Some(vulkan) = model.vulkan_backend() {
        vulkan.prompt_prefilled();
    }
    out.map(|_| hidden)
}

/// One chunk's pass for [`run_in_chunks`]: `part` at `pos` into the cache,
/// `last` for the final chunk.
type ChunkPass<'a> = dyn FnMut(&mut KvCache, &[u32], usize, bool) -> Result<Vec<f32>> + 'a;

/// The chunk loop itself: how wide each chunk is and when it runs, with
/// `run` doing one chunk's pass (`part` at `pos`, `last` for the final one)
/// and returning what the caller keeps from the last. Shared by a prompt's
/// prefill and an embeddings pass, so both are sized by the same budget.
#[allow(clippy::too_many_arguments)]
fn run_in_chunks(
    cost: &mut ChunkCost,
    cache: &mut KvCache,
    tokens: &[u32],
    start_pos: usize,
    chunking: Chunking,
    chunk: &mut ChunkPass<'_>,
    on_chunk: &mut dyn FnMut(usize) -> Result<()>,
) -> Result<Vec<f32>> {
    let Chunking {
        width: batch,
        policy,
    } = chunking;
    // **The width the NPU has graphs for**, when there is one. A compiled
    // graph has one static shape, so a chunk of any other size is not
    // offered the device — and left to itself this sizer issues almost
    // anything but that shape. Measured on a 980-token prompt with blocks
    // compiled at 16: `[16, 16, 25, 16, 25, 19, 16, 24, 17, 16, 24, ...]`,
    // 52 submissions of which about half matched, so well over half the
    // prompt never reached the device. The sizer is not wrong — it is
    // pricing submissions against a time budget and knows nothing about a
    // graph's shape — so the shape is told to it here.
    let npu_width = crate::engine::npu_ffn_service().and_then(|npu| npu.prefill_width());
    // Snapped to the device's width, up as well as down.
    //
    // Down is obviously safe. Up needs an argument, because the sizer's
    // width is a bound set by a time budget and by what this backend can
    // submit without being reset. The argument is that the bound is being
    // applied to the wrong thing: with the feed-forward on the device it is
    // about 82% of a layer's GPU time that is no longer submitted at all
    // (`ORANGU_GPU_TIMESTAMPS` puts ffn-side at 186.7 ms of a 260 ms step),
    // so a chunk of `npu` tokens is a *smaller* GPU submission than the
    // sizer's own choice was before the device took the network. The sizer
    // measures wall time, which now includes work that is not on the GPU,
    // and prices the submission far above what it is.
    //
    // Bounded by the probe's candidates, so this can never ask for more
    // than 128 tokens at once — see `npu_tool::probe_prefill_width`.
    //
    // Worth 2.8x on a prompt that reaches this loop at all. llama 3.2 3B
    // `Q8_0`, prefill rate with the device off, on, and on with the widths
    // snapped:
    //
    // | prompt | off | on, unsnapped | snapped to 16 | snapped to 64 |
    // |---|---|---|---|---|
    // | 631 tokens | 3.71 tok/s | - | 10.56 | |
    // | 980 tokens | 3.65 | 5.52 | 10.35 | **23.46** |
    // | 2080 tokens | 3.53 | - | 10.09 | |
    //
    // The last column is the same prompt once the device was asked what
    // width it wanted rather than told: 62 submissions became 17.
    // A cache whose windowed layers keep a ring of the window plus one chunk
    // (`LayerCache::set_mirror_ring`) cannot take a wider chunk than the
    // ring was sized for: the chunk's last positions would overwrite the
    // window its first ones attend over. Bounded here, once, for every
    // route to a chunk width — the configured batch, the one-pass short
    // prompt, and the streaming regime's whole-prompt pass.
    let ring_bound = cache.max_positions_per_call();
    let batch = match ring_bound {
        Some(bound) if batch == 0 => bound,
        Some(bound) => batch.min(bound),
        None => batch,
    };
    let snap = |width: usize| match ring_bound {
        Some(bound) => npu_width.unwrap_or(width).min(bound).max(1),
        None => npu_width.unwrap_or(width),
    };
    // The one shape that needs no bounding: a prompt that fits in a single
    // chunk *and* starts at position zero is the least work a prefill can be.
    // `batch == 0` is an explicit opt-out — see [`prefill_batch`].
    //
    // **Kept even when the device holds blocks**, which was worth measuring
    // because the device only ever sees a chunk of exactly `npu_width` and
    // this path issues one wide chunk instead. Declining it so that a short
    // prompt is chopped into device-width pieces makes short prompts
    // slower, not faster — llama 3.2 3B `Q8_0`, prefill rate, one pass
    // against snapped 16s:
    //
    // | prompt | one pass | chopped to 16 |
    // |---|---|---|
    // | 75 tokens | **10.72 tok/s** | 7.35 |
    // | 193 tokens | **19.81** | 10.80 |
    //
    // A prompt this size is one efficient batched submission, and the
    // per-chunk cost of twelve of them is more than the device gives back.
    // Past one batch the comparison inverts — see the table on `snap` —
    // because the sizer is no longer choosing between one pass and many but
    // between many narrow ones that miss the device and many that hit it.
    if (batch == 0 || (tokens.len() <= batch && start_pos == 0))
        && tokens.len() <= decode_yield_width(last_chunk_wall(), policy, batch)
    {
        // `chunk` rather than a bare `forward`: it is the one that also
        // feeds the drafter its state rows. The pass still has to be
        // counted — `forward_passes` is what the NPU work measures chunking
        // against.
        let logits = chunk(cache, tokens, start_pos, true)?;
        crate::engine::note_forward_pass();
        on_chunk(tokens.len())?;
        return Ok(logits);
    }

    let budget = prefill_chunk_budget();
    let mut logits = Vec::new();
    let mut pos = start_pos;
    let mut done = 0usize;
    // Start with a probe rather than a full-width chunk. Nothing here knows
    // this machine's cost curve, and a full-width chunk at a deep position is
    // exactly the submission that hangs the GPU; a probe is cheap at any
    // depth and turns the next choice into arithmetic instead of a guess.
    //
    // Under `ChunkPolicy::Flat` there is nothing to probe for: the probe
    // exists to price a submission against a driver limit this backend does
    // not have, and pricing a streamed model by the clock is what shrinks the
    // chunk into a read spiral. Start at the full width and stay there.
    // Evened only under `Adaptive`. `Flat` exists to take the full width and
    // stay there — a streamed model pays for every extra pass in re-reads —
    // so narrowing its opening chunk would be answering a question it was
    // built not to ask.
    let mut width = snap(match policy {
        // Evened only once the sizer has a *priced* opinion. The probe is
        // deliberately narrow and exists to learn this machine's cost; there
        // is nothing to plan a split around before it has run, and narrowing
        // it further buys nothing.
        ChunkPolicy::Adaptive => match cost.opening_width(start_pos, tokens.len(), budget, batch) {
            Some(priced) if even_chunks() => even_chunk_width(tokens.len(), priced),
            Some(priced) => priced,
            None => PREFILL_PROBE_TOKENS.max(chunk_floor()).min(batch),
        },
        ChunkPolicy::Flat => batch,
    });
    // One line per prefill, not one per submission — which is the whole point.
    // `ORANGU_PREFILL_TRACE` answers a different question and answers it by
    // writing to stderr inside the submission loop, so it changes the cost it
    // is measuring: under it this sizer reads inflated chunk costs and picks
    // narrower chunks, which is a real effect on the number being checked.
    // Recording the widths and printing them once afterwards costs nothing
    // measurable and is enough to see what the sizer actually chose.
    let chunks_report = chunk_widths_reported();
    let mut widths: Vec<usize> = Vec::new();
    while done < tokens.len() {
        // Checked per chunk: a decode that starts mid-prompt narrows the rest
        // of it, and one that finishes lets the prompt widen again.
        let cap = decode_yield_width(last_chunk_wall(), policy, batch);
        // A width of 0 is the no-limit opt-out, which reaches this loop only
        // when the cap applies.
        let uncapped = if width == 0 { usize::MAX } else { width };
        let n = uncapped.min(cap).min(tokens.len() - done);
        let n = keep_pictures_whole(&tokens[done..], n, ring_bound.unwrap_or(usize::MAX));
        let started = Instant::now();
        let submits_before = crate::engine::decode_stages::submissions_so_far();
        let last = done + n >= tokens.len();
        let out = chunk(cache, &tokens[done..done + n], pos, last)?;
        if last {
            logits = out;
        }
        crate::engine::note_forward_pass();
        let elapsed = started.elapsed();
        let submits = crate::engine::decode_stages::submissions_so_far() - submits_before;
        note_chunk_wall(n, elapsed);
        pos += n;
        done += n;
        // After the forward, not before: progress means work finished.
        on_chunk(done)?;
        if chunks_report {
            widths.push(n);
        }
        if policy == ChunkPolicy::Adaptive {
            if !chunk_is_priced(submits) {
                // **A chunk that put nothing in front of the driver cannot
                // have tripped the timeout the budget exists for**, whatever
                // it cost on the host. Priced by the clock anyway, a model
                // whose layers all run on the host is cut into chunks to fit
                // a limit it is not subject to, and every extra pass re-pays
                // each layer's fixed cost — for a mixture, its whole expert
                // stack. So such a chunk is not priced at all: the width goes
                // to the configured maximum and the cost model is left
                // untouched rather than fed a point that means nothing.
                //
                // Per chunk rather than once per model, because a split model
                // submits on some passes and not others, and the question is
                // always about the pass that just ran.
                width = snap(batch);
            } else {
                cost.observe(n, budget_elapsed(elapsed, submits));
                let asked = cost.next_width(budget, batch);
                width = snap(if even_chunks() {
                    even_chunk_width(tokens.len() - done, asked)
                } else {
                    asked
                });
            }
        }
    }
    if chunks_report {
        let total: usize = widths.iter().sum();
        eprintln!(
            "orangu-server: [prefill-chunks] {} tokens from {} in {} submissions: {:?}",
            total,
            start_pos,
            widths.len(),
            widths
        );
    }
    Ok(logits)
}

/// A chunk of `n` of `rest`'s tokens that does not end inside a picture's
/// rows (`engine::vision`), whose rows may see each other both ways and so
/// belong in one pass: cut before the picture, or — for a chunk that starts
/// with it — run to its end, at most `limit` tokens.
fn keep_pictures_whole(rest: &[u32], n: usize, limit: usize) -> usize {
    use super::vision::is_virtual;
    if n == 0 || n >= rest.len() || !is_virtual(rest[n - 1]) || !is_virtual(rest[n]) {
        return n;
    }
    let start = rest[..n]
        .iter()
        .rposition(|&t| !is_virtual(t))
        .map_or(0, |i| i + 1);
    if start > 0 {
        return start;
    }
    let end = rest
        .iter()
        .position(|&t| !is_virtual(t))
        .unwrap_or(rest.len());
    end.min(limit.max(n))
}

/// Whether a model with recurrent layers speculates: `ORANGU_SPEC_RECURRENT`
/// where it is set, otherwise where the model verifies on its device
/// (`ModelForward::verifies_recurrent_on_device`); always in this module's
/// tests, which check the rollback.
fn spec_recurrent(model: &dyn ModelForward) -> bool {
    static SET: std::sync::OnceLock<Option<bool>> = std::sync::OnceLock::new();
    cfg!(test)
        || SET
            .get_or_init(|| crate::engine::env::flag_setting("ORANGU_SPEC_RECURRENT"))
            .unwrap_or_else(|| model.verifies_recurrent_on_device())
}

/// Whether a request waits for a peer that is building the same new prompt
/// prefix instead of building it too (`PrefixIndex::wait_or_claim`). On
/// unless `ORANGU_PREFIX_WAIT=0`.
fn prefix_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::engine::env::flag_on_unless_disabled("ORANGU_PREFIX_WAIT"))
}

/// How many requests are in their decode loop.
static DECODING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

thread_local! {
    /// Whether this thread's request is in its decode loop — so the passes
    /// it runs there (a batched step, a verify) are not taken for another
    /// request's prompt.
    static DECODING_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// One request's membership of [`DECODING`], for as long as it decodes.
struct Decoding;

impl Decoding {
    fn enter() -> Self {
        DECODING.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        DECODING_HERE.with(|here| here.set(true));
        Self
    }
}

impl Drop for Decoding {
    fn drop(&mut self) {
        DECODING_HERE.with(|here| here.set(false));
        DECODING.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Whether a prompt on this thread should make way for decode steps between
/// its layers: another request is decoding, and this thread is not.
///
/// Requests share one device whose queue runs submissions in order, and a
/// prompt records a whole chunk's layers ahead of the device, so a decode
/// step issued mid-chunk waited for all of it — up to the chunk budget of
/// seconds. While this holds, a prompt submits one layer at a time and keeps
/// at most one more queued (`VulkanBackend::end_prefill_group`), so a decode
/// step waits for about two layers of the chunk. `ORANGU_PREFILL_YIELD_LAYERS=0`
/// turns it off.
pub(crate) fn prompt_yields_to_decode() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::engine::env::flag_on_unless_disabled("ORANGU_PREFILL_YIELD_LAYERS"))
        && !DECODING_HERE.with(|here| here.get())
        && DECODING.load(std::sync::atomic::Ordering::Relaxed) > 0
}

/// The widest prefill chunk the next pass may take, given that other
/// requests may be decoding: unbounded when none is.
///
/// Requests share one device and its queue runs submissions in order, so a
/// decode step issued behind a prefill chunk waits for the whole chunk. The
/// sizer's chunks are bounded by a time budget of seconds, which is what a
/// decoding stream would stall for. While another request decodes, the chunk
/// is sized to `ORANGU_PREFILL_YIELD_MS` instead — a time, not a width, so a
/// model whose chunks are already that short keeps them and pays nothing.
///
/// **Off unless set** (`0`). Prefill on a card is compute-bound and a narrow
/// chunk still pays for the GEMM's padded token tile, so narrowing multiplies
/// the prompt's cost; and a queued decode step still waits for whatever the
/// prompt has in flight. On a 3B model at 150 ms it made a 2 048-token prompt
/// seven times slower and left the decoding stream stalling for half a second.
/// [`prompt_yields_to_decode`] bounds the stall instead, by putting decode
/// steps between a prompt's layers.
///
/// Priced from `wall`, the last chunk's tokens and **wall time**
/// ([`note_chunk_wall`]), not from the sizer's estimate: that one prices a
/// chunk per submission (`ORANGU_PREFILL_CHUNK_SPREAD`), since its budget
/// guards one submission against the driver's timeout, and on the
/// device-resident stream it reads several times cheaper than the wait a
/// queued decode step sees. Only under the adaptive policy, and only once a
/// chunk has been timed: the flat policy streams its weights once a pass, and
/// narrowing it multiplies that read.
fn decode_yield_width(wall: Option<(usize, Duration)>, policy: ChunkPolicy, batch: usize) -> usize {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let ms = *MS.get_or_init(|| {
        std::env::var("ORANGU_PREFILL_YIELD_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(PREFILL_YIELD_MS_DEFAULT)
    });
    yield_width_at(ms, wall, policy, batch)
}

/// [`decode_yield_width`] at a budget of `ms`.
fn yield_width_at(
    ms: u64,
    wall: Option<(usize, Duration)>,
    policy: ChunkPolicy,
    batch: usize,
) -> usize {
    let Some((n, elapsed)) = wall else {
        return usize::MAX;
    };
    if ms == 0
        || policy != ChunkPolicy::Adaptive
        || DECODING.load(std::sync::atomic::Ordering::Relaxed) == 0
    {
        return usize::MAX;
    }
    let max_width = if batch == 0 { usize::MAX } else { batch };
    next_chunk_width(n, elapsed, Duration::from_millis(ms), max_width)
}

/// The last prefill chunk's width and wall time, whichever request ran it —
/// carried across requests as the cost estimate is, so a prompt that starts
/// while another decodes is capped from its first chunk.
static LAST_CHUNK_WALL: Mutex<Option<(usize, Duration)>> = Mutex::new(None);

fn note_chunk_wall(n: usize, elapsed: Duration) {
    if n > 0 {
        *LAST_CHUNK_WALL.lock().unwrap_or_else(|p| p.into_inner()) = Some((n, elapsed));
    }
}

fn last_chunk_wall() -> Option<(usize, Duration)> {
    *LAST_CHUNK_WALL.lock().unwrap_or_else(|p| p.into_inner())
}

const PREFILL_YIELD_MS_DEFAULT: u64 = 0;

/// Whether `ORANGU_PREFILL_CHUNKS=1` asked for one line per prefill naming the
/// widths the sizer chose.
/// `ORANGU_PREFILL_EVEN=0` takes the sizer's width as it comes — the control
/// arm for [`even_chunk_width`]. On unless turned off.
fn even_chunks() -> bool {
    crate::engine::env::flag_on_unless_disabled("ORANGU_PREFILL_EVEN")
}

/// The widest chunk that divides `remaining` into the **same number of
/// passes** as `width` would, so the passes come out even.
///
/// The sizer answers "how wide may the next chunk be"; it never asks what
/// that leaves for the ones after it. Under a 2048 ceiling a 2236-token
/// prompt is taken as `[2048, 188]` — two passes, the first at the ceiling —
/// where `[1118, 1118]` is also two passes and its widest chunk is **half**
/// as wide. Peak scratch is set by the widest chunk, and on a full card that
/// scratch is what pages.
///
/// Never wider than `width`, so a bound the caller established for the
/// driver's sake still holds, and never fewer passes, so nothing is made
/// more expensive to stream. `width == 0` is the no-limit opt-out and passes
/// through untouched.
fn even_chunk_width(remaining: usize, width: usize) -> usize {
    if width == 0 || remaining == 0 {
        return width;
    }
    let passes = remaining.div_ceil(width);
    remaining.div_ceil(passes.max(1))
}

/// What a chunk's wall time is worth against the budget, given how many
/// device submissions it was.
///
/// The budget exists for one reason: a submission the driver runs for
/// longer than its timeout resets the device. That is a bound on the
/// **longest submission**, not on the chunk. On the device-resident stream
/// a chunk is a handful of submissions and its wall time is a fair proxy
/// for the longest of them; on a host-orchestrated model it is hundreds —
/// one per matmul, the activations home between them — and the chunk's
/// wall time says nothing about any one of them. Sized by wall time, a
/// model whose 128-token chunk takes 34 seconds on an integrated GPU was
/// cut to eleven-token chunks, each paying every layer's fixed cost, and
/// its 572-token prompt ran at 1.5 tok/s where one chunk ran at 5.7.
///
/// So a chunk of `submits` submissions counts as `elapsed × spread /
/// submits` — its mean submission time with a margin for the longest —
/// and never more than `elapsed` itself. `spread` is
/// `ORANGU_PREFILL_CHUNK_SPREAD` (default 4; `0` counts the whole wall
/// time, the rule this replaces, for the control arm). A chunk with fewer
/// submissions than the spread is unchanged, which keeps the stream where
/// it was.
fn budget_elapsed(elapsed: Duration, submits: u64) -> Duration {
    let spread = chunk_submission_spread();
    if spread == 0 || submits <= spread {
        return elapsed;
    }
    elapsed.mul_f64(spread as f64 / submits as f64)
}

/// Whether a chunk of `submits` submissions is subject to the budget at all.
///
/// The budget bounds the longest **device submission**, because a submission
/// the driver runs past its timeout resets the device. A chunk that made no
/// submission has no such exposure, and its wall time — however long — is not
/// evidence about a limit it cannot reach.
///
/// `ORANGU_PREFILL_CHUNK_SPREAD=0` opts back out, with the rest of the
/// submission-aware pricing, so the whole rule has one control arm.
fn chunk_is_priced(submits: u64) -> bool {
    submits > 0 || chunk_submission_spread() == 0
}

fn chunk_submission_spread() -> u64 {
    static SPREAD: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *SPREAD.get_or_init(|| {
        std::env::var("ORANGU_PREFILL_CHUNK_SPREAD")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(4)
    })
}

fn chunk_widths_reported() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::engine::env::flag_on("ORANGU_PREFILL_CHUNKS"))
}

/// How wide the next chunk should be, reading one chunk's cost as if all of it
/// were per-token work.
///
/// This is the estimate available before two observations exist to separate a
/// submission's fixed cost from its marginal one ([`ChunkCost`]), and the one
/// deliberately used again whenever a chunk overruns its budget — charging
/// everything to per-token work under-sizes, and under-sizing is the safe
/// direction.
///
/// Cost per token is not constant: a prefill chunk attends over everything
/// before it, so the same 512 tokens that take 2.3 s at position 0 take 10.1 s
/// at position 6 656 — past the ~10 s the graphics driver allows a single
/// submission before it resets the device. Sizing by token count alone, as
/// `ORANGU_PREFILL_BATCH` did on its own, therefore stops protecting anything
/// once a prompt gets long: the *count* is bounded but the *work* is not.
///
/// Scaling the width by the rate just measured keeps the work per submission
/// roughly constant instead. The rate only drifts upward, and slowly, so
/// sizing from the previous chunk lands close; the budget is set well under
/// the driver's limit to absorb what it misses, and the estimate itself aims
/// below that budget again because the *next* chunk always starts deeper into
/// the prompt than the one just timed.
fn next_chunk_width(n: usize, elapsed: Duration, budget: Duration, max_width: usize) -> usize {
    let per_token = elapsed.as_secs_f64() / (n.max(1) as f64);
    if per_token <= 0.0 {
        return max_width;
    }
    clamp_chunk_width(
        budget.as_secs_f64() * PREFILL_BUDGET_HEADROOM / per_token,
        max_width,
    )
}

/// What one prefill submission costs on this machine, as a line through its
/// token count: `fixed + per_token · n` seconds.
///
/// A chunk's cost is not proportional to its width. A submission pays for
/// itself before it processes a single token — the encoder, the bind groups,
/// the host round trip, and the per-layer dispatches whose grids do not shrink
/// with the batch — and only then pays per token. Sizing from `elapsed / n`,
/// which is what this did while it modelled cost as pure proportion, reads all
/// of that fixed cost as if it were per-token work. The error is worst exactly
/// where the sizing starts: the opening probe is almost entirely fixed cost,
/// so its apparent per-token rate comes out several times the real one and the
/// next chunk is sized at a fraction of what the budget allows. The width then
/// climbs one chunk at a time, and on a prompt only a few chunks long the
/// climb *is* the prompt.
///
/// Two observations at different widths separate the two terms. That
/// separation is also what keeps this portable: `fixed` and `per_token` are
/// measured here, against this machine and this model, rather than carried in
/// as constants from whichever machine the code was written on. A card with a
/// cheaper round trip, a model with fewer layers, or a backend with no
/// per-submission cost at all describes itself correctly through the same two
/// numbers.
#[derive(Clone, Copy, Debug, Default)]
struct ChunkCost {
    /// The most recent observation: how wide a chunk was, and what it cost.
    last: Option<(usize, Duration)>,
    /// The line through the two most recent observations wide enough apart to
    /// support one, kept until a better-separated pair replaces it.
    fit: Option<CostFit>,
    /// Never form a fit, and so never offer an opening width — which leaves
    /// the proportional sizer this replaced, exactly. The control arm for
    /// `ORANGU_NO_CHUNK_COST_FIT`, carried in the value rather than read from
    /// the environment down here so the type stays pure and a test can have
    /// either behaviour without touching a process-wide switch.
    proportional_only: bool,
}

/// `fixed + per_token · n` seconds, both non-negative.
#[derive(Clone, Copy, Debug, PartialEq)]
struct CostFit {
    fixed: f64,
    per_token: f64,
}

/// How much wider than its partner the wider of two observations must be
/// before the pair is allowed to set a slope.
///
/// Two points near the same width put a small, noisy denominator under the
/// subtraction — the slope they imply is mostly the difference between two
/// timings of the same thing. Requiring real separation costs nothing, because
/// the widths this sizer produces vary by much more than this on the way up,
/// and a pair that fails the test leaves the previous fit standing rather than
/// replacing it with a coincidence.
const COST_FIT_SEPARATION: usize = 2;

impl CostFit {
    /// The line through two observations, or `None` when the pair cannot
    /// support one.
    ///
    /// Rejects a pair that is too close together to carry a slope, and one
    /// where the wider chunk did not cost more — which is noise, not a
    /// negative marginal cost, and would size the next chunk off a downward
    /// line.
    fn through(a: (usize, Duration), b: (usize, Duration)) -> Option<Self> {
        let (narrow, wide) = if a.0 <= b.0 { (a, b) } else { (b, a) };
        let (n_narrow, n_wide) = (narrow.0, wide.0);
        if n_narrow == 0 || n_wide < n_narrow.saturating_mul(COST_FIT_SEPARATION) {
            return None;
        }
        let (t_narrow, t_wide) = (narrow.1.as_secs_f64(), wide.1.as_secs_f64());
        let per_token = (t_wide - t_narrow) / ((n_wide - n_narrow) as f64);
        if !per_token.is_finite() || per_token <= 0.0 {
            return None;
        }
        // Clamped rather than rejected: a measured intercept below zero means
        // the narrow point sat above the line, which is ordinary scatter, and
        // the slope it came with is still the useful half of the answer.
        let fixed = (t_wide - per_token * n_wide as f64).max(0.0);
        if !fixed.is_finite() {
            return None;
        }
        Some(Self { fixed, per_token })
    }

    /// This line moved to pass through `point`, keeping its fixed term.
    ///
    /// The marginal cost is whatever is left of the observation once the fixed
    /// cost is taken off it. Floored above zero rather than at it: a point
    /// that lands under the intercept — ordinary scatter on a narrow chunk —
    /// would otherwise produce a zero or negative slope and size the next
    /// chunk at the ceiling on the strength of one cheap measurement.
    fn through_holding_fixed(self, (n, elapsed): (usize, Duration)) -> Self {
        if n == 0 {
            return self;
        }
        let marginal = (elapsed.as_secs_f64() - self.fixed) / n as f64;
        if !marginal.is_finite() || marginal <= 0.0 {
            return self;
        }
        Self {
            fixed: self.fixed,
            per_token: marginal,
        }
    }

    /// The widest chunk whose predicted cost still fits `target` seconds.
    fn fits_in(&self, target: f64) -> f64 {
        let room = target - self.fixed;
        if room <= 0.0 {
            return 0.0;
        }
        room / self.per_token
    }
}

impl ChunkCost {
    /// An estimate that has measured nothing yet.
    const fn new() -> Self {
        Self {
            last: None,
            fit: None,
            proportional_only: false,
        }
    }

    /// An estimate that will never fit — the control arm, which restores the
    /// single-point sizer this replaced.
    const fn proportional_only() -> Self {
        Self {
            last: None,
            fit: None,
            proportional_only: true,
        }
    }

    /// Record what a chunk of `n` tokens actually cost.
    fn observe(&mut self, n: usize, elapsed: Duration) {
        let point = (n, elapsed);
        if !self.proportional_only {
            match self.last.and_then(|prev| CostFit::through(prev, point)) {
                // Two well-separated widths price both terms.
                Some(fit) => self.fit = Some(fit),
                // One width cannot, but it must still be allowed to move the
                // estimate, and this is the case that occurs *most* — once the
                // sizer settles at a width it keeps choosing it, so every
                // later pair is the same width twice and can never separate.
                // Leaving the fit untouched there freezes it: whatever the
                // first two chunks happened to say becomes permanent, a single
                // unlucky pair sizes every prompt the server will ever serve,
                // and because the frozen width is then the only width chosen,
                // nothing can ever dislodge it. Re-derive the marginal cost
                // through the new point instead, holding the fixed cost — the
                // decomposition is what makes that sound. Fixed cost is a
                // property of the submission and is stable; marginal cost
                // climbs with context, so it is the term that has to track.
                None => {
                    if let Some(fit) = self.fit {
                        self.fit = Some(fit.through_holding_fixed(point));
                    }
                }
            }
        }
        self.last = Some(point);
    }

    /// How wide the next chunk of the prompt being processed should be.
    fn next_width(&self, budget: Duration, max_width: usize) -> usize {
        let target = budget.as_secs_f64() * PREFILL_BUDGET_HEADROOM;
        // No separate "did it overrun?" branch, because the fit already
        // shrinks on one and cannot fail to. Every observation moves the line
        // through the newest point, so a chunk of `n` tokens that cost `t`
        // sizes the next at `n · (target − fixed) / (t − fixed)`, which is
        // below `n` exactly when `t` exceeds the target — which is what an
        // overrun is. A guard in front of that would be safety code no test
        // could ever reach: it would read as a promise, and the honest way to
        // keep this one is the arithmetic above rather than a branch nothing
        // exercises.
        match (self.fit, self.last) {
            (Some(fit), _) => clamp_chunk_width(fit.fits_in(target), max_width),
            // One observation cannot separate the terms. Charging all of it to
            // per-token work is the conservative reading — it under-sizes, and
            // under-sizing costs a chunk where over-sizing costs a device.
            (None, Some((n, elapsed))) => next_chunk_width(n, elapsed, budget, max_width),
            (None, None) => max_width,
        }
    }

    /// How wide to open a prompt of `remaining` tokens at `start_pos`, for a
    /// machine already priced by an earlier one — or `None` to open with the
    /// probe.
    ///
    /// From position zero, the width this machine has been shown to sustain.
    /// The probe exists to keep the opening submission small at a depth where
    /// a token costs many times what it did at the start, and an estimate
    /// carried in from an earlier request knows nothing about the depth a
    /// continuation begins at — so a continuation probes, **unless the whole
    /// of it fits the budget many times over**: at [`continuation_margin`]
    /// times the carried per-token cost, which grows with the depth. A chat
    /// turn's few hundred new tokens are such a continuation, and probing
    /// them made two passes of one, each paying the GEMM's padded token tile.
    fn opening_width(
        &self,
        start_pos: usize,
        remaining: usize,
        budget: Duration,
        max_width: usize,
    ) -> Option<usize> {
        let fit = self.fit?;
        if start_pos == 0 {
            return Some(self.next_width(budget, max_width));
        }
        // The margin on the per-token term only: depth makes a token dearer,
        // not the submission around it.
        let cost = fit.fixed + continuation_margin(start_pos) * fit.per_token * remaining as f64;
        (remaining <= max_width && cost <= budget.as_secs_f64() * PREFILL_BUDGET_HEADROOM)
            .then_some(remaining)
    }
}

/// The bounds every adapted width lives inside, in one place.
///
/// `min(max_width)` on the floor as well: a caller that configured a batch
/// smaller than the floor asked for chunks that small, and `clamp` panics
/// outright when its own min exceeds its max.
fn clamp_chunk_width(fits: f64, max_width: usize) -> usize {
    let fits = if fits.is_finite() && fits >= 0.0 {
        fits as usize
    } else {
        0
    };
    fits.clamp(
        PREFILL_MIN_CHUNK_TOKENS.max(chunk_floor()).min(max_width),
        max_width,
    )
}

/// Only a fraction of the configured budget is spent when sizing the next
/// chunk. The measured chunk just completed at a shallower position than the
/// next one will start from, so using the full budget as the target leaves no
/// room for the cost curve to rise between them.
const PREFILL_BUDGET_HEADROOM: f64 = 0.75;

/// How many times the carried per-token cost a continuation at `start_pos`
/// is charged when deciding whether it may skip the probe: four, and one more
/// for every 4 096 positions of depth, since attention's share of a token
/// grows with the context it attends over.
fn continuation_margin(start_pos: usize) -> f64 {
    4.0 * (start_pos as f64 / 4096.0).max(1.0)
}

/// Tokens in the opening probe chunk. Small enough to stay well inside the
/// driver's limit even at the deepest context this server will accept.
const PREFILL_PROBE_TOKENS: usize = 16;

/// The floor on an adapted chunk. At a deep enough context even this exceeds
/// the budget — nothing can be done about that except take longer — but it
/// keeps the submission small enough that the driver does not give up on it.
const PREFILL_MIN_CHUNK_TOKENS: usize = 16;

/// Wall-clock target for one prefill submission — `ORANGU_PREFILL_CHUNK_MS`,
/// default [`PREFILL_CHUNK_BUDGET_MS_DEFAULT`].
///
/// This is a *timeout* budget, like `ORANGU_PREFILL_BATCH` before it, not a
/// memory one. The number that matters is the graphics driver's own limit on
/// how long one submission may run — around 10 s on the amdgpu/RADV stack this
/// was measured on, and not queryable from userspace. The default leaves room
/// for the estimate to be wrong and for the machine to be busier next time.
fn prefill_chunk_budget() -> Duration {
    static MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    Duration::from_millis(*MS.get_or_init(|| {
        std::env::var("ORANGU_PREFILL_CHUNK_MS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|ms| *ms > 0)
            .unwrap_or(PREFILL_CHUNK_BUDGET_MS_DEFAULT)
    }))
}

const PREFILL_CHUNK_BUDGET_MS_DEFAULT: u64 = 3_000;

/// Whether `ORANGU_GPU_TRACE` is set, cached.
///
/// Read once — this sits in the decode loop, which runs once per token.
fn gpu_trace() -> bool {
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE.get_or_init(|| crate::engine::env::flag_on("ORANGU_GPU_TRACE"))
}

/// Prompt lookup's draft length for an echo in progress
/// (`ORANGU_SPEC_ECHO_DRAFT`, `0` off); taken only while its verifies yield
/// more tokens a millisecond than the plain length's (`engine::draft_control`).
///
/// Off by default for a model with no device (`on_device` false). On the CPU
/// backend a verify's cost grows with its rows nearly as fast as an echo's
/// yield does, so the longer draft barely leads even when every token of it
/// is accepted, and an echo that breaks off — an edit — leaves it behind; the
/// measuring costs more than it finds.
fn spec_echo_draft(on_device: bool) -> usize {
    std::env::var("ORANGU_SPEC_ECHO_DRAFT")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(if on_device { 16 } else { 0 })
}

/// Set at start-up when the served model has no multi-position forward to
/// verify a draft with, which turns prompt lookup off for the process.
static NO_VERIFY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Turns prompt-lookup speculation off for this process: the served model
/// cannot verify a draft (`ModelForward::forward_verify` refused a probe).
pub fn disable_prompt_lookup() {
    NO_VERIFY.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Whether speculation is on for this process, before the model is asked.
pub fn speculation_enabled() -> bool {
    speculative_config().is_some()
}

fn trace_tokens() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| crate::engine::env::flag_on("ORANGU_TRACE_TOKENS"))
}

fn speculative_config() -> Option<(usize, usize)> {
    // On unless turned off: with `engine::draft_control` deciding per step,
    // a draft is verified only while verifying pays, and an empty one costs
    // nothing. Off under test, where a request's forward count is asserted.
    if cfg!(test)
        || NO_VERIFY.load(std::sync::atomic::Ordering::Relaxed)
        || !crate::engine::env::flag_on_unless_disabled("ORANGU_SPECULATIVE")
    {
        return None;
    }
    let read = |name: &str, default: usize| {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(default)
    };
    Some((read("ORANGU_SPEC_NGRAM", 2), read("ORANGU_SPEC_DRAFT", 4)))
}

/// How far back prompt lookup's match must reach for a block head to copy
/// it instead of drafting a block (`ORANGU_SPEC_COPY_CONTEXT`, by default
/// prompt lookup's own n-gram; `0` copies echoes alone, [`ECHO_CONTEXT`]).
/// A copy drafts the whole of prompt lookup's width where a block drafts
/// the head's, and prose seldom matches at all.
fn copy_context() -> usize {
    static CONTEXT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *CONTEXT.get_or_init(|| {
        match std::env::var("ORANGU_SPEC_COPY_CONTEXT")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
        {
            Some(0) => ECHO_CONTEXT,
            Some(n) => n,
            None => COPY_CONTEXT,
        }
    })
}

/// [`copy_context`]'s default.
const COPY_CONTEXT: usize = 2;

/// The most positions a block head copying an echo leaves out of its
/// context before caching them: each one holds a state row of every tapped
/// layer until it is.
const DEFERRED_CONTEXT: usize = 64;

/// Prompt lookup's settings, kept by a block head to draft an echo with
/// ([`Drafter::Mtp`]'s `echo`).
#[derive(Debug, Clone, Copy)]
struct EchoLookup {
    ngram: usize,
    max_draft: usize,
    echo_draft: usize,
}

/// Where a speculative step's candidate tokens come from.
///
/// The two sources have opposite cost profiles, and that is the whole reason
/// both exist. Prompt-lookup calls no model at all, so a wrong guess costs
/// nothing beyond the wider verification forward — but it produces nothing
/// unless the context happens to repeat itself. A draft model always produces
/// a draft and always pays a forward pass per token for it, so it wins on
/// ordinary prose and loses whenever the pair agrees too rarely to cover the
/// drafting cost.
///
/// Verification does not care which one produced the tokens: it re-derives
/// what the served model would have said and keeps the matching prefix, so the
/// emitted text is identical either way. A drafter can only change how fast
/// the answer arrives, never what it is.
enum Drafter<'a> {
    /// Copy the continuation of an earlier occurrence of the trailing n-gram:
    /// up to `max_draft` tokens, or `echo_draft` for a draft that
    /// [`Drafter::echoes`] where wider verifies pay
    /// (`engine::draft_control`).
    PromptLookup {
        ngram: usize,
        max_draft: usize,
        echo_draft: usize,
    },
    /// Run the served model's own multi-token-prediction head — one decoder
    /// block that reads the token just produced *and* the state the served
    /// model was in when it produced it (`engine::arch::qwen4exp_mtp`).
    ///
    /// Cheaper per drafted token than a draft model (one block, not a whole
    /// second trunk) and a much closer guesser (it was trained against this
    /// model's own states), at the cost of a state row that has to be kept
    /// in step with the served model token for token — which is what
    /// everything below is bookkeeping for.
    Mtp {
        head: &'a dyn MtpHead,
        tokens: usize,
        /// The head's own single-layer cache, indexed by the served model's
        /// absolute positions.
        cache: KvCache,
        /// How many positions of that cache are filled — placeholders
        /// included. The same quantity `Drafter::Model` tracks, and kept for
        /// the same reason.
        committed: usize,
        /// The earliest position the head has a *real* row for, or `None`
        /// until it has one. Everything below it is a placeholder standing
        /// in for a prompt prefix this request reused instead of forwarding,
        /// and must stay out of every attention window.
        first_pos: Option<usize>,
        /// The served model's own state rows from its most recent forward,
        /// and the absolute position of row 0. A draft step needs the state
        /// of the position *before* the token it reads, so this is the only
        /// thing that has to survive between one forward and the next.
        states: Vec<f32>,
        states_start: usize,
        /// One state row's width, from `ModelForward::mtp_state_width`.
        width: usize,
        /// For a head that drafts a block (`MtpHead::block_size`): prompt
        /// lookup's n-gram, plain width and echo width, with which an echo
        /// is drafted as prompt lookup would draft it — a copy is longer and
        /// surer than any block the head proposes. The head's context is
        /// cached on those steps all the same.
        echo: Option<EchoLookup>,
    },
    /// Run a draft head that attends over the served model's own cache
    /// (`MtpHead::reads_served_cache`, e.g. `engine::arch::gemma4_assistant`).
    /// It keeps no cache and runs no catch-up — every draft step reads the
    /// served model's committed rows at the position it is about to fill —
    /// so all it holds is the served model's state rows from its last
    /// forward, of which a step reads the one behind the token it starts
    /// from.
    Assistant {
        head: &'a dyn MtpHead,
        tokens: usize,
        states: Vec<f32>,
        states_start: usize,
        width: usize,
    },
    /// Run a second, smaller model autoregressively.
    Model {
        model: &'a dyn ModelForward,
        tokens: usize,
        /// The draft's own KV cache, which tracks the same committed token
        /// sequence as the target's.
        cache: KvCache,
        /// How many of `history` this cache holds. Kept explicitly rather
        /// than read off the cache, because a block-compressed slot's `len`
        /// is a row count and one row can stand for several tokens
        /// (`KvCache::committed_tokens`) — a draft architecture with a stride
        /// would otherwise desynchronise silently.
        committed: usize,
    },
}

impl Drafter<'_> {
    /// The tokens to speculate on, continuing after all of `history`.
    ///
    /// `room` caps the draft at what the KV cache can still hold. It is not a
    /// tuning knob: a verification forward appends the whole draft to the
    /// cache at once, so a draft longer than the remaining context writes past
    /// the allocation — on the GPU mirror, which is sized to the request's
    /// capacity, that is a write into whatever is next rather than an error.
    /// Only reachable in the last few tokens of a full context, which is why
    /// it survived unnoticed on the prompt-lookup path.
    fn draft(
        &mut self,
        history: &[u32],
        room: usize,
        served: &KvCache,
        slot_id: usize,
    ) -> Result<Vec<u32>> {
        if room == 0 {
            return Ok(Vec::new());
        }
        if let Drafter::Assistant { .. } = self {
            return self.draft_from_served(history, room, served);
        }
        // Dispatched before the match rather than inside it: the head's own
        // step borrows the drafter, and an arm already holds it.
        if self.wants_states() {
            return self.draft_from_head(history, room);
        }
        match self {
            Drafter::PromptLookup {
                ngram,
                max_draft,
                echo_draft,
            } => Ok(ngram_draft(
                history,
                *ngram,
                (*max_draft).max(*echo_draft).min(room),
            )),
            Drafter::Mtp { .. } | Drafter::Assistant { .. } => unreachable!("dispatched above"),
            Drafter::Model {
                model,
                tokens,
                cache,
                committed,
            } => {
                let tokens = &(*tokens).min(room);
                // Everything committed that this cache has not seen yet — the
                // whole prompt on the first step, one token on every step
                // after.
                let catch_up = &history[*committed..];
                if catch_up.is_empty() {
                    // `history` always ends with the token the target is about
                    // to forward, and `committed` never runs past it.
                    anyhow::bail!("draft cache is ahead of the committed history");
                }
                let mut logits =
                    draft_forward(*model, cache, catch_up, *committed, *tokens, slot_id)?;
                *committed = history.len();
                let mut drafted = Vec::with_capacity(*tokens);
                for i in 0..*tokens {
                    // Plain argmax, not the request's sampler: a draft is a
                    // guess, and its only effect is how often the guess is
                    // accepted. Running the caller's temperature here would
                    // make the *draft* random without making the output any
                    // less determined by the target.
                    let token = crate::engine::sampling::argmax(&logits);
                    drafted.push(token);
                    // The last drafted token is never fed back: nothing would
                    // read its logits, and committing it to the draft cache
                    // would put the cache a token ahead of what the target can
                    // possibly accept.
                    if i + 1 == *tokens {
                        break;
                    }
                    logits = draft_forward(*model, cache, &[token], *committed, *tokens, slot_id)?;
                    *committed += 1;
                }
                Ok(drafted)
            }
        }
    }

    /// Rolls the draft's own cache back to the tokens the target actually
    /// committed.
    ///
    /// Without this the rejected tail stays in the draft cache and every
    /// later position is written one slot too far along — the draft would
    /// keep producing tokens, they would simply stop resembling anything the
    /// target would say, and the only symptom would be an acceptance rate
    /// quietly falling to zero.
    fn commit(&mut self, committed_tokens: usize) -> Result<()> {
        match self {
            Drafter::Model {
                cache, committed, ..
            }
            | Drafter::Mtp {
                cache, committed, ..
            } if *committed > committed_tokens => {
                cache.truncate(committed_tokens)?;
                *committed = committed_tokens;
            }
            _ => {}
        }
        Ok(())
    }

    /// The [`Drafter::Mtp`] half of [`Self::draft`].
    ///
    /// The catch-up's own last position *is* the first draft step: it reads
    /// the token the served model is about to forward, paired with the state
    /// behind it, which is precisely what a guess at the next token is. Every
    /// step after it chains onto the head's own state and its own guess.
    fn draft_from_head(&mut self, history: &[u32], room: usize) -> Result<Vec<u32>> {
        let want = match self {
            Drafter::Mtp { tokens, .. } => (*tokens).min(room),
            _ => unreachable!("guarded by wants_states"),
        };
        let mut drafted = Vec::with_capacity(want);
        let committed = self.mtp_committed();
        anyhow::ensure!(
            committed < history.len(),
            "the draft head's cache is ahead of the committed history"
        );
        // A copy the context matches far enough back is drafted as prompt
        // lookup drafts it, at its width. Through an echo the head's context
        // waits for the next block it drafts: one catch-up over the whole
        // copy reads the head's weights once, where one a step reads them
        // every step.
        if let Drafter::Mtp {
            echo: Some(lookup), ..
        } = self
            && let Some((len, from)) = ngram_match(history, lookup.ngram)
            && from < history.len()
            && len >= copy_context()
        {
            let ngram = lookup.ngram;
            if len < ECHO_CONTEXT || history.len() - committed > DEFERRED_CONTEXT {
                self.catch_up(&history[committed..], committed)?;
            }
            return Ok(ngram_draft(history, ngram, room));
        }
        let Some(step) = self.catch_up(&history[committed..], committed)? else {
            return Ok(drafted);
        };
        // A block head drafts the whole run in one pass of its own, from
        // the context the catch-up just cached.
        if let Drafter::Mtp {
            head,
            cache,
            first_pos,
            ..
        } = self
            && head.block_size().is_some()
        {
            let pos = history.len() - 1;
            return head.draft_block(cache, history[pos], pos, want, first_pos.unwrap_or(0));
        }
        let mut logits = step.logits;
        let mut state = step.state;
        for i in 0..want {
            // Plain argmax, not the request's sampler, for the reason the
            // draft model uses one: a guess's only effect is how often it is
            // accepted.
            drafted.push(crate::engine::sampling::argmax(&logits));
            // Nothing would read the last guess's own logits, and caching
            // its row would put the head a position ahead of anything the
            // served model can accept.
            if i + 1 == want {
                break;
            }
            let step = self.mtp_step(drafted[i], &state)?;
            logits = step.logits;
            state = step.state;
        }
        Ok(drafted)
    }

    /// The [`Drafter::Assistant`] half of [`Self::draft`]: from the token
    /// the served model is about to read and its state behind it, each step
    /// guesses the next token and hands its own state to the step after,
    /// all at that token's position over the rows `served` committed.
    /// Empty while the state behind it is missing — the first step after a
    /// prompt, whose verify then exports one.
    fn draft_from_served(
        &self,
        history: &[u32],
        room: usize,
        served: &KvCache,
    ) -> Result<Vec<u32>> {
        let Drafter::Assistant { head, tokens, .. } = self else {
            unreachable!("only reached from the Assistant arm")
        };
        let want = (*tokens).min(room);
        let mut drafted = Vec::with_capacity(want);
        let Some(&last) = history.last() else {
            return Ok(drafted);
        };
        let pos = history.len() - 1;
        let Some(PrevState::Row(state)) = self.previous_state(pos) else {
            return Ok(drafted);
        };
        let (mut token, mut state) = (last, state.to_vec());
        while drafted.len() < want {
            let Some(step) = head.forward_served(served, token, &state, pos)? else {
                break;
            };
            // Plain argmax, for the reason every drafter here uses one.
            token = crate::engine::sampling::argmax(&step.logits);
            drafted.push(token);
            state = step.state;
        }
        Ok(drafted)
    }

    /// How many positions the MTP head's cache holds, or `0` for a drafter
    /// that keeps no state of the served model's.
    fn mtp_committed(&self) -> usize {
        match self {
            Drafter::Mtp { committed, .. } => *committed,
            _ => 0,
        }
    }

    /// Whether this drafter needs the served model to export its hidden
    /// state — the one thing a forward has to do differently for it.
    fn wants_states(&self) -> bool {
        matches!(self, Drafter::Mtp { .. })
    }

    /// Whether every verify has to export the served model's states — a
    /// head's next draft starts from one of them. Such a drafter verifies on
    /// every step, an empty draft included: a step taken any other way
    /// leaves the head nothing to start from.
    /// The most tokens one draft holds — a verify writes that many rows
    /// past the token it starts from.
    fn max_tokens(&self) -> usize {
        match self {
            Drafter::PromptLookup {
                max_draft,
                echo_draft,
                ..
            } => (*max_draft).max(*echo_draft),
            Drafter::Mtp {
                tokens,
                echo: Some(lookup),
                ..
            } => (*tokens).max(lookup.max_draft).max(lookup.echo_draft),
            Drafter::Mtp { tokens, .. }
            | Drafter::Assistant { tokens, .. }
            | Drafter::Model { tokens, .. } => *tokens,
        }
    }

    fn verifies_with_states(&self) -> bool {
        matches!(self, Drafter::Mtp { .. } | Drafter::Assistant { .. })
    }

    /// Whether the next draft copies a passage the reply is echoing: prompt
    /// lookup's match reaches [`ECHO_CONTEXT`] tokens back. Such a draft is
    /// verified even while verifying has not been paying
    /// (`engine::draft_control`): that verdict is an average over the
    /// request so far, and an echo that starts after prose would otherwise
    /// wait out the average.
    /// How many tokens this step's draft may hold: the echo length for an
    /// echo that `wide` says to verify wide, the drafter's own otherwise —
    /// on a block head's echo, prompt lookup's own.
    fn draft_len(&self, wide: bool) -> usize {
        if let Drafter::Mtp {
            echo: Some(lookup), ..
        } = self
        {
            return if wide {
                lookup.max_draft.max(lookup.echo_draft)
            } else {
                lookup.max_draft
            };
        }
        match self {
            Drafter::PromptLookup {
                max_draft,
                echo_draft,
                ..
            } if wide => (*max_draft).max(*echo_draft),
            Drafter::PromptLookup { max_draft, .. } => *max_draft,
            Drafter::Mtp { tokens, .. } => *tokens,
            _ => self.max_draft(),
        }
    }

    /// Whether this drafter has a wider draft for echoes than its own.
    fn has_echo_draft(&self) -> bool {
        matches!(self, Drafter::PromptLookup { max_draft, echo_draft, .. } if echo_draft > max_draft)
            || matches!(self, Drafter::Mtp { echo: Some(lookup), .. } if lookup.echo_draft > lookup.max_draft)
    }

    fn echoes(&self, history: &[u32]) -> bool {
        match self {
            Drafter::PromptLookup { ngram, .. } => echoing(history, *ngram),
            Drafter::Mtp {
                echo: Some(lookup), ..
            } => echoing(history, lookup.ngram),
            _ => false,
        }
    }

    /// Takes the served model's per-position state rows for the batch it has
    /// just forwarded. Row 0 is position `start_pos`.
    ///
    /// Only stored, never consumed here: what a draft step needs is the row
    /// *behind* the token it reads, and which token that is depends on how
    /// much of the batch the verification went on to accept.
    ///
    /// **One row of overlap is carried.** A batch's first token pairs with
    /// the last state of the batch before it, so dropping the old rows
    /// wholesale loses exactly the one the seam needs — and the head then
    /// gains a hole at every chunk boundary of a long prompt, which the
    /// rest of this refuses to paper over.
    ///
    /// A block head that copies an echo caches its context later, in one
    /// batch ([`Self::draft_from_head`]), so it keeps every row from the
    /// first its next catch-up reads — the rows a verify rewrote dropped,
    /// for this batch's in their place.
    fn observe_states(&mut self, start_pos: usize, rows: &[f32]) {
        if let Drafter::Mtp {
            states,
            states_start,
            width,
            committed,
            echo: Some(_),
            ..
        } = self
        {
            let end = *states_start + states.len() / *width;
            let keep_from = committed.saturating_sub(1).max(*states_start);
            if !states.is_empty() && (*states_start..=end).contains(&start_pos) {
                states.truncate((start_pos - *states_start) * *width);
                let stale = keep_from.min(start_pos) - *states_start;
                states.drain(..stale * *width);
                *states_start += stale;
            } else {
                states.clear();
                *states_start = start_pos;
            }
            states.extend_from_slice(rows);
            return;
        }
        if let Drafter::Mtp {
            states,
            states_start,
            width,
            ..
        }
        | Drafter::Assistant {
            states,
            states_start,
            width,
            ..
        } = self
        {
            let contiguous =
                !states.is_empty() && *states_start + states.len() / *width == start_pos;
            if contiguous {
                let last = states.split_off(states.len() - *width);
                *states = last;
                *states_start = start_pos - 1;
            } else {
                states.clear();
                *states_start = start_pos;
            }
            states.extend_from_slice(rows);
        }
    }

    /// The served model's state at the position *before* `pos`, if this
    /// drafter still holds it.
    ///
    /// Position 0 has no predecessor, and the head is trained against a zero
    /// state there — but only when the run actually began at 0. A batch that
    /// starts further in is a reused prefix, and *its* first positions have
    /// a predecessor whose state was simply never computed. The two cases
    /// look alike and are opposite, so they are different answers here
    /// rather than one zero row standing for both.
    fn previous_state(&self, pos: usize) -> Option<PrevState<'_>> {
        let (Drafter::Mtp {
            states,
            states_start,
            width,
            ..
        }
        | Drafter::Assistant {
            states,
            states_start,
            width,
            ..
        }) = self
        else {
            return None;
        };
        if pos == 0 {
            return (*states_start == 0).then_some(PrevState::Zero);
        }
        let row = (pos - 1).checked_sub(*states_start)?;
        states
            .get(row * *width..(row + 1) * *width)
            .map(PrevState::Row)
    }

    /// Runs the head over `tokens` — position `start_pos + i` — for every
    /// position its cache does not already hold, and hands back the last
    /// step. `None` when there was nothing to run.
    ///
    /// Positions whose paired state is missing are not run at all: the head
    /// holds them as placeholders (`MtpHead::forward` fills the gap) and
    /// keeps them out of every window. Feeding them a zero state instead
    /// would put a row that means nothing where the model expects one that
    /// means something, and the only symptom would be a lower acceptance
    /// rate.
    fn catch_up(&mut self, tokens: &[u32], start_pos: usize) -> Result<Option<MtpStep>> {
        if !self.wants_states() {
            return Ok(None);
        }
        let end = start_pos + tokens.len();
        let from = self.mtp_committed().max(start_pos);
        if from >= end {
            return Ok(None);
        }
        let first = (from..end).find(|&pos| self.previous_state(pos).is_some());
        let Some(first) = first else {
            // A whole batch with no state behind it: hold it all as
            // placeholders and wait for one that has.
            if let Drafter::Mtp { committed, .. } = self {
                *committed = end;
            }
            return Ok(None);
        };

        let width = match self {
            Drafter::Mtp { width, .. } => *width,
            _ => unreachable!("guarded by wants_states"),
        };
        let mut rows = Vec::with_capacity((end - first) * width);
        for pos in first..end {
            match self.previous_state(pos) {
                Some(PrevState::Row(row)) => rows.extend_from_slice(row),
                Some(PrevState::Zero) => rows.resize(rows.len() + width, 0.0),
                // A hole *after* the head has started caching is a
                // bookkeeping bug, not a reused prefix — and a silent one,
                // since a head fed the wrong row still drafts fluently.
                None => anyhow::bail!(
                    "the served model's state for position {} is missing mid-batch",
                    pos - 1
                ),
            }
        }

        let Drafter::Mtp {
            head,
            cache,
            committed,
            first_pos,
            ..
        } = self
        else {
            unreachable!("guarded by wants_states")
        };
        anyhow::ensure!(
            first_pos.is_none() || first == from,
            "the draft head's cache would gain a hole at position {from}"
        );
        let window_from = *first_pos.get_or_insert(first);
        let step = head.forward(
            cache,
            &tokens[first - start_pos..],
            &rows,
            first,
            window_from,
        )?;
        *committed = end;
        Ok(Some(step))
    }

    /// One chained draft step: the head reads its own last guess paired with
    /// its own state, one position past everything it has cached.
    fn mtp_step(&mut self, token: u32, state: &[f32]) -> Result<MtpStep> {
        let Drafter::Mtp {
            head,
            cache,
            committed,
            first_pos,
            ..
        } = self
        else {
            unreachable!("only reached from the Mtp arm")
        };
        let step = head.forward(cache, &[token], state, *committed, first_pos.unwrap_or(0))?;
        *committed += 1;
        Ok(step)
    }

    /// The most tokens [`Self::draft`] returns.
    fn max_draft(&self) -> usize {
        match self {
            Drafter::PromptLookup {
                max_draft,
                echo_draft,
                ..
            } => (*max_draft).max(*echo_draft),
            Drafter::Mtp { .. } => self.max_tokens(),
            Drafter::Assistant { tokens, .. } | Drafter::Model { tokens, .. } => *tokens,
        }
    }

    /// What the acceptance log calls this drafter.
    fn label(&self) -> &'static str {
        match self {
            Drafter::PromptLookup { .. } => "prompt-lookup",
            Drafter::Model { .. } => "draft model",
            Drafter::Mtp { .. } | Drafter::Assistant { .. } => "mtp head",
        }
    }
}

/// What sits behind the token a draft step reads — see
/// [`Drafter::previous_state`].
enum PrevState<'s> {
    /// The run began at position 0, where the head is trained against a
    /// state of zeros because there is no earlier position to have one.
    Zero,
    Row(&'s [f32]),
}

/// Runs the draft model forward and returns the last position's logits.
///
/// **The target's own calls, in the target's own order**: the prompt through
/// `forward`, as prefill runs it, and every step after it through
/// [`ModelForward::forward_verify`] — one token or a catch-up alike.
/// A draft cache is rolled back to what the target accepted and run again on
/// every step, which is the verifier's cache's life too, so it must be kept by
/// the same path the verifier keeps its own. A single-token `forward` alone is
/// not that path: on a backend with a fused decode step it writes the key and
/// value into the device mirror only (`KvCache::advance_gpu_only`), and an
/// architecture whose `forward_verify` is the host's `forward_all_logits`
/// would then read rows that were never written — one row past the end of
/// the host buffer, or, where the buffer was long enough, zeros and an
/// acceptance rate of zero from a model drafting for *itself*.
///
/// Chunked for the same reason [`prefill`] is: the first call carries the
/// whole prompt, and one large submission is what resets the device.
fn draft_forward(
    model: &dyn ModelForward,
    cache: &mut KvCache,
    tokens: &[u32],
    start_pos: usize,
    draft_tokens: usize,
    slot_id: usize,
) -> Result<Vec<f32>> {
    let mut last = Vec::new();
    if start_pos == 0 {
        // The draft model runs the same forward passes over the same prompt,
        // so it wants the same width for the same reasons.
        let batch = Chunking::for_prompt(tokens.len()).width;
        let chunk = if batch == 0 { tokens.len() } else { batch }.max(1);
        for (i, part) in tokens.chunks(chunk).enumerate() {
            last = model.forward(cache, part, i * chunk, slot_id)?;
        }
        return Ok(last);
    }
    // A catch-up longer than a verify (decoding went on undrafted for a
    // while) is verified a draft's width at a time.
    let chunk = draft_tokens + 1;
    for (i, part) in tokens.chunks(chunk).enumerate() {
        last = model
            .forward_verify(cache, part, start_pos + i * chunk, slot_id)?
            .pop()
            .ok_or_else(|| anyhow::anyhow!("the draft model returned no logits"))?;
    }
    Ok(last)
}

/// The longest trailing context prompt lookup compares.
const NGRAM_LONGEST: usize = 64;

/// Prompt-lookup draft: the continuation of the earlier occurrence of
/// `history`'s trailing context that matches it furthest back, at least
/// `shortest` tokens and at most [`NGRAM_LONGEST`], the most recent of equal
/// ones — a zero-cost (no model call) guess at what comes next, which pays
/// off whenever the output echoes the context (code, quotations,
/// structured/repetitive text). A longer matching context is the better
/// predictor: in a copied list every item ends the same way, and only the
/// context back to the item before tells the place being copied from the
/// copy just written. Empty when there's no match.
fn ngram_draft(history: &[u32], shortest: usize, max_draft: usize) -> Vec<u32> {
    ngram_match(history, shortest)
        .map(|(_, from)| {
            // A picture's rows are never a guess at the reply.
            history[from..(from + max_draft).min(history.len())]
                .iter()
                .take_while(|&&t| !super::vision::is_virtual(t))
                .copied()
                .collect()
        })
        .unwrap_or_default()
}

/// The trailing context matched at least this far back makes a draft an
/// echo in progress ([`Drafter::echoes`]): prose repeats a phrase of a few
/// tokens, a copy runs on for dozens.
const ECHO_CONTEXT: usize = 16;

/// Whether `history` is copying an earlier passage: its trailing context
/// matches [`ECHO_CONTEXT`] tokens back, with something left to copy.
fn echoing(history: &[u32], ngram: usize) -> bool {
    ngram_match(history, ngram)
        .is_some_and(|(len, from)| len >= ECHO_CONTEXT && from < history.len())
}

/// [`ngram_draft`]'s match: how many tokens of context it matched, and
/// where in `history` the continuation it copies starts.
fn ngram_match(history: &[u32], shortest: usize) -> Option<(usize, usize)> {
    let shortest = shortest.max(1);
    if history.len() <= shortest {
        return None;
    }
    let end = history.len();
    let suffix = &history[end - shortest..];
    // `(context length, start of the continuation)`.
    let mut best: Option<(usize, usize)> = None;
    for start in (0..end - shortest).rev() {
        if &history[start..start + shortest] != suffix {
            continue;
        }
        let from = start + shortest;
        let longest = NGRAM_LONGEST.min(from);
        let mut len = shortest;
        while len < longest && history[from - len - 1] == history[end - len - 1] {
            len += 1;
        }
        if best.is_none_or(|(l, _)| len > l) {
            best = Some((len, from));
            if len == longest {
                break;
            }
        }
    }
    best
}

/// One speculative step. Takes a draft from `drafter`, verifies the whole of
/// it in a single multi-position forward (as rows of the batched step, with
/// `batch`), keeps the longest prefix the model
/// would itself have produced greedily, and rolls the rejected tail off both
/// KV caches. Returns the model's own next token (identical to what plain
/// greedy decoding produces here) with the number of tokens the step
/// yielded, and pushes any further accepted tokens onto `spec_buf` for the
/// loop to emit before its next forward; `None` when there was no draft to
/// verify, and the loop decodes the token instead.
/// `accepted`/`steps` accumulate acceptance stats for the final log.
///
/// A draft token is accepted only when it equals the sampler's own pick at
/// that position, and that pick depends on the position alone
/// (`sampling::position_draw`), so the emitted sequence is what decoding
/// without a draft would emit, greedy or sampled.
#[allow(clippy::too_many_arguments)]
fn speculative_next(
    model: &dyn ModelForward,
    cache: &mut KvCache,
    sampler: &mut Sampler,
    history: &[u32],
    current: u32,
    start_pos: usize,
    capacity: usize,
    drafter: &mut Drafter<'_>,
    slot_id: usize,
    draft_len: usize,
    spec_buf: &mut VecDeque<u32>,
    accepted: &mut usize,
    steps: &mut usize,
    batch: Option<(
        &super::decode_batch::DecodeBatcher,
        &super::decode_batch::Registration<'_>,
    )>,
) -> Result<Option<(u32, usize)>> {
    // `current` takes the first of the remaining rows; the draft gets the
    // rest, up to the step's `draft_len`.
    let room = capacity.saturating_sub(start_pos + 1).min(draft_len);
    let draft = drafter.draft(history, room, cache, slot_id)?;
    // Nothing to verify: the plain decode step is the cheaper way to the
    // same token. A head-based drafter still needs this pass's states.
    if draft.is_empty() && !drafter.verifies_with_states() {
        return Ok(None);
    }
    let mut input = Vec::with_capacity(1 + draft.len());
    input.push(current);
    input.extend_from_slice(&draft);
    // A recurrent state takes in every token the verify runs, so it is saved
    // here, at the anchor, for a rejected draft to return to.
    let checkpoint = cache
        .has_recurrent()
        .then(|| cache.rollback_point(start_pos))
        .transpose()?;

    // Per-position logits for `current` and every drafted token, from one
    // forward that appends all of them to the cache. A head-based drafter
    // also needs each of those positions' hidden state, and takes it from
    // the same pass — the next draft reads the state behind whichever token
    // the verification below goes on to accept last, and only this forward
    // ever computes it.
    let logits = if drafter.verifies_with_states() {
        let (logits, states) =
            model.forward_all_logits_with_states(cache, &input, start_pos, slot_id)?;
        drafter.observe_states(start_pos, &states);
        logits
    } else {
        // Every non-empty draft verified at the step's full width (one of
        // two: the drafter's own and its echo width): a device keeps
        // resources per pass width, and a draft cut short near the end of
        // the history would otherwise add a width — and its share of device
        // memory — for each length it happened to have. The padding rows are computed, never read, and rolled off
        // with the rejected tail below.
        // Not for a recurrent cache: a padding row would step its state too,
        // and every verify that padded would have to be run again.
        let real = input.len();
        let width = 1 + room;
        if real > 1 && real < width && checkpoint.is_none() {
            input.resize(width, current);
        }
        // A slot in the batch verifies as rows of the batched step, so the
        // other slots step with it rather than wait for it.
        let mut logits = match batch {
            Some((batcher, registration)) => {
                batcher.verify(Some(registration), model, cache, &input, start_pos, slot_id)?
            }
            None => model.forward_verify(cache, &input, start_pos, slot_id)?,
        };
        logits.truncate(real);
        input.truncate(real);
        logits
    };

    // Verify with the request's own sampler, row by row. `recent` mirrors
    // what `history` would be as accepted tokens are committed, so the repeat
    // penalty (if any) sees exactly the context plain decoding would, and
    // row `m` is sampled for the position it fills — `start_pos + 1 + m` —
    // whose draw depends on nothing else (`sampling::position_draw`). A
    // constraint advances once per pick, so at row `m` it has seen exactly
    // the tokens committed before it. So the token chosen at every position
    // is the one plain decoding would choose there, sampled, greedy or
    // constrained, and a draft is accepted exactly as far as it agrees with
    // it.
    let rl = sampler.repeat_last_n();
    let mut recent: Vec<u32> = history[history.len().saturating_sub(rl)..].to_vec();
    let mut chosen = vec![sampler.sample(&logits[0], &recent, start_pos + 1)];
    recent.push(chosen[0]);
    let mut matched = 0usize;
    while matched < draft.len() && draft[matched] == chosen[matched] {
        let next = sampler.sample(&logits[matched + 1], &recent, start_pos + 2 + matched);
        recent.push(next);
        chosen.push(next);
        matched += 1;
    }

    // Keep `current` plus the `matched` accepted drafts; drop the rest. The
    // last element of `chosen` is the model's own token past the accepted
    // prefix — its key/value is not committed (it was never an accepted input),
    // so it becomes the next frontier the loop forwards.
    let committed_tokens = start_pos + chosen.len();
    match &checkpoint {
        // The state has taken in the rejected tokens too: back to the one
        // the verify left behind its last accepted token where the device
        // kept it, otherwise back to the anchor and the accepted ones again
        // — either way it holds exactly what plain decoding would have
        // given it. Their logits are not needed.
        Some(checkpoint) if chosen.len() < input.len() => {
            if !cache.keep_verified(checkpoint, chosen.len()) {
                cache.rollback(checkpoint)?;
                model.forward(cache, &input[..chosen.len()], start_pos, slot_id)?;
            }
        }
        // Every token verified was accepted: the state is already right.
        Some(_) => {}
        None => cache.truncate(committed_tokens)?,
    }
    // The draft's cache follows the target's exactly, which is what keeps the
    // two models looking at the same context. A drafter with no cache of its
    // own ignores this.
    drafter.commit(committed_tokens)?;
    *accepted += matched;
    *steps += 1;
    for &t in &chosen[1..] {
        spec_buf.push_back(t);
    }
    Ok(Some((chosen[0], chosen.len())))
}

#[cfg(test)]
mod message_header_tests {
    use super::{MessageHeader, REASONING_RECIPIENT};

    const START: u32 = 1;
    const MESSAGE: u32 = 2;
    const EOM: u32 = 3;
    /// The body-kind markers of the second format (`inkling`'s
    /// `<|content_thinking|>` and `<|content_text|>`).
    const THINKING: u32 = 4;
    const TEXT: u32 = 5;
    /// The framing of the third format (gemma-4's `<|channel>` /
    /// `<channel|>`).
    const CHANNEL_OPEN: u32 = 6;
    const CHANNEL_CLOSE: u32 = 7;

    /// One thing the model produced: a structural marker, or a run of
    /// ordinary text.
    enum Out {
        Marker(u32),
        Text(&'static str),
    }
    use Out::{Marker, Text};

    /// Everything a client would see for `stream`, given a reply that
    /// resumes inside a header (the ordinary generation prompt) and a role
    /// that either shows reasoning or suppresses it.
    fn shown(suppress_reasoning: bool, resumes_in_header: bool, stream: &[Out]) -> String {
        let mut header = MessageHeader {
            framing: Some((START, MESSAGE)),
            inside: resumes_in_header,
            suppress_reasoning,
            ..Default::default()
        };
        let mut out = String::new();
        for item in stream {
            match item {
                Marker(id) => {
                    if let Some(separator) = header.observe_marker(*id) {
                        out.push_str(separator);
                    }
                }
                Text(text) => {
                    if let Some(text) = header.observe_text((*text).to_string()) {
                        out.push_str(&text);
                    }
                }
            }
        }
        out
    }

    /// A whole `muse-glimmer` assistant turn: a reasoning message, then the
    /// answer. Every header is dropped — this is the bug that made replies
    /// begin `" to=self"` — and the two bodies are separated rather than
    /// run together (`…Final.Three primes larger than…`).
    #[test]
    fn a_showing_role_gets_both_messages_with_their_headers_dropped() {
        let turn = [
            Text(" to=self"),
            Marker(MESSAGE),
            Text("Let me think."),
            Marker(EOM),
            Marker(START),
            Text(" to=user"),
            Marker(MESSAGE),
            Text("23, 29, 31."),
        ];
        assert_eq!(shown(false, true, &turn), "Let me think.\n\n23, 29, 31.");
    }

    /// The same turn under a reasoning-suppressing role: the answer alone,
    /// with no separator in front of it (there is nothing to separate).
    #[test]
    fn a_suppressing_role_gets_only_the_message_addressed_to_the_caller() {
        let turn = [
            Text(" to=self"),
            Marker(MESSAGE),
            Text("Let me think."),
            Marker(EOM),
            Marker(START),
            Text(" to=user"),
            Marker(MESSAGE),
            Text("23, 29, 31."),
        ];
        assert_eq!(shown(true, true, &turn), "23, 29, 31.");
    }

    /// The recipient arrives one token at a time and need not land as a
    /// single piece, so the test that reads it must not assume it did.
    #[test]
    fn a_recipient_split_across_tokens_is_still_recognized() {
        let turn = [
            Text(" to"),
            Text("="),
            Text("se"),
            Text("lf"),
            Marker(MESSAGE),
            Text("thinking"),
        ];
        assert_eq!(shown(true, true, &turn), "");
        assert_eq!(shown(false, true, &turn), "thinking");
    }

    /// A prompt that does *not* end in a header — a prefilled partial reply,
    /// or any raw-completion prompt — resumes inside a body, and hiding its
    /// first tokens would swallow the start of the reply.
    #[test]
    fn a_reply_resuming_inside_a_body_hides_nothing() {
        assert_eq!(
            shown(false, false, &[Text("already answering")]),
            "already answering"
        );
        assert_eq!(
            shown(true, false, &[Text("already answering")]),
            "already answering"
        );
    }

    /// Every vocabulary without both markers keeps exactly its previous
    /// behavior: nothing is hidden and nothing is inserted, whatever the
    /// model emits and whatever the role is.
    #[test]
    fn a_vocabulary_without_the_markers_is_unaffected() {
        let mut header = MessageHeader {
            suppress_reasoning: true,
            ..Default::default()
        };
        assert_eq!(header.observe_marker(START), None);
        assert_eq!(header.observe_marker(MESSAGE), None);
        assert_eq!(
            header.observe_text("plain".to_string()),
            Some("plain".to_string())
        );
    }

    /// Everything a client would see from a gemma-4 reply, where a side
    /// channel is framed by `<|channel>` / `<channel|>` and names itself in
    /// the text right after the opener.
    fn shown_in_channels(suppress_reasoning: bool, stream: &[Out]) -> String {
        let mut header = MessageHeader {
            channel: Some((CHANNEL_OPEN, CHANNEL_CLOSE)),
            suppress_reasoning,
            ..Default::default()
        };
        let mut out = String::new();
        for item in stream {
            match item {
                Marker(id) => {
                    if let Some(separator) = header.observe_marker(*id) {
                        out.push_str(separator);
                    }
                }
                Text(text) => {
                    if let Some(text) = header.observe_text((*text).to_string()) {
                        if header.in_reasoning() {
                            out.push_str(&format!("[{text}]"));
                        } else {
                            out.push_str(&text);
                        }
                    }
                }
            }
        }
        out
    }

    /// The framing of the fourth format (`<think>` / `</think>`).
    const THINK_OPEN: u32 = 8;
    const THINK_CLOSE: u32 = 9;

    /// Everything a client would see for a `<think>`-marking model, split
    /// the way the stream splits it: `(reasoning, answer)`.
    ///
    /// `resumes_in_think` is what `prompt_ends_in_think` answers off the
    /// real prompt — true for every ordinary Qwen3.x generation prompt,
    /// which ends with the opening tag already written.
    fn thinking_and_answer(
        suppress_reasoning: bool,
        resumes_in_think: bool,
        stream: &[Out],
    ) -> (String, String) {
        let mut header = MessageHeader {
            think: Some((THINK_OPEN, THINK_CLOSE)),
            reasoning_body: resumes_in_think,
            hidden_body: resumes_in_think && suppress_reasoning,
            trim_leading: resumes_in_think,
            suppress_reasoning,
            ..Default::default()
        };
        let (mut reasoning, mut answer) = (String::new(), String::new());
        for item in stream {
            let emitted = match item {
                Marker(id) => header.observe_marker(*id).map(str::to_string),
                Text(text) => header.observe_text((*text).to_string()),
            };
            if let Some(text) = emitted {
                if header.in_reasoning() {
                    reasoning.push_str(&text);
                } else {
                    answer.push_str(&text);
                }
            }
        }
        (reasoning, answer)
    }

    /// The whole point: a Qwen3.x turn resumes inside the `<think>` block
    /// its own template opened, so everything up to `</think>` is thinking
    /// and everything after it is the answer. Before this, both arrived as
    /// one run of content and the answer was indistinguishable from the
    /// notes that led to it.
    #[test]
    fn a_think_block_is_separated_from_the_answer() {
        let turn = [
            Text("Let me plan this.\n\ntypedef struct Node { int v; } Node;"),
            Marker(THINK_CLOSE),
            Text("Here is a doubly linked list:\n\n```c\nint main(void){}\n```"),
        ];
        let (reasoning, answer) = thinking_and_answer(false, true, &turn);
        assert_eq!(
            reasoning,
            "Let me plan this.\n\ntypedef struct Node { int v; } Node;"
        );
        assert_eq!(
            answer,
            "Here is a doubly linked list:\n\n```c\nint main(void){}\n```"
        );
    }

    /// The blank lines the template writes around its markers do not
    /// survive into either pane. Qwen3.x opens with `<think>\n` and closes
    /// with `</think>\n\n`: the tags are hidden and the newlines are not,
    /// so before this every answer began with a blank line and every chain
    /// of thought with a newline — invisible in a rendered bubble,
    /// perfectly visible in `content`.
    #[test]
    fn the_whitespace_around_the_markers_is_trimmed() {
        let turn = [
            Text("\nThinking."),
            Marker(THINK_CLOSE),
            Text("\n\n"),
            Text("Answer."),
        ];
        assert_eq!(
            thinking_and_answer(false, true, &turn),
            ("Thinking.".to_string(), "Answer.".to_string())
        );
    }

    /// No separator is inserted where `</think>` was. The two sides leave
    /// as different events, so nothing downstream runs them together — and
    /// a leading blank line is visible whitespace once the answer is shown
    /// on its own.
    #[test]
    fn no_separator_is_owed_across_a_think_boundary() {
        let turn = [Text("Thinking."), Marker(THINK_CLOSE), Text("Answer.")];
        assert_eq!(
            thinking_and_answer(false, true, &turn),
            ("Thinking.".to_string(), "Answer.".to_string())
        );
    }

    /// A reasoning-suppressing role (`--review`) drops the body outright
    /// rather than merely labelling it — the `<think>` prefill usually
    /// means there is none to drop, but a model that opens one anyway must
    /// not have it reach the caller.
    #[test]
    fn a_suppressing_role_drops_a_think_block() {
        let turn = [
            Marker(THINK_OPEN),
            Text("Second-guessing myself."),
            Marker(THINK_CLOSE),
            Text("Answer."),
        ];
        assert_eq!(
            thinking_and_answer(true, false, &turn),
            (String::new(), "Answer.".to_string())
        );
    }

    /// A model that writes its own opening tag (no prefill in the prompt)
    /// is split on the tags it wrote, not on where the prompt ended.
    #[test]
    fn a_self_opened_think_block_is_split_too() {
        let turn = [
            Text("Sure!"),
            Marker(THINK_OPEN),
            Text("Actually, let me check."),
            Marker(THINK_CLOSE),
            Text("Yes."),
        ];
        let (reasoning, answer) = thinking_and_answer(false, false, &turn);
        assert_eq!(reasoning, "Actually, let me check.");
        assert_eq!(answer, "Sure!Yes.");
    }

    /// A vocabulary with no think framing is untouched: every byte is the
    /// answer, exactly as before.
    #[test]
    fn a_vocabulary_without_think_tokens_is_unaffected() {
        let mut header = MessageHeader::default();
        assert!(!header.in_reasoning());
        assert_eq!(
            header.observe_text("plain answer".to_string()),
            Some("plain answer".to_string())
        );
        assert!(!header.in_reasoning());
    }

    /// The channel *name* is framing and never reaches the reader, whatever
    /// the role. This is the bug that ended replies with a bare
    /// `"tool_code\nprint(create_directory(path='pacman_game'))"`.
    #[test]
    fn a_channel_name_is_never_shown() {
        let turn = [
            Text("Here is the plan."),
            Marker(CHANNEL_OPEN),
            Text("tool_code\nprint(create_directory(path='x'))"),
            Marker(CHANNEL_CLOSE),
        ];
        assert_eq!(
            shown_in_channels(false, &turn),
            "Here is the plan.\n\nprint(create_directory(path='x'))"
        );
        assert_eq!(shown_in_channels(true, &turn), "Here is the plan.");
    }

    /// A `thought` channel's body is reasoning, so it follows the same rule
    /// as the other formats' — marked as reasoning (bracketed here) apart
    /// from the answer, or dropped entirely when the role suppresses it.
    #[test]
    fn a_channel_body_follows_the_reasoning_rule() {
        let turn = [
            Marker(CHANNEL_OPEN),
            Text("thought\nLet me think."),
            Marker(CHANNEL_CLOSE),
            Text("23, 29, 31."),
        ];
        assert_eq!(
            shown_in_channels(false, &turn),
            "[Let me think.]23, 29, 31."
        );
        assert_eq!(shown_in_channels(true, &turn), "23, 29, 31.");
    }

    /// The name arrives one token at a time like everything else, and the
    /// token that ends it can carry the first of the body with it.
    #[test]
    fn a_channel_name_split_across_tokens_is_still_dropped() {
        let turn = [
            Marker(CHANNEL_OPEN),
            Text("tho"),
            Text("ught"),
            Text("\nLet me"),
            Text(" think."),
            Marker(CHANNEL_CLOSE),
            Text("Done."),
        ];
        assert_eq!(shown_in_channels(false, &turn), "[Let me][ think.]Done.");
    }

    /// A channel whose name never ends must not swallow the rest of the
    /// reply: past any name's length, what was withheld is content. Driven
    /// directly rather than through the helper, because the text it needs is
    /// longer than a literal worth writing out.
    #[test]
    fn an_unterminated_channel_name_releases_what_it_withheld() {
        let mut header = MessageHeader {
            channel: Some((CHANNEL_OPEN, CHANNEL_CLOSE)),
            ..Default::default()
        };
        assert_eq!(header.observe_marker(CHANNEL_OPEN), None);
        let long = "x".repeat(super::MAX_HEADER_LEN + 1);
        assert_eq!(header.observe_text(long.clone()), Some(long));
    }

    /// A reply that stops right after a channel gets no separator hung off
    /// the end of it.
    #[test]
    fn a_reply_ending_in_a_channel_has_no_trailing_separator() {
        let turn = [
            Text("Answer."),
            Marker(CHANNEL_OPEN),
            Text("thought\nafterthought"),
            Marker(CHANNEL_CLOSE),
        ];
        assert_eq!(shown_in_channels(true, &turn), "Answer.");
    }

    /// Everything a client would see from an `inkling`-style reply, where
    /// each body is opened by a marker naming its kind and there is no
    /// header at all.
    fn shown_by_kind(suppress_reasoning: bool, stream: &[Out]) -> String {
        let mut header = MessageHeader {
            kinds: Some(crate::engine::tokenizer::ContentKinds {
                reasoning: THINKING,
                other: vec![TEXT],
            }),
            suppress_reasoning,
            ..Default::default()
        };
        let mut out = String::new();
        for item in stream {
            match item {
                Marker(id) => {
                    if let Some(separator) = header.observe_marker(*id) {
                        out.push_str(separator);
                    }
                }
                Text(text) => {
                    if let Some(text) = header.observe_text((*text).to_string()) {
                        out.push_str(&text);
                    }
                }
            }
        }
        out
    }

    /// A whole `inkling` assistant turn: a thinking body, then the answer.
    /// The showing role gets both, separated; the suppressing role gets the
    /// answer alone, with no leading separator.
    #[test]
    fn a_typed_body_is_shown_or_hidden_by_its_own_marker() {
        let turn = [
            Marker(THINKING),
            Text("Let me think."),
            Marker(EOM),
            Marker(TEXT),
            Text("23, 29, 31."),
        ];
        assert_eq!(shown_by_kind(false, &turn), "Let me think.\n\n23, 29, 31.");
        assert_eq!(shown_by_kind(true, &turn), "23, 29, 31.");
    }

    /// The markers that are *not* body kinds — a turn marker, an end
    /// marker — must not be read as the start of a visible body. Taking
    /// `<|end_message|>` for one would un-hide the reasoning that follows
    /// it under a suppressing role.
    #[test]
    fn a_marker_that_is_not_a_body_kind_does_not_reopen_a_body() {
        let turn = [
            Marker(THINKING),
            Text("first thought."),
            Marker(EOM),
            Marker(START),
            Text("second thought."),
        ];
        assert_eq!(shown_by_kind(true, &turn), "");
        assert_eq!(shown_by_kind(false, &turn), "first thought.second thought.");
    }

    /// A reply that never writes `<|message|>` must not be swallowed. The
    /// prompt still ends in a header, so the filter starts by withholding —
    /// but only up to `MAX_HEADER_LEN`, after which it concludes this is
    /// not a header and releases what it held.
    ///
    /// The failure this guards is not hypothetical: appending an empty
    /// `<think>` block to a `<|start|>assistant` prompt (the old blanket
    /// reasoning-suppression prefill) produced exactly such a reply, and
    /// with no bound the whole answer came back empty.
    #[test]
    fn a_reply_that_never_leaves_the_header_is_released_rather_than_swallowed() {
        let long = "x".repeat(super::MAX_HEADER_LEN + 1);
        let stream = [Text(Box::leak(long.into_boxed_str()) as &'static str)];
        let out = shown(false, true, &stream);
        assert!(
            out.len() > super::MAX_HEADER_LEN,
            "withheld text must be released, got {} chars",
            out.len()
        );
        assert!(out.chars().all(|c| c == 'x'));
    }

    /// The recipient this module keys on is the one the format writes; a
    /// header naming anyone else is an ordinary message, shown under every
    /// role.
    #[test]
    fn only_the_reasoning_recipient_is_suppressed() {
        assert_eq!(REASONING_RECIPIENT, "to=self");
        let turn = [Text(" to=user"), Marker(MESSAGE), Text("the answer")];
        assert_eq!(shown(true, true, &turn), "the answer");
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn a_chunk_never_ends_inside_a_picture() {
        let v = super::super::vision::FIRST_ID;
        // Text, a picture of four rows, text.
        let tokens = [1, 2, v, v + 1, v + 2, v + 3, 3, 4];
        // Ending inside the picture: cut before it.
        assert_eq!(keep_pictures_whole(&tokens, 4, usize::MAX), 2);
        // Ending right after it, or before it: unchanged.
        assert_eq!(keep_pictures_whole(&tokens, 6, usize::MAX), 6);
        assert_eq!(keep_pictures_whole(&tokens, 2, usize::MAX), 2);
        // Starting with it: run to its end, within the limit.
        assert_eq!(keep_pictures_whole(&tokens[2..], 2, usize::MAX), 4);
        assert_eq!(keep_pictures_whole(&tokens[2..], 2, 3), 3);
        // A virtual id is a picture row whatever its value.
        assert!(!super::super::vision::is_virtual(v - 1));
    }
    use super::*;
    use crate::engine::kv_cache::KvCache;
    use crate::engine::loader::{ModelConfig, PoolingType};
    use crate::engine::scheduler::SlotPool;
    use orangu::gguf::{GgufFile, GgufValue};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A deterministic, model-math-free `ModelForward`: each position's
    /// key/value is a pure function of `(token, position)`, and the
    /// returned logits are a pure function of every cached key so far —
    /// so whether an earlier position's key was computed by *this* call or
    /// copied in from a previous request's cache
    /// (`KvCache::copy_prefix_from`) can't matter, exactly the property
    /// `prefix_cache_reuse_matches_a_full_recompute` below needs to isolate
    /// prefix reuse's own correctness from any real model's floating-point
    /// non-associativity across different batch shapes (a separate,
    /// already-present property of the real GPU backends, not something
    /// this module's own plumbing introduces).
    struct DeterministicModel {
        config: ModelConfig,
        /// Total tokens ever passed to `forward` — lets a test confirm
        /// prefix reuse actually skipped work, not just that it didn't
        /// change the result.
        forwarded_tokens: AtomicUsize,
        /// Disagree with the base rule on every `n`th call, so an instance
        /// can stand in for a *draft* model: one that mostly predicts what
        /// the target would say and sometimes does not. `0` never disagrees,
        /// which makes the draft a perfect oracle.
        ///
        /// Counted per call rather than per token so the disagreement lands
        /// at positions the caller cannot arrange for, which is the point —
        /// a draft that failed only where a test expected it to would not
        /// exercise the rollback at all.
        disagree_every: usize,
        calls: AtomicUsize,
        /// Entries into `forward`/`forward_all_logits`, however many tokens
        /// each carried. This — not the token count — is what speculation
        /// reduces: a verification forward is one call carrying the whole
        /// draft, where plain decoding would have made one call per token.
        forward_calls: AtomicUsize,
        /// Carry one recurrent layer beside the attention layer, the shape of
        /// a hybrid model: every token is added into its state, and the
        /// state weighs into the winner. A cache whose state saw other tokens
        /// than its attention rows then answers differently.
        recurrent: bool,
        /// How long each forward takes, so requests run side by side
        /// overlap.
        pause: Duration,
        /// Forwards at or past `meet_position` wait until this many requests
        /// reach that position, regardless of how their threads are scheduled.
        meet: usize,
        meet_position: usize,
        started: AtomicUsize,
    }

    impl DeterministicModel {
        fn new(n_vocab: usize) -> Self {
            Self {
                config: ModelConfig {
                    architecture: "test".to_string(),
                    n_vocab,
                    n_embd: 1,
                    n_layer: 1,
                    n_head: 1,
                    n_head_kv: 1,
                    head_dim: 4,
                    n_ctx_train: 1000,
                    rope_dim: 1,
                    rope_freq_base: 10000.0,
                    rms_eps: 1e-6,
                    pooling_type: PoolingType::Mean,
                },
                forwarded_tokens: AtomicUsize::new(0),
                disagree_every: 0,
                calls: AtomicUsize::new(0),
                forward_calls: AtomicUsize::new(0),
                recurrent: false,
                pause: Duration::ZERO,
                meet: 0,
                meet_position: 0,
                started: AtomicUsize::new(0),
            }
        }

        /// The same model with a recurrent layer — see `recurrent`.
        fn hybrid(n_vocab: usize) -> Self {
            Self {
                recurrent: true,
                ..Self::new(n_vocab)
            }
        }

        /// Pushes `token` at `pos` into every layer this model has.
        fn absorb(&self, cache: &mut KvCache, token: u32, pos: usize) {
            let val = token as f32 * 1000.0 + pos as f32;
            cache.layers[0].push(&[val], &[val]);
            if self.recurrent {
                cache.recurrent[0].delta_states_mut().0[0] += token as f32 + 1.0;
            }
        }

        /// The same model, wrong every `n`th call — a stand-in for a draft
        /// model that is usually but not always right.
        fn drafting(n_vocab: usize, disagree_every: usize) -> Self {
            Self {
                disagree_every,
                ..Self::new(n_vocab)
            }
        }

        /// The winner this model's rule picks for a cache in its current
        /// state, with the deliberate disagreement applied.
        fn winner(&self, cache: &mut KvCache) -> usize {
            let layer = &cache.layers[0];
            let mut acc = 0f32;
            for p in 0..layer.len {
                acc += layer.key_at(p, 0, 1)[0];
            }
            if self.recurrent {
                acc += 7.0 * cache.recurrent[0].delta_states_mut().0[0];
            }
            let base = (acc.abs() as u64 as usize) % self.config.n_vocab;
            let call = self.calls.fetch_add(1, Ordering::Relaxed);
            if self.disagree_every > 0 && call.is_multiple_of(self.disagree_every) {
                (base + 1) % self.config.n_vocab
            } else {
                base
            }
        }

        fn logits_for(&self, winner: usize) -> Vec<f32> {
            // Soft enough that a sampled request does not always take the
            // winner — about two in three at temperature 0.8.
            let mut logits = vec![0f32; self.config.n_vocab];
            logits[winner] = 2.0;
            logits
        }
    }

    impl ModelForward for DeterministicModel {
        fn config(&self) -> &ModelConfig {
            &self.config
        }

        fn new_kv_cache(&self, capacity: usize) -> KvCache {
            if self.recurrent {
                KvCache::new_mixed(
                    capacity,
                    &[1],
                    &[crate::engine::kv_cache::RecurrentSpec::delta_net(
                        0, 0, 1, 1,
                    )],
                )
            } else {
                KvCache::new(1, capacity, 1)
            }
        }

        fn forward(
            &self,
            cache: &mut KvCache,
            tokens: &[u32],
            start_pos: usize,
            _slot_id: usize,
        ) -> Result<Vec<f32>> {
            self.forwarded_tokens
                .fetch_add(tokens.len(), Ordering::Relaxed);
            self.forward_calls.fetch_add(1, Ordering::Relaxed);
            std::thread::sleep(self.pause);
            for (i, &t) in tokens.iter().enumerate() {
                self.absorb(cache, t, start_pos + i);
            }
            // Count completed forwards so every request holds its pages
            // before the rendezvous releases either request.
            if start_pos == self.meet_position {
                self.started.fetch_add(1, Ordering::AcqRel);
            }
            if start_pos >= self.meet_position {
                let deadline = Instant::now() + Duration::from_secs(10);
                while self.started.load(Ordering::Acquire) < self.meet && Instant::now() < deadline
                {
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
            let winner = self.winner(cache);
            Ok(self.logits_for(winner))
        }

        /// Per-position logits, each identical to what `forward` would have
        /// returned had the tokens been fed one at a time — the property
        /// speculative verification is built on, and the reason this is
        /// written as the same incremental rule rather than a batched one.
        fn forward_all_logits(
            &self,
            cache: &mut KvCache,
            tokens: &[u32],
            start_pos: usize,
            _slot_id: usize,
        ) -> Result<Vec<Vec<f32>>> {
            self.forwarded_tokens
                .fetch_add(tokens.len(), Ordering::Relaxed);
            self.forward_calls.fetch_add(1, Ordering::Relaxed);
            let mut out = Vec::with_capacity(tokens.len());
            for (i, &t) in tokens.iter().enumerate() {
                self.absorb(cache, t, start_pos + i);
                let winner = self.winner(cache);
                out.push(self.logits_for(winner));
            }
            Ok(out)
        }

        fn forward_hidden_states(&self, _tokens: &[u32]) -> Result<Vec<f32>> {
            unimplemented!("not exercised by this test")
        }
    }

    /// A minimal real `Tokenizer` (plain single-letter tokens, `"llama"`
    /// vocab kind so `decode` needs no byte-mapping table) — only
    /// `Tokenizer::decode` is exercised by `run`, to turn each sampled
    /// token id back into the streamed text this test compares.
    fn letter_tokenizer(n_vocab: usize) -> Tokenizer {
        let tokens: Vec<GgufValue> = (0..n_vocab)
            .map(|i| GgufValue::String(char::from_u32('a' as u32 + i as u32).unwrap().to_string()))
            .collect();
        let gguf = GgufFile {
            version: 3,
            metadata: vec![
                (
                    "tokenizer.ggml.tokens".to_string(),
                    GgufValue::Array(tokens),
                ),
                (
                    "tokenizer.ggml.model".to_string(),
                    GgufValue::String("llama".to_string()),
                ),
            ],
            tensors: vec![],
            alignment: 32,
            data_offset: 0,
        };
        Tokenizer::from_gguf(&gguf).unwrap()
    }

    /// [`letter_tokenizer`], but every token is a CONTROL token.
    ///
    /// Which makes every one of them *suppressed*: nothing reaches the stream,
    /// exactly as a chat format's structural markers and a hidden reasoning
    /// body do. That is the shape in which "stop when the client goes away"
    /// used to fail, so it is the shape a test for it has to use.
    fn suppressed_tokenizer(n_vocab: usize) -> Tokenizer {
        let tokens: Vec<GgufValue> = (0..n_vocab)
            .map(|i| GgufValue::String(char::from_u32('a' as u32 + i as u32).unwrap().to_string()))
            .collect();
        let gguf = GgufFile {
            version: 3,
            metadata: vec![
                (
                    "tokenizer.ggml.tokens".to_string(),
                    GgufValue::Array(tokens),
                ),
                (
                    "tokenizer.ggml.model".to_string(),
                    GgufValue::String("llama".to_string()),
                ),
                (
                    // 3 is llama.cpp's `LLAMA_TOKEN_TYPE_CONTROL`.
                    "tokenizer.ggml.token_type".to_string(),
                    GgufValue::Array((0..n_vocab).map(|_| GgufValue::I32(3)).collect()),
                ),
            ],
            tensors: vec![],
            alignment: 32,
            data_offset: 0,
        };
        Tokenizer::from_gguf(&gguf).unwrap()
    }

    fn greedy_params() -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            ..SamplingParams::default()
        }
    }

    /// Drains every event `run` already sent (it only returns after
    /// sending `Done`, so nothing is still in flight) into the
    /// concatenated streamed text plus whether it finished without error.
    fn drain(mut rx: UnboundedReceiver<StreamEvent>) -> (String, bool) {
        let mut text = String::new();
        let mut ok = false;
        while let Ok(event) = rx.try_recv() {
            match event {
                StreamEvent::Token(t) | StreamEvent::Reasoning(t) => text.push_str(&t),
                StreamEvent::PromptProgress { .. } | StreamEvent::Timings(_) => {}
                StreamEvent::Done { .. } => ok = true,
                StreamEvent::Error(e) => panic!("unexpected generation error: {e}"),
                StreamEvent::Overloaded => panic!("unexpected refusal: these pools are unbounded"),
            }
        }
        (text, ok)
    }

    fn run_request(
        model: &DeterministicModel,
        tokenizer: &Tokenizer,
        prefix_cache: Option<&PrefixCache>,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
    ) -> (String, bool) {
        run_request_cached(
            model,
            tokenizer,
            prefix_cache,
            prompt_tokens,
            max_tokens,
            true,
        )
    }

    /// `run_request`, with a draft model attached.
    ///
    /// Everything else is identical, which is what makes the comparison
    /// worth anything: speculation is only allowed to change how many
    /// forwards a request costs, never a byte of what it emits.
    fn run_request_drafted(
        model: &DeterministicModel,
        draft: Arc<DeterministicModel>,
        tokenizer: &Tokenizer,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        draft_tokens: usize,
    ) -> (String, bool) {
        let slots = SlotPool::new(1);
        let guard = pollster::block_on(slots.acquire());
        let (tx, rx) = mpsc::unbounded_channel();
        let req = GenerateRequest {
            json_output: false,
            json_schema: None,
            prompt_tokens,
            sampling: greedy_params(),
            max_tokens,
            stop_token_ids: vec![],
            cache_prompt: true,
            id_slot: None,
            timings_per_token: false,
            role: None,
            cache_point: None,
            background: false,
            reasoning_budget: None,
        };
        let draft = DraftModel {
            model: draft,
            tokens: draft_tokens,
            label: "test-draft".to_string(),
        };
        run(
            model,
            tokenizer,
            Some(&draft),
            None,
            None,
            None,
            None,
            &crate::engine::decode_batch::DecodeBatcher::new(),
            &guard,
            req,
            crate::config::Role::default(),
            crate::config::Role::default(),
            false,
            &crate::engine::metrics::ServerMetrics::new(),
            Instant::now(),
            tx,
            None,
            None,
        )
        .unwrap();
        drain(rx)
    }

    /// `run_request` with explicit control over whether the request is allowed
    /// to read a cache (`GenerateRequest::cache_prompt`).
    fn run_request_cached(
        model: &DeterministicModel,
        tokenizer: &Tokenizer,
        prefix_cache: Option<&PrefixCache>,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        cache_prompt: bool,
    ) -> (String, bool) {
        let slots = SlotPool::new(1);
        let guard = pollster::block_on(slots.acquire());
        let (tx, rx) = mpsc::unbounded_channel();
        let req = GenerateRequest {
            json_output: false,
            json_schema: None,
            prompt_tokens,
            sampling: greedy_params(),
            max_tokens,
            stop_token_ids: vec![],
            cache_prompt,
            id_slot: None,
            timings_per_token: false,
            role: None,
            cache_point: None,
            background: false,
            reasoning_budget: None,
        };
        run(
            model,
            tokenizer,
            None,
            None,
            prefix_cache,
            None,
            None,
            &crate::engine::decode_batch::DecodeBatcher::new(),
            &guard,
            req,
            crate::config::Role::default(),
            crate::config::Role::default(),
            false,
            &crate::engine::metrics::ServerMetrics::new(),
            Instant::now(),
            tx,
            None,
            None,
        )
        .unwrap();
        drain(rx)
    }

    /// `run_request` through a paged pool and its index, as a server with the
    /// paged cache on serves every request.
    fn run_request_paged(
        model: &DeterministicModel,
        tokenizer: &Tokenizer,
        paged: &(
            Arc<crate::engine::kv_pool::KvPool>,
            Arc<crate::engine::prefix_index::PrefixIndex>,
        ),
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
    ) -> (String, bool) {
        let slots = SlotPool::new(1);
        let guard = pollster::block_on(slots.acquire());
        run_on_slot(
            model,
            tokenizer,
            paged,
            &crate::engine::decode_batch::DecodeBatcher::new(),
            &guard,
            prompt_tokens,
            max_tokens,
        )
    }

    /// [`run_request_paged`] on a given slot and decode batch, admitted to
    /// the pool as a request is.
    fn run_on_slot(
        model: &DeterministicModel,
        tokenizer: &Tokenizer,
        paged: &(
            Arc<crate::engine::kv_pool::KvPool>,
            Arc<crate::engine::prefix_index::PrefixIndex>,
        ),
        batcher: &crate::engine::decode_batch::DecodeBatcher,
        guard: &crate::engine::scheduler::SlotGuard,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
    ) -> (String, bool) {
        let (tx, rx) = mpsc::unbounded_channel();
        let prompt_pages = wait_for_prompt_pages(&paged.0, prompt_tokens.len(), None, || false);
        let req = GenerateRequest {
            json_output: false,
            json_schema: None,
            prompt_tokens,
            sampling: greedy_params(),
            max_tokens,
            stop_token_ids: vec![],
            cache_prompt: true,
            id_slot: None,
            timings_per_token: false,
            role: None,
            cache_point: None,
            background: false,
            reasoning_budget: None,
        };
        run(
            model,
            tokenizer,
            None,
            None,
            None,
            None,
            Some(paged),
            batcher,
            guard,
            req,
            crate::config::Role::default(),
            crate::config::Role::default(),
            false,
            &crate::engine::metrics::ServerMetrics::new(),
            Instant::now(),
            tx,
            None,
            prompt_pages,
        )
        .unwrap();
        drain(rx)
    }

    /// Two requests whose answers together outgrow the pool both start at
    /// once — each promised its prompt, not its answer cap — and when the
    /// pool runs short the younger gives its pages back, waits for the
    /// older to finish, rebuilds and goes on: both answer exactly as each
    /// does alone, and every page and promise comes back. On a hybrid too,
    /// whose state the rebuild has to recompute rather than adopt.
    #[test]
    fn requests_that_outgrow_the_pool_together_take_turns_and_answer_as_alone() {
        use crate::engine::kv_pool::{KvPool, LayerGeometry, Policy};
        let tokenizer = letter_tokenizer(8);
        let prompts: [Vec<u32>; 2] = [
            vec![1, 2, 3, 4, 5, 6, 7, 1, 2],
            vec![7, 6, 5, 4, 3, 2, 1, 7, 6],
        ];
        const ANSWER: usize = 20;
        for hybrid in [false, true] {
            let make = || {
                let mut model = if hybrid {
                    DeterministicModel::hybrid(8)
                } else {
                    DeterministicModel::new(8)
                };
                model.pause = Duration::from_millis(2);
                model
            };
            let alone: Vec<(String, bool)> = prompts
                .iter()
                .map(|p| run_request(&make(), &tokenizer, None, p.clone(), ANSWER))
                .collect();

            // Both requests hold five pages before either can grow to six.
            // Meeting only at the prompt leaves later decode overlap to the
            // OS scheduler and can let one request finish without any yield.
            let mut model = make();
            model.meet = prompts.len();
            model.meet_position = 16;
            // Ten pages of four tokens: one request of 29 positions needs
            // eight, two together sixteen.
            const PAGE: usize = 4;
            let geom = LayerGeometry::of(&model.new_kv_cache(1));
            let pool = Arc::new(KvPool::with_policy(10, PAGE, geom, Policy::Lru));
            let paged = (
                pool.clone(),
                Arc::new(crate::engine::prefix_index::PrefixIndex::new(PAGE)),
            );
            let slots = SlotPool::new(2);
            let batcher = crate::engine::decode_batch::DecodeBatcher::new();
            let together: Vec<(String, bool)> = std::thread::scope(|scope| {
                let runs: Vec<_> = prompts
                    .iter()
                    .enumerate()
                    .map(|(i, prompt)| {
                        let (model, tokenizer, paged, batcher, slots) =
                            (&model, &tokenizer, &paged, &batcher, &slots);
                        scope.spawn(move || {
                            std::thread::sleep(Duration::from_millis(5 * i as u64));
                            let guard = pollster::block_on(slots.acquire());
                            run_on_slot(
                                model,
                                tokenizer,
                                paged,
                                batcher,
                                &guard,
                                prompt.clone(),
                                ANSWER,
                            )
                        })
                    })
                    .collect();
                runs.into_iter().map(|r| r.join().unwrap()).collect()
            });
            assert_eq!(together, alone, "an answer changed (hybrid={hybrid})");
            assert!(
                pool.yields() >= 1,
                "nobody gave pages back, so the test proves nothing (hybrid={hybrid})"
            );
            assert_eq!(
                pool.available(),
                10,
                "a page or promise leaked (hybrid={hybrid})"
            );
            assert_eq!(pool.stats().live_pages, 0);
        }
    }

    /// A pool and index sized for `model`'s own cache geometry.
    fn paged_pool_for(
        model: &DeterministicModel,
    ) -> (
        Arc<crate::engine::kv_pool::KvPool>,
        Arc<crate::engine::prefix_index::PrefixIndex>,
    ) {
        use crate::engine::kv_pool::{KvPool, LayerGeometry, Policy};
        const PAGE: usize = 4;
        let geom = LayerGeometry::of(&model.new_kv_cache(1));
        (
            Arc::new(KvPool::with_policy(64, PAGE, geom, Policy::Lru)),
            Arc::new(crate::engine::prefix_index::PrefixIndex::new(PAGE)),
        )
    }

    /// The second turn of a conversation answers the same whether its prefix
    /// came from the pool or was prefilled cold — and on a model without
    /// recurrent layers the pool really did skip the prefix, so the property
    /// is tested on a path that shares.
    #[test]
    fn a_shared_page_prefix_answers_as_a_cold_prefill() {
        let tokenizer = letter_tokenizer(8);
        let first: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 1, 2];
        let second: Vec<u32> = first.iter().copied().chain([3, 5, 7]).collect();
        for hybrid in [false, true] {
            let make = || {
                if hybrid {
                    DeterministicModel::hybrid(8)
                } else {
                    DeterministicModel::new(8)
                }
            };
            let cold_model = make();
            let cold = run_request(&cold_model, &tokenizer, None, second.clone(), 12);
            let cold_tokens = cold_model.forwarded_tokens.load(Ordering::Relaxed);

            let model = make();
            let paged = paged_pool_for(&model);
            run_request_paged(&model, &tokenizer, &paged, first.clone(), 4);
            let before = model.forwarded_tokens.load(Ordering::Relaxed);
            let warm = run_request_paged(&model, &tokenizer, &paged, second.clone(), 12);
            let warm_tokens = model.forwarded_tokens.load(Ordering::Relaxed) - before;

            assert_eq!(
                warm, cold,
                "a shared prefix changed the answer (hybrid={hybrid})"
            );
            if hybrid {
                assert_eq!(
                    warm_tokens, cold_tokens,
                    "a recurrent cache adopted pages its state does not match"
                );
            } else {
                assert!(
                    warm_tokens < cold_tokens,
                    "the pool shared nothing, so the test proves nothing: \
                     {warm_tokens} vs {cold_tokens}"
                );
            }
        }
    }

    /// The next turn of a conversation — the first turn's prompt, its reply,
    /// and a new message — finds the reply's pages, not only the prompt's,
    /// and answers as a cold prefill of the same tokens does.
    #[test]
    fn the_next_turn_shares_the_reply_pages_too() {
        let tokenizer = letter_tokenizer(8);
        // Nine prompt tokens: two whole pages and one the reply finishes.
        let first: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 1, 2];
        let model = DeterministicModel::new(8);
        let paged = paged_pool_for(&model);
        let (reply, _) = run_request_paged(&model, &tokenizer, &paged, first.clone(), 12);
        let reply: Vec<u32> = reply.chars().map(|c| c as u32 - 'a' as u32).collect();
        assert_eq!(reply.len(), 12);
        let second: Vec<u32> = first
            .iter()
            .chain(&reply)
            .copied()
            .chain([3, 5, 7])
            .collect();

        let cold_model = DeterministicModel::new(8);
        let cold = run_request(&cold_model, &tokenizer, None, second.clone(), 6);

        let before = model.forwarded_tokens.load(Ordering::Relaxed);
        let warm = run_request_paged(&model, &tokenizer, &paged, second.clone(), 6);
        let forwarded = model.forwarded_tokens.load(Ordering::Relaxed) - before;
        assert_eq!(warm, cold);
        // 24 tokens of the 21 the first turn committed (its last reply token
        // was never forwarded) make five whole pages; only the remainder of
        // the prompt and the answer should be computed.
        let prompt_pages_only = second.len() - 8;
        assert!(
            forwarded < prompt_pages_only,
            "the reply's pages were not shared: forwarded {forwarded} tokens"
        );
    }

    /// Speculation on a hybrid model emits exactly what plain decoding does:
    /// a rejected draft returns the recurrent state to the verify's anchor
    /// and runs the accepted tokens again, at every acceptance rate.
    #[test]
    fn a_hybrid_model_speculates_and_decodes_as_without_a_draft() {
        let tokenizer = letter_tokenizer(8);
        let prompt = vec![1u32, 2, 3, 4, 1, 2];
        let plain = {
            let model = DeterministicModel::hybrid(8);
            run_request(&model, &tokenizer, None, prompt.clone(), 40)
        };
        for disagree_every in [1usize, 2, 3] {
            let model = DeterministicModel::hybrid(8);
            let draft = Arc::new(DeterministicModel::drafting(8, disagree_every));
            let drafted =
                run_request_drafted(&model, draft.clone(), &tokenizer, prompt.clone(), 40, 4);
            assert_eq!(drafted, plain, "disagree_every={disagree_every}");
            assert!(
                draft.forwarded_tokens.load(Ordering::Relaxed) > 0,
                "no draft ran against the recurrent target"
            );
        }
    }

    /// A teacher-forced run sees the same distributions however many
    /// positions a pass covers, and one forward call per pass.
    #[test]
    fn teacher_forcing_is_the_same_by_decode_steps_or_verify_passes() {
        let tokens = [1u32, 2, 3, 4, 5, 6, 7, 1, 2, 3];
        let decoded = {
            let model = DeterministicModel::new(8);
            let out = crate::engine::teacher::teacher_forced(&model, &tokens, 1, 4, 0).unwrap();
            assert_eq!(model.forward_calls.load(Ordering::Relaxed), tokens.len());
            out
        };
        let model = DeterministicModel::new(8);
        let verified = crate::engine::teacher::teacher_forced(&model, &tokens, 4, 4, 0).unwrap();
        assert_eq!(model.forward_calls.load(Ordering::Relaxed), 3);
        assert_eq!(verified.len(), tokens.len());
        assert_eq!(verified, decoded);
    }

    /// A prompt makes way for decode steps only while *another* thread
    /// decodes: the decoding thread's own passes (a batched step, a verify)
    /// keep their groups.
    #[test]
    fn a_prompt_yields_between_layers_only_to_another_threads_decode() {
        let (entered, decoding) = std::sync::mpsc::channel();
        let (release, done) = std::sync::mpsc::channel::<()>();
        let decoder = std::thread::spawn(move || {
            let _decoding = Decoding::enter();
            entered
                .send((
                    prompt_yields_to_decode(),
                    crate::engine::arch::gemma::prefill_group_layers(4),
                ))
                .unwrap();
            done.recv().unwrap();
        });
        let (decoder_yields, decoder_group) = decoding.recv().unwrap();
        assert!(
            !decoder_yields,
            "the decoding thread's own passes do not yield"
        );
        assert_eq!(decoder_group, 4);
        assert!(prompt_yields_to_decode(), "another thread decodes");
        assert_eq!(crate::engine::arch::gemma::prefill_group_layers(4), 1);
        release.send(()).unwrap();
        decoder.join().unwrap();
    }

    /// While a request decodes, a timed prefill is sized to the yield budget
    /// from its wall time; an untimed or flat one is left alone.
    #[test]
    fn a_decoding_request_narrows_a_timed_prefill_to_the_yield_budget() {
        let _decoding = Decoding::enter();
        assert_eq!(
            yield_width_at(150, None, ChunkPolicy::Adaptive, 4096),
            usize::MAX,
            "nothing timed yet caps nothing"
        );
        // 10 ms a token of wall time: 150 ms is under the sizer's floor.
        let wall = Some((100, Duration::from_millis(1000)));
        let capped = yield_width_at(150, wall, ChunkPolicy::Adaptive, 4096);
        assert!(capped <= 16, "capped at {capped}");
        assert_eq!(
            yield_width_at(0, wall, ChunkPolicy::Adaptive, 4096),
            usize::MAX,
            "0 is off"
        );
        assert_eq!(
            yield_width_at(150, wall, ChunkPolicy::Flat, 4096),
            usize::MAX
        );
    }

    /// `run_request` with a slot store and a cache point, as a chat request
    /// to a hybrid model arrives.
    fn run_request_with_slots(
        model: &DeterministicModel,
        tokenizer: &Tokenizer,
        store: &crate::engine::slot_store::SlotStore,
        prompt_tokens: Vec<u32>,
        cache_point: Option<usize>,
        max_tokens: usize,
    ) -> (String, bool) {
        let slots = SlotPool::new(1);
        let guard = pollster::block_on(slots.acquire());
        let (tx, rx) = mpsc::unbounded_channel();
        let req = GenerateRequest {
            prompt_tokens,
            sampling: greedy_params(),
            max_tokens,
            cache_point,
            background: false,
            reasoning_budget: None,
            ..Default::default()
        };
        run(
            model,
            tokenizer,
            None,
            None,
            None,
            Some(store),
            None,
            &crate::engine::decode_batch::DecodeBatcher::new(),
            &guard,
            req,
            crate::config::Role::default(),
            crate::config::Role::default(),
            false,
            &crate::engine::metrics::ServerMetrics::new(),
            Instant::now(),
            tx,
            None,
            None,
        )
        .unwrap();
        drain(rx)
    }

    /// A hybrid model's next turn repeats this one only up to the end of the
    /// last user message — the reply comes back re-rendered — and must
    /// reuse exactly that much, from the state checkpointed there, and
    /// answer as a cold prefill of the same tokens does.
    #[test]
    fn a_hybrid_conversation_resumes_from_its_cache_point() {
        let tokenizer = letter_tokenizer(8);
        let dir = std::env::temp_dir().join(format!("orangu-m3-{}", std::process::id()));
        let store = crate::engine::slot_store::SlotStore::at(dir.clone(), "test".into(), 1);
        // Turn one: a "user message" of 6 tokens, then 3 of generation prompt.
        let first: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 7, 7];
        let model = DeterministicModel::hybrid(8);
        run_request_with_slots(&model, &tokenizer, &store, first.clone(), Some(6), 5);
        // Turn two: the same user message, then a reply rendered unlike the
        // tokens generated, and a new message.
        let second: Vec<u32> = first[..6].iter().copied().chain([2, 4, 6, 1, 3]).collect();
        let before = model.forwarded_tokens.load(Ordering::Relaxed);
        let warm = run_request_with_slots(&model, &tokenizer, &store, second.clone(), Some(9), 6);
        let forwarded = model.forwarded_tokens.load(Ordering::Relaxed) - before;

        let cold_model = DeterministicModel::hybrid(8);
        let cold = run_request(&cold_model, &tokenizer, None, second.clone(), 6);
        assert_eq!(warm, cold, "the checkpoint changed the answer");
        // Six tokens came from the checkpoint; the five after them and the
        // generated ones were computed.
        assert_eq!(
            forwarded,
            cold_model.forwarded_tokens.load(Ordering::Relaxed) - 6,
            "the checkpoint was not reused"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The rollback itself refuses a recurrent cache and leaves it as it was.
    #[test]
    fn truncating_a_recurrent_cache_is_an_error() {
        let model = DeterministicModel::hybrid(8);
        let mut cache = model.new_kv_cache(8);
        for (pos, t) in [1u32, 2, 3].into_iter().enumerate() {
            model.absorb(&mut cache, t, pos);
        }
        assert!(cache.truncate(1).is_err());
        assert_eq!(cache.layers[0].len, 3);
    }

    /// A rollback returns a hybrid cache to its checkpoint whole: the
    /// recurrent state as if the later tokens had never been pushed, and the
    /// attention rows cut back with it.
    #[test]
    fn a_recurrent_cache_rolls_back_to_its_checkpoint() {
        let model = DeterministicModel::hybrid(8);
        let mut cache = model.new_kv_cache(8);
        let mut reference = model.new_kv_cache(8);
        for (pos, t) in [1u32, 2].into_iter().enumerate() {
            model.absorb(&mut cache, t, pos);
            model.absorb(&mut reference, t, pos);
        }
        let checkpoint = cache.rollback_point(2).unwrap();
        for (pos, t) in [3u32, 4, 5].into_iter().enumerate() {
            model.absorb(&mut cache, t, pos + 2);
        }
        assert_ne!(
            cache.recurrent[0].host_snapshot().state,
            reference.recurrent[0].host_snapshot().state
        );
        cache.rollback(&checkpoint).unwrap();
        assert_eq!(cache.layers[0].len, 2);
        for (c, r) in cache.recurrent.iter().zip(&reference.recurrent) {
            assert_eq!(c.host_snapshot().state, r.host_snapshot().state);
            assert_eq!(c.host_snapshot().conv, r.host_snapshot().conv);
        }
        model.absorb(&mut cache, 6, 2);
        model.absorb(&mut reference, 6, 2);
        assert_eq!(
            cache.recurrent[0].host_snapshot().state,
            reference.recurrent[0].host_snapshot().state
        );
    }

    /// The correctness property prefix reuse must never break: a second,
    /// growing-conversation request (this exact model's own full first-
    /// turn history, plus a short new suffix — the shape `engine::
    /// prefix_cache`'s own doc comment calls the primary use case) must
    /// stream back *exactly* the same text whether or not a `PrefixCache`
    /// let it skip re-prefilling the shared part, and reuse must actually
    /// have skipped real work when it's available.
    #[test]
    fn prefix_cache_reuse_matches_a_full_recompute() {
        let n_vocab = 32;
        let tokenizer = letter_tokenizer(n_vocab);
        let turn1_prompt = vec![1u32, 2, 3, 4, 5];
        let turn2_suffix = vec![6u32, 7];

        // Baseline: no prefix cache at all, turn 2 is a full reprefill of
        // its own complete prompt from position 0 — today's behavior.
        let model = DeterministicModel::new(n_vocab);
        let (turn1_text, ok1) = run_request(&model, &tokenizer, None, turn1_prompt.clone(), 3);
        assert!(ok1);
        let mut turn2_prompt_baseline = turn1_prompt.clone();
        for ch in turn1_text.chars() {
            turn2_prompt_baseline.push(ch as u32 - 'a' as u32);
        }
        turn2_prompt_baseline.extend(turn2_suffix.clone());
        let (turn2_text_baseline, ok2) =
            run_request(&model, &tokenizer, None, turn2_prompt_baseline.clone(), 3);
        assert!(ok2);

        // Same two turns, this time through a shared `PrefixCache` — turn
        // 2's prompt is byte-for-byte `turn2_prompt_baseline` (same
        // tokenizer, same deterministic turn-1 output), so it should find
        // and reuse turn 1's entire cached history.
        let model = DeterministicModel::new(n_vocab);
        let pool = PrefixCache::new(4);
        let (turn1_text_reuse, ok1) =
            run_request(&model, &tokenizer, Some(&pool), turn1_prompt.clone(), 3);
        assert!(ok1);
        assert_eq!(
            turn1_text_reuse, turn1_text,
            "turn 1 has no prefix to reuse yet"
        );
        let mut turn2_prompt_reuse = turn1_prompt.clone();
        for ch in turn1_text_reuse.chars() {
            turn2_prompt_reuse.push(ch as u32 - 'a' as u32);
        }
        turn2_prompt_reuse.extend(turn2_suffix.clone());
        assert_eq!(
            turn2_prompt_reuse, turn2_prompt_baseline,
            "both runs' turn-2 prompts must be identical for this comparison to mean anything"
        );
        let forwarded_before_turn2 = model.forwarded_tokens.load(Ordering::Relaxed);
        let (turn2_text_reuse, ok2) =
            run_request(&model, &tokenizer, Some(&pool), turn2_prompt_reuse, 3);
        assert!(ok2);
        let reuse_forwarded = model.forwarded_tokens.load(Ordering::Relaxed);

        assert_eq!(
            turn2_text_reuse, turn2_text_baseline,
            "prefix reuse must produce byte-identical output to a full recompute"
        );
        // Turn 2's own forward-pass token count: reuse must have skipped
        // all but the very last of turn 1's 8-token history. `run`'s
        // decode loop stops as soon as `history.len()` reaches its target
        // capacity (`prompt.len() + max_tokens`) — which happens right
        // after the *last* generated token is appended to `history` but
        // *before* the forward call that would have pushed its own
        // key/value into the cache (`PrefixCache::take_best_match`'s own
        // doc comment covers this). Both turns here use `max_tokens = 3`
        // with no stop token ever reached, so this fires identically for
        // both: turn 1 leaves only 7 of its own 8 tokens actually cached,
        // and turn 2's own decode loop likewise only reaches 2 real
        // forward calls (its 3rd generated token's own forward call is
        // the one skipped this time). So turn 2 must forward: turn 1's
        // uncached 8th token (1), the 2 brand-new suffix tokens, plus 2
        // (not 3) decode-step forwards.
        let turn2_forwarded_reuse = reuse_forwarded - forwarded_before_turn2;
        assert_eq!(
            turn2_forwarded_reuse,
            1 + turn2_suffix.len() + 2,
            "reuse must skip turn 1's first 7 cached positions, forwarding only its own uncached 8th token, the new suffix, and this turn's own 2 real decode steps"
        );
    }

    /// `cache_prompt: false` must put every prompt token through a real
    /// forward pass even when a `PrefixCache` holds an exact match for the
    /// whole prompt — the property a prefill benchmark depends on, since a
    /// request served from cache reports a prompt time that measures the
    /// lookup and nothing else. Checked by counting forwarded tokens, not by
    /// timing: the same run repeated with the flag left at its default must
    /// skip that work, so the two counts bracket what the flag controls.
    #[test]
    fn cache_prompt_false_reprefills_a_prompt_the_cache_already_holds() {
        let n_vocab = 32;
        let tokenizer = letter_tokenizer(n_vocab);
        let prompt = vec![1u32, 2, 3, 4, 5];
        let model = DeterministicModel::new(n_vocab);
        let pool = PrefixCache::new(4);

        // Populate the pool, then re-send the identical prompt twice.
        let (first_text, ok) = run_request(&model, &tokenizer, Some(&pool), prompt.clone(), 3);
        assert!(ok);

        let before_uncached = model.forwarded_tokens.load(Ordering::Relaxed);
        let (uncached_text, ok) = run_request_cached(
            &model,
            &tokenizer,
            Some(&pool),
            prompt.clone(),
            3,
            /* cache_prompt */ false,
        );
        assert!(ok);
        let uncached_forwarded = model.forwarded_tokens.load(Ordering::Relaxed) - before_uncached;

        let before_cached = model.forwarded_tokens.load(Ordering::Relaxed);
        let (cached_text, ok) = run_request(&model, &tokenizer, Some(&pool), prompt.clone(), 3);
        assert!(ok);
        let cached_forwarded = model.forwarded_tokens.load(Ordering::Relaxed) - before_cached;

        // Skipping the cache changes cost, never output.
        assert_eq!(uncached_text, first_text);
        assert_eq!(cached_text, first_text);
        // All 5 prompt tokens forwarded, plus this run's own 2 real decode
        // steps (the 3rd generated token's forward call is skipped — see
        // `prefix_cache_reuse_matches_a_full_recompute` for why).
        assert_eq!(
            uncached_forwarded,
            prompt.len() + 2,
            "cache_prompt: false must prefill the whole prompt"
        );
        assert!(
            cached_forwarded < uncached_forwarded,
            "the same request with caching allowed must forward fewer tokens \
             ({cached_forwarded} vs {uncached_forwarded}) — otherwise this test \
             proves nothing about the flag"
        );
    }

    /// A `ModelForward` whose `forward` always panics — the deliberately
    /// broken model this module's own panic-recovery path
    /// (`Engine::generate`'s `catch_unwind` around `run`, `crate::
    /// panic_capture`) needs a real panic to exercise end to end, not
    /// just unit-test in isolation.
    struct PanickingModel {
        config: ModelConfig,
    }

    impl PanickingModel {
        fn new() -> Self {
            Self {
                config: ModelConfig {
                    architecture: "test".to_string(),
                    n_vocab: 8,
                    n_embd: 1,
                    n_layer: 1,
                    n_head: 1,
                    n_head_kv: 1,
                    head_dim: 4,
                    n_ctx_train: 1000,
                    rope_dim: 1,
                    rope_freq_base: 10000.0,
                    rms_eps: 1e-6,
                    pooling_type: PoolingType::Mean,
                },
            }
        }
    }

    /// A model whose hidden state at a position is the sum of every token
    /// up to it — causal by construction, and exact in `f32` for small ids,
    /// so a chunked run either reproduces the one-pass rows bit for bit or
    /// fed some token at the wrong position, twice, or not at all.
    struct PrefixSumModel {
        config: ModelConfig,
        causal: bool,
        chunks: std::sync::Mutex<Vec<(usize, usize)>>,
    }

    impl PrefixSumModel {
        fn new(causal: bool) -> Self {
            Self {
                config: DeterministicModel::new(8).config,
                causal,
                chunks: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    impl ModelForward for PrefixSumModel {
        fn config(&self) -> &ModelConfig {
            &self.config
        }

        fn new_kv_cache(&self, capacity: usize) -> KvCache {
            KvCache::new(1, capacity, 1)
        }

        fn forward(
            &self,
            _cache: &mut KvCache,
            _tokens: &[u32],
            _start_pos: usize,
            _slot_id: usize,
        ) -> Result<Vec<f32>> {
            unimplemented!("an embeddings pass produces no logits")
        }

        fn forward_hidden_states(&self, tokens: &[u32]) -> Result<Vec<f32>> {
            let mut cache = self.new_kv_cache(tokens.len());
            self.chunks.lock().unwrap().push((0, tokens.len()));
            self.prefix_sums(&mut cache, tokens, 0)
        }

        fn hidden_states_are_causal(&self) -> bool {
            self.causal
        }

        fn forward_hidden_states_at(
            &self,
            cache: &mut KvCache,
            tokens: &[u32],
            start_pos: usize,
        ) -> Result<Vec<f32>> {
            self.chunks.lock().unwrap().push((start_pos, tokens.len()));
            self.prefix_sums(cache, tokens, start_pos)
        }
    }

    impl PrefixSumModel {
        fn prefix_sums(
            &self,
            cache: &mut KvCache,
            tokens: &[u32],
            start_pos: usize,
        ) -> Result<Vec<f32>> {
            let layer = &mut cache.layers[0];
            anyhow::ensure!(
                layer.len == start_pos,
                "chunk at {start_pos}, cache at {}",
                layer.len
            );
            let mut acc: f32 = (0..layer.len).map(|p| layer.key_at(p, 0, 1)[0]).sum();
            let mut out = Vec::with_capacity(tokens.len());
            for &t in tokens {
                layer.push(&[t as f32], &[t as f32]);
                acc += t as f32;
                out.push(acc);
            }
            Ok(out)
        }
    }

    /// A causal model's embedding input is split into chunks, each at the
    /// position the last one ended, and the rows it gives are the one-pass
    /// rows exactly.
    #[test]
    fn a_causal_embedding_input_runs_in_chunks_with_the_one_pass_rows() {
        let tokens: Vec<u32> = (0..2000).map(|i| (i % 7) as u32).collect();
        let model = PrefixSumModel::new(true);
        let chunked = embedding_hidden_states(&model, &tokens).unwrap();
        let chunks = model.chunks.lock().unwrap().clone();
        assert!(chunks.len() > 1, "a 2000-token input ran as {chunks:?}");
        let mut next = 0;
        for &(pos, len) in &chunks {
            assert_eq!(pos, next, "chunks out of order: {chunks:?}");
            next += len;
        }
        assert_eq!(next, tokens.len());
        let one_pass = PrefixSumModel::new(true)
            .forward_hidden_states(&tokens)
            .unwrap();
        assert_eq!(chunked, one_pass);
    }

    /// A model whose tokens must see the whole input is never split.
    #[test]
    fn a_bidirectional_embedding_input_runs_in_one_pass() {
        let tokens: Vec<u32> = (0..2000).map(|i| (i % 7) as u32).collect();
        let model = PrefixSumModel::new(false);
        embedding_hidden_states(&model, &tokens).unwrap();
        assert_eq!(*model.chunks.lock().unwrap(), vec![(0, tokens.len())]);
    }

    impl ModelForward for PanickingModel {
        fn config(&self) -> &ModelConfig {
            &self.config
        }

        fn new_kv_cache(&self, capacity: usize) -> KvCache {
            KvCache::new(1, capacity, 1)
        }

        fn forward(
            &self,
            _cache: &mut KvCache,
            _tokens: &[u32],
            _start_pos: usize,
            _slot_id: usize,
        ) -> Result<Vec<f32>> {
            panic!("PANICKING_MODEL_DELIBERATE_TEST_PANIC");
        }

        fn forward_hidden_states(&self, _tokens: &[u32]) -> Result<Vec<f32>> {
            unimplemented!("not exercised by this test")
        }
    }

    /// A `ModelForward` that records the `(tokens, start_pos)` of every
    /// `forward` call instead of computing anything, so a test can see
    /// exactly how a prompt was fed to the model.
    struct RecordingModel {
        config: ModelConfig,
        calls: std::sync::Mutex<Vec<(Vec<u32>, usize)>>,
    }

    impl RecordingModel {
        fn new() -> Self {
            Self {
                config: PanickingModel::new().config,
                calls: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    impl ModelForward for RecordingModel {
        fn config(&self) -> &ModelConfig {
            &self.config
        }

        fn new_kv_cache(&self, capacity: usize) -> KvCache {
            KvCache::new(1, capacity, 1)
        }

        fn forward(
            &self,
            _cache: &mut KvCache,
            tokens: &[u32],
            start_pos: usize,
            _slot_id: usize,
        ) -> Result<Vec<f32>> {
            self.calls
                .lock()
                .unwrap()
                .push((tokens.to_vec(), start_pos));
            // Logits that identify which call produced them, so the caller
            // can be checked to keep the *last* chunk's.
            Ok(vec![start_pos as f32 + tokens.len() as f32; 8])
        }

        fn forward_hidden_states(&self, _tokens: &[u32]) -> Result<Vec<f32>> {
            unimplemented!("not exercised by this test")
        }
    }

    /// One state row per position, valued [`STATE_BASE`] above the position
    /// it belongs to, so a test can read straight off a head's input which
    /// position's state it was paired with — and so that the zero row
    /// position 0 is paired with cannot be mistaken for position 0's own.
    const STATE_WIDTH: usize = 2;
    const STATE_BASE: f32 = 100.0;

    /// A `ModelForward` that exports [`STATE_WIDTH`]-wide state rows —
    /// `[pos, pos]` for position `pos` — alongside whatever logits it is
    /// asked for, so a paired MTP head's bookkeeping can be checked without
    /// an architecture that has one.
    struct StatefulModel {
        config: ModelConfig,
    }

    impl StatefulModel {
        fn new() -> Self {
            Self {
                config: PanickingModel::new().config,
            }
        }

        fn states(tokens: &[u32], start_pos: usize) -> Vec<f32> {
            (0..tokens.len())
                .flat_map(|i| [STATE_BASE + (start_pos + i) as f32; STATE_WIDTH])
                .collect()
        }
    }

    impl ModelForward for StatefulModel {
        fn config(&self) -> &ModelConfig {
            &self.config
        }

        fn new_kv_cache(&self, capacity: usize) -> KvCache {
            KvCache::new(1, capacity, 1)
        }

        fn forward(
            &self,
            _cache: &mut KvCache,
            tokens: &[u32],
            start_pos: usize,
            _slot_id: usize,
        ) -> Result<Vec<f32>> {
            Ok(vec![start_pos as f32 + tokens.len() as f32; 8])
        }

        fn mtp_state_width(&self) -> Option<usize> {
            Some(STATE_WIDTH)
        }

        fn forward_with_states(
            &self,
            cache: &mut KvCache,
            tokens: &[u32],
            start_pos: usize,
            slot_id: usize,
        ) -> Result<(Vec<f32>, Vec<f32>)> {
            let logits = self.forward(cache, tokens, start_pos, slot_id)?;
            Ok((logits, Self::states(tokens, start_pos)))
        }

        fn forward_all_logits_with_states(
            &self,
            _cache: &mut KvCache,
            tokens: &[u32],
            start_pos: usize,
            _slot_id: usize,
        ) -> Result<(Vec<Vec<f32>>, Vec<f32>)> {
            let logits = tokens.iter().map(|_| vec![0.0; 8]).collect();
            Ok((logits, Self::states(tokens, start_pos)))
        }

        fn forward_hidden_states(&self, _tokens: &[u32]) -> Result<Vec<f32>> {
            unimplemented!("not exercised by this test")
        }
    }

    /// One call into a draft head, as the head saw it.
    #[derive(Debug, PartialEq)]
    struct HeadCall {
        tokens: Vec<u32>,
        /// The position each paired state row names — `-1.0` standing for
        /// the zero row a run beginning at position 0 gets.
        states: Vec<f32>,
        start_pos: usize,
        first_pos: usize,
    }

    /// An [`MtpHead`] that records what it was fed and guesses a fresh token
    /// every call, so a test can see the exact pairing the drafter built.
    struct RecordingHead {
        calls: std::sync::Mutex<Vec<HeadCall>>,
        next: std::sync::atomic::AtomicU32,
    }

    impl RecordingHead {
        fn new() -> Self {
            Self {
                calls: std::sync::Mutex::new(Vec::new()),
                next: std::sync::atomic::AtomicU32::new(1),
            }
        }

        fn calls(&self) -> std::sync::MutexGuard<'_, Vec<HeadCall>> {
            self.calls.lock().unwrap()
        }
    }

    impl MtpHead for RecordingHead {
        fn state_width(&self) -> usize {
            STATE_WIDTH
        }

        fn new_kv_cache(&self, capacity: usize) -> KvCache {
            KvCache::new(1, capacity, 1)
        }

        fn forward(
            &self,
            cache: &mut KvCache,
            tokens: &[u32],
            states: &[f32],
            start_pos: usize,
            first_pos: usize,
        ) -> Result<MtpStep> {
            // The real head holds the positions it never saw so its cache
            // stays indexed by the served model's, and appends a row per
            // token after them. A recorder that skipped this would let a
            // rollback test pass against a cache that was never filled.
            let slot = &mut cache.layers[0];
            let zeros = vec![0.0; 1];
            while slot.len < start_pos + tokens.len() {
                slot.push(&zeros, &zeros);
            }
            self.calls().push(HeadCall {
                tokens: tokens.to_vec(),
                states: states.chunks(STATE_WIDTH).map(|row| row[0]).collect(),
                start_pos,
                first_pos,
            });
            let token = self.next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut logits = vec![0.0; 8];
            logits[token as usize % 8] = 1.0;
            Ok(MtpStep {
                logits,
                state: vec![-(token as f32); STATE_WIDTH],
            })
        }
    }

    /// A [`RecordingHead`] that drafts a block (`MtpHead::block_size`), and
    /// records each block it is asked for: `(anchor, pos, n_draft,
    /// first_pos)`.
    struct BlockHead {
        context: RecordingHead,
        blocks: std::sync::Mutex<Vec<(u32, usize, usize, usize)>>,
    }

    impl MtpHead for BlockHead {
        fn state_width(&self) -> usize {
            STATE_WIDTH
        }

        fn new_kv_cache(&self, capacity: usize) -> KvCache {
            self.context.new_kv_cache(capacity)
        }

        fn forward(
            &self,
            cache: &mut KvCache,
            tokens: &[u32],
            states: &[f32],
            start_pos: usize,
            first_pos: usize,
        ) -> Result<MtpStep> {
            self.context
                .forward(cache, tokens, states, start_pos, first_pos)
        }

        fn block_size(&self) -> Option<usize> {
            Some(16)
        }

        fn draft_block(
            &self,
            _cache: &mut KvCache,
            anchor: u32,
            pos: usize,
            n_draft: usize,
            first_pos: usize,
        ) -> Result<Vec<u32>> {
            self.blocks
                .lock()
                .unwrap()
                .push((anchor, pos, n_draft, first_pos));
            Ok(vec![7; n_draft])
        }
    }

    fn block_drafter<'a>(head: &'a BlockHead, echo: Option<EchoLookup>) -> Drafter<'a> {
        Drafter::Mtp {
            head,
            tokens: 2,
            cache: head.new_kv_cache(64),
            committed: 0,
            first_pos: None,
            states: Vec::new(),
            states_start: 0,
            width: STATE_WIDTH,
            echo,
        }
    }

    /// A block head caches the committed context through the same catch-up
    /// a chained head runs — each token with the state behind it — and then
    /// drafts once, from the token the served model is about to read, at
    /// that token's position. No chained steps follow it.
    #[test]
    fn a_block_head_drafts_once_from_the_token_about_to_be_read() {
        let head = BlockHead {
            context: RecordingHead::new(),
            blocks: std::sync::Mutex::new(Vec::new()),
        };
        let mut drafter = block_drafter(&head, None);
        drafter.observe_states(0, &StatefulModel::states(&[0, 0, 0, 0], 0));
        let history: Vec<u32> = vec![10, 11, 12, 13, 14];
        let drafted = drafter
            .draft(&history, 8, &KvCache::new_with_dims(1, &[]), 0)
            .unwrap();
        assert_eq!(drafted, vec![7, 7]);
        let calls = head.context.calls();
        assert_eq!(calls.len(), 1, "the catch-up alone: {calls:?}");
        assert_eq!(calls[0].tokens, history);
        assert_eq!(calls[0].states, vec![0.0, 100.0, 101.0, 102.0, 103.0]);
        assert_eq!(*head.blocks.lock().unwrap(), vec![(14, 4, 2, 0)]);
    }

    /// While the reply copies an earlier passage, a block head's step copies
    /// it too, at the width the step was given, and leaves its context for
    /// later: the first block it drafts after the echo caches the whole
    /// copy in one catch-up, every token with the state behind it.
    #[test]
    fn a_block_head_copies_an_echo_and_keeps_its_context() {
        let head = BlockHead {
            context: RecordingHead::new(),
            blocks: std::sync::Mutex::new(Vec::new()),
        };
        let mut drafter = block_drafter(
            &head,
            Some(EchoLookup {
                ngram: 2,
                max_draft: 4,
                echo_draft: 0,
            }),
        );
        let passage: Vec<u32> = (20..40).collect();
        let mut history = passage.clone();
        history.push(99);
        history.extend_from_slice(&passage[..18]);
        drafter.observe_states(0, &StatefulModel::states(&history[..history.len() - 1], 0));
        assert!(drafter.echoes(&history));
        // Prompt lookup's own widths: no echo width, so no wide step.
        assert!(!drafter.has_echo_draft());
        assert_eq!(drafter.draft_len(false), 4);
        let drafted = drafter
            .draft(&history, 4, &KvCache::new_with_dims(1, &[]), 0)
            .unwrap();
        // What follows the copied passage, four tokens of it.
        assert_eq!(drafted, vec![38, 39, 99, 20]);
        assert!(head.blocks.lock().unwrap().is_empty());
        assert!(head.context.calls().is_empty(), "the context waits");

        // The served model accepted none of it and read a token that ends
        // the echo.
        let pos = history.len() - 1;
        drafter.observe_states(pos, &StatefulModel::states(&[history[pos]], pos));
        history.push(98);
        assert!(!drafter.echoes(&history));
        drafter
            .draft(&history, 4, &KvCache::new_with_dims(1, &[]), 0)
            .unwrap();
        let calls = head.context.calls();
        assert_eq!(calls.len(), 1, "one catch-up: {calls:?}");
        assert_eq!(calls[0].tokens, history);
        let mut behind = vec![0.0];
        behind.extend((0..history.len() - 1).map(|p| STATE_BASE + p as f32));
        assert_eq!(calls[0].states, behind);
        assert_eq!(head.blocks.lock().unwrap().len(), 1);
    }

    /// A shorter match than an echo's is copied too, with the context
    /// cached on the step as a block's would be: the next step may well be
    /// a block, and finds its context current.
    #[test]
    fn a_block_head_copies_a_short_match_with_its_context_current() {
        let head = BlockHead {
            context: RecordingHead::new(),
            blocks: std::sync::Mutex::new(Vec::new()),
        };
        let mut drafter = block_drafter(
            &head,
            Some(EchoLookup {
                ngram: 2,
                max_draft: 4,
                echo_draft: 0,
            }),
        );
        let history: Vec<u32> = vec![1, 2, 3, 9, 8, 1, 2];
        drafter.observe_states(0, &StatefulModel::states(&history[..history.len() - 1], 0));
        assert!(!drafter.echoes(&history));
        let drafted = drafter
            .draft(&history, 4, &KvCache::new_with_dims(1, &[]), 0)
            .unwrap();
        assert_eq!(drafted, vec![3, 9, 8, 1]);
        assert!(head.blocks.lock().unwrap().is_empty());
        let calls = head.context.calls();
        assert_eq!(calls.len(), 1, "the context is cached: {calls:?}");
        assert_eq!(calls[0].tokens, history);
    }

    fn mtp_drafter<'a>(head: &'a RecordingHead, tokens: usize) -> Drafter<'a> {
        Drafter::Mtp {
            head,
            tokens,
            cache: head.new_kv_cache(64),
            committed: 0,
            first_pos: None,
            states: Vec::new(),
            states_start: 0,
            width: STATE_WIDTH,
            echo: None,
        }
    }

    /// The pairing an MTP head is trained on: the token at position `p`
    /// with the served model's state at `p - 1`. Getting it off by one is
    /// invisible — the head still drafts fluent tokens, they are simply
    /// tokens for a context one position out, and the only symptom is an
    /// acceptance rate that never rises.
    ///
    /// Position 0 has no predecessor and takes the zero row the head is
    /// trained against there, which is why a real row is offset by
    /// [`STATE_BASE`]: the two would otherwise both read `0.0` and the test
    /// could not tell a correct pairing from a missing one.
    #[test]
    fn the_head_sees_each_prompt_token_paired_with_the_state_behind_it() {
        let model = StatefulModel::new();
        let head = RecordingHead::new();
        let mut drafter = mtp_drafter(&head, 2);
        let mut cache = model.new_kv_cache(64);
        let tokens: Vec<u32> = (100..110).collect();
        prefill_in_chunks(
            &mut ChunkCost::new(),
            &model,
            &mut cache,
            &tokens,
            0,
            0,
            Chunking {
                width: 4,
                policy: ChunkPolicy::Flat,
            },
            Some(&mut drafter),
            &mut |_| Ok(()),
        )
        .unwrap();

        let calls = head.calls();
        assert_eq!(calls.len(), 3, "one head pass per prompt chunk");
        assert_eq!(calls[0].tokens, vec![100, 101, 102, 103]);
        assert_eq!(calls[0].states, vec![0.0, 100.0, 101.0, 102.0]);
        assert_eq!(calls[0].start_pos, 0);
        assert_eq!(calls[1].tokens, vec![104, 105, 106, 107]);
        assert_eq!(calls[1].states, vec![103.0, 104.0, 105.0, 106.0]);
        assert_eq!(calls[1].start_pos, 4);
        assert_eq!(calls[2].states, vec![107.0, 108.0]);
        assert_eq!(calls[2].start_pos, 8);
        // Nothing was skipped, so every position is attendable.
        assert!(calls.iter().all(|c| c.first_pos == 0));
    }

    /// A prompt prefix served out of the KV cache never reaches the head:
    /// the states it would need were not recomputed. Those positions become
    /// placeholders and every window starts past them.
    ///
    /// The position the prefill *starts* at is a placeholder too, and that
    /// is the easy one to get wrong — its own state exists, but the state
    /// **behind** it does not.
    #[test]
    fn a_reused_prefix_is_held_as_placeholders_the_head_never_attends() {
        let model = StatefulModel::new();
        let head = RecordingHead::new();
        let mut drafter = mtp_drafter(&head, 2);
        let mut cache = model.new_kv_cache(64);
        let tokens: Vec<u32> = (200..206).collect();
        prefill_in_chunks(
            &mut ChunkCost::new(),
            &model,
            &mut cache,
            &tokens,
            5,
            0,
            Chunking {
                width: 4,
                policy: ChunkPolicy::Flat,
            },
            Some(&mut drafter),
            &mut |_| Ok(()),
        )
        .unwrap();

        let calls = head.calls();
        assert_eq!(calls[0].start_pos, 6, "position 5 has no state behind it");
        assert_eq!(calls[0].tokens, vec![201, 202, 203]);
        assert_eq!(calls[0].states, vec![105.0, 106.0, 107.0]);
        assert!(
            calls.iter().all(|c| c.first_pos == 6),
            "the reused prefix must stay out of every window: {calls:?}"
        );
    }

    /// A draft chains onto the head's *own* state and its *own* last guess,
    /// one position at a time, starting where the served model is about to
    /// forward. Feeding the served model's state again on the second step
    /// would draft the same token forever.
    #[test]
    fn a_draft_chains_the_head_onto_its_own_state() {
        let head = RecordingHead::new();
        let mut drafter = mtp_drafter(&head, 3);
        // As if the served model had just forwarded positions 0..4 and the
        // history now ends with the token it is about to forward at 4.
        drafter.observe_states(0, &StatefulModel::states(&[0, 0, 0, 0], 0));
        let history: Vec<u32> = vec![10, 11, 12, 13, 14];
        let drafted = drafter
            .draft(&history, 8, &KvCache::new_with_dims(1, &[]), 0)
            .unwrap();
        assert_eq!(drafted.len(), 3);

        let calls = head.calls();
        assert_eq!(calls.len(), 3, "a catch-up pass, then one per extra guess");
        // The catch-up ends on the token about to be forwarded — that last
        // position is the first guess, so no separate call for it.
        assert_eq!(calls[0].tokens, vec![10, 11, 12, 13, 14]);
        assert_eq!(calls[0].states, vec![0.0, 100.0, 101.0, 102.0, 103.0]);
        assert_eq!(calls[1].tokens, vec![drafted[0]]);
        assert_eq!(
            calls[1].start_pos, 5,
            "one past the position being forwarded"
        );
        assert_eq!(calls[2].tokens, vec![drafted[1]]);
        assert_eq!(calls[2].start_pos, 6);
        // The chained rows are the head's own, never the served model's.
        assert!(
            calls[1].states[0] < 0.0 && calls[2].states[0] < 0.0,
            "chained steps must read the head's own state: {calls:?}"
        );
    }

    /// Rejected drafts come off the head's cache with the served model's.
    /// Left there, every later position is written one slot too far along
    /// and the acceptance rate falls quietly to zero — the failure this
    /// engine has already had once, on the draft-model path.
    #[test]
    fn a_rejected_draft_is_rolled_off_the_head_cache_too() {
        let head = RecordingHead::new();
        let mut drafter = mtp_drafter(&head, 3);
        drafter.observe_states(0, &StatefulModel::states(&[0, 0, 0, 0], 0));
        let history: Vec<u32> = vec![10, 11, 12, 13, 14];
        drafter
            .draft(&history, 8, &KvCache::new_with_dims(1, &[]), 0)
            .unwrap();
        assert_eq!(
            drafter.mtp_committed(),
            7,
            "five committed plus two guesses"
        );

        // The served model accepted `current` and one drafted token.
        drafter.commit(6).unwrap();
        assert_eq!(drafter.mtp_committed(), 6);
        let Drafter::Mtp { cache, .. } = &drafter else {
            unreachable!()
        };
        assert_eq!(cache.layers[0].len, 6);
    }

    /// A prompt longer than one chunk is fed as consecutive `forward`
    /// calls, each starting where the previous one ended — covering every
    /// token exactly once, in order, with no gap or overlap. Anything else
    /// would corrupt the KV cache rather than merely slow things down.
    #[test]
    fn a_long_prompt_is_prefilled_in_consecutive_chunks() {
        let model = RecordingModel::new();
        let mut cache = model.new_kv_cache(64);
        let tokens: Vec<u32> = (0..25).collect();

        let logits = prefill_in_chunks(
            &mut ChunkCost::new(),
            &model,
            &mut cache,
            &tokens,
            0,
            0,
            Chunking {
                width: 10,
                policy: ChunkPolicy::Adaptive,
            },
            None,
            &mut |_| Ok(()),
        )
        .unwrap();

        let calls = model.calls.lock().unwrap().clone();
        assert_eq!(
            calls,
            vec![
                ((0..10).collect::<Vec<u32>>(), 0),
                ((10..20).collect::<Vec<u32>>(), 10),
                ((20..25).collect::<Vec<u32>>(), 20),
            ]
        );
        // The last chunk's logits are what sampling continues from; an
        // earlier chunk's would sample from the middle of the prompt.
        assert_eq!(logits[0], 25.0);
    }

    /// `Flat` does not probe. The probe exists to price one submission
    /// against a driver limit, and on a backend with no such limit it costs
    /// an extra pass over the whole model for nothing — which on a streamed
    /// model is the entire expense of a prefill.
    ///
    /// Asserted against `Adaptive` on the same input rather than in
    /// isolation, because the property is a *difference*: if both policies
    /// ever produce the same call sequence, this change did nothing.
    #[test]
    fn the_flat_policy_does_not_open_with_a_probe() {
        let widths = |policy| {
            let model = RecordingModel::new();
            let mut cache = model.new_kv_cache(128);
            let tokens: Vec<u32> = (0..40).collect();
            prefill_in_chunks(
                &mut ChunkCost::new(),
                &model,
                &mut cache,
                &tokens,
                0,
                0,
                Chunking { width: 32, policy },
                None,
                &mut |_| Ok(()),
            )
            .unwrap();
            let calls = model.calls.lock().unwrap().clone();
            calls.into_iter().map(|(t, _)| t.len()).collect::<Vec<_>>()
        };

        assert_eq!(widths(ChunkPolicy::Flat), vec![32, 8]);
        assert_eq!(
            widths(ChunkPolicy::Adaptive)[0],
            PREFILL_PROBE_TOKENS,
            "the adaptive path must still probe — this test is only \
             meaningful while the two policies differ"
        );
    }

    /// Dropping the adaptation must not drop the contract every prefill has:
    /// each token fed exactly once, in order, at its absolute position. This
    /// is the property whose violation corrupts a KV cache silently instead
    /// of failing, so it is asserted separately for each policy.
    #[test]
    fn the_flat_policy_still_feeds_every_token_once_in_order() {
        let model = RecordingModel::new();
        let mut cache = model.new_kv_cache(128);
        let tokens: Vec<u32> = (0..25).collect();

        prefill_in_chunks(
            &mut ChunkCost::new(),
            &model,
            &mut cache,
            &tokens,
            100,
            0,
            Chunking {
                width: 10,
                policy: ChunkPolicy::Flat,
            },
            None,
            &mut |_| Ok(()),
        )
        .unwrap();

        let calls = model.calls.lock().unwrap().clone();
        assert_eq!(
            calls,
            vec![
                ((0..10).collect::<Vec<u32>>(), 100),
                ((10..20).collect::<Vec<u32>>(), 110),
                ((20..25).collect::<Vec<u32>>(), 120),
            ]
        );
    }

    /// The direction of the mapping, which is the one thing here that is
    /// dangerous to get wrong: a backend that *has* a timeout must keep the
    /// chunker adapting, and an unregistered process must behave as though it
    /// does. Everything that is not the CPU backend — including CUDA, OpenCL
    /// and ROCm, none of which `as_wgpu` can distinguish from it — reaches
    /// this with `true`.
    /// The width choice is the one place D9 can still be catastrophically
    /// wrong, and in opposite directions in the two regimes: a narrow width on
    /// a streamed model re-reads the model per chunk (measured: 61x the
    /// bytes), while dropping the bound on a resident model costs tok/s and a
    /// third more peak memory. Both directions are asserted.
    #[test]
    fn a_streamed_model_prefills_in_one_pass() {
        assert_eq!(flat_width_for(None, 100_000, Some(0.5)), 0);
        assert_eq!(flat_width_for(None, 100_000, Some(0.0)), 0);
    }

    #[test]
    fn a_resident_model_keeps_the_narrow_width() {
        assert_eq!(
            flat_width_for(None, 100_000, Some(1.0)),
            PREFILL_BATCH_DEFAULT
        );
        assert_eq!(
            flat_width_for(None, 100_000, Some(RESIDENT_ENOUGH)),
            PREFILL_BATCH_DEFAULT
        );
    }

    /// The fail-safe: a platform where `mincore` is absent or refused must not
    /// be read as having said "streamed".
    #[test]
    fn unknowable_residency_keeps_todays_width() {
        assert_eq!(flat_width_for(None, 100_000, None), PREFILL_BATCH_DEFAULT);
    }

    #[test]
    fn an_explicit_batch_setting_outranks_the_regime() {
        // Including `0`, which is how an operator asks for one pass outright.
        assert_eq!(flat_width_for(Some(2048), 100_000, Some(0.1)), 2048);
        assert_eq!(flat_width_for(Some(2048), 100_000, Some(1.0)), 2048);
        assert_eq!(flat_width_for(Some(0), 100_000, Some(1.0)), 0);
    }

    /// A prompt that fits one pass at the narrow width must not be able to
    /// reach the streamed branch — it is the guard that keeps `mincore` off
    /// the path of every short prompt.
    #[test]
    fn a_short_prompt_ignores_the_regime_entirely() {
        for resident in [None, Some(0.0), Some(1.0)] {
            assert_eq!(
                flat_width_for(None, PREFILL_BATCH_DEFAULT, resident),
                PREFILL_BATCH_DEFAULT,
                "resident={resident:?}"
            );
        }
    }

    #[test]
    fn a_backend_with_a_submission_timeout_keeps_adapting() {
        assert_eq!(policy_for(true), ChunkPolicy::Adaptive);
        assert_eq!(policy_for(false), ChunkPolicy::Flat);
    }

    /// A prompt that starts partway in (the prefix cache reused its head)
    /// keeps counting positions from there — the chunk boundaries are
    /// relative to the prompt, the positions are absolute.
    #[test]
    fn chunking_starts_from_the_reused_prefix_position() {
        let model = RecordingModel::new();
        let mut cache = model.new_kv_cache(64);
        let tokens: Vec<u32> = (0..7).collect();

        prefill_in_chunks(
            &mut ChunkCost::new(),
            &model,
            &mut cache,
            &tokens,
            100,
            0,
            Chunking {
                width: 3,
                policy: ChunkPolicy::Adaptive,
            },
            None,
            &mut |_| Ok(()),
        )
        .unwrap();

        let starts: Vec<usize> = model
            .calls
            .lock()
            .unwrap()
            .iter()
            .map(|(_, pos)| *pos)
            .collect();
        assert_eq!(starts, vec![100, 103, 106]);
    }

    /// The whole point: a chunk that ran long makes the next one smaller, so
    /// the *work* per submission stays bounded even though the cost per token
    /// climbs with context. Sizing by token count alone does not — 512 tokens
    /// cost 2.3 s at position 0 and 10.1 s at position 6 656, past the ~10 s
    /// the driver allows before it resets the device.
    #[test]
    fn a_slow_chunk_shrinks_the_next_one() {
        let budget = Duration::from_millis(3_000);
        // 512 tokens took 10.1 s: ~19.7 ms each, so ~114 fit once the next
        // chunk leaves headroom for starting deeper in the prompt.
        let next = next_chunk_width(512, Duration::from_millis(10_069), budget, 512);
        assert!((105..=125).contains(&next), "sized {next}");
        // And a fast one is allowed back up to the configured maximum.
        assert_eq!(
            next_chunk_width(16, Duration::from_millis(70), budget, 512),
            512
        );
    }

    /// Whatever the measurement says, the width stays inside both bounds — the
    /// floor keeps a submission the driver can still finish, and the ceiling is
    /// the operator's own `ORANGU_PREFILL_BATCH`.
    #[test]
    fn an_adapted_width_stays_within_its_bounds() {
        let budget = Duration::from_millis(3_000);
        // Absurdly slow: would size to zero without the floor.
        assert_eq!(
            next_chunk_width(16, Duration::from_secs(600), budget, 512),
            PREFILL_MIN_CHUNK_TOKENS
        );
        // A batch smaller than the floor is honoured, not clamped upward —
        // and must not panic, which `clamp` does when min exceeds max.
        assert_eq!(next_chunk_width(3, Duration::from_secs(600), budget, 3), 3);
        // A free chunk cannot widen past the maximum.
        assert_eq!(next_chunk_width(8, Duration::ZERO, budget, 128), 128);
    }

    /// The bug this sizer was rebuilt around: a chunk's cost is not
    /// proportional to its width, and reading it as if it were turns the
    /// opening probe's fixed cost into a per-token rate several times the real
    /// one.
    ///
    /// The numbers are a real measurement — a 1120-token prompt on a 35-layer
    /// model, where a 16-token chunk cost 270 ms and a 122-token chunk 733 ms.
    /// Proportional sizing reads the probe at 16.9 ms/token and asks for ~133;
    /// the two-point fit recovers ~200 ms fixed and ~4.4 ms/token and asks for
    /// what the budget actually affords.
    #[test]
    fn two_observations_separate_fixed_cost_from_per_token_cost() {
        let budget = Duration::from_millis(3_000);
        let mut cost = ChunkCost::new();

        cost.observe(16, Duration::from_millis(270));
        let one_point = cost.next_width(budget, 512);

        cost.observe(122, Duration::from_millis(733));
        let two_points = cost.next_width(budget, 512);

        assert!(
            (120..=145).contains(&one_point),
            "one observation cannot do better than proportional: {one_point}"
        );
        assert!(
            two_points > one_point * 3,
            "the fit must reach a working width, not climb to it: \
             {one_point} then {two_points}"
        );
        let fit = cost.fit.expect("two separated observations must fit");
        assert!(
            (0.15..0.25).contains(&fit.fixed),
            "fixed cost {} s off the measured ~0.2 s",
            fit.fixed
        );
    }

    /// The safety property the budget exists for, preserved through the
    /// rewrite: a chunk that overran must shrink the next one. Over-sizing
    /// here is what resets the device, so it is asserted against a *standing*
    /// fit that still calls the wider chunk affordable.
    #[test]
    fn an_overrunning_chunk_shrinks_the_next_one_despite_a_fit() {
        let budget = Duration::from_millis(3_000);
        let mut cost = ChunkCost::new();
        // Two cheap, well-separated chunks: a fit that says 512 is affordable.
        cost.observe(16, Duration::from_millis(120));
        cost.observe(256, Duration::from_millis(600));
        assert_eq!(cost.next_width(budget, 512), 512);

        // Then the cost curve rises underneath it and 512 blows the budget.
        // This repeats the previous width deliberately: two observations that
        // close together cannot separate the two terms, so the *only* thing
        // that can shrink this is the fit tracking through the new point.
        cost.observe(512, Duration::from_millis(9_000));
        let after = cost.next_width(budget, 512);
        assert!(
            after < 256,
            "an overrun must shrink the next chunk, sized {after}"
        );
    }

    /// The estimate must never freeze.
    ///
    /// Once the sizer settles on a width it keeps choosing it, so every later
    /// pair is the same width twice and cannot separate the two terms. If that
    /// case left the fit untouched, whatever the opening pair happened to
    /// measure would size every prompt the process ever served — and since the
    /// frozen width is then the only width chosen, nothing could ever dislodge
    /// it. Measured on hardware before this was handled: a server whose
    /// opening pair was taken under profiling overhead pinned itself to
    /// 117-token chunks for every later request and ran *slower* than carrying
    /// no estimate at all.
    #[test]
    fn a_width_frozen_by_an_unlucky_opening_pair_recovers() {
        let budget = Duration::from_millis(3_000);
        let mut cost = ChunkCost::new();
        // An opening pair measured while something else had the machine: the
        // costs are inflated and the fit they produce is far too pessimistic.
        cost.observe(16, Duration::from_millis(270));
        cost.observe(106, Duration::from_millis(1_900));
        let frozen = cost.next_width(budget, 512);
        assert!(frozen < 200, "the unlucky pair should size low: {frozen}");

        // The interference goes away and that width is now cheap. Every later
        // observation is the *same* width, so nothing can separate the terms —
        // and the width must still climb back toward what the budget affords.
        let mut width = frozen;
        for _ in 0..4 {
            cost.observe(width, Duration::from_millis(300));
            width = cost.next_width(budget, 512);
        }
        assert_eq!(
            width, 512,
            "a frozen width must recover to the ceiling, reached {width}"
        );
    }

    /// A pair of observations too close in width cannot set a slope: the
    /// subtraction that produces it has a small, noisy denominator under it.
    /// Such a pair must leave the previous fit standing rather than replace it.
    #[test]
    fn observations_too_close_together_do_not_set_a_slope() {
        assert_eq!(
            CostFit::through(
                (500, Duration::from_millis(2_000)),
                (512, Duration::from_millis(2_010))
            ),
            None,
            "near-identical widths must not produce a fit"
        );
        // Nor may a wider chunk that somehow cost less — noise, not a negative
        // marginal cost.
        assert_eq!(
            CostFit::through(
                (16, Duration::from_millis(300)),
                (256, Duration::from_millis(200))
            ),
            None,
        );
    }

    /// A machine already priced by an earlier prompt opens the next one at a
    /// working width instead of re-running the ramp — that is the whole point
    /// of carrying the estimate — but only from position zero, because a
    /// carried fit knows nothing about the depth a continuation starts at.
    #[test]
    fn a_priced_machine_opens_a_fresh_prompt_at_width_but_still_probes_deep() {
        let budget = Duration::from_millis(3_000);
        let mut cost = ChunkCost::new();
        cost.observe(16, Duration::from_millis(270));
        cost.observe(122, Duration::from_millis(733));

        let fresh = cost
            .opening_width(0, 512, budget, 512)
            .expect("a priced machine must offer an opening width");
        assert!(fresh > 400, "opened only {fresh} tokens wide");
        assert_eq!(
            cost.opening_width(100_000, 600, budget, 512),
            None,
            "a deep continuation must still open with the probe"
        );
        // A turn's worth of tokens a little way in fits many times over and
        // goes in one pass; a long one at the same depth still probes.
        assert_eq!(cost.opening_width(2_048, 60, budget, 512), Some(60));
        assert_eq!(cost.opening_width(2_048, 4_000, budget, 512), None);
        // And an unpriced one has nothing to offer, at any position.
        assert_eq!(ChunkCost::new().opening_width(0, 512, budget, 512), None);
        assert_eq!(
            ChunkCost::new().opening_width(2_048, 120, budget, 512),
            None
        );
    }

    /// End to end through the chunk loop: a cold estimate opens with the probe
    /// and a carried one does not. Asserted on the widths the model was
    /// actually called with, so it covers the wiring and not just the
    /// arithmetic.
    ///
    /// Only the *opening* width is asserted, because only it is decided before
    /// any forward runs. Every later width comes from timing a `RecordingModel`
    /// call, which is a few microseconds of noise on an idle machine and
    /// whatever the scheduler decides on a busy one — an earlier version of
    /// this test asserted the submission *count* and failed intermittently for
    /// exactly that reason. The count belongs to
    /// [`the_carried_estimate_is_what_removes_the_submissions`], which models
    /// the cost curve instead of racing it.
    #[test]
    fn a_carried_estimate_opens_the_next_prompt_without_the_probe() {
        let widths = |cost: &mut ChunkCost| {
            let model = RecordingModel::new();
            let mut cache = model.new_kv_cache(4096);
            let tokens: Vec<u32> = (0..1120).collect();
            prefill_in_chunks(
                cost,
                &model,
                &mut cache,
                &tokens,
                0,
                0,
                Chunking {
                    width: 512,
                    policy: ChunkPolicy::Adaptive,
                },
                None,
                &mut |_| Ok(()),
            )
            .unwrap();
            let calls = model.calls.lock().unwrap().clone();
            calls.into_iter().map(|(t, _)| t.len()).collect::<Vec<_>>()
        };

        let cold = widths(&mut ChunkCost::new());
        assert_eq!(
            cold[0], PREFILL_PROBE_TOKENS,
            "a cold estimate must still probe: {cold:?}"
        );
        assert_eq!(cold.iter().sum::<usize>(), 1120, "every token fed once");

        // Primed with a fixed pair rather than with whatever the previous run
        // happened to measure, so the opening width is decided by arithmetic.
        let mut primed = ChunkCost::new();
        primed.observe(16, Duration::from_millis(270));
        primed.observe(122, Duration::from_millis(733));
        let warm = widths(&mut primed);
        // That pair prices the machine at 200 ms fixed and 4.37 ms a token,
        // which affords 469 tokens — and `even_chunk_width` then takes 374,
        // the widest width dividing 1120 into the same three passes. Both
        // facts matter: it opens at a *working* width rather than the
        // 16-token probe, and the passes come out even so the widest chunk
        // (and therefore the peak scratch) is 374 instead of 512.
        assert_eq!(
            warm[0], 374,
            "a priced machine must open at an evened working width: {warm:?}"
        );
        assert_eq!(
            warm.len(),
            1120_usize.div_ceil(469),
            "evening must not change the pass count: {warm:?}"
        );
        assert_eq!(warm.iter().sum::<usize>(), 1120);
    }

    /// The submission count, against a cost curve rather than a clock: a
    /// prefill chunk really costs `fixed + per_token · n`, so the sizer is
    /// driven here with the numbers measured on hardware
    /// (`fixed` 297 ms, `per_token` 3.57 ms) instead of by timing a mock.
    ///
    /// This is the claim the change is for, so it is the one worth pinning
    /// deterministically: carrying the estimate turns a 1120-token prompt from
    /// five submissions into three.
    #[test]
    fn the_carried_estimate_is_what_removes_the_submissions() {
        let budget = Duration::from_millis(3_000);
        let cost_of = |n: usize| Duration::from_secs_f64(0.297 + 0.00357 * n as f64);

        // Run the sizer over a 1120-token prompt and report the widths it picks.
        let run = |cost: &mut ChunkCost| {
            let mut widths = Vec::new();
            let mut done = 0usize;
            let mut width = cost
                .opening_width(0, 512, budget, 512)
                .unwrap_or(PREFILL_PROBE_TOKENS.min(512));
            while done < 1120 {
                let n = width.min(1120 - done);
                widths.push(n);
                done += n;
                cost.observe(n, cost_of(n));
                width = cost.next_width(budget, 512);
            }
            widths
        };

        let mut carried = ChunkCost::new();
        let cold = run(&mut carried);
        let warm = run(&mut carried);

        assert_eq!(cold, vec![16, 101, 512, 491], "cold ramp: {cold:?}");
        assert_eq!(warm, vec![512, 512, 96], "warm run: {warm:?}");
        assert!(
            warm.len() < cold.len(),
            "carrying the estimate must cost fewer submissions"
        );
        // And it stays converged: a third prompt does not drift off it.
        assert_eq!(run(&mut carried), vec![512, 512, 96]);
        assert_eq!(cold.iter().sum::<usize>(), 1120);
        assert_eq!(warm.iter().sum::<usize>(), 1120);
    }

    /// A deep continuation is the shape that used to hang the GPU: a full-width
    /// chunk submitted at a position where each token costs ten times what it
    /// did at the start. It must open with the small probe instead.
    #[test]
    fn a_continuation_opens_with_a_probe_not_a_full_width_chunk() {
        let model = RecordingModel::new();
        let mut cache = model.new_kv_cache(200_000);
        let tokens: Vec<u32> = (0..600).collect();

        prefill_in_chunks(
            &mut ChunkCost::new(),
            &model,
            &mut cache,
            &tokens,
            100_000,
            0,
            Chunking {
                width: 512,
                policy: ChunkPolicy::Adaptive,
            },
            None,
            &mut |_| Ok(()),
        )
        .unwrap();

        let first = model.calls.lock().unwrap()[0].0.len();
        assert_eq!(first, PREFILL_PROBE_TOKENS, "opened {first} tokens wide");
    }

    /// Progress must be reported *after* each chunk's forward and count
    /// cumulatively, so a client can render a bar that only ever moves
    /// forward and reaches the total exactly once.
    #[test]
    fn every_prefill_chunk_reports_cumulative_progress() {
        let model = RecordingModel::new();
        let mut cache = model.new_kv_cache(64);
        let tokens: Vec<u32> = (0..25).collect();
        let mut seen = Vec::new();

        prefill_in_chunks(
            &mut ChunkCost::new(),
            &model,
            &mut cache,
            &tokens,
            0,
            0,
            Chunking {
                width: 10,
                policy: ChunkPolicy::Adaptive,
            },
            None,
            &mut |done| {
                seen.push((done, model.calls.lock().unwrap().len()));
                Ok(())
            },
        )
        .unwrap();

        // (tokens done, forwards completed) — the second element proves the
        // report follows the work rather than announcing it.
        assert_eq!(seen, vec![(10, 1), (20, 2), (25, 3)]);
    }

    /// A report that answers an error — the client is gone — stops the
    /// prefill there: no further chunk runs, and the error says why.
    #[test]
    fn a_prefill_stops_at_the_chunk_its_client_left_during() {
        let model = RecordingModel::new();
        let mut cache = model.new_kv_cache(64);
        let tokens: Vec<u32> = (0..25).collect();
        let err = prefill_in_chunks(
            &mut ChunkCost::new(),
            &model,
            &mut cache,
            &tokens,
            0,
            0,
            Chunking {
                width: 10,
                policy: ChunkPolicy::Adaptive,
            },
            None,
            &mut |_| Err(anyhow::Error::new(PrefillCancelled)),
        )
        .unwrap_err();
        assert!(err.is::<PrefillCancelled>(), "{err:#}");
        assert_eq!(model.calls.lock().unwrap().len(), 1);
    }

    /// A one-chunk prompt still reports, once, at the total — otherwise a
    /// short prompt's progress bar would never appear at all.
    #[test]
    fn a_single_chunk_prompt_still_reports_once() {
        let model = RecordingModel::new();
        let mut cache = model.new_kv_cache(64);
        let tokens: Vec<u32> = (0..5).collect();
        let mut seen = Vec::new();

        prefill_in_chunks(
            &mut ChunkCost::new(),
            &model,
            &mut cache,
            &tokens,
            0,
            0,
            Chunking {
                width: 512,
                policy: ChunkPolicy::Adaptive,
            },
            None,
            &mut |done| {
                seen.push(done);
                Ok(())
            },
        )
        .unwrap();

        assert_eq!(seen, vec![5]);
    }

    /// The split must come out even without ever widening a chunk or adding
    /// a pass.
    ///
    /// The case this exists for: 2236 tokens under a 2048 ceiling is two
    /// passes either way, but `[2048, 188]` peaks at 2048 of scratch and
    /// `[1118, 1118]` at 1118.
    #[test]
    fn evening_a_split_keeps_the_pass_count_and_lowers_the_peak() {
        // The case from the task list.
        assert_eq!(even_chunk_width(2236, 2048), 1118);
        // A prompt that already fits one pass is untouched.
        assert_eq!(even_chunk_width(1000, 2048), 1000);
        assert_eq!(even_chunk_width(2048, 2048), 2048);
        // A probe-width opening still divides evenly.
        assert_eq!(even_chunk_width(2236, 256), 249);
        // Never wider than asked, never more passes than asked.
        for remaining in [1usize, 7, 100, 513, 2236, 4097, 22_000] {
            for width in [1usize, 16, 249, 256, 1118, 2048, 8192] {
                let got = even_chunk_width(remaining, width);
                assert!(got <= width.max(1), "{remaining}/{width} widened to {got}");
                assert!(got >= 1, "{remaining}/{width} gave zero");
                assert_eq!(
                    remaining.div_ceil(got),
                    remaining.div_ceil(width),
                    "{remaining}/{width} changed the pass count"
                );
            }
        }
        // The no-limit opt-out passes through.
        assert_eq!(even_chunk_width(2236, 0), 0);
        assert_eq!(even_chunk_width(0, 2048), 2048);
    }

    /// A chunk that made no device submission is not priced against the
    /// budget, because the budget bounds a driver timeout it cannot reach.
    /// A chunk that made some is, however few — that is the stream, and
    /// leaving it where it was is what `budget_elapsed`'s spread rule is
    /// for.
    ///
    /// The case this exists for, measured: a host-run mixture's 2239-token
    /// prompt was cut to 114-token chunks by its wall time alone, and each
    /// chunk re-paid every layer's expert stack.
    #[test]
    fn a_chunk_that_submitted_nothing_is_not_priced_against_the_driver() {
        // SAFETY: single-threaded test, and the value is restored below.
        let was = std::env::var("ORANGU_PREFILL_CHUNK_SPREAD").ok();
        assert!(!chunk_is_priced(0), "no submission, no driver exposure");
        assert!(chunk_is_priced(1));
        assert!(chunk_is_priced(1_000));
        assert_eq!(was, std::env::var("ORANGU_PREFILL_CHUNK_SPREAD").ok());
    }

    /// A prompt that fits in one chunk — and `ORANGU_PREFILL_BATCH=0` at
    /// any length — is exactly one `forward` call, the same single pass
    /// this code made before chunking existed.
    #[test]
    fn a_short_prompt_is_still_one_forward_call() {
        for (len, batch) in [(5usize, 512usize), (5000, 0)] {
            let model = RecordingModel::new();
            let mut cache = model.new_kv_cache(8192);
            let tokens: Vec<u32> = (0..len as u32).collect();

            prefill_in_chunks(
                &mut ChunkCost::new(),
                &model,
                &mut cache,
                &tokens,
                0,
                0,
                Chunking {
                    width: batch,
                    policy: ChunkPolicy::Adaptive,
                },
                None,
                &mut |_| Ok(()),
            )
            .unwrap();

            let calls = model.calls.lock().unwrap();
            assert_eq!(calls.len(), 1, "len {len}, batch {batch}");
            assert_eq!(calls[0].0.len(), len);
        }
    }

    /// A panic during generation must reach the client as a real,
    /// detailed `StreamEvent::Error` — the panic's own message plus a
    /// captured backtrace (`panic_capture`) — not the generic "task
    /// panicked" note `tokio::task::JoinError`'s own `Display` would give
    /// on its own, and the generation channel must still terminate
    /// cleanly (one `Error` event, not a hang or a second event).
    #[tokio::test]
    async fn a_panic_during_generation_reaches_the_client_with_a_captured_backtrace() {
        // Prints the panic to stderr too (`panic_capture::install`'s hook
        // chains to, not replaces, the default one) — expected noise for a
        // test that deliberately panics, left alone rather than swapped
        // out for a silencing hook: `std::panic::set_hook` is a process-
        // global slot, and a hook that panics itself (or races another
        // concurrently-running test's own hook swap) aborts the whole
        // process outright, which a first attempt at silencing this
        // actually hit.
        crate::panic_capture::install();

        let engine = Engine {
            mtp: None,
            paged_kv: None,
            metrics: Arc::new(crate::engine::metrics::ServerMetrics::new()),
            draft: None,
            model: Arc::new(PanickingModel::new()),
            tokenizer: Arc::new(letter_tokenizer(8)),
            chat_template_source: None,
            chat_template: None,
            recurrent: false,
            slots: SlotPool::new(1),
            prefix_cache: None,
            decode_batcher: Arc::new(crate::engine::decode_batch::DecodeBatcher::new()),
            slot_store: None,
            role: crate::config::Role::default(),
            reasoning_effort: None,
            live_stats: false,
            image: None,
            video: None,
            vision: None,
            switch: None,
        };

        let mut rx = engine
            .generate(GenerateRequest {
                json_output: false,
                json_schema: None,
                prompt_tokens: vec![1, 2, 3],
                sampling: greedy_params(),
                max_tokens: 4,
                stop_token_ids: vec![],
                cache_prompt: true,
                id_slot: None,
                timings_per_token: false,
                role: None,
                cache_point: None,
                background: false,
                reasoning_budget: None,
            })
            .await;

        let event = rx
            .recv()
            .await
            .expect("generate() must send exactly one event before closing the channel on a panic");
        let StreamEvent::Error(detail) = event else {
            panic!("expected a StreamEvent::Error, got something else");
        };
        assert!(
            detail.contains("PANICKING_MODEL_DELIBERATE_TEST_PANIC"),
            "error detail must include the panic's own message, got: {detail}"
        );
        assert!(
            detail.contains("backtrace:"),
            "error detail must include a captured backtrace, got: {detail}"
        );

        assert!(
            rx.recv().await.is_none(),
            "the channel must close after the one error event, not send anything further"
        );
    }

    /// An ordinary panic keeps its full detail (the test above proves that
    /// end to end); a lost GPU device is replaced by the one sentence that
    /// is actually actionable. The panic's `wgpu` backtrace is not a
    /// diagnosis of the caller's request, and it must not be what they get
    /// instead of an explanation — the detail stays in the server's log.
    #[test]
    fn a_device_lost_panic_is_reported_as_one_sentence_not_a_backtrace() {
        let detail = "panicked at vulkan.rs:6183:\nmapping a buffer failed\n\nbacktrace:\n...";

        let ordinary = panic_report(detail.to_string(), false);
        assert_eq!(ordinary, detail, "an ordinary panic keeps its detail");

        let lost = panic_report(detail.to_string(), true);
        assert_eq!(lost, crate::device_lost::CLIENT_MESSAGE);
        assert!(!lost.contains("backtrace"));
        assert!(!lost.contains("vulkan.rs"));
    }

    /// A request whose client has gone away stops, even when every token it
    /// produces is suppressed.
    ///
    /// **This is the case the old check missed**, and it missed it silently.
    /// The disconnect test used to ride on sending a token, so a generation
    /// that rendered nothing — structural markers, or a reasoning body the
    /// role hides — never asked whether anyone was listening and ran to
    /// `max_tokens`, holding the slot every other request is queued for.
    ///
    /// Asserted on the target's forward calls rather than on elapsed time, so
    /// it fails on a slow machine for the right reason or not at all.
    #[test]
    fn generation_stops_when_the_client_goes_away_even_with_nothing_to_send() {
        let model = DeterministicModel::new(8);
        let tokenizer = suppressed_tokenizer(8);
        let slots = SlotPool::new(1);
        let guard = pollster::block_on(slots.acquire());
        let (tx, rx) = mpsc::unbounded_channel();
        // The client is already gone before the first token.
        drop(rx);

        let max_tokens = 500;
        run(
            &model,
            &tokenizer,
            None,
            None,
            None,
            None,
            None,
            &crate::engine::decode_batch::DecodeBatcher::new(),
            &guard,
            GenerateRequest {
                prompt_tokens: vec![1, 2, 3],
                sampling: greedy_params(),
                max_tokens,
                ..Default::default()
            },
            crate::config::Role::default(),
            crate::config::Role::default(),
            false,
            &crate::engine::metrics::ServerMetrics::new(),
            Instant::now(),
            tx,
            None,
            None,
        )
        .unwrap();

        // One prefill plus a couple of decode steps, not five hundred.
        let calls = model.forward_calls.load(Ordering::Relaxed);
        assert!(
            calls < 5,
            "kept generating for a client that had gone: {calls} forward calls"
        );
    }

    /// **The property the whole feature rests on.** A draft model may only
    /// change how fast an answer arrives, never what it says — so greedy
    /// output with a draft attached must be byte-identical to greedy output
    /// without one, whatever the draft proposes.
    ///
    /// Swept over how often the draft is wrong (never, every other call,
    /// every third, always) and over the draft depth, because the interesting
    /// failures are all in the rollback: a cache left one token long, or a
    /// draft cache that quietly desynchronises from the target's, shows up as
    /// divergence only at particular acceptance patterns.
    #[test]
    fn a_draft_model_never_changes_what_greedy_decoding_emits() {
        let tokenizer = letter_tokenizer(8);
        let prompt = vec![1u32, 2, 3, 4, 1, 2];
        let plain = {
            let model = DeterministicModel::new(8);
            run_request(&model, &tokenizer, None, prompt.clone(), 40)
        };
        assert!(plain.1, "the baseline run must finish");
        assert!(!plain.0.is_empty(), "the baseline run must emit something");

        for disagree_every in [0usize, 1, 2, 3, 5] {
            for draft_tokens in [1usize, 2, 4, 7] {
                let model = DeterministicModel::new(8);
                let draft = Arc::new(DeterministicModel::drafting(8, disagree_every));
                let drafted = run_request_drafted(
                    &model,
                    draft.clone(),
                    &tokenizer,
                    prompt.clone(),
                    40,
                    draft_tokens,
                );
                assert_eq!(
                    drafted.0, plain.0,
                    "speculation changed the output (disagree_every={disagree_every}, \
                     draft_tokens={draft_tokens})"
                );
                assert!(drafted.1, "the drafted run must finish");
                assert!(
                    draft.forwarded_tokens.load(Ordering::Relaxed) > 0,
                    "the draft model was never actually run"
                );
            }
        }
    }

    /// `run_request_drafted` under any sampling, with or without a draft.
    fn run_request_sampling(
        model: &DeterministicModel,
        draft: Option<Arc<DeterministicModel>>,
        tokenizer: &Tokenizer,
        prompt_tokens: Vec<u32>,
        max_tokens: usize,
        sampling: SamplingParams,
    ) -> (String, bool) {
        let slots = SlotPool::new(1);
        let guard = pollster::block_on(slots.acquire());
        let (tx, rx) = mpsc::unbounded_channel();
        let req = GenerateRequest {
            prompt_tokens,
            sampling,
            max_tokens,
            ..Default::default()
        };
        let draft = draft.map(|model| DraftModel {
            model,
            tokens: 4,
            label: "test-draft".to_string(),
        });
        run(
            model,
            tokenizer,
            draft.as_ref(),
            None,
            None,
            None,
            None,
            &crate::engine::decode_batch::DecodeBatcher::new(),
            &guard,
            req,
            crate::config::Role::default(),
            crate::config::Role::default(),
            false,
            &crate::engine::metrics::ServerMetrics::new(),
            Instant::now(),
            tx,
            None,
            None,
        )
        .unwrap();
        drain(rx)
    }

    /// Under sampling, too, a draft changes how many forwards a request costs
    /// and nothing it emits: the verifier samples each drafted position as
    /// plain decoding samples it.
    #[test]
    fn a_draft_never_changes_what_sampled_decoding_emits() {
        let tokenizer = letter_tokenizer(8);
        let prompt = vec![1u32, 2, 3, 4, 1, 2];
        for seed in [1u64, 7, 42] {
            let sampling = SamplingParams {
                temperature: 0.8,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                repeat_penalty: 1.0,
                repeat_last_n: 0,
                seed,
            };
            let plain = run_request_sampling(
                &DeterministicModel::new(8),
                None,
                &tokenizer,
                prompt.clone(),
                40,
                sampling.clone(),
            );
            for disagree_every in [0usize, 2, 1] {
                let model = DeterministicModel::new(8);
                let draft = Arc::new(DeterministicModel::drafting(8, disagree_every));
                let drafted = run_request_sampling(
                    &model,
                    Some(draft.clone()),
                    &tokenizer,
                    prompt.clone(),
                    40,
                    sampling.clone(),
                );
                assert_eq!(
                    drafted, plain,
                    "seed {seed}, disagree_every {disagree_every}"
                );
                assert!(
                    draft.forwarded_tokens.load(Ordering::Relaxed) > 0,
                    "the draft never ran"
                );
            }
        }
    }

    /// [`letter_tokenizer`] over single JSON characters, with `e` as the stop
    /// token (id 7). Every grammar state has a legal token, and inside a
    /// string every token but the stop is legal, so a draft is often accepted.
    fn json_tokenizer() -> Tokenizer {
        let tokens: Vec<GgufValue> = ["{", "\"", "a", "b", "c", ":", "}", "e", ","]
            .iter()
            .map(|t| GgufValue::String(t.to_string()))
            .collect();
        let gguf = GgufFile {
            version: 3,
            metadata: vec![
                (
                    "tokenizer.ggml.tokens".to_string(),
                    GgufValue::Array(tokens),
                ),
                (
                    "tokenizer.ggml.model".to_string(),
                    GgufValue::String("llama".to_string()),
                ),
            ],
            tensors: vec![],
            alignment: 32,
            data_offset: 0,
        };
        Tokenizer::from_gguf(&gguf).unwrap()
    }

    /// A JSON-constrained request, with or without a draft of `draft_tokens`.
    fn run_request_json(
        model: &DeterministicModel,
        draft: Option<(Arc<DeterministicModel>, usize)>,
        prompt_tokens: Vec<u32>,
        sampling: SamplingParams,
    ) -> (String, bool) {
        let tokenizer = json_tokenizer();
        let slots = SlotPool::new(1);
        let guard = pollster::block_on(slots.acquire());
        let (tx, rx) = mpsc::unbounded_channel();
        let req = GenerateRequest {
            json_output: true,
            json_schema: None,
            prompt_tokens,
            sampling,
            max_tokens: 60,
            stop_token_ids: vec![7],
            ..Default::default()
        };
        let draft = draft.map(|(model, tokens)| DraftModel {
            model,
            tokens,
            label: "test-draft".to_string(),
        });
        run(
            model,
            &tokenizer,
            draft.as_ref(),
            None,
            None,
            None,
            None,
            &crate::engine::decode_batch::DecodeBatcher::new(),
            &guard,
            req,
            crate::config::Role::default(),
            crate::config::Role::default(),
            false,
            &crate::engine::metrics::ServerMetrics::new(),
            Instant::now(),
            tx,
            None,
            None,
        )
        .unwrap();
        drain(rx)
    }

    /// Under a JSON constraint a draft still changes nothing that is emitted:
    /// the draft model proposes unconstrained tokens, and verification picks
    /// each position under the constraint as plain decoding does, so the
    /// constrained answer is byte-identical, greedy or sampled.
    #[test]
    fn a_draft_never_changes_what_constrained_decoding_emits() {
        let prompt = vec![1u32, 2, 3, 4, 1, 2];
        let sampled = SamplingParams {
            temperature: 0.8,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            seed: 7,
        };
        for sampling in [greedy_params(), sampled] {
            let baseline = DeterministicModel::new(9);
            let plain = run_request_json(&baseline, None, prompt.clone(), sampling.clone());
            let plain_calls = baseline.forward_calls.load(Ordering::Relaxed);
            let mut fewest_calls = usize::MAX;
            assert!(plain.1, "the baseline run must finish");
            // A run may end at `max_tokens` inside the object, so what holds
            // is that the output is a legal start of one.
            assert!(
                plain.0.starts_with('{')
                    && crate::engine::constraint::JsonPrefix::object().allows(plain.0.as_bytes()),
                "the constraint did not hold: {:?}",
                plain.0
            );
            for disagree_every in [0usize, 1, 2, 3] {
                for draft_tokens in [1usize, 2, 4] {
                    let model = DeterministicModel::new(9);
                    let draft = Arc::new(DeterministicModel::drafting(9, disagree_every));
                    let drafted = run_request_json(
                        &model,
                        Some((draft.clone(), draft_tokens)),
                        prompt.clone(),
                        sampling.clone(),
                    );
                    assert_eq!(
                        drafted, plain,
                        "speculation changed the constrained output (temperature {}, \
                         disagree_every {disagree_every}, draft_tokens {draft_tokens})",
                        sampling.temperature
                    );
                    assert!(
                        draft.forwarded_tokens.load(Ordering::Relaxed) > 0,
                        "the draft never ran under a constraint"
                    );
                    fewest_calls = fewest_calls.min(model.forward_calls.load(Ordering::Relaxed));
                }
            }
            // Some draft was accepted, or every run above compared a
            // rejected draft with plain decoding and proved nothing.
            assert!(
                fewest_calls < plain_calls,
                "no draft was ever accepted under the constraint ({fewest_calls} forward calls \
                 against {plain_calls}; output {:?})",
                plain.0
            );
        }
    }

    /// A perfect draft must actually save forwards, or the path is inert and
    /// the test above would pass on a `Drafter` that returned nothing.
    ///
    /// Counted in *forward calls*, not tokens: the target still processes
    /// every token — a verification forward carries the whole draft — so
    /// tokens are exactly what speculation does not reduce. What it reduces
    /// is how many times the model has to be entered, which on a
    /// bandwidth-bound decode is one full pass over the weights each.
    #[test]
    fn a_perfect_draft_costs_the_target_fewer_forward_calls() {
        let tokenizer = letter_tokenizer(8);
        let prompt = vec![1u32, 2, 3, 4, 1, 2];

        let plain_model = DeterministicModel::new(8);
        let plain = run_request(&plain_model, &tokenizer, None, prompt.clone(), 40);
        let plain_calls = plain_model.forward_calls.load(Ordering::Relaxed);

        let model = DeterministicModel::new(8);
        // `disagree_every = 0` never disagrees, so every drafted token is
        // exactly what the target would have produced.
        let draft = Arc::new(DeterministicModel::drafting(8, 0));
        let drafted = run_request_drafted(&model, draft, &tokenizer, prompt, 40, 4);
        assert_eq!(drafted.0, plain.0);

        let calls = model.forward_calls.load(Ordering::Relaxed);
        // One prefill plus one verification per five tokens, against one
        // prefill plus one call per token.
        assert!(
            calls * 3 < plain_calls,
            "a perfect 4-token draft should cut the target's forward calls several-fold: \
             {calls} vs {plain_calls}"
        );
        assert!(
            model.forwarded_tokens.load(Ordering::Relaxed) > 0,
            "the target must still have run"
        );
    }

    /// A draft that is always wrong must still be correct, and must not
    /// wedge: every step rejects everything, so the loop makes progress only
    /// through the target's own token.
    #[test]
    fn an_always_wrong_draft_still_terminates_with_the_right_answer() {
        let tokenizer = letter_tokenizer(8);
        let prompt = vec![5u32, 5, 5];
        let plain = {
            let model = DeterministicModel::new(8);
            run_request(&model, &tokenizer, None, prompt.clone(), 25)
        };
        let model = DeterministicModel::new(8);
        let draft = Arc::new(DeterministicModel::drafting(8, 1));
        let drafted = run_request_drafted(&model, draft, &tokenizer, prompt, 25, 4);
        assert_eq!(drafted.0, plain.0);
        assert!(drafted.1);
    }

    /// No drafter may propose more tokens than the KV cache can still hold.
    ///
    /// A verification forward appends `1 + draft.len()` positions at once, so
    /// a draft longer than the remaining context writes past the allocation —
    /// and on the GPU mirror, sized to the request's capacity, that is a write
    /// into whatever is next rather than an error. Only reachable in the last
    /// few tokens of a full context, which is exactly why it needs a test
    /// rather than a run to find it.
    #[test]
    fn a_draft_never_runs_past_the_end_of_the_context() {
        let history: Vec<u32> = vec![1, 2, 3, 1, 2, 3, 1, 2, 3];
        for room in 0..6 {
            let mut lookup = Drafter::PromptLookup {
                ngram: 2,
                max_draft: 8,
                echo_draft: 16,
            };
            let drafted = lookup
                .draft(&history, room, &KvCache::new_with_dims(1, &[]), 0)
                .expect("prompt lookup");
            assert!(
                drafted.len() <= room,
                "prompt-lookup drafted {} with room for {room}",
                drafted.len()
            );

            let draft_model = DeterministicModel::new(8);
            let cache = draft_model.new_kv_cache(64);
            let mut model = Drafter::Model {
                model: &draft_model,
                tokens: 8,
                cache,
                committed: 0,
            };
            let drafted = model
                .draft(&history, room, &KvCache::new_with_dims(1, &[]), 0)
                .expect("draft model");
            assert!(
                drafted.len() <= room,
                "the draft model drafted {} with room for {room}",
                drafted.len()
            );
        }
    }

    /// The draft cache is rolled back to exactly what the target committed.
    ///
    /// Asserted directly rather than through output, because the symptom of
    /// getting it wrong is not a wrong answer — the target re-derives every
    /// token either way — but an acceptance rate silently falling to zero as
    /// the draft reads a context shifted out from under it.
    #[test]
    fn commit_rolls_the_draft_cache_back_to_what_the_target_kept() {
        let draft_model = DeterministicModel::new(8);
        let mut cache = draft_model.new_kv_cache(64);
        // Ten tokens in the draft's cache, standing for eight committed plus
        // two drafted-and-about-to-be-rejected.
        for pos in 0..10 {
            draft_model
                .forward(&mut cache, &[1], pos, 0)
                .expect("test model");
        }
        let mut drafter = Drafter::Model {
            model: &draft_model,
            tokens: 4,
            cache,
            committed: 10,
        };
        drafter.commit(8).unwrap();
        match &drafter {
            Drafter::Model {
                cache, committed, ..
            } => {
                assert_eq!(*committed, 8);
                assert_eq!(
                    cache.layers[0].len, 8,
                    "the rejected tail was not rolled off"
                );
            }
            _ => unreachable!(),
        }
        // Committing further ahead than the cache reaches is the
        // all-accepted case, and must leave the cache alone for the next
        // step's catch-up to fill in rather than pretend it holds more.
        drafter.commit(12).unwrap();
        match &drafter {
            Drafter::Model {
                cache, committed, ..
            } => {
                assert_eq!(*committed, 8);
                assert_eq!(cache.layers[0].len, 8);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn ngram_draft_copies_the_continuation_of_the_latest_matching_context() {
        // Last two tokens are [1, 2]; their most recent earlier occurrence is
        // at index 4, followed by [3, 9, 9] — so a 3-token draft is [3, 9, 9].
        let history = [1u32, 2, 3, 7, 1, 2, 3, 9, 9, 1, 2];
        assert_eq!(ngram_draft(&history, 2, 3), vec![3, 9, 9]);
        // max_draft caps the length.
        assert_eq!(ngram_draft(&history, 2, 1), vec![3]);
    }

    #[test]
    fn ngram_draft_prefers_the_longest_matching_context() {
        // [1, 2] last recurs at index 6, followed by [7, 7]; but [5, 1, 2]
        // recurs at index 0, followed by [8, 8], and the longer context wins.
        let history = [5u32, 1, 2, 8, 8, 6, 1, 2, 7, 7, 5, 1, 2];
        assert_eq!(ngram_draft(&history, 2, 2), vec![8, 8]);
    }

    #[test]
    fn ngram_draft_copies_a_list_from_the_prompt_not_from_its_own_copy() {
        // A prompt listing items `[n, 50, 51, 52, 53, 54]` for n = 1..=6,
        // then the reply copying it, three items in. Every item ends the same
        // five tokens, so the newest occurrence of the trailing context is
        // the reply's own item 2, whose continuation is item 3 again. Matched
        // back past item 3's number, the prompt's copy is the place being
        // copied, and the draft runs on through items 4 and 5.
        let item = |n: u32| [n, 50, 51, 52, 53, 54];
        let mut history: Vec<u32> = vec![99];
        for n in 1..=6 {
            history.extend(item(n));
        }
        history.push(98);
        for n in 1..=3 {
            history.extend(item(n));
        }
        let mut want = Vec::new();
        for n in 4..=5 {
            want.extend(item(n));
        }
        assert_eq!(ngram_draft(&history, 2, 12), want);
    }

    #[test]
    fn a_draft_echoes_only_on_a_long_matched_context() {
        let lookup = Drafter::PromptLookup {
            ngram: 2,
            max_draft: 4,
            echo_draft: 16,
        };
        // A passage of 20 distinct tokens, then the reply copying its first
        // 17: matched 17 back, an echo.
        let passage: Vec<u32> = (100..120).collect();
        let mut history = passage.clone();
        history.extend(&passage[..17]);
        assert!(lookup.echoes(&history));
        // The same copy only 4 tokens in: a phrase, not an echo.
        let mut short = passage.clone();
        short.extend(&passage[..4]);
        assert!(!lookup.echoes(&short));
        assert!(!ngram_draft(&short, 2, 4).is_empty());
    }

    #[test]
    fn ngram_draft_is_empty_without_a_match_or_enough_history() {
        assert!(ngram_draft(&[1, 2], 2, 4).is_empty()); // suffix is the whole history
        assert!(ngram_draft(&[1, 2, 3, 4, 5], 2, 4).is_empty()); // [4,5] never recurs
        assert!(ngram_draft(&[], 2, 4).is_empty());
    }

    #[test]
    fn kv_cache_truncate_rolls_back_length_and_regrows_cleanly() {
        // A plain (no-GPU-mirror) cache: push a few positions, roll back, and
        // confirm the length moves and re-pushing overwrites in place.
        let mut cache = KvCache::new(2, 8, 4);
        for i in 0..6u32 {
            for layer in &mut cache.layers {
                let v = vec![i as f32; 4];
                layer.push(&v, &v);
            }
        }
        assert_eq!(cache.layers[0].len, 6);
        cache.truncate(4).unwrap();
        assert_eq!(cache.layers[0].len, 4);
        cache.truncate(10).unwrap(); // no-op: never grows
        assert_eq!(cache.layers[0].len, 4);
    }
}

#[cfg(test)]
mod admission_tests {
    use super::*;
    use crate::engine::kv_pool::{KvPool, LayerGeometry};

    /// Eight pages of four tokens.
    fn pool() -> Arc<KvPool> {
        Arc::new(KvPool::new(
            8,
            4,
            vec![LayerGeometry {
                kv_dim: 8,
                stride: 1,
                ring: None,
            }],
        ))
    }

    /// A prompt the pool has room for is admitted at once and its pages
    /// promised until the promise is dropped.
    #[test]
    fn a_prompt_with_room_is_admitted_and_its_pages_held() {
        let pool = pool();
        // 11 tokens and the one generated: three pages of four.
        let held = wait_for_prompt_pages(&pool, 11, None, || false).expect("admitted");
        assert_eq!(held.pages, 3);
        assert_eq!(pool.available(), 5);
        drop(held);
        assert_eq!(pool.available(), 8);
    }

    /// A prompt arriving while others hold the pages it needs waits until
    /// they are given back, then holds them itself; two arriving together
    /// cannot both be admitted against the same free pages.
    #[test]
    fn a_prompt_waits_for_the_pages_others_hold() {
        let pool = pool();
        let first = wait_for_prompt_pages(&pool, 19, None, || false).expect("admitted");
        assert_eq!(first.pages, 5);
        let started = Instant::now();
        let releaser = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(100));
            drop(first);
        });
        let second = wait_for_prompt_pages(&pool, 19, None, || false).expect("admitted after");
        assert!(started.elapsed() >= std::time::Duration::from_millis(90));
        assert_eq!(pool.available(), 3);
        releaser.join().unwrap();
        drop(second);
        assert_eq!(pool.available(), 8);
    }

    /// A slot's retained conversation holding the pages a prompt needs is
    /// dropped for it: an idle cache never gives them back by itself, and a
    /// prompt waiting on it would wait for the next request on that slot.
    #[test]
    fn a_prompt_does_not_wait_behind_an_idle_retained_cache() {
        let pool = pool();
        let mut idle =
            KvCache::new_with_strided_dims(24, &crate::engine::kv_cache::strided_dims(&pool))
                .try_into_paged(pool.clone(), 0)
                .unwrap_or_else(|_| panic!("the pool has room"));
        for r in 0..24 {
            let row = vec![r as f32; 8];
            idle.layers[0].push(&row, &row);
        }
        idle.commit_pages();
        let store = crate::engine::slot_store::SlotStore::at(
            std::env::temp_dir().join("orangu-admission-test"),
            "test".into(),
            1,
        );
        store.retain(0, (0..24).collect(), idle);
        assert_eq!(pool.available(), 2, "the retained cache holds six pages");
        let started = Instant::now();
        let gave_up = || started.elapsed() > std::time::Duration::from_millis(300);
        let held = wait_for_prompt_pages(&pool, 19, Some(&store), gave_up)
            .expect("admitted once the idle cache is dropped");
        assert_eq!(held.pages, 5);
        assert!(started.elapsed() < std::time::Duration::from_millis(250));
    }

    /// Nothing to wait for: a prompt larger than the whole pool, or a
    /// client that has gone, is not held back and takes no promise.
    #[test]
    fn a_prompt_that_cannot_be_helped_by_waiting_does_not_wait() {
        let pool = pool();
        assert!(wait_for_prompt_pages(&pool, 40, None, || false).is_none());
        let _all = wait_for_prompt_pages(&pool, 31, None, || false).expect("admitted");
        let started = Instant::now();
        assert!(wait_for_prompt_pages(&pool, 3, None, || true).is_none());
        assert!(started.elapsed() < std::time::Duration::from_millis(50));
    }
}
