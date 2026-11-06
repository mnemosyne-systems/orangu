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

//! Next-token sampling: repetition penalty, then temperature + top-k +
//! top-p + min-p, matching llama.cpp's own default sampler chain order
//! closely enough for these parameters. `temperature <= 0.0` means greedy
//! (always the highest-logit token) and is fully deterministic.

#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    /// Divisor applied to the logit of any token seen in the last
    /// [`Self::repeat_last_n`]. **`1.0` — off — by default**, matching the
    /// reference implementation; see [`Default`].
    pub repeat_penalty: f32,
    /// How many of the most recent tokens the penalties look at.
    pub repeat_last_n: usize,
    /// Subtracted from the logit of any token seen in the last
    /// [`Self::repeat_last_n`] tokens, however often. `0.0` is off; a negative
    /// value favours what was already said.
    pub presence_penalty: f32,
    /// Subtracted from the logit of a token once per time it occurs in the
    /// last [`Self::repeat_last_n`] tokens. `0.0` is off.
    pub frequency_penalty: f32,
    /// Added to a token's logit before anything else looks at it.
    pub logit_bias: Vec<(u32, f32)>,
    pub seed: u64,
}

impl Default for SamplingParams {
    /// **`repeat_penalty` is `1.0`, which means the penalty is off.**
    ///
    /// It was `1.1`, and that is a bad default for anything structured. The
    /// penalty is applied per *token id*, so it falls hardest on whichever
    /// token repeats most — and in source code that is the **newline**. What
    /// the model reaches for once the newline has been pushed down is
    /// whatever else is plausible there, which for a block comment is a rule
    /// of dashes or a `|`; both are already in its top eight at those
    /// positions. The reported symptom was exactly that: generated C coming
    /// back with `-------------------------` runs and `|` characters where
    /// the line breaks belonged.
    ///
    /// Measured on `gemma-4-26B-A4B`, same server and prompt, penalty the
    /// only difference: `1.1` continues a block comment with
    /// `'---------------------'`, `1.0` continues it with `'\n * @param'` —
    /// which is what real `llama.cpp` produces for the same tokens. Over 90
    /// tokens of C, `1.0` emits twice the newlines.
    ///
    /// `1.0` is also what `llama.cpp` itself defaults to, so a prompt now
    /// behaves the same way through either engine unless a caller asks for
    /// otherwise. Repetition is a real failure mode on some workloads, but it
    /// is the caller's to opt into rather than something to impose on every
    /// request — and imposing it silently corrupts the one output format
    /// whose whitespace is load-bearing.
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            logit_bias: Vec::new(),
            seed: 0,
        }
    }
}

impl SamplingParams {
    /// The base sampling parameters an HTTP request's own (all optional)
    /// `temperature`/`top_p`/`top_k`/`min_p` fields override — `config::
    /// Role::Explorer`'s mapped `llama-server --temp 0.7 --top-p 0.8
    /// --top-k 20 --min-p 0` (tuned for broader, more varied output);
    /// every other role keeps this type's own [`Default`].
    pub fn default_for_role(role: crate::config::Role) -> Self {
        match role {
            crate::config::Role::Explorer => Self {
                temperature: 0.7,
                top_k: 20,
                top_p: 0.8,
                min_p: 0.0,
                ..Self::default()
            },
            crate::config::Role::All
            | crate::config::Role::Code
            | crate::config::Role::Review
            | crate::config::Role::Embedding
            | crate::config::Role::Image
            | crate::config::Role::Video
            | crate::config::Role::Audio => Self::default(),
        }
    }
}

/// The grammar a constraint holds output to.
enum Grammar {
    /// Any JSON object — `response_format: json_object`.
    Json(crate::engine::constraint::JsonPrefix),
    /// A document a JSON Schema describes.
    Schema(Box<crate::engine::json_schema::SchemaPrefix>),
    /// The start of a reply, held to one of a fixed set of strings.
    OneOf(crate::engine::constraint::OneOfPrefix),
    /// What a request's `grammar` describes.
    Text(crate::engine::grammar::Matcher),
    /// One tool call, after which the constraint lets go.
    Call(crate::engine::grammar::Matcher),
}

impl Grammar {
    fn is_complete(&self) -> bool {
        match self {
            Self::Json(g) => g.is_complete(),
            Self::Schema(g) => g.is_complete(),
            Self::OneOf(g) => g.is_complete(),
            Self::Text(g) | Self::Call(g) => g.is_complete(),
        }
    }

    /// Whether a reply ends the moment it is complete. A document does; a
    /// grammar can describe text that goes on, and leaves that to the model.
    fn ends_when_complete(&self) -> bool {
        match self {
            Self::Json(_) | Self::Schema(_) | Self::OneOf(_) | Self::Call(_) => true,
            Self::Text(g) => !g.can_continue(),
        }
    }

    fn allows(&mut self, bytes: &[u8]) -> bool {
        match self {
            Self::Json(g) => g.allows(bytes),
            Self::Schema(g) => g.allows(bytes),
            Self::OneOf(g) => g.allows(bytes),
            Self::Text(g) | Self::Call(g) => g.allows(bytes),
        }
    }

    fn push_bytes(&mut self, bytes: &[u8]) -> bool {
        match self {
            Self::Json(g) => g.push_bytes(bytes),
            Self::Schema(g) => g.push_bytes(bytes),
            Self::OneOf(g) => g.push_bytes(bytes),
            Self::Text(g) | Self::Call(g) => g.push_bytes(bytes),
        }
    }
}

/// What a constrained request is holding while it generates.
pub struct Constraint {
    grammar: Grammar,
    /// Every token's emitted bytes, from the tokenizer — shared, built once.
    token_bytes: std::sync::Arc<Vec<Vec<u8>>>,
    /// Ids that end generation. Masked until the document is complete, so a
    /// model cannot stop half-way through an object and leave a caller
    /// parsing `{"a":`.
    stop_ids: Vec<u32>,
}

impl Constraint {
    pub fn json(token_bytes: std::sync::Arc<Vec<Vec<u8>>>, stop_ids: Vec<u32>) -> Self {
        Self {
            grammar: Grammar::Json(crate::engine::constraint::JsonPrefix::object()),
            token_bytes,
            stop_ids,
        }
    }

    /// Holds output to a document `schema` describes.
    pub fn schema(
        schema: std::sync::Arc<crate::engine::json_schema::Schema>,
        token_bytes: std::sync::Arc<Vec<Vec<u8>>>,
        stop_ids: Vec<u32>,
    ) -> Self {
        Self {
            grammar: Grammar::Schema(Box::new(crate::engine::json_schema::SchemaPrefix::new(
                schema,
            ))),
            token_bytes,
            stop_ids,
        }
    }

    /// Holds the start of the reply to one of `candidates`, and lets go after
    /// it: the rest of the reply is the model's own.
    pub fn one_of(
        candidates: Vec<Vec<u8>>,
        token_bytes: std::sync::Arc<Vec<Vec<u8>>>,
        stop_ids: Vec<u32>,
    ) -> Self {
        Self {
            grammar: Grammar::OneOf(crate::engine::constraint::OneOfPrefix::new(candidates)),
            token_bytes,
            stop_ids,
        }
    }

    /// Holds output to what `grammar` describes.
    pub fn grammar(
        grammar: std::sync::Arc<crate::engine::grammar::Grammar>,
        token_bytes: std::sync::Arc<Vec<Vec<u8>>>,
        stop_ids: Vec<u32>,
    ) -> Self {
        Self {
            grammar: Grammar::Text(crate::engine::grammar::Matcher::new(grammar)),
            token_bytes,
            stop_ids,
        }
    }

    /// Holds output to one tool call, and lets go at its end.
    fn call(
        grammar: std::sync::Arc<crate::engine::grammar::Grammar>,
        token_bytes: std::sync::Arc<Vec<Vec<u8>>>,
        stop_ids: Vec<u32>,
    ) -> Self {
        Self {
            grammar: Grammar::Call(crate::engine::grammar::Matcher::new(grammar)),
            token_bytes,
            stop_ids,
        }
    }

    /// Whether this constraint has nothing left to hold.
    fn is_released(&self) -> bool {
        match &self.grammar {
            Grammar::OneOf(g) => g.is_finished(),
            Grammar::Call(g) => g.is_complete() || !g.can_continue(),
            _ => false,
        }
    }

    /// Whether this token may be sampled next.
    fn allows(&mut self, id: u32) -> bool {
        if self.stop_ids.contains(&id) {
            return self.grammar.is_complete();
        }
        // Once the document is complete, **only** stopping is allowed. The
        // grammar would go on accepting trailing whitespace forever, and a
        // model that has just closed its object reaches for a newline far more
        // readily than for end-of-sequence — measured, before this: `{}`
        // followed by four hundred characters of blank lines, running all the
        // way to `max_tokens`. A constrained request asked for one document;
        // once it has one there is nothing left to generate.
        //
        // Skipped when the request declared no stop tokens at all, since then
        // masking everything would leave the sampler with no legal move.
        //
        // Whitespace *before* and *within* the document stays legal, and that
        // is deliberate. Banning whitespace-only tokens was tried and made the
        // output worse: with no room to hesitate the model opens `{` on the
        // first step and, having committed with nothing planned, closes it
        // again — `{}` where allowing it to pause produced
        // `{"Name": "John Doe", "Age": 32}`.
        if self.grammar.is_complete()
            && self.grammar.ends_when_complete()
            && !self.stop_ids.is_empty()
        {
            return false;
        }
        match self.token_bytes.get(id as usize) {
            // A token with no printable bytes (a special marker) cannot make
            // the document invalid, but it also must not appear inside one —
            // letting it through is how a `<|im_end|>` ends up in the middle
            // of a string.
            Some(bytes) if bytes.is_empty() => false,
            Some(bytes) => self.grammar.allows(bytes),
            None => false,
        }
    }

    /// Commits a token that was actually chosen.
    fn accept(&mut self, id: u32) {
        if self.stop_ids.contains(&id) {
            return;
        }
        if let Some(bytes) = self.token_bytes.get(id as usize) {
            self.grammar.push_bytes(bytes);
        }
    }
}

/// The model's own distribution at one generated position, in log
/// probabilities: what `logprobs` reports.
///
/// Taken over the whole vocabulary from the logits as the model wrote them,
/// before temperature, penalties, bias and truncation: those decide what gets
/// picked, and a figure that moved with them could not be compared across
/// requests.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenLogprob {
    pub id: u32,
    pub logprob: f32,
    /// The most likely tokens at this position, most likely first.
    pub top: Vec<(u32, f32)>,
}

impl TokenLogprob {
    fn from_logits(logits: &[f32], id: u32, top: usize) -> Self {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let norm = logits.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
        let logprob_of = |v: f32| v - max - norm;
        let mut best: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v))
            .collect();
        let top = top.min(best.len());
        if top > 0 && top < best.len() {
            best.select_nth_unstable_by(top - 1, |a, b| b.1.total_cmp(&a.1));
        }
        best.truncate(top);
        best.sort_by(|a, b| b.1.total_cmp(&a.1));
        for (_, v) in best.iter_mut() {
            *v = logprob_of(*v);
        }
        Self {
            id,
            logprob: logits
                .get(id as usize)
                .map_or(f32::NEG_INFINITY, |&v| logprob_of(v)),
            top: best,
        }
    }
}

/// A tool-call grammar waiting for a call to open. While it waits, a reply is
/// ordinary text, on the device's fast paths.
struct WaitingCall {
    calls: std::sync::Arc<crate::engine::tool_learn::CallGrammar>,
    token_bytes: std::sync::Arc<Vec<Vec<u8>>>,
    stop_ids: Vec<u32>,
    /// The latest text, as long as the trigger could still be spanning it.
    tail: Vec<u8>,
    /// Text inside a reasoning body can mention a call without making one.
    paused: bool,
}

pub struct Sampler {
    params: SamplingParams,
    /// `None` for an unconstrained request, which is every request that did
    /// not ask for a `response_format`.
    constraint: Option<Constraint>,
    waiting_call: Option<WaitingCall>,
    /// The token the next pick is, whatever the logits say — see
    /// [`Self::force`].
    forced: Option<u32>,
    /// How many alternatives a recorded [`TokenLogprob`] lists; `None` when
    /// the request did not ask for any.
    logprobs: Option<usize>,
    /// The picks made and not yet claimed by [`Self::take_logprob`]. A pick
    /// comes ahead of the token's turn to be emitted, by several tokens when a
    /// speculative step verifies a draft.
    recorded: std::collections::VecDeque<TokenLogprob>,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        Self {
            params,
            constraint: None,
            waiting_call: None,
            forced: None,
            logprobs: None,
            recorded: std::collections::VecDeque::new(),
        }
    }

    /// Records each pick's [`TokenLogprob`], listing `top` alternatives.
    pub fn with_logprobs(mut self, top: usize) -> Self {
        self.logprobs = Some(top);
        self
    }

    /// The record of the pick that chose `id`, when it is next in line. A token
    /// no pick chose has none: [`Self::force`] writes one without looking at
    /// the logits.
    pub fn take_logprob(&mut self, id: u32) -> Option<TokenLogprob> {
        if self.recorded.front()?.id == id {
            self.recorded.pop_front()
        } else {
            None
        }
    }

    /// Makes `token` the next pick, once — how a reasoning budget closes
    /// the reasoning. The steps until then sample here rather than on the
    /// device, as [`Self::needs_host_logits`] says.
    pub fn force(&mut self, token: u32) {
        self.forced = Some(token);
    }

    /// Restricts this sampler to tokens the constraint still allows.
    pub fn with_constraint(mut self, constraint: Constraint) -> Self {
        self.constraint = Some(constraint);
        self
    }

    /// Holds tool calls to `calls`: from the start where it says what a forced
    /// call must be, and whenever a call opens after that.
    pub fn with_calls(
        mut self,
        calls: std::sync::Arc<crate::engine::tool_learn::CallGrammar>,
        token_bytes: std::sync::Arc<Vec<Vec<u8>>>,
        stop_ids: Vec<u32>,
    ) -> Self {
        if let Some(initial) = calls.initial.clone() {
            self.constraint = Some(Constraint::call(
                initial,
                token_bytes.clone(),
                stop_ids.clone(),
            ));
        }
        self.waiting_call = Some(WaitingCall {
            calls,
            token_bytes,
            stop_ids,
            tail: Vec::new(),
            paused: false,
        });
        self
    }

    /// Whether what is being written is reasoning, where a call's opener is
    /// only a mention of one.
    pub fn set_reasoning(&mut self, reasoning: bool) {
        if let Some(waiting) = self.waiting_call.as_mut() {
            waiting.paused = reasoning;
            if reasoning {
                waiting.tail.clear();
            }
        }
    }

    /// A token the device picked, which no pick here saw.
    pub fn note_device_pick(&mut self, id: u32) {
        self.accept_into_constraint(id);
    }

    /// Whether a pick needs every logit on the host — the decode loop's cue
    /// that its GPU fast paths cannot be used, since they know the repeat
    /// penalty and `top_k` and nothing else: not a grammar's mask, a forced
    /// token, a bias, the other two penalties, or what `logprobs` reports.
    pub fn needs_host_logits(&self) -> bool {
        self.constraint.is_some()
            || self.forced.is_some()
            || self.logprobs.is_some()
            || !self.params.logit_bias.is_empty()
            || self.params.presence_penalty != 0.0
            || self.params.frequency_penalty != 0.0
    }

    /// `true` iff `sample` would take its argmax fast path (`temperature
    /// <= 0.0`) — the only case `engine::arch::ModelForward::forward_
    /// maybe_sampling`'s GPU fast path can replicate; top-k/top-p/min-p
    /// stay CPU-only.
    pub fn is_greedy(&self) -> bool {
        self.params.temperature <= 0.0
    }

    pub fn repeat_penalty(&self) -> f32 {
        self.params.repeat_penalty
    }

    pub fn repeat_last_n(&self) -> usize {
        self.params.repeat_last_n
    }

    /// Picks the token for absolute position `pos` from `logits` (one score
    /// per vocab id), penalizing any token id present in `recent_tokens`' last
    /// `repeat_last_n` entries.
    ///
    /// The random draw is a function of the seed and `pos` alone
    /// ([`position_draw`]), not of how many draws came before. So the token
    /// chosen for a position depends only on that position's logits, which
    /// is what lets speculative decoding verify a draft under sampling
    /// exactly: the target's choice at each drafted position is the one plain
    /// decoding would have made there.
    pub fn sample(&mut self, logits: &[f32], recent_tokens: &[u32], pos: usize) -> u32 {
        if let Some(id) = self.forced.take() {
            return id;
        }
        let id = self.pick(logits, recent_tokens, pos);
        self.accept_into_constraint(id);
        if let Some(top) = self.logprobs {
            self.recorded
                .push_back(TokenLogprob::from_logits(logits, id, top));
        }
        id
    }

    /// Commits a chosen token to the constraint, and drops one that has let
    /// go, so that the steps after it can use the device's fast paths again.
    fn accept_into_constraint(&mut self, id: u32) {
        if let Some(constraint) = self.constraint.as_mut() {
            constraint.accept(id);
            if constraint.is_released() {
                self.constraint = None;
            }
            return;
        }
        let Some(waiting) = self.waiting_call.as_mut().filter(|w| !w.paused) else {
            return;
        };
        let Some(bytes) = waiting.token_bytes.get(id as usize) else {
            return;
        };
        waiting.tail.extend_from_slice(bytes);
        let trigger = waiting.calls.trigger.as_bytes();
        let Some(at) = waiting
            .tail
            .windows(trigger.len())
            .position(|w| w == trigger)
        else {
            let keep = trigger.len().saturating_sub(1);
            let drop = waiting.tail.len().saturating_sub(keep);
            waiting.tail.drain(..drop);
            return;
        };
        // What the token held past the trigger is the call's first text.
        let rest = waiting.tail.split_off(at + trigger.len());
        waiting.tail.clear();
        let mut constraint = Constraint::call(
            waiting.calls.after_trigger.clone(),
            waiting.token_bytes.clone(),
            waiting.stop_ids.clone(),
        );
        if constraint.grammar.push_bytes(&rest) && !constraint.is_released() {
            self.constraint = Some(constraint);
        }
    }

    fn pick(&mut self, logits: &[f32], recent_tokens: &[u32], pos: usize) -> u32 {
        let mut logits: Vec<f32> = logits.to_vec();
        apply_logit_bias(&mut logits, &self.params.logit_bias);
        apply_repeat_penalty(&mut logits, recent_tokens, &self.params);
        apply_occurrence_penalties(&mut logits, recent_tokens, &self.params);

        if self.params.temperature <= 0.0 {
            return match &mut self.constraint {
                // Exactly the argmax over allowed tokens, and usually one
                // check: walking the vocabulary in descending logit order and
                // stopping at the first token the grammar accepts is the same
                // answer as masking everything else to negative infinity, at a
                // fraction of the work. The model's preferred token is legal
                // far more often than not.
                Some(c) => argmax_allowed(&logits, c),
                None => argmax(&logits),
            };
        }

        for v in logits.iter_mut() {
            *v /= self.params.temperature;
        }

        let mut candidates: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v))
            .collect();

        // A full `sort_by` here is an O(n log n) pass over the entire
        // vocabulary (262k tokens for Gemma) on every sampled token. When
        // top_k narrows the field first, partition around the k-th largest
        // logit in O(n) with `select_nth_unstable_by` and only sort that
        // small prefix — top_p/min_p below still need descending order,
        // just over `top_k` elements instead of the whole vocab.
        if self.params.top_k > 0 && self.params.top_k < candidates.len() {
            let k = self.params.top_k;
            candidates.select_nth_unstable_by(k - 1, |a, b| b.1.total_cmp(&a.1));
            candidates.truncate(k);
        }
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

        // Masked here — after `top_k`, before the softmax — so the surviving
        // probabilities are renormalized over the allowed tokens rather than
        // over all of them. Filtering after `top_p`/`min_p` instead would let
        // those thresholds spend their budget on tokens the grammar was going
        // to reject anyway.
        //
        // Exact given `top_k`, which is the truncation the caller already
        // asked for (40 by default). If every one of the top `k` is rejected,
        // the field is refilled from the whole vocabulary in descending order
        // rather than failing — a legal token always exists while the document
        // is unfinished, it may simply be an unlikely one.
        if let Some(c) = &mut self.constraint {
            candidates.retain(|&(id, _)| c.allows(id));
            if candidates.is_empty() {
                let mut all: Vec<(u32, f32)> = logits
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (i as u32, v))
                    .filter(|&(id, _)| c.allows(id))
                    .collect();
                all.sort_by(|a, b| b.1.total_cmp(&a.1));
                all.truncate(self.params.top_k.max(1));
                candidates = all;
            }
        }

        softmax_pairs(&mut candidates);
        apply_top_p(&mut candidates, self.params.top_p);
        apply_min_p(&mut candidates, self.params.min_p);

        self.draw(&candidates, pos)
    }

    /// One weighted draw over `candidates`, whose second field is a
    /// probability, for position `pos`.
    fn draw(&self, candidates: &[(u32, f32)], pos: usize) -> u32 {
        let total: f32 = candidates.iter().map(|(_, p)| p).sum();
        let mut draw = position_draw(self.params.seed, pos) * total;
        for &(id, p) in candidates {
            draw -= p;
            if draw <= 0.0 {
                return id;
            }
        }
        candidates.first().map(|(id, _)| *id).unwrap_or(0)
    }

    /// How many candidates the device may pre-select for this sampler — its
    /// `top_k` when the whole pick can be made from those alone, `0` when
    /// it cannot (greedy, no `top_k`, a `top_k` past the device's cap, or
    /// [`Self::needs_host_logits`]). See
    /// [`Self::sample_from_candidates`].
    pub fn device_top_k(&self) -> u32 {
        const CAP: usize = 64;
        // `ORANGU_NO_DEVICE_TOPK=1` keeps the whole-vocabulary path, as the
        // control arm of a sweep.
        static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *OFF.get_or_init(|| crate::engine::env::flag_on("ORANGU_NO_DEVICE_TOPK")) {
            return 0;
        }
        if self.is_greedy() || self.needs_host_logits() {
            return 0;
        }
        let k = self.params.top_k;
        if k == 0 || k > CAP {
            return 0;
        }
        k as u32
    }

    /// [`Self::sample`] for a step whose repeat penalty and `top_k` were
    /// already applied by the device, which handed back the `top_k`
    /// `(token, logit)` pairs ([`Self::device_top_k`] said how many). What
    /// is left is exactly what [`Self::pick`] does past its own `top_k`:
    /// temperature, the descending order, softmax, `top_p`, `min_p`, the
    /// draw — the same candidates in the same order, so the same
    /// distribution.
    pub fn sample_from_candidates(&mut self, candidates: Vec<(u32, f32)>, pos: usize) -> u32 {
        let mut candidates = candidates;
        for (_, v) in candidates.iter_mut() {
            *v /= self.params.temperature;
        }
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        softmax_pairs(&mut candidates);
        apply_top_p(&mut candidates, self.params.top_p);
        apply_min_p(&mut candidates, self.params.min_p);
        let id = self.draw(&candidates, pos);
        self.accept_into_constraint(id);
        id
    }
}

/// A uniform number in `[0, 1)` for the draw at absolute position `pos`
/// under `seed`: SplitMix64 over the two, so every position has its own
/// independent draw and nothing depends on the order draws are made in.
pub fn position_draw(seed: u64, pos: usize) -> f32 {
    let mut z = seed
        ^ (pos as u64)
            .wrapping_add(1)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // The top 24 bits: exactly representable in an `f32`, so the value is
    // strictly below 1.
    (z >> 40) as f32 / (1u64 << 24) as f32
}

/// The highest-scoring token the constraint still allows.
///
/// Falls back to the plain argmax when nothing is allowed, which should not
/// happen while a document is unfinished — but a sampler that returned no
/// token at all would hang the decode loop, and a wrong token is recoverable
/// where a hang is not.
fn argmax_allowed(logits: &[f32], constraint: &mut Constraint) -> u32 {
    // The overwhelmingly common case: the model's own preferred token is
    // already legal, so one grammar probe settles it and nothing is sorted.
    // Sorting 262k entries on every generated token to answer a question that
    // is almost always "yes" would make the constraint cost more than the
    // forward pass it decorates.
    let best = argmax(logits);
    if constraint.allows(best) {
        return best;
    }
    let mut order: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as u32, v))
        .collect();
    order.sort_by(|a, b| b.1.total_cmp(&a.1));
    order
        .iter()
        .find(|&&(id, _)| constraint.allows(id))
        .map(|&(id, _)| id)
        .unwrap_or(best)
}

pub(crate) fn argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}

fn apply_repeat_penalty(logits: &mut [f32], recent_tokens: &[u32], params: &SamplingParams) {
    if params.repeat_penalty == 1.0 || recent_tokens.is_empty() {
        return;
    }
    let start = recent_tokens.len().saturating_sub(params.repeat_last_n);
    for &tok in &recent_tokens[start..] {
        if let Some(v) = logits.get_mut(tok as usize) {
            *v = if *v > 0.0 {
                *v / params.repeat_penalty
            } else {
                *v * params.repeat_penalty
            };
        }
    }
}

/// Adds each token's bias to its logit. An id past the vocabulary is refused
/// where the request is read; one that reaches here does nothing.
fn apply_logit_bias(logits: &mut [f32], bias: &[(u32, f32)]) {
    for &(id, value) in bias {
        if let Some(v) = logits.get_mut(id as usize) {
            *v += value;
        }
    }
}

/// `presence_penalty` and `frequency_penalty`, over the same window as the
/// repeat penalty. Unlike it they subtract, so they treat a token the same
/// whichever side of zero its logit is on.
fn apply_occurrence_penalties(logits: &mut [f32], recent_tokens: &[u32], params: &SamplingParams) {
    if (params.presence_penalty == 0.0 && params.frequency_penalty == 0.0)
        || recent_tokens.is_empty()
    {
        return;
    }
    let start = recent_tokens.len().saturating_sub(params.repeat_last_n);
    let mut counts: rustc_hash::FxHashMap<u32, u32> = rustc_hash::FxHashMap::default();
    for &tok in &recent_tokens[start..] {
        *counts.entry(tok).or_default() += 1;
    }
    for (tok, count) in counts {
        if let Some(v) = logits.get_mut(tok as usize) {
            *v -= params.presence_penalty + params.frequency_penalty * count as f32;
        }
    }
}

/// `tensor::softmax_inplace`, over `(token, logit)` pairs.
///
/// A second copy of that computation rather than a call to it, because the
/// values are interleaved with their token ids and extracting them would
/// mean an allocation per sample. **The two must agree bit for bit** — the
/// same max, the same left-to-right accumulation, the same `sum > 0.0`
/// guard — and `sampling_softmax_agrees_with_the_tensor_one` is what holds
/// them together.
fn softmax_pairs(candidates: &mut [(u32, f32)]) {
    let max = candidates
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for (_, v) in candidates.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for (_, v) in candidates.iter_mut() {
            *v /= sum;
        }
    }
}

/// Keeps the smallest prefix of `candidates` (already sorted by descending
/// probability) whose cumulative probability reaches `top_p`.
fn apply_top_p(candidates: &mut Vec<(u32, f32)>, top_p: f32) {
    if top_p >= 1.0 {
        return;
    }
    let mut cumulative = 0.0;
    let mut cutoff = candidates.len();
    for (i, &(_, p)) in candidates.iter().enumerate() {
        cumulative += p;
        if cumulative >= top_p {
            cutoff = i + 1;
            break;
        }
    }
    candidates.truncate(cutoff.max(1));
}

/// Drops any candidate whose probability is below `min_p * max_probability`
/// — llama.cpp's min-p sampler.
fn apply_min_p(candidates: &mut Vec<(u32, f32)>, min_p: f32) {
    if min_p <= 0.0 || candidates.is_empty() {
        return;
    }
    let max_p = candidates[0].1;
    let threshold = min_p * max_p;
    let keep = candidates
        .iter()
        .take_while(|(_, p)| *p >= threshold)
        .count();
    candidates.truncate(keep.max(1));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two softmaxes in this engine — `tensor::softmax_inplace` over a
    /// row and [`softmax_pairs`] over `(token, logit)` pairs — are separate
    /// copies of one computation, so nothing but a test stops them drifting.
    ///
    /// Bit-for-bit, not approximately: this one decides sampled probabilities
    /// and the other decides attention weights, and "close enough" between
    /// two copies of the same formula is how a divergence hides.
    #[test]
    fn sampling_softmax_agrees_with_the_tensor_one() {
        let mut lcg = 0x2468_ACE0u32;
        let mut next = || {
            lcg = lcg.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (lcg >> 8) as f32 / 8_388_608.0 - 0.5
        };
        // The last row makes the accumulation order decisive: one element at
        // the maximum (exponential exactly `1.0`) and three hundred an order
        // of magnitude below its half-ulp. Left to right they are all
        // swallowed; in any other order they accumulate first and survive.
        // A random spread alone does *not* separate the two orders — that
        // was checked by mutation, not assumed.
        let mut rows: Vec<Vec<f32>> = [0usize, 1, 2, 7, 8, 33, 129]
            .iter()
            .map(|&len| (0..len).map(|i| next() * 25.0 - (i % 5) as f32).collect())
            .collect();
        rows.push(
            std::iter::once(0.0f32)
                .chain(std::iter::repeat_n(-18.4f32, 300))
                .collect(),
        );

        for logits in rows {
            let len = logits.len();

            let mut pairs: Vec<(u32, f32)> = logits
                .iter()
                .enumerate()
                .map(|(i, &v)| (i as u32, v))
                .collect();
            softmax_pairs(&mut pairs);

            let mut row = logits.clone();
            crate::engine::tensor::softmax_inplace(&mut row);

            assert_eq!(
                pairs.iter().map(|(_, v)| v.to_bits()).collect::<Vec<_>>(),
                row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "len {len}: the sampler's softmax has drifted from the tensor one"
            );
        }
        // An all `-inf` row. `-inf - -inf` is `NaN`, so both copies return
        // `NaN` rather than zeros — see `tensor::softmax_inplace` for why
        // that is unreachable in production and deliberately not
        // special-cased. What this pins is that the two copies agree *there
        // too*, which is where a divergence would be easiest to introduce by
        // "fixing" one of them alone.
        let mut pairs: Vec<(u32, f32)> = (0..4).map(|i| (i, f32::NEG_INFINITY)).collect();
        softmax_pairs(&mut pairs);
        let mut row = vec![f32::NEG_INFINITY; 4];
        crate::engine::tensor::softmax_inplace(&mut row);
        assert_eq!(
            pairs.iter().map(|(_, v)| v.to_bits()).collect::<Vec<_>>(),
            row.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "the two copies disagree on an all -inf row"
        );
    }

    #[test]
    fn greedy_sampling_is_deterministic_and_picks_the_max() {
        let logits = [0.1, 5.0, 2.0, -1.0];
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(params);
        for _ in 0..5 {
            assert_eq!(sampler.sample(&logits, &[], 0), 1);
        }
    }

    #[test]
    fn repeat_penalty_discourages_a_recently_used_token() {
        let logits = [5.0, 5.0, 5.0];
        let params = SamplingParams {
            temperature: 0.0,
            repeat_penalty: 1.5,
            ..Default::default()
        };
        let mut sampler = Sampler::new(params);
        // Token 0 was just used; greedy sampling with equal raw logits
        // should now prefer a different (unpenalized) token.
        let chosen = sampler.sample(&logits, &[0], 1);
        assert_ne!(chosen, 0);
    }

    #[test]
    fn same_seed_reproduces_the_same_sequence() {
        let logits = [1.0, 2.0, 3.0, 0.5, 0.1];
        let params = SamplingParams {
            temperature: 1.0,
            seed: 42,
            ..Default::default()
        };
        let mut a = Sampler::new(params.clone());
        let mut b = Sampler::new(params);
        let seq_a: Vec<u32> = (0..10).map(|p| a.sample(&logits, &[], p)).collect();
        let seq_b: Vec<u32> = (0..10).map(|p| b.sample(&logits, &[], p)).collect();
        assert_eq!(seq_a, seq_b);
    }

    /// A position's draw does not depend on which positions were drawn
    /// before it — the property exact speculative verification rests on.
    #[test]
    fn a_positions_draw_is_independent_of_the_draws_before_it() {
        let logits = [1.0, 2.0, 3.0, 0.5, 0.1];
        let params = SamplingParams {
            temperature: 1.0,
            seed: 7,
            ..Default::default()
        };
        let mut in_order = Sampler::new(params.clone());
        let all: Vec<u32> = (0..32).map(|p| in_order.sample(&logits, &[], p)).collect();
        let mut alone = Sampler::new(params);
        for p in [31usize, 3, 17, 0] {
            assert_eq!(alone.sample(&logits, &[], p), all[p], "position {p}");
        }
        // And the draws are spread over [0, 1).
        let draws: Vec<f32> = (0..10_000).map(|p| position_draw(7, p)).collect();
        let mean = draws.iter().sum::<f32>() / draws.len() as f32;
        assert!((mean - 0.5).abs() < 0.02, "mean draw {mean}");
        assert!(draws.iter().all(|&d| (0.0..1.0).contains(&d)));
    }

    fn greedy() -> SamplingParams {
        SamplingParams {
            temperature: 0.0,
            ..Default::default()
        }
    }

    #[test]
    fn presence_penalty_counts_a_token_once_and_frequency_per_occurrence() {
        let params = |presence_penalty, frequency_penalty| SamplingParams {
            presence_penalty,
            frequency_penalty,
            ..greedy()
        };
        let recent = [0, 0, 0, 1];
        let mut logits = vec![10.0, 10.0, 8.0];
        apply_occurrence_penalties(&mut logits, &recent, &params(1.0, 0.0));
        assert_eq!(logits, [9.0, 9.0, 8.0]);
        let mut logits = vec![10.0, 10.0, 8.0];
        apply_occurrence_penalties(&mut logits, &recent, &params(0.0, 1.0));
        assert_eq!(logits, [7.0, 9.0, 8.0]);
        // A negative penalty favours what was said.
        let mut sampler = Sampler::new(params(-2.0, 0.0));
        assert_eq!(sampler.sample(&[1.0, 2.0, 0.0], &[0], 1), 0);
    }

    /// A penalty looks at the repeat window and not at the whole history.
    #[test]
    fn occurrence_penalties_stop_at_the_repeat_window() {
        let params = SamplingParams {
            frequency_penalty: 1.0,
            repeat_last_n: 2,
            ..greedy()
        };
        let mut logits = vec![5.0, 5.0];
        apply_occurrence_penalties(&mut logits, &[0, 0, 0, 1, 1], &params);
        assert_eq!(logits, [5.0, 3.0]);
    }

    /// A bias decides the pick before temperature, on both paths.
    #[test]
    fn logit_bias_can_ban_a_token_and_can_force_one() {
        let logits = [5.0, 1.0, 0.0];
        for temperature in [0.0, 0.8] {
            let bias = |logit_bias| SamplingParams {
                temperature,
                logit_bias,
                ..Default::default()
            };
            let mut banned = Sampler::new(bias(vec![(0, -100.0), (2, -100.0)]));
            assert_eq!(banned.sample(&logits, &[], 0), 1);
            let mut forced = Sampler::new(bias(vec![(2, 100.0)]));
            assert_eq!(forced.sample(&logits, &[], 0), 2);
        }
        // An id past the vocabulary is nothing, not a panic.
        let mut outside = Sampler::new(SamplingParams {
            logit_bias: vec![(99, 100.0)],
            ..greedy()
        });
        assert_eq!(outside.sample(&logits, &[], 0), 0);
    }

    /// What the device cannot do must keep the pick on the host: a greedy
    /// device argmax would otherwise answer without any of it.
    #[test]
    fn what_only_the_host_can_do_takes_the_device_paths_away() {
        let top_k = SamplingParams::default();
        assert!(!Sampler::new(top_k.clone()).needs_host_logits());
        assert!(Sampler::new(top_k.clone()).device_top_k() > 0);
        for params in [
            SamplingParams {
                presence_penalty: 0.5,
                ..top_k.clone()
            },
            SamplingParams {
                frequency_penalty: 0.5,
                ..top_k.clone()
            },
            SamplingParams {
                logit_bias: vec![(0, 1.0)],
                ..top_k.clone()
            },
        ] {
            let sampler = Sampler::new(params);
            assert!(sampler.needs_host_logits());
            assert_eq!(sampler.device_top_k(), 0);
        }
        let asked = Sampler::new(top_k).with_logprobs(0);
        assert!(asked.needs_host_logits());
        assert_eq!(asked.device_top_k(), 0);
    }

    /// The figures are the model's: temperature and bias move the pick, never
    /// what is reported about it.
    #[test]
    fn logprobs_report_the_models_own_distribution() {
        let logits = [2.0f32, 1.0, 0.0, -1.0];
        let mut sampler = Sampler::new(SamplingParams {
            logit_bias: vec![(2, 100.0)],
            ..greedy()
        })
        .with_logprobs(2);
        assert_eq!(sampler.sample(&logits, &[], 0), 2);
        let record = sampler.take_logprob(2).expect("recorded");

        let exp: Vec<f32> = logits.iter().map(|v| v.exp()).collect();
        let sum: f32 = exp.iter().sum();
        assert!((record.logprob - (exp[2] / sum).ln()).abs() < 1e-5);
        assert_eq!(record.id, 2);
        assert_eq!(record.top.len(), 2);
        assert_eq!(record.top[0].0, 0, "most likely first");
        assert_eq!(record.top[1].0, 1);
        assert!((record.top[0].1 - (exp[0] / sum).ln()).abs() < 1e-5);
        assert!(sampler.take_logprob(2).is_none(), "claimed once");
    }

    /// Picks made ahead of their turn, as a verified draft makes them, are
    /// claimed in order; a token no pick chose has no record and takes nobody
    /// else's.
    #[test]
    fn logprob_records_are_claimed_in_the_order_they_were_picked() {
        let mut sampler = Sampler::new(greedy()).with_logprobs(0);
        assert_eq!(sampler.sample(&[1.0, 2.0], &[], 0), 1);
        assert_eq!(sampler.sample(&[2.0, 1.0], &[], 1), 0);
        assert!(sampler.take_logprob(7).is_none());
        assert_eq!(sampler.take_logprob(1).map(|r| r.id), Some(1));
        sampler.force(5);
        assert_eq!(sampler.sample(&[1.0, 2.0], &[], 2), 5);
        assert!(sampler.take_logprob(5).is_none());
        assert_eq!(sampler.take_logprob(0).map(|r| r.id), Some(0));
    }

    #[test]
    fn top_p_keeps_at_least_one_candidate() {
        let mut candidates = vec![(0u32, 0.9f32), (1, 0.05), (2, 0.05)];
        apply_top_p(&mut candidates, 0.0001);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].0, 0);
    }
}

#[cfg(test)]
mod constraint_tests {
    use super::*;
    use std::sync::Arc;

    /// A tiny vocabulary whose ids are indices into this table.
    fn vocab() -> Arc<Vec<Vec<u8>>> {
        Arc::new(vec![
            b"{".to_vec(),     // 0
            b"}".to_vec(),     // 1
            b"\"a\"".to_vec(), // 2
            b":".to_vec(),     // 3
            b"1".to_vec(),     // 4
            b"hello".to_vec(), // 5
            b" ".to_vec(),     // 6
            Vec::new(),        // 7 — a special token with no printable bytes
        ])
    }
    const EOS: u32 = 99;

    fn tool_name_sampler(temperature: f32) -> Sampler {
        let names = vec![b"ab\"".to_vec(), b"abc\"".to_vec()];
        let table = Arc::new(vec![
            b"a".to_vec(),
            b"b".to_vec(),
            b"c".to_vec(),
            b"\"".to_vec(),
            b"zz".to_vec(),
            b"bc\",".to_vec(),
            Vec::new(),
        ]);
        let params = SamplingParams {
            temperature,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repeat_penalty: 1.0,
            ..Default::default()
        };
        Sampler::new(params).with_constraint(Constraint::one_of(names, table, vec![6]))
    }

    /// The model wants the undeclared token, then to stop, then something
    /// else again; only a declared name can come out, and what follows it is
    /// the model's own, with the device's fast paths available again.
    #[test]
    fn a_tool_name_comes_out_declared_and_then_the_constraint_lets_go() {
        for temperature in [0.0, 0.7] {
            let mut sampler = tool_name_sampler(temperature);
            assert!(sampler.needs_host_logits());
            let prefers = |wanted: usize, second: usize| {
                let mut logits = vec![-50.0_f32; 7];
                logits[wanted] = 50.0;
                logits[second] = 40.0;
                logits
            };
            assert_eq!(sampler.sample(&prefers(4, 0), &[], 0), 0);
            assert!(sampler.needs_host_logits());
            assert_eq!(sampler.sample(&prefers(6, 1), &[], 1), 1);
            assert_eq!(sampler.sample(&prefers(4, 3), &[], 2), 3);
            assert!(!sampler.needs_host_logits());
            assert_eq!(sampler.sample(&prefers(4, 0), &[], 3), 4);
        }
    }

    /// One token can hold a name's last letters and what comes after them.
    #[test]
    fn a_token_may_run_past_the_end_of_a_tool_name() {
        let mut sampler = tool_name_sampler(0.0);
        let mut logits = vec![-50.0_f32; 7];
        logits[0] = 1.0;
        assert_eq!(sampler.sample(&logits, &[], 0), 0);
        let mut logits = vec![-50.0_f32; 7];
        logits[5] = 1.0;
        assert_eq!(sampler.sample(&logits, &[], 1), 5);
        assert!(!sampler.needs_host_logits());
    }

    /// A vocabulary of one token per piece of a call, and a sampler for it.
    fn call_sampler(initial: bool) -> Sampler {
        use crate::engine::grammar::Grammar;
        let after = Grammar::parse("root ::= \"x\" \"y\"\n").unwrap();
        let calls = crate::engine::tool_learn::CallGrammar {
            trigger: "<c>".to_string(),
            after_trigger: after.clone(),
            initial: initial.then(|| Grammar::parse("root ::= \"y\"\n").unwrap()),
        };
        let table = Arc::new(vec![
            b"<c>".to_vec(),
            b"x".to_vec(),
            b"y".to_vec(),
            b"z".to_vec(),
            b"<c>x".to_vec(),
            Vec::new(),
        ]);
        let params = SamplingParams {
            temperature: 0.0,
            repeat_penalty: 1.0,
            ..Default::default()
        };
        Sampler::new(params).with_calls(Arc::new(calls), table, vec![5])
    }

    fn wanting(id: usize, second: usize) -> Vec<f32> {
        let mut logits = vec![-50.0_f32; 6];
        logits[id] = 50.0;
        logits[second] = 40.0;
        logits
    }

    /// Until a call opens the reply is ordinary, and on the fast paths; once it
    /// opens, only the call can be written, not even a stop; after the call the
    /// reply is ordinary again.
    #[test]
    fn a_call_grammar_waits_for_the_call_and_lets_go_after_it() {
        let mut sampler = call_sampler(false);
        assert!(!sampler.needs_host_logits());
        assert_eq!(sampler.sample(&wanting(3, 1), &[], 0), 3);
        assert!(!sampler.needs_host_logits());
        assert_eq!(sampler.sample(&wanting(0, 1), &[], 1), 0);
        assert!(sampler.needs_host_logits(), "the call has opened");
        assert_eq!(sampler.sample(&wanting(3, 1), &[], 2), 1);
        assert_eq!(
            sampler.sample(&wanting(5, 2), &[], 3),
            2,
            "no stopping inside"
        );
        assert!(!sampler.needs_host_logits(), "the call has closed");
        assert_eq!(sampler.sample(&wanting(3, 1), &[], 4), 3);
        // And the next call is held the same way.
        assert_eq!(sampler.sample(&wanting(0, 1), &[], 5), 0);
        assert!(sampler.needs_host_logits());
    }

    /// The device's pick goes through nothing here, and has to be seen all the
    /// same, with the text past a trigger in the one token counting as the call.
    #[test]
    fn a_call_is_seen_opening_from_a_device_pick_and_within_a_token() {
        let mut sampler = call_sampler(false);
        sampler.note_device_pick(0);
        assert!(sampler.needs_host_logits());
        let mut sampler = call_sampler(false);
        sampler.note_device_pick(4);
        assert!(sampler.needs_host_logits());
        assert_eq!(sampler.sample(&wanting(3, 2), &[], 0), 2);
        assert!(!sampler.needs_host_logits());
    }

    #[test]
    fn a_call_mentioned_in_reasoning_is_only_a_mention() {
        let mut sampler = call_sampler(false);
        sampler.set_reasoning(true);
        sampler.note_device_pick(0);
        assert!(!sampler.needs_host_logits());
        sampler.set_reasoning(false);
        sampler.note_device_pick(0);
        assert!(sampler.needs_host_logits());
    }

    /// A reply forced to begin with a call is held from its first token.
    #[test]
    fn a_forced_call_is_held_from_the_start() {
        let mut sampler = call_sampler(true);
        assert!(sampler.needs_host_logits());
        assert_eq!(sampler.sample(&wanting(3, 2), &[], 0), 2);
        assert!(!sampler.needs_host_logits());
    }

    /// A grammar that can go on leaves stopping to the model once it may, and
    /// one that cannot leaves only stopping.
    #[test]
    fn a_grammar_says_when_a_reply_may_end() {
        use crate::engine::grammar::Grammar;
        let table = Arc::new(vec![b"a".to_vec(), b"b".to_vec(), Vec::new()]);
        let params = SamplingParams {
            temperature: 0.0,
            repeat_penalty: 1.0,
            ..Default::default()
        };
        let open = Grammar::parse("root ::= \"a\"+\n").unwrap();
        let mut sampler = Sampler::new(params.clone()).with_constraint(Constraint::grammar(
            open,
            table.clone(),
            vec![2],
        ));
        let logits = |id: usize, next: usize| {
            let mut l = vec![-50.0_f32; 3];
            l[id] = 50.0;
            l[next] = 40.0;
            l
        };
        assert_eq!(sampler.sample(&logits(2, 0), &[], 0), 0, "not complete yet");
        assert_eq!(sampler.sample(&logits(0, 2), &[], 1), 0, "it may go on");
        assert_eq!(sampler.sample(&logits(2, 0), &[], 2), 2, "it may stop");

        let closed = Grammar::parse("root ::= \"a\"\n").unwrap();
        let mut sampler =
            Sampler::new(params).with_constraint(Constraint::grammar(closed, table, vec![2]));
        assert_eq!(sampler.sample(&logits(1, 0), &[], 0), 0);
        assert_eq!(
            sampler.sample(&logits(0, 2), &[], 1),
            2,
            "only stopping is left"
        );
    }

    fn json_constraint() -> Constraint {
        Constraint::json(vocab(), vec![EOS])
    }

    /// The document has to open with `{`, and prose cannot get in.
    #[test]
    fn only_tokens_keeping_the_output_valid_are_allowed() {
        let mut c = json_constraint();
        assert!(c.allows(0), "`{{` opens the object");
        assert!(c.allows(6), "leading whitespace is legal JSON");
        assert!(!c.allows(5), "`hello` is not a JSON document");
        assert!(!c.allows(1), "`}}` cannot open one");
        assert!(!c.allows(2), "a bare string is not an object");
    }

    /// A token that decodes to nothing is a special marker, and one of those
    /// landing mid-document is how `<|im_end|>` ends up inside a string.
    #[test]
    fn tokens_with_no_printable_bytes_are_refused() {
        assert!(!json_constraint().allows(7));
    }

    /// Stopping is masked until the document is complete — the failure this
    /// exists to prevent is a model emitting `{"a":` and calling it done.
    #[test]
    fn stopping_is_masked_until_the_document_is_complete() {
        let mut c = json_constraint();
        assert!(!c.allows(EOS), "nothing has been written yet");
        for id in [0, 2, 3] {
            assert!(c.allows(id));
            c.accept(id);
        }
        assert!(!c.allows(EOS), "`{{\"a\":` is a prefix, not a document");
        c.accept(4);
        assert!(!c.allows(EOS), "`{{\"a\":1` still has an object open");
        c.accept(1);
        assert!(c.allows(EOS), "`{{\"a\":1}}` is a whole document");
    }

    /// And once it *is* complete, stopping is the only thing left — otherwise
    /// the model trails whitespace to `max_tokens`.
    #[test]
    fn a_complete_document_allows_nothing_but_stopping() {
        let mut c = json_constraint();
        c.accept(0);
        c.accept(1);
        assert!(c.allows(EOS));
        for id in 0..8u32 {
            assert!(!c.allows(id), "token {id} after a complete document");
        }
    }

    /// With no stop tokens declared there is nothing to steer towards, so the
    /// mask must not close down to nothing — a sampler with no legal move
    /// cannot make progress at all.
    #[test]
    fn a_request_with_no_stop_tokens_is_never_left_without_a_move() {
        let mut c = Constraint::json(vocab(), Vec::new());
        c.accept(0);
        c.accept(1);
        assert!(
            (0..8u32).any(|id| c.allows(id)),
            "every token masked and no stop token to pick"
        );
    }

    /// The sampler honours the mask on the greedy path even when the model
    /// would rather write prose — the logit for `hello` is the highest here.
    #[test]
    fn greedy_sampling_takes_the_best_allowed_token() {
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let mut sampler = Sampler::new(params.clone()).with_constraint(json_constraint());
        // `hello` (5) is the model's favourite; `{` (0) is its second.
        let logits = vec![9.0, 1.0, 1.0, 1.0, 1.0, 10.0, 0.5, 0.0];
        assert_eq!(sampler.sample(&logits, &[], 0), 0, "must not pick `hello`");
        // Unconstrained, the same logits give the prose token.
        let mut plain = Sampler::new(params);
        assert_eq!(plain.sample(&logits, &[], 0), 5);
    }

    /// A schema narrows what JSON alone allows: with `a` a required string,
    /// `1` — valid JSON, the model's favourite — is masked after `{"a":`,
    /// and stopping waits until `a` has its string.
    #[test]
    fn a_schema_masks_what_its_shape_rules_out() {
        let schema = crate::engine::json_schema::compile(&serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
        }))
        .unwrap();
        let mut c = Constraint::schema(Arc::new(schema), vocab(), vec![EOS]);
        for id in [0, 2, 3] {
            assert!(c.allows(id), "token {id}");
            c.accept(id);
        }
        assert!(!c.allows(4), "`1` is JSON, but `a` is a string");
        assert!(c.allows(2), "`\"a\"` is a string value");
        c.accept(2);
        assert!(!c.allows(EOS), "the object is still open");
        c.accept(1);
        assert!(c.allows(EOS), "`{{\"a\":\"a\"}}` is the whole document");
        let empty = crate::engine::json_schema::compile(&serde_json::json!({
            "type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"],
        }))
        .unwrap();
        let mut c = Constraint::schema(Arc::new(empty), vocab(), vec![EOS]);
        c.accept(0);
        assert!(!c.allows(1), "`{{}}` leaves out the required `a`");
    }
}
