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

use rand::{RngExt, SeedableRng, rngs::StdRng};

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
    /// How many of the most recent generated tokens the repeat penalty
    /// looks at.
    pub repeat_last_n: usize,
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
            | crate::config::Role::Embedding => Self::default(),
        }
    }
}

/// What a constrained request is holding while it generates.
pub struct Constraint {
    grammar: crate::engine::constraint::JsonPrefix,
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
            grammar: crate::engine::constraint::JsonPrefix::object(),
            token_bytes,
            stop_ids,
        }
    }

    /// Whether this token may be sampled next.
    fn allows(&self, id: u32) -> bool {
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
        if self.grammar.is_complete() && !self.stop_ids.is_empty() {
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

pub struct Sampler {
    params: SamplingParams,
    rng: StdRng,
    /// `None` for an unconstrained request, which is every request that did
    /// not ask for a `response_format`.
    constraint: Option<Constraint>,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        let rng = StdRng::seed_from_u64(params.seed);
        Self {
            params,
            rng,
            constraint: None,
        }
    }

    /// Restricts this sampler to tokens the constraint still allows.
    pub fn with_constraint(mut self, constraint: Constraint) -> Self {
        self.constraint = Some(constraint);
        self
    }

    /// Whether a constraint is in force — the decode loop's cue that its GPU
    /// argmax fast path cannot be used, since that samples without ever
    /// consulting the mask.
    pub fn is_constrained(&self) -> bool {
        self.constraint.is_some()
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

    /// Picks the next token from `logits` (one score per vocab id),
    /// penalizing any token id present in `recent_tokens`' last
    /// `repeat_last_n` entries.
    pub fn sample(&mut self, logits: &[f32], recent_tokens: &[u32]) -> u32 {
        let id = self.pick(logits, recent_tokens);
        if let Some(constraint) = self.constraint.as_mut() {
            constraint.accept(id);
        }
        id
    }

    fn pick(&mut self, logits: &[f32], recent_tokens: &[u32]) -> u32 {
        let mut logits: Vec<f32> = logits.to_vec();
        apply_repeat_penalty(&mut logits, recent_tokens, &self.params);

        if self.params.temperature <= 0.0 {
            return match &self.constraint {
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
        if let Some(c) = &self.constraint {
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

        let total: f32 = candidates.iter().map(|(_, p)| p).sum();
        let mut draw = self.rng.random::<f32>() * total;
        for &(id, p) in &candidates {
            draw -= p;
            if draw <= 0.0 {
                return id;
            }
        }
        candidates.first().map(|(id, _)| *id).unwrap_or(0)
    }
}

/// The highest-scoring token the constraint still allows.
///
/// Falls back to the plain argmax when nothing is allowed, which should not
/// happen while a document is unfinished — but a sampler that returned no
/// token at all would hang the decode loop, and a wrong token is recoverable
/// where a hang is not.
fn argmax_allowed(logits: &[f32], constraint: &Constraint) -> u32 {
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
            assert_eq!(sampler.sample(&logits, &[]), 1);
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
        let chosen = sampler.sample(&logits, &[0]);
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
        let seq_a: Vec<u32> = (0..10).map(|_| a.sample(&logits, &[])).collect();
        let seq_b: Vec<u32> = (0..10).map(|_| b.sample(&logits, &[])).collect();
        assert_eq!(seq_a, seq_b);
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

    fn json_constraint() -> Constraint {
        Constraint::json(vocab(), vec![EOS])
    }

    /// The document has to open with `{`, and prose cannot get in.
    #[test]
    fn only_tokens_keeping_the_output_valid_are_allowed() {
        let c = json_constraint();
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
        assert_eq!(sampler.sample(&logits, &[]), 0, "must not pick `hello`");
        // Unconstrained, the same logits give the prose token.
        let mut plain = Sampler::new(params);
        assert_eq!(plain.sample(&logits, &[]), 5);
    }
}
