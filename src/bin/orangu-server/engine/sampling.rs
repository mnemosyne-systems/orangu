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

use rand::{Rng, SeedableRng, rngs::StdRng};

#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    /// How many of the most recent generated tokens the repeat penalty
    /// looks at.
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            repeat_penalty: 1.1,
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

pub struct Sampler {
    params: SamplingParams,
    rng: StdRng,
}

impl Sampler {
    pub fn new(params: SamplingParams) -> Self {
        let rng = StdRng::seed_from_u64(params.seed);
        Self { params, rng }
    }

    /// `true` iff `sample` would take its argmax fast path (`temperature
    /// <= 0.0`) — the only case `engine::arch::ModelForward::forward_
    /// maybe_sampling`'s GPU fast path can replicate (`doc/SERVER_OPUS.md`
    /// Section 9's Step 11); top-k/top-p/min-p stay CPU-only.
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
        let mut logits: Vec<f32> = logits.to_vec();
        apply_repeat_penalty(&mut logits, recent_tokens, &self.params);

        if self.params.temperature <= 0.0 {
            return argmax(&logits);
        }

        for v in logits.iter_mut() {
            *v /= self.params.temperature;
        }

        let mut candidates: Vec<(u32, f32)> = logits
            .iter()
            .enumerate()
            .map(|(i, &v)| (i as u32, v))
            .collect();
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));

        if self.params.top_k > 0 && self.params.top_k < candidates.len() {
            candidates.truncate(self.params.top_k);
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

fn argmax(logits: &[f32]) -> u32 {
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
