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

//! A type per tensor, chosen for size and speed: the choice.
//!
//! Every tensor starts at its fastest type. While the file's predicted error
//! is over its budget, the one change that removes the most error for the
//! time it adds is made, a tensor at a time, never past a size limit. Time
//! is a token's decode plus a prompt token's share of prefill, each as a
//! share of the reference file's, so neither drowns the other; added bytes
//! count a little, so of two changes that cost the same time the smaller
//! wins. A file over a size limit at the start is first brought under it by
//! the changes that add the least error and time for the bytes they save.
//!
//! A card runs a model on one kernel family, chosen by timing the whole
//! model, so the choice is made for each family and the faster result kept.
//!
//! The predicted error is the sum of the tensors' error-table entries; its
//! budget defaults to the reference file's own sum (the standard mixture
//! of the same name, with the same importance matrix), which makes the
//! scale between that sum and the KL it predicts irrelevant to the default.
//!
//! The same inputs give the same choice: every scan runs in tensor order
//! and only a strictly better change replaces the one held.

use crate::auto::Cost;

/// One tensor the choice is made for.
#[derive(Clone, Debug)]
pub struct Tensor {
    /// The error table's entries: one per candidate type.
    pub costs: Vec<Cost>,
    /// Microseconds this tensor's products take at each candidate type
    /// (in `costs` order), per kernel family (float, integer dot): one
    /// token's decode, and one prompt token's share of prefill.
    pub time_us: Vec<[[f64; 2]; 2]>,
    /// Whether its bytes are held on the card.
    pub on_device: bool,
}

/// What the chosen file must stay within.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// The most predicted error, summed over the tensors.
    pub max_error: f64,
    /// The most bytes the tensors may take.
    pub max_bytes: u64,
    /// The most bytes the tensors held on the card may take.
    pub max_device_bytes: u64,
    /// The most microseconds a decoded token, and a prompt token, may take:
    /// a faster file is never bought with a slower one of the two.
    pub max_decode_us: f64,
    pub max_prefill_us: f64,
}

/// The chosen type of every tensor and what it is predicted to cost.
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    pub types: Vec<u32>,
    /// The kernel family the times are for: 0 float, 1 integer dot.
    pub family: usize,
    pub error: f64,
    pub bytes: u64,
    pub device_bytes: u64,
    /// Microseconds per decoded token, and per prompt token.
    pub decode_us: f64,
    pub prefill_us: f64,
    /// Whether every limit is met.
    pub within: bool,
}

/// What added bytes cost against added time, both as shares of the
/// reference: a quarter.
const BYTES_WEIGHT: f64 = 0.25;

/// A change that costs nothing is taken before any that costs: its cost is
/// held at this many shares, so of free changes the one removing the most
/// error comes first.
const FREE: f64 = 1e-12;

struct State<'a> {
    tensors: &'a [Tensor],
    family: usize,
    at: Vec<usize>,
    error: f64,
    bytes: u64,
    device_bytes: u64,
    decode: f64,
    prefill: f64,
    /// The reference's decode and prefill microseconds and bytes, the
    /// shares a change is priced in.
    reference: (f64, f64, f64),
}

impl State<'_> {
    fn time(&self, t: usize, c: usize) -> [f64; 2] {
        [
            self.tensors[t].time_us[c][0][self.family],
            self.tensors[t].time_us[c][1][self.family],
        ]
    }

    /// A change's added time and bytes, as shares of the reference.
    fn price(&self, t: usize, to: usize) -> f64 {
        let from = self.at[t];
        let (d0, p0) = (self.time(t, from)[0], self.time(t, from)[1]);
        let (d1, p1) = (self.time(t, to)[0], self.time(t, to)[1]);
        let costs = &self.tensors[t].costs;
        let bytes = costs[to].bytes as f64 - costs[from].bytes as f64;
        let (rd, rp, rb) = self.reference;
        (d1 - d0) / rd + (p1 - p0) / rp + BYTES_WEIGHT * bytes / rb
    }

    fn fits(&self, t: usize, to: usize, limits: &Limits) -> bool {
        let costs = &self.tensors[t].costs;
        let added = costs[to].bytes as i128 - costs[self.at[t]].bytes as i128;
        let bytes = self.bytes as i128 + added;
        let device = self.device_bytes as i128 + if self.tensors[t].on_device { added } else { 0 };
        let (a, b) = (self.time(t, self.at[t]), self.time(t, to));
        bytes <= limits.max_bytes as i128
            && device <= limits.max_device_bytes as i128
            && self.decode + b[0] - a[0] <= limits.max_decode_us
            && self.prefill + b[1] - a[1] <= limits.max_prefill_us
    }

    fn apply(&mut self, t: usize, to: usize) {
        let from = self.at[t];
        let costs = &self.tensors[t].costs;
        self.error += costs[to].error - costs[from].error;
        self.bytes = self.bytes + costs[to].bytes - costs[from].bytes;
        if self.tensors[t].on_device {
            self.device_bytes = self.device_bytes + costs[to].bytes - costs[from].bytes;
        }
        let (a, b) = (self.time(t, from), self.time(t, to));
        self.decode += b[0] - a[0];
        self.prefill += b[1] - a[1];
        self.at[t] = to;
    }

    fn over_size(&self, limits: &Limits) -> bool {
        self.bytes > limits.max_bytes || self.device_bytes > limits.max_device_bytes
    }

    /// Brings the file under its size limits by the changes that add the
    /// least error and time for the bytes they save. `false` when no change
    /// saves bytes any more.
    fn shrink(&mut self, limits: &Limits) -> bool {
        while self.over_size(limits) {
            let mut best: Option<(f64, usize, usize)> = None;
            for t in 0..self.tensors.len() {
                let device_bound = self.device_bytes > limits.max_device_bytes;
                if device_bound && self.bytes <= limits.max_bytes && !self.tensors[t].on_device {
                    continue;
                }
                let costs = &self.tensors[t].costs;
                let from = self.at[t];
                for to in 0..costs.len() {
                    if costs[to].bytes >= costs[from].bytes {
                        continue;
                    }
                    let saved = (costs[from].bytes - costs[to].bytes) as f64 / self.reference.2;
                    let added = (costs[to].error - costs[from].error).max(0.0)
                        / self.error.max(1e-30)
                        + self.price(t, to).max(0.0);
                    let score = added / saved;
                    if best.is_none_or(|(s, _, _)| score < s) {
                        best = Some((score, t, to));
                    }
                }
            }
            let Some((_, t, to)) = best else {
                return false;
            };
            self.apply(t, to);
        }
        true
    }

    /// Spends time on error until the error is within its limit, never past
    /// a size limit. `false` when no change that fits removes error.
    fn improve(&mut self, limits: &Limits) -> bool {
        while self.error > limits.max_error {
            let mut best: Option<(f64, usize, usize)> = None;
            for t in 0..self.tensors.len() {
                let costs = &self.tensors[t].costs;
                let from = self.at[t];
                for to in 0..costs.len() {
                    let removed = costs[from].error - costs[to].error;
                    if removed <= 0.0 || !self.fits(t, to, limits) {
                        continue;
                    }
                    let score = removed / self.price(t, to).max(FREE);
                    if best.is_none_or(|(s, _, _)| score > s) {
                        best = Some((score, t, to));
                    }
                }
            }
            let Some((_, t, to)) = best else {
                return false;
            };
            self.apply(t, to);
        }
        true
    }

    fn choice(&self, within: bool) -> Choice {
        Choice {
            types: self
                .at
                .iter()
                .zip(self.tensors)
                .map(|(&c, t)| t.costs[c].ggml_type)
                .collect(),
            family: self.family,
            error: self.error,
            bytes: self.bytes,
            device_bytes: self.device_bytes,
            decode_us: self.decode,
            prefill_us: self.prefill,
            within,
        }
    }
}

/// The candidate index of `ggml_type` in each tensor's costs, or `None`
/// when a tensor has no entry for its type.
pub fn positions(tensors: &[Tensor], types: &[u32]) -> Option<Vec<usize>> {
    tensors
        .iter()
        .zip(types)
        .map(|(t, &ty)| t.costs.iter().position(|c| c.ggml_type == ty))
        .collect()
}

/// What `types` — one per tensor, each among its candidates — is predicted
/// to cost on `family`.
pub fn evaluate(tensors: &[Tensor], types: &[u32], family: usize) -> Option<Choice> {
    let at = positions(tensors, types)?;
    let mut choice = Choice {
        types: types.to_vec(),
        family,
        error: 0.0,
        bytes: 0,
        device_bytes: 0,
        decode_us: 0.0,
        prefill_us: 0.0,
        within: true,
    };
    for (t, &c) in tensors.iter().zip(&at) {
        choice.error += t.costs[c].error;
        choice.bytes += t.costs[c].bytes;
        if t.on_device {
            choice.device_bytes += t.costs[c].bytes;
        }
        choice.decode_us += t.time_us[c][0][family];
        choice.prefill_us += t.time_us[c][1][family];
    }
    Some(choice)
}

/// The reference file's cost on the family it runs faster on: the one its
/// card would choose.
pub fn reference(tensors: &[Tensor], types: &[u32]) -> Option<Choice> {
    let float = evaluate(tensors, types, 0)?;
    let integer = evaluate(tensors, types, 1)?;
    Some(if integer.decode_us < float.decode_us {
        integer
    } else {
        float
    })
}

/// The limits a reference file sets by default: no more error and no more
/// bytes than it has, and no more on the card than `device_bytes` (the
/// card's room for weights beside its KV pages; `None`, no card limit). No
/// slower to decode or to read a prompt than it is.
pub fn default_limits(reference: &Choice, device_bytes: Option<u64>) -> Limits {
    Limits {
        max_error: reference.error,
        max_bytes: reference.bytes,
        max_device_bytes: device_bytes.unwrap_or(u64::MAX),
        max_decode_us: reference.decode_us,
        max_prefill_us: reference.prefill_us,
    }
}

/// Chooses a type for every tensor within `limits`, fastest first, priced
/// against `reference` (decode µs, prefill µs, bytes). When no choice meets
/// every limit, the closest is returned with `within` false.
pub fn choose(tensors: &[Tensor], limits: &Limits, reference: (f64, f64, f64)) -> Choice {
    let reference = (
        reference.0.max(1e-9),
        reference.1.max(1e-9),
        reference.2.max(1.0),
    );
    let mut best: Option<Choice> = None;
    for family in 0..2 {
        let at: Vec<usize> = tensors
            .iter()
            .map(|t| {
                let mut fastest = 0;
                for c in 1..t.costs.len() {
                    let key = |c: usize| {
                        (
                            t.time_us[c][0][family] / reference.0
                                + t.time_us[c][1][family] / reference.1,
                            t.costs[c].error,
                        )
                    };
                    let (a, b) = (key(c), key(fastest));
                    if a.0 < b.0 || (a.0 == b.0 && a.1 < b.1) {
                        fastest = c;
                    }
                }
                fastest
            })
            .collect();
        let mut state = State {
            tensors,
            family,
            at: vec![0; tensors.len()],
            error: 0.0,
            bytes: 0,
            device_bytes: 0,
            decode: 0.0,
            prefill: 0.0,
            reference,
        };
        let start = evaluate(
            tensors,
            &at.iter()
                .zip(tensors)
                .map(|(&c, t)| t.costs[c].ggml_type)
                .collect::<Vec<_>>(),
            family,
        )
        .expect("the start is drawn from each tensor's own candidates");
        state.at = at;
        state.error = start.error;
        state.bytes = start.bytes;
        state.device_bytes = start.device_bytes;
        state.decode = start.decode_us;
        state.prefill = start.prefill_us;
        let within = state.shrink(limits) && state.improve(limits);
        let choice = state.choice(within);
        let better = match &best {
            None => true,
            Some(held) => {
                (
                    choice.within,
                    -(choice.decode_us / reference.0 + choice.prefill_us / reference.1),
                ) > (
                    held.within,
                    -(held.decode_us / reference.0 + held.prefill_us / reference.1),
                )
            }
        };
        if better {
            best = Some(choice);
        }
    }
    best.expect("two families were tried")
}

#[cfg(test)]
mod tests {
    use super::*;
    use orangu::quantize as q;

    const Q4: u32 = q::GGML_TYPE_Q4_K;
    const Q5: u32 = q::GGML_TYPE_Q5_K;
    const Q6: u32 = q::GGML_TYPE_Q6_K;
    const Q3: u32 = q::GGML_TYPE_Q3_K;

    /// A tensor with (type, bytes, error, decode µs) candidates, the same
    /// time on both families and a prompt token at a hundredth of decode.
    fn tensor(candidates: &[(u32, u64, f64, f64)]) -> Tensor {
        Tensor {
            costs: candidates
                .iter()
                .map(|&(ggml_type, bytes, error, _)| Cost {
                    ggml_type,
                    bytes,
                    error,
                })
                .collect(),
            time_us: candidates
                .iter()
                .map(|&(_, _, _, us)| [[us, us], [us / 100.0, us / 100.0]])
                .collect(),
            on_device: true,
        }
    }

    fn ladder(scale: f64, sensitivity: f64) -> Tensor {
        tensor(&[
            (Q3, (300.0 * scale) as u64, 4.0 * sensitivity, 12.0 * scale),
            (Q4, (400.0 * scale) as u64, 2.0 * sensitivity, 10.0 * scale),
            (Q5, (500.0 * scale) as u64, 1.0 * sensitivity, 11.0 * scale),
            (Q6, (600.0 * scale) as u64, 0.5 * sensitivity, 13.0 * scale),
        ])
    }

    fn reference_of(tensors: &[Tensor], types: &[u32]) -> (Choice, (f64, f64, f64)) {
        let r = reference(tensors, types).unwrap();
        let prices = (r.decode_us, r.prefill_us, r.bytes as f64);
        (r, prices)
    }

    #[test]
    fn a_loose_budget_keeps_every_tensor_at_its_fastest_type() {
        let tensors = vec![ladder(1.0, 1.0), ladder(2.0, 1.0)];
        let (r, prices) = reference_of(&tensors, &[Q6, Q6]);
        let limits = Limits {
            max_error: f64::MAX,
            max_bytes: u64::MAX,
            max_device_bytes: u64::MAX,
            max_decode_us: f64::MAX,
            max_prefill_us: f64::MAX,
        };
        let chosen = choose(&tensors, &limits, prices);
        assert_eq!(chosen.types, [Q4, Q4]);
        assert!(chosen.within && chosen.decode_us < r.decode_us);
    }

    #[test]
    fn error_is_bought_where_it_is_cheapest() {
        // The second tensor's error falls as far for an upgrade but its
        // time costs twice as much: the first is upgraded.
        let tensors = vec![ladder(1.0, 4.0), ladder(2.0, 4.0)];
        let limits = Limits {
            max_error: 14.0,
            max_bytes: u64::MAX,
            max_device_bytes: u64::MAX,
            max_decode_us: f64::MAX,
            max_prefill_us: f64::MAX,
        };
        let chosen = choose(&tensors, &limits, (30.0, 0.3, 1200.0));
        assert_eq!(chosen.types, [Q5, Q4]);
        assert!(chosen.within && chosen.error <= 14.0);
    }

    #[test]
    fn a_size_limit_is_never_crossed() {
        let tensors = vec![ladder(1.0, 4.0), ladder(1.0, 4.0)];
        let limits = Limits {
            max_error: 0.0,
            max_bytes: 1000,
            max_device_bytes: u64::MAX,
            max_decode_us: f64::MAX,
            max_prefill_us: f64::MAX,
        };
        let chosen = choose(&tensors, &limits, (20.0, 0.2, 800.0));
        assert!(chosen.bytes <= 1000);
        assert!(!chosen.within, "an error of 0 cannot be met");
    }

    #[test]
    fn an_oversized_start_is_shrunk_first() {
        let tensors = vec![ladder(1.0, 1.0), ladder(1.0, 1.0)];
        let limits = Limits {
            max_error: f64::MAX,
            max_bytes: 700,
            max_device_bytes: u64::MAX,
            max_decode_us: f64::MAX,
            max_prefill_us: f64::MAX,
        };
        let chosen = choose(&tensors, &limits, (20.0, 0.2, 800.0));
        assert_eq!(chosen.types, [Q3, Q4]);
        assert!(chosen.within && chosen.bytes == 700);
    }

    #[test]
    fn only_the_cards_bytes_count_against_its_room() {
        let mut host = ladder(1.0, 4.0);
        host.on_device = false;
        let tensors = vec![ladder(1.0, 4.0), host];
        let limits = Limits {
            max_error: 10.0,
            max_bytes: u64::MAX,
            max_device_bytes: 400,
            max_decode_us: f64::MAX,
            max_prefill_us: f64::MAX,
        };
        let chosen = choose(&tensors, &limits, (20.0, 0.2, 800.0));
        assert_eq!(chosen.types, [Q4, Q6]);
        assert!(chosen.device_bytes <= 400 && chosen.within);
    }

    #[test]
    fn a_free_upgrade_comes_before_a_paid_one() {
        // A looked-up embedding: no time at any type.
        let embedding = tensor(&[(Q4, 400, 2.0, 0.0), (Q6, 600, 0.5, 0.0)]);
        let tensors = vec![ladder(1.0, 1.0), embedding];
        let limits = Limits {
            max_error: 2.5,
            max_bytes: u64::MAX,
            max_device_bytes: u64::MAX,
            max_decode_us: f64::MAX,
            max_prefill_us: f64::MAX,
        };
        let chosen = choose(&tensors, &limits, (10.0, 0.1, 800.0));
        assert_eq!(chosen.types, [Q4, Q6]);
    }

    #[test]
    fn the_faster_kernel_family_is_kept() {
        let mut t = ladder(1.0, 1.0);
        // Q3 is the fastest type on the integer dot only.
        t.time_us[0] = [[12.0, 5.0], [0.12, 0.05]];
        let tensors = vec![t];
        let limits = Limits {
            max_error: f64::MAX,
            max_bytes: u64::MAX,
            max_device_bytes: u64::MAX,
            max_decode_us: f64::MAX,
            max_prefill_us: f64::MAX,
        };
        let chosen = choose(&tensors, &limits, (10.0, 0.1, 400.0));
        assert_eq!((chosen.types[0], chosen.family), (Q3, 1));
    }

    #[test]
    fn the_same_inputs_give_the_same_choice() {
        let tensors: Vec<Tensor> = (0..40)
            .map(|i| ladder(1.0 + (i % 7) as f64 * 0.3, 1.0 + (i % 5) as f64))
            .collect();
        let types = vec![Q5; tensors.len()];
        let (r, prices) = reference_of(&tensors, &types);
        let limits = default_limits(&r, None);
        let a = choose(&tensors, &limits, prices);
        let b = choose(&tensors, &limits, prices);
        assert_eq!(a, b);
        assert!(a.within && a.error <= r.error && a.bytes <= r.bytes);
        assert!(
            a.decode_us <= r.decode_us,
            "{} > {}",
            a.decode_us,
            r.decode_us
        );
    }
}

#[cfg(test)]
mod time_limit_tests {
    use super::*;
    use orangu::quantize as q;

    #[test]
    fn a_file_is_never_made_slower_to_meet_its_error() {
        // Only the slower type removes enough error.
        let t = Tensor {
            costs: vec![
                Cost {
                    ggml_type: q::GGML_TYPE_Q4_K,
                    bytes: 400,
                    error: 2.0,
                },
                Cost {
                    ggml_type: q::GGML_TYPE_Q6_K,
                    bytes: 600,
                    error: 0.5,
                },
            ],
            time_us: vec![[[10.0; 2], [0.1; 2]], [[13.0; 2], [0.13; 2]]],
            on_device: true,
        };
        let limits = Limits {
            max_error: 1.0,
            max_bytes: u64::MAX,
            max_device_bytes: u64::MAX,
            max_decode_us: 12.0,
            max_prefill_us: f64::MAX,
        };
        let chosen = choose(&[t], &limits, (10.0, 0.1, 400.0));
        assert_eq!(chosen.types, [q::GGML_TYPE_Q4_K]);
        assert!(!chosen.within);
    }
}
