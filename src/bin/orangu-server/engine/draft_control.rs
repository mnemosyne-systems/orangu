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

//! Whether a speculative step is worth taking, and how long its draft is.
//!
//! A verify pass yields the tokens of the draft it accepts plus one; a plain
//! decode step yields one. Which is faster depends on the machine — a verify
//! over five positions costs one decode step where the weights do not fit
//! the device and several where they do — and on the text: a reply that
//! echoes its context accepts drafts deep, prose rarely. So both are
//! measured as decoding goes.
//!
//! **Acceptance, by depth.** A request keeps, for each depth of a draft, how
//! often the drafted token there was accepted given every one before it was.
//! A draft of `L` tokens is then expected to yield `1 + p₁ + p₁p₂ + … +
//! p₁…p_L` tokens — so where acceptance falls off after two tokens, a
//! sixteen-token draft costs sixteen rows and yields what a two-token one
//! does. Lengths are chosen from [`LENGTHS`] (up to the drafter's own), so a
//! device keeps resources for a few widths and not every one.
//!
//! **Prices, by what was priced.** A verify's time is kept per drafter, per
//! width and per how many requests share the step (`engine::decode_batch`); a
//! plain step's per that sharing. Each price is the median of its last three
//! measurements, so one slow pass — a page fault, a clock dip — does not move
//! it; the first measurement at a price is dropped, since it pays the switch
//! to a width the device had not built resources for. Prices are kept per
//! depth of context too ([`depth_bin`]), since a step's cost moves with the
//! rows it attends to, and a price older than [`STALE`] is measured again
//! rather than trusted, so one early measurement does not decide a width for
//! good. A batched plain step is timed without the time it
//! waited for its partners, which is theirs.
//!
//! A step verifies with the length whose `(expected tokens + rows − 1) /
//! verify time` (in the batch) or `expected tokens / verify time` (outside
//! it) is highest, while that beats `rows / step time`. While verifying
//! loses, one chance in [`PROBE_EVERY`] verifies anyway, so a stretch of
//! echoed text is noticed when it starts, and every [`RETIME_PLAIN_EVERY`]
//! verifies one plain step is taken, so the plain price follows the context.
//!
//! A reply that echoes its context is verified whatever the gate says (the
//! drafter knows: its match runs far back), and the control picks the
//! width of those verifies instead: the drafter's own, or its wider echo
//! length, whichever has yielded more tokens a millisecond on echoes, the
//! other once in [`PROBE_EVERY`] echoes so it stays measured.
//!
//! `ORANGU_SPEC_ADAPTIVE=0` verifies every draft at the drafter's length, an
//! echo at its wider one.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

/// While verifying loses, verify anyway once in this many chances.
pub const PROBE_EVERY: u32 = 16;
/// The draft lengths a step chooses from (up to the drafter's own).
pub const LENGTHS: [usize; 5] = [1, 2, 4, 8, 16];
/// The deepest draft whose acceptance is kept.
const DEPTHS: usize = 16;
/// Confidence bins a drafter reports a proposal in — see [`bin_of_match`]
/// and [`bin_of_probability`].
pub const BINS: usize = 4;
/// Weight of the newest pass in a depth's acceptance, once it has seen
/// enough passes; before that each pass weighs as much as every one before
/// it and the prior together ([`learn`]).
const ALPHA: f64 = 0.1;
/// Acceptance assumed at a depth before any draft reached it: optimistic,
/// so a longer draft is tried rather than written off unmeasured.
const PRIOR_ACCEPT: f64 = 0.6;
/// Verifies after which, with no plain step timed yet, one plain step is
/// taken to time it.
const VERIFIES_BEFORE_A_TIMED_STEP: u32 = 4;
/// Verifies between plain steps taken to time the plain step again.
pub const RETIME_PLAIN_EVERY: u32 = 64;

/// What a price is the price of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Priced {
    /// A plain decode step.
    Plain,
    /// A verify of this many rows by this drafter.
    Verify { drafter: u8, rows: usize },
}

/// A price, with the requests sharing the step, whether the verify rode in
/// it and how deep the context was.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Key {
    what: Priced,
    sharing: usize,
    in_batch: bool,
    /// The context's depth bin ([`depth_bin`]): a step's attention, and so
    /// its price, grows with the rows behind it.
    depth: u8,
}

/// The bin of a step at position `pos`: one per doubling of the context
/// past 512 tokens.
fn depth_bin(pos: usize) -> u8 {
    (pos / 512 + 1).ilog2().min(15) as u8
}

/// A price older than this is measured again.
const STALE: Duration = Duration::from_secs(120);

/// The last three measurements of one price.
#[derive(Clone, Copy, Debug, Default)]
struct Price {
    recent: [f64; 3],
    kept: usize,
    seen: u32,
    at: Option<Instant>,
}

impl Price {
    fn note(&mut self, ms: f64) {
        self.seen += 1;
        self.at = Some(Instant::now());
        // The first at a price pays the switch to it.
        if self.seen == 1 {
            return;
        }
        self.recent[self.kept % 3] = ms;
        self.kept += 1;
    }

    /// The median of the last three (or fewer), unless it is stale.
    fn value(&self) -> Option<f64> {
        self.value_at(Instant::now())
    }

    fn value_at(&self, now: Instant) -> Option<f64> {
        if self.kept == 0 || self.at.is_none_or(|at| now.duration_since(at) > STALE) {
            return None;
        }
        let mut v: Vec<f64> = self.recent[..self.kept.min(3)].to_vec();
        v.sort_by(f64::total_cmp);
        Some(v[v.len() / 2])
    }
}

/// Every price, for the process: a property of the machine and the model.
static PRICES: Mutex<Option<HashMap<Key, Price>>> = Mutex::new(None);

fn price(key: Key) -> Option<f64> {
    PRICES
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .as_ref()?
        .get(&key)?
        .value()
}

fn note_price(key: Key, took: Duration) {
    PRICES
        .lock()
        .unwrap_or_else(|p| p.into_inner())
        .get_or_insert_with(HashMap::new)
        .entry(key)
        .or_default()
        .note(took.as_secs_f64() * 1000.0);
}

/// One more pass at a depth, `hit` 1 where it was accepted: the prior counts
/// as one pass and each pass as one more, so the first replaces the prior
/// halfway (one miss does not write a depth off) and a few agreeing ones
/// settle it, until the newest weighs [`ALPHA`].
fn learn(accept: &mut f64, seen: &mut u32, hit: f64) {
    *seen += 1;
    let weight = (1.0 / f64::from(*seen + 1)).max(ALPHA);
    *accept += weight * (hit - *accept);
}

/// What a step was planned against: the cost of each verify width.
#[derive(Clone, Debug)]
struct Planned {
    /// `(draft length, verify ms)`, priced or estimated.
    widths: Vec<(usize, f64)>,
}

/// A prompt-lookup proposal's confidence bin, by how many tokens of context
/// its match covers: a match a few tokens long is a phrase that recurs, one
/// many tokens long is a passage being copied.
pub fn bin_of_match(context: usize) -> u8 {
    match context {
        0..=3 => 0,
        4..=7 => 1,
        8..=15 => 2,
        _ => 3,
    }
}

/// A model's proposal's confidence bin, by the probability its drafter gave it.
pub fn bin_of_probability(p: f32) -> u8 {
    if p < 0.3 {
        0
    } else if p < 0.6 {
        1
    } else if p < 0.9 {
        2
    } else {
        3
    }
}

/// One request's view: its acceptance by depth, and the next draft's length.
pub struct DraftControl {
    adaptive: bool,
    /// Which drafter's verify prices this request reads and writes.
    drafter: u8,
    /// Per depth: the share of drafts that reached it and were accepted there.
    accept: [f64; DEPTHS],
    /// Per depth: how many drafts reached it.
    reached: [u32; DEPTHS],
    /// Per depth and confidence bin: the share of proposals there that were
    /// accepted, and how many were verified.
    accept_bin: [[f64; BINS]; DEPTHS],
    reached_bin: [[u32; BINS]; DEPTHS],
    /// The prices a step was planned against, for [`Self::cut`].
    planned: Option<Planned>,
    skipped: u32,
    verifies: u32,
    since_plain: u32,
    /// Tokens per verify of an echo, at the drafter's own width and at its
    /// echo width.
    echo_yield: [Option<f64>; 2],
    echoes: u32,
    /// The depth bin of the step being decided ([`Self::at`]).
    depth: u8,
}

impl DraftControl {
    /// A request's control for the drafter `drafter` names (one byte per
    /// kind of drafter: prompt lookup, a draft model, a head).
    pub fn new(drafter: u8) -> Self {
        Self::with(
            cfg!(not(test)) && crate::engine::env::flag_on_unless_disabled("ORANGU_SPEC_ADAPTIVE"),
            drafter,
        )
    }

    fn with(adaptive: bool, drafter: u8) -> Self {
        Self {
            adaptive,
            drafter,
            accept: [PRIOR_ACCEPT; DEPTHS],
            reached: [0; DEPTHS],
            accept_bin: [[PRIOR_ACCEPT; BINS]; DEPTHS],
            reached_bin: [[0; BINS]; DEPTHS],
            planned: None,
            skipped: 0,
            verifies: 0,
            since_plain: 0,
            echo_yield: [None, None],
            echoes: 0,
            depth: 0,
        }
    }

    /// The next step runs at position `pos`: its prices are read and kept
    /// with others at the same depth of context.
    pub fn at(&mut self, pos: usize) {
        self.depth = depth_bin(pos);
    }

    /// Tokens a draft of `len` is expected to yield, the model's own included.
    fn expected(&self, len: usize) -> f64 {
        let mut reach = 1.0;
        let mut total = 1.0;
        let mut ceiling = 1.0f64;
        for d in 0..len.min(DEPTHS) {
            // A depth no draft reached yet is assumed no likelier than the
            // depth before it: a miss there lowers the prior behind it too.
            let p = if self.reached[d] > 0 {
                self.accept[d]
            } else {
                self.accept[d].min(ceiling)
            };
            ceiling = p;
            reach *= p;
            total += reach;
        }
        total
    }

    /// The next step's draft length — `None` to take a plain step — with
    /// `sharing` requests in the decode step, the verify riding in it when
    /// `in_batch`, and the drafter able to draft `max_len`.
    pub fn plan(&mut self, sharing: usize, in_batch: bool, max_len: usize) -> Option<usize> {
        if !self.adaptive || max_len == 0 {
            return (max_len > 0).then_some(max_len);
        }
        self.decide(sharing, in_batch, max_len)
    }

    fn decide(&mut self, sharing: usize, in_batch: bool, max_len: usize) -> Option<usize> {
        let plain_key = Key {
            what: Priced::Plain,
            sharing,
            in_batch: false,
            depth: self.depth,
        };
        let Some(step) = price(plain_key) else {
            // A missing plain price is measured by taking a plain step, after
            // a few verifies have had their turn.
            return (self.verifies < VERIFIES_BEFORE_A_TIMED_STEP).then_some(max_len);
        };
        if self.since_plain >= RETIME_PLAIN_EVERY {
            self.since_plain = 0;
            return None;
        }
        let rows = sharing.max(1) as f64;
        let others = if in_batch { rows - 1.0 } else { 0.0 };
        let lengths: Vec<usize> = LENGTHS
            .iter()
            .copied()
            .filter(|&l| l < max_len)
            .chain(std::iter::once(max_len))
            .collect();
        // A length not priced yet is priced as the nearest one that is, or as
        // a plain step a row: what it costs is found by taking it.
        let priced: Vec<(usize, Option<f64>)> = lengths
            .iter()
            .map(|&l| {
                let key = Key {
                    what: Priced::Verify {
                        drafter: self.drafter,
                        rows: l + 1,
                    },
                    sharing,
                    in_batch,
                    depth: self.depth,
                };
                (l, price(key))
            })
            .collect();
        let estimate = |l: usize| -> f64 {
            priced
                .iter()
                .filter_map(|&(m, p)| p.map(|p| (m.abs_diff(l), p)))
                .min_by_key(|&(d, _)| d)
                .map_or(step * (1.0 + 0.25 * l as f64), |(_, p)| p)
        };
        self.planned = Some(Planned {
            widths: priced
                .iter()
                .map(|&(l, p)| (l, p.unwrap_or_else(|| estimate(l))))
                .collect(),
        });
        let (best, rate) = priced
            .iter()
            .map(|&(l, p)| {
                (
                    l,
                    (self.expected(l) + others) / p.unwrap_or_else(|| estimate(l)),
                )
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))?;
        if rate >= rows / step {
            self.skipped = 0;
            return Some(best);
        }
        self.skipped += 1;
        if self.skipped >= PROBE_EVERY {
            self.skipped = 0;
            return Some(best);
        }
        None
    }

    /// How many of a draft's proposals to verify, given each one's confidence
    /// bin: the proposals up to the width whose expected tokens a millisecond
    /// is best, each proposal's chance of being reached and accepted taken
    /// from what proposals at its depth and bin have done. A proposal its
    /// drafter is unsure of, deep in a draft, costs a row and rarely yields
    /// a token. The whole draft when the step was not planned against prices
    /// (an echo, a head, no adaptation).
    pub fn cut(&self, bins: &[u8]) -> usize {
        let Some(planned) = self.planned.as_ref().filter(|_| self.adaptive) else {
            return bins.len();
        };
        let mut reach = 1.0;
        let mut expected = vec![1.0];
        for (d, &bin) in bins.iter().enumerate().take(DEPTHS) {
            reach *= self.accept_bin[d][usize::from(bin).min(BINS - 1)];
            expected.push(expected[d] + reach);
        }
        planned
            .widths
            .iter()
            .filter(|&&(l, _)| l <= bins.len())
            .map(|&(l, ms)| (l, expected[l.min(expected.len() - 1)] / ms))
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map_or(bins.len(), |(l, _)| l)
    }

    /// Whether the next echo is verified at the drafter's echo width, given
    /// the two lengths and the step's sharing.
    pub fn echo_wide(&mut self, own: usize, wide: usize, sharing: usize, in_batch: bool) -> bool {
        if !self.adaptive {
            return true;
        }
        let ms = |len: usize| {
            price(Key {
                what: Priced::Verify {
                    drafter: self.drafter,
                    rows: len + 1,
                },
                sharing,
                in_batch,
                depth: self.depth,
            })
        };
        self.choose_wide(ms(own), ms(wide))
    }

    fn choose_wide(&mut self, own_ms: Option<f64>, wide_ms: Option<f64>) -> bool {
        self.echoes += 1;
        // An unmeasured width is measured by taking it, the wide one first.
        let (Some(wide_ms), Some(wide_yield)) = (wide_ms, self.echo_yield[1]) else {
            return true;
        };
        let (Some(own_ms), Some(own_yield)) = (own_ms, self.echo_yield[0]) else {
            return false;
        };
        let wide = wide_yield / wide_ms >= own_yield / own_ms;
        if self.echoes.is_multiple_of(PROBE_EVERY) {
            !wide
        } else {
            wide
        }
    }

    /// A verify of a `drafted`-token draft over `rows` rows (the padding to
    /// its width included) took `took` and yielded `tokens` (accepted drafts
    /// plus the model's own next token), with `sharing`
    /// requests in the step and the verify in it when `in_batch`; `echo` is
    /// the width of a verify taken for an echo — `Some(true)` the echo width
    /// — and `None` for any other.
    #[allow(clippy::too_many_arguments)]
    pub fn verified(
        &mut self,
        tokens: usize,
        drafted: usize,
        bins: &[u8],
        rows: usize,
        took: Duration,
        echo: Option<bool>,
        sharing: usize,
        in_batch: bool,
    ) {
        self.verifies += 1;
        self.since_plain += 1;
        let accepted = tokens.saturating_sub(1).min(drafted);
        // The gate's acceptance is of drafts it chose the length of; an echo
        // verified wide says nothing about that.
        if echo != Some(true) {
            for d in 0..(accepted + 1).min(drafted).min(DEPTHS) {
                let hit = if d < accepted { 1.0 } else { 0.0 };
                learn(&mut self.accept[d], &mut self.reached[d], hit);
                if let Some(&bin) = bins.get(d) {
                    let b = usize::from(bin).min(BINS - 1);
                    learn(&mut self.accept_bin[d][b], &mut self.reached_bin[d][b], hit);
                }
            }
        }
        if let Some(w) = echo {
            let slot = &mut self.echo_yield[usize::from(w)];
            *slot = Some(slot.map_or(tokens as f64, |old| old + ALPHA * (tokens as f64 - old)));
        }
        note_price(
            Key {
                what: Priced::Verify {
                    drafter: self.drafter,
                    rows,
                },
                sharing,
                in_batch,
                depth: self.depth,
            },
            took,
        );
    }

    /// A plain decode step, with `sharing` requests in it, computed for
    /// `took` — its wait for its partners excluded.
    pub fn stepped(&mut self, took: Duration, sharing: usize) {
        self.since_plain = 0;
        note_price(
            Key {
                what: Priced::Plain,
                sharing,
                in_batch: false,
                depth: self.depth,
            },
            took,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Prices for these tests under a drafter id of their own, so tests
    /// running together do not read each other's.
    fn priced(drafter: u8, sharing: usize, in_batch: bool, step: f64, verify: &[(usize, f64)]) {
        let ms = |v: f64| Duration::from_secs_f64(v / 1000.0);
        for _ in 0..3 {
            note_price(
                Key {
                    what: Priced::Plain,
                    sharing,
                    in_batch: false,
                    depth: 0,
                },
                ms(step),
            );
            for &(rows, v) in verify {
                note_price(
                    Key {
                        what: Priced::Verify { drafter, rows },
                        sharing,
                        in_batch,
                        depth: 0,
                    },
                    ms(v),
                );
            }
        }
    }

    /// A price is the median of its last three, its first dropped, and none
    /// once stale.
    #[test]
    fn a_price_is_the_median_of_three_after_the_first() {
        let mut p = Price::default();
        p.note(500.0);
        assert_eq!(p.value(), None, "the first pays the switch");
        for v in [10.0, 90.0, 12.0] {
            p.note(v);
        }
        assert_eq!(p.value(), Some(12.0));
        p.note(11.0);
        assert_eq!(p.value(), Some(12.0), "the last three: 90, 12, 11");
        let later = Instant::now() + STALE + Duration::from_secs(1);
        assert_eq!(p.value_at(later), None);
    }

    /// Deep acceptance takes a long draft, shallow acceptance a short one,
    /// at the same prices.
    #[test]
    fn the_draft_length_follows_acceptance_by_depth() {
        let verify: Vec<(usize, f64)> = LENGTHS
            .iter()
            .map(|&l| (l + 1, 20.0 + 2.0 * l as f64))
            .collect();
        priced(201, 1, false, 16.0, &verify);
        let mut deep = DraftControl::with(true, 201);
        deep.accept = [0.95; DEPTHS];
        assert_eq!(deep.decide(1, false, 16), Some(16));
        let mut shallow = DraftControl::with(true, 201);
        shallow.accept = [0.0; DEPTHS];
        shallow.accept[0] = 0.9;
        shallow.accept[1] = 0.9;
        let len = shallow.decide(1, false, 16);
        assert!(matches!(len, Some(2) | Some(4)), "{len:?}");
    }

    /// Prose a drafter keeps missing on stops being verified within a few
    /// drafts: each early miss weighs as much as the prior, and the depths
    /// behind the first are not assumed likelier than it.
    #[test]
    fn a_few_misses_stop_the_verifies() {
        let verify: Vec<(usize, f64)> = LENGTHS.iter().map(|&l| (l + 1, 12.0 + l as f64)).collect();
        priced(211, 1, false, 10.0, &verify);
        let mut c = DraftControl::with(true, 211);
        let mut verifies = 0;
        while let Some(len) = c.decide(1, false, 4) {
            verifies += 1;
            assert!(verifies <= 3, "still verifying after {verifies} misses");
            c.verified(
                1,
                len,
                &[],
                len + 1,
                Duration::from_millis(15),
                None,
                1,
                false,
            );
        }
    }

    /// Acceptance is learned per depth: a draft accepted to depth 2 of 4
    /// raises depths 1–2 and lowers depth 3, and leaves depth 4 alone.
    #[test]
    fn a_verify_teaches_each_depth_it_reached() {
        let mut c = DraftControl::with(true, 202);
        c.verified(3, 4, &[], 5, Duration::from_millis(1), None, 1, false);
        assert!(c.accept[0] > PRIOR_ACCEPT && c.accept[1] > PRIOR_ACCEPT);
        assert!(c.accept[2] < PRIOR_ACCEPT);
        assert_eq!(c.accept[3], PRIOR_ACCEPT);
        assert_eq!(c.reached[3], 0);
    }

    /// A short draft padded to a wide verify is priced at the width it ran,
    /// so the wide width's cost is known and its own short length's is not
    /// charged for the padding.
    #[test]
    fn a_padded_verify_is_priced_at_the_rows_it_ran() {
        let mut c = DraftControl::with(true, 209);
        for _ in 0..3 {
            c.verified(
                4,
                3,
                &[],
                17,
                Duration::from_millis(60),
                Some(true),
                1,
                false,
            );
        }
        let key = |rows| Key {
            what: Priced::Verify { drafter: 209, rows },
            sharing: 1,
            in_batch: false,
            depth: 0,
        };
        assert!(price(key(17)).is_some());
        assert!(price(key(4)).is_none());
    }

    /// A price measured near the start of a context is not read deep into
    /// one, where every row attends to thousands more.
    #[test]
    fn a_price_is_kept_per_depth_of_context() {
        let mut c = DraftControl::with(true, 210);
        c.at(100);
        for _ in 0..3 {
            c.stepped(Duration::from_millis(10), 1);
        }
        let plain = |depth| Key {
            what: Priced::Plain,
            sharing: 1,
            in_batch: false,
            depth,
        };
        assert!(price(plain(depth_bin(100))).is_some());
        c.at(4096);
        assert_ne!(depth_bin(100), c.depth);
        assert!(price(plain(c.depth)).is_none());
    }

    /// A verify that loses to the plain step is not taken, but is probed
    /// once in [`PROBE_EVERY`].
    #[test]
    fn a_losing_verify_is_probed_now_and_then() {
        let verify: Vec<(usize, f64)> = LENGTHS.iter().map(|&l| (l + 1, 100.0)).collect();
        priced(203, 1, false, 16.0, &verify);
        let mut c = DraftControl::with(true, 203);
        c.accept = [0.3; DEPTHS];
        let taken = (0..PROBE_EVERY * 3)
            .filter(|_| c.decide(1, false, 8).is_some())
            .count();
        assert_eq!(taken, 3);
    }

    /// A verify inside the batch counts the other rows' tokens; outside it
    /// has to beat every row of the step on its own.
    #[test]
    fn a_verify_in_the_batch_is_charged_only_what_it_adds() {
        let verify: Vec<(usize, f64)> = LENGTHS.iter().map(|&l| (l + 1, 36.0)).collect();
        priced(204, 4, false, 30.0, &verify);
        priced(204, 4, true, 30.0, &verify);
        let mut c = DraftControl::with(true, 204);
        c.accept = [0.0; DEPTHS];
        c.accept[0] = 1.0;
        assert_eq!(c.decide(4, false, 4), None);
        assert!(c.decide(4, true, 4).is_some());
    }

    /// With no plain price, a few verifies go first and then a plain step
    /// is taken to price it; and every [`RETIME_PLAIN_EVERY`] verifies the
    /// plain step is timed again.
    #[test]
    fn the_plain_step_is_priced_and_priced_again() {
        let mut c = DraftControl::with(true, 205);
        assert_eq!(c.decide(9, false, 4), Some(4));
        c.verifies = VERIFIES_BEFORE_A_TIMED_STEP;
        assert_eq!(c.decide(9, false, 4), None);
        let verify: Vec<(usize, f64)> = LENGTHS.iter().map(|&l| (l + 1, 2.0)).collect();
        priced(205, 9, false, 16.0, &verify);
        // A verify of about 2.3 tokens in 2 ms beats nine rows in 16.
        c.since_plain = RETIME_PLAIN_EVERY;
        assert_eq!(c.decide(9, false, 4), None);
        assert!(c.decide(9, false, 4).is_some());
    }

    /// The echo width is tried first, then whichever width yields more
    /// tokens a millisecond is taken, the other once in [`PROBE_EVERY`].
    #[test]
    fn an_echo_takes_the_width_with_more_tokens_a_millisecond() {
        let mut c = DraftControl::with(true, 206);
        assert!(c.choose_wide(Some(40.0), None));
        c.echo_yield[1] = Some(17.0);
        assert!(!c.choose_wide(Some(40.0), Some(100.0)));
        c.echo_yield[0] = Some(5.0);
        c.echoes = 0;
        let taken = (0..PROBE_EVERY)
            .filter(|_| c.choose_wide(Some(40.0), Some(100.0)))
            .count();
        assert_eq!(taken as u32, PROBE_EVERY - 1);
        c.echoes = 0;
        let taken = (0..PROBE_EVERY)
            .filter(|_| c.choose_wide(Some(40.0), Some(200.0)))
            .count();
        assert_eq!(taken, 1);
    }

    #[test]
    fn without_adaptation_every_draft_is_verified_at_its_length() {
        let mut c = DraftControl::with(false, 207);
        assert_eq!(c.plan(4, false, 6), Some(6));
        assert_eq!(c.plan(4, false, 0), None);
    }

    /// A draft is cut where its proposals stop paying for their rows: past a
    /// confident run, an unsure proposal is not verified.
    #[test]
    fn an_unsure_proposal_deep_in_a_draft_is_not_verified() {
        let mut c = DraftControl::with(true, 208);
        c.planned = Some(Planned {
            widths: vec![(1, 18.0), (2, 19.0), (4, 21.0), (8, 25.0)],
        });
        for d in 0..DEPTHS {
            c.accept_bin[d][3] = 0.95;
            c.accept_bin[d][0] = 0.05;
        }
        assert_eq!(c.cut(&[3, 3, 3, 3, 3, 3, 3, 3]), 8);
        assert_eq!(c.cut(&[3, 3, 0, 0, 0, 0, 0, 0]), 2);
        // Unplanned (an echo): everything.
        c.planned = None;
        assert_eq!(c.cut(&[0, 0, 0]), 3);
        assert_eq!(bin_of_match(2), 0);
        assert_eq!(bin_of_match(20), 3);
        assert_eq!(bin_of_probability(0.95), 3);
        assert_eq!(bin_of_probability(0.1), 0);
    }
}
