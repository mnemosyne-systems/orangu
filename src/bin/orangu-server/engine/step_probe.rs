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

//! **Choosing between ways of running a decode step by running them.**
//!
//! Some choices a decode step makes have no answer that holds across
//! machines — which processor takes the vocabulary projection, how the
//! step's recording is split into submissions — and the only honest way
//! to make them is to time the step each way on the machine at hand and
//! keep the fastest. This module is that experiment, with the shape the
//! two choices it serves have needed:
//!
//! * **Blocks, not alternation.** Each arm is timed over a block of
//!   consecutive steps rather than the arms alternating step by step. A
//!   step that leaves the card idle parks its clock, and the other arm's
//!   step after it would be timed at the parked clock — the arm loses to
//!   its neighbour's presence.
//! * **Settling first.** The first steps of a process, and the first after
//!   a long settled stretch, run at a rate the settled step does not (the
//!   card's clock climbing, the path's first-use costs); a block timed on
//!   them can be twice the settled step. Nothing is timed until the step
//!   has settled.
//! * **Rounds and a vote.** One block per arm is decided by whatever else
//!   those steps were doing; the arms are timed over several rounds, each
//!   round's medians are compared on their own, and the answer is the arm
//!   that won most rounds. The default arm wins a round unless another
//!   beats it by a margin — a near-tie is noise, and the default is the
//!   arm that costs nothing elsewhere.
//! * **Windows.** An answer holds for a window of steps and is then asked
//!   again — the first window short, so a call made on a process's first
//!   request is soon checked against its settled rate.

/// The experiment: `arms` ways of running a step, arm 0 the default.
pub(crate) struct ArmProbe {
    arms: usize,
    /// Steps to discard per block: the first steps on an arm pay for
    /// whatever its path touches for the first time, and for the card's
    /// clock settling to the arm's own rhythm.
    warmup: usize,
    /// Steps to time per block, past the discarded ones.
    timed: usize,
    /// Device steps before the first block of a window is timed.
    settle: usize,
    /// Rounds voted per answer.
    rounds: usize,
    /// Steps under the first answer before the arms are timed again.
    reprobe_first: usize,
    /// Steps under any later answer before the arms are timed again.
    reprobe: usize,
    /// Another arm is taken only when it is this much faster than the
    /// default.
    margin_percent: u64,

    /// The arm the current step is running, and the one [`ArmProbe::note`]
    /// will credit. Latched rather than recomputed because a single step
    /// may ask twice, and the two must agree.
    arm: usize,
    /// Steps taken on the default arm before the first block is timed.
    settled: usize,
    /// Step times per arm of the round in progress.
    samples: Vec<Vec<u64>>,
    /// Each completed round's winner.
    winners: Vec<usize>,
    /// The answer, once enough rounds have been voted.
    decided: Option<usize>,
    /// Steps taken under the current answer.
    settled_steps: usize,
    /// How many answers have been given; the first holds a short window.
    answers: usize,
}

/// What one completed round measured, for tracing.
pub(crate) struct Round {
    /// The round's number, from one.
    pub number: usize,
    /// Each arm's median step in nanoseconds.
    pub medians: Vec<u64>,
    /// The arm that won it.
    pub winner: usize,
}

/// What a completed vote decided, for tracing.
pub(crate) struct Verdict {
    /// Rounds won per arm.
    pub wins: Vec<usize>,
    /// The arm taken.
    pub arm: usize,
}

/// What a step's note may have completed.
pub(crate) enum Noted {
    Nothing,
    Round(Round),
    Decided(Round, Verdict),
}

impl ArmProbe {
    /// A probe over `arms` arms with this project's settled shape: three
    /// warm-up and four timed steps per block, 32 settling steps, three
    /// rounds, a first window of 1024 steps and later ones of 4096, and a
    /// `margin_percent` another arm must beat the default by. The blocks
    /// are short and the windows long because a losing arm's steps are
    /// paid for in full: a host-tail step on a small model is 1.6× the
    /// device's, and thirty of them in a request's first two hundred steps
    /// were a visible dent in its rate.
    pub(crate) const fn new(arms: usize, margin_percent: u64) -> Self {
        Self {
            arms,
            warmup: 3,
            timed: 4,
            settle: 32,
            rounds: 3,
            reprobe_first: 1024,
            reprobe: 4096,
            margin_percent,
            arm: 0,
            settled: 0,
            samples: Vec::new(),
            winners: Vec::new(),
            decided: None,
            settled_steps: 0,
            answers: 0,
        }
    }

    #[cfg(test)]
    const fn warmup(&self) -> usize {
        self.warmup
    }

    #[cfg(test)]
    const fn timed(&self) -> usize {
        self.timed
    }

    #[cfg(test)]
    const fn settle(&self) -> usize {
        self.settle
    }

    #[cfg(test)]
    const fn rounds(&self) -> usize {
        self.rounds
    }

    #[cfg(test)]
    const fn reprobe_first(&self) -> usize {
        self.reprobe_first
    }

    #[cfg(test)]
    const fn reprobe(&self) -> usize {
        self.reprobe
    }

    /// The arm the next step runs.
    pub(crate) fn arm(&self) -> usize {
        self.decided.unwrap_or(self.arm)
    }

    /// Whether an answer stands — the probe is not timing arms.
    pub(crate) fn is_decided(&self) -> bool {
        self.decided.is_some()
    }

    /// Drops the round in progress, keeping the rounds already voted: for
    /// a stretch of steps that another experiment is shaping, which must
    /// not be credited to an arm of this one.
    pub(crate) fn pause(&mut self) {
        if self.decided.is_none() {
            self.samples.clear();
            self.arm = 0;
        }
    }

    /// Records a step that took `elapsed` on the arm [`ArmProbe::arm`]
    /// handed out for it.
    pub(crate) fn note(&mut self, elapsed: std::time::Duration) -> Noted {
        if self.decided.is_some() {
            self.settled_steps += 1;
            let window = if self.answers <= 1 {
                self.reprobe_first
            } else {
                self.reprobe
            };
            if self.settled_steps >= window {
                self.decided = None;
                self.settled_steps = 0;
                self.settled = 0;
                self.samples.clear();
                self.winners.clear();
                self.arm = 0;
            }
            return Noted::Nothing;
        }
        if self.settled < self.settle {
            self.settled += 1;
            return Noted::Nothing;
        }
        if self.samples.len() < self.arms {
            self.samples.resize_with(self.arms, Vec::new);
        }
        let block = self.warmup + self.timed;
        self.samples[self.arm].push(u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX));
        // Arm by arm, the default's block first; the arm changes only when
        // a block is complete.
        if self.samples[self.arm].len() >= block && self.arm + 1 < self.arms {
            self.arm += 1;
        }
        if self.samples.iter().any(|s| s.len() < block) {
            return Noted::Nothing;
        }
        let medians: Vec<u64> = self
            .samples
            .iter()
            .map(|s| {
                let mut steps: Vec<u64> = s[self.warmup..].to_vec();
                steps.sort_unstable();
                steps[steps.len() / 2]
            })
            .collect();
        let winner = self.round_winner(&medians);
        self.samples.clear();
        self.arm = 0;
        self.winners.push(winner);
        let round = Round {
            number: self.winners.len(),
            medians,
            winner,
        };
        if self.winners.len() < self.rounds {
            return Noted::Round(round);
        }
        let mut wins = vec![0usize; self.arms];
        for &w in &self.winners {
            wins[w] += 1;
        }
        // The most rounds; the lower arm on a tie, so the default keeps a
        // split vote.
        let arm = (0..self.arms)
            .max_by(|&a, &b| wins[a].cmp(&wins[b]).then(b.cmp(&a)))
            .unwrap_or(0);
        self.decided = Some(arm);
        self.arm = arm;
        self.answers += 1;
        Noted::Decided(round, Verdict { wins, arm })
    }

    /// The fastest arm of a round, if it beats the default by the margin;
    /// the default otherwise.
    fn round_winner(&self, medians: &[u64]) -> usize {
        let base = medians[0];
        let best = (1..self.arms).min_by_key(|&a| medians[a]).unwrap_or(0);
        if best != 0 && medians[best] * 100 < base * (100 - self.margin_percent) {
            best
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ms(n: u64) -> std::time::Duration {
        std::time::Duration::from_millis(n)
    }

    /// Feeds `probe` one round: a block per arm at the given step times.
    fn round(probe: &mut ArmProbe, arm_ms: &[u64]) {
        let block = probe.warmup() + probe.timed();
        for (arm, &t) in arm_ms.iter().enumerate() {
            for _ in 0..block {
                assert_eq!(probe.arm(), arm, "arms run in order, in blocks");
                probe.note(ms(t));
            }
        }
    }

    fn settle(probe: &mut ArmProbe, t: u64) {
        for _ in 0..probe.settle() {
            assert_eq!(probe.arm(), 0);
            probe.note(ms(t));
        }
    }

    /// Steps before the first block are not timed, nothing is decided until
    /// every round is through, and within a round each arm is timed over a
    /// block of consecutive steps — the arms never alternate.
    #[test]
    fn settles_then_times_each_arm_in_blocks() {
        let mut probe = ArmProbe::new(3, 10);
        settle(&mut probe, 100);
        assert!(probe.samples.is_empty(), "settling steps are not timed");
        for r in 0..probe.rounds() {
            assert!(!probe.is_decided(), "undecided before round {r}");
            round(&mut probe, &[100, 50, 80]);
        }
        assert_eq!(probe.arm(), 1);
    }

    /// A near-tie keeps the default; another arm has to win by the margin.
    #[test]
    fn needs_a_margin_to_leave_the_default() {
        let decide = |arms: &[u64]| {
            let mut probe = ArmProbe::new(arms.len(), 10);
            settle(&mut probe, arms[0]);
            for _ in 0..probe.rounds() {
                round(&mut probe, arms);
            }
            probe.arm()
        };
        assert_eq!(decide(&[100, 95]), 0, "a 5% edge is noise");
        assert_eq!(decide(&[100, 80]), 1, "a clear win is taken");
        assert_eq!(decide(&[100, 120]), 0);
        assert_eq!(decide(&[100, 95, 85]), 2, "the fastest of the rest");
    }

    /// One round another arm wins while the card is cold does not decide:
    /// the answer is the majority of the rounds, and a split vote keeps the
    /// default.
    #[test]
    fn votes_across_rounds() {
        let mut probe = ArmProbe::new(2, 10);
        settle(&mut probe, 100);
        round(&mut probe, &[200, 100]);
        round(&mut probe, &[60, 100]);
        round(&mut probe, &[60, 100]);
        assert_eq!(probe.decided, Some(0), "two default rounds outvote one");

        let mut probe = ArmProbe::new(3, 10);
        settle(&mut probe, 100);
        round(&mut probe, &[100, 50, 100]);
        round(&mut probe, &[100, 100, 50]);
        round(&mut probe, &[100, 100, 100]);
        assert_eq!(probe.decided, Some(0), "a split vote keeps the default");
    }

    /// The answer is revisited: the first after a short window, later ones
    /// after the long one, and a different outcome replaces the old one.
    #[test]
    fn revisits_its_answer() {
        let mut probe = ArmProbe::new(2, 10);
        settle(&mut probe, 100);
        for _ in 0..probe.rounds() {
            round(&mut probe, &[100, 50]);
        }
        assert_eq!(probe.arm(), 1, "the other arm won the first window");
        for _ in 0..probe.reprobe_first() {
            probe.note(ms(50));
        }
        assert!(!probe.is_decided(), "the first window is short");
        assert_eq!(probe.arm(), 0, "and the next starts on the default");
        settle(&mut probe, 40);
        for _ in 0..probe.rounds() {
            round(&mut probe, &[40, 50]);
        }
        assert_eq!(probe.arm(), 0, "the default won the second window");
        for _ in 0..probe.reprobe_first() {
            probe.note(ms(40));
        }
        assert!(probe.is_decided(), "the second window is the long one");
        for _ in probe.reprobe_first()..probe.reprobe() {
            probe.note(ms(40));
        }
        assert!(!probe.is_decided());
    }

    /// A pause drops the round in progress and keeps the rounds voted.
    #[test]
    fn a_pause_drops_the_round_in_progress() {
        let mut probe = ArmProbe::new(2, 10);
        settle(&mut probe, 100);
        round(&mut probe, &[100, 50]);
        for _ in 0..3 {
            probe.note(ms(100));
        }
        probe.pause();
        assert_eq!(probe.winners.len(), 1);
        assert!(probe.samples.is_empty());
        assert_eq!(probe.arm(), 0);
        round(&mut probe, &[100, 50]);
        round(&mut probe, &[100, 50]);
        assert_eq!(probe.arm(), 1);
    }
}
