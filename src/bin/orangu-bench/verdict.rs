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

//! A two-arm sweep's answer, judged against its own noise.
//!
//! A sweep of two builds (`ARM=old,new,new,old`) prints each point's mean per
//! run; whether a 3% difference is the change or the machine is left to the
//! reader. This decides it, per point, from two measured spreads:
//!
//! - **within a run**: the standard error of the difference of the two arms'
//!   means, from each run's own repetitions;
//! - **between runs of one arm**: how far the same build's two runs landed
//!   apart. Drift over a sweep (heat, clocks, the page cache) moves a whole
//!   run, and the repetitions inside it cannot see that.
//!
//! The threshold is the larger of twice the first and the second. A
//! difference beyond it is the change; anything inside it is not shown to be.
//!
//! The second spread needs two runs of each arm, so an arm with fewer gets no
//! verdict ([`MIN_RUNS`]): one run cannot tell the change from the drift.

/// Runs of each arm a verdict needs.
pub const MIN_RUNS: usize = 2;

/// One run's summary of one point, as the history records it.
#[derive(Clone, Copy, Debug)]
pub struct Run {
    pub mean: f64,
    /// Sample standard deviation of the run's repetitions, when there was
    /// more than one.
    pub sd: Option<f64>,
    pub reps: u32,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Outcome {
    Better,
    Worse,
    WithinNoise,
}

#[derive(Debug, Clone, Copy)]
pub struct Verdict {
    /// Candidate over control, as a fraction (`0.05` is 5% higher).
    pub change: f64,
    /// The threshold the difference was held to, as a fraction of the
    /// control's mean.
    pub noise: f64,
    pub outcome: Outcome,
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

/// Judges `candidate` against `control` for one point. `None` when either arm
/// has fewer than [`MIN_RUNS`] runs or the control's mean is not positive.
pub fn judge(control: &[Run], candidate: &[Run], lower_is_better: bool) -> Option<Verdict> {
    if control.len() < MIN_RUNS || candidate.len() < MIN_RUNS {
        return None;
    }
    let c = mean(&control.iter().map(|r| r.mean).collect::<Vec<_>>());
    let t = mean(&candidate.iter().map(|r| r.mean).collect::<Vec<_>>());
    if c <= 0.0 {
        return None;
    }
    // Standard error of an arm's mean over all its repetitions.
    let se = |runs: &[Run]| -> f64 {
        let n: f64 = runs.iter().map(|r| f64::from(r.reps.max(1))).sum();
        let var = runs
            .iter()
            .map(|r| r.sd.unwrap_or(0.0).powi(2) * f64::from(r.reps.max(1)))
            .sum::<f64>()
            / n;
        (var / n).sqrt()
    };
    let se_diff = (se(control).powi(2) + se(candidate).powi(2)).sqrt();
    let spread = |runs: &[Run]| -> f64 {
        let lo = runs.iter().map(|r| r.mean).fold(f64::INFINITY, f64::min);
        let hi = runs
            .iter()
            .map(|r| r.mean)
            .fold(f64::NEG_INFINITY, f64::max);
        hi - lo
    };
    let noise = (2.0 * se_diff).max(spread(control)).max(spread(candidate));
    let delta = t - c;
    let outcome = if delta.abs() <= noise {
        Outcome::WithinNoise
    } else if (delta > 0.0) != lower_is_better {
        Outcome::Better
    } else {
        Outcome::Worse
    };
    Some(Verdict {
        change: delta / c,
        noise: noise / c,
        outcome,
    })
}

/// Modes whose recorded value is a cost, so a smaller one is the better.
pub fn lower_is_better(mode: &str) -> bool {
    matches!(mode, "cpu" | "itl" | "kl" | "ttft" | "ttft-cold")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(mean: f64, sd: f64) -> Run {
        Run {
            mean,
            sd: Some(sd),
            reps: 3,
        }
    }

    #[test]
    fn a_change_larger_than_both_spreads_is_reported() {
        let v = judge(
            &[run(100.0, 1.0), run(101.0, 1.0)],
            &[run(110.0, 1.0), run(111.0, 1.0)],
            false,
        )
        .unwrap();
        assert_eq!(v.outcome, Outcome::Better);
        assert!((v.change - 0.0995).abs() < 1e-3);
    }

    #[test]
    fn drift_between_runs_of_one_arm_widens_the_threshold() {
        // Tight repetitions, but the control's own runs are 8 apart: a 5
        // difference is inside what the machine did on its own.
        let v = judge(
            &[run(100.0, 0.1), run(108.0, 0.1)],
            &[run(109.0, 0.1), run(109.0, 0.1)],
            false,
        )
        .unwrap();
        assert_eq!(v.outcome, Outcome::WithinNoise);
    }

    #[test]
    fn a_cost_that_rises_is_worse() {
        let v = judge(
            &[run(4.0, 0.1), run(4.0, 0.1)],
            &[run(7.0, 0.1), run(7.0, 0.1)],
            true,
        )
        .unwrap();
        assert_eq!(v.outcome, Outcome::Worse);
        assert!(lower_is_better("itl") && lower_is_better("kl") && !lower_is_better("tg"));
        assert!(lower_is_better("ttft") && lower_is_better("ttft-cold"));
    }

    #[test]
    fn an_arm_with_fewer_than_two_runs_has_no_verdict() {
        assert!(judge(&[], &[run(1.0, 0.0)], false).is_none());
        // A large, tight difference still needs each arm's drift measured.
        assert!(judge(&[run(100.0, 0.1)], &[run(150.0, 0.1)], false).is_none());
        assert!(judge(&[run(100.0, 0.1); 2], &[run(150.0, 0.1)], false).is_none());
        assert!(judge(&[run(100.0, 0.1); 2], &[run(150.0, 0.1); 2], false).is_some());
    }
}
