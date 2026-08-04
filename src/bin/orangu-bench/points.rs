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

//! The sweep points a list-valued flag accepts: plain numbers, and ranges.
//!
//! `--pp 128,512,1024,2048` is four numbers typed out. The sweeps this project
//! actually runs are doublings, and typing them out is both tedious and the
//! kind of thing that ends up wrong in one arm of an A/B. `128-2048*2` says the
//! same thing and cannot be mistyped asymmetrically.
//!
//! Three forms, matching what the wider ecosystem's benchmarks accept so a
//! sweep can be copied between them:
//!
//! - `first-last` — every value from `first` to `last`, stepping by one
//! - `first-last+step` — stepping by `step`
//! - `first-last*mult` — multiplying by `mult` (the doubling sweep)
//!
//! Shared by the command line and the web console on purpose: the console
//! validates what it is about to run before it runs it, and a second copy of
//! this grammar would eventually accept something the CLI refuses.

/// The most points one range may expand to.
///
/// `128-4096` is a legal range and means 3969 separate measurements — hours of
/// benchmarking from a typo that looks exactly like a doubling sweep missing
/// its `*2`. The cap is far above any sweep this project runs (the longest is
/// eight points) and far below the size at which a mistake stops being
/// recoverable.
const MAX_POINTS: usize = 256;

/// Expand one comma-separated list into its points.
pub fn expand_list<S: AsRef<str>>(items: &[S]) -> anyhow::Result<Vec<u32>> {
    let mut out = Vec::new();
    for item in items {
        for value in expand(item.as_ref())? {
            out.push(value);
        }
    }
    Ok(out)
}

/// Expand one item: a number, or one of the three range forms.
pub fn expand(spec: &str) -> anyhow::Result<Vec<u32>> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Ok(Vec::new());
    }
    let number = |s: &str| -> anyhow::Result<u32> {
        s.trim().parse::<u32>().map_err(|_| {
            anyhow::anyhow!(
                "{s:?} is not a number — a point is a number, or a range like \
                 128-2048*2 (multiply), 0-2048+512 (step) or 1-8"
            )
        })
    };

    // A leading `-` cannot start a range (there are no negative token counts),
    // so the separator is looked for after the first character.
    let Some(dash) = spec[1..].find('-').map(|i| i + 1) else {
        return Ok(vec![number(spec)?]);
    };
    let first = number(&spec[..dash])?;
    let rest = &spec[dash + 1..];

    // `+step` and `*mult` bind to the end of the range, not to `first`.
    let (last, step) = match rest.find(['+', '*']) {
        Some(i) => {
            let operator = rest.as_bytes()[i];
            let amount = number(&rest[i + 1..])?;
            if amount == 0 || (operator == b'*' && amount == 1) {
                anyhow::bail!(
                    "{spec:?} would never reach its end — a step of 0 or a multiplier of 1 \
                     repeats the same point forever"
                );
            }
            (
                number(&rest[..i])?,
                if operator == b'*' {
                    Step::Multiply(amount)
                } else {
                    Step::Add(amount)
                },
            )
        }
        None => (number(rest)?, Step::Add(1)),
    };

    if last < first {
        anyhow::bail!("{spec:?} ends before it starts ({first} to {last})");
    }
    // A multiplying range from zero never moves; caught here rather than by
    // the cap, so the message says why.
    if first == 0 && matches!(step, Step::Multiply(_)) {
        anyhow::bail!("{spec:?} multiplies from 0, which never advances — start at 1 or higher");
    }

    let mut out = Vec::new();
    let mut value = first;
    loop {
        out.push(value);
        if out.len() > MAX_POINTS {
            anyhow::bail!(
                "{spec:?} expands to more than {MAX_POINTS} points — did a doubling sweep \
                 lose its `*2`?"
            );
        }
        value = match step {
            Step::Add(n) => value.saturating_add(n),
            Step::Multiply(n) => value.saturating_mul(n),
        };
        if value > last {
            break;
        }
    }
    Ok(out)
}

#[derive(Clone, Copy)]
enum Step {
    Add(u32),
    Multiply(u32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_plain_number_is_still_a_plain_number() {
        assert_eq!(expand("512").unwrap(), vec![512]);
        assert_eq!(expand(" 512 ").unwrap(), vec![512]);
        assert_eq!(
            expand_list(&["0", "512", "1024"]).unwrap(),
            vec![0, 512, 1024]
        );
    }

    #[test]
    fn the_three_range_forms_expand_as_the_ecosystem_spells_them() {
        // The doubling sweep this repository actually runs.
        assert_eq!(
            expand("128-2048*2").unwrap(),
            vec![128, 256, 512, 1024, 2048]
        );
        assert_eq!(
            expand("0-2048+512").unwrap(),
            vec![0, 512, 1024, 1536, 2048]
        );
        assert_eq!(expand("1-8").unwrap(), vec![1, 2, 3, 4, 5, 6, 7, 8]);
        // The end is a bound, not a member: a range that does not land on it
        // stops below it rather than overshooting.
        assert_eq!(
            expand("128-3000*2").unwrap(),
            vec![128, 256, 512, 1024, 2048]
        );
        // One point is a legitimate range.
        assert_eq!(expand("512-512").unwrap(), vec![512]);
    }

    #[test]
    fn lists_and_ranges_mix() {
        assert_eq!(
            expand_list(&["0", "128-512*2", "3072"]).unwrap(),
            vec![0, 128, 256, 512, 3072]
        );
    }

    /// Every one of these is a plausible typo, and every one of them would
    /// otherwise cost the user a run — or, worse, produce a sweep that looks
    /// deliberate.
    #[test]
    fn a_range_that_cannot_work_is_refused_with_its_reason() {
        for bad in [
            "2048-128",   // backwards
            "128-2048*1", // never advances
            "128-2048+0", // never advances
            "0-2048*2",   // multiplying from zero
            "128-",       // no end
            "128-2048*",  // no multiplier
            "12a-2048",   // not a number
            "128-2048*abc",
        ] {
            let err = expand(bad).expect_err(&format!("accepted {bad:?}"));
            assert!(!err.to_string().is_empty(), "{bad:?}");
        }
    }

    /// `128-4096` is legal, means 3969 measurements, and looks exactly like a
    /// doubling sweep that lost its `*2`. Hours of benchmarking should not
    /// start on that.
    #[test]
    fn a_runaway_expansion_is_refused_rather_than_run() {
        let err = expand("128-4096").expect_err("should be capped");
        assert!(err.to_string().contains("more than"), "{err}");
        // Right up to the cap is still allowed.
        assert_eq!(expand("1-256").unwrap().len(), 256);
    }
}
