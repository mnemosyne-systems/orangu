//! What to drop when a conversation outgrows the context window.
//!
//! The first tokens stay, because that is where a conversation's instructions
//! are, and whole steps of the oldest tokens after them go. A step is a fixed
//! share of the window, so where a cut falls depends on the window alone. A
//! conversation's next turn, longer than this one, therefore cuts in the same
//! place until it has grown by another whole step, and its trimmed prompt
//! begins with what the cache already holds. That reuse is exact: the rows are
//! those of the same tokens, computed in the same order.

/// `[orangu-server].context_shift`, when it is on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContextShift {
    /// How many of the first tokens stay, unless a request says otherwise.
    pub keep: usize,
    /// The share of the tokens after those that one step drops.
    pub drop_fraction: f32,
}

/// The fewest tokens a window can hold and still have something to shift.
const SMALLEST_WINDOW: usize = 4;

impl ContextShift {
    /// The tokens that stay at the start, and how many one step drops. `None`
    /// for a window too small to do this in.
    pub fn plan(&self, window: usize, n_keep: Option<usize>) -> Option<(usize, usize)> {
        if window < SMALLEST_WINDOW {
            return None;
        }
        // The first token stays whatever is asked, as it can be a BOS, and at
        // most half the window, or there would be little left to answer in.
        let keep = n_keep.unwrap_or(self.keep).clamp(1, window / 2);
        let step = ((window - keep) as f32 * self.drop_fraction) as usize;
        Some((keep, step.clamp(1, window - keep)))
    }

    /// `tokens` with as many steps dropped as leaves room for one more token.
    /// `None` when they fit already, or the window is too small.
    pub fn trim(
        &self,
        tokens: &[u32],
        window: usize,
        n_keep: Option<usize>,
    ) -> Option<(Vec<u32>, usize)> {
        let over = (tokens.len() + 1).checked_sub(window).filter(|&o| o > 0)?;
        let (keep, step) = self.plan(window, n_keep)?;
        let dropped = over.div_ceil(step) * step;
        // A step is at most what follows `keep`, and the tokens over the window
        // are fewer than the tokens after it, so this stays inside the prompt
        // whenever anything after `keep` is left to keep.
        let dropped = dropped.min(tokens.len().saturating_sub(keep + 1));
        let mut trimmed = Vec::with_capacity(tokens.len() - dropped);
        trimmed.extend_from_slice(&tokens[..keep]);
        trimmed.extend_from_slice(&tokens[keep + dropped..]);
        Some((trimmed, dropped))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HALF: ContextShift = ContextShift {
        keep: 2,
        drop_fraction: 0.5,
    };

    fn counting(n: u32) -> Vec<u32> {
        (0..n).collect()
    }

    #[test]
    fn what_fits_stays_as_it_is() {
        assert_eq!(HALF.trim(&counting(9), 10, None), None);
        assert!(HALF.trim(&counting(10), 10, None).is_some());
    }

    #[test]
    fn the_first_tokens_and_the_newest_stay() {
        let (trimmed, dropped) = HALF.trim(&counting(14), 10, None).unwrap();
        assert_eq!(dropped, 8);
        assert_eq!(trimmed, vec![0, 1, 10, 11, 12, 13]);
    }

    #[test]
    fn a_request_can_keep_more_and_at_least_one_token_always_stays() {
        let (trimmed, _) = HALF.trim(&counting(12), 10, Some(4)).unwrap();
        assert_eq!(&trimmed[..4], &[0, 1, 2, 3]);
        let (trimmed, _) = HALF.trim(&counting(12), 10, Some(0)).unwrap();
        assert_eq!(trimmed[0], 0);
        assert_eq!(trimmed.last(), Some(&11));
        // More than half the window is not left to the first tokens.
        assert_eq!(HALF.plan(10, Some(9)), Some((5, 2)));
    }

    #[test]
    fn nothing_is_left_out_of_range_at_any_length() {
        for window in SMALLEST_WINDOW..40 {
            for len in 0..200u32 {
                for fraction in [0.01, 0.25, 0.5, 0.99, 1.0] {
                    let shift = ContextShift {
                        keep: 3,
                        drop_fraction: fraction,
                    };
                    let tokens = counting(len);
                    let Some((trimmed, dropped)) = shift.trim(&tokens, window, None) else {
                        assert!((len as usize) < window);
                        continue;
                    };
                    assert!(trimmed.len() < window, "{window} {len} {fraction}");
                    assert_eq!(trimmed.len() + dropped, tokens.len());
                    assert_eq!(trimmed.last(), tokens.last());
                }
            }
        }
    }

    /// The property cache reuse rests on: as a conversation grows, each trimmed
    /// prompt begins with the whole of the one before it, until it has grown by
    /// a further step.
    #[test]
    fn a_longer_conversation_cuts_in_the_same_place() {
        let window = 100;
        let conversation = counting(1000);
        let (_, step) = HALF.plan(window, None).unwrap();
        let mut cuts = 0;
        let mut reused = 0;
        let mut previous: Option<(Vec<u32>, usize)> = None;
        for len in window..1000 {
            let current = HALF.trim(&conversation[..len], window, None).unwrap();
            assert_eq!(current.1 % step, 0);
            if let Some((before, dropped)) = &previous {
                if *dropped == current.1 {
                    assert!(current.0.starts_with(before), "at {len}");
                    reused += 1;
                } else {
                    cuts += 1;
                }
            }
            previous = Some(current);
        }
        assert!(reused > cuts * (step - 2), "{reused} reused, {cuts} cuts");
    }

    /// Cutting one step at a time as a reply grows lands where cutting the
    /// whole conversation at once does, so the next turn's prompt matches,
    /// unless what that turn adds carries it over a further step, when it has
    /// to cut again and what follows the cut is read afresh.
    #[test]
    fn a_shift_during_a_reply_matches_the_next_turns_prompt() {
        let window = 64;
        let conversation = counting(400);
        let (keep, step) = HALF.plan(window, None).unwrap();
        let mut live = conversation[..window - 1].to_vec();
        let mut live_dropped = 0;
        let (mut matched, mut crossed) = (0, 0);
        for sent in window..conversation.len() {
            live.push(conversation[sent - 1]);
            if live.len() >= window {
                live.drain(keep..keep + step);
                live_dropped += step;
            }
            let (next, dropped) = HALF.trim(&conversation[..=sent], window, None).unwrap();
            if dropped == live_dropped {
                assert!(next.starts_with(&live), "at {sent}");
                matched += 1;
            } else {
                assert_eq!(dropped, live_dropped + step, "at {sent}");
                crossed += 1;
            }
        }
        assert!(
            matched > crossed * (step - 2),
            "{matched} matched, {crossed} crossed"
        );
    }
}
