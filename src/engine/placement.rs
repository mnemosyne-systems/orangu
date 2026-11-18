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

//! Which device runs which layer, when a model is spread across more than
//! one of them.
//!
//! `engine::backend::device` picks the device *set*; this decides what each
//! member of that set holds. Pure arithmetic over capacities and layer
//! counts — no `wgpu`, no driver, nothing to mock — so the policy can be
//! held to its promises by ordinary tests.
//!
//! # Contiguous ranges, always
//!
//! Layers are handed out in unbroken runs: `0..k` to the first device,
//! `k..m` to the second, and so on. Never round-robin, never interleaved.
//! A forward pass walks layers in order, so the number of times the hidden
//! state has to cross the bus is exactly the number of *boundaries* between
//! runs — `devices - 1` with contiguous ranges, and `n_layer - 1` with
//! interleaving. On a 48-layer model split over two cards that is the
//! difference between one crossing per token and forty-seven.
//!
//! # Proportional to capacity, not equal
//!
//! Two cards are rarely the same size, and an equal split makes the smaller
//! one the ceiling for both. Shares come from each device's reported memory
//! (`llama.cpp` does the same, from *free* memory, in `llama_model::
//! load_tensors`), falling back to equal shares when any device in the set
//! declines to report a size — an equal split of unknowns is at least
//! predictable, whereas treating "unknown" as zero would silently starve a
//! perfectly good card.
//!
//! # When to split at all
//!
//! Splitting is **not** free and in this engine it is not even cheap: see
//! `engine::backend::multi` for what a spread model gives up. So
//! [`SplitMode::Auto`] only splits when the weights do not fit the first
//! device on their own — the case where the alternative is the driver
//! paging VRAM on every token — and `Off`, the default, never does.

use std::fmt;

/// How `[orangu-server].device_split` was answered.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum SplitMode {
    /// One device runs the whole model. The default.
    #[default]
    Off,
    /// Split only when the weights exceed the first device.
    Auto,
    /// Always split across every selected device.
    All,
    /// Explicit proportions, one per selected device — `llama.cpp`'s `-ts`.
    /// Values are relative, not absolute: `3,1` is three quarters and one
    /// quarter.
    Ratios(Vec<f64>),
    /// Fill the devices with as many layers as fit, in order, and run the
    /// rest on the **CPU** — `llama.cpp`'s partial offload (`-ngl`), decided
    /// from capacity rather than typed by hand.
    ///
    /// The one mode that is a *fill* rather than a share, and it has to be:
    /// the host's budget is system RAM, so giving it a proportional share
    /// would hand it most of the model. Here it gets only what the devices
    /// could not hold.
    Cpu,
}

impl SplitMode {
    /// Parses `off`, `auto`, `all`, or a comma-separated ratio list.
    pub fn parse(raw: &str) -> Result<Self, String> {
        let raw = raw.trim();
        match raw.to_ascii_lowercase().as_str() {
            "" | "off" | "no" | "none" => return Ok(Self::Off),
            "auto" => return Ok(Self::Auto),
            "all" | "yes" => return Ok(Self::All),
            "cpu" | "overflow" => return Ok(Self::Cpu),
            _ => {}
        }
        let ratios = raw
            .split(',')
            .map(|part| {
                part.trim()
                    .parse::<f64>()
                    .map_err(|_| format!("{part:?} is not a number"))
                    .and_then(|value| {
                        // A negative or non-finite share has no meaning and
                        // would silently produce an empty or reversed
                        // assignment rather than an error.
                        (value.is_finite() && value >= 0.0)
                            .then_some(value)
                            .ok_or_else(|| format!("{value} is not a share of a model"))
                    })
            })
            .collect::<Result<Vec<f64>, String>>()?;
        if ratios.iter().sum::<f64>() <= 0.0 {
            return Err("every share is zero, which places no layers anywhere".to_string());
        }
        Ok(Self::Ratios(ratios))
    }

    pub fn is_off(&self) -> bool {
        matches!(self, Self::Off)
    }
}

impl fmt::Display for SplitMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Off => write!(f, "off"),
            Self::Auto => write!(f, "auto"),
            Self::All => write!(f, "all"),
            Self::Cpu => write!(f, "cpu"),
            Self::Ratios(ratios) => write!(
                f,
                "{}",
                ratios
                    .iter()
                    .map(|r| r.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        }
    }
}

/// The decision: which device holds each layer, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SplitPlan {
    /// One entry per layer — a position in the selected device set, not an
    /// enumeration index.
    pub layer_device: Vec<usize>,
    /// How many layers each selected device ended up with, so the report
    /// can state the split without re-deriving it.
    pub per_device_layers: Vec<usize>,
}

impl SplitPlan {
    /// How many times the hidden state crosses between devices in one
    /// forward pass. The cost of the split, in one number.
    pub fn boundaries(&self) -> usize {
        self.per_device_layers
            .iter()
            .filter(|n| **n > 0)
            .count()
            .saturating_sub(1)
    }

    /// Whether this plan actually spreads the model. A one-device plan is
    /// built and then discarded rather than special-cased at every call
    /// site.
    pub fn is_split(&self) -> bool {
        self.per_device_layers.iter().filter(|n| **n > 0).count() > 1
    }

    /// Whether the last entry — the host, under [`SplitMode::Cpu`] — ended
    /// up holding any layers.
    pub fn runs_on_host(&self) -> bool {
        self.per_device_layers.last().copied().unwrap_or(0) > 0
    }

    /// Whether this plan puts anything anywhere other than device 0 — a
    /// *relocation*, of which a split is one kind.
    ///
    /// [`is_split`](Self::is_split) asks whether the layers ended up on more
    /// than one device, which is the wrong question for a fill: moving every
    /// layer off device 0 onto device 1 is not a split by that test and is
    /// emphatically a plan worth keeping. Discarding it hands the model back
    /// to device 0 — the device the fill just decided could not hold it.
    ///
    /// Measured: a `BF16` model whose embeddings and `lm_head` alone exceed
    /// the discrete card zeroed that card's budget, so the fill correctly put
    /// all 35 layers on the 21 GiB integrated GPU — and `plan` threw the
    /// answer away because only one device had layers.
    pub fn relocates(&self) -> bool {
        self.per_device_layers
            .iter()
            .skip(1)
            .any(|&layers| layers > 0)
    }

    /// `layers 0-23 -> device 0, 24-31 -> device 1`, for the banner.
    pub fn describe(&self, device_names: &[String]) -> String {
        let mut parts = Vec::new();
        let mut start = 0usize;
        for (device, &count) in self.per_device_layers.iter().enumerate() {
            if count == 0 {
                continue;
            }
            let end = start + count - 1;
            let name = device_names
                .get(device)
                .cloned()
                .unwrap_or_else(|| format!("device {device}"));
            parts.push(if count == 1 {
                format!("layer {start} -> {name}")
            } else {
                format!("layers {start}-{end} -> {name}")
            });
            start += count;
        }
        parts.join(", ")
    }
}

/// Turns `mode` and the selected devices' capacities into a plan, or `None`
/// when the model should stay on one device.
///
/// `weights_bytes` is the model's device-resident weight total (see
/// `engine::footprint`) and is only consulted by [`SplitMode::Auto`], which
/// is the one mode that decides for itself whether a split is warranted.
pub fn plan(
    mode: &SplitMode,
    per_layer_bytes: &[u64],
    capacities: &[Option<u64>],
    weights_bytes: u64,
) -> Option<SplitPlan> {
    let n_layer = per_layer_bytes.len();
    if n_layer == 0 || capacities.len() < 2 {
        return None;
    }
    let shares: Vec<f64> = match mode {
        SplitMode::Off => return None,
        // Not a share at all — see the variant's own doc.
        SplitMode::Cpu => {
            let plan = fill_in_order(per_layer_bytes, capacities);
            // `relocates`, not just `is_split`: a fill that put *every*
            // layer somewhere other than device 0 — the host, or a second
            // card — is still a plan worth returning, because the
            // embeddings and `lm_head` stay on device 0 either way
            // (`LoadedModel::device_for_tensor`). Discarding it as "not a
            // split" would hand the whole model back to device 0 — which is
            // the paging this mode exists to avoid, arrived at by asking
            // for the opposite. `relocates` covers the host case
            // `runs_on_host` used to and the second-card case it missed.
            return (plan.is_split() || plan.relocates()).then_some(plan);
        }
        SplitMode::Auto => {
            // The whole of `Auto`: only spread a model that does not fit
            // where it would otherwise go. Anything else trades a working
            // fast path for a working slow one.
            let head = capacities.first().copied().flatten()?;
            if weights_bytes <= head {
                return None;
            }
            capacity_shares(capacities)
        }
        SplitMode::All => capacity_shares(capacities),
        SplitMode::Ratios(ratios) => {
            let mut shares = ratios.clone();
            // A short list leaves the remaining devices out entirely, which
            // is a legitimate way to say "these two of my three cards"; a
            // long one is trimmed rather than rejected, since the device set
            // can shrink between runs (a card removed, a driver gone) and
            // failing to start over a stale trailing zero would be worse
            // than ignoring it.
            shares.resize(capacities.len(), 0.0);
            shares
        }
    };
    let plan = assign_layers(n_layer, &shares);
    plan.is_split().then_some(plan)
}

/// Hands layers out in order, moving to the next device when the current
/// one is full — the shape [`SplitMode::Cpu`] needs.
///
/// A device whose capacity is unknown is treated as unbounded rather than
/// as zero, for the same reason the proportional path falls back to equal
/// shares: "unknown" is not "full", and a device that declines to report a
/// size would otherwise be skipped entirely. The last entry is the host,
/// whose capacity is system RAM and which therefore takes whatever is left.
///
/// Contiguous by construction, since layers are walked in their own order.
fn fill_in_order(per_layer_bytes: &[u64], capacities: &[Option<u64>]) -> SplitPlan {
    let mut counts = vec![0usize; capacities.len()];
    let mut used = 0u64;
    let mut device = 0usize;
    for &bytes in per_layer_bytes {
        while device + 1 < capacities.len() {
            match capacities[device] {
                Some(budget) if used.saturating_add(bytes) > budget => {
                    device += 1;
                    used = 0;
                }
                _ => break,
            }
        }
        counts[device] += 1;
        used = used.saturating_add(bytes);
    }
    finish(counts, per_layer_bytes.len())
}

/// Equal shares when any device's capacity is unknown, proportional shares
/// otherwise. See the module doc for why unknown is not zero.
fn capacity_shares(capacities: &[Option<u64>]) -> Vec<f64> {
    if capacities.iter().any(Option::is_none) {
        return vec![1.0; capacities.len()];
    }
    capacities
        .iter()
        .map(|c| c.unwrap_or(0) as f64)
        .collect::<Vec<f64>>()
}

/// Hands `n_layer` layers out in contiguous runs sized by `shares`.
///
/// Every device with a non-zero share gets at least one layer as long as
/// there are layers to go round: a share that rounds to nothing would
/// otherwise produce a device that was selected, reported, and given
/// nothing to do, which reads as a bug rather than as rounding.
fn assign_layers(n_layer: usize, shares: &[f64]) -> SplitPlan {
    let total: f64 = shares.iter().sum();
    let mut counts = vec![0usize; shares.len()];
    if total <= 0.0 {
        counts[0] = n_layer;
        return finish(counts, n_layer);
    }
    // Largest-remainder apportionment: floor every exact share, then hand
    // the leftovers to the largest fractional parts. Plain rounding can
    // distribute more or fewer layers than exist.
    let exact: Vec<f64> = shares.iter().map(|s| s / total * n_layer as f64).collect();
    for (count, value) in counts.iter_mut().zip(&exact) {
        *count = value.floor() as usize;
    }
    let mut remaining = n_layer.saturating_sub(counts.iter().sum::<usize>());
    let mut order: Vec<usize> = (0..shares.len()).collect();
    order.sort_by(|&a, &b| {
        let fa = exact[a] - exact[a].floor();
        let fb = exact[b] - exact[b].floor();
        fb.partial_cmp(&fa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    for &device in order.iter().cycle().take(remaining) {
        counts[device] += 1;
    }
    remaining = 0;
    let _ = remaining;

    // A selected device with a real share and no layers is worse than a
    // slightly uneven split: borrow from the largest holder.
    for device in 0..counts.len() {
        if shares[device] > 0.0 && counts[device] == 0 {
            let (donor, _) = counts
                .iter()
                .enumerate()
                .max_by_key(|(_, n)| **n)
                .expect("counts is non-empty");
            if counts[donor] > 1 {
                counts[donor] -= 1;
                counts[device] += 1;
            }
        }
    }
    finish(counts, n_layer)
}

fn finish(counts: Vec<usize>, n_layer: usize) -> SplitPlan {
    let mut layer_device = Vec::with_capacity(n_layer);
    for (device, &count) in counts.iter().enumerate() {
        layer_device.extend(std::iter::repeat_n(device, count));
    }
    debug_assert_eq!(layer_device.len(), n_layer);
    SplitPlan {
        layer_device,
        per_device_layers: counts,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    /// `n` layers of no particular size — for the proportional modes, which
    /// look only at the count.
    fn layers(n: usize) -> Vec<u64> {
        vec![0; n]
    }

    /// The reported crash, reduced: a fill whose head device has no budget
    /// left puts every layer on the *second* device. That is one device, so
    /// `is_split` is false — and returning `None` for it handed the model
    /// back to the device the fill had just rejected.
    #[test]
    fn a_fill_that_moves_every_layer_off_device_zero_is_kept() {
        let per_layer = vec![256 * 1024 * 1024u64; 35];
        // Device 0's budget is zero: the embeddings and `lm_head` already
        // exceeded it, which is exactly what `apply_device_split` computes
        // for a large-vocabulary BF16 model on a small card.
        let caps = vec![Some(0), Some(18 * 1024 * 1024 * 1024), None];
        let plan = plan(&SplitMode::Cpu, &per_layer, &caps, 9 * 1024 * 1024 * 1024)
            .expect("a fill that relocates every layer is a plan worth keeping");
        assert!(!plan.is_split(), "all layers landed on one device");
        assert!(plan.relocates(), "but not on device 0");
        assert!(!plan.runs_on_host(), "and not on the host either");
        assert!(plan.layer_device.iter().all(|&d| d == 1));
    }

    /// `relocates` is about device 0, not about the host, so it still covers
    /// the case `runs_on_host` used to.
    #[test]
    fn a_fill_that_spills_to_the_host_still_relocates() {
        let per_layer = vec![4 * 1024 * 1024 * 1024u64; 4];
        let caps = vec![Some(0), None];
        let plan = plan(&SplitMode::Cpu, &per_layer, &caps, 16 * 1024 * 1024 * 1024)
            .expect("everything on the host is a plan");
        assert!(plan.runs_on_host());
        assert!(plan.relocates());
    }

    /// A fill that changes nothing is still nothing — device 0 holding every
    /// layer is the placement that already happens without a split.
    #[test]
    fn a_fill_that_keeps_everything_on_device_zero_is_not_a_plan() {
        let per_layer = vec![64 * 1024 * 1024u64; 4];
        let caps = vec![Some(64 * 1024 * 1024 * 1024), None];
        assert!(plan(&SplitMode::Cpu, &per_layer, &caps, 256 * 1024 * 1024).is_none());
    }

    #[test]
    fn parses_every_spelling_of_the_mode() {
        assert_eq!(SplitMode::parse("off"), Ok(SplitMode::Off));
        assert_eq!(SplitMode::parse(""), Ok(SplitMode::Off));
        assert_eq!(SplitMode::parse("AUTO"), Ok(SplitMode::Auto));
        assert_eq!(SplitMode::parse(" all "), Ok(SplitMode::All));
        assert_eq!(
            SplitMode::parse("3,1"),
            Ok(SplitMode::Ratios(vec![3.0, 1.0]))
        );
        assert!(SplitMode::parse("3,banana").is_err());
        assert!(SplitMode::parse("-1,2").is_err());
        assert!(SplitMode::parse("0,0").is_err());
    }

    /// The default must be exactly what orangu did before this module
    /// existed.
    #[test]
    fn off_never_splits() {
        assert_eq!(
            plan(
                &SplitMode::Off,
                &layers(32),
                &[Some(4 * GIB), Some(24 * GIB)],
                100 * GIB
            ),
            None
        );
    }

    /// `auto` is a capacity decision, not a "use everything" decision: a
    /// model that fits the first card stays on it, because a split costs
    /// real speed in this engine.
    #[test]
    fn auto_leaves_a_model_that_fits_on_one_device() {
        assert_eq!(
            plan(
                &SplitMode::Auto,
                &layers(32),
                &[Some(24 * GIB), Some(24 * GIB)],
                8 * GIB
            ),
            None
        );
    }

    #[test]
    fn auto_splits_a_model_that_does_not_fit_the_first_device() {
        let split = plan(
            &SplitMode::Auto,
            &layers(32),
            &[Some(8 * GIB), Some(8 * GIB)],
            12 * GIB,
        )
        .expect("a split");
        assert_eq!(split.per_device_layers, vec![16, 16]);
        assert_eq!(split.boundaries(), 1);
    }

    /// Shares follow capacity: a 24 GiB card beside an 8 GiB one takes
    /// three quarters of the model, not half.
    #[test]
    fn shares_follow_capacity() {
        let split = plan(
            &SplitMode::All,
            &layers(32),
            &[Some(24 * GIB), Some(8 * GIB)],
            0,
        )
        .expect("a split");
        assert_eq!(split.per_device_layers, vec![24, 8]);
    }

    /// Every layer is placed exactly once, and the runs are contiguous —
    /// which is what keeps the number of bus crossings at one per boundary.
    #[test]
    fn layers_are_contiguous_and_every_one_is_placed() {
        let split = plan(
            &SplitMode::All,
            &layers(10),
            &[Some(GIB), Some(GIB), Some(2 * GIB)],
            0,
        )
        .expect("a split");
        assert_eq!(split.layer_device.len(), 10);
        // Shares 1:1:2 over 10 layers are 2.5, 2.5, 5 exactly; the odd
        // layer goes to the first of the two tied devices, which is what
        // largest-remainder apportionment does and what keeps the total at
        // 10 rather than 9 or 11.
        assert_eq!(split.layer_device, vec![0, 0, 0, 1, 1, 2, 2, 2, 2, 2]);
        // Contiguous means the device index never decreases.
        assert!(split.layer_device.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(split.boundaries(), 2);
    }

    /// Layer counts must sum to the model's own layer count even when the
    /// exact shares do not divide evenly — the reason this apportions by
    /// largest remainder rather than rounding each share.
    #[test]
    fn an_uneven_split_still_places_every_layer() {
        for n_layer in 1..=64 {
            for shares in [
                vec![Some(3 * GIB), Some(GIB)],
                vec![Some(7 * GIB), Some(5 * GIB), Some(GIB)],
                vec![Some(GIB), Some(GIB), Some(GIB)],
            ] {
                let Some(split) = plan(&SplitMode::All, &layers(n_layer), &shares, 0) else {
                    // One layer cannot be spread over two devices; that is
                    // reported as "no split", not as a bad plan.
                    assert!(n_layer < shares.len(), "n_layer {n_layer}");
                    continue;
                };
                assert_eq!(
                    split.per_device_layers.iter().sum::<usize>(),
                    n_layer,
                    "{n_layer} layers over {shares:?}"
                );
                assert_eq!(split.layer_device.len(), n_layer);
            }
        }
    }

    /// A device that was selected and reported must not end up holding
    /// nothing — that reads as a bug in the split rather than as rounding.
    #[test]
    fn a_tiny_share_still_gets_a_layer() {
        let split = plan(
            &SplitMode::All,
            &layers(32),
            &[Some(1000 * GIB), Some(GIB / 64)],
            0,
        )
        .expect("a split");
        assert!(split.per_device_layers[1] >= 1, "{split:?}");
        assert_eq!(split.per_device_layers.iter().sum::<usize>(), 32);
    }

    /// Unknown capacity is not zero: a device that declines to report a
    /// size still gets its share, and the whole set falls back to equal
    /// rather than starving it.
    #[test]
    fn an_unknown_capacity_falls_back_to_equal_shares() {
        let split =
            plan(&SplitMode::All, &layers(32), &[Some(24 * GIB), None], 0).expect("a split");
        assert_eq!(split.per_device_layers, vec![16, 16]);
    }

    #[test]
    fn explicit_ratios_override_capacity() {
        let split = plan(
            &SplitMode::Ratios(vec![1.0, 3.0]),
            &layers(32),
            &[Some(24 * GIB), Some(8 * GIB)],
            0,
        )
        .expect("a split");
        assert_eq!(split.per_device_layers, vec![8, 24]);
    }

    /// A zero share means "not this one" — a legitimate way to exclude a
    /// selected device without changing `device`.
    #[test]
    fn a_zero_ratio_excludes_a_device() {
        let split = plan(
            &SplitMode::Ratios(vec![1.0, 0.0, 1.0]),
            &layers(32),
            &[Some(GIB), Some(GIB), Some(GIB)],
            0,
        )
        .expect("a split");
        assert_eq!(split.per_device_layers, vec![16, 0, 16]);
        assert_eq!(split.boundaries(), 1);
        assert!(!split.layer_device.contains(&1));
    }

    /// A ratio list shorter than the device set places nothing on the rest,
    /// and a longer one is trimmed rather than rejected.
    #[test]
    fn a_ratio_list_of_the_wrong_length_is_taken_at_face_value() {
        let short = plan(
            &SplitMode::Ratios(vec![1.0, 1.0]),
            &layers(32),
            &[Some(GIB), Some(GIB), Some(GIB)],
            0,
        )
        .expect("a split");
        assert_eq!(short.per_device_layers, vec![16, 16, 0]);

        let long = plan(
            &SplitMode::Ratios(vec![1.0, 1.0, 5.0]),
            &layers(32),
            &[Some(GIB), Some(GIB)],
            0,
        )
        .expect("a split");
        assert_eq!(long.per_device_layers, vec![16, 16]);
    }

    /// One device is never a split, whatever the mode says.
    #[test]
    fn a_single_device_is_never_a_split() {
        assert_eq!(
            plan(&SplitMode::All, &layers(32), &[Some(GIB)], 100 * GIB),
            None
        );
        assert_eq!(
            plan(&SplitMode::Auto, &layers(32), &[Some(GIB)], 100 * GIB),
            None
        );
    }

    /// `cpu` is a *fill*, not a share: the device takes as many layers as
    /// fit and the host takes the rest. A share would hand the host most of
    /// the model, since its budget is system RAM.
    #[test]
    fn cpu_overflow_fills_the_device_then_spills_to_the_host() {
        // 8 layers of 1 GiB each, a 3 GiB card, and the host behind it.
        let split =
            plan(&SplitMode::Cpu, &[GIB; 8], &[Some(3 * GIB), None], 8 * GIB).expect("a split");
        assert_eq!(split.per_device_layers, vec![3, 5]);
        assert!(split.runs_on_host());
        assert_eq!(split.boundaries(), 1);
        assert_eq!(split.layer_device, vec![0, 0, 0, 1, 1, 1, 1, 1]);
    }

    /// A model that fits the device leaves the host with nothing — and a
    /// plan that places every layer on one device is not a split at all,
    /// so the ordinary single-device path keeps its fused kernels.
    #[test]
    fn cpu_overflow_does_not_split_a_model_that_fits() {
        assert_eq!(
            plan(&SplitMode::Cpu, &[GIB; 2], &[Some(8 * GIB), None], 2 * GIB),
            None
        );
    }

    /// Layers are not all the same size, which is exactly why the fill
    /// works per layer rather than from an average.
    #[test]
    fn cpu_overflow_measures_each_layer_rather_than_an_average() {
        // 4 GiB of budget: 3 + 1 fits two layers, the 3 GiB third does not.
        let split = plan(
            &SplitMode::Cpu,
            &[3 * GIB, GIB, 3 * GIB, GIB],
            &[Some(4 * GIB), None],
            8 * GIB,
        )
        .expect("a split");
        assert_eq!(split.per_device_layers, vec![2, 2]);
    }

    /// With two devices and the host, the second device fills before
    /// anything reaches the CPU.
    #[test]
    fn cpu_overflow_uses_every_device_before_the_host() {
        let split = plan(
            &SplitMode::Cpu,
            &[GIB; 8],
            &[Some(2 * GIB), Some(3 * GIB), None],
            8 * GIB,
        )
        .expect("a split");
        assert_eq!(split.per_device_layers, vec![2, 3, 3]);
    }

    /// A device that declines to report a size is unbounded, not full —
    /// the same "unknown is not zero" rule the proportional path follows.
    /// Here that means nothing spills past it.
    #[test]
    fn cpu_overflow_treats_an_unknown_capacity_as_unbounded() {
        assert_eq!(
            plan(&SplitMode::Cpu, &[GIB; 8], &[None, None], 8 * GIB),
            None,
            "an unsized device takes everything, so there is no split"
        );
    }

    /// A single layer larger than the card must not wedge the fill: it
    /// moves on rather than looping.
    #[test]
    fn cpu_overflow_passes_a_layer_too_large_for_the_device_to_the_host() {
        let split = plan(
            &SplitMode::Cpu,
            &[8 * GIB, GIB, GIB],
            &[Some(2 * GIB), None],
            10 * GIB,
        )
        .expect("a split");
        // The first layer does not fit, so the device is skipped for it and
        // everything from there on is the host's — contiguity is preserved.
        // Still returned as a plan rather than discarded: the embeddings and
        // `lm_head` stay on the device, and handing the model back to the
        // GPU wholesale is exactly the paging this mode avoids.
        assert_eq!(split.per_device_layers, vec![0, 3]);
        assert!(split.layer_device.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn the_description_names_each_run() {
        let split =
            plan(&SplitMode::All, &layers(32), &[Some(3 * GIB), Some(GIB)], 0).expect("a split");
        let text = split.describe(&["RX 7900".to_string(), "RX 5500M".to_string()]);
        assert_eq!(text, "layers 0-23 -> RX 7900, layers 24-31 -> RX 5500M");
    }
}
