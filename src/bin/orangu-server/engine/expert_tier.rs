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

//! Which routed experts a *device* could hold, and what holding them would
//! be worth.
//!
//! `engine::placement` spreads a model's dense layers across devices;
//! `engine::expert_store` keeps hot experts in owned host memory under a
//! byte budget. This is the third question, and the one a second card
//! actually poses: given the VRAM left after the dense weights, which
//! experts should live there?
//!
//! # Whole experts, hottest first, fastest device first
//!
//! Three rules, all borrowed from `colibri`, which is the only engine
//! either reference tree has that ships a VRAM expert tier:
//!
//! - **Whole experts only.** An expert is never split across devices —
//!   `colibri/docs/cuda.md`: "a single expert is not sharded". Half an
//!   expert on a card is half an expert's worth of round trips for none of
//!   the benefit.
//! - **Hottest first.** Capacity is not the thing that decides whether a
//!   tier pays; the routing profile is. colibri measured the *same* 150 GB
//!   tier at 0.94–1.64 tok/s filled hot-first against 0.29 tok/s filled
//!   without routing heat — a 3–5× difference from ordering alone.
//! - **Fastest device first.** An expert goes to the first device with room
//!   for it, so the primary card fills before a secondary one is touched.
//!   That is `COLI_VK_DEV2`'s rule: the second device holds the *next*
//!   heat-ranked experts after the first device's budget stops, and the
//!   primary hot path is left alone.
//!
//! # What this is for right now
//!
//! **The tier executes.** `main::plan_expert_tier` fills it from real routing
//! heat (`ORANGU_EXPERT_USAGE`) or by size, records it through
//! `LoadedModel::set_expert_residency`, and every `ExpertQuantMatrix` built
//! afterwards carries the per-expert flags that `arch::device_expert_admissible`
//! gates the device dispatch on. It runs behind `ORANGU_GPU_EXPERTS`, and the
//! startup line names how many experts it placed and which of heat or size
//! filled it.
//!
//! This paragraph used to say the opposite — that nothing executed on a device
//! expert tier yet. That stopped being true when `LLAMA.md` L1 added the
//! residency check to `gpu_project_expert`, and it stayed wrong long enough
//! that an outside reader of the source drew the wrong conclusion from it.
//! **A doc comment is part of the source, and this one was lying.**
//!
//! [`coverage`] is still the number the decision turns on: the share of real
//! routing traffic a tier of a given size would serve. A tier that covers 4%
//! of selections cannot pay for itself whatever the kernels look like; one
//! that covers 60% might.
//!
//! # And the measurement came back negative
//!
//! The honest prior was already unfavourable — colibri's own conclusion is
//! that a GPU expert tier "earns its VRAM only when the CPU is the weak
//! link", and orangu's expert matmul is a tuned AVX2/rayon path over host
//! memory that a small card has to beat while also paying a round trip per
//! dispatch.
//!
//! It is no longer a prior. `LLAMA.md` L1 measured the device expert path
//! losing at 11.8% and at 82.8% residency, per-expert and batched, at decode
//! and at prefill. `DISK.md`'s D2b then reached **44.4% coverage** — three and
//! a half times the ceiling L1 was stuck at — on a model chosen to make the
//! tier large relative to the card, and measured decode **9.4% slower with the
//! tier than without it**, prefill inside the noise.
//!
//! So residency was never the limiter, and this module's value is now the
//! projection rather than the placement: [`coverage`] is what tells an
//! operator, before the first token, that a tier on their card would serve a
//! share too small to pay for the VRAM it takes from the KV cache.

/// One expert a tier could hold: what it costs, and how much traffic it
/// would serve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertHeat {
    /// The expert's own quantized bytes — [`crate::engine::loader::
    /// ExpertQuantMatrix::expert_bytes`].
    pub bytes: u64,
    /// How often routing selected it. Any consistent unit: the ordering and
    /// the ratio are what matter, not the scale.
    ///
    /// `1` for every expert models "no profile" — a tier filled without
    /// routing heat, which is the case colibri measured as 3–5× worse and
    /// which [`coverage`] will duly report as covering only the share of
    /// experts it holds.
    pub heat: u64,
}

/// Which experts a tier holds, and where.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ExpertTierPlan {
    /// `Some(device)` for a resident expert, `None` for one left on the
    /// host — parallel to the `heat` slice it was planned from.
    pub device_of: Vec<Option<usize>>,
    pub per_device_bytes: Vec<u64>,
    pub resident_heat: u64,
    pub total_heat: u64,
}

impl ExpertTierPlan {
    pub fn resident_count(&self) -> usize {
        self.device_of.iter().filter(|d| d.is_some()).count()
    }

    /// The share of routing traffic this tier would serve, `0.0..=1.0`.
    ///
    /// The number the decision actually turns on. `None` when nothing was
    /// routed at all, which is not zero coverage — it is no evidence.
    pub fn coverage(&self) -> Option<f64> {
        (self.total_heat > 0).then(|| self.resident_heat as f64 / self.total_heat as f64)
    }

    pub fn resident_bytes(&self) -> u64 {
        self.per_device_bytes.iter().sum()
    }
}

/// The coverage below which a resident tier is not built.
///
/// **On a small card one subsystem's allocator is another's ceiling.** VRAM
/// the tier takes is VRAM the KV cache and the transient arenas do not get,
/// so a tier serving a small share of routing is not merely useless — it is
/// paid for by the things that are working.
///
/// A quarter is a guard against the futile case, not a tuned optimum, and
/// saying so matters: the *only* coverage this project has measured
/// end-to-end is **44.4%, and it was 9.4% slower on decode** (see `DISK.md`
/// D2b). So a tier passing this floor is not thereby known to pay — it is
/// merely not known to be pointless. The floor exists to stop the case where
/// the budget bought 12% of the traffic, which needs no measurement to reject.
const COVERAGE_FLOOR_DEFAULT: f64 = 0.25;

/// `ORANGU_EXPERT_TIER_FLOOR`, as a percentage — the same units the startup
/// line prints, so an operator reading "12.7%" can type `13` and get the
/// behaviour they expected. Out-of-range and unparseable values are ignored
/// rather than clamped: a typo must not silently pick a different policy.
pub fn coverage_floor() -> f64 {
    static FLOOR: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *FLOOR.get_or_init(|| {
        std::env::var("ORANGU_EXPERT_TIER_FLOOR")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|percent| (0.0..=100.0).contains(percent))
            .map(|percent| percent / 100.0)
            .unwrap_or(COVERAGE_FLOOR_DEFAULT)
    })
}

/// Whether a planned tier is worth the VRAM, given what is known about it.
///
/// Three cases, and the middle one is the whole design:
///
/// - **Coverage measured and below the floor** — decline. The budget would
///   buy a share of routing too small to pay for the VRAM it costs.
/// - **Coverage unknown** — build it. `None` means nothing was routed yet, so
///   the tier was filled by size and there is *no evidence either way*; that
///   is the state every first run is in, and refusing there would make the
///   feature unreachable without a profiling session the operator cannot run
///   until the feature works. An operator who set `ORANGU_GPU_EXPERTS` asked
///   for this; absence of evidence must not read as evidence of absence.
/// - **Coverage at or above the floor** — build it, and let the operator's
///   own measurement decide from there.
pub fn worth_building(coverage: Option<f64>, floor: f64) -> bool {
    match coverage {
        Some(coverage) => coverage >= floor,
        None => true,
    }
}

/// Fills `budgets` with the hottest experts that fit, whole, fastest device
/// first.
///
/// An expert too large for the device being filled is passed over rather
/// than ending the fill: the next expert down the heat order may still fit,
/// and stopping at the first non-fitting one would leave usable VRAM empty
/// for no reason. Ties in heat break by index, so the same profile always
/// produces the same tier.
pub fn plan(heat: &[ExpertHeat], budgets: &[u64]) -> ExpertTierPlan {
    let mut order: Vec<usize> = (0..heat.len()).collect();
    order.sort_by(|&a, &b| heat[b].heat.cmp(&heat[a].heat).then(a.cmp(&b)));

    let mut plan = ExpertTierPlan {
        device_of: vec![None; heat.len()],
        per_device_bytes: vec![0; budgets.len()],
        resident_heat: 0,
        total_heat: heat.iter().map(|e| e.heat).sum(),
    };
    for index in order {
        let expert = heat[index];
        // Zero-heat experts are admitted too, once the hot ones are placed:
        // unused VRAM serves nobody, and colibri fills the remainder the
        // same way (`PIN_FILL`). They simply sort last, so they can never
        // displace an expert that routing actually asked for.
        let Some(device) = budgets
            .iter()
            .enumerate()
            .position(|(d, budget)| plan.per_device_bytes[d] + expert.bytes <= *budget)
        else {
            continue;
        };
        plan.device_of[index] = Some(device);
        plan.per_device_bytes[device] += expert.bytes;
        plan.resident_heat += expert.heat;
    }
    plan
}

/// What a tier of `budgets` would be worth against `heat`, in one line
/// each — a projection, and labelled as one *when it is one*.
///
/// Printed at startup for a MoE model on a GPU, because the alternative is
/// that the question "would a VRAM expert tier help on this machine?" can
/// only be answered by building one first.
///
/// `active` is whether a tier is really running (`ORANGU_GPU_EXPERTS`).
/// Without it the closing line claimed "no tier is active" unconditionally,
/// which on a run that had just printed `expert tier: N of M experts on
/// device` is not a stale caveat but a direct contradiction of the line
/// above it — and the reader who believes the wrong one of the two goes
/// looking for a bug in the wrong half of the engine.
pub fn projection(
    api: &str,
    plan: &ExpertTierPlan,
    n_expert: usize,
    uniform: bool,
    active: bool,
) -> Vec<String> {
    let mut lines = Vec::new();
    let share = if n_expert > 0 {
        100.0 * plan.resident_count() as f64 / n_expert as f64
    } else {
        0.0
    };
    lines.push(format!(
        "orangu-server: [{api}] a device expert tier in the free VRAM would hold {} of {} \
         experts ({:.1}%, {})",
        plan.resident_count(),
        n_expert,
        share,
        orangu::format::format_bytes(plan.resident_bytes()),
    ));
    // The distinction is the whole point of reporting coverage separately
    // from capacity: with a real profile, a small tier can serve a large
    // share of traffic, and without one it cannot do better than its size.
    match (plan.coverage(), uniform) {
        (Some(coverage), false) => lines.push(format!(
            "orangu-server: [{api}] measured routing would find {:.1}% of its selections there",
            coverage * 100.0
        )),
        _ => lines.push(format!(
            "orangu-server: [{api}] no routing profile, so that is also its expected hit \
             rate — a tier filled by heat serves far more traffic than one filled by size"
        )),
    }
    lines.push(if active {
        format!(
            "orangu-server: [{api}] the tier above is active: a resident expert runs on the \
             device, the rest on the CPU. See the manual's \"Expert tiers\" section."
        )
    } else {
        format!(
            "orangu-server: [{api}] projection only: experts run on the CPU, and no tier is \
             active. See the manual's \"Expert tiers\" section."
        )
    });
    lines
}

#[cfg(test)]
mod tests {

    /// D2c's acceptance test: the threshold, from both sides, with no gap
    /// between them. `>=` rather than `>` is deliberate — a tier that exactly
    /// meets the floor meets it.
    #[test]
    fn the_coverage_floor_is_tested_from_both_sides() {
        let floor = 0.25;
        assert!(
            !worth_building(Some(0.249), floor),
            "just under must decline"
        );
        assert!(
            worth_building(Some(0.25), floor),
            "exactly at the floor is in"
        );
        assert!(worth_building(Some(0.251), floor), "just over must build");
        // The case the floor exists for.
        assert!(!worth_building(Some(0.127), floor));
        // And the case it must not catch.
        assert!(worth_building(Some(1.0), floor));
    }

    /// Unknown coverage is not low coverage. `None` means nothing has been
    /// routed yet, which is the state of every first run — refusing there
    /// would make the tier unreachable, because the routing profile that
    /// would raise coverage above the floor can only be recorded by running
    /// with the tier on.
    #[test]
    fn unknown_coverage_builds_the_tier_rather_than_declining_it() {
        for floor in [0.0, 0.25, 0.99, 1.0] {
            assert!(
                worth_building(None, floor),
                "no routing evidence must not read as evidence of no routing (floor {floor})"
            );
        }
    }

    /// A floor of zero has to mean "never decline", or an operator disabling
    /// the guard would still hit it on a tier covering exactly nothing.
    #[test]
    fn a_zero_floor_never_declines() {
        assert!(worth_building(Some(0.0), 0.0));
    }
    use super::*;

    const MIB: u64 = 1024 * 1024;

    fn experts(heats: &[u64], bytes: u64) -> Vec<ExpertHeat> {
        heats
            .iter()
            .map(|&heat| ExpertHeat { bytes, heat })
            .collect()
    }

    /// The rule the whole tier turns on: the hottest experts get the seats.
    #[test]
    fn the_hottest_experts_are_resident() {
        let heat = experts(&[1, 50, 3, 100], MIB);
        let plan = plan(&heat, &[2 * MIB]);
        assert_eq!(plan.device_of, vec![None, Some(0), None, Some(0)]);
        assert_eq!(plan.resident_count(), 2);
    }

    /// Coverage, not capacity, is what says whether a tier pays: two of four
    /// experts here is 97% of the traffic.
    #[test]
    fn coverage_reports_traffic_served_not_seats_filled() {
        let heat = experts(&[1, 50, 3, 100], MIB);
        let plan = plan(&heat, &[2 * MIB]);
        assert_eq!(plan.coverage(), Some(150.0 / 154.0));
        assert!(plan.coverage().unwrap() > 0.97);
    }

    /// Without a profile a tier cannot do better than its own size — which
    /// is exactly the case colibri measured as 3-5x worse, and the reason
    /// the report says so out loud.
    #[test]
    fn uniform_heat_makes_coverage_equal_the_share_of_experts_held() {
        let heat = experts(&[1, 1, 1, 1], MIB);
        let plan = plan(&heat, &[2 * MIB]);
        assert_eq!(plan.resident_count(), 2);
        assert_eq!(plan.coverage(), Some(0.5));
    }

    /// The primary device fills before a secondary one is touched — colibri's
    /// `COLI_VK_DEV2` rule, which keeps the fast card's hot path intact.
    #[test]
    fn the_first_device_fills_before_the_second() {
        let heat = experts(&[100, 90, 80, 70], MIB);
        let plan = plan(&heat, &[2 * MIB, 2 * MIB]);
        assert_eq!(
            plan.device_of,
            vec![Some(0), Some(0), Some(1), Some(1)],
            "hottest two on device 0"
        );
        assert_eq!(plan.per_device_bytes, vec![2 * MIB, 2 * MIB]);
    }

    /// Never sharded: an expert bigger than every budget stays on the host
    /// rather than being cut in half.
    #[test]
    fn an_expert_too_large_for_any_device_stays_on_the_host() {
        let heat = vec![
            ExpertHeat {
                bytes: 8 * MIB,
                heat: 1000,
            },
            ExpertHeat {
                bytes: MIB,
                heat: 1,
            },
        ];
        let plan = plan(&heat, &[2 * MIB]);
        assert_eq!(plan.device_of, vec![None, Some(0)]);
        // And its traffic is honestly *not* counted as covered.
        assert_eq!(plan.resident_heat, 1);
    }

    /// One expert that doesn't fit must not end the fill — the next one down
    /// may, and leaving VRAM empty for no reason is worse than a slightly
    /// out-of-order tier.
    #[test]
    fn a_non_fitting_expert_is_passed_over_rather_than_stopping_the_fill() {
        let heat = vec![
            ExpertHeat {
                bytes: MIB,
                heat: 100,
            },
            ExpertHeat {
                bytes: 8 * MIB,
                heat: 90,
            },
            ExpertHeat {
                bytes: MIB,
                heat: 80,
            },
        ];
        let plan = plan(&heat, &[2 * MIB]);
        assert_eq!(plan.device_of, vec![Some(0), None, Some(0)]);
    }

    /// A budget of nothing holds nothing, and says so as zero coverage
    /// rather than as no evidence.
    #[test]
    fn an_empty_budget_holds_nothing() {
        let heat = experts(&[5, 5], MIB);
        let plan = plan(&heat, &[0]);
        assert_eq!(plan.resident_count(), 0);
        assert_eq!(plan.coverage(), Some(0.0));
        assert_eq!(plan.resident_bytes(), 0);
    }

    /// No routing at all is *no evidence*, which is a different answer from
    /// zero coverage and must not be reported as one.
    #[test]
    fn no_routing_traffic_is_no_evidence_rather_than_zero_coverage() {
        let heat = experts(&[0, 0], MIB);
        let plan = plan(&heat, &[4 * MIB]);
        assert_eq!(
            plan.resident_count(),
            2,
            "cold experts still fill free VRAM"
        );
        assert_eq!(plan.coverage(), None);
    }

    /// The same profile must always produce the same tier, or an A/B between
    /// two tier sizes measures the tie-break as well as the size.
    #[test]
    fn equal_heat_breaks_ties_by_index() {
        let heat = experts(&[7, 7, 7, 7], MIB);
        let plan = plan(&heat, &[2 * MIB]);
        assert_eq!(plan.device_of, vec![Some(0), Some(0), None, None]);
    }

    #[test]
    fn the_projection_states_that_it_is_a_projection() {
        let heat = experts(&[3, 2, 1], MIB);
        let plan = plan(&heat, &[MIB]);
        let text = projection("vulkan", &plan, 3, false, false).join("\n");
        assert!(text.contains("1 of 3 experts"), "{text}");
        assert!(text.contains("projection only"), "{text}");
    }

    /// An active tier must not be described as one that does not exist.
    ///
    /// The two lines are emitted together, so a closing caveat that ignores
    /// `active` contradicts the capacity line directly above it rather than
    /// merely being out of date.
    #[test]
    fn an_active_tier_is_not_reported_as_projection_only() {
        let heat = experts(&[3, 2, 1], MIB);
        let plan = plan(&heat, &[MIB]);
        let text = projection("vulkan", &plan, 3, false, true).join("\n");
        assert!(text.contains("1 of 3 experts"), "{text}");
        assert!(
            !text.contains("projection only"),
            "an active tier was reported as a projection: {text}"
        );
        assert!(text.contains("active"), "{text}");
    }
}
