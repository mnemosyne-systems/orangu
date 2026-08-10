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

//! *Which* physical device a GPU backend runs on — the policy, shared by
//! every backend, kept apart from any one API's bring-up code.
//!
//! `engine::backend::mod` picks the *API* (Vulkan, Metal, CUDA, OpenCL,
//! ROCm). Everything here answers the question that comes after it: a
//! machine with two cards, or the ordinary laptop with a discrete GPU
//! beside the CPU's integrated one, offers several devices through that
//! one API, and until this module existed orangu took whichever one the
//! driver handed back first.
//!
//! That was not a neutral default. `wgpu`'s `request_adapter` with
//! `PowerPreference::HighPerformance` is a *hint*: the answer is the
//! loader's, it varies by driver and by machine, and on a dual-GPU box it
//! is routinely the integrated one. The same trap is well documented on
//! the other side of the fence — `llama-server` has to be told
//! `--device Vulkan1` on this project's own dev machine to stop it
//! measuring the iGPU. A throughput number from an unnamed device is not
//! a throughput number.
//!
//! # The policy
//!
//! Rank by class first, then by size:
//!
//! 1. **Discrete** — a real card with its own VRAM. Largest first.
//! 2. **Other** — a device the API declined to classify. Treated as a real
//!    card rather than as a shared one, which is `llama.cpp`'s rule (only
//!    `VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU` becomes its `IGPU` device
//!    type; everything else that is not a CPU becomes a plain `GPU`).
//!    `colibri` ranks `OTHER` last instead; the disagreement only matters
//!    for hardware neither project has seen, and guessing "real card" fails
//!    in the safer direction — a wrong guess here is a slow run, whereas
//!    demoting a real card below an iGPU is the exact failure this module
//!    exists to prevent.
//! 3. **Virtual** — passthrough/hosted. Still a card, just not a local one.
//! 4. **Integrated** — an iGPU or APU, whose "VRAM" is a carve-out of the
//!    same system RAM the CPU is using. Real, and much better than nothing,
//!    but last among GPUs.
//! 5. **Software** — a CPU rasterizer pretending to be a GPU (llvmpipe,
//!    lavapipe, WARP). **Never selected automatically.** orangu already has
//!    a real CPU backend with AVX2 kernels; routing the forward pass
//!    through a software Vulkan driver instead would be slower and would
//!    report itself as a GPU run. Naming one explicitly still works — it is
//!    a legitimate way to exercise the GPU code path in CI — and says so
//!    when it happens.
//!
//! Ties inside a class break by reported VRAM (largest first), then by
//! enumeration index, so the choice is deterministic on a machine with two
//! identical cards.
//!
//! A device that reports no VRAM at all is ranked *below* one of the same
//! class that does, but is never excluded: "unknown" is not "zero". That is
//! `llama.cpp`'s own rule for its `--fit` accounting, which refuses to
//! *place* on a device reporting `0/0` but still lists it.
//!
//! # Being told, rather than guessing
//!
//! [`DeviceRequest`] is the override: an index, or a substring of the
//! device's name. A request that matches nothing is a startup error that
//! prints the whole inventory — never a silent fall-back to a different
//! device, which is `colibri`'s rule for `COLI_GPUS` and the only way an
//! A/B between two cards can be trusted.

use std::fmt;

use orangu::format::format_bytes;

/// What kind of device this is, in the order the selector prefers them.
///
/// Ordering is by [`DeviceClass::rank`] rather than by declaration order,
/// so adding a class later can't silently re-rank the existing ones.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceClass {
    /// A card with its own memory.
    Discrete,
    /// A real device the API didn't classify. See the module doc for why
    /// this outranks [`DeviceClass::Integrated`].
    Other,
    /// Passthrough or hosted.
    Virtual,
    /// iGPU/APU — shares the system RAM pool with the CPU.
    Integrated,
    /// CPU rasterizer. Never selected automatically.
    Software,
}

impl DeviceClass {
    /// Higher is preferred. See the module doc for the reasoning behind
    /// each step.
    pub fn rank(self) -> u8 {
        match self {
            Self::Discrete => 4,
            Self::Other => 3,
            Self::Virtual => 2,
            Self::Integrated => 1,
            Self::Software => 0,
        }
    }

    /// The one-word form used in the startup inventory and in `/props`.
    pub fn label(self) -> &'static str {
        match self {
            Self::Discrete => "discrete",
            Self::Other => "other",
            Self::Virtual => "virtual",
            Self::Integrated => "integrated",
            Self::Software => "software",
        }
    }

    /// Whether [`select`] refuses to reach this device without being asked
    /// for it by name or index.
    pub fn is_software(self) -> bool {
        self == Self::Software
    }
}

/// One device a backend's API reported, as far as it can be described
/// *before* committing to it — enumeration only, no device creation.
///
/// Every field except `index` and `name` is best-effort: the four backends
/// this feeds expose wildly different amounts of metadata, and a `None`
/// here means "this API didn't say", never "zero".
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceCandidate {
    /// Position in the backend's own enumeration, which is what an
    /// `index` [`DeviceRequest`] names. Stable within a process; the
    /// device *id* below is what stays stable across runs.
    pub index: usize,
    pub name: String,
    pub class: DeviceClass,
    /// Device-local memory, when the API reports it. A *capacity*, not a
    /// budget: nothing here knows what else is resident on the card, so
    /// this ranks devices and is not fit to decide whether a model fits.
    pub vram_total_bytes: Option<u64>,
    /// A stable identifier for the physical device — a PCI bus id where
    /// the API has one. Diagnostic today; the key a future multi-device
    /// path would deduplicate on, the way `llama.cpp` dedupes a card seen
    /// through two backends at once.
    pub id: Option<String>,
    pub driver: Option<String>,
}

impl DeviceCandidate {
    /// One line for the startup inventory, e.g.
    /// `0: AMD Radeon RX 5500M [discrete, 4.00 GiB, 0000:03:00.0]`.
    pub fn describe(&self) -> String {
        let mut detail = vec![self.class.label().to_string()];
        if let Some(vram) = self.vram_total_bytes {
            detail.push(format_bytes(vram));
        }
        if let Some(id) = &self.id {
            detail.push(id.clone());
        }
        format!("{}: {} [{}]", self.index, self.name, detail.join(", "))
    }

    /// Sort key for [`preference_order`]: class rank, then size, both
    /// descending. `Reverse` on the index keeps the overall sort ascending
    /// there while everything before it is descending.
    fn sort_key(&self) -> (std::cmp::Reverse<u8>, std::cmp::Reverse<u64>, usize) {
        (
            std::cmp::Reverse(self.class.rank()),
            std::cmp::Reverse(self.vram_total_bytes.unwrap_or(0)),
            self.index,
        )
    }
}

/// Which device the operator asked for, if any.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub enum DeviceRequest {
    /// Apply the ranking policy. The default.
    #[default]
    Auto,
    /// The device at this enumeration index.
    Index(usize),
    /// The one device whose name contains this, case-insensitively.
    Name(String),
}

impl DeviceRequest {
    /// Parses `auto`, a decimal index, or a name substring.
    ///
    /// Infallible on purpose: anything that isn't `auto` or a number is a
    /// name, and whether that name matches is a question only the
    /// enumerated device list can answer — which is where the error
    /// belongs, since that is where the list to print alongside it is.
    pub fn parse(raw: &str) -> Self {
        let raw = raw.trim();
        if raw.is_empty() || raw.eq_ignore_ascii_case("auto") {
            return Self::Auto;
        }
        match raw.parse::<usize>() {
            Ok(index) => Self::Index(index),
            Err(_) => Self::Name(raw.to_string()),
        }
    }

    /// Whether this is the ranking policy rather than a specific device.
    ///
    /// The caller needs the distinction because the two want opposite
    /// failure handling: `auto` finding nothing means "try the next
    /// backend, then the CPU", while a named device that isn't there means
    /// "stop and say so".
    pub fn is_auto(&self) -> bool {
        matches!(self, Self::Auto)
    }
}

impl fmt::Display for DeviceRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::Index(index) => write!(f, "{index}"),
            Self::Name(name) => write!(f, "{name}"),
        }
    }
}

/// Why [`select`] couldn't hand back a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceErrorKind {
    /// Nothing usable is present. Not the operator's fault, and not
    /// necessarily an error at all: under `auto` this is how a machine
    /// without a GPU reports itself, and the caller moves on to the next
    /// backend.
    Absent,
    /// The operator named a device this machine doesn't have. Always an
    /// error, never a fall-back — the whole point of naming one is that a
    /// different device silently answering would go unnoticed.
    Rejected,
}

/// A selection failure, carrying the message the operator should see —
/// which always includes the full inventory, because "device 3 not found"
/// without the list of what *is* there just prompts the next question.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceError {
    pub kind: DeviceErrorKind,
    message: String,
}

impl fmt::Display for DeviceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DeviceError {}

/// Every candidate's index, best first, by the policy in the module doc.
pub fn preference_order(candidates: &[DeviceCandidate]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by_key(|&i| candidates[i].sort_key());
    order
}

/// Every candidate `request` admits, best first, as positions in
/// `candidates`.
///
/// This is the shape the rest of the server works in, and the two answers
/// it can give are the two things an operator can mean:
///
/// - `Auto` — **the whole ranked set**, minus software rasterizers. The
///   default. One device runs the model today, and it is the head of this
///   list; the tail is what a device-splitting placement pass would walk,
///   in the order it should walk it.
/// - an index or a name — **exactly that device, dedicated**. Nothing else
///   is offered, so a run pinned to one card cannot quietly spill onto
///   another.
///
/// The distinction matters even while only the head is used: it is the
/// difference between "this machine has three devices and orangu chose the
/// RX 5500M" and "orangu was told to use the RX 5500M and nothing else",
/// and only the second survives someone plugging in a second card.
pub fn select_all(
    candidates: &[DeviceCandidate],
    request: &DeviceRequest,
) -> Result<Vec<usize>, DeviceError> {
    match request {
        DeviceRequest::Auto => {
            let selected: Vec<usize> = preference_order(candidates)
                .into_iter()
                .filter(|&i| !candidates[i].class.is_software())
                .collect();
            if selected.is_empty() {
                // The one-device path builds exactly this error, and its
                // two forms ("nothing at all" vs "only software") are the
                // whole point of routing through it rather than repeating
                // the message here.
                return Err(select(candidates, request)
                    .expect_err("an empty hardware set cannot select a device"));
            }
            Ok(selected)
        }
        _ => select(candidates, request).map(|position| vec![position]),
    }
}

/// The single candidate `request` names, as a position in `candidates`.
///
/// `Auto` gives the highest-ranked one and skips [`DeviceClass::Software`]
/// entirely (see the module doc); an explicit index or name reaches it.
/// [`select_all`] is the fuller answer — this is its head.
pub fn select(
    candidates: &[DeviceCandidate],
    request: &DeviceRequest,
) -> Result<usize, DeviceError> {
    let listing = || {
        if candidates.is_empty() {
            "no devices were reported".to_string()
        } else {
            let lines: Vec<String> = candidates.iter().map(|c| c.describe()).collect();
            format!("available devices:\n  {}", lines.join("\n  "))
        }
    };
    let reject = |message: String| DeviceError {
        kind: DeviceErrorKind::Rejected,
        message: format!("{message}; {}", listing()),
    };

    match request {
        DeviceRequest::Auto => preference_order(candidates)
            .into_iter()
            .find(|&i| !candidates[i].class.is_software())
            .ok_or_else(|| DeviceError {
                kind: DeviceErrorKind::Absent,
                // Deliberately two different sentences: "there is no
                // device" and "there is only a software rasterizer" lead to
                // completely different next steps, and the second one is
                // invisible unless it is said.
                message: if candidates.is_empty() {
                    "no device was found".to_string()
                } else {
                    format!(
                        "no hardware device was found — only software rasterizers, which \
                         are slower than orangu's own CPU backend and so are never chosen \
                         automatically; {}",
                        listing()
                    )
                },
            }),
        DeviceRequest::Index(index) => candidates
            .iter()
            .position(|c| c.index == *index)
            .ok_or_else(|| reject(format!("device {index} does not exist"))),
        DeviceRequest::Name(name) => {
            let needle = name.to_lowercase();
            let matches: Vec<usize> = candidates
                .iter()
                .enumerate()
                .filter(|(_, c)| c.name.to_lowercase().contains(&needle))
                .map(|(i, _)| i)
                .collect();
            match matches.as_slice() {
                [only] => Ok(*only),
                [] => Err(reject(format!("no device name contains {name:?}"))),
                // Ambiguity is rejected rather than resolved by rank: two
                // identical cards is exactly when picking one silently
                // would make an A/B between them meaningless.
                _ => Err(reject(format!(
                    "{} devices' names contain {name:?} — use an index instead",
                    matches.len()
                ))),
            }
        }
    }
}

/// What the selector decided about one device — the state the startup
/// inventory and `/props` both report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceRole {
    /// Running the model.
    InUse,
    /// Selected, but idle. Only reachable under `auto`, which selects the
    /// whole ranked set while one device runs the model.
    ///
    /// Reported rather than hidden, and reported as *idle* rather than as
    /// "available": a machine with a second card and a busy first one is a
    /// question ("why isn't it being used?") that deserves an answer on the
    /// same screen as the number that prompted it.
    Idle,
    /// Not selected, and why.
    Excluded(&'static str),
}

impl DeviceRole {
    /// The trailing phrase on this device's inventory line.
    pub fn label(self) -> &'static str {
        match self {
            Self::InUse => "<- in use",
            Self::Idle => "— selected, idle",
            Self::Excluded(why) => why,
        }
    }

    /// The `/props` spelling — a word, not a sentence.
    pub fn tag(self) -> &'static str {
        match self {
            Self::InUse => "in_use",
            Self::Idle => "idle",
            Self::Excluded(_) => "excluded",
        }
    }
}

/// Each candidate's role given `selected` (positions in `candidates`, best
/// first, as [`select_all`] returns them), in ranked order.
fn roles(candidates: &[DeviceCandidate], selected: &[usize]) -> Vec<(usize, DeviceRole)> {
    preference_order(candidates)
        .into_iter()
        .map(|position| {
            let role = match selected.iter().position(|&s| s == position) {
                Some(0) => DeviceRole::InUse,
                Some(_) => DeviceRole::Idle,
                None if candidates[position].class.is_software() => {
                    DeviceRole::Excluded("— not selected: software rasterizer")
                }
                None => DeviceRole::Excluded("— not selected"),
            };
            (position, role)
        })
        .collect()
}

/// The startup inventory: one line per device, **in ranked order**, saying
/// what each one is doing.
///
/// Printed unconditionally rather than behind a verbosity flag. These are
/// the lines that answer "which card did that measurement come from", and
/// they are worth nothing if they have to be turned on before the run they
/// describe. Ranked rather than enumeration order because the ranking is
/// the decision being reported — the enumeration index stays on every line,
/// since that is what `device = <n>` names.
pub fn inventory(api: &str, candidates: &[DeviceCandidate], selected: &[usize]) -> Vec<String> {
    roles(candidates, selected)
        .into_iter()
        .map(|(position, role)| {
            format!(
                "orangu-server: [{api}] {} {}",
                candidates[position].describe(),
                role.label()
            )
        })
        .collect()
}

/// The same inventory for `/props`, so a benchmark result carries the
/// device set it was taken on rather than just the name of the winner.
pub fn inventory_json(candidates: &[DeviceCandidate], selected: &[usize]) -> serde_json::Value {
    roles(candidates, selected)
        .into_iter()
        .map(|(position, role)| {
            let candidate = &candidates[position];
            serde_json::json!({
                "index": candidate.index,
                "name": candidate.name,
                "class": candidate.class.label(),
                "vram_total_bytes": candidate.vram_total_bytes,
                "id": candidate.id,
                "driver": candidate.driver,
                "role": role.tag(),
                // Kept alongside `role` because it is what every existing
                // reader of this field asks, and a reader that only knows
                // the old shape should not silently start seeing `false`
                // for the device that is running the model.
                "in_use": role == DeviceRole::InUse,
            })
        })
        .collect::<Vec<_>>()
        .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn device(index: usize, name: &str, class: DeviceClass, vram: Option<u64>) -> DeviceCandidate {
        DeviceCandidate {
            index,
            name: name.to_string(),
            class,
            vram_total_bytes: vram,
            id: None,
            driver: None,
        }
    }

    /// The case this module exists for: a laptop with a discrete card and
    /// the CPU's integrated one, where `request_adapter` routinely answered
    /// with the iGPU.
    #[test]
    fn auto_prefers_a_discrete_gpu_over_an_integrated_one() {
        let candidates = vec![
            device(
                0,
                "Intel UHD Graphics",
                DeviceClass::Integrated,
                Some(16 * GIB),
            ),
            device(
                1,
                "AMD Radeon RX 5500M",
                DeviceClass::Discrete,
                Some(4 * GIB),
            ),
        ];
        let chosen = select(&candidates, &DeviceRequest::Auto).expect("a device");
        assert_eq!(candidates[chosen].name, "AMD Radeon RX 5500M");
    }

    /// An iGPU's "VRAM" is the system RAM total, so it is routinely the
    /// biggest number in the list. Size must never outrank class.
    #[test]
    fn auto_ranks_class_above_size() {
        let candidates = vec![
            device(0, "iGPU", DeviceClass::Integrated, Some(64 * GIB)),
            device(1, "small card", DeviceClass::Discrete, Some(4 * GIB)),
        ];
        let chosen = select(&candidates, &DeviceRequest::Auto).expect("a device");
        assert_eq!(candidates[chosen].name, "small card");
    }

    #[test]
    fn auto_takes_the_largest_of_several_discrete_gpus() {
        let candidates = vec![
            device(0, "RX 5500M", DeviceClass::Discrete, Some(4 * GIB)),
            device(1, "RX 7900 XTX", DeviceClass::Discrete, Some(24 * GIB)),
            device(2, "RX 6600", DeviceClass::Discrete, Some(8 * GIB)),
        ];
        let chosen = select(&candidates, &DeviceRequest::Auto).expect("a device");
        assert_eq!(candidates[chosen].name, "RX 7900 XTX");
        // And the rest stay ranked, so a future second-device tier has an
        // order to walk rather than a winner and a bag.
        let order = preference_order(&candidates);
        let names: Vec<&str> = order.iter().map(|&i| candidates[i].name.as_str()).collect();
        assert_eq!(names, vec!["RX 7900 XTX", "RX 6600", "RX 5500M"]);
    }

    /// "Unknown" is not "zero": a card whose VRAM the API declined to
    /// report still outranks every lower class, it just sorts behind cards
    /// of its own class that did report.
    #[test]
    fn unknown_vram_ranks_last_within_its_class_but_still_beats_a_lower_class() {
        let candidates = vec![
            device(0, "iGPU", DeviceClass::Integrated, Some(32 * GIB)),
            device(1, "mystery card", DeviceClass::Discrete, None),
            device(2, "known card", DeviceClass::Discrete, Some(8 * GIB)),
        ];
        let order = preference_order(&candidates);
        let names: Vec<&str> = order.iter().map(|&i| candidates[i].name.as_str()).collect();
        assert_eq!(names, vec!["known card", "mystery card", "iGPU"]);
    }

    /// Two identical cards must not reorder between runs.
    #[test]
    fn identical_devices_keep_enumeration_order() {
        let candidates = vec![
            device(0, "RTX 4090", DeviceClass::Discrete, Some(24 * GIB)),
            device(1, "RTX 4090", DeviceClass::Discrete, Some(24 * GIB)),
        ];
        assert_eq!(preference_order(&candidates), vec![0, 1]);
        assert_eq!(select(&candidates, &DeviceRequest::Auto), Ok(0));
    }

    /// A software rasterizer is worse than `CpuBackend`, so `auto` must
    /// report *absence* and let the caller fall back — not select it.
    #[test]
    fn auto_refuses_a_software_rasterizer() {
        let candidates = vec![device(0, "llvmpipe (LLVM 19)", DeviceClass::Software, None)];
        let err = select(&candidates, &DeviceRequest::Auto).expect_err("no hardware device");
        assert_eq!(err.kind, DeviceErrorKind::Absent);
        assert!(err.to_string().contains("llvmpipe"), "{err}");
    }

    /// ...but naming it explicitly works, which is how the GPU code path
    /// gets exercised on a machine with no GPU.
    #[test]
    fn an_explicit_request_still_reaches_a_software_rasterizer() {
        let candidates = vec![
            device(0, "llvmpipe (LLVM 19)", DeviceClass::Software, None),
            device(1, "RX 7900 XTX", DeviceClass::Discrete, Some(24 * GIB)),
        ];
        assert_eq!(select(&candidates, &DeviceRequest::Index(0)), Ok(0));
        assert_eq!(
            select(&candidates, &DeviceRequest::Name("llvm".into())),
            Ok(0)
        );
    }

    #[test]
    fn an_empty_device_list_is_absent_not_rejected() {
        let err = select(&[], &DeviceRequest::Auto).expect_err("no devices");
        assert_eq!(err.kind, DeviceErrorKind::Absent);
    }

    /// The error a wrong index produces has to name what *is* there, or it
    /// only prompts the next question.
    #[test]
    fn a_bad_index_is_rejected_and_lists_every_device() {
        let candidates = vec![
            device(0, "RX 5500M", DeviceClass::Discrete, Some(4 * GIB)),
            device(1, "Intel UHD", DeviceClass::Integrated, Some(16 * GIB)),
        ];
        let err = select(&candidates, &DeviceRequest::Index(7)).expect_err("no device 7");
        assert_eq!(err.kind, DeviceErrorKind::Rejected);
        let message = err.to_string();
        assert!(message.contains("device 7 does not exist"), "{message}");
        assert!(message.contains("RX 5500M"), "{message}");
        assert!(message.contains("Intel UHD"), "{message}");
    }

    #[test]
    fn a_name_matches_case_insensitively_on_a_substring() {
        let candidates = vec![
            device(0, "Intel UHD Graphics", DeviceClass::Integrated, None),
            device(
                1,
                "AMD Radeon RX 5500M (RADV NAVI14)",
                DeviceClass::Discrete,
                None,
            ),
        ];
        assert_eq!(
            select(&candidates, &DeviceRequest::Name("navi".into())),
            Ok(1)
        );
    }

    #[test]
    fn an_ambiguous_name_is_rejected_rather_than_ranked() {
        let candidates = vec![
            device(0, "RTX 4090", DeviceClass::Discrete, Some(24 * GIB)),
            device(1, "RTX 4090", DeviceClass::Discrete, Some(24 * GIB)),
        ];
        let err = select(&candidates, &DeviceRequest::Name("4090".into())).expect_err("ambiguous");
        assert_eq!(err.kind, DeviceErrorKind::Rejected);
        assert!(err.to_string().contains("use an index"), "{err}");
    }

    /// An index names the *enumeration* position, which is not the position
    /// in the ranked list — a device list that skipped an unusable adapter
    /// must still answer to the number it prints.
    #[test]
    fn an_index_names_the_enumeration_position_not_the_rank() {
        let candidates = vec![
            device(3, "Intel UHD", DeviceClass::Integrated, None),
            device(5, "RX 7900 XTX", DeviceClass::Discrete, Some(24 * GIB)),
        ];
        assert_eq!(select(&candidates, &DeviceRequest::Index(5)), Ok(1));
        assert!(select(&candidates, &DeviceRequest::Index(1)).is_err());
    }

    #[test]
    fn parse_reads_auto_an_index_and_a_name() {
        assert_eq!(DeviceRequest::parse("auto"), DeviceRequest::Auto);
        assert_eq!(DeviceRequest::parse("  AUTO "), DeviceRequest::Auto);
        assert_eq!(DeviceRequest::parse(""), DeviceRequest::Auto);
        assert_eq!(DeviceRequest::parse("2"), DeviceRequest::Index(2));
        assert_eq!(
            DeviceRequest::parse("RX 7900"),
            DeviceRequest::Name("RX 7900".to_string())
        );
        // A negative number is not an index, and must not be silently read
        // as one — it becomes a name that will fail to match and print the
        // inventory, which is the more useful outcome than "device
        // 18446744073709551615 does not exist".
        assert_eq!(
            DeviceRequest::parse("-1"),
            DeviceRequest::Name("-1".to_string())
        );
    }

    /// The inventory reports in *ranked* order, so the first line is the
    /// device running the model — and every line still carries the
    /// enumeration index, which is what an operator would type back.
    #[test]
    fn the_inventory_leads_with_the_device_in_use() {
        let candidates = vec![
            device(0, "Intel UHD", DeviceClass::Integrated, Some(16 * GIB)),
            device(1, "RX 5500M", DeviceClass::Discrete, Some(4 * GIB)),
        ];
        let selected = select_all(&candidates, &DeviceRequest::Auto).expect("a device");
        let lines = inventory("vulkan", &candidates, &selected);
        assert_eq!(lines.len(), 2);
        assert!(
            lines[0].starts_with("orangu-server: [vulkan] 1: RX 5500M"),
            "{}",
            lines[0]
        );
        assert!(lines[0].contains("<- in use"), "{}", lines[0]);
        assert!(lines[0].contains("discrete"), "{}", lines[0]);
        assert!(lines[0].contains("4.00 GiB"), "{}", lines[0]);
        assert!(lines[1].contains("selected, idle"), "{}", lines[1]);
    }

    /// `auto` is every usable device, best first — the tail is what a
    /// device-splitting pass would walk, and reporting it is how a second
    /// idle card stops being invisible.
    #[test]
    fn auto_selects_every_hardware_device_in_ranked_order() {
        let candidates = vec![
            device(0, "iGPU", DeviceClass::Integrated, Some(32 * GIB)),
            device(1, "RX 5500M", DeviceClass::Discrete, Some(4 * GIB)),
            device(2, "llvmpipe", DeviceClass::Software, Some(64 * GIB)),
            device(3, "RX 7900 XTX", DeviceClass::Discrete, Some(24 * GIB)),
        ];
        let selected = select_all(&candidates, &DeviceRequest::Auto).expect("devices");
        let names: Vec<&str> = selected
            .iter()
            .map(|&i| candidates[i].name.as_str())
            .collect();
        assert_eq!(names, vec!["RX 7900 XTX", "RX 5500M", "iGPU"]);
        // The software rasterizer is excluded from the set, not merely
        // ranked last in it.
        assert!(!selected.contains(&2));
    }

    /// A named device is *dedicated*: the set is exactly one, so a run
    /// pinned to one card cannot spill onto another when placement across
    /// devices arrives.
    #[test]
    fn a_named_device_is_the_whole_set() {
        let candidates = vec![
            device(0, "iGPU", DeviceClass::Integrated, Some(32 * GIB)),
            device(1, "RX 5500M", DeviceClass::Discrete, Some(4 * GIB)),
        ];
        assert_eq!(
            select_all(&candidates, &DeviceRequest::Index(0)),
            Ok(vec![0])
        );
        assert_eq!(
            select_all(&candidates, &DeviceRequest::Name("5500".into())),
            Ok(vec![1])
        );
    }

    /// `select_all` must fail exactly the way `select` does, message and
    /// all — the software-only case is the one an operator most needs
    /// spelled out.
    #[test]
    fn select_all_reports_absence_the_same_way_the_single_pick_does() {
        let software = vec![device(0, "llvmpipe", DeviceClass::Software, None)];
        for candidates in [Vec::new(), software] {
            let one = select(&candidates, &DeviceRequest::Auto).expect_err("no hardware");
            let all = select_all(&candidates, &DeviceRequest::Auto).expect_err("no hardware");
            assert_eq!(all, one);
        }
    }

    #[test]
    fn an_excluded_device_says_why_it_was_excluded() {
        let candidates = vec![
            device(0, "RX 5500M", DeviceClass::Discrete, Some(4 * GIB)),
            device(1, "llvmpipe", DeviceClass::Software, None),
            device(2, "iGPU", DeviceClass::Integrated, Some(32 * GIB)),
        ];
        // Pinned to device 0: the iGPU is excluded because it wasn't asked
        // for, llvmpipe because of what it is. Two different answers.
        let selected = select_all(&candidates, &DeviceRequest::Index(0)).expect("a device");
        let lines = inventory("vulkan", &candidates, &selected);
        let iggy = lines
            .iter()
            .find(|l| l.contains("iGPU"))
            .expect("the iGPU line");
        let soft = lines
            .iter()
            .find(|l| l.contains("llvmpipe"))
            .expect("the llvmpipe line");
        assert!(iggy.ends_with("— not selected"), "{iggy}");
        assert!(soft.ends_with("software rasterizer"), "{soft}");
    }
}
