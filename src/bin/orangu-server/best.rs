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

//! `--sort best`: the installed models ranked for this machine, for
//! `orangu-server list` (all of them) and `orangu-server suggest` (the top
//! ten, one row per repo).
//!
//! A model that fits comes before one that only runs by streaming its
//! experts, which comes before one that does not fit, which comes before one
//! this build cannot serve. Within each of those, newer comes first. A
//! model's date is when its Hugging Face repo was created, or its file's
//! date when there is no repo or no answer. The version in its name refines
//! that: inside one family (`Qwen3.5`, `Qwen3.8`) a higher version is never
//! ranked as older than a lower one, whatever order the repos were created in.

use std::collections::{HashMap, HashSet};
use std::time::UNIX_EPOCH;

use orangu::format::format_bytes;
use orangu::gguf::GgufFile;
use orangu::hardware::{self, CpuInfo, GpuInfo};
use orangu::model_spec::{ListView, ModelGroup, ModelSupport, RankCells};
use orangu::npu::NpuInfo;
use orangu::os::OsInfo;
use rayon::prelude::*;

use crate::config::{BackendPreference, ServerConfiguration};
use crate::engine;

/// How many rows `suggest --sort best` prints.
const TOP: usize = 10;

/// The memory a model is weighed against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Memory {
    ram: u64,
    /// The largest dedicated card a backend can use, or `0`.
    card: u64,
    /// What `plan --json` calls this machine's capacity.
    capacity: u64,
}

/// What a model needs, from its tensor tables.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Needs {
    dense: u64,
    expert: u64,
    device: u64,
    kv: u64,
}

/// Where a model runs on this machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fit {
    /// What a backend uploads fits on the card, the rest in RAM.
    Gpu,
    /// All of it fits in RAM.
    Ram,
    /// All of it fits only across RAM and the card together.
    Split,
    /// The dense part fits; experts are read from disk as they are routed.
    Streams,
    /// Not even the dense part fits.
    No,
}

impl Fit {
    fn of(needs: Needs, memory: Memory) -> Self {
        let dense = needs.dense + needs.kv;
        let whole = dense + needs.expert;
        if dense > memory.capacity {
            return Fit::No;
        }
        if whole > memory.capacity {
            return Fit::Streams;
        }
        let on_card = needs.device + needs.kv;
        if on_card <= memory.card && whole.saturating_sub(on_card) <= memory.ram {
            Fit::Gpu
        } else if whole <= memory.ram {
            Fit::Ram
        } else {
            Fit::Split
        }
    }

    fn cell(self) -> &'static str {
        match self {
            Fit::Gpu => "GPU",
            Fit::Ram => "RAM",
            Fit::Split => "GPU+RAM",
            Fit::Streams => "Streams",
            Fit::No => "No",
        }
    }
}

/// One installed model, as the ranking sees it.
#[derive(Clone, Debug)]
struct Candidate<'a> {
    /// The repo id or file label the version is read from.
    name: &'a str,
    quant: Option<&'a str>,
    size: u64,
    /// `None` when this build cannot serve it.
    fit: Option<Fit>,
    date: Option<u64>,
}

/// Reads `(family, version)` from a model's name: `unsloth/Qwen3.8-27B-GGUF`
/// is `qwen` 3.8, `Llama-3.2-1B` is `llama` 3.2, `DeepSeek-V4.1-Flash` is
/// `deepseek-v` 4.1. The first number decides, and a size (`27B`, `A3B`)
/// reached before any version means the name has none.
fn family_and_version(name: &str) -> Option<(String, Vec<u32>)> {
    let name = name.rsplit('/').next()?.split(':').next()?;
    let mut family = Vec::new();
    for token in name.split(['-', '_', ' ']) {
        let Some(digit) = token.find(|c: char| c.is_ascii_digit()) else {
            family.push(token.to_ascii_lowercase());
            continue;
        };
        let (prefix, number) = token.split_at(digit);
        let prefix = prefix.strip_suffix('.').unwrap_or(prefix);
        if !prefix.chars().all(|c| c.is_ascii_alphabetic()) {
            return None;
        }
        let version = parse_version(number)?;
        if !prefix.is_empty() {
            family.push(prefix.to_ascii_lowercase());
        }
        if family.is_empty() {
            return None;
        }
        return Some((family.join("-"), version));
    }
    None
}

/// `3.10` as `[3, 10]`. Trailing zeros are dropped so that `3.0` and `3` are
/// one version.
fn parse_version(text: &str) -> Option<Vec<u32>> {
    let mut parts = text
        .split('.')
        .map(|part| part.parse().ok())
        .collect::<Option<Vec<u32>>>()?;
    while parts.last() == Some(&0) {
        parts.pop();
    }
    Some(parts)
}

/// How `download` would choose among a repo's quantizations: its default
/// tags, then anything quantized, then plain floats.
fn quant_preference(quant: Option<&str>) -> usize {
    let quant = quant.unwrap_or_default();
    if let Some(index) = orangu::model_download::DEFAULT_TAG_PREFERENCE
        .iter()
        .position(|tag| *tag == quant)
    {
        return index;
    }
    if matches!(quant, "F32" | "F16" | "BF16") {
        orangu::model_download::DEFAULT_TAG_PREFERENCE.len() + 1
    } else {
        orangu::model_download::DEFAULT_TAG_PREFERENCE.len()
    }
}

/// Each candidate's date for ranking: its own, or the newest of any lower
/// version in its family when that is later. A single number per model keeps
/// the order total, which comparing versions pairwise beside dates would not.
fn ranking_dates(candidates: &[Candidate], versions: &[Option<(String, Vec<u32>)>]) -> Vec<u64> {
    candidates
        .iter()
        .zip(versions)
        .map(|(candidate, version)| {
            let own = candidate.date.unwrap_or(0);
            let Some((family, version)) = version else {
                return own;
            };
            candidates
                .iter()
                .zip(versions)
                .filter(|(other, _)| other.fit.is_some())
                .filter_map(|(other, other_version)| {
                    let (other_family, other_version) = other_version.as_ref()?;
                    (other_family == family && other_version < version)
                        .then_some(other.date)
                        .flatten()
                })
                .fold(own, u64::max)
        })
        .collect()
}

/// `candidates`' indices, best first.
fn rank(candidates: &[Candidate]) -> Vec<usize> {
    let versions: Vec<_> = candidates
        .iter()
        .map(|candidate| family_and_version(candidate.name))
        .collect();
    let dates = ranking_dates(candidates, &versions);
    let tier = |fit: Option<Fit>| match fit {
        Some(Fit::Gpu | Fit::Ram | Fit::Split) => 0,
        Some(Fit::Streams) => 1,
        Some(Fit::No) => 2,
        None => 3,
    };
    let placement = |fit: Option<Fit>| match fit {
        Some(Fit::Gpu) => 0,
        Some(Fit::Ram) => 1,
        _ => 2,
    };
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by_cached_key(|&index| {
        let candidate = &candidates[index];
        (
            tier(candidate.fit),
            std::cmp::Reverse(dates[index]),
            std::cmp::Reverse(versions[index].as_ref().map(|(_, version)| version.clone())),
            placement(candidate.fit),
            quant_preference(candidate.quant),
            std::cmp::Reverse(candidate.size),
        )
    });
    order
}

/// The installed models ranked for this machine.
pub(crate) struct Ranking {
    /// Indices into `groups`, best first.
    pub(crate) order: Vec<usize>,
    /// The `DATE` and `FIT` cells, aligned to `groups`.
    pub(crate) cells: Vec<RankCells>,
    fits: Vec<Option<Fit>>,
    memory: Memory,
    context: usize,
}

impl Ranking {
    /// The first [`TOP`] models that run here, one per repo: the rest of a
    /// repo's quantizations are the same model again.
    fn top(&self, groups: &[ModelGroup]) -> Vec<usize> {
        let mut repos = HashSet::new();
        self.order
            .iter()
            .copied()
            .filter(|&index| {
                matches!(
                    self.fits[index],
                    Some(Fit::Gpu | Fit::Ram | Fit::Split | Fit::Streams)
                )
            })
            .filter(|&index| repos.insert(repo_of(&groups[index])))
            .take(TOP)
            .collect()
    }
}

fn repo_of(group: &ModelGroup) -> &str {
    group.hf_repo.as_deref().unwrap_or(&group.label)
}

/// Each repo's creation time: the registry's, then the Hub's for what the
/// registry does not have yet, which it then keeps.
fn repo_dates(groups: &[ModelGroup]) -> HashMap<String, u64> {
    let mut known = orangu::model_registry::repo_created_times();
    let missing: Vec<String> = groups
        .iter()
        .filter_map(|group| group.hf_repo.clone())
        .filter(|repo| !known.contains_key(repo))
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let found = orangu::model_download::repo_created_times(&missing);
    // Only a cache: an unwritable registry costs a lookup on the next run.
    let _ = orangu::model_registry::record_repo_created(&found);
    known.extend(found);
    known
}

/// The newest modification time among a model's files, in Unix seconds.
fn file_date(group: &ModelGroup) -> Option<u64> {
    group
        .paths
        .iter()
        .filter_map(|path| std::fs::metadata(path).ok()?.modified().ok())
        .filter_map(|time| Some(time.duration_since(UNIX_EPOCH).ok()?.as_secs()))
        .max()
}

fn format_date(timestamp: u64) -> String {
    chrono::DateTime::from_timestamp(timestamp as i64, 0)
        .map(|time| time.format("%Y-%m-%d").to_string())
        .unwrap_or_else(|| "-".to_string())
}

/// What `group` needs at `context`, from its tensor tables and hyperparameters.
fn needs_of(group: &ModelGroup, context: usize) -> Option<Needs> {
    let plan =
        engine::plan::analyze_shards(group.paths.iter().map(|path| GgufFile::open_summary(path)))
            .ok()?;
    Some(Needs {
        dense: plan.dense_bytes,
        expert: plan.expert_bytes,
        device: plan.device_bytes,
        kv: crate::kv_bytes_at(&group.representative_path, context).unwrap_or(0),
    })
}

/// Ranks `groups` against this machine's memory, at the configured context.
pub(crate) fn rank_installed(
    groups: &[ModelGroup],
    support: &[ModelSupport],
    conf: &ServerConfiguration,
    cpu: &CpuInfo,
    gpus: &[GpuInfo],
) -> Ranking {
    let memory = Memory {
        ram: cpu.total_memory_bytes,
        card: if matches!(conf.backend, BackendPreference::Cpu) {
            0
        } else {
            crate::suggest::largest_dedicated_gpu(gpus)
                .and_then(|gpu| gpu.vram_total_bytes)
                .unwrap_or(0)
        },
        capacity: cpu.total_memory_bytes + crate::device_memory_bytes(conf, gpus),
    };
    let context = crate::requested_context(None, conf.context).unwrap_or(crate::PLAN_KV_CONTEXT);

    let fits: Vec<Option<Fit>> = groups
        .par_iter()
        .zip(support)
        .map(|(group, support)| {
            let servable =
                group.errors.is_empty() && support.loadable() && support.companion_of.is_none();
            servable
                .then(|| needs_of(group, context))
                .flatten()
                .map(|needs| Fit::of(needs, memory))
        })
        .collect();

    let repos = repo_dates(groups);
    let dates: Vec<(Option<u64>, bool)> = groups
        .iter()
        .map(|group| {
            match group
                .hf_repo
                .as_ref()
                .and_then(|repo| repos.get(repo).copied())
            {
                Some(created) => (Some(created), false),
                None => (file_date(group), true),
            }
        })
        .collect();

    let candidates: Vec<Candidate> = groups
        .iter()
        .enumerate()
        .map(|(index, group)| Candidate {
            name: repo_of(group),
            quant: group.quantization.as_deref(),
            size: group.size_bytes,
            fit: fits[index],
            date: dates[index].0,
        })
        .collect();

    let cells = dates
        .iter()
        .zip(&fits)
        .map(|(&(date, estimated), fit)| RankCells {
            date: match date {
                Some(date) if estimated => format!("~{}", format_date(date)),
                Some(date) => format_date(date),
                None => "-".to_string(),
            },
            fit: fit.map_or("-", Fit::cell).to_string(),
        })
        .collect();

    Ranking {
        order: rank(&candidates),
        cells,
        fits,
        memory,
        context,
    }
}

/// `suggest --sort best`'s report: the machine, then its top models.
pub(crate) fn format_top(
    os: &OsInfo,
    cpu: &CpuInfo,
    gpus: &[GpuInfo],
    npu: Option<&NpuInfo>,
    groups: &[ModelGroup],
    support: &[ModelSupport],
    ranking: &Ranking,
) -> String {
    let mut out = hardware::format_report(os, cpu, gpus, npu, &hardware::detect_power());
    let order = ranking.top(groups);
    out.push_str(&format!("\nBest installed models (top {})\n", order.len()));
    out.push_str(&format!(
        "  Memory           : {} RAM",
        format_bytes(ranking.memory.ram)
    ));
    if ranking.memory.card > 0 {
        out.push_str(&format!(", {} GPU", format_bytes(ranking.memory.card)));
    }
    out.push_str(&format!("\n  Context          : {}\n\n", ranking.context));
    if order.is_empty() {
        out.push_str("No installed model runs on this machine.\n");
        return out;
    }
    out.push_str(&orangu::model_spec::format_groups_in_view(
        groups,
        std::path::Path::new(""),
        &HashMap::new(),
        support,
        orangu::model_spec::Dimming::Off,
        ListView {
            last_used: Some(&orangu::model_registry::last_used_for(
                groups.iter().map(|group| group.paths.as_slice()),
            )),
            ranking: Some(&ranking.cells),
            order: &order,
        },
    ));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    fn candidate<'a>(name: &'a str, date: u64) -> Candidate<'a> {
        Candidate {
            name,
            quant: Some("Q4_K_M"),
            size: GIB,
            fit: Some(Fit::Ram),
            date: Some(date),
        }
    }

    fn names<'a>(candidates: &[Candidate<'a>]) -> Vec<&'a str> {
        rank(candidates)
            .into_iter()
            .map(|index| candidates[index].name)
            .collect()
    }

    #[test]
    fn versions_read_from_the_names_on_disk() {
        for (name, family, version) in [
            ("unsloth/Qwen3.8-27B-GGUF", "qwen", vec![3, 8]),
            ("unsloth/Qwen3-4B-GGUF", "qwen", vec![3]),
            (
                "unsloth/Qwen2.5-Coder-0.5B-Instruct-GGUF",
                "qwen",
                vec![2, 5],
            ),
            (
                "unsloth/Qwen-Image-2.1-Turbo-GGUF",
                "qwen-image",
                vec![2, 1],
            ),
            ("unsloth/gemma-4-E2B-it-GGUF", "gemma", vec![4]),
            ("bartowski/Llama-3.2-1B-Instruct-GGUF", "llama", vec![3, 2]),
            ("bartowski/SmolLM2-360M-Instruct-GGUF", "smollm", vec![2]),
            ("unsloth/FLUX.2-klein-base-4B-GGUF", "flux", vec![2]),
            (
                "unsloth/DeepSeek-V4.1-Flash-1017-GGUF",
                "deepseek-v",
                vec![4, 1],
            ),
            ("unsloth/MiniMax-M3.1-Flash-GGUF", "minimax-m", vec![3, 1]),
            ("unsloth/embeddinggemma-2-GGUF", "embeddinggemma", vec![2]),
            ("bartowski/Ling-3.0-tiny-GGUF", "ling", vec![3]),
        ] {
            assert_eq!(
                family_and_version(name),
                Some((family.to_string(), version)),
                "{name}"
            );
        }
    }

    /// A size is not a version, and neither is the `A3B` of a MoE's name.
    #[test]
    fn a_size_is_not_a_version() {
        for name in [
            "unsloth/Muse-Glimmer-30B-GGUF",
            "ggml-org/embeddinggemma-300M-GGUF",
            "unsloth/Inkling-Small-GGUF",
            "A3B-GGUF",
        ] {
            assert_eq!(family_and_version(name), None, "{name}");
        }
    }

    /// Numeric, not textual: 3.10 is after 3.8.
    #[test]
    fn versions_compare_by_number() {
        assert!(parse_version("3.10") > parse_version("3.8"));
        assert_eq!(parse_version("3.0"), parse_version("3"));
    }

    #[test]
    fn newer_comes_first() {
        let candidates = [candidate("a-model", 10), candidate("b-model", 30)];
        assert_eq!(names(&candidates), ["b-model", "a-model"]);
    }

    /// The repo for 3.8 being older than the repo for 3.5 does not make 3.8
    /// the older model.
    #[test]
    fn a_higher_version_outranks_a_lower_one_of_any_date() {
        let candidates = [
            candidate("unsloth/Qwen3.8-27B-GGUF", 10),
            candidate("unsloth/Qwen3.5-9B-GGUF", 20),
            candidate("unsloth/Other-1-GGUF", 15),
        ];
        assert_eq!(
            names(&candidates),
            [
                "unsloth/Qwen3.8-27B-GGUF",
                "unsloth/Qwen3.5-9B-GGUF",
                "unsloth/Other-1-GGUF",
            ]
        );
    }

    /// Fit decides before the date does: the newest model is no use if it
    /// cannot run here.
    #[test]
    fn what_fits_comes_before_what_is_newer() {
        let with = |name, date, fit| Candidate {
            fit,
            ..candidate(name, date)
        };
        let candidates = [
            with("unservable", 50, None),
            with("too-big", 40, Some(Fit::No)),
            with("streams", 30, Some(Fit::Streams)),
            with("fits-old", 10, Some(Fit::Ram)),
            with("fits-new", 20, Some(Fit::Gpu)),
        ];
        assert_eq!(
            names(&candidates),
            ["fits-new", "fits-old", "streams", "too-big", "unservable"]
        );
    }

    /// A model this build cannot serve lifts nothing.
    #[test]
    fn an_unservable_model_lifts_nothing() {
        let candidates = [
            Candidate {
                fit: None,
                ..candidate("Qwen-1-GGUF", 100)
            },
            candidate("Qwen-2-GGUF", 10),
            candidate("Other-1-GGUF", 20),
        ];
        assert_eq!(names(&candidates)[..2], ["Other-1-GGUF", "Qwen-2-GGUF"]);
    }

    /// One repo's quantizations share a date: the card comes first, then
    /// `download`'s default, then anything quantized before a plain float.
    #[test]
    fn a_repo_s_own_rows_order_by_placement_and_quantization() {
        let row = |quant, size, fit| Candidate {
            name: "repo",
            quant: Some(quant),
            size,
            fit: Some(fit),
            date: Some(1),
        };
        let candidates = [
            row("BF16", 8 * GIB, Fit::Ram),
            row("Q6_K", 4 * GIB, Fit::Ram),
            row("Q4_K_M", 3 * GIB, Fit::Ram),
            row("IQ1_S", GIB, Fit::Gpu),
        ];
        let quants: Vec<_> = rank(&candidates)
            .into_iter()
            .map(|index| candidates[index].quant.unwrap())
            .collect();
        assert_eq!(quants, ["IQ1_S", "Q4_K_M", "Q6_K", "BF16"]);
    }

    #[test]
    fn fit_weighs_the_card_ram_and_experts() {
        let memory = Memory {
            ram: 64 * GIB,
            card: 8 * GIB,
            capacity: 72 * GIB,
        };
        let needs = |dense: u64, expert: u64, device: u64| Needs {
            dense: dense * GIB,
            expert: expert * GIB,
            device: device * GIB,
            kv: 0,
        };
        assert_eq!(Fit::of(needs(4, 0, 4), memory), Fit::Gpu);
        assert_eq!(Fit::of(needs(32, 0, 32), memory), Fit::Ram);
        assert_eq!(Fit::of(needs(70, 0, 70), memory), Fit::Split);
        assert_eq!(Fit::of(needs(32, 100, 32), memory), Fit::Streams);
        assert_eq!(Fit::of(needs(80, 0, 80), memory), Fit::No);
        // The KV cache counts: it is what tips this one off the card.
        let with_kv = Needs {
            kv: 5 * GIB,
            ..needs(4, 0, 4)
        };
        assert_eq!(Fit::of(with_kv, memory), Fit::Ram);
    }

    /// Without a dedicated card nothing is placed on one.
    #[test]
    fn no_card_means_ram() {
        let memory = Memory {
            ram: 64 * GIB,
            card: 0,
            capacity: 64 * GIB,
        };
        let needs = Needs {
            dense: GIB,
            device: GIB,
            ..Needs::default()
        };
        assert_eq!(Fit::of(needs, memory), Fit::Ram);
    }
}
