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

//! What a model would need to run here, answered **without loading it**.
//!
//! A 434 GB model takes a long time to find out about the hard way. Every
//! number here comes from the GGUF tensor table — names, shapes and types —
//! which is a few hundred kilobytes at the head of each shard, so a plan for a
//! model far larger than this machine costs about as long as `ls`.
//!
//! # What it is for
//!
//! On a mixture-of-experts model the question is not "does it fit" but "what
//! has to be *resident* and what can stream". Those are different quantities
//! and only one of them is the file size:
//!
//! - the **dense** part — attention, norms, embeddings, shared experts — is
//!   touched by every single token, so it has to be in memory or the model is
//!   unusable at any speed;
//! - the **routed experts** are touched a handful at a time, so they can live
//!   on disk and be fetched as the router asks for them.
//!
//! A model whose dense part fits and whose experts do not is *slow*. A model
//! whose dense part does not fit is *not going to work*, and that is worth
//! knowing in seconds rather than after a thirty-minute load.

use std::path::Path;

use anyhow::{Result, anyhow};

use crate::engine::loader;
use crate::engine::loader::block_index;
use crate::engine::quant;
use orangu::gguf::GgufFile;

/// What the model needs, in bytes, split by how it would be used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    pub shards: usize,
    pub architecture: String,
    /// Every tensor's bytes, as the files store them — including the draft
    /// head, which is on disk whether or not anything runs it.
    pub total_bytes: u64,
    /// Weights every token touches: everything that is not a routed expert
    /// and not part of the draft head.
    pub dense_bytes: u64,
    /// Weights only the router's choices touch.
    pub expert_bytes: u64,
    /// Weights a GPU backend would upload, by
    /// [`crate::engine::backend::is_cpu_only_tensor`]'s own rule rather than
    /// a second copy of it.
    ///
    /// Not the same as [`dense_bytes`](Self::dense_bytes), and the difference
    /// is the point: routed *and* shared experts have no GPU path, so on a
    /// mixture-of-experts model the device holds less than the dense part —
    /// shared experts are dense (every token runs them) yet still live in
    /// host memory. Charging them to a card would overstate what a plan says
    /// the GPU has to hold.
    pub device_bytes: u64,
    /// Weights belonging to a trailing multi-token-prediction (draft) block,
    /// which this engine does not run — see [`trunk_block_count`]. `0` for
    /// the models that have none, which is most of them.
    pub draft_bytes: u64,
    /// Experts per MoE layer, and how many of them a token uses.
    pub n_expert: usize,
    pub n_expert_used: usize,
    /// Layers that carry routed experts.
    pub moe_layers: usize,
    /// One expert's weights, across every per-expert tensor of one layer.
    pub bytes_per_expert: u64,
}

impl Plan {
    /// Expert bytes one token's routing touches, across the whole model.
    ///
    /// The quantity that decides whether a streaming model is usable: it is
    /// what must arrive from wherever the experts live, **per token**, and no
    /// placement policy can make it smaller. Only a cache that already holds
    /// them can make it free.
    pub fn expert_bytes_per_token(&self) -> u64 {
        self.bytes_per_expert * self.n_expert_used as u64 * self.moe_layers as u64
    }

    pub fn is_moe(&self) -> bool {
        self.moe_layers > 0 && self.n_expert > 0
    }

    /// Whether the model can run here **at all**, as distinct from running
    /// slowly.
    ///
    /// The dense part is what every token touches, so it is the part that has
    /// to be resident: below this line a model does not work, and above it a
    /// model whose experts also do not fit is merely slow, because experts
    /// stream. One expression covers both kinds of model — on a dense one,
    /// everything *is* the dense part.
    ///
    /// This is the same test both verdict lines are written from, and the one
    /// `orangu-server download` asks before deciding whether fetching the
    /// model is worth confirming. Keeping it here is what stops the prompt and
    /// the printed verdict from ever disagreeing.
    pub fn dense_fits_in(&self, available_ram: u64) -> bool {
        self.dense_bytes <= available_ram
    }

    /// How much larger than `vram` the weights a GPU backend would upload
    /// are, or `None` when they fit.
    ///
    /// The same question `engine::footprint`'s `shortfall_on` asks *after*
    /// loading, asked here from the file's tensor table instead. It exists
    /// because the two used to disagree by construction: a plan that only
    /// weighed system RAM answered "fits in RAM with 23.9 GiB to spare" for
    /// a model that then reported being 17.3 GiB larger than the card the
    /// server actually selected. A verdict is only useful against the
    /// hardware the model is going to land on.
    pub fn device_shortfall(&self, vram: u64) -> Option<u64> {
        (self.device_bytes > vram).then(|| self.device_bytes - vram)
    }
}

/// How many of a model's `block_count` blocks this engine actually runs.
///
/// Some architectures append a **multi-token-prediction** block: a
/// self-contained draft head that predicts a token beyond the next one, for a
/// speculative decoder to check. It is counted in `block_count` and it carries
/// a full set of tensors — including, on a mixture-of-experts model, an
/// entire layer's worth of experts — but nothing in the trunk reads it, and
/// this engine has no second-model speculative path to use it with, so
/// `engine::arch`'s `glm`, `deepseek4` and `nemotron` all stop loading before
/// it.
///
/// A plan that counted its tensors as weights to be held would overstate
/// exactly the number a reader is consulting the plan for. `None` when the
/// file names no such block, which is the common case.
fn trunk_block_count(gguf: &GgufFile, architecture: &str) -> Option<usize> {
    let meta = |suffix: &str| -> Option<u64> {
        let key = format!("{architecture}.{suffix}");
        gguf.metadata
            .iter()
            .find(|(k, _)| *k == key)
            .and_then(|(_, v)| v.as_u64())
    };
    let n_draft = meta("nextn_predict_layers").filter(|&n| n > 0)? as usize;
    meta("block_count")?
        .checked_sub(n_draft as u64)
        .map(|n| n as usize)
}

/// Reads every shard's tensor table and classifies it.
///
/// A tensor is a routed expert when its name ends in `_exps.weight` — the GGUF
/// convention for the stacked `[in, out, n_expert]` tensors
/// (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`, and the fused
/// `ffn_gate_up_exps`). Everything else, including the *shared* expert
/// (`ffn_*_shexp`), is dense: a shared expert runs for every token, so it has
/// to be resident whatever the router decides.
///
/// A tensor belonging to a trailing draft block is neither — see
/// [`trunk_block_count`]. It counts toward the on-disk total and toward
/// nothing else.
pub fn analyze(path: &Path) -> Result<Plan> {
    let first = GgufFile::open(path)?;
    // Shard 1's table is already parsed; re-opening it would be a second read
    // of the same few hundred kilobytes.
    let rest: Vec<_> = loader::shard_paths(path, &first)?
        .into_iter()
        .skip(1)
        .collect();
    analyze_shards(std::iter::once(Ok(first)).chain(rest.iter().map(|shard| GgufFile::open(shard))))
}

/// [`analyze`] over shards that have already been parsed, or that come from
/// somewhere other than this machine's disk.
///
/// Every number a [`Plan`] carries comes from tensor tables and a handful of
/// metadata keys, and neither depends on where the bytes were read from. That
/// is what lets `orangu-server download` plan a model **it has not
/// downloaded** — `orangu::model_download::RemoteModel::headers` yields the
/// same `GgufFile` values over HTTP that [`analyze`] gets from `open`, and
/// this classifier cannot tell the difference.
///
/// The first shard supplies the architecture and the expert-count metadata;
/// every shard, the first included, supplies tensors. Shards are consumed
/// lazily and dropped as they go, so a model with dozens of them never holds
/// more than one table at a time.
pub fn analyze_shards<I>(shards: I) -> Result<Plan>
where
    I: IntoIterator<Item = Result<GgufFile>>,
{
    let mut shards = shards.into_iter();
    let first = shards
        .next()
        .ok_or_else(|| anyhow!("a model needs at least one shard to plan"))??;

    let architecture = first
        .metadata
        .iter()
        .find(|(k, _)| k == "general.architecture")
        .and_then(|(_, v)| match v {
            orangu::gguf::GgufValue::String(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_else(|| "unknown".to_string());
    let (n_expert, n_expert_used) = {
        let meta_u64 = |suffix: &str| -> Option<u64> {
            let key = format!("{architecture}.{suffix}");
            first
                .metadata
                .iter()
                .find(|(k, _)| *k == key)
                .and_then(|(_, v)| v.as_u64())
        };
        (
            meta_u64("expert_count").unwrap_or(0) as usize,
            meta_u64("expert_used_count").unwrap_or(0) as usize,
        )
    };

    let trunk_blocks = trunk_block_count(&first, &architecture);

    let mut shard_count = 0usize;
    let mut total_bytes = 0u64;
    let mut draft_bytes = 0u64;
    let mut device_bytes = 0u64;
    let mut expert_bytes = 0u64;
    let mut moe_layers = std::collections::HashSet::new();
    let mut per_layer_expert_bytes = 0u64;
    let mut seen_first_moe_layer: Option<String> = None;

    for gguf in std::iter::once(Ok(first)).chain(shards) {
        let gguf = gguf?;
        shard_count += 1;
        for tensor in &gguf.tensors {
            let elements: u64 = tensor.dims.iter().product();
            let bytes = quant::tensor_byte_size(tensor.ggml_type, elements).unwrap_or(0);
            total_bytes += bytes;
            // The draft head is on disk but never loaded, so it is neither
            // resident nor streamable — it is simply not part of what running
            // this model costs.
            if let (Some(trunk), Some(block)) = (trunk_blocks, block_index(&tensor.name))
                && block >= trunk
            {
                draft_bytes += bytes;
                continue;
            }
            // Asked of the backend rather than decided here, so a plan's
            // "what has to fit on the card" cannot drift from what the
            // backend actually uploads.
            if !crate::engine::backend::is_cpu_only_tensor(&tensor.name) {
                device_bytes += bytes;
            }
            if !tensor.name.ends_with("_exps.weight") {
                continue;
            }
            expert_bytes += bytes;
            let layer = tensor
                .name
                .split('.')
                .nth(1)
                .unwrap_or_default()
                .to_string();
            moe_layers.insert(layer.clone());
            // One layer's per-expert tensors are enough to size an expert;
            // summing every layer's would multiply by the layer count twice.
            match &seen_first_moe_layer {
                None => {
                    seen_first_moe_layer = Some(layer);
                    per_layer_expert_bytes += bytes;
                }
                Some(first_layer) if *first_layer == layer => {
                    per_layer_expert_bytes += bytes;
                }
                Some(_) => {}
            }
        }
    }

    let bytes_per_expert = if n_expert > 0 {
        per_layer_expert_bytes / n_expert as u64
    } else {
        0
    };
    Ok(Plan {
        shards: shard_count,
        architecture,
        total_bytes,
        dense_bytes: total_bytes
            .saturating_sub(expert_bytes)
            .saturating_sub(draft_bytes),
        expert_bytes,
        device_bytes,
        draft_bytes,
        n_expert,
        n_expert_used,
        moe_layers: moe_layers.len(),
        bytes_per_expert,
    })
}

/// One byte figure, in whichever unit reads best for it.
///
/// `orangu::format::format_bytes` rather than a fixed unit, and rather than a
/// formatter of this module's own: it is what `list` prints a model's size
/// with and what `engine::footprint` prints the startup weight lines with, so
/// a plan and the server now render agreeing numbers identically instead of
/// merely agreeing about them.
///
/// Every figure here used to carry a hardcoded unit — `{:.1} GiB` for the
/// large ones, `{:.1} MiB` for the per-expert ones — which broke at both ends
/// of the range a plan has to cover. A 318 MiB embedding model read
/// `0.3 GiB on disk`, throwing away most of the precision it had; a large
/// mixture-of-experts model read `Per token 14473.1 MiB`, a number nobody
/// can weigh against the GiB figures three lines above it.
fn size(bytes: u64) -> String {
    orangu::format::format_bytes(bytes)
}

/// The GPU a plan is judged against: the card the server would actually
/// select, its capacity, and its name.
///
/// A pair rather than a bare byte count because a machine with several GPUs
/// makes a lone number unreadable — the integrated one reports the whole of
/// system RAM as its memory, so a plan that named no card would look like it
/// had tens of gigabytes to play with on a box whose real target is a 4 GiB
/// discrete card. `None` on a machine with no dedicated GPU, where system
/// RAM is the only ceiling that matters.
pub type PlanDevice<'a> = Option<(&'a str, u64)>;

/// The plan as a report, with the machine's own memory beside it.
///
/// Deliberately states the *verdict* rather than only the numbers. The numbers
/// are what a reader would have to combine themselves to answer the question
/// they actually have, which is whether to press on.
///
/// Both ceilings are reported, because a model has to clear both and they fail
/// differently: too big for RAM is fatal, while too big for the card is the
/// driver paging weights in and out on every token — slow rather than broken,
/// and invisible unless something says so.
pub fn format_plan(plan: &Plan, available_ram: u64, device: PlanDevice<'_>) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Model      {} · {} shard{} · {} on disk\n",
        plan.architecture,
        plan.shards,
        if plan.shards == 1 { "" } else { "s" },
        size(plan.total_bytes),
    ));

    if !plan.is_moe() {
        out.push_str(&format!(
            "Dense      {} — every byte is touched by every token\n",
            size(plan.dense_bytes)
        ));
        out.push_str(&draft_line(plan));
        out.push_str(&this_box(available_ram, device));
        // Only worth saying when the model has to stream: for one that fits,
        // the per-token figure is a number nothing reads from disk.
        if !plan.dense_fits_in(available_ram) {
            out.push_str(&per_token_dense(plan));
        }
        out.push_str(&verdict_dense(plan, available_ram));
        out.push_str(&device_verdict(plan, device));
        return out;
    }

    out.push_str(&format!(
        "Dense      {} — attention, norms, embeddings, shared experts. Must be resident.\n",
        size(plan.dense_bytes)
    ));
    out.push_str(&format!(
        "Experts    {} — {} per layer x {} layers, {} each. Can stream.\n",
        size(plan.expert_bytes),
        plan.n_expert,
        plan.moe_layers,
        size(plan.bytes_per_expert),
    ));
    out.push_str(&format!(
        "Per token  {} of experts ({} of {} per layer, {} layers)\n",
        size(plan.expert_bytes_per_token()),
        plan.n_expert_used,
        plan.n_expert,
        plan.moe_layers,
    ));
    out.push_str(&draft_line(plan));
    out.push_str(&this_box(available_ram, device));
    out.push_str(&verdict_moe(plan, available_ram));
    out.push_str(&device_verdict(plan, device));
    out
}

/// What the machine has, on one line — RAM always, and the GPU by name when
/// there is one worth judging against.
///
/// Printed for a dense model too. It used to appear only under the
/// mixture-of-experts branch, which meant the *most* common case got a
/// verdict with nothing beside it to check the verdict against.
fn this_box(available_ram: u64, device: PlanDevice<'_>) -> String {
    format!(
        "This box   {} RAM available{}\n",
        size(available_ram),
        device.map_or(String::new(), |(name, vram)| format!(
            ", {} VRAM ({name})",
            size(vram)
        )),
    )
}

/// The GPU half of the verdict: silent when the weights fit the card, or
/// when there is no card to fit them.
///
/// The wording deliberately matches what `engine::footprint` prints at
/// startup once the model is loaded, because it is the same finding — the
/// point of saying it here is that here it arrives while it can still change
/// the decision.
fn device_verdict(plan: &Plan, device: PlanDevice<'_>) -> String {
    // The card is named once, on the `This box` line above. Repeating it here
    // is what made this line wrap: real GPU names run to sixty characters
    // ("Navi 14 [Radeon RX 5500/5500M / Pro 5300/5300M/5500M]").
    let Some((_, vram)) = device else {
        return String::new();
    };
    let Some(short) = plan.device_shortfall(vram) else {
        return format!(
            "Device     {} of weights on a {} GPU — fits, {} spare\n",
            size(plan.device_bytes),
            size(vram),
            size(vram - plan.device_bytes),
        );
    };
    format!(
        concat!(
            "Device     {} too large for this GPU ({}) — the server will spread the model\n",
            "           across every device in order and run the remainder on the CPU, which\n",
            "           is slow rather than fatal. A smaller quantization avoids it.\n",
        ),
        size(short),
        size(vram),
    )
}

/// The draft-head line, or nothing at all for a model without one.
///
/// Stated rather than silently dropped: a reader adding the printed figures
/// up against the file size is entitled to find the difference accounted for,
/// and "this much of the file is never read" is itself worth knowing.
fn draft_line(plan: &Plan) -> String {
    if plan.draft_bytes == 0 {
        return String::new();
    }
    format!(
        "Draft head {} — a multi-token-prediction block this engine does not run. Never loaded.\n",
        size(plan.draft_bytes),
    )
}

/// The per-token line for a dense model, which is the one figure a reader
/// takes away and the one that used to be missing.
///
/// **Prefill and decode are stated separately because they differ by three
/// orders of magnitude**, and quoting only the decode number — which is what
/// "per token" naturally means — reads as "this model is unusable" for a
/// workload where it is perfectly usable. Measured under a 4 GiB cap: 6,533
/// MiB per token at decode against 6.33 MiB per prompt token at prefill,
/// because decode walks the whole model to produce one token and prefill
/// walks it once for the entire prompt.
///
/// **No seconds figure, deliberately.** It would need a read rate, and the
/// only honest one is a measurement: the drive this was sized against runs at
/// 210 MB/s in a short probe and 52 MB/s under the sustained reading that
/// streaming actually does — a factor of four, decided by duty cycle rather
/// than by the drive. Quoting either number alone misleads, probing at plan
/// time costs the `ls`-speed this module's whole design rests on, and bytes
/// are exact and free. So the bytes are stated and the arithmetic is left to
/// a reader who knows their own storage.
fn per_token_dense(plan: &Plan) -> String {
    format!(
        concat!(
            "Per token  {} at decode — every byte, for every token. At prefill the same {}\n",
            "           covers the whole prompt, so a long prompt costs a fraction of that.\n",
        ),
        size(plan.dense_bytes),
        size(plan.dense_bytes),
    )
}

fn verdict_dense(plan: &Plan, available_ram: u64) -> String {
    let total = plan.dense_bytes;
    if plan.dense_fits_in(available_ram) {
        format!(
            "Verdict    fits in RAM with {} to spare\n",
            size(available_ram - total)
        )
    } else {
        // Not "nothing to stream": a dense model streams the whole of itself,
        // which is slow rather than impossible. The old wording said the
        // model would not run, and it does — measured at 992 s/token on a
        // 57 GiB model against 35 GiB of RAM, with no cgroup involved.
        format!(
            concat!(
                "Verdict    runnable by streaming: {} more than RAM holds, so that much comes\n",
                "           off disk every token. The storage under the model sets the speed.\n",
            ),
            size(total - available_ram)
        )
    }
}

fn verdict_moe(plan: &Plan, available_ram: u64) -> String {
    if !plan.dense_fits_in(available_ram) {
        return format!(
            concat!(
                "Verdict    runnable but slow: the dense part alone is {} short of RAM and is\n",
                "           touched by every token, so it streams as well as the experts do.\n",
            ),
            size(plan.dense_bytes - available_ram)
        );
    }
    let spare = available_ram - plan.dense_bytes;
    if plan.expert_bytes <= spare {
        format!(
            "Verdict    fits entirely in RAM ({} to spare); nothing needs to stream\n",
            size(spare - plan.expert_bytes)
        )
    } else {
        format!(
            "Verdict    runnable by streaming: dense fits, {} of experts do not and will \
             come off disk\n           at {} per token, the storage under the model sets \
             the speed\n",
            size(plan.expert_bytes - spare),
            size(plan.expert_bytes_per_token()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moe_plan() -> Plan {
        Plan {
            shards: 11,
            architecture: "glm-dsa".into(),
            total_bytes: 400 << 30,
            dense_bytes: 10 << 30,
            expert_bytes: 390 << 30,
            device_bytes: 10 << 30,
            draft_bytes: 0,
            n_expert: 256,
            n_expert_used: 8,
            moe_layers: 75,
            bytes_per_expert: 20 << 20,
        }
    }

    /// The number that decides whether a streaming model is usable at all,
    /// and the one nobody can work out by looking at a file size.
    #[test]
    fn per_token_expert_bytes_multiply_out_across_layers() {
        let plan = moe_plan();
        assert_eq!(plan.expert_bytes_per_token(), (20 << 20) * 8 * 75);
    }

    /// A dense part that does not fit is a different answer from experts that
    /// do not fit: one is "slow", the other is "no".
    #[test]
    fn a_dense_part_larger_than_ram_is_reported_as_unworkable() {
        let mut plan = moe_plan();
        plan.dense_bytes = 40 << 30;
        let report = format_plan(&plan, 20 << 30, None);
        assert!(report.contains("runnable but slow"), "{report}");
        assert!(report.contains("20.00 GiB short"), "{report}");
    }

    #[test]
    fn experts_larger_than_the_spare_ram_are_reported_as_streaming() {
        let report = format_plan(&moe_plan(), 60 << 30, None);
        assert!(report.contains("runnable by streaming"), "{report}");
        assert!(report.contains("per token"), "{report}");
    }

    #[test]
    fn a_model_that_fits_entirely_says_nothing_needs_to_stream() {
        let mut plan = moe_plan();
        plan.expert_bytes = 5 << 30;
        let report = format_plan(&plan, 60 << 30, None);
        assert!(report.contains("fits entirely in RAM"), "{report}");
    }

    /// A draft head is on disk but never loaded, so it must count toward the
    /// file size and toward neither of the two figures a reader is actually
    /// consulting the plan for. Before this, its experts were counted as
    /// streamable and its per-layer share inflated `Per token`.
    #[test]
    fn a_draft_head_is_charged_to_neither_resident_nor_streamed() {
        let mut plan = moe_plan();
        // Carved out of the dense side, where an unfixed `analyze` would have
        // left it: the three parts still have to sum to the file.
        plan.draft_bytes = 5 << 30;
        plan.dense_bytes = 5 << 30;
        let report = format_plan(&plan, 60 << 30, None);
        assert!(report.contains("Draft head 5.00 GiB"), "{report}");
        assert!(report.contains("does not run"), "{report}");
        // And the three parts still account for the whole file.
        assert_eq!(
            plan.dense_bytes + plan.expert_bytes + plan.draft_bytes,
            plan.total_bytes
        );
    }

    /// The overwhelming majority of models have no draft head, and their
    /// reports must not grow a line about one.
    #[test]
    fn a_model_without_a_draft_head_says_nothing_about_one() {
        let report = format_plan(&moe_plan(), 60 << 30, None);
        assert!(!report.contains("Draft head"), "{report}");
    }

    /// `block_count` counts the draft block, so the trunk is what remains —
    /// and a model that names no draft block has no bound at all.
    #[test]
    fn trunk_block_count_subtracts_only_a_declared_draft_block() {
        let gguf = |kv: &[(&str, u64)]| orangu::gguf::GgufFile {
            metadata: kv
                .iter()
                .map(|(k, v)| (k.to_string(), orangu::gguf::GgufValue::U64(*v)))
                .collect(),
            tensors: Vec::new(),
            data_offset: 0,
            alignment: 32,
            version: 3,
        };
        assert_eq!(
            trunk_block_count(
                &gguf(&[("a.block_count", 53), ("a.nextn_predict_layers", 1)]),
                "a"
            ),
            Some(52)
        );
        // No key, and an explicit zero, both mean "every block is trunk".
        assert_eq!(
            trunk_block_count(&gguf(&[("a.block_count", 53)]), "a"),
            None
        );
        assert_eq!(
            trunk_block_count(
                &gguf(&[("a.block_count", 53), ("a.nextn_predict_layers", 0)]),
                "a"
            ),
            None
        );
    }

    #[test]
    fn block_index_reads_only_a_block_scoped_tensor_name() {
        assert_eq!(block_index("blk.52.ffn_up_exps.weight"), Some(52));
        assert_eq!(block_index("blk.7.attn_norm.weight"), Some(7));
        assert_eq!(block_index("token_embd.weight"), None);
        assert_eq!(block_index("output_norm.weight"), None);
    }

    /// A dense model has no experts to stream, so "short of RAM" is terminal
    /// rather than merely slow — and the report must not offer streaming as
    /// an option that does not exist.
    #[test]
    fn a_dense_model_is_judged_on_its_whole_size() {
        let plan = Plan {
            shards: 1,
            architecture: "llama".into(),
            total_bytes: 30 << 30,
            dense_bytes: 30 << 30,
            expert_bytes: 0,
            device_bytes: 30 << 30,
            draft_bytes: 0,
            n_expert: 0,
            n_expert_used: 0,
            moe_layers: 0,
            bytes_per_expert: 0,
        };
        assert!(!plan.is_moe());
        let short = format_plan(&plan, 8 << 30, None);
        assert!(short.contains("runnable by streaming"), "{short}");
        assert!(
            short.contains("comes\n           off disk every token"),
            "{short}"
        );
        assert!(format_plan(&plan, 60 << 30, None).contains("fits in RAM"));
    }

    /// One `f32` tensor of `elements` elements, named `name`. `f32` because
    /// its byte size is exactly four per element, so a test can state the
    /// numbers it expects rather than reproduce a quantization's block
    /// arithmetic to predict them.
    fn f32_tensor(name: &str, elements: u64) -> orangu::gguf::TensorInfo {
        orangu::gguf::TensorInfo {
            name: name.to_string(),
            dims: vec![elements],
            // `GGML_TYPE_F32`.
            ggml_type: 0,
            offset: 0,
        }
    }

    fn shard(
        metadata: &[(&str, u64)],
        tensors: Vec<orangu::gguf::TensorInfo>,
    ) -> orangu::gguf::GgufFile {
        orangu::gguf::GgufFile {
            metadata: metadata
                .iter()
                .map(|(k, v)| {
                    (
                        k.to_string(),
                        if *k == "general.architecture" {
                            orangu::gguf::GgufValue::String("testmoe".to_string())
                        } else {
                            orangu::gguf::GgufValue::U64(*v)
                        },
                    )
                })
                .collect(),
            tensors,
            data_offset: 0,
            alignment: 32,
            version: 3,
        }
    }

    /// The classifier is the same one whether the tables came off this disk or
    /// off the network, and this is the test that says so: nothing here opens
    /// a file. It is what lets `download` plan a repo it has not fetched.
    ///
    /// Also pins the two things easiest to get wrong when summing across
    /// shards — that a tensor is classified by its *name* wherever it lands,
    /// and that one expert's size comes from **one** layer rather than from
    /// every layer summed.
    #[test]
    fn analyze_shards_classifies_tables_from_anywhere() {
        // 4 experts, 2 used, two MoE layers split across two shards. Each
        // layer carries one 4096-element expert tensor = 16 KiB, so one
        // expert is 4 KiB.
        let meta = &[
            ("general.architecture", 0),
            ("testmoe.expert_count", 4),
            ("testmoe.expert_used_count", 2),
        ];
        let plan = analyze_shards(vec![
            Ok(shard(
                meta,
                vec![
                    f32_tensor("token_embd.weight", 1024),
                    f32_tensor("blk.0.attn_norm.weight", 1024),
                    f32_tensor("blk.0.ffn_gate_exps.weight", 4096),
                ],
            )),
            Ok(shard(
                meta,
                vec![
                    f32_tensor("blk.1.attn_norm.weight", 1024),
                    f32_tensor("blk.1.ffn_gate_exps.weight", 4096),
                    // A *shared* expert runs for every token, so it is dense
                    // however much its name looks like a routed one.
                    f32_tensor("blk.1.ffn_gate_shexp.weight", 1024),
                ],
            )),
        ])
        .unwrap();

        assert_eq!(plan.shards, 2);
        assert_eq!(plan.architecture, "testmoe");
        assert_eq!(plan.moe_layers, 2);
        assert_eq!(plan.expert_bytes, 2 * 4096 * 4);
        // One layer's expert tensors over `n_expert`, not two layers' worth.
        assert_eq!(plan.bytes_per_expert, 4096 * 4 / 4);
        assert_eq!(plan.dense_bytes, 4 * 1024 * 4);
        assert_eq!(plan.total_bytes, plan.dense_bytes + plan.expert_bytes);
        assert_eq!(plan.expert_bytes_per_token(), (4096 * 4 / 4) * 2 * 2);
    }

    /// Metadata comes from the first shard, and later shards contribute
    /// tensors only — a shard that repeats the architecture key must not be
    /// able to change the answer, and a plan of one shard must not differ
    /// from a plan of the same shard followed by more.
    #[test]
    fn analyze_shards_takes_its_metadata_from_the_first_shard() {
        let plan = analyze_shards(vec![Ok(shard(
            &[("general.architecture", 0)],
            vec![f32_tensor("token_embd.weight", 1024)],
        ))])
        .unwrap();
        assert_eq!(plan.shards, 1);
        assert_eq!(plan.n_expert, 0);
        assert!(!plan.is_moe());
        assert_eq!(plan.dense_bytes, 1024 * 4);
    }

    /// A shard that will not parse — a truncated header, a dropped
    /// connection part-way through a remote fetch — must fail the plan
    /// rather than silently produce one that undercounts by a shard.
    #[test]
    fn analyze_shards_propagates_a_failed_shard() {
        let err = analyze_shards(vec![
            Ok(shard(
                &[("general.architecture", 0)],
                vec![f32_tensor("token_embd.weight", 1024)],
            )),
            Err(anyhow!("connection reset")),
        ])
        .unwrap_err();
        assert!(err.to_string().contains("connection reset"), "{err}");
    }

    #[test]
    fn analyze_shards_rejects_a_model_with_no_shards() {
        assert!(analyze_shards(Vec::new()).is_err());
    }

    /// The prompt `download` shows and the verdict `format_plan` prints are
    /// two readings of one predicate, and this is what keeps them from
    /// drifting: for every case, "the report says it will not work" and
    /// "`dense_fits_in` says no" must be the same answer. A model whose
    /// *experts* overflow is explicitly not that case — it streams, so it is
    /// slow rather than broken, and `download` must not stop to warn about
    /// the workload orangu is built for.
    #[test]
    fn dense_fits_in_agrees_with_every_printed_verdict() {
        let unworkable_moe = Plan {
            dense_bytes: 40 << 30,
            ..moe_plan()
        };
        let dense = Plan {
            shards: 1,
            architecture: "llama".into(),
            total_bytes: 30 << 30,
            dense_bytes: 30 << 30,
            expert_bytes: 0,
            device_bytes: 30 << 30,
            draft_bytes: 0,
            n_expert: 0,
            n_expert_used: 0,
            moe_layers: 0,
            bytes_per_expert: 0,
        };
        for (plan, ram) in [
            (&moe_plan(), 60 << 30),     // streams: runnable
            (&moe_plan(), 400 << 30),    // fits outright
            (&unworkable_moe, 20 << 30), // dense part short
            (&dense, 60 << 30),          // fits
            (&dense, 8 << 30),           // short
        ] {
            let report = format_plan(plan, ram, None);
            // Both over-RAM verdicts now say "runnable" — the model streams
            // rather than failing — so the marker is what the verdict warns
            // about, not a claim that it cannot run.
            let says_no = report.contains("comes\n           off disk every token")
                || report.contains("runnable but slow");
            assert_eq!(
                plan.dense_fits_in(ram),
                !says_no,
                "predicate and verdict disagree at {ram} bytes:\n{report}"
            );
        }
    }

    /// The bug this whole device path exists for.
    ///
    /// `Qwen3.8-27B-Q6_K` planned as "fits in RAM with 23.9 GiB to spare" and
    /// then, on being served, reported that its weights were 17.3 GiB larger
    /// than the card the server had just selected and would be paged in and
    /// out on every token. Both statements were true; the plan was simply
    /// answering a question about the wrong ceiling, and printing no `This
    /// box` line for a dense model meant there was nothing beside the verdict
    /// to notice that with.
    #[test]
    fn a_model_that_fits_ram_but_not_the_card_says_so() {
        let plan = Plan {
            shards: 1,
            architecture: "qwen35".into(),
            total_bytes: 21300 << 20,
            dense_bytes: 21000 << 20,
            expert_bytes: 0,
            device_bytes: 21000 << 20,
            draft_bytes: 300 << 20,
            n_expert: 0,
            n_expert_used: 0,
            moe_layers: 0,
            bytes_per_expert: 0,
        };
        let report = format_plan(&plan, 44 << 30, Some(("AMD Radeon RX 5500M", 3980 << 20)));

        // The RAM verdict is unchanged and still true.
        assert!(report.contains("fits in RAM"), "{report}");
        // What used to be missing entirely.
        assert!(report.contains("This box"), "{report}");
        assert!(report.contains("AMD Radeon RX 5500M"), "{report}");
        assert!(report.contains("Device"), "{report}");
        assert!(report.contains("too large for this GPU"), "{report}");
        // The wording changed when the server learned to spread a too-large
        // model across devices instead of paging it: it now says what will
        // happen rather than naming a knob the operator has to reach for.
        assert!(report.contains("spread the model"), "{report}");
        assert!(report.contains("run the remainder on the CPU"), "{report}");
        assert_eq!(plan.device_shortfall(3980 << 20), Some(17020 << 20));
    }

    /// A dense model used to print no `This box` line at all — that block sat
    /// under the mixture-of-experts branch, so the commonest kind of model got
    /// a verdict with nothing to check it against.
    #[test]
    fn a_dense_plan_states_the_machine_it_was_judged_against() {
        let mut plan = moe_plan();
        plan.moe_layers = 0;
        plan.n_expert = 0;
        assert!(!plan.is_moe());
        assert!(format_plan(&plan, 60 << 30, None).contains("This box"));
    }

    /// With no dedicated card there is no second ceiling, and the report must
    /// not invent one — nor name a GPU it was not given.
    #[test]
    fn no_device_means_no_device_verdict() {
        for plan in [
            moe_plan(),
            Plan {
                moe_layers: 0,
                n_expert: 0,
                ..moe_plan()
            },
        ] {
            let report = format_plan(&plan, 60 << 30, None);
            assert!(!report.contains("Device"), "{report}");
            assert!(!report.contains("VRAM"), "{report}");
        }
    }

    /// Shared experts are dense — every token runs them — but they have no
    /// GPU path, so they belong to `dense_bytes` and *not* to `device_bytes`.
    /// Conflating the two would have a plan tell a card to hold weights that
    /// never reach it, overstating an MoE model's device footprint by most of
    /// its shared-expert mass.
    #[test]
    fn shared_experts_are_dense_but_not_device_resident() {
        let meta = &[
            ("general.architecture", 0),
            ("testmoe.expert_count", 4),
            ("testmoe.expert_used_count", 2),
        ];
        let plan = analyze_shards(vec![Ok(shard(
            meta,
            vec![
                f32_tensor("token_embd.weight", 1024),
                f32_tensor("blk.0.attn_norm.weight", 1024),
                f32_tensor("blk.0.ffn_gate_exps.weight", 4096),
                f32_tensor("blk.0.ffn_gate_shexp.weight", 2048),
            ],
        ))])
        .unwrap();

        // Dense counts the shared expert; the device does not.
        assert_eq!(plan.dense_bytes, (1024 + 1024 + 2048) * 4);
        assert_eq!(plan.device_bytes, (1024 + 1024) * 4);
        assert_eq!(plan.expert_bytes, 4096 * 4);
    }

    /// A draft block is never uploaded either, so it must not be charged to
    /// the card any more than it is charged to RAM.
    #[test]
    fn the_draft_block_is_not_charged_to_the_device() {
        let plan = analyze_shards(vec![Ok(shard(
            &[
                ("general.architecture", 0),
                ("testmoe.block_count", 2),
                ("testmoe.nextn_predict_layers", 1),
            ],
            vec![
                f32_tensor("blk.0.attn_norm.weight", 1024),
                f32_tensor("blk.1.attn_norm.weight", 4096),
            ],
        ))])
        .unwrap();
        assert_eq!(plan.draft_bytes, 4096 * 4);
        assert_eq!(plan.dense_bytes, 1024 * 4);
        assert_eq!(plan.device_bytes, 1024 * 4);
    }

    /// Every figure scales to its own size, at both ends of the range a plan
    /// has to cover.
    ///
    /// A fixed unit is wrong twice over, and the report used to carry two of
    /// them. `{:.1} MiB` on the per-token line made a large mixture-of-experts
    /// model read `Per token 14473.1 MiB` — four digits nobody can weigh
    /// against the `GiB` figures three lines above. `{:.1} GiB` on the size
    /// lines rounded a 318 MiB embedding model down to `0.3 GiB on disk`,
    /// discarding most of the precision it had.
    ///
    /// Asserted as "the wrong unit does not appear" rather than by matching
    /// the exact rendering, so this keeps biting if the formatter's decimals
    /// or spacing change but its scaling does not.
    #[test]
    fn every_figure_is_printed_in_a_unit_that_suits_it() {
        // 20 MiB per expert x 8 used x 75 layers = 11.72 GiB per token.
        let report = format_plan(&moe_plan(), 60 << 30, None);
        let per_token = report
            .lines()
            .find(|line| line.starts_with("Per token"))
            .expect("a per-token line");
        assert!(
            per_token.contains("GiB"),
            "11.72 GiB per token printed in the wrong unit: {per_token}"
        );
        // The per-expert figure on the line above is genuinely MiB-scale and
        // must stay that way — scaling everything to GiB would be the same
        // mistake pointing the other way.
        let experts = report
            .lines()
            .find(|line| line.starts_with("Experts"))
            .expect("an experts line");
        assert!(
            experts.contains("20.00 MiB each"),
            "a 20 MiB expert printed in the wrong unit: {experts}"
        );

        // A model far below a gigabyte keeps its precision.
        let small = Plan {
            shards: 1,
            architecture: "gemma-embedding".into(),
            total_bytes: 318 << 20,
            dense_bytes: 318 << 20,
            expert_bytes: 0,
            device_bytes: 318 << 20,
            draft_bytes: 0,
            n_expert: 0,
            n_expert_used: 0,
            moe_layers: 0,
            bytes_per_expert: 0,
        };
        let report = format_plan(&small, 60 << 30, None);
        assert!(report.contains("318.00 MiB on disk"), "{report}");
        assert!(!report.contains("0.3 GiB"), "{report}");
    }
}
