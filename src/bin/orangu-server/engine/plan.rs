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

use anyhow::Result;

use crate::engine::loader;
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

/// The block index in a `blk.<i>.<...>` tensor name, or `None` for a tensor
/// that belongs to no block (`token_embd`, `output_norm`, `output`).
fn block_index(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
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
    let architecture = first
        .metadata
        .iter()
        .find(|(k, _)| k == "general.architecture")
        .and_then(|(_, v)| match v {
            orangu::gguf::GgufValue::String(s) => Some(s.clone()),
            _ => None,
        })
        .unwrap_or_else(|| "unknown".to_string());
    let meta_u64 = |suffix: &str| -> Option<u64> {
        let key = format!("{architecture}.{suffix}");
        first
            .metadata
            .iter()
            .find(|(k, _)| *k == key)
            .and_then(|(_, v)| v.as_u64())
    };
    let n_expert = meta_u64("expert_count").unwrap_or(0) as usize;
    let n_expert_used = meta_u64("expert_used_count").unwrap_or(0) as usize;

    let trunk_blocks = trunk_block_count(&first, &architecture);

    let shards = loader::shard_paths(path, &first)?;
    let mut total_bytes = 0u64;
    let mut draft_bytes = 0u64;
    let mut expert_bytes = 0u64;
    let mut moe_layers = std::collections::HashSet::new();
    let mut per_layer_expert_bytes = 0u64;
    let mut seen_first_moe_layer: Option<String> = None;

    for (index, shard) in shards.iter().enumerate() {
        // Shard 1's table is already parsed; re-opening it would be a second
        // read of the same few hundred kilobytes.
        let reopened;
        let gguf = if index == 0 {
            &first
        } else {
            reopened = GgufFile::open(shard)?;
            &reopened
        };
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
        shards: shards.len(),
        architecture,
        total_bytes,
        dense_bytes: total_bytes
            .saturating_sub(expert_bytes)
            .saturating_sub(draft_bytes),
        expert_bytes,
        draft_bytes,
        n_expert,
        n_expert_used,
        moe_layers: moe_layers.len(),
        bytes_per_expert,
    })
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// The plan as a report, with the machine's own memory beside it.
///
/// Deliberately states the *verdict* rather than only the numbers. The numbers
/// are what a reader would have to combine themselves to answer the question
/// they actually have, which is whether to press on.
pub fn format_plan(plan: &Plan, available_ram: u64, vram: Option<u64>) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Model      {} · {} shard{} · {:.1} GiB on disk\n",
        plan.architecture,
        plan.shards,
        if plan.shards == 1 { "" } else { "s" },
        gib(plan.total_bytes),
    ));

    if !plan.is_moe() {
        out.push_str(&format!(
            "Dense      {:.1} GiB — every byte is touched by every token\n",
            gib(plan.dense_bytes)
        ));
        out.push_str(&draft_line(plan));
        out.push_str(&verdict_dense(plan.dense_bytes, available_ram));
        return out;
    }

    out.push_str(&format!(
        "Dense      {:.1} GiB — attention, norms, embeddings, shared experts. Must be resident.\n",
        gib(plan.dense_bytes)
    ));
    out.push_str(&format!(
        "Experts    {:.1} GiB — {} per layer x {} layers, {:.1} MiB each. Can stream.\n",
        gib(plan.expert_bytes),
        plan.n_expert,
        plan.moe_layers,
        plan.bytes_per_expert as f64 / (1024.0 * 1024.0),
    ));
    out.push_str(&format!(
        "Per token  {:.1} MiB of experts ({} of {} per layer, {} layers)\n",
        plan.expert_bytes_per_token() as f64 / (1024.0 * 1024.0),
        plan.n_expert_used,
        plan.n_expert,
        plan.moe_layers,
    ));
    out.push_str(&draft_line(plan));
    out.push_str(&format!(
        "This box   {:.1} GiB RAM available{}\n",
        gib(available_ram),
        vram.map_or(String::new(), |v| format!(", {:.1} GiB VRAM", gib(v))),
    ));
    out.push_str(&verdict_moe(plan, available_ram));
    out
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
        "Draft head {:.1} GiB — a multi-token-prediction block this engine does not run. Never loaded.\n",
        gib(plan.draft_bytes),
    )
}

fn verdict_dense(total: u64, available_ram: u64) -> String {
    if total <= available_ram {
        format!(
            "Verdict    fits in RAM with {:.1} GiB to spare\n",
            gib(available_ram - total)
        )
    } else {
        format!(
            "Verdict    does NOT fit: {:.1} GiB short, and a dense model has nothing to stream\n",
            gib(total - available_ram)
        )
    }
}

fn verdict_moe(plan: &Plan, available_ram: u64) -> String {
    if plan.dense_bytes > available_ram {
        return format!(
            "Verdict    will NOT work: the dense part alone is {:.1} GiB short of RAM, and it is \
             touched by every token\n",
            gib(plan.dense_bytes - available_ram)
        );
    }
    let spare = available_ram - plan.dense_bytes;
    if plan.expert_bytes <= spare {
        format!(
            "Verdict    fits entirely in RAM ({:.1} GiB to spare); nothing needs to stream\n",
            gib(spare - plan.expert_bytes)
        )
    } else {
        format!(
            "Verdict    runnable by streaming: dense fits, {:.1} GiB of experts do not and will \
             come off disk\n           at {:.1} MiB per token, the storage under the model sets \
             the speed\n",
            gib(plan.expert_bytes - spare),
            plan.expert_bytes_per_token() as f64 / (1024.0 * 1024.0),
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
        assert!(report.contains("will NOT work"), "{report}");
        assert!(report.contains("20.0 GiB short"), "{report}");
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
        assert!(report.contains("Draft head 5.0 GiB"), "{report}");
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
            draft_bytes: 0,
            n_expert: 0,
            n_expert_used: 0,
            moe_layers: 0,
            bytes_per_expert: 0,
        };
        assert!(!plan.is_moe());
        let short = format_plan(&plan, 8 << 30, None);
        assert!(short.contains("does NOT fit"), "{short}");
        assert!(short.contains("nothing to stream"), "{short}");
        assert!(format_plan(&plan, 60 << 30, None).contains("fits in RAM"));
    }
}
