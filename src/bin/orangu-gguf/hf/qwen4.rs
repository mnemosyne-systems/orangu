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

//! `qwen4exp` — Qwen 4 (Qwen3.8-Flash-Next): the Qwen 3.5 mixture's gated
//! delta rule and gated full attention, with three additions.
//!
//! - **Hyper-connections in place of every layer norm.** The residual is
//!   `hyper_connection.count` streams wide; before each sub-layer a mixer
//!   (a norm over every stream, a low-rank down and up projection, and an
//!   inject weight) reads them, and one more mixer at the end replaces the
//!   output norm. Their norms are stored as offsets from one.
//! - **Sparse attention on the full layers.** An indexer picks which keys
//!   each query attends to; its query and key projections are one tensor in
//!   the checkpoint, split here, and its norms are offsets from one.
//! - **N-gram hash embeddings (PLE) on one layer.** A table indexed by a
//!   hash of the last few tokens, stored in shards and written as one
//!   tensor a shard at a time — it is far larger than memory — beside the
//!   hash's integer constants, written exactly as metadata.
//!
//! The vision tower belongs to a projector file and the next-token
//! prediction head to a drafter; neither is converted here.

use anyhow::{Context, Result, anyhow, bail};
use orangu::gguf::GgufValue;
use orangu::safetensors::SafeTensorsSet;
use serde_json::Value;

use super::qwen35::{ATTENTION, Heads, LINEAR, SHARED};
use super::{
    Conversion, Mapped, Source, Transform, check_all_read, find, find_usize, require_usize,
};

/// Checkpoint prefixes of the parts other files carry.
const ELSEWHERE: [&str; 3] = ["model.visual.", "mtp.", "model.mtp."];

/// A hyper-connection mixer's checkpoint suffixes and GGUF suffixes.
const MIXER: [(&str, &str); 4] = [
    ("hc_norm.weight", "norm.weight"),
    ("input_mix_weight_down.weight", "down.weight"),
    ("input_mix_weight_up.weight", "up.weight"),
    ("block_inject_weight.weight", "inject.weight"),
];

/// The PLE layer's own tensors.
const PLE: [(&str, &str); 6] = [
    ("ple.key_proj.weight", "ple_key.weight"),
    ("ple.value_proj.weight", "ple_value.weight"),
    ("ple.norm_key.weight", "ple_norm_key.weight"),
    ("ple.norm_query.weight", "ple_norm_query.weight"),
    ("ple.norm_conv.weight", "ple_norm_conv.weight"),
    ("ple.conv1d.weight", "ple_conv1d.weight"),
];

/// A checkpoint's int64 tensor, exactly.
fn integers(weights: &SafeTensorsSet, name: &str) -> Result<Vec<i64>> {
    match weights.info(name) {
        Some((_, "I64")) => {}
        Some((_, other)) => bail!("{name} is {other}, expected I64"),
        None => bail!("the checkpoint has no {name}"),
    }
    Ok(weights
        .bytes(name)?
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| i64::from_le_bytes(*b))
        .collect())
}

pub fn plan(hf: &str, config: &Value, weights: &SafeTensorsSet) -> Result<Option<Conversion>> {
    if !matches!(
        hf,
        "Qwen4ExpForConditionalGeneration" | "Qwen4ExpForCausalLM"
    ) {
        return Ok(None);
    }
    let arch = "qwen4exp";
    let outer = config;
    let config = config
        .get("text_config")
        .filter(|v| v.is_object())
        .unwrap_or(config);
    let need = |k: &str| require_usize(config, &[k]);
    let layers = need("num_hidden_layers")?;
    let embed = need("hidden_size")?;
    let heads = need("num_attention_heads")?;
    let kv_heads = find_usize(config, &["num_key_value_heads"]).unwrap_or(heads);
    let vocab_size = need("vocab_size")?;
    let head_dim = find_usize(config, &["head_dim"]).unwrap_or(embed / heads);
    let interval = find_usize(config, &["full_attention_interval"]).unwrap_or(4);
    let delta = Heads {
        key: need("linear_num_key_heads")?,
        value: need("linear_num_value_heads")?,
        key_dim: need("linear_key_head_dim")?,
        value_dim: need("linear_value_head_dim")?,
    };
    if delta.key == 0 || !delta.value.is_multiple_of(delta.key) {
        bail!(
            "{} value heads do not divide among {} key heads",
            delta.value,
            delta.key
        );
    }
    let full = |layer: usize| (layer + 1).is_multiple_of(interval);
    if let Some(types) = config["layer_types"].as_array() {
        for (layer, kind) in types.iter().enumerate() {
            if (kind.as_str() == Some("full_attention")) != full(layer) {
                bail!("layer {layer}'s type does not follow full_attention_interval {interval}");
            }
        }
    }
    let indexer_heads = need("indexer_n_heads")?;
    let indexer_dim = need("indexer_head_dim")?;

    let rope = config
        .get("rope_parameters")
        .filter(|v| v.is_object())
        .unwrap_or(config);
    let rotated = rope["partial_rotary_factor"].as_f64().unwrap_or(0.25);
    let mut sections: Vec<u32> = rope["mrope_section"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(Value::as_u64)
                .map(|v| v as u32)
                .collect()
        })
        .unwrap_or_else(|| vec![11, 11, 10]);
    sections.resize(4, 0);
    let theta = find(rope, &["rope_theta"])
        .and_then(Value::as_f64)
        .ok_or_else(|| anyhow!("config.json has no rope_theta"))?;
    let eps = find(config, &["rms_norm_eps"])
        .and_then(Value::as_f64)
        .unwrap_or(1e-6);

    let key = |k: &str| format!("{arch}.{k}");
    let u = |v: usize| GgufValue::U32(v as u32);
    let mut metadata: Vec<(String, GgufValue)> = vec![
        (key("block_count"), u(layers)),
        (key("context_length"), u(need("max_position_embeddings")?)),
        (key("embedding_length"), u(embed)),
        (key("attention.head_count"), u(heads)),
        (key("attention.head_count_kv"), u(kv_heads)),
        (
            key("rope.dimension_sections"),
            GgufValue::Array(
                sections
                    .into_iter()
                    .map(|s| GgufValue::I32(s as i32))
                    .collect(),
            ),
        ),
        (key("rope.freq_base"), GgufValue::F32(theta as f32)),
        (
            key("attention.layer_norm_rms_epsilon"),
            GgufValue::F32(eps as f32),
        ),
        (key("expert_count"), u(need("num_experts")?)),
        (key("expert_used_count"), u(need("num_experts_per_tok")?)),
        (key("attention.key_length"), u(head_dim)),
        (key("attention.value_length"), u(head_dim)),
        (
            key("expert_feed_forward_length"),
            u(need("moe_intermediate_size")?),
        ),
        (
            key("expert_shared_feed_forward_length"),
            u(need("shared_expert_intermediate_size")?),
        ),
        (key("ssm.conv_kernel"), u(need("linear_conv_kernel_dim")?)),
        (key("ssm.state_size"), u(delta.key_dim)),
        (key("ssm.group_count"), u(delta.key)),
        (key("ssm.time_step_rank"), u(delta.value)),
        (key("ssm.inner_size"), u(delta.value * delta.value_dim)),
        (key("full_attention_interval"), u(interval)),
        (
            key("rope.dimension_count"),
            u((head_dim as f64 * rotated) as usize),
        ),
        (key("hyper_connection.count"), u(need("hc_count")?)),
        (key("hyper_connection.low_rank"), u(need("hc_lowrank")?)),
        (key("attention.indexer.head_count"), u(indexer_heads)),
        (key("attention.indexer.key_length"), u(indexer_dim)),
        (key("attention.indexer.top_k"), u(need("indexer_budget")?)),
        (
            key("attention.compress_ratios"),
            GgufValue::Array(
                (0..layers)
                    .map(|l| {
                        GgufValue::I32(if full(l) {
                            need("indexer_compress_ratio").unwrap_or(0) as i32
                        } else {
                            0
                        })
                    })
                    .collect(),
            ),
        ),
    ];

    let root = ["model.language_model", "model"]
        .into_iter()
        .find(|r| weights.has(&format!("{r}.embed_tokens.weight")))
        .ok_or_else(|| anyhow!("the checkpoint has no embed_tokens.weight"))?;
    let at = |layer: usize, s: &str| format!("{root}.layers.{layer}.{s}");

    // The PLE layers, 1-based in the configuration.
    let ple_layers: Vec<usize> = config["ple_layer_ids"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_u64)
        .map(|l| l as usize - 1)
        .collect();
    let mut skipped: Vec<String> = weights
        .names()
        .into_iter()
        .filter(|n| ELSEWHERE.iter().any(|p| n.starts_with(p)))
        .map(str::to_string)
        .collect();
    let mut tensors = vec![Mapped::new(
        "token_embd.weight",
        Source::tensor(format!("{root}.embed_tokens.weight")),
    )];
    let mut ple_table: Option<Mapped> = None;
    if !ple_layers.is_empty() {
        if ple_layers.len() != 1 {
            bail!("more than one PLE layer is not converted");
        }
        let layer = ple_layers[0];
        let parts = need("split_ngram_parts")?;
        let shards: Vec<String> = (0..parts)
            .map(|i| {
                at(
                    layer,
                    &format!("ple.ple_embedding.ngram_embedding.shard_{i}.weight"),
                )
            })
            .collect();
        let row_dim = weights
            .info(&shards[0])
            .map(|(shape, _)| shape[shape.len() - 1])
            .ok_or_else(|| anyhow!("the checkpoint has no {}", shards[0]))?;
        let mut constant = |s: &str| -> Result<GgufValue> {
            let name = at(layer, &format!("ple.ple_embedding.{s}"));
            skipped.push(name.clone());
            Ok(GgufValue::Array(
                integers(weights, &name)?
                    .into_iter()
                    .map(|v| GgufValue::U64(v as u64))
                    .collect(),
            ))
        };
        let eos = match &config["eos_token_id"] {
            Value::Array(ids) => ids.last().and_then(Value::as_u64),
            other => other.as_u64(),
        }
        .ok_or_else(|| anyhow!("PLE needs eos_token_id: its hash restarts on it"))?;
        metadata.extend([
            (
                key("ple.layers"),
                GgufValue::Array(vec![GgufValue::I32(layer as i32)]),
            ),
            (key("ple.ngram_size"), u(need("ngram_size")?)),
            (key("ple.heads_per_ngram"), u(need("heads_per_ngram")?)),
            (key("ple.conv_kernel"), u(need("ple_conv_kernel_size")?)),
            (key("ple.eos_token_id"), u(eos as usize)),
        ]);
        let image = config["image_token_id"]
            .as_u64()
            .or(outer["image_token_id"].as_u64());
        if let Some(image) = image {
            metadata.push((key("ple.image_token_id"), u(image as usize)));
        }
        metadata.extend([
            (key("embedding_length_per_layer_input"), u(row_dim)),
            (key("ple.layer_multipliers"), constant("layer_multipliers")?),
            (key("ple.head_offsets"), constant("ngram_heads_offsets")?),
            (
                key("ple.head_vocab_sizes"),
                constant("ngram_heads_vocab_sizes")?,
            ),
        ]);
        ple_table = Some(Mapped::new(
            "per_layer_token_embd.weight",
            Source::Rows(shards),
        ));
    }

    let mixer = |tensors: &mut Vec<Mapped>, from: &str, to: &str| {
        for (suffix, name) in MIXER {
            let mut mapped = Mapped::new(
                format!("{to}_{name}"),
                Source::tensor(format!("{from}.{suffix}")),
            );
            if name == "norm.weight" {
                mapped = mapped.with(Transform::AddOne);
            }
            tensors.push(mapped);
        }
    };
    for layer in 0..layers {
        let blk = |s: &str| format!("blk.{layer}.{s}");
        mixer(
            &mut tensors,
            &at(layer, "attn_hyper_connection"),
            &blk("hc_attn"),
        );
        mixer(
            &mut tensors,
            &at(layer, "mlp_hyper_connection"),
            &blk("hc_ffn"),
        );
        let own: &[(&str, &str)] = if full(layer) { &ATTENTION } else { &LINEAR };
        for &(suffix, name) in own.iter().chain(&SHARED) {
            let source = at(layer, suffix);
            if !weights.has(&source) {
                continue;
            }
            let mut mapped = Mapped::new(blk(name), Source::tensor(source));
            if name.ends_with("norm.weight") && name != "ssm_norm.weight" {
                mapped = mapped.with(Transform::AddOne);
            }
            if name == "ffn_gate_inp_shexp.weight" {
                mapped = mapped.with(Transform::Squeeze);
            }
            for transform in delta.transforms(name) {
                mapped = mapped.with(transform);
            }
            tensors.push(mapped);
        }
        if full(layer) {
            // One projection feeds the indexer's queries and keys.
            let qk = at(layer, "self_attn.indexer.index_qk_proj.weight");
            for (name, offset, count) in [
                ("indexer.q_proj.weight", 0, indexer_heads * indexer_dim),
                (
                    "indexer.k_proj.weight",
                    indexer_heads * indexer_dim,
                    indexer_dim,
                ),
            ] {
                tensors.push(
                    Mapped::new(
                        blk(name),
                        Source::HeadRows {
                            name: qk.clone(),
                            heads: 1,
                            offset,
                            count,
                            transpose: false,
                        },
                    )
                    .with(Transform::Squeeze),
                );
            }
            for (from, to) in [
                (
                    "self_attn.indexer.q_layernorm.weight",
                    "indexer.q_norm.weight",
                ),
                (
                    "self_attn.indexer.k_layernorm.weight",
                    "indexer.k_norm.weight",
                ),
            ] {
                tensors.push(
                    Mapped::new(blk(to), Source::tensor(at(layer, from))).with(Transform::AddOne),
                );
            }
        }
        if ple_layers.contains(&layer) {
            for (suffix, name) in PLE {
                let mut mapped = Mapped::new(blk(name), Source::tensor(at(layer, suffix)));
                if name.ends_with("norm_key.weight")
                    || name.ends_with("norm_query.weight")
                    || name.ends_with("norm_conv.weight")
                {
                    mapped = mapped.with(Transform::AddOne);
                }
                if name == "ple_conv1d.weight" {
                    mapped = mapped.with(Transform::Squeeze);
                }
                tensors.push(mapped);
            }
        }
        tensors.extend(super::llama::experts(
            config,
            weights,
            &format!("{root}.layers"),
            layer,
        )?);
    }
    mixer(
        &mut tensors,
        &format!("{root}.hyper_connection_mixer"),
        "output_hc",
    );
    // The final mixer reads the streams and writes one: it has no inject.
    tensors.retain(|m| m.name != "output_hc_inject.weight");
    if let Some(table) = ple_table {
        tensors.push(table);
    }
    let tied = find(config, &["tie_word_embeddings"])
        .or(find(outer, &["tie_word_embeddings"]))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if weights.has("lm_head.weight") {
        if tied {
            skipped.push("lm_head.weight".into());
        } else {
            tensors.push(Mapped::new(
                "output.weight",
                Source::tensor("lm_head.weight"),
            ));
        }
    }
    check_all_read(weights, &tensors, &skipped, arch).context("mapping the Qwen 4 checkpoint")?;

    Ok(Some(Conversion {
        architecture: arch,
        metadata,
        tensors,
        layers,
        vocab_size,
        tokenizer: super::tokenizer::Style::ByteLevel,
        tokenizer_overrides: Vec::new(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::path::Path;

    #[test]
    fn hash_constants_are_read_exactly() {
        let dir = tempfile::tempdir().unwrap();
        let values: [i64; 2] = [23_703_573_157_769, -3];
        let header = r#"{"c":{"dtype":"I64","shape":[2],"data_offsets":[0,16]},"f":{"dtype":"F32","shape":[1],"data_offsets":[16,20]}}"#;
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header.as_bytes());
        bytes.extend(values.iter().flat_map(|v| v.to_le_bytes()));
        bytes.extend(1.0f32.to_le_bytes());
        std::fs::write(dir.path().join("m.safetensors"), bytes).unwrap();
        let weights = SafeTensorsSet::open_dir(dir.path()).unwrap();
        assert_eq!(integers(&weights, "c").unwrap(), values);
        assert!(integers(&weights, "f").is_err() && integers(&weights, "x").is_err());
    }

    /// The conversion planned from a checkpoint's headers alone
    /// (`ORANGU_QWEN4_STUB`: its JSON files and every shard's header, the
    /// data left sparse) against a published file's tensor table and
    /// `qwen4exp.*` metadata (`ORANGU_QWEN4_GGUF`: a directory of its
    /// shards): every tensor, its shape, and every value.
    #[test]
    #[ignore = "needs ORANGU_QWEN4_STUB and ORANGU_QWEN4_GGUF"]
    fn the_planned_table_is_the_published_one() {
        let stub = std::path::PathBuf::from(std::env::var("ORANGU_QWEN4_STUB").unwrap());
        let shards = std::path::PathBuf::from(std::env::var("ORANGU_QWEN4_GGUF").unwrap());
        let weights = SafeTensorsSet::open_dir(&stub).unwrap();
        let config = super::super::read_config(&stub).unwrap();
        let conversion = super::super::plan(&config, &weights).unwrap();

        let mut ours: BTreeMap<String, Vec<u64>> = BTreeMap::new();
        for m in &conversion.tensors {
            let shape = m.shape(&weights).unwrap();
            ours.insert(
                m.name.clone(),
                shape.iter().rev().map(|&d| d as u64).collect(),
            );
        }
        let mut published: BTreeMap<String, Vec<u64>> = BTreeMap::new();
        let mut meta: BTreeMap<String, GgufValue> = BTreeMap::new();
        let mut files: Vec<_> = std::fs::read_dir(&shards)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|e| e == "gguf"))
            .collect();
        files.sort();
        for file in &files {
            let gguf = orangu::gguf::GgufFile::open(Path::new(file)).unwrap();
            for t in &gguf.tensors {
                published.insert(t.name.clone(), t.dims.clone());
            }
            for (k, v) in &gguf.metadata {
                if k.starts_with("qwen4exp.") {
                    meta.insert(k.clone(), v.clone());
                }
            }
        }
        let mut problems = Vec::new();
        for (name, dims) in &published {
            match ours.get(name) {
                None => problems.push(format!("missing {name}")),
                Some(d) if d != dims => problems.push(format!("{name}: {d:?} != {dims:?}")),
                _ => {}
            }
        }
        for name in ours.keys().filter(|n| !published.contains_key(*n)) {
            problems.push(format!("extra {name}"));
        }
        let planned: BTreeMap<&str, &GgufValue> = conversion
            .metadata
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect();
        for (k, v) in &meta {
            // The hash constants are data, which a stub does not carry: only
            // their type and count are compared.
            let data = k.ends_with("ple.layer_multipliers")
                || k.ends_with("ple.head_offsets")
                || k.ends_with("ple.head_vocab_sizes");
            let shape = |v: &GgufValue| match v {
                GgufValue::Array(a) => {
                    format!("{} x {:?}", a.len(), a.first().map(std::mem::discriminant))
                }
                other => format!("{other:?}"),
            };
            if data {
                if planned.get(k.as_str()).map(|p| shape(p)) != Some(shape(v)) {
                    problems.push(format!("metadata {k}: not {}", shape(v)));
                }
                continue;
            }
            match planned.get(k.as_str()) {
                None => problems.push(format!("metadata {k} missing")),
                Some(p) if format!("{p:?}") != format!("{v:?}") => {
                    problems.push(format!("metadata {k}: {p:?} != {v:?}"))
                }
                _ => {}
            }
        }
        for k in planned.keys().filter(|k| !meta.contains_key(**k)) {
            problems.push(format!("metadata {k} extra"));
        }
        println!(
            "{} tensors planned, {} published, {} metadata keys",
            ours.len(),
            published.len(),
            meta.len()
        );
        for p in &problems {
            println!("  {p}");
        }
        assert!(problems.is_empty(), "{} differences", problems.len());
    }
}
