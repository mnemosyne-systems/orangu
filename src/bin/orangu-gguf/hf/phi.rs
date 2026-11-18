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

//! `phi3` — Phi-3 and Phi-4-mini.
//!
//! The block stores Q, K and V as one projection and the gate and up
//! projections as one, and both go into the file fused as they are. Rotary
//! positions cover only part of each head (`partial_rotary_factor`), and a
//! long-context checkpoint stretches them with per-frequency factors
//! (`longrope`): those factors are not checkpoint tensors but lists in
//! `config.json`, written as the two `rope_factors_*` tensors, with the
//! attention scale the stretch implies as `rope.scaling.attn_factor`.

use anyhow::{Result, bail};
use orangu::gguf::GgufValue;
use orangu::safetensors::SafeTensorsSet;
use serde_json::Value;

use super::{Conversion, Mapped, Source, check_all_read, find, find_usize, require_usize};

const BLOCK: [(&str, &str); 6] = [
    ("input_layernorm.weight", "attn_norm.weight"),
    ("post_attention_layernorm.weight", "ffn_norm.weight"),
    ("self_attn.qkv_proj.weight", "attn_qkv.weight"),
    ("self_attn.o_proj.weight", "attn_output.weight"),
    ("mlp.gate_up_proj.weight", "ffn_up.weight"),
    ("mlp.down_proj.weight", "ffn_down.weight"),
];

pub fn plan(hf: &str, config: &Value, weights: &SafeTensorsSet) -> Result<Option<Conversion>> {
    if hf != "Phi3ForCausalLM" {
        return Ok(None);
    }
    let arch = "phi3";
    let layers = require_usize(config, &["num_hidden_layers"])?;
    let embed = require_usize(config, &["hidden_size"])?;
    let heads = require_usize(config, &["num_attention_heads"])?;
    let kv_heads = find_usize(config, &["num_key_value_heads"]).unwrap_or(heads);
    let vocab_size = require_usize(config, &["vocab_size"])?;
    let context = require_usize(config, &["max_position_embeddings", "n_positions"])?;
    let rope = config
        .get("rope_parameters")
        .filter(|v| v.is_object())
        .unwrap_or(config);
    let scaling = find(rope, &["rope_scaling"])
        .cloned()
        .unwrap_or(Value::Null);
    let original = find_usize(rope, &["original_max_position_embeddings"])
        .or_else(|| {
            scaling["original_max_position_embeddings"]
                .as_u64()
                .map(|v| v as usize)
        })
        .unwrap_or(context);
    let partial = find(rope, &["partial_rotary_factor"])
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    let rope_dims = (partial * embed as f64) as usize / heads;

    let key = |k: &str| format!("{arch}.{k}");
    let mut metadata: Vec<(String, GgufValue)> = vec![
        (key("context_length"), GgufValue::U32(context as u32)),
        (
            key("rope.scaling.original_context_length"),
            GgufValue::U32(original as u32),
        ),
        (key("embedding_length"), GgufValue::U32(embed as u32)),
        (
            key("feed_forward_length"),
            GgufValue::U32(require_usize(config, &["intermediate_size"])? as u32),
        ),
        (key("block_count"), GgufValue::U32(layers as u32)),
        (key("attention.head_count"), GgufValue::U32(heads as u32)),
        (
            key("attention.head_count_kv"),
            GgufValue::U32(kv_heads as u32),
        ),
        (
            key("attention.layer_norm_rms_epsilon"),
            GgufValue::F32(
                find(config, &["rms_norm_eps"])
                    .and_then(Value::as_f64)
                    .unwrap_or(1e-5) as f32,
            ),
        ),
        (
            key("rope.dimension_count"),
            GgufValue::U32(rope_dims as u32),
        ),
        (
            key("rope.freq_base"),
            GgufValue::F32(
                find(rope, &["rope_theta"])
                    .and_then(Value::as_f64)
                    .unwrap_or(10000.0) as f32,
            ),
        ),
        (
            key("attention.sliding_window"),
            GgufValue::U32(find_usize(config, &["sliding_window"]).unwrap_or(0) as u32),
        ),
    ];

    let mut tensors = Vec::new();
    let factors = |name: &str| -> Option<Vec<f32>> {
        scaling[name].as_array().map(|a| {
            a.iter()
                .filter_map(Value::as_f64)
                .map(|v| v as f32)
                .collect()
        })
    };
    if let (Some(long), Some(short)) = (factors("long_factor"), factors("short_factor")) {
        if long.len() != rope_dims / 2 || short.len() != rope_dims / 2 {
            bail!(
                "longrope factors must have {} entries each, not {} and {}",
                rope_dims / 2,
                long.len(),
                short.len()
            );
        }
        let kind = scaling["rope_type"]
            .as_str()
            .or_else(|| scaling["type"].as_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let stretch = context as f64 / original as f64;
        let attn_factor = match kind.as_str() {
            "su" | "longrope" if stretch > 1.0 => {
                (1.0 + stretch.ln() / (original as f64).ln()).sqrt()
            }
            "yarn" if stretch > 1.0 => 0.1 * stretch.ln() + 1.0,
            "su" | "longrope" | "yarn" => 1.0,
            other => bail!("rope scaling {other:?} is not converted"),
        };
        metadata.insert(
            0,
            (
                key("rope.scaling.attn_factor"),
                GgufValue::F32(attn_factor as f32),
            ),
        );
        let n = long.len();
        tensors.push(Mapped::new(
            "rope_factors_long.weight",
            Source::Values {
                values: long,
                shape: vec![n],
            },
        ));
        tensors.push(Mapped::new(
            "rope_factors_short.weight",
            Source::Values {
                values: short,
                shape: vec![n],
            },
        ));
    }

    tensors.push(Mapped::new(
        "token_embd.weight",
        Source::tensor("model.embed_tokens.weight"),
    ));
    for layer in 0..layers {
        for (suffix, name) in BLOCK {
            tensors.push(Mapped::new(
                format!("blk.{layer}.{name}"),
                Source::tensor(format!("model.layers.{layer}.{suffix}")),
            ));
        }
    }
    tensors.push(Mapped::new(
        "output_norm.weight",
        Source::tensor("model.norm.weight"),
    ));
    let mut skipped = Vec::new();
    if weights.has("lm_head.weight") {
        if config["tie_word_embeddings"].as_bool().unwrap_or(false) {
            skipped.push("lm_head.weight".to_string());
        } else {
            tensors.push(Mapped::new(
                "output.weight",
                Source::tensor("lm_head.weight"),
            ));
        }
    }
    check_all_read(weights, &tensors, &skipped, arch)?;

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
