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

//! `qwen35` and `qwen35moe` — the Qwen 3.5 hybrids, dense and routed.
//!
//! What sets the architecture apart, and therefore the table:
//!
//! - **Two kinds of layer.** Three of every four layers are linear
//!   attention (a gated delta rule over a short causal convolution); every
//!   `full_attention_interval`-th layer is gated full attention, whose
//!   query projection carries the output gate beside each head.
//! - **Norms stored as offsets from one.** Every RMSNorm weight except the
//!   delta rule's own output norm is written with one added.
//! - **The decay is stored as its logarithm** (`A_log`) and written as
//!   `-exp(A_log)`; the convolution kernel loses its unit dimension.
//! - **Value heads tiled.** With more value heads than key heads, the
//!   checkpoint keeps each key head's value heads in a row; the file wants
//!   them tiled, so every tensor indexed by value head is regrouped
//!   ([`Transform::TileHeads`]).
//! - **Multi-axis rotation** in three sections, written as four.
//!
//! The vision tower belongs to a projector file and the multi-token
//! prediction head to a drafter; neither is converted here.

use anyhow::{Result, anyhow, bail};
use orangu::gguf::GgufValue;
use orangu::safetensors::SafeTensorsSet;
use serde_json::Value;

use super::{
    Conversion, Mapped, Source, Transform, check_all_read, find, find_usize, require_usize,
};

/// The GGUF architecture for a checkpoint's `architectures[0]`.
fn architecture(hf: &str) -> Option<&'static str> {
    Some(match hf {
        "Qwen3_5ForConditionalGeneration" | "Qwen3_5ForCausalLM" => "qwen35",
        "Qwen3_5MoeForConditionalGeneration" | "Qwen3_5MoeForCausalLM" => "qwen35moe",
        _ => return None,
    })
}

/// Checkpoint prefixes of the parts other files carry.
const ELSEWHERE: [&str; 3] = ["model.visual.", "mtp.", "model.mtp."];

/// Per-block suffixes shared by both kinds of layer.
const COMMON: [(&str, &str); 5] = [
    ("input_layernorm.weight", "attn_norm.weight"),
    (
        "post_attention_layernorm.weight",
        "post_attention_norm.weight",
    ),
    ("mlp.gate_proj.weight", "ffn_gate.weight"),
    ("mlp.up_proj.weight", "ffn_up.weight"),
    ("mlp.down_proj.weight", "ffn_down.weight"),
];

/// A full-attention layer's suffixes.
pub(super) const ATTENTION: [(&str, &str); 6] = [
    ("self_attn.q_proj.weight", "attn_q.weight"),
    ("self_attn.k_proj.weight", "attn_k.weight"),
    ("self_attn.v_proj.weight", "attn_v.weight"),
    ("self_attn.o_proj.weight", "attn_output.weight"),
    ("self_attn.q_norm.weight", "attn_q_norm.weight"),
    ("self_attn.k_norm.weight", "attn_k_norm.weight"),
];

/// A linear-attention layer's suffixes.
pub(super) const LINEAR: [(&str, &str); 9] = [
    ("linear_attn.in_proj_qkv.weight", "attn_qkv.weight"),
    ("linear_attn.in_proj_z.weight", "attn_gate.weight"),
    ("linear_attn.in_proj_b.weight", "ssm_beta.weight"),
    ("linear_attn.in_proj_a.weight", "ssm_alpha.weight"),
    ("linear_attn.conv1d.weight", "ssm_conv1d.weight"),
    ("linear_attn.A_log", "ssm_a"),
    ("linear_attn.dt_bias", "ssm_dt.bias"),
    ("linear_attn.norm.weight", "ssm_norm.weight"),
    ("linear_attn.out_proj.weight", "ssm_out.weight"),
];

/// A routed layer's shared expert and its gate.
pub(super) const SHARED: [(&str, &str); 4] = [
    (
        "mlp.shared_expert.gate_proj.weight",
        "ffn_gate_shexp.weight",
    ),
    ("mlp.shared_expert.up_proj.weight", "ffn_up_shexp.weight"),
    (
        "mlp.shared_expert.down_proj.weight",
        "ffn_down_shexp.weight",
    ),
    ("mlp.shared_expert_gate.weight", "ffn_gate_inp_shexp.weight"),
];

/// The delta rule's head geometry.
#[derive(Clone, Copy)]
pub(super) struct Heads {
    pub key: usize,
    pub value: usize,
    pub key_dim: usize,
    pub value_dim: usize,
}

impl Heads {
    /// How a tensor indexed by value head is regrouped: `head_dim` rows (or
    /// columns) a head after `skip`; nothing when the counts are equal.
    fn tile(&self, skip: usize, head_dim: usize, columns: bool) -> Transform {
        if self.value == self.key {
            return Transform::None;
        }
        Transform::TileHeads {
            skip,
            groups: self.key,
            per_group: self.value / self.key,
            head_dim,
            columns,
        }
    }

    /// The transforms of a linear-attention tensor, by GGUF name.
    pub fn transforms(&self, name: &str) -> Vec<Transform> {
        let qk = 2 * self.key * self.key_dim;
        match name {
            "attn_qkv.weight" => vec![self.tile(qk, self.value_dim, false)],
            "attn_gate.weight" => vec![self.tile(0, self.value_dim, false)],
            "ssm_beta.weight" | "ssm_alpha.weight" | "ssm_dt.bias" => {
                vec![self.tile(0, 1, false)]
            }
            "ssm_a" => vec![Transform::NegExp, self.tile(0, 1, false)],
            "ssm_conv1d.weight" => vec![Transform::Squeeze, self.tile(qk, self.value_dim, false)],
            "ssm_out.weight" => vec![self.tile(0, self.value_dim, true)],
            _ => Vec::new(),
        }
    }
}

pub fn plan(hf: &str, config: &Value, weights: &SafeTensorsSet) -> Result<Option<Conversion>> {
    let Some(arch) = architecture(hf) else {
        return Ok(None);
    };
    let layers = require_usize(config, &["num_hidden_layers"])?;
    let embed = require_usize(config, &["hidden_size"])?;
    let heads = require_usize(config, &["num_attention_heads"])?;
    let kv_heads = find_usize(config, &["num_key_value_heads"]).unwrap_or(heads);
    let vocab_size = require_usize(config, &["vocab_size"])?;
    let head_dim = find_usize(config, &["head_dim"]).unwrap_or(embed / heads);
    let interval = find_usize(config, &["full_attention_interval"]).unwrap_or(4);
    let delta = Heads {
        key: require_usize(config, &["linear_num_key_heads"])?,
        value: require_usize(config, &["linear_num_value_heads"])?,
        key_dim: require_usize(config, &["linear_key_head_dim"])?,
        value_dim: require_usize(config, &["linear_value_head_dim"])?,
    };
    if delta.key == 0 || !delta.value.is_multiple_of(delta.key) {
        bail!(
            "{} value heads do not divide among {} key heads",
            delta.value,
            delta.key
        );
    }
    // The file says which layers are full attention by an interval only.
    if let Some(types) = config["layer_types"].as_array() {
        for (layer, kind) in types.iter().enumerate() {
            let full = kind.as_str() == Some("full_attention");
            if full != ((layer + 1) % interval == 0) {
                bail!("layer {layer}'s type does not follow full_attention_interval {interval}");
            }
        }
    }

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

    let key = |k: &str| format!("{arch}.{k}");
    let mut metadata: Vec<(String, GgufValue)> = vec![
        (key("block_count"), GgufValue::U32(layers as u32)),
        (
            key("context_length"),
            GgufValue::U32(require_usize(config, &["max_position_embeddings"])? as u32),
        ),
        (key("embedding_length"), GgufValue::U32(embed as u32)),
    ];
    if arch == "qwen35" {
        metadata.push((
            key("feed_forward_length"),
            GgufValue::U32(require_usize(config, &["intermediate_size"])? as u32),
        ));
    }
    metadata.push((key("attention.head_count"), GgufValue::U32(heads as u32)));
    metadata.push((
        key("attention.head_count_kv"),
        GgufValue::U32(kv_heads as u32),
    ));
    metadata.push((
        key("rope.dimension_sections"),
        GgufValue::Array(sections.into_iter().map(GgufValue::U32).collect()),
    ));
    let theta = find(rope, &["rope_theta"])
        .and_then(Value::as_f64)
        .ok_or_else(|| anyhow!("config.json has no rope_theta"))?;
    metadata.push((key("rope.freq_base"), GgufValue::F32(theta as f32)));
    let eps = find(config, &["rms_norm_eps"])
        .and_then(Value::as_f64)
        .unwrap_or(1e-6);
    metadata.push((
        key("attention.layer_norm_rms_epsilon"),
        GgufValue::F32(eps as f32),
    ));
    if arch == "qwen35moe" {
        metadata.push((
            key("expert_count"),
            GgufValue::U32(require_usize(config, &["num_experts"])? as u32),
        ));
        metadata.push((
            key("expert_used_count"),
            GgufValue::U32(require_usize(config, &["num_experts_per_tok"])? as u32),
        ));
    }
    metadata.push((key("attention.key_length"), GgufValue::U32(head_dim as u32)));
    metadata.push((
        key("attention.value_length"),
        GgufValue::U32(head_dim as u32),
    ));
    if arch == "qwen35moe" {
        metadata.push((
            key("expert_feed_forward_length"),
            GgufValue::U32(require_usize(config, &["moe_intermediate_size"])? as u32),
        ));
        metadata.push((
            key("expert_shared_feed_forward_length"),
            GgufValue::U32(require_usize(config, &["shared_expert_intermediate_size"])? as u32),
        ));
    }
    metadata.extend([
        (
            key("ssm.conv_kernel"),
            GgufValue::U32(require_usize(config, &["linear_conv_kernel_dim"])? as u32),
        ),
        (key("ssm.state_size"), GgufValue::U32(delta.key_dim as u32)),
        (key("ssm.group_count"), GgufValue::U32(delta.key as u32)),
        (
            key("ssm.time_step_rank"),
            GgufValue::U32(delta.value as u32),
        ),
        (
            key("ssm.inner_size"),
            GgufValue::U32((delta.value * delta.value_dim) as u32),
        ),
        (
            key("full_attention_interval"),
            GgufValue::U32(interval as u32),
        ),
        (
            key("rope.dimension_count"),
            GgufValue::U32((head_dim as f64 * rotated) as u32),
        ),
    ]);

    let root = ["model.language_model", "model"]
        .into_iter()
        .find(|r| weights.has(&format!("{r}.embed_tokens.weight")))
        .ok_or_else(|| anyhow!("the checkpoint has no embed_tokens.weight"))?;
    let skipped: Vec<String> = weights
        .names()
        .into_iter()
        .filter(|n| ELSEWHERE.iter().any(|p| n.starts_with(p)))
        .map(str::to_string)
        .collect();
    // Every RMSNorm but the delta rule's output norm is an offset from one.
    let offset = |name: &str| name.ends_with("norm.weight") && name != "ssm_norm.weight";

    let mut tensors = vec![Mapped::new(
        "token_embd.weight",
        Source::tensor(format!("{root}.embed_tokens.weight")),
    )];
    let layer_prefix = format!("{root}.layers");
    for layer in 0..layers {
        let full = (layer + 1) % interval == 0;
        let own: &[(&str, &str)] = if full { &ATTENTION } else { &LINEAR };
        for &(suffix, name) in COMMON.iter().chain(own).chain(&SHARED) {
            let source = format!("{layer_prefix}.{layer}.{suffix}");
            if !weights.has(&source) {
                continue;
            }
            let mut mapped = Mapped::new(format!("blk.{layer}.{name}"), Source::tensor(source));
            if offset(name) {
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
        tensors.extend(super::llama::experts(
            config,
            weights,
            &layer_prefix,
            layer,
        )?);
    }
    tensors.push(
        Mapped::new(
            "output_norm.weight",
            Source::tensor(format!("{root}.norm.weight")),
        )
        .with(Transform::AddOne),
    );
    let tied = find(config, &["tie_word_embeddings"])
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let mut skipped = skipped;
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
