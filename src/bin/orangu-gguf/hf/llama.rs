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

//! The dense llama family: `llama` (Llama, Mistral, SmolLM), `granite`,
//! `mistral3` (Ministral 3), `qwen2` and `qwen3`.
//!
//! One block shape — RMSNorm, grouped-query attention with rotary
//! positions, a SwiGLU feed-forward — with four differences that matter
//! here: `qwen2` carries Q/K/V biases, `qwen3` carries per-head Q/K norms,
//! `llama`, `granite` and `mistral3` rotate neighbouring pairs, so their Q
//! and K rows are regrouped from the checkpoint's half-split order
//! ([`Transform::PermuteRotary`]), and `granite` scales the attention
//! scores, the embeddings, each sub-layer's output and the logits by
//! factors its configuration gives.
//!
//! `mistral3` rotates as `llama` does, with YaRN scaling and a temperature
//! on its queries that grows with position; it ships with a vision tower,
//! so its settings are in `text_config` and its tensors under
//! `language_model.`, and the tower is left to a projector file.

use anyhow::{Result, bail};
use orangu::gguf::GgufValue;
use orangu::safetensors::SafeTensorsSet;
use serde_json::Value;

use super::{
    Conversion, Mapped, Source, Transform, check_all_read, find, find_usize, require_usize,
};

/// The GGUF architecture for a checkpoint's `architectures[0]`.
fn architecture(hf: &str) -> Option<&'static str> {
    Some(match hf {
        "LlamaForCausalLM" | "LLaMAForCausalLM" | "MistralForCausalLM" => "llama",
        "Qwen2ForCausalLM" => "qwen2",
        "Qwen3ForCausalLM" => "qwen3",
        "Qwen3MoeForCausalLM" => "qwen3moe",
        "GraniteForCausalLM" => "granite",
        "Mistral3ForConditionalGeneration" | "Ministral3ForCausalLM" => "mistral3",
        _ => return None,
    })
}

/// Per-block checkpoint suffixes and their GGUF names.
const BLOCK: [(&str, &str); 15] = [
    ("input_layernorm.weight", "attn_norm.weight"),
    ("post_attention_layernorm.weight", "ffn_norm.weight"),
    ("self_attn.q_proj.weight", "attn_q.weight"),
    ("self_attn.q_proj.bias", "attn_q.bias"),
    ("self_attn.k_proj.weight", "attn_k.weight"),
    ("self_attn.k_proj.bias", "attn_k.bias"),
    ("self_attn.v_proj.weight", "attn_v.weight"),
    ("self_attn.v_proj.bias", "attn_v.bias"),
    ("self_attn.o_proj.weight", "attn_output.weight"),
    ("self_attn.q_norm.weight", "attn_q_norm.weight"),
    ("self_attn.k_norm.weight", "attn_k_norm.weight"),
    ("mlp.gate_proj.weight", "ffn_gate.weight"),
    ("mlp.up_proj.weight", "ffn_up.weight"),
    ("mlp.down_proj.weight", "ffn_down.weight"),
    // Present in some checkpoints, never read: the rotary frequencies are
    // computed from `rope.freq_base`.
    ("self_attn.rotary_emb.inv_freq", ""),
];

/// A routed feed-forward's router and expert tensors, for one layer of
/// those under `layers` (the checkpoint's prefix, `model.layers`).
///
/// The experts come one tensor per expert and projection, stacked here
/// into one tensor per projection, or already stacked with the gate and up
/// projections fused, split here into two.
pub fn experts(
    config: &Value,
    weights: &SafeTensorsSet,
    layers: &str,
    layer: usize,
) -> Result<Vec<Mapped>> {
    let Some(count) = find_usize(config, &["num_experts", "num_local_experts"]) else {
        return Ok(Vec::new());
    };
    let at = |s: &str| format!("{layers}.{layer}.mlp.{s}");
    // A layer listed in `mlp_only_layers` keeps a dense feed-forward.
    if !weights.has(&at("gate.weight")) {
        return Ok(Vec::new());
    }
    let mut out = vec![Mapped::new(
        format!("blk.{layer}.ffn_gate_inp.weight"),
        Source::tensor(at("gate.weight")),
    )];
    let fused = ["experts.gate_up_proj", "experts.gate_up_proj.weight"]
        .into_iter()
        .map(at)
        .find(|n| weights.has(n));
    if let Some(fused) = fused {
        let down = ["experts.down_proj", "experts.down_proj.weight"]
            .into_iter()
            .map(at)
            .find(|n| weights.has(n))
            .ok_or_else(|| anyhow::anyhow!("layer {layer} has fused experts but no down_proj"))?;
        out.push(Mapped::new(
            format!("blk.{layer}.ffn_gate_exps.weight"),
            Source::Half {
                name: fused.clone(),
                second: false,
            },
        ));
        out.push(Mapped::new(
            format!("blk.{layer}.ffn_up_exps.weight"),
            Source::Half {
                name: fused,
                second: true,
            },
        ));
        out.push(Mapped::new(
            format!("blk.{layer}.ffn_down_exps.weight"),
            Source::Tensor(down),
        ));
        return Ok(out);
    }
    for (projection, name) in [
        ("gate_proj", "ffn_gate_exps"),
        ("up_proj", "ffn_up_exps"),
        ("down_proj", "ffn_down_exps"),
    ] {
        out.push(Mapped::new(
            format!("blk.{layer}.{name}.weight"),
            Source::Stack(
                (0..count)
                    .map(|e| at(&format!("experts.{e}.{projection}.weight")))
                    .collect(),
            ),
        ));
    }
    Ok(out)
}

pub fn plan(hf: &str, config: &Value, weights: &SafeTensorsSet) -> Result<Option<Conversion>> {
    let Some(arch) = architecture(hf) else {
        return Ok(None);
    };
    // A checkpoint with vision keeps its language model's settings in
    // `text_config` and its tensors under a prefix of their own.
    let config = config
        .get("text_config")
        .filter(|v| v.is_object())
        .unwrap_or(config);
    let prefix = ["language_model.model", "model.language_model", "model"]
        .into_iter()
        .find(|p| weights.has(&format!("{p}.embed_tokens.weight")))
        .ok_or_else(|| anyhow::anyhow!("the checkpoint has no token embedding"))?;
    let layers = require_usize(config, &["num_hidden_layers", "n_layer"])?;
    let embed = require_usize(config, &["hidden_size", "n_embd"])?;
    let heads = require_usize(config, &["num_attention_heads", "n_head"])?;
    let kv_heads = find_usize(config, &["num_key_value_heads", "n_kv_heads"]).unwrap_or(heads);
    let vocab_size = require_usize(config, &["vocab_size"])?;
    // Granite sizes its heads from the width alone, whatever `head_dim`
    // says.
    let head_dim = find_usize(config, &["head_dim"]).filter(|_| arch != "granite");

    let key = |k: &str| format!("{arch}.{k}");
    let mut metadata: Vec<(String, GgufValue)> = vec![
        (key("block_count"), GgufValue::U32(layers as u32)),
        (key("embedding_length"), GgufValue::U32(embed as u32)),
    ];
    if let Some(ctx) = find_usize(
        config,
        &[
            "max_position_embeddings",
            "n_ctx",
            "n_positions",
            "max_length",
        ],
    ) {
        metadata.push((key("context_length"), GgufValue::U32(ctx as u32)));
    }
    if let Some(ff) = find_usize(config, &["intermediate_size", "n_inner"]) {
        metadata.push((key("feed_forward_length"), GgufValue::U32(ff as u32)));
    }
    metadata.push((key("attention.head_count"), GgufValue::U32(heads as u32)));
    metadata.push((
        key("attention.head_count_kv"),
        GgufValue::U32(kv_heads as u32),
    ));

    let rope = config
        .get("rope_parameters")
        .filter(|v| v.is_object())
        .unwrap_or(config);
    // Scaling sits in `rope_scaling`, or in `rope_parameters` itself.
    let scaling = find(rope, &["rope_scaling"])
        .filter(|v| v.is_object())
        .unwrap_or(rope);
    let kind = scaling["rope_type"]
        .as_str()
        .or_else(|| scaling["type"].as_str())
        .unwrap_or("");
    let factor = scaling["factor"].as_f64();
    let f = |v: f64| GgufValue::F32(v as f32);
    match (kind, factor) {
        ("" | "default", _) => {}
        ("linear", Some(factor)) => {
            metadata.push((key("rope.scaling.type"), GgufValue::String("linear".into())));
            metadata.push((key("rope.scaling.factor"), f(factor)));
        }
        ("yarn", Some(factor)) => {
            metadata.push((key("rope.scaling.type"), GgufValue::String("yarn".into())));
            metadata.push((key("rope.scaling.factor"), f(factor)));
            let original = scaling["original_max_position_embeddings"]
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("yarn scaling without its original context"))?;
            metadata.push((
                key("rope.scaling.original_context_length"),
                GgufValue::U32(original as u32),
            ));
            for (from, to) in [
                ("truncate", "rope.scaling.yarn_truncate"),
                ("extrapolation_factor", "rope.scaling.yarn_ext_factor"),
                ("attention_factor", "rope.scaling.yarn_attn_factor"),
                ("attn_factor", "rope.scaling.yarn_attn_factor"),
                ("beta_fast", "rope.scaling.yarn_beta_fast"),
                ("beta_slow", "rope.scaling.yarn_beta_slow"),
            ] {
                if let Some(v) = scaling[from].as_f64()
                    && !metadata.iter().any(|(k, _)| *k == key(to))
                {
                    metadata.push((key(to), f(v)));
                }
            }
            // Ministral 3: the YaRN log multiplier, and the temperature its
            // queries are scaled by with position.
            if arch == "mistral3" {
                if let Some(v) = scaling["mscale_all_dim"].as_f64() {
                    metadata.push((key("rope.scaling.yarn_log_multiplier"), f(v)));
                }
                if let Some(v) = scaling["llama_4_scaling_beta"].as_f64() {
                    metadata.push((key("attention.temperature_scale"), f(v)));
                }
            }
        }
        (other, _) => bail!("rope scaling {other:?} is not converted yet"),
    }
    if let Some(theta) = find(rope, &["rope_theta"]).and_then(Value::as_f64) {
        metadata.push((key("rope.freq_base"), GgufValue::F32(theta as f32)));
    }
    if let Some(eps) = find(config, &["rms_norm_eps", "norm_eps"]).and_then(Value::as_f64) {
        metadata.push((
            key("attention.layer_norm_rms_epsilon"),
            GgufValue::F32(eps as f32),
        ));
    }
    if let Some(dim) = head_dim {
        metadata.push((key("attention.key_length"), GgufValue::U32(dim as u32)));
        metadata.push((key("attention.value_length"), GgufValue::U32(dim as u32)));
    }
    if let Some(experts) = find_usize(config, &["num_experts", "num_local_experts"]) {
        metadata.push((key("expert_count"), GgufValue::U32(experts as u32)));
        if let Some(used) = find_usize(config, &["num_experts_per_tok"]) {
            metadata.push((key("expert_used_count"), GgufValue::U32(used as u32)));
        }
        if let Some(ff) = find_usize(config, &["moe_intermediate_size"]) {
            metadata.push((key("expert_feed_forward_length"), GgufValue::U32(ff as u32)));
        }
    }
    if matches!(arch, "llama" | "granite" | "mistral3") {
        metadata.push((key("vocab_size"), GgufValue::U32(vocab_size as u32)));
        let rope_dim = head_dim.unwrap_or(embed / heads);
        metadata.push((key("rope.dimension_count"), GgufValue::U32(rope_dim as u32)));
    }
    // Granite's four multipliers: on the attention scores, the token
    // embeddings, each sub-layer's output, and the logits (divided).
    if arch == "granite" {
        for (from, to) in [
            ("attention_multiplier", "attention.scale"),
            ("embedding_multiplier", "embedding_scale"),
            ("residual_multiplier", "residual_scale"),
            ("logits_scaling", "logit_scale"),
        ] {
            if let Some(v) = config[from].as_f64().filter(|v| *v != 0.0) {
                metadata.push((key(to), GgufValue::F32(v as f32)));
            }
        }
    }

    let mut tensors = Vec::new();
    // Checkpoint tensors read for nothing, on purpose.
    let mut skipped: Vec<String> = Vec::new();
    let take = |tensors: &mut Vec<Mapped>, source: &str, name: String, transform| {
        tensors.push(Mapped::new(name, Source::tensor(source)).with(transform));
    };
    let embedding = format!("{prefix}.embed_tokens.weight");
    take(
        &mut tensors,
        &embedding,
        "token_embd.weight".into(),
        Transform::None,
    );
    let block_prefix = format!("{prefix}.layers");
    for layer in 0..layers {
        for (suffix, name) in BLOCK {
            let source = format!("{block_prefix}.{layer}.{suffix}");
            if !weights.has(&source) {
                continue;
            }
            if name.is_empty() {
                skipped.push(source);
                continue;
            }
            let transform = match (arch, name) {
                ("llama" | "granite" | "mistral3", "attn_q.weight" | "attn_q.bias") => {
                    Transform::PermuteRotary { heads }
                }
                ("llama" | "granite" | "mistral3", "attn_k.weight" | "attn_k.bias") => {
                    Transform::PermuteRotary { heads: kv_heads }
                }
                _ => Transform::None,
            };
            take(
                &mut tensors,
                &source,
                format!("blk.{layer}.{name}"),
                transform,
            );
        }
        tensors.extend(experts(config, weights, &block_prefix, layer)?);
    }
    take(
        &mut tensors,
        &format!("{prefix}.norm.weight"),
        "output_norm.weight".into(),
        Transform::None,
    );
    let tied = config["tie_word_embeddings"].as_bool().unwrap_or(false);
    for head in ["lm_head.weight", "language_model.lm_head.weight"] {
        if !weights.has(head) {
            continue;
        }
        if tied {
            skipped.push(head.into());
        } else {
            take(&mut tensors, head, "output.weight".into(), Transform::None);
        }
    }
    // A vision tower and its projector are a separate projector file.
    for name in weights.names() {
        if [
            "vision_tower.",
            "multi_modal_projector.",
            "model.vision_tower.",
            "model.multi_modal_projector.",
        ]
        .iter()
        .any(|p| name.starts_with(p))
        {
            skipped.push(name.to_string());
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
