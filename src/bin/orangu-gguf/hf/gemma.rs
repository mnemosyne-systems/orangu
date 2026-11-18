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

//! `gemma4` — the Gemma 4 models: dense (E2B, E4B, 12B, 31B) and the
//! 26B-A4B mixture.
//!
//! What sets the architecture apart, and therefore the table:
//!
//! - **Two kinds of attention layer.** `layer_types` marks each layer
//!   sliding-window or full, and the two have different head sizes, KV head
//!   counts and rotary bases, so those keys come in pairs (`*_swa`) and the
//!   pattern is written as one boolean per layer.
//! - **Partial rotation on the full layers.** A full layer rotates only a
//!   fraction of each head (`partial_rotary_factor`), placed so the
//!   rotation cannot simply be cut short: the unrotated pairs get a
//!   frequency factor of 1e30 in `rope_freqs`, which turns their angle to
//!   zero.
//! - **Per-layer embeddings** (the "E" models): a second embedding table
//!   with a slice per layer, a projection into it, and a gate and
//!   projection inside each block.
//! - **A mixture beside the dense feed-forward** (26B-A4B): a router with
//!   its own input scale, 128 experts with their gate and up projections
//!   fused, and a scale per expert on the down projection.
//! - **Shared KV.** The last `num_kv_shared_layers` layers reuse an earlier
//!   layer's keys and values; with `use_double_wide_mlp` their feed-forward
//!   is twice as wide, so the width is written per layer.
//!
//! Vision and audio towers are a separate projector file and are not
//! converted here.

use anyhow::{Result, anyhow, bail};
use orangu::gguf::GgufValue;
use orangu::safetensors::SafeTensorsSet;
use serde_json::Value;

use super::tokenizer::Style;
use super::{Conversion, Mapped, Source, check_all_read, find, find_usize, require_usize};

/// Per-block checkpoint suffixes and their GGUF names.
const BLOCK: [(&str, &str); 25] = [
    ("input_layernorm.weight", "attn_norm.weight"),
    (
        "post_attention_layernorm.weight",
        "post_attention_norm.weight",
    ),
    ("pre_feedforward_layernorm.weight", "ffn_norm.weight"),
    ("post_feedforward_layernorm.weight", "post_ffw_norm.weight"),
    ("post_per_layer_input_norm.weight", "post_norm.weight"),
    ("layer_scalar", "layer_output_scale.weight"),
    ("per_layer_input_gate.weight", "inp_gate.weight"),
    ("per_layer_projection.weight", "proj.weight"),
    ("self_attn.q_proj.weight", "attn_q.weight"),
    ("self_attn.k_proj.weight", "attn_k.weight"),
    ("self_attn.v_proj.weight", "attn_v.weight"),
    ("self_attn.o_proj.weight", "attn_output.weight"),
    ("self_attn.q_norm.weight", "attn_q_norm.weight"),
    ("self_attn.k_norm.weight", "attn_k_norm.weight"),
    ("mlp.gate_proj.weight", "ffn_gate.weight"),
    ("mlp.up_proj.weight", "ffn_up.weight"),
    ("mlp.down_proj.weight", "ffn_down.weight"),
    // The mixture (26B-A4B): a routed feed-forward beside the dense one,
    // each with its own norms, the experts' gate and up projections kept
    // fused.
    (
        "pre_feedforward_layernorm_2.weight",
        "pre_ffw_norm_2.weight",
    ),
    (
        "post_feedforward_layernorm_1.weight",
        "post_ffw_norm_1.weight",
    ),
    (
        "post_feedforward_layernorm_2.weight",
        "post_ffw_norm_2.weight",
    ),
    ("router.proj.weight", "ffn_gate_inp.weight"),
    ("router.scale", "ffn_gate_inp.scale"),
    ("router.per_expert_scale", "ffn_down_exps.scale"),
    ("experts.gate_up_proj", "ffn_gate_up_exps.weight"),
    ("experts.down_proj", "ffn_down_exps.weight"),
];

/// Top-level checkpoint tensors and their GGUF names.
const TOP: [(&str, &str); 5] = [
    ("embed_tokens.weight", "token_embd.weight"),
    (
        "embed_tokens_per_layer.weight",
        "per_layer_token_embd.weight",
    ),
    (
        "per_layer_model_projection.weight",
        "per_layer_model_proj.weight",
    ),
    (
        "per_layer_projection_norm.weight",
        "per_layer_proj_norm.weight",
    ),
    ("norm.weight", "output_norm.weight"),
];

/// Special tokens a chat parser reads from the output, so the server must
/// show them rather than hide them as control tokens.
const VISIBLE: &[&str] = &[
    "<|channel>",
    "<channel|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    "<|\"|>",
];

/// Checkpoint prefixes of the parts that belong to the projector file.
const PROJECTOR: [&str; 4] = [
    "model.vision_tower.",
    "model.audio_tower.",
    "model.embed_vision.",
    "model.embed_audio.",
];

pub fn plan(hf: &str, config: &Value, weights: &SafeTensorsSet) -> Result<Option<Conversion>> {
    if !matches!(hf, "Gemma4ForConditionalGeneration" | "Gemma4ForCausalLM") {
        return Ok(None);
    }
    let arch = "gemma4";
    let layers = require_usize(config, &["num_hidden_layers"])?;
    let embed = require_usize(config, &["hidden_size"])?;
    let heads = require_usize(config, &["num_attention_heads"])?;
    let kv_heads = require_usize(config, &["num_key_value_heads"])?;
    let vocab_size = require_usize(config, &["vocab_size"])?;
    let head_dim = require_usize(config, &["head_dim"])?;
    let head_dim_full = require_usize(config, &["global_head_dim"])?;
    let shared = find_usize(config, &["num_kv_shared_layers"]).unwrap_or(0);
    let types: Vec<bool> = config["layer_types"]
        .as_array()
        .ok_or_else(|| anyhow!("config.json has no layer_types"))?
        .iter()
        .map(|t| t.as_str() == Some("sliding_attention"))
        .collect();
    if types.len() != layers {
        bail!(
            "layer_types has {} entries for {layers} layers",
            types.len()
        );
    }
    let rope = &config["rope_parameters"];
    let full = &rope["full_attention"];
    let swa = &rope["sliding_attention"];
    if full["rope_type"].as_str() != Some("proportional") {
        bail!("Gemma 4 full-attention layers are expected to use proportional rotation");
    }

    let key = |k: &str| format!("{arch}.{k}");
    let u32v = |v: usize| GgufValue::U32(v as u32);
    let f32v = |v: Option<f64>| -> Result<GgufValue> {
        Ok(GgufValue::F32(
            v.ok_or_else(|| anyhow!("config.json lacks a value Gemma 4 needs"))? as f32,
        ))
    };
    let ff = require_usize(config, &["intermediate_size"])?;
    let feed_forward = if config["use_double_wide_mlp"].as_bool().unwrap_or(false) {
        GgufValue::Array(
            (0..layers)
                .map(|l| u32v(if l < layers - shared { ff } else { 2 * ff }))
                .collect(),
        )
    } else {
        u32v(ff)
    };
    let head_count_kv = match find_usize(config, &["num_global_key_value_heads"]) {
        Some(global) => GgufValue::Array(
            types
                .iter()
                .map(|&sliding| u32v(if sliding { kv_heads } else { global }))
                .collect(),
        ),
        None => u32v(kv_heads),
    };
    let swa_partial = swa["partial_rotary_factor"].as_f64().unwrap_or(1.0);
    let mut metadata: Vec<(String, GgufValue)> = vec![
        (key("block_count"), u32v(layers)),
        (
            key("context_length"),
            u32v(require_usize(config, &["max_position_embeddings"])?),
        ),
        (key("embedding_length"), u32v(embed)),
        (key("feed_forward_length"), feed_forward),
        (key("attention.head_count"), u32v(heads)),
        (key("attention.head_count_kv"), head_count_kv),
        (key("rope.freq_base"), f32v(full["rope_theta"].as_f64())?),
        (key("rope.freq_base_swa"), f32v(swa["rope_theta"].as_f64())?),
        (
            key("attention.layer_norm_rms_epsilon"),
            f32v(find(config, &["rms_norm_eps"]).and_then(Value::as_f64))?,
        ),
        (key("attention.key_length"), u32v(head_dim_full)),
        (key("attention.value_length"), u32v(head_dim_full)),
        (
            key("attention.sliding_window"),
            u32v(require_usize(config, &["sliding_window"])?),
        ),
        (key("attention.shared_kv_layers"), u32v(shared)),
        (
            key("embedding_length_per_layer_input"),
            u32v(find_usize(config, &["hidden_size_per_layer_input"]).unwrap_or(0)),
        ),
        (
            key("attention.sliding_window_pattern"),
            GgufValue::Array(types.iter().map(|&b| GgufValue::Bool(b)).collect()),
        ),
        (key("attention.key_length_swa"), u32v(head_dim)),
        (key("attention.value_length_swa"), u32v(head_dim)),
        (key("rope.dimension_count"), u32v(head_dim_full)),
        (
            key("rope.dimension_count_swa"),
            u32v((head_dim as f64 * swa_partial) as usize),
        ),
    ];
    if let Some(cap) = config["final_logit_softcapping"].as_f64() {
        metadata.push((key("final_logit_softcapping"), GgufValue::F32(cap as f32)));
    }
    if config["enable_moe_block"].as_bool().unwrap_or(false) {
        metadata.push((
            key("expert_count"),
            u32v(require_usize(config, &["num_experts"])?),
        ));
        metadata.push((
            key("expert_used_count"),
            u32v(require_usize(config, &["top_k_experts"])?),
        ));
        metadata.push((
            key("expert_feed_forward_length"),
            u32v(require_usize(config, &["moe_intermediate_size"])?),
        ));
    }

    // Full layers rotate `partial_rotary_factor` of each head; the rest of
    // the pairs get an infinite wavelength.
    let partial = full["partial_rotary_factor"].as_f64().unwrap_or(1.0);
    let rotated = (head_dim_full as f64 * partial / 2.0) as usize;
    let pairs = head_dim_full / 2;
    let mut freqs = vec![1.0f32; rotated];
    freqs.resize(pairs, 1e30);
    let mut tensors = vec![Mapped::new(
        "rope_freqs.weight",
        Source::Values {
            values: freqs,
            shape: vec![pairs],
        },
    )];

    let prefix = if weights.has("model.language_model.embed_tokens.weight") {
        "model.language_model."
    } else {
        "model."
    };
    let mut skipped: Vec<String> = weights
        .names()
        .into_iter()
        .filter(|n| PROJECTOR.iter().any(|p| n.starts_with(p)))
        .map(str::to_string)
        .collect();
    for (suffix, name) in TOP {
        let source = format!("{prefix}{suffix}");
        if weights.has(&source) {
            tensors.push(Mapped::new(name, Source::Tensor(source)));
        }
    }
    for layer in 0..layers {
        for (suffix, name) in BLOCK {
            let source = format!("{prefix}layers.{layer}.{suffix}");
            if weights.has(&source) {
                tensors.push(Mapped::new(
                    format!("blk.{layer}.{name}"),
                    Source::Tensor(source),
                ));
            }
        }
    }
    if weights.has("lm_head.weight") {
        skipped.push("lm_head.weight".into());
    }
    check_all_read(weights, &tensors, &skipped, arch)?;

    Ok(Some(Conversion {
        architecture: arch,
        metadata,
        tensors,
        layers,
        vocab_size,
        tokenizer: Style::ByteFallback {
            model: "gemma4",
            visible: VISIBLE,
        },
        tokenizer_overrides: vec![
            (
                "tokenizer.ggml.add_space_prefix".into(),
                GgufValue::Bool(false),
            ),
            ("tokenizer.ggml.add_bos_token".into(), GgufValue::Bool(true)),
        ],
    }))
}
