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

//! `gemma3` — Gemma 3, text only; and `gemma2`, Gemma 2, mapped but refused,
//! since the server does not run it.
//!
//! One block shape: RMSNorm before and after both the attention and the
//! feed-forward ("sandwich" norms), a GEGLU feed-forward, grouped-query
//! attention whose layers alternate between a sliding window and the full
//! context. What the table has to know:
//!
//! - **Every norm is stored as its offset from one**; the file carries the
//!   weight itself ([`Transform::AddOne`]), the per-head Q/K norms of
//!   Gemma 3 included.
//! - **Gemma 2 caps its logits**, in the attention and at the output; both
//!   caps are written. Gemma 3 has per-head Q/K norms instead of the
//!   attention cap.
//! - **Gemma 3's two kinds of layer rotate with different bases**: the
//!   full layers' base, and the window layers' under `*_swa`.
//! - **A Gemma 3 checkpoint with vision** keeps its language model under
//!   `language_model.` with its settings in `text_config`; the vision tower
//!   and its projector are a separate projector file and are not converted.
//! - **The vocabulary is a SentencePiece model** (`tokenizer.model`),
//!   written with each piece's own score.
//!
//! The output projection is the token embedding, so no `output.weight` is
//! written.

use anyhow::{Result, bail};
use orangu::gguf::GgufValue;
use orangu::safetensors::SafeTensorsSet;
use serde_json::Value;

use super::tokenizer::Style;
use super::{
    Conversion, Mapped, Source, Transform, check_all_read, find, find_usize, require_usize,
};

/// Per-block checkpoint suffixes and their GGUF names.
const BLOCK: [(&str, &str); 13] = [
    ("input_layernorm.weight", "attn_norm.weight"),
    (
        "post_attention_layernorm.weight",
        "post_attention_norm.weight",
    ),
    ("pre_feedforward_layernorm.weight", "ffn_norm.weight"),
    ("post_feedforward_layernorm.weight", "post_ffw_norm.weight"),
    ("self_attn.q_proj.weight", "attn_q.weight"),
    ("self_attn.k_proj.weight", "attn_k.weight"),
    ("self_attn.v_proj.weight", "attn_v.weight"),
    ("self_attn.o_proj.weight", "attn_output.weight"),
    ("self_attn.q_norm.weight", "attn_q_norm.weight"),
    ("self_attn.k_norm.weight", "attn_k_norm.weight"),
    ("mlp.gate_proj.weight", "ffn_gate.weight"),
    ("mlp.up_proj.weight", "ffn_up.weight"),
    ("mlp.down_proj.weight", "ffn_down.weight"),
];

/// Checkpoint prefixes of a multimodal Gemma 3's parts that are not the
/// language model.
const NOT_TEXT: [&str; 3] = [
    "vision_tower.",
    "multi_modal_projector.",
    "model.vision_tower.",
];

pub fn plan(hf: &str, config: &Value, weights: &SafeTensorsSet) -> Result<Option<Conversion>> {
    let arch = match hf {
        "Gemma2ForCausalLM" => "gemma2",
        "Gemma3ForCausalLM" | "Gemma3ForConditionalGeneration" => "gemma3",
        _ => return Ok(None),
    };
    let text = config
        .get("text_config")
        .filter(|v| v.is_object())
        .unwrap_or(config);
    // Where the language model's tensors live.
    let prefix = ["language_model.model", "model.language_model", "model"]
        .into_iter()
        .find(|p| weights.has(&format!("{p}.embed_tokens.weight")))
        .ok_or_else(|| anyhow::anyhow!("the checkpoint has no token embedding"))?;

    let layers = require_usize(text, &["num_hidden_layers"])?;
    let embed = require_usize(text, &["hidden_size"])?;
    let gemma3 = arch == "gemma3";
    let heads = if gemma3 {
        find_usize(text, &["num_attention_heads"]).unwrap_or(8)
    } else {
        require_usize(text, &["num_attention_heads"])?
    };
    let kv_heads =
        find_usize(text, &["num_key_value_heads"]).unwrap_or(if gemma3 { 4 } else { heads });
    let head_dim = if gemma3 {
        find_usize(text, &["head_dim"]).unwrap_or(256)
    } else {
        require_usize(text, &["head_dim"])?
    };
    let vocab_size =
        require_usize(text, &["vocab_size"]).or_else(|_| require_usize(config, &["vocab_size"]))?;

    let key = |k: &str| format!("{arch}.{k}");
    let u = |v: usize| GgufValue::U32(v as u32);
    let f = |v: f64| GgufValue::F32(v as f32);
    let context = find_usize(text, &["max_position_embeddings"]).unwrap_or(131072);
    let mut metadata: Vec<(String, GgufValue)> = vec![
        (key("context_length"), u(context)),
        (key("embedding_length"), u(embed)),
        (key("block_count"), u(layers)),
    ];
    if let Some(ff) = find_usize(text, &["intermediate_size"]) {
        metadata.push((key("feed_forward_length"), u(ff)));
    }
    metadata.push((key("attention.head_count"), u(heads)));
    metadata.push((key("attention.head_count_kv"), u(kv_heads)));
    let eps = find(text, &["rms_norm_eps"])
        .and_then(Value::as_f64)
        .unwrap_or(1e-6);
    metadata.push((key("attention.layer_norm_rms_epsilon"), f(eps)));
    metadata.push((key("attention.key_length"), u(head_dim)));
    metadata.push((key("attention.value_length"), u(head_dim)));

    if gemma3 {
        // The full layers' rotation, and the window layers' under `*_swa`:
        // in `rope_parameters` per kind of layer, or the older flat keys.
        let rope = text.get("rope_parameters").filter(|v| v.is_object());
        let full = rope
            .and_then(|r| r.get("full_attention"))
            .or(rope)
            .unwrap_or(text);
        let theta = full["rope_theta"].as_f64().unwrap_or(1_000_000.0);
        let scaling = full
            .get("rope_scaling")
            .filter(|v| v.is_object())
            .unwrap_or(full);
        let kind = scaling["rope_type"].as_str().or(scaling["type"].as_str());
        match (kind, scaling["factor"].as_f64()) {
            (None | Some("default"), _) => {}
            (Some("linear"), Some(factor)) => {
                metadata.push((key("rope.scaling.type"), GgufValue::String("linear".into())));
                metadata.push((key("rope.scaling.factor"), f(factor)));
            }
            (Some(other), _) => bail!("Gemma 3 rope scaling {other:?} is not converted"),
        }
        metadata.push((key("rope.freq_base"), f(theta)));
        let local = rope
            .and_then(|r| r.get("sliding_attention"))
            .and_then(|s| s["rope_theta"].as_f64())
            .or_else(|| text["rope_local_base_freq"].as_f64());
        if let Some(local) = local {
            metadata.push((key("rope.freq_base_swa"), f(local)));
        }
        if let Some(cap) = text["final_logit_softcapping"].as_f64() {
            metadata.push((key("final_logit_softcapping"), f(cap)));
        }
        if text["sliding_window_pattern"].as_u64() != Some(1)
            && let Some(window) = find_usize(text, &["sliding_window"])
        {
            metadata.push((key("attention.sliding_window"), u(window)));
        }
    } else {
        for (from, to) in [
            ("attn_logit_softcapping", "attn_logit_softcapping"),
            ("final_logit_softcapping", "final_logit_softcapping"),
        ] {
            let cap = text[from]
                .as_f64()
                .ok_or_else(|| anyhow::anyhow!("config.json has no {from}"))?;
            metadata.push((key(to), f(cap)));
        }
        let window = require_usize(text, &["sliding_window"])?;
        metadata.push((key("attention.sliding_window"), u(window)));
    }

    let norm = |name: &str| {
        if name.ends_with("norm.weight") {
            Transform::AddOne
        } else {
            Transform::None
        }
    };
    let mut tensors = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    let embedding = format!("{prefix}.embed_tokens.weight");
    if let Some((shape, _)) = weights.info(&embedding)
        && shape[0] != vocab_size
    {
        bail!(
            "the token embedding has {} rows for a vocabulary of {vocab_size}",
            shape[0]
        );
    }
    tensors.push(Mapped::new("token_embd.weight", Source::tensor(embedding)));
    for layer in 0..layers {
        for (suffix, name) in BLOCK {
            let source = format!("{prefix}.layers.{layer}.{suffix}");
            if !weights.has(&source) {
                continue;
            }
            tensors.push(
                Mapped::new(format!("blk.{layer}.{name}"), Source::tensor(source)).with(norm(name)),
            );
        }
    }
    tensors.push(
        Mapped::new(
            "output_norm.weight",
            Source::tensor(format!("{prefix}.norm.weight")),
        )
        .with(Transform::AddOne),
    );
    for name in weights.names() {
        if name == "lm_head.weight" || NOT_TEXT.iter().any(|p| name.starts_with(p)) {
            skipped.push(name.to_string());
        }
    }
    check_all_read(weights, &tensors, &skipped, arch)?;
    if arch == "gemma2" {
        bail!(
            "orangu-server does not run Gemma 2: it caps its attention logits, which the \
             server does not implement, so the file would be refused"
        );
    }

    Ok(Some(Conversion {
        architecture: arch,
        metadata,
        tensors,
        layers,
        vocab_size,
        tokenizer: Style::SentencePiece,
        tokenizer_overrides: vec![(
            "tokenizer.ggml.add_space_prefix".into(),
            GgufValue::Bool(false),
        )],
    }))
}
