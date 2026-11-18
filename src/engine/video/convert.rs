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

//! LTX-2.5's text encoder as published
//! (`text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors`, 24.5 GiB
//! of `bf16`) turned into a GGUF the engine's Gemma 4 loads: once, at the
//! first start that needs it, kept under `<models>/orangu-converted/`.
//!
//! The file is Hugging Face's `Gemma4UnifiedForConditionalGeneration` —
//! the language model under `model.`, a vision tower and two multimodal
//! projectors this pipeline never runs, LTX's two projection heads under
//! `text_embedding_projection.`, the model's `config.json` in the header
//! (`gemma_config`) and its tokenizer as an embedded file
//! (`tokenizer_json`). The conversion is llama.cpp's own for Gemma 4
//! (`Gemma4Model` in `convert_hf_to_gguf`): the same tensor names, the same
//! hyperparameter keys, no shift on the norms, the `rope_freqs` that make
//! the global layers' rotation partial, and the vocabulary's tokens, merges,
//! scores and types derived from `tokenizer.json` the way `LlamaHfVocab`
//! derives them. The language model's linears and the heads are written as
//! `Q6_K`, which keeps them within a rounding of the `bf16` and fits beside
//! the transformer; the norms stay `f32`.

use crate::engine::image::safetensors::SafeTensors;
use crate::gguf::GgufValue;
use crate::gguf_write::TensorPlan;
use crate::quantize::{GGML_TYPE_F32, GGML_TYPE_Q6_K};
use anyhow::{Context, Result, anyhow, bail, ensure};
use std::path::{Path, PathBuf};

/// The tokens a Gemma 4 chat parser must always see rendered
/// (`Gemma4Model.set_vocab`'s `visible_tokens`): typed user-defined.
const VISIBLE_TOKENS: [&str; 7] = [
    "<|channel>",
    "<channel|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    "<|\"|>",
];

/// GGUF token types (`gguf.TokenType`).
const NORMAL: i32 = 1;
const CONTROL: i32 = 3;
const USER_DEFINED: i32 = 4;
const BYTE: i32 = 6;

/// Where the GGUF converted from `source` is kept.
pub fn converted_path(models_dir: &Path, source: &Path) -> PathBuf {
    let stem = source
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("text-encoder")
        .trim_end_matches("-bf16");
    models_dir
        .join("orangu-converted")
        .join(format!("{stem}-Q6_K.gguf"))
}

/// The GGUF tokenizer arrays from `tokenizer.json`: tokens in id order,
/// their types, and the BPE merges.
pub struct Vocab {
    pub tokens: Vec<String>,
    pub types: Vec<i32>,
    pub merges: Vec<String>,
}

pub fn vocab_from_tokenizer_json(json: &serde_json::Value, size: usize) -> Result<Vocab> {
    let model = &json["model"];
    ensure!(
        model["type"].as_str() == Some("BPE"),
        "the text encoder's tokenizer is not BPE"
    );
    let mut tokens = vec![None::<String>; size];
    for (text, id) in model["vocab"]
        .as_object()
        .ok_or_else(|| anyhow!("the tokenizer has no vocab"))?
    {
        let id = id
            .as_u64()
            .ok_or_else(|| anyhow!("a vocab id is not a number"))? as usize;
        ensure!(id < size, "token id {id} is past the vocabulary of {size}");
        tokens[id] = Some(text.clone());
    }
    let mut special = vec![false; size];
    let mut added = vec![false; size];
    for token in json["added_tokens"].as_array().into_iter().flatten() {
        let id = token["id"].as_u64().unwrap_or(u64::MAX) as usize;
        ensure!(id < size, "added token id {id} is past the vocabulary");
        tokens[id] = token["content"].as_str().map(str::to_string);
        added[id] = true;
        special[id] = token["special"].as_bool().unwrap_or(false);
    }
    let tokens: Vec<String> = tokens
        .into_iter()
        .enumerate()
        .map(|(i, t)| t.ok_or_else(|| anyhow!("token id {i} has no text")))
        .collect::<Result<_>>()?;
    let is_byte = |t: &str| {
        t.len() == 6
            && t.starts_with("<0x")
            && t.ends_with('>')
            && t[3..5].chars().all(|c| c.is_ascii_hexdigit())
    };
    let types = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| {
            if VISIBLE_TOKENS.contains(&t.as_str()) {
                USER_DEFINED
            } else if is_byte(t) {
                BYTE
            } else if special[i] {
                CONTROL
            } else if added[i] {
                USER_DEFINED
            } else {
                NORMAL
            }
        })
        .collect();
    let merges = model["merges"]
        .as_array()
        .ok_or_else(|| anyhow!("the tokenizer has no merges"))?
        .iter()
        .map(|m| match m {
            serde_json::Value::String(s) => Ok(s.clone()),
            serde_json::Value::Array(pair) => Ok(pair
                .iter()
                .filter_map(|p| p.as_str())
                .collect::<Vec<_>>()
                .join(" ")),
            _ => Err(anyhow!("a merge is neither a string nor a pair")),
        })
        .collect::<Result<_>>()?;
    Ok(Vocab {
        tokens,
        types,
        merges,
    })
}

/// The Hugging Face name of every tensor the GGUF carries, under the name
/// llama.cpp gives it.
fn gguf_name(hf: &str) -> Option<String> {
    if hf == "model.embed_tokens.weight" {
        return Some("token_embd.weight".into());
    }
    if hf == "model.norm.weight" {
        return Some("output_norm.weight".into());
    }
    if hf.starts_with("text_embedding_projection.") {
        return Some(hf.to_string());
    }
    let rest = hf.strip_prefix("model.layers.")?;
    let (layer, name) = rest.split_once('.')?;
    let mapped = match name {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "post_attention_norm.weight",
        "pre_feedforward_layernorm.weight" => "ffn_norm.weight",
        "post_feedforward_layernorm.weight" => "post_ffw_norm.weight",
        "layer_scalar" => "layer_output_scale.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_norm.weight" => "attn_q_norm.weight",
        "self_attn.k_norm.weight" => "attn_k_norm.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        _ => return None,
    };
    Some(format!("blk.{layer}.{mapped}"))
}

/// Converts `source` into the GGUF at `target` (written beside it under a
/// temporary name and moved into place, so a stopped conversion never
/// leaves a file that looks finished). `progress` hears `(done, total)`
/// tensors.
pub fn convert(source: &Path, target: &Path, progress: &dyn Fn(usize, usize)) -> Result<()> {
    let file = SafeTensors::open(source)?;
    let config: serde_json::Value = serde_json::from_str(
        file.metadata("gemma_config")
            .ok_or_else(|| anyhow!("{} has no gemma_config", source.display()))?,
    )
    .context("parsing gemma_config")?;
    let t = &config["text_config"];
    let int = |key: &str| -> Result<u32> {
        t[key]
            .as_u64()
            .map(|v| v as u32)
            .ok_or_else(|| anyhow!("gemma_config has no text_config.{key}"))
    };
    let num = |v: &serde_json::Value, what: &str| -> Result<f32> {
        v.as_f64()
            .map(|v| v as f32)
            .ok_or_else(|| anyhow!("gemma_config has no {what}"))
    };
    let n_layer = int("num_hidden_layers")?;
    let layer_types: Vec<bool> = t["layer_types"]
        .as_array()
        .ok_or_else(|| anyhow!("gemma_config has no layer_types"))?
        .iter()
        .map(|v| v.as_str() == Some("sliding_attention"))
        .collect();
    ensure!(
        layer_types.len() == n_layer as usize,
        "gemma_config lists {} layer types for {n_layer} layers",
        layer_types.len()
    );
    ensure!(
        !t["enable_moe_block"].as_bool().unwrap_or(false)
            && t["hidden_size_per_layer_input"].as_u64().unwrap_or(0) == 0,
        "the text encoder is a Gemma 4 variant (experts or per-layer inputs) this conversion does \
         not handle"
    );
    let head_dim_swa = int("head_dim")?;
    let head_dim_full = int("global_head_dim")?;
    let kv_swa = int("num_key_value_heads")?;
    let kv_full = int("num_global_key_value_heads")?;
    let full_rope = &t["rope_parameters"]["full_attention"];
    let swa_rope = &t["rope_parameters"]["sliding_attention"];
    ensure!(
        full_rope["rope_type"].as_str() == Some("proportional"),
        "the global layers' rotation is not the proportional one"
    );
    let partial_full = full_rope["partial_rotary_factor"].as_f64().unwrap_or(1.0);
    let partial_swa = swa_rope["partial_rotary_factor"].as_f64().unwrap_or(1.0);
    let vocab_size = int("vocab_size")? as usize;
    let tokenizer: serde_json::Value = serde_json::from_slice(file.bytes("tokenizer_json")?)
        .context("parsing the embedded tokenizer.json")?;
    let vocab = vocab_from_tokenizer_json(&tokenizer, vocab_size)?;
    let asset = |name: &str| -> Option<serde_json::Value> {
        file.bytes(name)
            .ok()
            .and_then(|b| serde_json::from_slice(b).ok())
    };
    let tokenizer_config = asset("hf_asset__tokenizer_config.json").unwrap_or_default();
    let generation = asset("hf_asset__generation_config.json").unwrap_or_default();
    let token_id = |key: &str, default: u32| -> u32 {
        tokenizer_config[key]
            .as_str()
            .and_then(|text| vocab.tokens.iter().position(|t| t == text))
            .map(|i| i as u32)
            .unwrap_or(default)
    };
    let chat_template = file
        .bytes("hf_asset__chat_template.jinja")
        .ok()
        .and_then(|b| std::str::from_utf8(b).ok())
        .map(str::to_string);

    let u32v = |v: u32| GgufValue::U32(v);
    let f32v = |v: f32| GgufValue::F32(v);
    let mut metadata: Vec<(String, GgufValue)> = vec![
        (
            "general.architecture".into(),
            GgufValue::String("gemma4".into()),
        ),
        ("general.type".into(), GgufValue::String("model".into())),
        (
            "general.name".into(),
            GgufValue::String("LTX-2.5 Gemma 4 12B text encoder".into()),
        ),
        (
            "general.source.file".into(),
            GgufValue::String(
                source
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default()
                    .to_string(),
            ),
        ),
        ("gemma4.block_count".into(), u32v(n_layer)),
        (
            "gemma4.context_length".into(),
            u32v(int("max_position_embeddings")?),
        ),
        ("gemma4.embedding_length".into(), u32v(int("hidden_size")?)),
        (
            "gemma4.feed_forward_length".into(),
            u32v(int("intermediate_size")?),
        ),
        (
            "gemma4.attention.head_count".into(),
            u32v(int("num_attention_heads")?),
        ),
        (
            "gemma4.attention.head_count_kv".into(),
            GgufValue::Array(
                layer_types
                    .iter()
                    .map(|&swa| u32v(if swa { kv_swa } else { kv_full }))
                    .collect(),
            ),
        ),
        (
            "gemma4.rope.freq_base".into(),
            f32v(num(&full_rope["rope_theta"], "the global rope_theta")?),
        ),
        (
            "gemma4.rope.freq_base_swa".into(),
            f32v(num(&swa_rope["rope_theta"], "the sliding rope_theta")?),
        ),
        (
            "gemma4.attention.layer_norm_rms_epsilon".into(),
            f32v(num(&t["rms_norm_eps"], "rms_norm_eps")?),
        ),
        ("gemma4.attention.key_length".into(), u32v(head_dim_full)),
        ("gemma4.attention.value_length".into(), u32v(head_dim_full)),
        (
            "gemma4.final_logit_softcapping".into(),
            f32v(num(
                &t["final_logit_softcapping"],
                "final_logit_softcapping",
            )?),
        ),
        (
            "gemma4.attention.sliding_window".into(),
            u32v(int("sliding_window")?),
        ),
        (
            "gemma4.attention.shared_kv_layers".into(),
            u32v(t["num_kv_shared_layers"].as_u64().unwrap_or(0) as u32),
        ),
        ("gemma4.embedding_length_per_layer_input".into(), u32v(0)),
        (
            "gemma4.attention.sliding_window_pattern".into(),
            GgufValue::Array(layer_types.iter().map(|&b| GgufValue::Bool(b)).collect()),
        ),
        ("gemma4.attention.key_length_swa".into(), u32v(head_dim_swa)),
        (
            "gemma4.attention.value_length_swa".into(),
            u32v(head_dim_swa),
        ),
        ("gemma4.rope.dimension_count".into(), u32v(head_dim_full)),
        (
            "gemma4.rope.dimension_count_swa".into(),
            u32v((head_dim_swa as f64 * partial_swa) as u32),
        ),
        (
            "tokenizer.ggml.model".into(),
            GgufValue::String("gemma4".into()),
        ),
        (
            "tokenizer.ggml.tokens".into(),
            GgufValue::Array(
                vocab
                    .tokens
                    .iter()
                    .cloned()
                    .map(GgufValue::String)
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.scores".into(),
            GgufValue::Array(vec![GgufValue::F32(-1000.0); vocab.tokens.len()]),
        ),
        (
            "tokenizer.ggml.token_type".into(),
            GgufValue::Array(vocab.types.iter().map(|&t| GgufValue::I32(t)).collect()),
        ),
        (
            "tokenizer.ggml.merges".into(),
            GgufValue::Array(
                vocab
                    .merges
                    .iter()
                    .cloned()
                    .map(GgufValue::String)
                    .collect(),
            ),
        ),
        (
            "tokenizer.ggml.bos_token_id".into(),
            u32v(token_id("bos_token", 2)),
        ),
        (
            "tokenizer.ggml.eos_token_id".into(),
            u32v(token_id("eos_token", 1)),
        ),
        (
            "tokenizer.ggml.unknown_token_id".into(),
            u32v(token_id("unk_token", 3)),
        ),
        (
            "tokenizer.ggml.padding_token_id".into(),
            u32v(token_id("pad_token", 0)),
        ),
        (
            "tokenizer.ggml.mask_token_id".into(),
            u32v(token_id("mask_token", 4)),
        ),
        (
            "tokenizer.ggml.add_space_prefix".into(),
            GgufValue::Bool(false),
        ),
        ("tokenizer.ggml.add_bos_token".into(), GgufValue::Bool(true)),
    ];
    if let Some(suppress) = generation["suppress_tokens"].as_array() {
        metadata.push((
            "tokenizer.ggml.suppress_tokens".into(),
            GgufValue::Array(
                suppress
                    .iter()
                    .filter_map(|v| v.as_u64())
                    .map(|v| GgufValue::I32(v as i32))
                    .collect(),
            ),
        ));
    }
    if let Some(template) = chat_template {
        metadata.push((
            "tokenizer.chat_template".into(),
            GgufValue::String(template),
        ));
    }

    // The rotation of the global layers: the first `partial` of each head's
    // frequency pairs turn at their own rate, the rest are held still by a
    // factor of 1e30 (`Gemma4Model.generate_extra_tensors`).
    let n_rot = (head_dim_full as f64 * partial_full / 2.0) as usize;
    let n_unrot = head_dim_full as usize / 2 - n_rot;
    let rope_freqs: Vec<f32> = std::iter::repeat_n(1.0f32, n_rot)
        .chain(std::iter::repeat_n(1e30f32, n_unrot))
        .collect();

    // The plan, in the reference's order: the norms and the embeddings,
    // then layer by layer, then the heads.
    let mut names: Vec<(String, String)> = Vec::new();
    let mut hf_names: Vec<String> = file
        .names()
        .into_iter()
        .map(str::to_string)
        .filter(|n| gguf_name(n).is_some())
        .collect();
    let layer_of = |n: &str| -> usize {
        n.strip_prefix("model.layers.")
            .and_then(|r| r.split('.').next())
            .and_then(|l| l.parse().ok())
            .unwrap_or(usize::MAX)
    };
    hf_names.sort_by_key(|n| (layer_of(n), n.clone()));
    for hf in hf_names {
        let g = gguf_name(&hf).expect("filtered above");
        names.push((g, hf));
    }
    let mut plans = vec![TensorPlan {
        name: "rope_freqs.weight".into(),
        dims: vec![rope_freqs.len() as u64],
        ggml_type: GGML_TYPE_F32,
    }];
    for (g, hf) in &names {
        let (shape, _) = file
            .info(hf)
            .ok_or_else(|| anyhow!("{hf} vanished from the header"))?;
        // GGUF order: the row length first.
        let dims: Vec<u64> = shape.iter().rev().map(|&d| d as u64).collect();
        let ncols = dims[0] as usize;
        let ggml_type = if dims.len() == 2 && ncols.is_multiple_of(256) {
            GGML_TYPE_Q6_K
        } else {
            GGML_TYPE_F32
        };
        plans.push(TensorPlan {
            name: g.clone(),
            dims,
            ggml_type,
        });
    }
    ensure!(
        plans.iter().any(|p| p.name == "token_embd.weight")
            && plans.iter().any(|p| p.name == "output_norm.weight")
            && plans
                .iter()
                .any(|p| p.name == "text_embedding_projection.audio_aggregate_embed.weight"),
        "{} is missing the embeddings, the final norm or a projection head",
        source.display()
    );
    let parent = target
        .parent()
        .ok_or_else(|| anyhow!("{} has no directory", target.display()))?;
    std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    let partial = target.with_extension("gguf.partial");
    let lookup: std::collections::HashMap<&str, &str> = names
        .iter()
        .map(|(g, h)| (g.as_str(), h.as_str()))
        .collect();
    let total = plans.len();
    let result = crate::gguf_write::write(
        &partial,
        &metadata,
        &plans,
        |plan| {
            if plan.name == "rope_freqs.weight" {
                return Ok(rope_freqs.clone());
            }
            let hf = lookup
                .get(plan.name.as_str())
                .ok_or_else(|| anyhow!("no source for {}", plan.name))?;
            Ok(file.tensor(hf)?.0)
        },
        &|n, _, _| progress(n, total),
    );
    if let Err(err) = result {
        let _ = std::fs::remove_file(&partial);
        return Err(err);
    }
    std::fs::rename(&partial, target)
        .with_context(|| format!("moving the converted encoder to {}", target.display()))?;
    if !crate::model_spec::is_ltx_text_encoder(target) {
        bail!(
            "the converted encoder {} does not read back as one",
            target.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hugging_face_names_map_to_llama_cpp_ones() {
        assert_eq!(
            gguf_name("model.embed_tokens.weight").unwrap(),
            "token_embd.weight"
        );
        assert_eq!(
            gguf_name("model.layers.5.layer_scalar").unwrap(),
            "blk.5.layer_output_scale.weight"
        );
        assert_eq!(
            gguf_name("model.layers.0.pre_feedforward_layernorm.weight").unwrap(),
            "blk.0.ffn_norm.weight"
        );
        assert_eq!(
            gguf_name("model.layers.47.self_attn.o_proj.weight").unwrap(),
            "blk.47.attn_output.weight"
        );
        assert!(gguf_name("vision_model.patch_dense.weight").is_none());
        assert!(gguf_name("multi_modal_projector.embedding_projection.weight").is_none());
        assert_eq!(
            gguf_name("text_embedding_projection.video_aggregate_embed.bias").unwrap(),
            "text_embedding_projection.video_aggregate_embed.bias"
        );
    }

    #[test]
    fn token_types_follow_llama_cpp() {
        let json = serde_json::json!({
            "model": {"type": "BPE", "vocab": {"<pad>": 0, "a": 1, "<0x0A>": 2, "<|channel>": 3, "<x>": 4},
                      "merges": [["a", "a"], "a b"]},
            "added_tokens": [
                {"id": 0, "content": "<pad>", "special": true},
                {"id": 3, "content": "<|channel>", "special": true},
                {"id": 4, "content": "<x>", "special": false}
            ]
        });
        let v = vocab_from_tokenizer_json(&json, 5).unwrap();
        assert_eq!(
            v.types,
            vec![CONTROL, NORMAL, BYTE, USER_DEFINED, USER_DEFINED]
        );
        assert_eq!(v.merges, vec!["a a".to_string(), "a b".to_string()]);
    }

    #[test]
    fn the_converted_file_is_named_for_its_source_and_type() {
        let p = converted_path(
            Path::new("/m"),
            Path::new("/x/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"),
        );
        assert_eq!(
            p,
            Path::new("/m/orangu-converted/gemma4-12b-with-proj-ltx-2.5-Q6_K.gguf")
        );
    }
}
