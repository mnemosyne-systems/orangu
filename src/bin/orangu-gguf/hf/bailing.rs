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

//! `bailingmoe3` — Ling 3.0: a hybrid of linear-attention layers and
//! latent-attention layers, over a mixture of experts.
//!
//! What the table has to know:
//!
//! - **Two kinds of layer.** Every `layer_group_size`-th layer (and every
//!   layer past the last whole group) attends over the full context through
//!   a compressed latent (MLA); the rest are linear attention (KDA) with
//!   short convolutions over their queries, keys and values. The file says
//!   which is which with a KV head count per layer: one for a latent layer,
//!   none for a linear one.
//! - **The latent layers' `kv_b` projection is split per head** into the
//!   key part, transposed, and the value part.
//! - **The linear layers' decay is stored as its logarithm** and written as
//!   the rate itself, as a column; their convolutions are written 4-D.
//! - **The first `first_k_dense_replace` layers are dense**; the rest route
//!   among `num_experts` experts in groups, with a shared expert and a bias
//!   on the router's scores.
//! - The gates: a latent layer's per-head output gate, a linear layer's
//!   forget-gate and output-gate projections, under different names.
//!
//! Configurations this does not cover — KDA low-rank projections, unsafe
//! gates, gates other than per head — are refused, as are next-token
//! prediction layers.

use anyhow::{Result, bail};
use orangu::gguf::GgufValue;
use orangu::safetensors::SafeTensorsSet;
use serde_json::Value;

use super::{Conversion, Mapped, Source, Transform, check_all_read, find_usize, require_usize};

/// Every layer's checkpoint suffixes and their GGUF names.
const COMMON: [(&str, &str); 2] = [
    ("input_layernorm.weight", "attn_norm.weight"),
    ("post_attention_layernorm.weight", "ffn_norm.weight"),
];

/// A linear-attention layer's.
const LINEAR: [(&str, &str); 8] = [
    ("attention.q_proj.weight", "attn_q.weight"),
    ("attention.k_proj.weight", "attn_k.weight"),
    ("attention.v_proj.weight", "attn_v.weight"),
    ("attention.o_proj.weight", "attn_output.weight"),
    ("attention.o_norm.weight", "ssm_norm.weight"),
    ("attention.b_proj.weight", "ssm_beta.weight"),
    ("attention.f_proj.weight", "ssm_f_a.weight"),
    ("attention.g_proj.weight", "ssm_g_a.weight"),
];

/// A latent-attention layer's, `kv_b_proj` apart.
const LATENT: [(&str, &str); 7] = [
    ("attention.dense.weight", "attn_output.weight"),
    ("attention.g_proj.weight", "attn_gate.weight"),
    ("attention.q_a_proj.weight", "attn_q_a.weight"),
    ("attention.q_a_layernorm.weight", "attn_q_a_norm.weight"),
    ("attention.q_b_proj.weight", "attn_q_b.weight"),
    (
        "attention.kv_a_proj_with_mqa.weight",
        "attn_kv_a_mqa.weight",
    ),
    ("attention.kv_a_layernorm.weight", "attn_kv_a_norm.weight"),
];

/// A dense layer's feed-forward.
const DENSE: [(&str, &str); 3] = [
    ("mlp.gate_proj.weight", "ffn_gate.weight"),
    ("mlp.up_proj.weight", "ffn_up.weight"),
    ("mlp.down_proj.weight", "ffn_down.weight"),
];

/// A routed layer's router and shared expert.
const ROUTED: [(&str, &str); 5] = [
    ("mlp.gate.weight", "ffn_gate_inp.weight"),
    ("mlp.gate.expert_bias", "exp_probs_b.bias"),
    (
        "mlp.shared_experts.gate_proj.weight",
        "ffn_gate_shexp.weight",
    ),
    ("mlp.shared_experts.up_proj.weight", "ffn_up_shexp.weight"),
    (
        "mlp.shared_experts.down_proj.weight",
        "ffn_down_shexp.weight",
    ),
];

/// Whether layer `layer` of `layers` attends through the latent: the last
/// of every group, and every layer past the last whole group.
fn is_latent(layer: usize, layers: usize, group: usize) -> bool {
    (layer + 1).is_multiple_of(group) || layer >= layers / group * group
}

pub fn plan(hf: &str, config: &Value, weights: &SafeTensorsSet) -> Result<Option<Conversion>> {
    if hf != "BailingMoeV3ForCausalLM" {
        return Ok(None);
    }
    let arch = "bailingmoe3";
    let flag = |k: &str| config[k].as_bool().unwrap_or(false);
    if !flag("no_kda_lora") {
        bail!("Ling 3.0 with KDA low-rank projections is not converted");
    }
    if !flag("kda_safe_gate") {
        bail!("Ling 3.0 with unsafe KDA gates is not converted");
    }
    if config["gated_attention_proj_granularity_type"].as_str() != Some("head_wise") {
        bail!("Ling 3.0 needs per-head attention gates");
    }
    if find_usize(config, &["num_nextn_predict_layers"]).unwrap_or(0) != 0 {
        bail!("Ling 3.0's next-token prediction layers are not converted");
    }

    let need = |k: &str| require_usize(config, &[k]);
    let layers = need("num_hidden_layers")?;
    let group = need("layer_group_size")?;
    let heads = need("num_attention_heads")?;
    let head_dim = need("head_dim")?;
    let kv_lora = need("kv_lora_rank")?;
    let nope = need("qk_nope_head_dim")?;
    let rope = need("qk_rope_head_dim")?;
    let v_dim = need("v_head_dim")?;
    let experts = need("num_experts")?;
    let dense_first = need("first_k_dense_replace")?;
    let vocab_size = need("vocab_size")?;

    let key = |k: &str| format!("{arch}.{k}");
    let u = |v: usize| GgufValue::U32(v as u32);
    let f = |v: f64| GgufValue::F32(v as f32);
    let num = |k: &str| {
        config[k]
            .as_f64()
            .ok_or_else(|| anyhow::anyhow!("config.json has no {k}"))
    };
    let mut metadata: Vec<(String, GgufValue)> = vec![
        (key("block_count"), u(layers)),
        (key("context_length"), u(need("max_position_embeddings")?)),
        (key("embedding_length"), u(need("hidden_size")?)),
        (key("feed_forward_length"), u(need("intermediate_size")?)),
        (key("attention.head_count"), u(heads)),
        (
            key("attention.head_count_kv"),
            GgufValue::Array(
                (0..layers)
                    .map(|l| GgufValue::U32(u32::from(is_latent(l, layers, group))))
                    .collect(),
            ),
        ),
        (key("rope.freq_base"), f(num("rope_theta")?)),
        (
            key("attention.layer_norm_rms_epsilon"),
            f(num("rms_norm_eps")?),
        ),
        (key("expert_count"), u(experts)),
        (key("expert_used_count"), u(need("num_experts_per_tok")?)),
    ];
    if let Some(groups) = find_usize(config, &["n_group"]) {
        metadata.push((key("expert_group_count"), u(groups)));
    }
    if let Some(used) = find_usize(config, &["topk_group"]) {
        metadata.push((key("expert_group_used_count"), u(used)));
    }
    let score = config["score_function"]
        .as_str()
        .or(config["scoring_func"].as_str());
    match score {
        Some("sigmoid") => metadata.push((key("expert_gating_func"), u(2))),
        Some("softmax") => metadata.push((key("expert_gating_func"), u(1))),
        None => {}
        Some(other) => bail!("expert scoring {other:?} is not converted"),
    }
    metadata.extend([
        (key("attention.key_length"), u(kv_lora + rope)),
        (key("attention.value_length"), u(head_dim)),
        (key("vocab_size"), u(vocab_size)),
        (key("ssm.conv_kernel"), u(need("short_conv_kernel_size")?)),
        (key("kda.head_dim"), u(head_dim)),
        (key("kda.safe_gate"), GgufValue::Bool(true)),
        (key("kda.gate_lower_bound"), f(num("kda_lower_bound")?)),
    ]);
    if let Some(q_lora) = find_usize(config, &["q_lora_rank"]) {
        metadata.push((key("attention.q_lora_rank"), u(q_lora)));
    }
    metadata.extend([
        (key("attention.kv_lora_rank"), u(kv_lora)),
        (key("rope.dimension_count"), u(rope)),
        (key("attention.key_length_mla"), u(nope + rope)),
        (key("attention.value_length_mla"), u(v_dim)),
        (
            key("expert_feed_forward_length"),
            u(need("moe_intermediate_size")?),
        ),
        (
            key("expert_shared_feed_forward_length"),
            u(need("moe_shared_expert_intermediate_size")?),
        ),
        (
            key("expert_shared_count"),
            u(find_usize(config, &["num_shared_experts"]).unwrap_or(1)),
        ),
        (key("leading_dense_block_count"), u(dense_first)),
        (
            key("expert_weights_scale"),
            f(num("routed_scaling_factor")?),
        ),
        (
            key("expert_weights_norm"),
            GgufValue::Bool(flag("norm_topk_prob")),
        ),
    ]);
    for (from, to) in [
        ("expert_swiglu_limit_list", "swiglu_clamp_exp"),
        ("share_expert_swiglu_limit_list", "swiglu_clamp_shexp"),
    ] {
        if let Some(list) = config[from].as_array() {
            let mut values: Vec<GgufValue> = list
                .iter()
                .take(layers)
                .map(|v| GgufValue::F32(v.as_f64().unwrap_or(0.0) as f32))
                .collect();
            values.resize(layers, GgufValue::F32(0.0));
            metadata.push((key(to), GgufValue::Array(values)));
        }
    }

    let mut tensors = Vec::new();
    let plain = |tensors: &mut Vec<Mapped>, source: String, name: String| {
        tensors.push(Mapped::new(name, Source::Tensor(source)));
    };
    plain(
        &mut tensors,
        "model.word_embeddings.weight".into(),
        "token_embd.weight".into(),
    );
    for layer in 0..layers {
        let at = |s: &str| format!("model.layers.{layer}.{s}");
        let blk = |s: &str| format!("blk.{layer}.{s}");
        let latent = is_latent(layer, layers, group);
        let table: &[(&str, &str)] = if latent { &LATENT } else { &LINEAR };
        for (from, to) in COMMON.iter().chain(table) {
            plain(&mut tensors, at(from), blk(to));
        }
        if latent {
            let kv_b = at("attention.kv_b_proj.weight");
            tensors.push(Mapped::new(
                blk("attn_k_b.weight"),
                Source::HeadRows {
                    name: kv_b.clone(),
                    heads,
                    offset: 0,
                    count: nope,
                    transpose: true,
                },
            ));
            tensors.push(Mapped::new(
                blk("attn_v_b.weight"),
                Source::HeadRows {
                    name: kv_b,
                    heads,
                    offset: nope,
                    count: v_dim,
                    transpose: false,
                },
            ));
        } else {
            tensors.push(
                Mapped::new(blk("ssm_a"), Source::Tensor(at("attention.A_log")))
                    .with(Transform::Exp)
                    .with(Transform::Column),
            );
            plain(&mut tensors, at("attention.dt_bias"), blk("ssm_dt.bias"));
            for x in ["q", "k", "v"] {
                tensors.push(
                    Mapped::new(
                        blk(&format!("ssm_conv1d_{x}.weight")),
                        Source::Tensor(at(&format!("attention.{x}_conv1d.weight"))),
                    )
                    .with(Transform::Conv4d),
                );
            }
        }
        if layer < dense_first {
            for (from, to) in DENSE {
                plain(&mut tensors, at(from), blk(to));
            }
            continue;
        }
        for (from, to) in ROUTED {
            plain(&mut tensors, at(from), blk(to));
        }
        for (projection, name) in [
            ("gate_proj", "ffn_gate_exps"),
            ("up_proj", "ffn_up_exps"),
            ("down_proj", "ffn_down_exps"),
        ] {
            tensors.push(Mapped::new(
                blk(&format!("{name}.weight")),
                Source::Stack(
                    (0..experts)
                        .map(|e| at(&format!("mlp.experts.{e}.{projection}.weight")))
                        .collect(),
                ),
            ));
        }
    }
    plain(
        &mut tensors,
        "model.norm.weight".into(),
        "output_norm.weight".into(),
    );
    let mut skipped = Vec::new();
    if config["tie_word_embeddings"].as_bool().unwrap_or(false) {
        skipped.push("lm_head.weight".to_string());
    } else {
        plain(
            &mut tensors,
            "lm_head.weight".into(),
            "output.weight".into(),
        );
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_last_of_every_group_and_the_remainder_attend_through_the_latent() {
        let latent: Vec<usize> = (0..10).filter(|&l| is_latent(l, 10, 4)).collect();
        assert_eq!(latent, [3, 7, 8, 9]);
        let ling: Vec<usize> = (0..24).filter(|&l| is_latent(l, 24, 4)).collect();
        assert_eq!(ling, [3, 7, 11, 15, 19, 23]);
    }
}
