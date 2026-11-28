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

//! `whisper` — OpenAI's Whisper speech recognizer.
//!
//! An encoder over the log-mel spectrogram (two convolutions, then
//! pre-norm transformer blocks with learned positions) and a decoder that
//! writes text tokens, attending to its own tokens and to the encoder's
//! output. The output head is the token embedding, so `proj_out` is not
//! written. The convolutions keep the checkpoint's `[out, in, kernel]`.

use anyhow::Result;
use orangu::gguf::GgufValue;
use orangu::safetensors::SafeTensorsSet;
use serde_json::Value;

use super::{Conversion, Mapped, Source, check_all_read, require_usize};

const ATTN: [(&str, &str); 7] = [
    ("q_proj.weight", "q.weight"),
    ("q_proj.bias", "q.bias"),
    ("k_proj.weight", "k.weight"),
    ("v_proj.weight", "v.weight"),
    ("v_proj.bias", "v.bias"),
    ("out_proj.weight", "output.weight"),
    ("out_proj.bias", "output.bias"),
];

const FFN: [(&str, &str); 6] = [
    ("final_layer_norm.weight", "ffn_norm.weight"),
    ("final_layer_norm.bias", "ffn_norm.bias"),
    ("fc1.weight", "ffn_up.weight"),
    ("fc1.bias", "ffn_up.bias"),
    ("fc2.weight", "ffn_down.weight"),
    ("fc2.bias", "ffn_down.bias"),
];

pub fn plan(hf: &str, config: &Value, weights: &SafeTensorsSet) -> Result<Option<Conversion>> {
    if hf != "WhisperForConditionalGeneration" {
        return Ok(None);
    }
    let arch = "whisper";
    let enc_layers = require_usize(config, &["encoder_layers"])?;
    let dec_layers = require_usize(config, &["decoder_layers"])?;
    let vocab_size = require_usize(config, &["vocab_size"])?;
    let u32_of = |keys: &[&str]| -> Result<GgufValue> {
        Ok(GgufValue::U32(require_usize(config, keys)? as u32))
    };
    let key = |k: &str| format!("{arch}.{k}");
    let ids = |name: &str| {
        GgufValue::Array(
            config[name]
                .as_array()
                .or_else(|| config["generation_config"][name].as_array())
                .into_iter()
                .flatten()
                .filter_map(Value::as_u64)
                .map(|v| GgufValue::U32(v as u32))
                .collect(),
        )
    };
    let metadata = vec![
        (key("suppress_tokens"), ids("suppress_tokens")),
        (key("begin_suppress_tokens"), ids("begin_suppress_tokens")),
        (key("embedding_length"), u32_of(&["d_model"])?),
        (key("block_count"), GgufValue::U32(dec_layers as u32)),
        (
            key("attention.head_count"),
            u32_of(&["decoder_attention_heads"])?,
        ),
        (key("feed_forward_length"), u32_of(&["decoder_ffn_dim"])?),
        (key("context_length"), u32_of(&["max_target_positions"])?),
        (
            key("encoder.block_count"),
            GgufValue::U32(enc_layers as u32),
        ),
        (
            key("encoder.attention.head_count"),
            u32_of(&["encoder_attention_heads"])?,
        ),
        (
            key("encoder.context_length"),
            u32_of(&["max_source_positions"])?,
        ),
        (key("audio.n_mels"), u32_of(&["num_mel_bins"])?),
        (
            key("decoder_start_token_id"),
            u32_of(&["decoder_start_token_id"])?,
        ),
        (key("eos_token_id"), u32_of(&["eos_token_id"])?),
    ];

    let mut tensors = Vec::new();
    let mut put =
        |name: String, from: String| tensors.push(Mapped::new(name, Source::tensor(from)));
    for conv in ["conv1", "conv2"] {
        for part in ["weight", "bias"] {
            put(
                format!("enc.{conv}.{part}"),
                format!("model.encoder.{conv}.{part}"),
            );
        }
    }
    put(
        "enc.pos_embd.weight".into(),
        "model.encoder.embed_positions.weight".into(),
    );
    put(
        "dec.token_embd.weight".into(),
        "model.decoder.embed_tokens.weight".into(),
    );
    put(
        "dec.pos_embd.weight".into(),
        "model.decoder.embed_positions.weight".into(),
    );
    for (side, layers) in [("encoder", enc_layers), ("decoder", dec_layers)] {
        let short = &side[..3];
        for part in ["weight", "bias"] {
            put(
                format!("{short}.output_norm.{part}"),
                format!("model.{side}.layer_norm.{part}"),
            );
        }
        for layer in 0..layers {
            let from = format!("model.{side}.layers.{layer}");
            let to = format!("{short}.blk.{layer}");
            let mut blocks = vec![("self_attn", "attn")];
            if side == "decoder" {
                blocks.push(("encoder_attn", "cross_attn"));
            }
            for (hf_block, block) in blocks {
                for part in ["weight", "bias"] {
                    put(
                        format!("{to}.{block}_norm.{part}"),
                        format!("{from}.{hf_block}_layer_norm.{part}"),
                    );
                }
                for (suffix, name) in ATTN {
                    put(
                        format!("{to}.{block}_{name}"),
                        format!("{from}.{hf_block}.{suffix}"),
                    );
                }
            }
            for (suffix, name) in FFN {
                put(format!("{to}.{name}"), format!("{from}.{suffix}"));
            }
        }
    }
    let skipped: Vec<String> = ["proj_out.weight"]
        .into_iter()
        .filter(|n| weights.has(n))
        .map(String::from)
        .collect();
    check_all_read(weights, &tensors, &skipped, arch)?;

    Ok(Some(Conversion {
        architecture: arch,
        metadata,
        tensors,
        layers: dec_layers,
        vocab_size,
        tokenizer: super::tokenizer::Style::ByteLevel,
        tokenizer_overrides: Vec::new(),
    }))
}
