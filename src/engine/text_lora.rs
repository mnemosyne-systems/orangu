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

//! A `peft` LoRA for a chat model (`[orangu-server].text_lora`), merged
//! into the GGUF weights at load so every architecture and backend runs it
//! unchanged. What cannot be merged exactly (embedding adapters, DoRA,
//! `modules_to_save`, rank/alpha patterns, fused linears) is refused.

use crate::engine::image::lora::{LoraPair, MergedCache};
use crate::engine::loader::{LoadedModel, ModelConfig, QuantMatrix};
use anyhow::{Context, Result, bail, ensure};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

const PEFT_WEIGHTS: &str = "adapter_model.safetensors";
const PEFT_CONFIG: &str = "adapter_config.json";

/// Adapted linears by GGUF tensor name.
pub struct TextLora {
    pairs: Vec<(String, LoraPair)>,
}

/// The files [`MergedCache`] keys a merge on.
pub struct CacheKey<'a> {
    pub models_dir: &'a Path,
    pub model: &'a Path,
    pub lora: &'a Path,
}

pub struct Merged {
    pub tensors: usize,
    pub from_cache: bool,
}

/// A safetensors file, or the directory `peft` saved one in, as given or under `models_dir`.
pub fn resolve(models_dir: &Path, spec: &str) -> Result<PathBuf> {
    for candidate in [PathBuf::from(spec), models_dir.join(spec)] {
        if candidate.is_file() {
            return Ok(candidate);
        }
        let inside = candidate.join(PEFT_WEIGHTS);
        if inside.is_file() {
            return Ok(inside);
        }
    }
    bail!(
        "text_lora {spec}: no such file (or directory holding {PEFT_WEIGHTS}), as given or under {}",
        models_dir.display()
    )
}

impl TextLora {
    pub fn open(path: &Path, config: &ModelConfig) -> Result<Self> {
        let settings = path
            .parent()
            .map(|dir| dir.join(PEFT_CONFIG))
            .filter(|p| p.is_file())
            .map(|p| AdapterConfig::read(&p))
            .transpose()?;
        let file = crate::safetensors::SafeTensors::open(path)
            .with_context(|| format!("opening the LoRA {}", path.display()))?;
        let mut halves: HashMap<&str, [Option<&str>; 2]> = HashMap::new();
        let mut alphas: HashMap<&str, &str> = HashMap::new();
        for name in file.names() {
            match split_key(name) {
                Some((base, Half::Down)) => halves.entry(base).or_default()[0] = Some(name),
                Some((base, Half::Up)) => halves.entry(base).or_default()[1] = Some(name),
                None => match name.strip_suffix(".alpha") {
                    Some(base) => {
                        alphas.insert(base, name);
                    }
                    None => bail!(
                        "{}: {name} is not a LoRA half — embedding adapters, DoRA and \
                         modules_to_save cannot be merged",
                        path.display()
                    ),
                },
            }
        }
        let mut pairs = Vec::with_capacity(halves.len());
        for (base, [down, up]) in halves {
            let (Some(down), Some(up)) = (down, up) else {
                bail!("LoRA {base} has only one of its two projections");
            };
            let target = gguf_name(base).with_context(|| {
                format!("LoRA {base}: not a linear orangu can map onto a GGUF tensor")
            })?;
            let (down, down_shape) = file.tensor(down)?;
            let (mut up, up_shape) = file.tensor(up)?;
            ensure!(
                down_shape.len() == 2 && up_shape.len() == 2 && down_shape[0] == up_shape[1],
                "LoRA {base}: down is {down_shape:?}, up is {up_shape:?}"
            );
            let (rank, in_dim, out_dim) = (down_shape[0], down_shape[1], up_shape[0]);
            let alpha = match alphas.get(base) {
                Some(name) => file.tensor(name)?.0.first().copied(),
                None => None,
            };
            let scale = match (alpha, &settings) {
                (Some(alpha), _) => alpha / rank as f32,
                (None, Some(settings)) => settings.scale(rank),
                (None, None) => 1.0,
            };
            if let Some(heads) = rotary_heads(config, &target) {
                ensure!(
                    heads > 0 && out_dim.is_multiple_of(2 * heads),
                    "LoRA {base}: {out_dim} rows do not split into {heads} rotary heads"
                );
                up = permute_rotary(&up, out_dim, heads);
            }
            pairs.push((
                target,
                LoraPair {
                    down: QuantMatrix::from_f32_rows(down, in_dim, rank),
                    up: QuantMatrix::from_f32_rows(up, rank, out_dim),
                    rank,
                    scale,
                },
            ));
        }
        ensure!(!pairs.is_empty(), "{} holds no LoRA pairs", path.display());
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        for pair in pairs.windows(2) {
            ensure!(pair[0].0 != pair[1].0, "LoRA adapts {} twice", pair[0].0);
        }
        Ok(Self { pairs })
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Folds every pair into `model`'s weights; must run before the model is built.
    pub fn merge(self, model: &mut LoadedModel, cache: Option<CacheKey<'_>>) -> Result<Merged> {
        let reopen = |key: &CacheKey<'_>| MergedCache::open(key.models_dir, key.model, key.lora);
        let cache = cache.as_ref().map(|key| (key, reopen(key)));
        let mut bases = Vec::with_capacity(self.pairs.len());
        for (name, pair) in &self.pairs {
            let w = model.matrix(name).with_context(|| {
                format!("the LoRA adapts {name}, which this model does not have as its own tensor")
            })?;
            ensure!(
                w.in_dim == pair.down.in_dim && w.out_dim == pair.up.out_dim,
                "LoRA {name}: adapter is {}→{}, the model's tensor {}→{}",
                pair.down.in_dim,
                pair.up.out_dim,
                w.in_dim,
                w.out_dim
            );
            bases.push(w);
        }
        if let Some((_, cache)) = &cache
            && cache.is_hit()
            && self.map_cached(model, cache, &bases)?
        {
            return Ok(Merged {
                tensors: self.pairs.len(),
                from_cache: true,
            });
        }
        let mut cache = cache;
        for ((name, pair), w) in self.pairs.iter().zip(&bases) {
            let merged = pair.merge_into(w, name)?;
            replace(model, name, &merged)?;
            if let Some((_, cache)) = cache.as_mut() {
                cache.put(name, &merged);
            }
        }
        if let Some((key, mut cache)) = cache {
            match cache.flush() {
                Ok(_) => {
                    let again = reopen(key);
                    if again.is_hit() {
                        self.map_cached(model, &again, &bases)?;
                    }
                }
                Err(e) => log::warn!(
                    "orangu-server: [text] could not write the merged adapter to {}: {e:#} — \
                     the next start merges again",
                    cache.path().display()
                ),
            }
        }
        Ok(Merged {
            tensors: self.pairs.len(),
            from_cache: false,
        })
    }

    /// `false`, changing nothing, when `cache` lacks any pair.
    fn map_cached(
        &self,
        model: &mut LoadedModel,
        cache: &MergedCache,
        bases: &[QuantMatrix],
    ) -> Result<bool> {
        let mut found = Vec::with_capacity(bases.len());
        for ((name, _), w) in self.pairs.iter().zip(bases) {
            match cache.get(name, w) {
                Some(m) => found.push(m),
                None => return Ok(false),
            }
        }
        for ((name, _), m) in self.pairs.iter().zip(&found) {
            replace(model, name, m)?;
        }
        Ok(true)
    }
}

fn replace(model: &mut LoadedModel, name: &str, m: &QuantMatrix) -> Result<()> {
    model.replace_tensor_range(
        name,
        m.ggml_type(),
        m.shared_bytes(),
        m.byte_start(),
        m.raw_bytes().len(),
    )
}

struct AdapterConfig {
    alpha: f32,
    rslora: bool,
}

impl AdapterConfig {
    fn read(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let json: serde_json::Value =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        Self::parse(&json).with_context(|| format!("in {}", path.display()))
    }

    fn parse(json: &serde_json::Value) -> Result<Self> {
        for key in ["rank_pattern", "alpha_pattern"] {
            ensure!(
                json[key].as_object().is_none_or(|m| m.is_empty()),
                "{key} (a rank or alpha per module) is not supported"
            );
        }
        ensure!(
            !json["use_dora"].as_bool().unwrap_or(false),
            "DoRA adapters (use_dora) cannot be merged as a plain LoRA"
        );
        let alpha = json["lora_alpha"].as_f64().context("no lora_alpha")? as f32;
        Ok(Self {
            alpha,
            rslora: json["use_rslora"].as_bool().unwrap_or(false),
        })
    }

    fn scale(&self, rank: usize) -> f32 {
        if self.rslora {
            self.alpha / (rank as f32).sqrt()
        } else {
            self.alpha / rank as f32
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Half {
    Down,
    Up,
}

fn split_key(name: &str) -> Option<(&str, Half)> {
    for (suffix, half) in [
        (".lora_A.weight", Half::Down),
        (".lora_down.weight", Half::Down),
        (".lora_B.weight", Half::Up),
        (".lora_up.weight", Half::Up),
    ] {
        if let Some(base) = name.strip_suffix(suffix) {
            return Some((base, half));
        }
    }
    None
}

const LINEARS: [(&str, &str); 7] = [
    ("self_attn.q_proj", "attn_q"),
    ("self_attn.k_proj", "attn_k"),
    ("self_attn.v_proj", "attn_v"),
    ("self_attn.o_proj", "attn_output"),
    ("mlp.gate_proj", "ffn_gate"),
    ("mlp.up_proj", "ffn_up"),
    ("mlp.down_proj", "ffn_down"),
];

/// `…layers.3.self_attn.q_proj` → `blk.3.attn_q.weight`; `…lm_head` → `output.weight`.
fn gguf_name(base: &str) -> Option<String> {
    let base = base.strip_prefix("base_model.model.").unwrap_or(base);
    if base == "lm_head" || base.ends_with(".lm_head") {
        return Some("output.weight".to_string());
    }
    let at = if base.starts_with("layers.") {
        0
    } else {
        base.find(".layers.")? + 1
    };
    let rest = &base[at + "layers.".len()..];
    let (layer, linear) = rest.split_once('.')?;
    let layer: usize = layer.parse().ok()?;
    let (_, gguf) = LINEARS.iter().find(|(hf, _)| *hf == linear)?;
    Some(format!("blk.{layer}.{gguf}.weight"))
}

/// Heads Q/K rows were regrouped by at conversion (`orangu-gguf`'s `PermuteRotary`).
fn rotary_heads(config: &ModelConfig, target: &str) -> Option<usize> {
    if !matches!(
        config.architecture.as_str(),
        "llama" | "granite" | "mistral3"
    ) {
        return None;
    }
    if target.ends_with(".attn_q.weight") {
        Some(config.n_head)
    } else if target.ends_with(".attn_k.weight") {
        Some(config.n_head_kv)
    } else {
        None
    }
}

/// Per head, row `j · half + i` becomes row `2i + j`.
fn permute_rotary(values: &[f32], rows: usize, heads: usize) -> Vec<f32> {
    let width = values.len() / rows;
    let per_head = rows / heads;
    let half = per_head / 2;
    let mut out = vec![0.0f32; values.len()];
    for h in 0..heads {
        for i in 0..half {
            for j in 0..2 {
                let from = h * per_head + j * half + i;
                let to = h * per_head + 2 * i + j;
                out[to * width..(to + 1) * width]
                    .copy_from_slice(&values[from * width..(from + 1) * width]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn peft_names_map_onto_gguf_tensors() {
        let cases = [
            (
                "base_model.model.model.layers.3.self_attn.q_proj",
                Some("blk.3.attn_q.weight"),
            ),
            (
                "base_model.model.model.layers.12.mlp.down_proj",
                Some("blk.12.ffn_down.weight"),
            ),
            (
                "base_model.model.language_model.model.layers.0.self_attn.o_proj",
                Some("blk.0.attn_output.weight"),
            ),
            (
                "model.layers.7.mlp.gate_proj",
                Some("blk.7.ffn_gate.weight"),
            ),
            ("base_model.model.lm_head", Some("output.weight")),
            ("base_model.model.model.layers.3.self_attn.qkv_proj", None),
            ("base_model.model.model.embed_tokens", None),
            ("base_model.model.model.layers.x.mlp.up_proj", None),
        ];
        for (base, want) in cases {
            assert_eq!(gguf_name(base).as_deref(), want, "{base}");
        }
    }

    #[test]
    fn the_scale_follows_adapter_config() {
        let plain = AdapterConfig::parse(&serde_json::json!({"r": 16, "lora_alpha": 32})).unwrap();
        assert_eq!(plain.scale(16), 2.0);
        let rs = AdapterConfig::parse(
            &serde_json::json!({"r": 16, "lora_alpha": 32, "use_rslora": true}),
        )
        .unwrap();
        assert_eq!(rs.scale(16), 8.0);
        assert!(
            AdapterConfig::parse(&serde_json::json!({"lora_alpha": 8, "use_dora": true})).is_err()
        );
        assert!(
            AdapterConfig::parse(
                &serde_json::json!({"lora_alpha": 8, "rank_pattern": {"q_proj": 4}})
            )
            .is_err()
        );
        assert!(
            AdapterConfig::parse(&serde_json::json!({"lora_alpha": 8, "rank_pattern": {}})).is_ok()
        );
    }

    /// `P · (B·A) = (P·B) · A`, so regrouping `up` alone is enough.
    #[test]
    fn regrouping_up_regroups_the_delta() {
        let (rows, rank, width, heads) = (8usize, 2usize, 3usize, 2usize);
        let up: Vec<f32> = (0..rows * rank).map(|i| i as f32).collect();
        let down: Vec<f32> = (0..rank * width).map(|i| (i as f32) * 0.5 - 1.0).collect();
        let delta = |up: &[f32]| -> Vec<f32> {
            let mut d = vec![0.0f32; rows * width];
            for o in 0..rows {
                for c in 0..width {
                    for r in 0..rank {
                        d[o * width + c] += up[o * rank + r] * down[r * width + c];
                    }
                }
            }
            d
        };
        let after = delta(&permute_rotary(&up, rows, heads));
        let want = permute_rotary(&delta(&up), rows, heads);
        assert_eq!(after, want);
        let ids: Vec<f32> = (0..rows).map(|i| i as f32).collect();
        assert_eq!(
            permute_rotary(&ids, rows, heads),
            [0.0, 2.0, 1.0, 3.0, 4.0, 6.0, 5.0, 7.0]
        );
    }

    fn write_safetensors(path: &Path, tensors: &[(String, Vec<usize>, Vec<f32>)]) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, shape, values) in tensors {
            let begin = data.len();
            for v in values {
                data.extend_from_slice(&v.to_le_bytes());
            }
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [begin, data.len()]}),
            );
        }
        let header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend(header);
        bytes.extend(data);
        std::fs::write(path, bytes).unwrap();
    }

    fn config(architecture: &str) -> ModelConfig {
        ModelConfig {
            architecture: architecture.to_string(),
            n_vocab: 0,
            n_embd: 4,
            n_layer: 1,
            n_head: 2,
            n_head_kv: 1,
            head_dim: 2,
            n_ctx_train: 0,
            rope_dim: 2,
            rope_freq_base: 10000.0,
            rms_eps: 1e-6,
            pooling_type: crate::engine::loader::PoolingType::Mean,
        }
    }

    #[test]
    fn a_peft_directory_opens_with_its_scale_and_regrouped_query() {
        let dir = tempfile::tempdir().unwrap();
        let prefix = "base_model.model.model.layers.0";
        let up: Vec<f32> = (0..4).map(|i| i as f32).collect();
        write_safetensors(
            &dir.path().join(PEFT_WEIGHTS),
            &[
                (
                    format!("{prefix}.self_attn.q_proj.lora_A.weight"),
                    vec![1, 4],
                    vec![1.0; 4],
                ),
                (
                    format!("{prefix}.self_attn.q_proj.lora_B.weight"),
                    vec![4, 1],
                    up.clone(),
                ),
                (
                    format!("{prefix}.mlp.up_proj.lora_A.weight"),
                    vec![1, 4],
                    vec![1.0; 4],
                ),
                (
                    format!("{prefix}.mlp.up_proj.lora_B.weight"),
                    vec![4, 1],
                    up.clone(),
                ),
            ],
        );
        std::fs::write(
            dir.path().join(PEFT_CONFIG),
            r#"{"r": 1, "lora_alpha": 4, "peft_type": "LORA"}"#,
        )
        .unwrap();
        let path = resolve(Path::new("/nowhere"), dir.path().to_str().unwrap()).unwrap();
        let lora = TextLora::open(&path, &config("llama")).unwrap();
        let names: Vec<&str> = lora.pairs.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, ["blk.0.attn_q.weight", "blk.0.ffn_up.weight"]);
        let rows =
            |m: &QuantMatrix| -> Vec<f32> { (0..m.out_dim).flat_map(|r| m.row(r)).collect() };
        let (_, q) = &lora.pairs[0];
        assert_eq!(q.scale, 4.0);
        assert_eq!(
            rows(&q.up),
            [0.0, 1.0, 2.0, 3.0],
            "one row per head: nothing to swap"
        );
        let (_, ffn) = &lora.pairs[1];
        assert_eq!(rows(&ffn.up), up);

        let qwen = TextLora::open(&path, &config("qwen3")).unwrap();
        assert_eq!(rows(&qwen.pairs[0].1.up), up);
    }

    #[test]
    fn what_cannot_be_merged_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.safetensors");
        write_safetensors(
            &path,
            &[(
                "base_model.model.model.embed_tokens.lora_embedding_A".to_string(),
                vec![1, 4],
                vec![0.0; 4],
            )],
        );
        assert!(TextLora::open(&path, &config("qwen3")).is_err());
        write_safetensors(
            &path,
            &[(
                "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_string(),
                vec![1, 4],
                vec![0.0; 4],
            )],
        );
        let err = TextLora::open(&path, &config("qwen3")).err().unwrap();
        assert!(format!("{err:#}").contains("only one"), "{err:#}");
    }
}
