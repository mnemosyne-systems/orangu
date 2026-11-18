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

//! The `ltxv` diffusion transformer — Lightricks' `LTXModel` (LTX-2's
//! 22-billion-parameter audio-video DiT, `AVTransformer3DModel` in the
//! file's `config`), read from the GGUF `unsloth/LTX-2.5-GGUF` publishes.
//!
//! Two streams run side by side through 48 identical blocks: **video**
//! tokens (one per latent pixel of every latent frame, 4096 wide, 32 heads
//! of 128) and **audio** tokens (one per latent audio frame, 2048 wide, 32
//! heads of 64). Each block, per stream:
//!
//! 1. self-attention, behind an AdaLN shift, scale and gate;
//! 2. cross-attention to the prompt (the text connectors' output), the
//!    query modulated by the step's timestep and the prompt's keys and
//!    values by the sigma (`cross_attention_adaln`);
//! 3. then the two streams meet: audio-to-video attention (video queries,
//!    audio keys) and video-to-audio, each from the streams as they stood
//!    before either was added, with their own modulation tables and gates;
//! 4. a GELU feed-forward, behind its own shift, scale and gate.
//!
//! Every attention has a per-head gate (`2 · sigmoid(to_gate_logits(x))`,
//! `apply_gated_attention`), RMS-normed queries and keys, and LTX's *split*
//! rotary embedding — video over (time in seconds, height and width in
//! pixels), audio over time in seconds, and the cross-modal attentions over
//! time alone, so a frame and the sound under it line up.
//!
//! The network predicts a **velocity**; the pipeline turns it into the
//! denoised latent as `x − v · σ`.

use super::ops::{AdaLayerNorm, Attention, FeedForward, KeyValues, Linear, Rope, vector};
use super::ops::{gated_add, layer_norm_rows, modulate, rms_norm_rows};
use crate::engine::backend::Backend;
use crate::engine::loader::LoadedModel;
use anyhow::{Context, Result, anyhow, ensure};
use std::sync::Arc;

/// The hyperparameters in the GGUF's `config` metadata (`transformer`).
#[derive(Debug, Clone)]
pub struct LtxConfig {
    pub dim: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub audio_dim: usize,
    pub audio_heads: usize,
    pub audio_head_dim: usize,
    pub n_layer: usize,
    pub in_channels: usize,
    pub audio_in_channels: usize,
    pub eps: f32,
    pub theta: f64,
    pub max_pos: [f32; 3],
    pub audio_max_pos: f32,
    /// The cross-modal attentions' extent in time: the larger of the two
    /// streams' own.
    pub cross_max_pos: f32,
    /// The rotary width of the cross-modal attentions
    /// (`audio_cross_attention_dim`).
    pub cross_rope_dim: usize,
    pub timestep_scale: f32,
    /// `av_ca_timestep_scale_multiplier / timestep_scale_multiplier`.
    pub av_ca_factor: f32,
    pub connector_heads: usize,
    pub audio_connector_heads: usize,
    pub connector_max_pos: f32,
    /// The text encoder's `gemma_version` the checkpoint was trained with.
    pub gemma_version: Option<String>,
}

impl LtxConfig {
    pub fn from(loaded: &LoadedModel) -> Result<Self> {
        let raw = loaded
            .metadata
            .iter()
            .find_map(|(k, v)| match v {
                crate::gguf::GgufValue::String(s) if k == "config" => Some(s.as_str()),
                _ => None,
            })
            .ok_or_else(|| anyhow!("ltxv: the file has no `config` metadata"))?;
        let json: serde_json::Value =
            serde_json::from_str(raw).context("ltxv: parsing the `config` metadata")?;
        let t = &json["transformer"];
        let num = |key: &str, default: f64| t[key].as_f64().unwrap_or(default);
        let int = |key: &str, default: u64| t[key].as_u64().unwrap_or(default) as usize;
        let list = |key: &str, default: &[f64]| -> Vec<f32> {
            t[key]
                .as_array()
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_f64())
                        .map(|v| v as f32)
                        .collect()
                })
                .unwrap_or_else(|| default.iter().map(|&v| v as f32).collect())
        };
        // The pieces this engine runs only one way of: refuse the others
        // rather than run them wrongly.
        let expect = |key: &str, want: &serde_json::Value| -> Result<()> {
            let have = &t[key];
            ensure!(
                have.is_null() || have == want,
                "ltxv: config `{key}` is {have}, and this engine runs only {want}"
            );
            Ok(())
        };
        expect("rope_type", &serde_json::json!("split"))?;
        expect("use_middle_indices_grid", &serde_json::json!(true))?;
        expect("apply_gated_attention", &serde_json::json!(true))?;
        expect("cross_attention_adaln", &serde_json::json!(true))?;
        expect("caption_proj_before_connector", &serde_json::json!(true))?;
        let heads = int("num_attention_heads", 32);
        let head_dim = int("attention_head_dim", 128);
        let audio_heads = int("audio_num_attention_heads", 32);
        let audio_head_dim = int("audio_attention_head_dim", 64);
        let max_pos = list("positional_embedding_max_pos", &[20.0, 2048.0, 2048.0]);
        ensure!(
            max_pos.len() == 3,
            "ltxv: positional_embedding_max_pos is not three axes"
        );
        let audio_max_pos = list("audio_positional_embedding_max_pos", &[20.0]);
        let connector_max_pos = list("connector_positional_embedding_max_pos", &[1.0]);
        let ts = num("timestep_scale_multiplier", 1000.0) as f32;
        let gemma_version = json
            .get("gemma_source_checkpoint")
            .and_then(|g| g["gemma_version"].as_str())
            .map(str::to_string)
            .or_else(|| {
                loaded.metadata.iter().find_map(|(k, v)| match v {
                    crate::gguf::GgufValue::String(s) if k == "gemma_source_checkpoint" => {
                        serde_json::from_str::<serde_json::Value>(s)
                            .ok()?
                            .get("gemma_version")?
                            .as_str()
                            .map(str::to_string)
                    }
                    _ => None,
                })
            });
        Ok(Self {
            dim: heads * head_dim,
            heads,
            head_dim,
            audio_dim: audio_heads * audio_head_dim,
            audio_heads,
            audio_head_dim,
            n_layer: int("num_layers", 48),
            in_channels: int("in_channels", 128),
            audio_in_channels: int("audio_in_channels", 128),
            eps: num("norm_eps", 1e-6) as f32,
            theta: num("positional_embedding_theta", 10000.0),
            max_pos: [max_pos[0], max_pos[1], max_pos[2]],
            audio_max_pos: audio_max_pos[0],
            cross_max_pos: max_pos[0].max(audio_max_pos[0]),
            cross_rope_dim: int("audio_cross_attention_dim", 2048),
            timestep_scale: ts,
            av_ca_factor: num("av_ca_timestep_scale_multiplier", 1.0) as f32 / ts,
            connector_heads: int("connector_num_attention_heads", 30),
            audio_connector_heads: t["audio_connector_num_attention_heads"]
                .as_u64()
                .map(|v| v as usize)
                .unwrap_or_else(|| int("connector_num_attention_heads", 30)),
            connector_max_pos: connector_max_pos[0],
            gemma_version,
        })
    }
}

/// The modulation tables of one block: rows of `[dim]`.
struct Tables {
    /// `scale_shift_table`, 9 rows: self-attention shift, scale, gate;
    /// feed-forward shift, scale, gate; cross-attention query shift, scale,
    /// gate.
    main: Vec<f32>,
    /// `prompt_scale_shift_table`, 2 rows: the prompt keys' shift, scale.
    prompt: Vec<f32>,
    /// `scale_shift_table_a2v_ca_*`, 5 rows: scale and shift for
    /// audio-to-video, scale and shift for video-to-audio, and the gate.
    cross: Vec<f32>,
}

struct Block {
    attn1: Attention,
    attn2: Attention,
    ff: FeedForward,
    video: Tables,
    audio_attn1: Attention,
    audio_attn2: Attention,
    audio_ff: FeedForward,
    audio: Tables,
    /// Video queries, audio keys.
    a2v: Attention,
    /// Audio queries, video keys.
    v2a: Attention,
}

/// One stream's inputs to a pass.
pub struct StreamInput<'a> {
    /// `[n, in_channels]` latent tokens.
    pub latent: &'a [f32],
    /// Each token's timestep — σ, or zero for a conditioning token.
    pub timesteps: &'a [f32],
    /// The stream's σ for this pass, what the prompt's keys and the other
    /// stream's gates are modulated by.
    pub sigma: f32,
    /// The text connectors' output for this stream, `[n_ctx, dim]`.
    pub context: &'a [f32],
}

/// Rotary tables for a stage: fixed by the latent's shape, built once and
/// read by every pass.
pub struct Geometry {
    pub n_video: usize,
    pub n_audio: usize,
    /// Tokens of the first latent frame, which carry the keyframe marker.
    pub keyframe_tokens: usize,
    video: Rope,
    audio: Rope,
    video_cross: Rope,
    audio_cross: Rope,
}

impl Geometry {
    /// `video_positions` is `[n_video][3]` (seconds, pixels, pixels) and
    /// `audio_positions` `[n_audio]` seconds, each already the middle of
    /// its token's span.
    pub fn new(
        c: &LtxConfig,
        video_positions: &[f32],
        audio_positions: &[f32],
        keyframe_tokens: usize,
    ) -> Self {
        let n_video = video_positions.len() / 3;
        let video_time: Vec<f32> = video_positions.chunks(3).map(|p| p[0]).collect();
        Self {
            n_video,
            n_audio: audio_positions.len(),
            keyframe_tokens,
            video: Rope::new(video_positions, 3, &c.max_pos, c.dim, c.heads, c.theta),
            audio: Rope::new(
                audio_positions,
                1,
                &[c.audio_max_pos],
                c.audio_dim,
                c.audio_heads,
                c.theta,
            ),
            video_cross: Rope::new(
                &video_time,
                1,
                &[c.cross_max_pos],
                c.cross_rope_dim,
                c.heads,
                c.theta,
            ),
            audio_cross: Rope::new(
                audio_positions,
                1,
                &[c.cross_max_pos],
                c.cross_rope_dim,
                c.audio_heads,
                c.theta,
            ),
        }
    }
}

/// The distinct timesteps of a stream's tokens and which one each token
/// has — every modulation is computed once per distinct value.
fn groups(timesteps: &[f32]) -> (Vec<f32>, Vec<usize>) {
    let mut values: Vec<f32> = Vec::new();
    let group = timesteps
        .iter()
        .map(|&t| match values.iter().position(|&v| v == t) {
            Some(i) => i,
            None => {
                values.push(t);
                values.len() - 1
            }
        })
        .collect();
    (values, group)
}

/// `table` (`[rows, dim]`) plus each group's `[rows, dim]` modulation —
/// `[groups][rows][dim]`.
fn add_table(table: &[f32], modulation: &[f32], rows: usize, dim: usize) -> Vec<f32> {
    let per = rows * dim;
    let mut out = modulation.to_vec();
    for g in out.chunks_mut(per) {
        for (v, t) in g.iter_mut().zip(table) {
            *v += t;
        }
    }
    out
}

/// Row `row` of every group of a `[groups][rows][dim]` table, as
/// `[groups][dim]`.
fn pick(sum: &[f32], rows: usize, dim: usize, row: usize) -> Vec<f32> {
    sum.chunks(rows * dim)
        .flat_map(|g| g[row * dim..(row + 1) * dim].iter().copied())
        .collect()
}

/// One stream's per-pass modulation.
struct StreamModulation {
    /// The token groups and each token's group.
    group: Vec<usize>,
    /// `adaln_single` per group, `[groups][9 · dim]`.
    main: Vec<f32>,
    /// `adaln_single`'s embedded timestep per group, `[groups][dim]`.
    embedded: Vec<f32>,
    /// `prompt_adaln_single` of the stream's σ, `[2 · dim]`.
    prompt: Vec<f32>,
    /// `av_ca_*_scale_shift_adaln_single` per group, `[groups][4 · dim]`.
    cross: Vec<f32>,
    /// `av_ca_*_gate_adaln_single` of the *other* stream's σ, `[dim]`.
    cross_gate: Vec<f32>,
}

pub struct LtxTransformer {
    backend: Arc<dyn Backend>,
    pub config: LtxConfig,
    patchify: Linear,
    audio_patchify: Linear,
    keyframes: Option<Vec<f32>>,
    adaln: AdaLayerNorm,
    audio_adaln: AdaLayerNorm,
    prompt_adaln: AdaLayerNorm,
    audio_prompt_adaln: AdaLayerNorm,
    video_cross_ss: AdaLayerNorm,
    audio_cross_ss: AdaLayerNorm,
    a2v_gate: AdaLayerNorm,
    v2a_gate: AdaLayerNorm,
    out_table: Vec<f32>,
    audio_out_table: Vec<f32>,
    proj_out: Linear,
    audio_proj_out: Linear,
    blocks: Vec<Block>,
}

impl LtxTransformer {
    pub fn load(loaded: &LoadedModel, backend: Arc<dyn Backend>) -> Result<Self> {
        let config = LtxConfig::from(loaded)?;
        let c = &config;
        let (d, ad) = (c.dim, c.audio_dim);
        let mut blocks = Vec::with_capacity(c.n_layer);
        for i in 0..c.n_layer {
            let p = format!("transformer_blocks.{i}");
            let attn = |name: &str, heads| Attention::load(loaded, &format!("{p}.{name}"), heads);
            let tables = |main: &str, prompt: &str, cross: &str, dim: usize| -> Result<Tables> {
                Ok(Tables {
                    main: vector(loaded, &format!("{p}.{main}"), 9 * dim)?,
                    prompt: vector(loaded, &format!("{p}.{prompt}"), 2 * dim)?,
                    cross: vector(loaded, &format!("{p}.{cross}"), 5 * dim)?,
                })
            };
            blocks.push(Block {
                attn1: attn("attn1", c.heads)?,
                attn2: attn("attn2", c.heads)?,
                ff: FeedForward::load(loaded, &format!("{p}.ff"))?,
                video: tables(
                    "scale_shift_table",
                    "prompt_scale_shift_table",
                    "scale_shift_table_a2v_ca_video",
                    d,
                )?,
                audio_attn1: attn("audio_attn1", c.audio_heads)?,
                audio_attn2: attn("audio_attn2", c.audio_heads)?,
                audio_ff: FeedForward::load(loaded, &format!("{p}.audio_ff"))?,
                audio: tables(
                    "audio_scale_shift_table",
                    "audio_prompt_scale_shift_table",
                    "scale_shift_table_a2v_ca_audio",
                    ad,
                )?,
                a2v: attn("audio_to_video_attn", c.audio_heads)?,
                v2a: attn("video_to_audio_attn", c.audio_heads)?,
            });
        }
        let keyframes = loaded
            .has_tensor("keyframes_abs_pos_embedding")
            .then(|| vector(loaded, "keyframes_abs_pos_embedding", d))
            .transpose()?;
        let model = Self {
            patchify: Linear::load(loaded, "patchify_proj")?,
            audio_patchify: Linear::load(loaded, "audio_patchify_proj")?,
            keyframes,
            adaln: AdaLayerNorm::load(loaded, "adaln_single")?,
            audio_adaln: AdaLayerNorm::load(loaded, "audio_adaln_single")?,
            prompt_adaln: AdaLayerNorm::load(loaded, "prompt_adaln_single")?,
            audio_prompt_adaln: AdaLayerNorm::load(loaded, "audio_prompt_adaln_single")?,
            video_cross_ss: AdaLayerNorm::load(loaded, "av_ca_video_scale_shift_adaln_single")?,
            audio_cross_ss: AdaLayerNorm::load(loaded, "av_ca_audio_scale_shift_adaln_single")?,
            a2v_gate: AdaLayerNorm::load(loaded, "av_ca_a2v_gate_adaln_single")?,
            v2a_gate: AdaLayerNorm::load(loaded, "av_ca_v2a_gate_adaln_single")?,
            out_table: vector(loaded, "scale_shift_table", 2 * d)?,
            audio_out_table: vector(loaded, "audio_scale_shift_table", 2 * ad)?,
            proj_out: Linear::load(loaded, "proj_out")?,
            audio_proj_out: Linear::load(loaded, "audio_proj_out")?,
            blocks,
            backend,
            config,
        };
        let c = &model.config;
        ensure!(
            model.patchify.in_dim() == c.in_channels
                && model.patchify.out_dim() == c.dim
                && model.audio_patchify.in_dim() == c.audio_in_channels
                && model.audio_patchify.out_dim() == c.audio_dim
                && model.proj_out.out_dim() == c.in_channels
                && model.audio_proj_out.out_dim() == c.audio_in_channels,
            "ltxv: the patchify and output projections do not match the stream widths"
        );
        for (i, b) in model.blocks.iter().enumerate() {
            ensure!(
                b.attn1.inner() == c.dim
                    && b.attn2.to_k.in_dim() == c.dim
                    && b.audio_attn1.inner() == c.audio_dim
                    && b.audio_attn2.to_k.in_dim() == c.audio_dim
                    && b.a2v.to_q.in_dim() == c.dim
                    && b.a2v.to_k.in_dim() == c.audio_dim
                    && b.v2a.to_q.in_dim() == c.audio_dim
                    && b.v2a.to_k.in_dim() == c.dim
                    && b.a2v.head_dim == c.cross_rope_dim / c.heads,
                "ltxv: transformer_blocks.{i} does not match the config's widths"
            );
        }
        Ok(model)
    }

    /// Seconds per video token per block, and the block count: block 0's
    /// feed-forward timed on `n` tokens (after one untimed run, which
    /// pages its weights in), scaled by the share of a token's multiply-adds
    /// a block's feed-forward is (`8 d²` of `14 d² + 4 d · d_audio`).
    pub fn calibrate(&self, backend: &dyn Backend, n: usize) -> (f64, usize) {
        let c = &self.config;
        let block = &self.blocks[0];
        let x: Vec<f32> = (0..n * c.dim)
            .map(|i| ((i * 7919) % 2003) as f32 / 2003.0 - 0.5)
            .collect();
        let _ = block.ff.forward(backend, &x[..c.dim], 1);
        let started = std::time::Instant::now();
        let _ = block.ff.forward(backend, &x, n);
        let ff = started.elapsed().as_secs_f64() / n as f64;
        let (d, ad) = (c.dim as f64, c.audio_dim as f64);
        let share = (14.0 * d * d + 4.0 * d * ad) / (8.0 * d * d);
        (ff * share, c.n_layer)
    }

    /// The modulation of one stream for a pass.
    #[allow(clippy::too_many_arguments)]
    fn stream_modulation(
        &self,
        timesteps: &[f32],
        sigma: f32,
        other_sigma: f32,
        adaln: &AdaLayerNorm,
        prompt: &AdaLayerNorm,
        cross_ss: &AdaLayerNorm,
        cross_gate: &AdaLayerNorm,
    ) -> StreamModulation {
        let b = &*self.backend;
        let ts = self.config.timestep_scale;
        let (values, group) = groups(timesteps);
        let scaled: Vec<f32> = values.iter().map(|v| v * ts).collect();
        let (main, embedded) = adaln.forward(b, &scaled);
        let (prompt, _) = prompt.forward(b, &[sigma * ts]);
        let (cross, _) = cross_ss.forward(b, &scaled);
        let (cross_gate, _) = cross_gate.forward(b, &[other_sigma * ts * self.config.av_ca_factor]);
        StreamModulation {
            group,
            main,
            embedded,
            prompt,
            cross,
            cross_gate,
        }
    }

    /// One pass: the velocity of each stream, `[n_video, in_channels]` and
    /// `[n_audio, audio_in_channels]`.
    pub fn forward(
        &self,
        geometry: &Geometry,
        video: &StreamInput<'_>,
        audio: &StreamInput<'_>,
        cancel: Option<&std::sync::atomic::AtomicBool>,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let c = &self.config;
        let b = &*self.backend;
        let (d, ad, eps) = (c.dim, c.audio_dim, c.eps);
        let (nv, na) = (geometry.n_video, geometry.n_audio);
        ensure!(
            video.latent.len() == nv * c.in_channels && video.timesteps.len() == nv,
            "ltxv: the video latent does not match the stage's geometry"
        );
        ensure!(
            audio.latent.len() == na * c.audio_in_channels && audio.timesteps.len() == na,
            "ltxv: the audio latent does not match the stage's geometry"
        );
        let n_ctx = video.context.len() / d;
        let n_actx = audio.context.len() / ad;

        let mut x = self.patchify.forward(b, video.latent, nv);
        if let Some(k) = &self.keyframes {
            for row in x.chunks_mut(d).take(geometry.keyframe_tokens) {
                for (v, k) in row.iter_mut().zip(k) {
                    *v += k;
                }
            }
        }
        let mut a = self.audio_patchify.forward(b, audio.latent, na);

        let vm = self.stream_modulation(
            video.timesteps,
            video.sigma,
            audio.sigma,
            &self.adaln,
            &self.prompt_adaln,
            &self.video_cross_ss,
            &self.a2v_gate,
        );
        let am = self.stream_modulation(
            audio.timesteps,
            audio.sigma,
            video.sigma,
            &self.audio_adaln,
            &self.audio_prompt_adaln,
            &self.audio_cross_ss,
            &self.v2a_gate,
        );
        let single = [0usize];
        let ones_v = vec![0usize; n_ctx];
        let ones_a = vec![0usize; n_actx];

        for block in &self.blocks {
            if super::cancelled(cancel) {
                anyhow::bail!("cancelled");
            }
            let vt = add_table(&block.video.main, &vm.main, 9, d);
            let at = add_table(&block.audio.main, &am.main, 9, ad);

            // Self-attention.
            let mut h = rms_norm_rows(&x, d, eps);
            modulate(
                &mut h,
                d,
                &pick(&vt, 9, d, 0),
                &pick(&vt, 9, d, 1),
                &vm.group,
            );
            let y = block
                .attn1
                .self_attend(b, &h, nv, Some(&geometry.video), eps);
            gated_add(&mut x, &y, d, &pick(&vt, 9, d, 2), &vm.group);
            let mut h = rms_norm_rows(&a, ad, eps);
            modulate(
                &mut h,
                ad,
                &pick(&at, 9, ad, 0),
                &pick(&at, 9, ad, 1),
                &am.group,
            );
            let y = block
                .audio_attn1
                .self_attend(b, &h, na, Some(&geometry.audio), eps);
            gated_add(&mut a, &y, ad, &pick(&at, 9, ad, 2), &am.group);

            // Cross-attention to the prompt.
            let text = |stream: &[f32],
                        n: usize,
                        dim: usize,
                        t: &[f32],
                        m: &StreamModulation,
                        tables: &Tables,
                        attn: &Attention,
                        ctx: &[f32],
                        n_ctx: usize,
                        ctx_group: &[usize]|
             -> Vec<f32> {
                let mut q = rms_norm_rows(stream, dim, eps);
                modulate(
                    &mut q,
                    dim,
                    &pick(t, 9, dim, 6),
                    &pick(t, 9, dim, 7),
                    &m.group,
                );
                let kv_mod = add_table(&tables.prompt, &m.prompt, 2, dim);
                let mut ctx = ctx.to_vec();
                modulate(&mut ctx, dim, &kv_mod[..dim], &kv_mod[dim..], ctx_group);
                let kv = attn.key_values(b, &ctx, n_ctx, None, eps);
                attn.attend(b, &q, n, &kv, None, eps)
            };
            let y = text(
                &x,
                nv,
                d,
                &vt,
                &vm,
                &block.video,
                &block.attn2,
                video.context,
                n_ctx,
                &ones_v,
            );
            gated_add(&mut x, &y, d, &pick(&vt, 9, d, 8), &vm.group);
            let y = text(
                &a,
                na,
                ad,
                &at,
                &am,
                &block.audio,
                &block.audio_attn2,
                audio.context,
                n_actx,
                &ones_a,
            );
            gated_add(&mut a, &y, ad, &pick(&at, 9, ad, 8), &am.group);

            // Audio and video attend to each other, both from where they
            // stood before either was added to.
            if nv > 0 && na > 0 {
                let vc = add_table(&block.video.cross[..4 * d], &vm.cross, 4, d);
                let ac = add_table(&block.audio.cross[..4 * ad], &am.cross, 4, ad);
                let nx = rms_norm_rows(&x, d, eps);
                let nax = rms_norm_rows(&a, ad, eps);
                let scaled = |base: &[f32], dim, table: &[f32], scale_row, m: &StreamModulation| {
                    let mut h = base.to_vec();
                    modulate(
                        &mut h,
                        dim,
                        &pick(table, 4, dim, scale_row + 1),
                        &pick(table, 4, dim, scale_row),
                        &m.group,
                    );
                    h
                };
                let gate = |tables: &Tables, m: &StreamModulation, dim: usize| -> Vec<f32> {
                    tables.cross[4 * dim..5 * dim]
                        .iter()
                        .zip(&m.cross_gate)
                        .map(|(t, g)| t + g)
                        .collect()
                };
                // Audio to video: rows 0 (scale) and 1 (shift).
                let q = scaled(&nx, d, &vc, 0, &vm);
                let ctx = scaled(&nax, ad, &ac, 0, &am);
                let kv: KeyValues =
                    block
                        .a2v
                        .key_values(b, &ctx, na, Some(&geometry.audio_cross), eps);
                let y_a2v = block
                    .a2v
                    .attend(b, &q, nv, &kv, Some(&geometry.video_cross), eps);
                // Video to audio: rows 2 (scale) and 3 (shift).
                let q = scaled(&nax, ad, &ac, 2, &am);
                let ctx = scaled(&nx, d, &vc, 2, &vm);
                let kv = block
                    .v2a
                    .key_values(b, &ctx, nv, Some(&geometry.video_cross), eps);
                let y_v2a = block
                    .v2a
                    .attend(b, &q, na, &kv, Some(&geometry.audio_cross), eps);
                gated_add(
                    &mut x,
                    &y_a2v,
                    d,
                    &gate(&block.video, &vm, d),
                    &single.repeat(nv),
                );
                gated_add(
                    &mut a,
                    &y_v2a,
                    ad,
                    &gate(&block.audio, &am, ad),
                    &single.repeat(na),
                );
            }

            // Feed-forward.
            let mut h = rms_norm_rows(&x, d, eps);
            modulate(
                &mut h,
                d,
                &pick(&vt, 9, d, 3),
                &pick(&vt, 9, d, 4),
                &vm.group,
            );
            let y = block.ff.forward(b, &h, nv);
            gated_add(&mut x, &y, d, &pick(&vt, 9, d, 5), &vm.group);
            let mut h = rms_norm_rows(&a, ad, eps);
            modulate(
                &mut h,
                ad,
                &pick(&at, 9, ad, 3),
                &pick(&at, 9, ad, 4),
                &am.group,
            );
            let y = block.audio_ff.forward(b, &h, na);
            gated_add(&mut a, &y, ad, &pick(&at, 9, ad, 5), &am.group);
        }

        let out = |x: &[f32], n, dim, table: &[f32], m: &StreamModulation, proj: &Linear| {
            let mut h = layer_norm_rows(x, dim, eps);
            let shift = add_table(&table[..dim], &m.embedded, 1, dim);
            let scale = add_table(&table[dim..], &m.embedded, 1, dim);
            modulate(&mut h, dim, &shift, &scale, &m.group);
            proj.forward(b, &h, n)
        };
        Ok((
            out(&x, nv, d, &self.out_table, &vm, &self.proj_out),
            out(&a, na, ad, &self.audio_out_table, &am, &self.audio_proj_out),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_are_grouped_by_their_timestep() {
        let (values, group) = groups(&[0.5, 0.5, 0.0, 0.5]);
        assert_eq!(values, vec![0.5, 0.0]);
        assert_eq!(group, vec![0, 0, 1, 0]);
    }

    #[test]
    fn a_table_row_is_picked_per_group() {
        // Two groups of a 2-row, 2-wide table.
        let sum = add_table(
            &[1.0, 1.0, 2.0, 2.0],
            &[0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 20.0, 20.0],
            2,
            2,
        );
        assert_eq!(pick(&sum, 2, 2, 1), vec![2.0, 2.0, 22.0, 22.0]);
    }
}
