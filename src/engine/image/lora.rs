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

//! A low-rank adapter for the picture transformer — what turns Qwen-Image
//! into Qwen-Image-Lightning.
//!
//! A LoRA adds `scale · (x · Aᵀ) · Bᵀ` to a linear's output: `A` (the
//! *down* projection, `[rank, in]`) and `B` (*up*, `[out, rank]`) are the
//! only new weights, and at rank 64 against widths of 3072–12288 the two
//! small products are a few percent of the linear they ride on. The
//! distillations published as LoRAs (`lightx2v/Qwen-Image-2512-Lightning`)
//! make a picture at 4 or 8 steps without guidance that the base model
//! needs 50 guided steps for — an order of magnitude fewer transformer
//! passes, which is more than every kernel in this tree bought together.
//!
//! The file is `safetensors` in the kohya layout every trainer writes:
//! `<linear>.lora_down.weight`, `<linear>.lora_up.weight` and a scalar
//! `<linear>.alpha`, the linear named as diffusers names it
//! (`transformer_blocks.7.attn.to_q`), with or without a `transformer.`
//! prefix; the diffusers spelling `lora_A`/`lora_B` is read too. The scale
//! is `alpha / rank`, or 1 when no alpha is stored.

use crate::engine::backend::{Backend, MatmulOp};
use crate::engine::loader::QuantMatrix;
use anyhow::{Context, Result, ensure};
use std::collections::HashMap;
use std::path::Path;

/// One linear's adapter: `y += scale · (x · down) · up`.
pub struct LoraPair {
    /// `[rank, in]` as `in`-long rows — `x · downᵀ` is a matmul against it.
    pub down: QuantMatrix,
    /// `[out, rank]` as `rank`-long rows.
    pub up: QuantMatrix,
    pub rank: usize,
    pub scale: f32,
}

impl LoraPair {
    /// The adapter's contribution added into `out` (`[n, out_dim]`) for the
    /// activations `x` (`[n, in_dim]`).
    ///
    /// On the CPU this is one parallel region over blocks of tokens, each
    /// task taking its tokens through both products and adding into its
    /// own rows of `out` — the two products are tiny (rank 64), and run as
    /// two backend matmuls they were two parallel regions of a few
    /// milliseconds each, mostly tails: 21% of a 1024-token pass for 4% of
    /// its arithmetic. Other backends get the two matmuls.
    pub fn apply(&self, backend: &dyn Backend, x: &[f32], n: usize, out: &mut [f32]) {
        #[cfg(target_arch = "aarch64")]
        if backend.as_wgpu().is_none()
            && self.rank.is_multiple_of(crate::engine::vecdot::F32_ROWS)
            && self
                .up
                .out_dim
                .is_multiple_of(crate::engine::vecdot::F32_ROWS)
        {
            self.apply_cpu(x, n, out);
            return;
        }
        let h = backend
            .matmul_batch(&[MatmulOp {
                x,
                n_tokens: n,
                w: &self.down,
            }])
            .pop()
            .expect("one op in, one result out");
        let delta = backend
            .matmul_batch(&[MatmulOp {
                x: &h,
                n_tokens: n,
                w: &self.up,
            }])
            .pop()
            .expect("one op in, one result out");
        debug_assert_eq!(delta.len(), out.len());
        for (o, d) in out.iter_mut().zip(&delta) {
            *o += self.scale * d;
        }
    }
}

#[cfg(target_arch = "aarch64")]
impl LoraPair {
    /// The down product in token blocks (64 rows against a block of
    /// tokens, each task's `h` rows its own), then the up product through
    /// the backend's own two-dimensional float matmul — whose row groups
    /// keep each task's slice of `up` (3 MiB at 12288 × 64) in cache,
    /// where a token-blocked up product streamed the whole of it per
    /// block — and the scaled add.
    fn apply_cpu(&self, x: &[f32], n: usize, out: &mut [f32]) {
        use crate::engine::vecdot::{F32_ROWS, gemm_f32_rows};
        use rayon::prelude::*;
        let (in_dim, rank) = (self.down.in_dim, self.rank);
        debug_assert_eq!(x.len(), n * in_dim);
        // Safety: `from_f32_rows` built the matrix from `f32` rows in a
        // `Vec<f32>`, so the bytes are aligned `f32`s.
        let down: &[f32] = unsafe { self.down.raw_bytes().align_to::<f32>().1 };
        let block = LORA_TOKEN_BLOCK;
        let mut h = vec![0f32; n * rank];
        h.par_chunks_mut(block * rank).enumerate().for_each_init(
            || vec![0f32; rank * block],
            |h_t, (b, dst)| {
                let t0 = b * block;
                let nt = dst.len() / rank;
                let xs = &x[t0 * in_dim..(t0 + nt) * in_dim];
                // `h_t` is `[rank][nt]`, the kernel's own layout,
                // transposed into the token-major `h` below.
                for r0 in (0..rank).step_by(F32_ROWS) {
                    let (y0, rest) = h_t[r0 * nt..].split_at_mut(nt);
                    let (y1, rest) = rest.split_at_mut(nt);
                    let (y2, rest) = rest.split_at_mut(nt);
                    let (y3, _) = rest.split_at_mut(nt);
                    let row = |r: usize| &down[(r0 + r) * in_dim..(r0 + r + 1) * in_dim];
                    gemm_f32_rows(
                        [row(0), row(1), row(2), row(3)],
                        xs,
                        in_dim,
                        [y0, y1, y2, y3],
                    );
                }
                for r in 0..rank {
                    for t in 0..nt {
                        dst[t * rank + r] = h_t[r * nt + t];
                    }
                }
            },
        );
        let delta = crate::engine::backend::CpuBackend.matmul(&h, n, &self.up);
        debug_assert_eq!(delta.len(), out.len());
        out.par_chunks_mut(self.up.out_dim)
            .zip(delta.par_chunks(self.up.out_dim))
            .for_each(|(o, d)| {
                for (o, d) in o.iter_mut().zip(d) {
                    *o += self.scale * d;
                }
            });
    }
}

/// Tokens per [`LoraPair::apply_cpu`] down-product task: a 1,024-token
/// picture is 43 tasks, a 256-token one 11.
#[cfg(target_arch = "aarch64")]
const LORA_TOKEN_BLOCK: usize = 24;

impl LoraPair {
    /// The linear's weights with this adapter folded in — `W + scale · B·A`
    /// — re-encoded in the weights' own type, so the linear runs at its
    /// full speed with nothing to apply per pass. Rows in parallel: each
    /// is dequantized, the low-rank product added (`rank` `axpy`s of
    /// `in_dim`), and encoded again by the same encoder that wrote the
    /// file (`crate::quantize`).
    ///
    /// The re-encoding rounds a second time, which is the cost of merging:
    /// Q4_K's four bits are refit to the merged row rather than the
    /// original, and the adapter's contribution — small beside the weight
    /// — is carried at the weight's precision. The alternative, applying
    /// the adapter in `f32` every pass, was 13–28% of a pass on the CPU;
    /// `[orangu-server].image_lora_merge = no` keeps it for a picture that
    /// wants the adapter exact.
    pub fn merge_into(&self, w: &QuantMatrix, name: &str) -> Result<QuantMatrix> {
        use rayon::prelude::*;
        let ggml_type = w.ggml_type();
        ensure!(
            crate::quantize::row_bytes(ggml_type, w.in_dim) == w.row_bytes()
                && crate::quantize::row_bytes(ggml_type, w.in_dim) > 0,
            "{name}: no encoder for tensor type {ggml_type}, so the adapter cannot be merged"
        );
        let (in_dim, out_dim, rank) = (w.in_dim, w.out_dim, self.rank);
        // Safety: `from_f32_rows` built both from `Vec<f32>`s.
        let down: &[f32] = unsafe { self.down.raw_bytes().align_to::<f32>().1 };
        let up: &[f32] = unsafe { self.up.raw_bytes().align_to::<f32>().1 };
        let raw = w.raw_bytes();
        let row_bytes = w.row_bytes();
        let rows: Vec<Result<Vec<u8>>> = (0..out_dim)
            .into_par_iter()
            .map(|o| {
                let mut row = crate::engine::quant::dequantize(
                    ggml_type,
                    &raw[o * row_bytes..(o + 1) * row_bytes],
                    in_dim,
                )
                .with_context(|| format!("{name}: dequantizing row {o}"))?;
                for r in 0..rank {
                    let b = self.scale * up[o * rank + r];
                    if b == 0.0 {
                        continue;
                    }
                    crate::engine::tensor::axpy_inplace(
                        &mut row,
                        &down[r * in_dim..(r + 1) * in_dim],
                        b,
                    );
                }
                Ok(crate::quantize::encode_chunk(ggml_type, &row))
            })
            .collect();
        let mut bytes = Vec::with_capacity(out_dim * row_bytes);
        for row in rows {
            bytes.extend(row?);
        }
        Ok(QuantMatrix::from_encoded_rows(
            bytes,
            ggml_type,
            in_dim,
            out_dim,
            Some(w),
        ))
    }
}

/// Every adapter in one file, keyed by the linear's diffusers name.
pub struct Lora {
    pairs: HashMap<String, LoraPair>,
}

impl Lora {
    /// Reads every `lora_down`/`lora_up` (or `lora_A`/`lora_B`) pair in the
    /// file, widened to `f32`. A pair whose shapes disagree, or a `down`
    /// without its `up`, is an error: an adapter half-applied is a picture
    /// nobody asked for.
    pub fn open(path: &Path) -> Result<Self> {
        let file = super::safetensors::SafeTensors::open(path)
            .with_context(|| format!("opening the LoRA {}", path.display()))?;
        let mut pairs = HashMap::new();
        for name in file.names() {
            let Some((base, kind)) = split_key(name) else {
                continue;
            };
            if kind != "down" {
                continue;
            }
            let up_name = [
                format!("{base}.lora_up.weight"),
                format!("{base}.lora_B.weight"),
            ]
            .into_iter()
            .find(|n| file.has(n))
            .with_context(|| format!("LoRA {name} has no matching up projection"))?;
            let (down, down_shape) = file.tensor(name)?;
            let (up, up_shape) = file.tensor(&up_name)?;
            ensure!(
                down_shape.len() == 2 && up_shape.len() == 2 && down_shape[0] == up_shape[1],
                "LoRA {base}: down is {down_shape:?}, up is {up_shape:?}"
            );
            let (rank, in_dim, out_dim) = (down_shape[0], down_shape[1], up_shape[0]);
            let alpha = file
                .has(&format!("{base}.alpha"))
                .then(|| file.tensor(&format!("{base}.alpha")))
                .transpose()?
                .and_then(|(v, _)| v.first().copied());
            let scale = alpha.map_or(1.0, |alpha| alpha / rank as f32);
            let linear = base
                .strip_prefix("transformer.")
                .unwrap_or(base)
                .to_string();
            pairs.insert(
                linear,
                LoraPair {
                    down: QuantMatrix::from_f32_rows(down, in_dim, rank),
                    up: QuantMatrix::from_f32_rows(up, rank, out_dim),
                    rank,
                    scale,
                },
            );
        }
        ensure!(!pairs.is_empty(), "{} holds no LoRA pairs", path.display());
        Ok(Self { pairs })
    }

    /// The adapter for `linear` (a diffusers name such as
    /// `transformer_blocks.7.attn.to_q`), if the file has one.
    pub fn take(&mut self, linear: &str) -> Option<LoraPair> {
        self.pairs.remove(linear)
    }

    /// How many pairs are still unclaimed — the ones naming linears this
    /// transformer does not have, worth a warning.
    pub fn remaining(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.pairs.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

/// `transformer_blocks.7.attn.to_q.lora_down.weight` →
/// `("transformer_blocks.7.attn.to_q", "down")`; `lora_A` reads as `down`,
/// `lora_up`/`lora_B` as `up`.
fn split_key(name: &str) -> Option<(&str, &'static str)> {
    for (suffix, kind) in [
        (".lora_down.weight", "down"),
        (".lora_A.weight", "down"),
        (".lora_up.weight", "up"),
        (".lora_B.weight", "up"),
    ] {
        if let Some(base) = name.strip_suffix(suffix) {
            return Some((base, kind));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_split_into_the_linear_and_the_half() {
        assert_eq!(
            split_key("transformer_blocks.7.attn.to_q.lora_down.weight"),
            Some(("transformer_blocks.7.attn.to_q", "down"))
        );
        assert_eq!(
            split_key("transformer.transformer_blocks.0.img_mlp.net.2.lora_B.weight"),
            Some(("transformer.transformer_blocks.0.img_mlp.net.2", "up"))
        );
        assert_eq!(split_key("transformer_blocks.7.attn.to_q.alpha"), None);
    }

    /// A rank-2 adapter on a 3→2 linear, applied by hand: `y += (alpha /
    /// rank) · (x · Aᵀ) · Bᵀ`.
    #[test]
    fn a_pair_adds_the_scaled_low_rank_product() {
        let down = QuantMatrix::from_f32_rows(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 3, 2);
        let up = QuantMatrix::from_f32_rows(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
        let pair = LoraPair {
            down,
            up,
            rank: 2,
            scale: 0.5,
        };
        let x = [1.0f32, 2.0, 3.0];
        let mut out = [10.0f32, 20.0];
        pair.apply(&crate::engine::backend::CpuBackend, &x, 1, &mut out);
        // h = [1, 2]; delta = [1*1 + 2*2, 1*3 + 2*4] = [5, 11]; scaled by 0.5.
        assert!(
            (out[0] - 12.5).abs() < 1e-5 && (out[1] - 25.5).abs() < 1e-5,
            "{out:?}"
        );
    }

    /// A merged matrix answers as the base matrix plus the adapter, within
    /// the weight type's own rounding — here `Q6_K`, six bits, whose error
    /// is small enough to check against a few percent of the largest
    /// output. (`Q4_K` merges the same way; its rounding is the picture's
    /// to judge.)
    #[test]
    fn merging_matches_applying_within_the_weights_rounding() {
        use crate::engine::backend::CpuBackend;
        use crate::engine::loader::test_quant_matrix;
        use crate::engine::quant::GGML_TYPE_Q6_K;
        let (in_dim, out_dim, rank, n) = (256usize, 8usize, 4usize, 3usize);
        let value = |i: usize| ((i * 7 % 13) as f32 - 6.0) * 0.05;
        let base: Vec<f32> = (0..out_dim * in_dim).map(value).collect();
        let bytes = crate::quantize::encode(GGML_TYPE_Q6_K, &base, in_dim);
        let w = test_quant_matrix(&bytes, GGML_TYPE_Q6_K, in_dim, out_dim);
        let pair = LoraPair {
            down: QuantMatrix::from_f32_rows(
                (0..rank * in_dim).map(|i| value(i + 5) * 0.2).collect(),
                in_dim,
                rank,
            ),
            up: QuantMatrix::from_f32_rows(
                (0..out_dim * rank).map(|i| value(i + 9) * 0.2).collect(),
                rank,
                out_dim,
            ),
            rank,
            scale: 1.5,
        };
        let merged = pair.merge_into(&w, "test").unwrap();
        let x: Vec<f32> = (0..n * in_dim).map(|i| value(i + 2)).collect();
        let mut want = CpuBackend.matmul_dequant(&x, n, &w);
        pair.apply(&CpuBackend, &x, n, &mut want);
        let got = CpuBackend.matmul_dequant(&x, n, &merged);
        let scale = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        for (i, (g, e)) in got.iter().zip(&want).enumerate() {
            assert!((g - e).abs() <= 0.03 * scale, "at {i}: {g} vs {e}");
        }
    }

    /// A merged matrix written to the cache comes back byte-identical
    /// through a mapping, keyed by both files' identities: the same pair
    /// hits, a changed adapter misses.
    #[test]
    fn the_merged_cache_round_trips_and_keys_on_both_files() {
        use crate::engine::loader::test_quant_matrix;
        use crate::engine::quant::GGML_TYPE_Q6_K;
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model.gguf");
        let lora = dir.path().join("lora.safetensors");
        std::fs::write(&model, b"model").unwrap();
        std::fs::write(&lora, b"lora").unwrap();
        let (in_dim, out_dim) = (256usize, 3usize);
        let values: Vec<f32> = (0..in_dim * out_dim)
            .map(|i| (i % 7) as f32 * 0.1)
            .collect();
        let bytes = crate::quantize::encode(GGML_TYPE_Q6_K, &values, in_dim);
        let w = test_quant_matrix(&bytes, GGML_TYPE_Q6_K, in_dim, out_dim);
        let merged = QuantMatrix::from_encoded_rows(
            bytes.clone(),
            GGML_TYPE_Q6_K,
            in_dim,
            out_dim,
            Some(&w),
        );

        let mut cache = MergedCache::open(dir.path(), &model, &lora);
        assert!(!cache.is_hit());
        cache.put("blk.to_q", &merged);
        assert!(cache.flush().unwrap() > 0);

        let again = MergedCache::open(dir.path(), &model, &lora);
        assert!(again.is_hit());
        let back = again.get("blk.to_q", &w).expect("cached");
        assert_eq!(back.raw_bytes(), merged.raw_bytes());
        assert!(again.get("blk.to_k", &w).is_none());

        std::fs::write(&lora, b"another lora").unwrap();
        assert!(!MergedCache::open(dir.path(), &model, &lora).is_hit());
    }

    /// The token-blocked CPU path is the two-matmul form to `f32` rounding,
    /// at a rank and width that are multiples of the tile and a token count
    /// that leaves a short last block.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn the_blocked_cpu_path_matches_the_two_matmuls() {
        let (in_dim, rank, out_dim, n) = (24usize, 8usize, 12usize, 53usize);
        let value = |i: usize| ((i * 7 % 13) as f32 - 6.0) * 0.05;
        let pair = LoraPair {
            down: QuantMatrix::from_f32_rows((0..rank * in_dim).map(value).collect(), in_dim, rank),
            up: QuantMatrix::from_f32_rows(
                (0..out_dim * rank).map(|i| value(i + 3)).collect(),
                rank,
                out_dim,
            ),
            rank,
            scale: 0.75,
        };
        let x: Vec<f32> = (0..n * in_dim).map(|i| value(i + 11)).collect();
        let mut fast = vec![1.0f32; n * out_dim];
        pair.apply_cpu(&x, n, &mut fast);
        let backend = crate::engine::backend::CpuBackend;
        let h = backend.matmul(&x, n, &pair.down);
        let delta = backend.matmul(&h, n, &pair.up);
        for (i, (f, d)) in fast.iter().zip(&delta).enumerate() {
            let want = 1.0 + pair.scale * d;
            assert!(
                (f - want).abs() <= 1e-5 * want.abs().max(1.0),
                "at {i}: {f} vs {want}"
            );
        }
    }
}

/// The merged tensors of one (model, adapter) pair kept on disk, so the
/// merge is paid once rather than at every start.
///
/// A file under `<models>/orangu-merged/`, named by a hash of the two
/// files' identities (path, size, modification time) and this format's
/// version: a magic, a JSON header naming every tensor's type, shape and
/// byte range, then the rows exactly as [`LoraPair::merge_into`] encoded
/// them. It is mapped, not read, so a start pays page faults as the base
/// model does rather than a 12 GiB copy — and the merged weights are page
/// cache rather than heap. A file whose header does not match both
/// identities is ignored and rewritten; a file that cannot be written is a
/// warning and a slower start, never a failure.
pub struct MergedCache {
    path: std::path::PathBuf,
    identity: serde_json::Value,
    /// The mapped file and its header, when a matching one was found.
    hit: Option<(std::sync::Arc<memmap2::Mmap>, Vec<MergedEntry>)>,
    /// What this start merged, to be written when the load is done.
    pending: Vec<(MergedEntry, crate::engine::loader::TensorBytes)>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
struct MergedEntry {
    name: String,
    ggml_type: u32,
    in_dim: usize,
    out_dim: usize,
    offset: usize,
    len: usize,
}

const MERGED_MAGIC: &[u8; 8] = b"ORMERGE1";
const MERGED_ALIGN: usize = 64;

impl MergedCache {
    /// Looks for a cached merge of `model` with `lora` under `models_dir`.
    pub fn open(models_dir: &Path, model: &Path, lora: &Path) -> Self {
        let identity = serde_json::json!({
            "version": 1,
            "model": file_identity(model),
            "lora": file_identity(lora),
        });
        use sha2::Digest;
        let digest = sha2::Sha256::digest(identity.to_string().as_bytes());
        let key: String = digest.iter().take(16).map(|b| format!("{b:02x}")).collect();
        let path = models_dir.join("orangu-merged").join(format!("{key}.bin"));
        let hit = Self::map(&path, &identity);
        Self {
            path,
            identity,
            hit,
            pending: Vec::new(),
        }
    }

    fn map(
        path: &Path,
        identity: &serde_json::Value,
    ) -> Option<(std::sync::Arc<memmap2::Mmap>, Vec<MergedEntry>)> {
        let file = std::fs::File::open(path).ok()?;
        // Safety: the file is ours and never written while mapped; a
        // corrupt one fails the checks below rather than being trusted.
        let map = unsafe { memmap2::Mmap::map(&file) }.ok()?;
        if map.len() < 16 || &map[..8] != MERGED_MAGIC {
            return None;
        }
        let header_len = u64::from_le_bytes(map[8..16].try_into().ok()?) as usize;
        let header: serde_json::Value =
            serde_json::from_slice(map.get(16..16 + header_len)?).ok()?;
        if header.get("identity") != Some(identity) {
            return None;
        }
        let entries: Vec<MergedEntry> =
            serde_json::from_value(header.get("tensors")?.clone()).ok()?;
        if entries.iter().any(|e| {
            e.offset
                .checked_add(e.len)
                .is_none_or(|end| end > map.len())
        }) {
            return None;
        }
        Some((std::sync::Arc::new(map), entries))
    }

    /// Whether a matching file was found — the merge can be skipped.
    pub fn is_hit(&self) -> bool {
        self.hit.is_some()
    }

    /// The cached merged matrix for `name`, shaped like `like`.
    pub fn get(&self, name: &str, like: &QuantMatrix) -> Option<QuantMatrix> {
        let (map, entries) = self.hit.as_ref()?;
        let entry = entries.iter().find(|e| e.name == name)?;
        if entry.in_dim != like.in_dim
            || entry.out_dim != like.out_dim
            || entry.ggml_type != like.ggml_type()
        {
            return None;
        }
        Some(QuantMatrix::from_shared_bytes(
            map.clone(),
            entry.offset,
            entry.len,
            entry.ggml_type,
            entry.in_dim,
            entry.out_dim,
            like,
        ))
    }

    /// Records a matrix this start merged, for [`Self::flush`].
    pub fn put(&mut self, name: &str, merged: &QuantMatrix) {
        self.pending.push((
            MergedEntry {
                name: name.to_string(),
                ggml_type: merged.ggml_type(),
                in_dim: merged.in_dim,
                out_dim: merged.out_dim,
                offset: 0,
                len: merged.raw_bytes().len(),
            },
            merged.shared_bytes(),
        ));
    }

    /// Writes what was merged, to a temporary file renamed into place, and
    /// returns the bytes written. Nothing pending writes nothing.
    pub fn flush(&mut self) -> Result<u64> {
        use std::io::Write;
        if self.pending.is_empty() {
            return Ok(0);
        }
        let dir = self.path.parent().context("cache path has no parent")?;
        std::fs::create_dir_all(dir)?;
        let mut entries = Vec::with_capacity(self.pending.len());
        let mut header_probe = serde_json::json!({ "identity": self.identity, "tensors": [] });
        // Two passes over the header: its length decides where the data
        // starts, and the offsets in it depend on that.
        let mut data_start: usize = 0;
        for _ in 0..2 {
            entries.clear();
            let mut offset = data_start;
            for (entry, bytes) in &self.pending {
                offset = offset.div_ceil(MERGED_ALIGN) * MERGED_ALIGN;
                entries.push(MergedEntry {
                    offset,
                    len: bytes.len(),
                    ..entry.clone()
                });
                offset += bytes.len();
            }
            header_probe["tensors"] = serde_json::to_value(&entries)?;
            let header_len = serde_json::to_vec(&header_probe)?.len();
            data_start = (16 + header_len).div_ceil(MERGED_ALIGN) * MERGED_ALIGN;
        }
        let header = serde_json::to_vec(&header_probe)?;
        let tmp = self.path.with_extension("part");
        let mut file = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        file.write_all(MERGED_MAGIC)?;
        file.write_all(&(header.len() as u64).to_le_bytes())?;
        file.write_all(&header)?;
        let mut written = 16 + header.len();
        for (entry, (_, bytes)) in entries.iter().zip(&self.pending) {
            let pad = entry.offset - written;
            file.write_all(&vec![0u8; pad])?;
            file.write_all(&bytes[..entry.len])?;
            written = entry.offset + bytes.len();
        }
        file.flush()?;
        drop(file);
        std::fs::rename(&tmp, &self.path)?;
        self.pending.clear();
        Ok(written as u64)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// What identifies a file for the cache key: where it is, how big it is,
/// and when it last changed — a re-downloaded model or adapter changes at
/// least one.
fn file_identity(path: &Path) -> serde_json::Value {
    let meta = std::fs::metadata(path).ok();
    serde_json::json!({
        "path": path.display().to_string(),
        "len": meta.as_ref().map(|m| m.len()),
        "mtime": meta
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs()),
    })
}
