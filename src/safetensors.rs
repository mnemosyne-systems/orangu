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

//! A reader for the `safetensors` container: diffusion VAEs and adapters
//! as published, and the language-model checkpoints `orangu-gguf` converts.
//!
//! The format is small enough to read directly rather than take a crate
//! for: eight little-endian bytes of header length, a JSON header mapping
//! each tensor name to its `dtype`, `shape` and `[begin, end)` byte range,
//! and the tensor bytes after it, in row-major order.
//!
//! [`SafeTensors::tensor`] widens `F32`, `F16`, `BF16`, `F8_E4M3` and
//! `F8_E5M2` to `f32`; any other type is indexed so the file opens, and
//! read as bytes with [`SafeTensors::bytes`].
//!
//! An eight-bit float weight is read already multiplied by its scales: a
//! `<name>_scale_inv` (one scale per block of the weight, 128x128 as
//! published) or a `<name>_scale` (one for the whole weight, one per row,
//! or one per block). The scales, and the activation scales such a
//! checkpoint carries as `<layer>.input_scale`, are left out of
//! [`SafeTensors::names`], so a reader sees the weights it would see in a
//! checkpoint stored at full width. The file is mapped rather than read: a mapped
//! file costs nothing until a tensor is actually touched.
//!
//! A checkpoint larger than one file is a shard set with a
//! `model.safetensors.index.json` naming which shard holds each tensor;
//! [`SafeTensorsSet`] opens the set as one.

use anyhow::{Context, Result, anyhow, bail};
use memmap2::Mmap;
use rayon::prelude::*;
use std::{collections::HashMap, fs::File, path::Path};

/// One tensor's entry in the header.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    /// Byte range inside the data section (after the header).
    begin: usize,
    end: usize,
}

pub struct SafeTensors {
    map: Mmap,
    data_start: usize,
    tensors: HashMap<String, TensorInfo>,
    /// The header's `__metadata__`: string keys to string values (LTX-2's
    /// components keep their `config` JSON there).
    metadata: HashMap<String, String>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> Result<Self> {
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        // Safety: opened read-only and never written by this process; the
        // same caveat every mapped model file in the loader accepts.
        let map = unsafe { Mmap::map(&file) }
            .with_context(|| format!("failed to mmap {}", path.display()))?;
        Self::parse(map).with_context(|| format!("reading {}", path.display()))
    }

    fn parse(map: Mmap) -> Result<Self> {
        if map.len() < 8 {
            bail!("file is too short to be a safetensors file");
        }
        let header_len =
            u64::from_le_bytes(map[..8].try_into().expect("eight bytes were checked")) as usize;
        let data_start = 8usize
            .checked_add(header_len)
            .filter(|&end| end <= map.len())
            .ok_or_else(|| anyhow!("safetensors header length {header_len} exceeds the file"))?;
        let header: serde_json::Map<String, serde_json::Value> =
            serde_json::from_slice(&map[8..data_start]).context("parsing safetensors header")?;
        let data_len = map.len() - data_start;
        let mut tensors = HashMap::with_capacity(header.len());
        let mut metadata = HashMap::new();
        for (name, entry) in header {
            if name == "__metadata__" {
                if let Some(map) = entry.as_object() {
                    for (k, v) in map {
                        if let Some(s) = v.as_str() {
                            metadata.insert(k.clone(), s.to_string());
                        }
                    }
                }
                continue;
            }
            let dtype = entry["dtype"]
                .as_str()
                .ok_or_else(|| anyhow!("tensor '{name}' has no dtype"))?
                .to_string();
            let shape: Vec<usize> = entry["shape"]
                .as_array()
                .ok_or_else(|| anyhow!("tensor '{name}' has no shape"))?
                .iter()
                .map(|v| {
                    v.as_u64()
                        .map(|d| d as usize)
                        .ok_or_else(|| anyhow!("tensor '{name}' has a non-integer dimension"))
                })
                .collect::<Result<_>>()?;
            let offsets = entry["data_offsets"]
                .as_array()
                .filter(|o| o.len() == 2)
                .ok_or_else(|| anyhow!("tensor '{name}' has no data_offsets"))?;
            let begin = offsets[0].as_u64().unwrap_or(u64::MAX) as usize;
            let end = offsets[1].as_u64().unwrap_or(u64::MAX) as usize;
            if begin > end || end > data_len {
                bail!("tensor '{name}' has byte range {begin}..{end} outside the data section");
            }
            let elements: usize = shape.iter().product();
            let width = match dtype.as_str() {
                "F32" => 4,
                "F16" | "BF16" => 2,
                // Raw bytes: a checkpoint's embedded files (LTX-2's text
                // encoder carries its tokenizer this way), read with `bytes`.
                "U8" | "I8" | "BOOL" => 1,
                // Eight-bit floats, with their scales as separate tensors.
                "F8_E4M3" | "F8_E5M2" => 1,
                // Integer bookkeeping a checkpoint carries beside its
                // weights (a BatchNorm's `num_batches_tracked`): indexed so
                // the file opens, refused by `tensor` if ever read.
                "I16" | "U16" => 2,
                "I32" | "U32" => 4,
                "I64" | "U64" | "F64" => 8,
                other => {
                    bail!("tensor '{name}' is stored as {other}, which this reader does not read")
                }
            };
            if elements * width != end - begin {
                bail!(
                    "tensor '{name}': {} {dtype} elements do not fill {} bytes",
                    elements,
                    end - begin
                );
            }
            tensors.insert(
                name,
                TensorInfo {
                    dtype,
                    shape,
                    begin,
                    end,
                },
            );
        }
        Ok(Self {
            map,
            data_start,
            tensors,
            metadata,
        })
    }

    /// A tensor's raw bytes, whatever its type — for a `U8` tensor that
    /// is an embedded file.
    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("safetensors file is missing tensor '{name}'"))?;
        Ok(&self.map[self.data_start + info.begin..self.data_start + info.end])
    }

    /// A tensor's shape and type, without reading it.
    pub fn info(&self, name: &str) -> Option<(&[usize], &str)> {
        self.tensors
            .get(name)
            .map(|t| (t.shape.as_slice(), t.dtype.as_str()))
    }

    /// A `__metadata__` value.
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Every tensor name in the file, sorted, without an eight-bit float
    /// weight's scales.
    pub fn names(&self) -> Vec<&str> {
        let dtype = |n: &str| self.tensors.get(n).map(|t| t.dtype.as_str());
        let mut names: Vec<&str> = self
            .tensors
            .keys()
            .map(String::as_str)
            .filter(|n| !is_scale_of_fp8(n, &dtype))
            .collect();
        names.sort_unstable();
        names
    }

    /// Tensor `name` widened to `f32`, with its shape; an eight-bit float
    /// weight multiplied by its scales.
    pub fn tensor(&self, name: &str) -> Result<(Vec<f32>, &[usize])> {
        let (mut values, shape, dtype) = self.decoded(name)?;
        if is_fp8(dtype) {
            scale_fp8(&mut values, shape, name, |n| {
                self.has(n)
                    .then(|| self.decoded(n).map(|(v, s, _)| (v, s.to_vec())))
            })?;
        }
        Ok((values, shape))
    }

    /// Tensor `name` widened to `f32` as stored, with its shape and type.
    fn decoded(&self, name: &str) -> Result<(Vec<f32>, &[usize], &str)> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow!("safetensors file is missing tensor '{name}'"))?;
        let bytes = &self.map[self.data_start + info.begin..self.data_start + info.end];
        let values = match info.dtype.as_str() {
            "F32" => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
            "F16" => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half::f16::from_le_bytes(*b).to_f32())
                .collect(),
            "BF16" => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half::bf16::from_le_bytes(*b).to_f32())
                .collect(),
            "F8_E4M3" => {
                let table = e4m3_table();
                bytes.par_iter().map(|&b| table[b as usize]).collect()
            }
            // E5M2 is the top byte of an F16.
            "F8_E5M2" => bytes
                .par_iter()
                .map(|&b| half::f16::from_bits(u16::from(b) << 8).to_f32())
                .collect(),
            other => bail!("tensor '{name}' is stored as {other}"),
        };
        Ok((values, &info.shape, info.dtype.as_str()))
    }
}

fn is_fp8(dtype: &str) -> bool {
    matches!(dtype, "F8_E4M3" | "F8_E5M2")
}

/// Every `F8_E4M3` byte's value: four exponent bits biased by 7, three
/// mantissa bits, no infinities, and `S.1111.111` not a number.
fn e4m3_table() -> &'static [f32; 256] {
    static TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        std::array::from_fn(|b| {
            let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
            let exponent = ((b >> 3) & 0xF) as i32;
            let mantissa = (b & 7) as f32;
            let magnitude = if exponent == 0 {
                mantissa / 8.0 * 2f32.powi(-6)
            } else if exponent == 15 && b & 7 == 7 {
                f32::NAN
            } else {
                (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
            };
            sign * magnitude
        })
    })
}

/// Whether `name` is a scale an eight-bit float weight carries: its
/// `_scale_inv` or `_scale`, or the `input_scale` beside it.
fn is_scale_of_fp8<'a>(name: &str, dtype: &impl Fn(&str) -> Option<&'a str>) -> bool {
    let fp8 = |n: &str| dtype(n).is_some_and(is_fp8);
    if let Some(base) = name
        .strip_suffix("_scale_inv")
        .or_else(|| name.strip_suffix("_scale"))
        && fp8(base)
    {
        return true;
    }
    name.strip_suffix("input_scale")
        .is_some_and(|layer| fp8(&format!("{layer}weight")))
}

/// Multiplies the eight-bit float weight `name`, read into `values` with
/// `shape`, by its scales, which `scale` reads by name: `<name>_scale_inv`
/// or `<name>_scale`, of one value, one per row, or one per block of rows
/// and columns. A weight without either is left as stored.
fn scale_fp8(
    values: &mut [f32],
    shape: &[usize],
    name: &str,
    scale: impl Fn(&str) -> Option<Result<(Vec<f32>, Vec<usize>)>>,
) -> Result<()> {
    let Some(found) =
        scale(&format!("{name}_scale_inv")).or_else(|| scale(&format!("{name}_scale")))
    else {
        return Ok(());
    };
    let (scales, scale_shape) = found?;
    let columns = shape.last().copied().unwrap_or(1).max(1);
    let rows = values.len() / columns;
    let (block_rows, block_columns) = match scale_shape.as_slice() {
        [] | [1] => (1, 1),
        [n] => (*n, 1),
        [a, b] => (*a, *b),
        other => bail!("'{name}' has scales of shape {other:?}, which this reader does not apply"),
    };
    if block_rows * block_columns != scales.len() || block_rows == 0 || block_columns == 0 {
        bail!(
            "'{name}' has {} scales for a {block_rows}x{block_columns} grid",
            scales.len()
        );
    }
    let per_row = rows.div_ceil(block_rows);
    let per_column = columns.div_ceil(block_columns);
    values
        .par_chunks_mut(columns)
        .enumerate()
        .for_each(|(r, row)| {
            let grid = &scales[(r / per_row) * block_columns..][..block_columns];
            for (c, v) in row.iter_mut().enumerate() {
                *v *= grid[c / per_column];
            }
        });
    Ok(())
}

/// A checkpoint split over several files, opened as one.
///
/// Built from `model.safetensors.index.json` when the directory has one, and
/// otherwise from every `*.safetensors` file in it. A tensor named by two
/// shards is an error: which one a converter read would depend on the order
/// the directory listed them in.
pub struct SafeTensorsSet {
    shards: Vec<SafeTensors>,
    owner: HashMap<String, usize>,
}

impl SafeTensorsSet {
    pub fn open_dir(dir: &Path) -> Result<Self> {
        let index = dir.join("model.safetensors.index.json");
        let mut files: Vec<std::path::PathBuf> = if index.is_file() {
            let text = std::fs::read_to_string(&index)
                .with_context(|| format!("reading {}", index.display()))?;
            let json: serde_json::Value = serde_json::from_str(&text)
                .with_context(|| format!("parsing {}", index.display()))?;
            let map = json["weight_map"]
                .as_object()
                .ok_or_else(|| anyhow!("{} has no weight_map", index.display()))?;
            let mut names: Vec<String> = map
                .values()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect();
            names.sort();
            names.dedup();
            names.into_iter().map(|n| dir.join(n)).collect()
        } else {
            std::fs::read_dir(dir)
                .with_context(|| format!("listing {}", dir.display()))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
                .collect()
        };
        files.sort();
        if files.is_empty() {
            bail!("{} holds no .safetensors files", dir.display());
        }
        let shards = files
            .iter()
            .map(|f| SafeTensors::open(f))
            .collect::<Result<Vec<_>>>()?;
        let mut owner = HashMap::new();
        for (i, shard) in shards.iter().enumerate() {
            for name in shard.tensors.keys() {
                if owner.insert(name.clone(), i).is_some() {
                    bail!(
                        "tensor '{name}' is in more than one shard of {}",
                        dir.display()
                    );
                }
            }
        }
        Ok(Self { shards, owner })
    }

    fn shard(&self, name: &str) -> Result<&SafeTensors> {
        self.owner
            .get(name)
            .map(|&i| &self.shards[i])
            .ok_or_else(|| anyhow!("the checkpoint has no tensor '{name}'"))
    }

    pub fn has(&self, name: &str) -> bool {
        self.owner.contains_key(name)
    }

    /// Every tensor name in the set, sorted, without an eight-bit float
    /// weight's scales.
    pub fn names(&self) -> Vec<&str> {
        let dtype = |n: &str| self.info(n).map(|(_, d)| d);
        let mut names: Vec<&str> = self
            .owner
            .keys()
            .map(String::as_str)
            .filter(|n| !is_scale_of_fp8(n, &dtype))
            .collect();
        names.sort_unstable();
        names
    }

    /// Where `name` comes in the checkpoint's own order: shard by shard,
    /// and by name within a shard.
    pub fn position(&self, name: &str) -> Option<(usize, &str)> {
        let (key, &shard) = self.owner.get_key_value(name)?;
        Some((shard, key.as_str()))
    }

    pub fn info(&self, name: &str) -> Option<(&[usize], &str)> {
        self.shard(name).ok()?.info(name)
    }

    pub fn bytes(&self, name: &str) -> Result<&[u8]> {
        self.shard(name)?.bytes(name)
    }

    /// Tensor `name` widened to `f32`, with its shape; an eight-bit float
    /// weight multiplied by its scales, from whichever shard holds them.
    pub fn tensor(&self, name: &str) -> Result<(Vec<f32>, &[usize])> {
        let (mut values, shape, dtype) = self.shard(name)?.decoded(name)?;
        if is_fp8(dtype) {
            scale_fp8(&mut values, shape, name, |n| {
                self.has(n)
                    .then(|| self.shard(n)?.decoded(n).map(|(v, s, _)| (v, s.to_vec())))
            })?;
        }
        Ok((values, shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Writes a two-tensor file by hand, so the reader is checked against the
    /// format's specification rather than against itself.
    fn write_fixture(path: &Path) {
        let a: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let b: Vec<u8> = [0.5f32, -1.5]
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect();
        let header = format!(
            "{{\"__metadata__\":{{\"format\":\"pt\"}},\
             \"a\":{{\"dtype\":\"F32\",\"shape\":[2,3],\"data_offsets\":[0,{}]}},\
             \"b\":{{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[{},{}]}}}}",
            a.len(),
            a.len(),
            a.len() + b.len()
        );
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        file.write_all(&a).unwrap();
        file.write_all(&b).unwrap();
    }

    #[test]
    fn reads_f32_and_bf16_tensors_with_their_shapes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.safetensors");
        write_fixture(&path);
        let file = SafeTensors::open(&path).unwrap();
        let (a, shape) = file.tensor("a").unwrap();
        assert_eq!(shape, &[2, 3]);
        assert_eq!(a, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let (b, shape) = file.tensor("b").unwrap();
        assert_eq!(shape, &[2]);
        assert_eq!(b, vec![0.5, -1.5]);
        assert!(file.has("a"));
        assert!(!file.has("__metadata__"));
        assert!(file.tensor("c").is_err());
    }

    #[test]
    fn a_directory_of_shards_opens_as_one_and_refuses_a_duplicate() {
        let dir = tempfile::tempdir().unwrap();
        write_fixture(&dir.path().join("model-00001-of-00002.safetensors"));
        let header = "{\"c\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}";
        let mut file = File::create(dir.path().join("model-00002-of-00002.safetensors")).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        file.write_all(&7.0f32.to_le_bytes()).unwrap();
        drop(file);
        let set = SafeTensorsSet::open_dir(dir.path()).unwrap();
        assert_eq!(set.names(), vec!["a", "b", "c"]);
        assert_eq!(set.tensor("c").unwrap().0, vec![7.0]);
        write_fixture(&dir.path().join("model-00003-of-00003.safetensors"));
        assert!(SafeTensorsSet::open_dir(dir.path()).is_err());
    }

    /// Writes `tensors` (name, type, shape, bytes) as one file.
    fn write_tensors(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<u8>)]) {
        let mut entries = Vec::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            entries.push(format!(
                "\"{name}\":{{\"dtype\":\"{dtype}\",\"shape\":{shape:?},\"data_offsets\":[{},{}]}}",
                data.len(),
                data.len() + bytes.len()
            ));
            data.extend_from_slice(bytes);
        }
        let header = format!("{{{}}}", entries.join(","));
        let mut file = File::create(path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        file.write_all(&data).unwrap();
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// Every E4M3 and E5M2 byte decodes to the value the format defines,
    /// checked against the definition written out by hand.
    #[test]
    fn eight_bit_floats_decode_by_their_definition() {
        let t = e4m3_table();
        assert_eq!(t[0x00], 0.0);
        assert_eq!(t[0x38], 1.0); // exponent 7, mantissa 0
        assert_eq!(t[0x3C], 1.5);
        assert_eq!(t[0xB8], -1.0);
        assert_eq!(t[0x7E], 448.0); // the largest
        assert_eq!(t[0x01], 2f32.powi(-9)); // the smallest subnormal
        assert_eq!(t[0x08], 2f32.powi(-6)); // the smallest normal
        assert!(t[0x7F].is_nan() && t[0xFF].is_nan());
        let e5m2 = |b: u8| half::f16::from_bits(u16::from(b) << 8).to_f32();
        assert_eq!(e5m2(0x3C), 1.0);
        assert_eq!(e5m2(0x7B), 57344.0);
        assert_eq!(e5m2(0xC0), -2.0);
    }

    /// An eight-bit float weight reads multiplied by its block scales, in
    /// one file and across shards, and its scales are not listed.
    #[test]
    fn an_fp8_weight_reads_scaled_and_its_scales_are_hidden() {
        let dir = tempfile::tempdir().unwrap();
        // A 3x4 weight of 1.0 (0x38), scaled in 2x2 blocks of a 2x2 grid:
        // rows 0-1 and row 2, columns 0-1 and 2-3.
        let weight = vec![0x38u8; 12];
        let blocks = f32_bytes(&[1.0, 2.0, 3.0, 4.0]);
        write_tensors(
            &dir.path().join("model-00001-of-00002.safetensors"),
            &[
                ("l.weight", "F8_E4M3", vec![3, 4], weight.clone()),
                ("l.input_scale", "F32", vec![], f32_bytes(&[0.5])),
                ("r.weight", "F8_E5M2", vec![2, 2], vec![0x3C; 4]),
            ],
        );
        write_tensors(
            &dir.path().join("model-00002-of-00002.safetensors"),
            &[
                ("l.weight_scale_inv", "F32", vec![2, 2], blocks),
                ("r.weight_scale", "F32", vec![2, 1], f32_bytes(&[3.0, 5.0])),
            ],
        );
        let set = SafeTensorsSet::open_dir(dir.path()).unwrap();
        assert_eq!(set.names(), vec!["l.weight", "r.weight"]);
        let (l, shape) = set.tensor("l.weight").unwrap();
        assert_eq!(shape, &[3, 4]);
        assert_eq!(
            l,
            vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]
        );
        assert_eq!(set.tensor("r.weight").unwrap().0, vec![3.0, 3.0, 5.0, 5.0]);

        // In one file, a single scale for the whole weight.
        let one = dir.path().join("one.safetensors");
        write_tensors(
            &one,
            &[
                ("w", "F8_E4M3", vec![2], vec![0x38, 0x3C]),
                ("w_scale", "F32", vec![1], f32_bytes(&[2.0])),
            ],
        );
        let file = SafeTensors::open(&one).unwrap();
        assert_eq!(file.names(), vec!["w"]);
        assert_eq!(file.tensor("w").unwrap().0, vec![2.0, 3.0]);
    }

    #[test]
    fn a_range_past_the_data_section_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.safetensors");
        let header = "{\"a\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}}";
        let mut file = File::create(&path).unwrap();
        file.write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        file.write_all(header.as_bytes()).unwrap();
        file.write_all(&[0u8; 8]).unwrap();
        assert!(SafeTensors::open(&path).is_err());
    }
}
