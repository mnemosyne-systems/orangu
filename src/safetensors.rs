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
//! [`SafeTensors::tensor`] widens `F32`, `F16` and `BF16` to `f32`; any
//! other type is indexed so the file opens, and read as bytes with
//! [`SafeTensors::bytes`]. The file is mapped rather than read: a mapped
//! file costs nothing until a tensor is actually touched.
//!
//! A checkpoint larger than one file is a shard set with a
//! `model.safetensors.index.json` naming which shard holds each tensor;
//! [`SafeTensorsSet`] opens the set as one.

use anyhow::{Context, Result, anyhow, bail};
use memmap2::Mmap;
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

    /// Every tensor name in the file, sorted.
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.tensors.keys().map(String::as_str).collect();
        names.sort_unstable();
        names
    }

    /// Tensor `name` widened to `f32`, with its shape.
    pub fn tensor(&self, name: &str) -> Result<(Vec<f32>, &[usize])> {
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
            other => bail!("tensor '{name}' is stored as {other}"),
        };
        Ok((values, &info.shape))
    }
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

    /// Every tensor name in the set, sorted.
    pub fn names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.owner.keys().map(String::as_str).collect();
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

    pub fn tensor(&self, name: &str) -> Result<(Vec<f32>, &[usize])> {
        self.shard(name)?.tensor(name)
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
