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

//! A reader for the `safetensors` container the Qwen-Image VAE ships in.
//!
//! The format is small enough to read directly rather than take a crate
//! for: eight little-endian bytes of header length, a JSON header mapping
//! each tensor name to its `dtype`, `shape` and `[begin, end)` byte range,
//! and the tensor bytes after it, in row-major order. Every diffusion VAE
//! published for ComfyUI or stable-diffusion.cpp is one of these; nobody
//! converts the VAE to GGUF, so the engine reads it as published.
//!
//! Only `F32`, `F16` and `BF16` are read — the three a VAE is ever stored
//! in — and every tensor comes back as `f32`, since the convolutions run
//! through `engine::backend` as `F32` matrices and the norms want floats
//! anyway. The file is mapped rather than read: 254 MiB is not much, but a
//! mapped file costs nothing until a tensor is actually touched.

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
