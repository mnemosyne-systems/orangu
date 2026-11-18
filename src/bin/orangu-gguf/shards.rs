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

//! Shard sets and metadata.
//!
//! A model too large for one file is a shard set:
//! `<prefix>-00001-of-0000N.gguf` … `<prefix>-0000N-of-0000N.gguf`. The first
//! shard carries every metadata key; each shard carries `split.no` (from
//! 0), `split.count` and `split.tensors.count`, and its own share of the
//! tensors.
//!
//! [`Source`] reads a single file or a whole set as one model. [`groups`]
//! decides which tensors a written set puts in which shard, [`copy`] writes
//! tensors through byte for byte, and [`edit`] changes metadata — in place
//! when the new header fits in the space the old one took.

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use orangu::gguf::{GgufFile, GgufValue, TensorInfo};
use orangu::gguf_write as write;
use std::{
    fs::{self, File},
    io::{BufWriter, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

const SPLIT_NO: &str = "split.no";
const SPLIT_COUNT: &str = "split.count";
const SPLIT_TENSORS: &str = "split.tensors.count";

/// A model file, or every shard of a set, read as one.
pub struct Source {
    pub paths: Vec<PathBuf>,
    /// The first shard's keys, without the `split.*` ones.
    pub metadata: Vec<(String, GgufValue)>,
    pub alignment: u64,
    /// Every tensor, shard by shard in file order.
    pub tensors: Vec<TensorInfo>,
    /// The shard each tensor is stored in.
    pub shard_of: Vec<usize>,
    /// Each tensor's stored bytes: absolute start and length in its shard.
    extents: Vec<(usize, usize)>,
    maps: Vec<Mmap>,
    first: GgufFile,
}

impl Source {
    pub fn open(path: &Path) -> Result<Self> {
        let first = GgufFile::open(path).with_context(|| format!("reading {}", path.display()))?;
        let paths = orangu::engine::loader::shard_paths(path, &first)?;
        let mut tensors = Vec::new();
        let mut shard_of = Vec::new();
        let mut extents = Vec::new();
        let mut maps = Vec::new();
        for (s, shard) in paths.iter().enumerate() {
            let file = if s == 0 {
                None
            } else {
                Some(
                    GgufFile::open(shard)
                        .with_context(|| format!("reading {}", shard.display()))?,
                )
            };
            let file_ref = file.as_ref().unwrap_or(&first);
            let handle =
                File::open(shard).with_context(|| format!("opening {}", shard.display()))?;
            // Safety: opened read-only; this process never writes to it.
            let map = unsafe { Mmap::map(&handle) }
                .with_context(|| format!("mapping {}", shard.display()))?;
            let base = file_ref.data_offset;
            let mut offsets: Vec<u64> = file_ref.tensors.iter().map(|t| t.offset).collect();
            offsets.sort_unstable();
            let end = (map.len() as u64).saturating_sub(base);
            for t in &file_ref.tensors {
                // A type this build cannot size runs to the next tensor,
                // which includes the padding: zeros, and harmless to copy.
                let next = offsets
                    .iter()
                    .find(|&&o| o > t.offset)
                    .copied()
                    .unwrap_or(end);
                let len = orangu::dequantize::tensor_byte_size(t.ggml_type, t.element_count())
                    .unwrap_or(next - t.offset);
                let start = base + t.offset;
                if start + len > map.len() as u64 {
                    bail!("{} is cut short inside {}", shard.display(), t.name);
                }
                extents.push((start as usize, len as usize));
                tensors.push(t.clone());
                shard_of.push(s);
            }
            maps.push(map);
        }
        let metadata = first
            .metadata
            .iter()
            .filter(|(k, _)| !k.starts_with("split."))
            .cloned()
            .collect();
        Ok(Source {
            paths,
            metadata,
            alignment: first.alignment,
            tensors,
            shard_of,
            extents,
            maps,
            first,
        })
    }

    pub fn is_split(&self) -> bool {
        self.paths.len() > 1
    }

    /// Tensor `i`'s stored bytes.
    pub fn bytes(&self, i: usize) -> &[u8] {
        let (start, len) = self.extents[i];
        &self.maps[self.shard_of[i]][start..start + len]
    }

    /// Tensor `i`'s values. A quantized tensor is read only with
    /// `requantize`; without it, only the float types are.
    pub fn values(&self, i: usize, requantize: bool) -> Result<Vec<f32>> {
        let tensor = &self.tensors[i];
        let elements = tensor.element_count() as usize;
        if requantize {
            orangu::dequantize::dequantize(tensor.ggml_type, self.bytes(i), elements)
        } else {
            crate::quant::decode(tensor.ggml_type, self.bytes(i), elements)
        }
        .with_context(|| tensor.name.clone())
    }

    /// The index of the tensor called `name`.
    pub fn index_of(&self, name: &str) -> Option<usize> {
        self.tensors.iter().position(|t| t.name == name)
    }

    /// The model as one file's header: every tensor, the set's metadata.
    /// Offsets are each shard's own.
    pub fn as_file(&self) -> GgufFile {
        GgufFile {
            version: self.first.version,
            metadata: self.metadata.clone(),
            tensors: self.tensors.clone(),
            alignment: self.alignment,
            data_offset: self.first.data_offset,
        }
    }

    /// The tensors of each source shard, as indices.
    pub fn shard_groups(&self) -> Vec<Vec<usize>> {
        let mut out = vec![Vec::new(); self.paths.len()];
        for (i, &s) in self.shard_of.iter().enumerate() {
            out[s].push(i);
        }
        out
    }
}

/// How large a written shard may be.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Split {
    pub max_bytes: Option<u64>,
    pub max_tensors: Option<usize>,
}

impl Split {
    pub fn is_set(&self) -> bool {
        self.max_bytes.is_some() || self.max_tensors.is_some()
    }
}

/// Tensors of `sizes` bytes (padding included), in order, cut into shards
/// no larger than `split` allows, the first shard's header taking `first`
/// bytes and every other's `other`. A tensor larger than the byte limit has
/// a shard to itself.
pub fn groups(sizes: &[u64], split: &Split, (first, other): (u64, u64)) -> Vec<Vec<usize>> {
    let mut out: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut bytes = first;
    for (i, &size) in sizes.iter().enumerate() {
        let over_bytes = split.max_bytes.is_some_and(|m| bytes + size > m);
        let over_count = split.max_tensors.is_some_and(|m| current.len() >= m);
        if !current.is_empty() && (over_bytes || over_count) {
            out.push(std::mem::take(&mut current));
            bytes = other;
        }
        current.push(i);
        bytes += size;
    }
    if !current.is_empty() || out.is_empty() {
        out.push(current);
    }
    out
}

/// What a shard's header can take: the first shard's, carrying every key,
/// and any other's, carrying the split keys. Both are bounded by every
/// tensor's record, since a shard holds no more than all of them.
pub fn overheads(
    metadata: &[(String, GgufValue)],
    tensors: &[(&str, &[u64], u32)],
    alignment: u64,
) -> Result<(u64, u64)> {
    let entries: Vec<write::HeaderTensor<'_>> = tensors
        .iter()
        .map(|&(name, dims, ggml_type)| write::HeaderTensor {
            name,
            dims,
            ggml_type,
            offset: 0,
        })
        .collect();
    let split = shard_metadata(&[], 1, 2, tensors.len());
    let mut keys = metadata.to_vec();
    keys.extend(split.iter().cloned());
    let first = write::header(&keys, &entries)?.len() as u64 + alignment;
    let other = write::header(&split, &entries)?.len() as u64 + alignment;
    Ok((first, other))
}

/// Shard `no` (from 0) of `count` written to `output`:
/// `<stem>-0000{no+1}-of-0000{count}.gguf`, or `output` itself for one.
pub fn shard_path(output: &Path, no: usize, count: usize) -> PathBuf {
    if count <= 1 {
        return output.to_path_buf();
    }
    let stem = output
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| n.strip_suffix(".gguf").unwrap_or(n))
        .unwrap_or("model");
    output.with_file_name(format!("{stem}-{:05}-of-{count:05}.gguf", no + 1))
}

/// A shard set's first file name with its numbering taken off:
/// `m-00001-of-00003.gguf` is `m`.
pub fn set_stem(path: &Path) -> String {
    let stem = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model")
        .to_string();
    match stem.rsplit_once("-of-") {
        Some((head, tail)) if tail.len() == 5 && tail.bytes().all(|b| b.is_ascii_digit()) => {
            match head.rsplit_once('-') {
                Some((prefix, no)) if no.len() == 5 && no.bytes().all(|b| b.is_ascii_digit()) => {
                    prefix.to_string()
                }
                _ => stem,
            }
        }
        _ => stem,
    }
}

/// Shard `no`'s keys: all of `metadata` on the first, the `split.*` keys
/// on every one. A single file has no `split.*` keys.
pub fn shard_metadata(
    metadata: &[(String, GgufValue)],
    no: usize,
    count: usize,
    tensors: usize,
) -> Vec<(String, GgufValue)> {
    let mut out: Vec<(String, GgufValue)> = if no == 0 {
        metadata
            .iter()
            .filter(|(k, _)| !k.starts_with("split."))
            .cloned()
            .collect()
    } else {
        Vec::new()
    };
    if count > 1 {
        out.push((SPLIT_NO.into(), GgufValue::U16(no as u16)));
        out.push((SPLIT_COUNT.into(), GgufValue::U16(count as u16)));
        out.push((SPLIT_TENSORS.into(), GgufValue::I32(tensors as i32)));
    }
    out
}

/// Writes `source`'s tensors byte for byte into the shards `groups` makes
/// of them, with `metadata` on the first: the bytes written.
pub fn copy(
    source: &Source,
    output: &Path,
    metadata: &[(String, GgufValue)],
    groups: &[Vec<usize>],
) -> Result<u64> {
    let count = groups.len();
    if count > u16::MAX as usize {
        bail!("{count} shards is more than a shard set can number");
    }
    let mut alignment_keys = metadata.to_vec();
    if source.alignment != 32 && !metadata.iter().any(|(k, _)| k == "general.alignment") {
        alignment_keys.push((
            "general.alignment".into(),
            GgufValue::U32(source.alignment as u32),
        ));
    }
    let mut total = 0u64;
    for (no, group) in groups.iter().enumerate() {
        let path = shard_path(output, no, count);
        for input in &source.paths {
            if same_file(&path, input) {
                bail!("that would overwrite {} while reading it", input.display());
            }
        }
        let keys = shard_metadata(&alignment_keys, no, count, source.tensors.len());
        // Every shard is aligned as the source is; only the first says so.
        let keys = if no == 0 || source.alignment == 32 {
            keys
        } else {
            let mut k = keys;
            k.push((
                "general.alignment".into(),
                GgufValue::U32(source.alignment as u32),
            ));
            k
        };
        let tensors: Vec<write::RawTensor<'_>> = group
            .iter()
            .map(|&i| write::RawTensor {
                name: &source.tensors[i].name,
                dims: &source.tensors[i].dims,
                ggml_type: source.tensors[i].ggml_type,
                bytes: source.bytes(i),
            })
            .collect();
        total += write::write_raw(&path, &keys, &tensors, source.alignment, &|n, all, name| {
            print!("\r  {} {}/{all} {name}    ", path.display(), n + 1);
            let _ = std::io::stdout().flush();
        })?;
        println!();
    }
    Ok(total)
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (fs::canonicalize(a), fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

/// One change to a file's metadata.
#[derive(Debug, Clone)]
pub enum Edit {
    Set(String, Value),
    Delete(String),
}

/// A value as the command line gives it: typed, or `int`/`float`, which
/// take the existing key's type.
#[derive(Debug, Clone)]
pub enum Value {
    Typed(GgufValue),
    Int(i64),
    Float(f64),
}

/// `key=type:value`, where `type` is `u8`…`u64`, `i8`…`i64`, `f32`, `f64`,
/// `bool`, `str`, or `int`/`float` for the key's own type.
pub fn parse_set(text: &str) -> Result<Edit> {
    let (key, rest) = text
        .split_once('=')
        .with_context(|| format!("{text:?} is not key=type:value"))?;
    let (kind, value) = rest
        .split_once(':')
        .with_context(|| format!("{text:?} is not key=type:value"))?;
    let key = key.trim();
    if key.is_empty() {
        bail!("{text:?} names no key");
    }
    let bad = || anyhow::anyhow!("{value:?} is not a {kind}");
    let v = match kind.trim().to_ascii_lowercase().as_str() {
        "u8" => Value::Typed(GgufValue::U8(value.parse().map_err(|_| bad())?)),
        "i8" => Value::Typed(GgufValue::I8(value.parse().map_err(|_| bad())?)),
        "u16" => Value::Typed(GgufValue::U16(value.parse().map_err(|_| bad())?)),
        "i16" => Value::Typed(GgufValue::I16(value.parse().map_err(|_| bad())?)),
        "u32" => Value::Typed(GgufValue::U32(value.parse().map_err(|_| bad())?)),
        "i32" => Value::Typed(GgufValue::I32(value.parse().map_err(|_| bad())?)),
        "u64" => Value::Typed(GgufValue::U64(value.parse().map_err(|_| bad())?)),
        "i64" => Value::Typed(GgufValue::I64(value.parse().map_err(|_| bad())?)),
        "f32" => Value::Typed(GgufValue::F32(value.parse().map_err(|_| bad())?)),
        "f64" => Value::Typed(GgufValue::F64(value.parse().map_err(|_| bad())?)),
        "bool" => Value::Typed(GgufValue::Bool(match value {
            "true" | "1" => true,
            "false" | "0" => false,
            _ => return Err(bad()),
        })),
        "str" | "string" => Value::Typed(GgufValue::String(value.to_string())),
        "int" => Value::Int(value.parse().map_err(|_| bad())?),
        "float" => Value::Float(value.parse().map_err(|_| bad())?),
        other => bail!(
            "{other:?} is not a metadata type: one of u8, i8, u16, i16, u32, i32, u64, i64, \
             f32, f64, bool, str, int, float"
        ),
    };
    Ok(Edit::Set(key.to_string(), v))
}

/// `value` in the type of `existing`, when it is a number of another width.
fn resolve(value: &Value, existing: Option<&GgufValue>) -> Result<GgufValue> {
    use GgufValue as G;
    Ok(match (value, existing) {
        (Value::Typed(v), _) => v.clone(),
        (Value::Int(i), Some(e)) => {
            let i = *i;
            let fit = |ok: bool| {
                if ok {
                    Ok(())
                } else {
                    Err(anyhow::anyhow!("{i} does not fit the key's type"))
                }
            };
            match e {
                G::U8(_) => {
                    fit(u8::try_from(i).is_ok())?;
                    G::U8(i as u8)
                }
                G::I8(_) => {
                    fit(i8::try_from(i).is_ok())?;
                    G::I8(i as i8)
                }
                G::U16(_) => {
                    fit(u16::try_from(i).is_ok())?;
                    G::U16(i as u16)
                }
                G::I16(_) => {
                    fit(i16::try_from(i).is_ok())?;
                    G::I16(i as i16)
                }
                G::U32(_) => {
                    fit(u32::try_from(i).is_ok())?;
                    G::U32(i as u32)
                }
                G::I32(_) => {
                    fit(i32::try_from(i).is_ok())?;
                    G::I32(i as i32)
                }
                G::U64(_) => {
                    fit(i >= 0)?;
                    G::U64(i as u64)
                }
                _ => G::I64(i),
            }
        }
        (Value::Int(i), None) => G::I64(*i),
        (Value::Float(f), Some(G::F32(_))) => G::F32(*f as f32),
        (Value::Float(f), _) => G::F64(*f),
    })
}

/// `metadata` with `edits` applied in order. A set replaces the key where
/// it is, or adds it at the end.
pub fn apply(metadata: &mut Vec<(String, GgufValue)>, edits: &[Edit]) -> Result<()> {
    for edit in edits {
        match edit {
            Edit::Set(key, value) => {
                if key.starts_with("split.") {
                    bail!("{key} describes the shard set and is written by the tool");
                }
                let at = metadata.iter().position(|(k, _)| k == key);
                let v = resolve(value, at.map(|i| &metadata[i].1))?;
                match at {
                    Some(i) => metadata[i].1 = v,
                    None => metadata.push((key.clone(), v)),
                }
            }
            Edit::Delete(key) => {
                let before = metadata.len();
                metadata.retain(|(k, _)| k != key);
                if metadata.len() == before {
                    bail!("there is no key {key} to delete");
                }
            }
        }
    }
    Ok(())
}

/// Applies `edits` to the metadata of `path` — the first shard of a set.
/// In place when the new header fits the space the old one took (the
/// header is padded to the alignment, so a small change usually does);
/// otherwise the file is rewritten beside itself and renamed over, the
/// tensor data copied unchanged. Returns whether it was in place.
pub fn edit(path: &Path, edits: &[Edit]) -> Result<bool> {
    let file = GgufFile::open(path).with_context(|| format!("reading {}", path.display()))?;
    let mut metadata = file.metadata.clone();
    apply(&mut metadata, edits)?;
    let entries: Vec<write::HeaderTensor<'_>> = file
        .tensors
        .iter()
        .map(|t| write::HeaderTensor {
            name: &t.name,
            dims: &t.dims,
            ggml_type: t.ggml_type,
            offset: t.offset,
        })
        .collect();
    let header = write::header(&metadata, &entries)?;
    let padded = write::aligned(header.len() as u64, file.alignment);
    if padded == file.data_offset {
        let mut out = fs::OpenOptions::new()
            .write(true)
            .open(path)
            .with_context(|| format!("opening {} to write", path.display()))?;
        out.seek(SeekFrom::Start(0))?;
        out.write_all(&header)?;
        out.write_all(&vec![0u8; (padded - header.len() as u64) as usize])?;
        out.sync_all()?;
        return Ok(true);
    }
    let temporary = path.with_extension("gguf.tmp");
    {
        let mut source = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        source.seek(SeekFrom::Start(file.data_offset))?;
        let mut out = BufWriter::with_capacity(
            4 << 20,
            File::create(&temporary)
                .with_context(|| format!("creating {}", temporary.display()))?,
        );
        out.write_all(&header)?;
        out.write_all(&vec![0u8; (padded - header.len() as u64) as usize])?;
        std::io::copy(&mut source, &mut out)?;
        out.flush()?;
    }
    fs::rename(&temporary, path)
        .with_context(|| format!("renaming {} into place", temporary.display()))?;
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shards_fill_to_their_limit_and_a_large_tensor_stands_alone() {
        let split = Split {
            max_bytes: Some(10),
            max_tensors: None,
        };
        assert_eq!(
            groups(&[4, 4, 4, 30, 2], &split, (0, 0)),
            vec![vec![0, 1], vec![2], vec![3], vec![4]]
        );
        let split = Split {
            max_bytes: None,
            max_tensors: Some(2),
        };
        assert_eq!(
            groups(&[1, 1, 1], &split, (0, 0)),
            vec![vec![0, 1], vec![2]]
        );
        assert_eq!(groups(&[], &split, (0, 0)), vec![Vec::<usize>::new()]);
        // The first shard's header counts against its limit.
        let split = Split {
            max_bytes: Some(10),
            max_tensors: None,
        };
        assert_eq!(groups(&[4, 4], &split, (3, 0)), vec![vec![0], vec![1]]);
    }

    #[test]
    fn shard_names_are_the_ones_the_loader_follows() {
        let out = Path::new("/m/model.gguf");
        assert_eq!(shard_path(out, 0, 1), out);
        assert_eq!(
            shard_path(out, 1, 3),
            Path::new("/m/model-00002-of-00003.gguf")
        );
        assert_eq!(set_stem(Path::new("/m/model-00001-of-00003.gguf")), "model");
        assert_eq!(set_stem(Path::new("/m/model-Q4_K_M.gguf")), "model-Q4_K_M");
    }

    #[test]
    fn a_number_takes_the_type_of_the_key_it_replaces() {
        let mut metadata = vec![
            ("a.count".to_string(), GgufValue::U32(4)),
            ("a.eps".to_string(), GgufValue::F32(1e-5)),
            ("a.name".to_string(), GgufValue::String("x".into())),
        ];
        let edits = vec![
            parse_set("a.count=int:8").unwrap(),
            parse_set("a.eps=float:1e-6").unwrap(),
            parse_set("a.new=str:hello").unwrap(),
            Edit::Delete("a.name".into()),
        ];
        apply(&mut metadata, &edits).unwrap();
        assert_eq!(
            format!("{metadata:?}"),
            format!(
                "{:?}",
                vec![
                    ("a.count".to_string(), GgufValue::U32(8)),
                    ("a.eps".to_string(), GgufValue::F32(1e-6)),
                    ("a.new".to_string(), GgufValue::String("hello".into())),
                ]
            )
        );
        assert!(apply(&mut metadata, &[parse_set("a.count=int:-1").unwrap()]).is_err());
        assert!(apply(&mut metadata, &[Edit::Delete("absent".into())]).is_err());
        assert!(parse_set("a=u32").is_err());
        assert!(parse_set("a=u33:1").is_err());
    }
}
