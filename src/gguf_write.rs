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

//! The GGUF writer: the magic, the metadata key/value pairs, one info
//! record per tensor, and then the tensor data.
//!
//! Tensor data is written one tensor at a time and never held in full. The
//! header can be built first because every offset follows from a tensor's
//! type and shape alone, so nothing has to be encoded to know where it will
//! land — which is what keeps writing a file larger than memory an ordinary
//! thing to do.

use crate::gguf::GgufValue;
use anyhow::{Context, Result, bail};
use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use crate::quantize as quant;

const MAGIC: &[u8; 4] = b"GGUF";
const VERSION: u32 = 3;
/// The alignment the tensor-data section is padded to, and the value
/// `general.alignment` would carry. This is the default, so the key is not
/// written.
const ALIGNMENT: u64 = 32;

/// The reader's `gguf_metadata_value_type` numbering.
mod value_type {
    pub const UINT8: u32 = 0;
    pub const INT8: u32 = 1;
    pub const UINT16: u32 = 2;
    pub const INT16: u32 = 3;
    pub const UINT32: u32 = 4;
    pub const INT32: u32 = 5;
    pub const FLOAT32: u32 = 6;
    pub const BOOL: u32 = 7;
    pub const STRING: u32 = 8;
    pub const ARRAY: u32 = 9;
    pub const UINT64: u32 = 10;
    pub const INT64: u32 = 11;
    pub const FLOAT64: u32 = 12;
}

/// One tensor's identity in the file, decided before any of it is encoded.
#[derive(Debug, Clone)]
pub struct TensorPlan {
    pub name: String,
    /// GGUF order: `dims[0]` is the row length.
    pub dims: Vec<u64>,
    pub ggml_type: u32,
}

impl TensorPlan {
    pub fn elements(&self) -> usize {
        self.dims.iter().map(|&d| d as usize).product()
    }

    pub fn ncols(&self) -> usize {
        self.dims[0] as usize
    }

    pub fn bytes(&self) -> usize {
        quant::row_bytes(self.ggml_type, self.ncols()) * (self.elements() / self.ncols())
    }
}

/// Writes a complete GGUF file.
///
/// `values` is asked for one tensor at a time, in file order, and hands
/// back that tensor's weights as `f32`; the encoding to the planned type
/// happens here.
pub fn write(
    path: &Path,
    metadata: &[(String, GgufValue)],
    tensors: &[TensorPlan],
    values: impl FnMut(&TensorPlan) -> Result<Vec<f32>>,
    progress: &dyn Fn(usize, usize, &TensorPlan),
) -> Result<u64> {
    write_weighted(path, metadata, tensors, values, &|_| None, progress)
}

/// [`write`], fitting each tensor `importance` has an entry for to it
/// ([`quant::encode_weighted`]).
pub fn write_weighted<'a>(
    path: &Path,
    metadata: &[(String, GgufValue)],
    tensors: &[TensorPlan],
    mut values: impl FnMut(&TensorPlan) -> Result<Vec<f32>>,
    importance: &dyn Fn(&TensorPlan) -> Option<&'a [f32]>,
    progress: &dyn Fn(usize, usize, &TensorPlan),
) -> Result<u64> {
    write_streamed(
        path,
        metadata,
        tensors,
        |t| values(t).map(Weights::All),
        importance,
        progress,
    )
}

/// A tensor's values as [`write_streamed`] takes them.
pub enum Weights<'a> {
    /// Every value at once, row-major.
    All(Vec<f32>),
    /// Blocks of whole rows, in order: a tensor too large to hold, encoded
    /// and written a block at a time.
    Rows(Box<dyn Iterator<Item = Result<Vec<f32>>> + 'a>),
}

/// [`write_weighted`], each tensor's values taken whole or a block of rows
/// at a time ([`Weights`]). Rows are independent in every type written, so
/// a tensor written in blocks is the same bytes as one written whole.
pub fn write_streamed<'a, 'v>(
    path: &Path,
    metadata: &[(String, GgufValue)],
    tensors: &[TensorPlan],
    mut values: impl FnMut(&TensorPlan) -> Result<Weights<'v>>,
    importance: &dyn Fn(&TensorPlan) -> Option<&'a [f32]>,
    progress: &dyn Fn(usize, usize, &TensorPlan),
) -> Result<u64> {
    let mut offset = 0u64;
    let entries: Vec<HeaderTensor<'_>> = tensors
        .iter()
        .map(|t| {
            let at = offset;
            offset += pad_to(t.bytes() as u64, ALIGNMENT);
            HeaderTensor {
                name: &t.name,
                dims: &t.dims,
                ggml_type: t.ggml_type,
                offset: at,
            }
        })
        .collect();
    let header = header(metadata, &entries)?;

    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut out = BufWriter::with_capacity(4 << 20, file);
    out.write_all(&header)?;
    let mut written = header.len() as u64;
    written += pad(&mut out, written, ALIGNMENT)?;

    let data_start = written;
    for (n, tensor) in tensors.iter().enumerate() {
        progress(n, tensors.len(), tensor);
        let encode = |raw: &[f32]| match importance(tensor) {
            Some(weights) => quant::encode_weighted(tensor.ggml_type, raw, tensor.ncols(), weights),
            None => quant::encode(tensor.ggml_type, raw, tensor.ncols()),
        };
        debug_assert_eq!(written - data_start, offset_of(tensors, n));
        let read = || format!("reading the weights of {}", tensor.name);
        let mut count = 0usize;
        match values(tensor).with_context(read)? {
            Weights::All(raw) => {
                count = raw.len();
                if count == tensor.elements() {
                    let bytes = encode(&raw);
                    out.write_all(&bytes)?;
                    written += bytes.len() as u64;
                }
            }
            Weights::Rows(blocks) => {
                for block in blocks {
                    let block = block.with_context(read)?;
                    if !block.len().is_multiple_of(tensor.ncols().max(1))
                        || count + block.len() > tensor.elements()
                    {
                        bail!(
                            "{}: a block of {} weights is not whole rows of this tensor",
                            tensor.name,
                            block.len()
                        );
                    }
                    count += block.len();
                    let bytes = encode(&block);
                    out.write_all(&bytes)?;
                    written += bytes.len() as u64;
                }
            }
        }
        if count != tensor.elements() {
            bail!(
                "{}: {} weights for a tensor of {}",
                tensor.name,
                count,
                tensor.elements()
            );
        }
        written += pad(&mut out, written, ALIGNMENT)?;
    }
    out.flush()?;
    Ok(written)
}

/// Writes a GGUF file whose tensor bytes come from `produce`: called once
/// per tensor in order, it hands that tensor's encoded bytes to the sink it
/// is given, in as many pieces as it likes. A tensor whose pieces do not
/// add up to its planned size is refused.
pub fn write_with(
    path: &Path,
    metadata: &[(String, GgufValue)],
    tensors: &[TensorPlan],
    mut produce: impl FnMut(&TensorPlan, &mut (dyn FnMut(&[u8]) -> Result<()> + Send)) -> Result<()>,
    progress: &dyn Fn(usize, usize, &TensorPlan),
) -> Result<u64> {
    let mut offset = 0u64;
    let entries: Vec<HeaderTensor<'_>> = tensors
        .iter()
        .map(|t| {
            let at = offset;
            offset += pad_to(t.bytes() as u64, ALIGNMENT);
            HeaderTensor {
                name: &t.name,
                dims: &t.dims,
                ggml_type: t.ggml_type,
                offset: at,
            }
        })
        .collect();
    let header = header(metadata, &entries)?;

    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut out = BufWriter::with_capacity(4 << 20, file);
    out.write_all(&header)?;
    let mut written = header.len() as u64;
    written += pad(&mut out, written, ALIGNMENT)?;
    for (n, tensor) in tensors.iter().enumerate() {
        progress(n, tensors.len(), tensor);
        let mut count = 0u64;
        produce(tensor, &mut |bytes: &[u8]| {
            out.write_all(bytes)?;
            count += bytes.len() as u64;
            Ok(())
        })
        .with_context(|| format!("writing {}", tensor.name))?;
        if count != tensor.bytes() as u64 {
            bail!(
                "{}: {count} bytes for a tensor of {}",
                tensor.name,
                tensor.bytes()
            );
        }
        written += count;
        written += pad(&mut out, written, ALIGNMENT)?;
    }
    out.flush()?;
    Ok(written)
}

/// The data-section offset the header recorded for tensor `n` — the
/// invariant the write loop asserts against as it goes, since a header that
/// disagrees with the data by one padding byte produces a file that reads
/// as plausible garbage.
fn offset_of(tensors: &[TensorPlan], n: usize) -> u64 {
    tensors[..n]
        .iter()
        .map(|t| pad_to(t.bytes() as u64, ALIGNMENT))
        .sum()
}

fn pad_to(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

fn pad(out: &mut impl Write, at: u64, alignment: u64) -> Result<u64> {
    let padding = pad_to(at, alignment) - at;
    if padding > 0 {
        out.write_all(&vec![0u8; padding as usize])?;
    }
    Ok(padding)
}

fn put_string(buf: &mut Vec<u8>, value: &str) {
    buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

fn put_value(buf: &mut Vec<u8>, value: &GgufValue) -> Result<()> {
    buf.extend_from_slice(&type_of(value)?.to_le_bytes());
    put_payload(buf, value)
}

fn type_of(value: &GgufValue) -> Result<u32> {
    Ok(match value {
        GgufValue::U8(_) => value_type::UINT8,
        GgufValue::I8(_) => value_type::INT8,
        GgufValue::U16(_) => value_type::UINT16,
        GgufValue::I16(_) => value_type::INT16,
        GgufValue::U32(_) => value_type::UINT32,
        GgufValue::I32(_) => value_type::INT32,
        GgufValue::F32(_) => value_type::FLOAT32,
        GgufValue::Bool(_) => value_type::BOOL,
        GgufValue::String(_) => value_type::STRING,
        GgufValue::Array(_) => value_type::ARRAY,
        GgufValue::U64(_) => value_type::UINT64,
        GgufValue::I64(_) => value_type::INT64,
        GgufValue::F64(_) => value_type::FLOAT64,
    })
}

fn put_payload(buf: &mut Vec<u8>, value: &GgufValue) -> Result<()> {
    match value {
        GgufValue::U8(v) => buf.push(*v),
        GgufValue::I8(v) => buf.push(*v as u8),
        GgufValue::U16(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I16(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::U32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::F32(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Bool(v) => buf.push(*v as u8),
        GgufValue::String(v) => put_string(buf, v),
        GgufValue::U64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::F64(v) => buf.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Array(items) => {
            // An array declares one element type for all of its members. A
            // mixed array cannot be written, and an empty one has no type
            // to infer, so it is written as an empty array of strings —
            // which is what an absent list of tokens or merges is.
            let element = match items.first() {
                Some(first) => type_of(first)?,
                None => value_type::STRING,
            };
            for item in items {
                if type_of(item)? != element {
                    bail!("a metadata array mixes value types");
                }
            }
            buf.extend_from_slice(&element.to_le_bytes());
            buf.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                put_payload(buf, item)?;
            }
        }
    }
    Ok(())
}

/// What the tensor data will take, before it is written.
///
/// Worth knowing in advance: a full-precision export of a multi-billion
/// parameter model is several gigabytes, and finding that out by running
/// out of disk halfway through is the wrong time.
pub fn planned_bytes(tensors: &[TensorPlan]) -> u64 {
    tensors
        .iter()
        .map(|t| pad_to(t.bytes() as u64, ALIGNMENT))
        .sum()
}

/// A human-readable size for `general.size_label`.
pub fn size_label(parameters: usize) -> String {
    let billions = parameters as f64 / 1e9;
    if billions >= 1.0 {
        format!("{billions:.1}B")
    } else {
        format!("{:.0}M", parameters as f64 / 1e6)
    }
}

/// One tensor's record in a header: its offset is relative to the start of
/// the tensor-data section.
pub struct HeaderTensor<'a> {
    pub name: &'a str,
    pub dims: &'a [u64],
    pub ggml_type: u32,
    pub offset: u64,
}

/// The magic, version, metadata and tensor records, unpadded.
pub fn header(metadata: &[(String, GgufValue)], tensors: &[HeaderTensor<'_>]) -> Result<Vec<u8>> {
    let mut out: Vec<u8> = Vec::with_capacity(1 << 20);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        put_string(&mut out, key);
        put_value(&mut out, value)?;
    }
    for tensor in tensors {
        if tensor.dims.is_empty() {
            bail!("tensor {} has no dimensions", tensor.name);
        }
        put_string(&mut out, tensor.name);
        out.extend_from_slice(&(tensor.dims.len() as u32).to_le_bytes());
        for &dim in tensor.dims {
            out.extend_from_slice(&dim.to_le_bytes());
        }
        out.extend_from_slice(&tensor.ggml_type.to_le_bytes());
        out.extend_from_slice(&tensor.offset.to_le_bytes());
    }
    Ok(out)
}

/// One tensor copied as it is stored.
pub struct RawTensor<'a> {
    pub name: &'a str,
    pub dims: &'a [u64],
    pub ggml_type: u32,
    pub bytes: &'a [u8],
}

/// Writes a GGUF file whose tensors are copied byte for byte, each padded to
/// `alignment` — which `metadata` must declare as `general.alignment` when
/// it is not the default 32.
pub fn write_raw(
    path: &Path,
    metadata: &[(String, GgufValue)],
    tensors: &[RawTensor<'_>],
    alignment: u64,
    progress: &dyn Fn(usize, usize, &str),
) -> Result<u64> {
    let mut offset = 0u64;
    let entries: Vec<HeaderTensor<'_>> = tensors
        .iter()
        .map(|t| {
            let at = offset;
            offset += pad_to(t.bytes.len() as u64, alignment);
            HeaderTensor {
                name: t.name,
                dims: t.dims,
                ggml_type: t.ggml_type,
                offset: at,
            }
        })
        .collect();
    let header = header(metadata, &entries)?;
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut out = BufWriter::with_capacity(4 << 20, file);
    out.write_all(&header)?;
    let mut written = header.len() as u64;
    written += pad(&mut out, written, alignment)?;
    for (n, tensor) in tensors.iter().enumerate() {
        progress(n, tensors.len(), tensor.name);
        out.write_all(tensor.bytes)?;
        written += tensor.bytes.len() as u64;
        written += pad(&mut out, written, alignment)?;
    }
    out.flush()?;
    Ok(written)
}

/// `value` rounded up to a multiple of `alignment`.
pub fn aligned(value: u64, alignment: u64) -> u64 {
    pad_to(value, alignment)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufFile;

    fn metadata() -> Vec<(String, GgufValue)> {
        vec![
            (
                "general.architecture".into(),
                GgufValue::String("qwen3".into()),
            ),
            ("general.file_type".into(), GgufValue::U32(32)),
            (
                "qwen3.attention.layer_norm_rms_epsilon".into(),
                GgufValue::F32(1e-5),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                GgufValue::Array(vec![
                    GgufValue::String("a".into()),
                    GgufValue::String("b".into()),
                ]),
            ),
            (
                "tokenizer.ggml.token_type".into(),
                GgufValue::Array(vec![GgufValue::I32(1), GgufValue::I32(3)]),
            ),
            (
                "tokenizer.ggml.add_bos_token".into(),
                GgufValue::Bool(false),
            ),
        ]
    }

    /// The whole point of the writer: what it writes, the reader reads back
    /// as the same thing.
    #[test]
    fn a_written_file_reads_back_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.gguf");
        let tensors = vec![
            TensorPlan {
                name: "token_embd.weight".into(),
                dims: vec![256, 4],
                ggml_type: quant::GGML_TYPE_BF16,
            },
            TensorPlan {
                name: "output_norm.weight".into(),
                dims: vec![256],
                ggml_type: quant::GGML_TYPE_F32,
            },
            TensorPlan {
                name: "blk.0.attn_q.weight".into(),
                dims: vec![256, 256],
                ggml_type: quant::GGML_TYPE_Q4_K,
            },
        ];
        let size = write(
            &path,
            &metadata(),
            &tensors,
            |t| Ok((0..t.elements()).map(|i| (i % 17) as f32 * 0.01).collect()),
            &|_, _, _| {},
        )
        .unwrap();
        assert_eq!(size, std::fs::metadata(&path).unwrap().len());

        let file = GgufFile::open(&path).unwrap();
        let get = |key: &str| {
            file.metadata
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.clone())
                .unwrap()
        };
        assert!(matches!(get("general.architecture"), GgufValue::String(s) if s == "qwen3"));
        assert!(matches!(get("general.file_type"), GgufValue::U32(32)));
        assert!(matches!(
            get("tokenizer.ggml.add_bos_token"),
            GgufValue::Bool(false)
        ));
        match get("tokenizer.ggml.tokens") {
            GgufValue::Array(items) => assert_eq!(items.len(), 2),
            other => panic!("{other:?}"),
        }
        assert_eq!(file.tensors.len(), 3);
        for (written, read) in tensors.iter().zip(file.tensors.iter()) {
            assert_eq!(written.name, read.name);
            assert_eq!(written.dims, read.dims);
            assert_eq!(written.ggml_type, read.ggml_type);
        }
    }

    /// Every tensor's data has to start where its info record says it does,
    /// after alignment padding — the failure this catches reads back as
    /// plausible noise rather than as an error.
    #[test]
    fn tensor_offsets_are_aligned_and_land_on_the_data() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.gguf");
        let tensors = vec![
            TensorPlan {
                name: "a.weight".into(),
                dims: vec![3],
                ggml_type: quant::GGML_TYPE_F32,
            },
            TensorPlan {
                name: "b.weight".into(),
                dims: vec![5],
                ggml_type: quant::GGML_TYPE_F32,
            },
        ];
        write(
            &path,
            &metadata(),
            &tensors,
            |t| Ok((0..t.elements()).map(|i| i as f32 + 1.0).collect()),
            &|_, _, _| {},
        )
        .unwrap();

        let file = GgufFile::open(&path).unwrap();
        assert_eq!(file.tensors[0].offset, 0);
        assert_eq!(
            file.tensors[1].offset, 32,
            "the second tensor is padded to 32"
        );
        let bytes = std::fs::read(&path).unwrap();
        let at = (file.data_offset + file.tensors[1].offset) as usize;
        assert_eq!(
            f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()),
            1.0
        );
    }

    #[test]
    fn the_planned_size_is_the_size_that_gets_written() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.gguf");
        let tensors = vec![
            TensorPlan {
                name: "a.weight".into(),
                dims: vec![256, 8],
                ggml_type: quant::GGML_TYPE_Q4_K,
            },
            TensorPlan {
                name: "b.weight".into(),
                dims: vec![7],
                ggml_type: quant::GGML_TYPE_F32,
            },
        ];
        let written = write(
            &path,
            &metadata(),
            &tensors,
            |t| Ok(vec![0.5; t.elements()]),
            &|_, _, _| {},
        )
        .unwrap();
        let file = GgufFile::open(&path).unwrap();
        assert_eq!(written - file.data_offset, planned_bytes(&tensors));
    }

    #[test]
    fn size_labels_read_the_way_a_model_name_does() {
        assert_eq!(size_label(2_020_000_000), "2.0B");
        assert_eq!(size_label(10_000_000), "10M");
    }
}
