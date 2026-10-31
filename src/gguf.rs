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

//! Reader for the GGUF model file format, following
//! <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>: a `GGUF`
//! magic, a header naming how many tensors and metadata key/value pairs
//! follow, the key/value pairs themselves (scalars, strings, or arrays of
//! either, arbitrarily nested), then one tensor-info record per tensor
//! (name/shape/type/offset — never the tensor data itself, which is what
//! makes this cheap to read even for huge multi-gigabyte model files).
//!
//! Only little-endian GGUF is supported: the spec itself notes there is
//! currently no reliable way to detect a big-endian file, and every model in
//! the wild is little-endian.

use anyhow::{Context, Result, anyhow, bail};
use std::{
    fs::File,
    io::{BufReader, Read, Seek, SeekFrom},
    path::Path,
};

const MAGIC: &[u8; 4] = b"GGUF";
/// GGUFv1 used `uint32` tensor/metadata counts instead of `uint64` and is
/// long deprecated upstream; nothing still produces it.
const MIN_SUPPORTED_VERSION: u32 = 2;
/// Circuit breakers against a corrupt or hostile length prefix trying to
/// force a multi-gigabyte allocation before a single byte of it is verified
/// to exist in the file.
const MAX_STRING_BYTES: u64 = 100 * 1024 * 1024;
const MAX_ARRAY_ELEMENTS: u64 = 200_000_000;
/// Arrays longer than this are skipped by [`GgufFile::open_summary`] — long
/// enough for every per-layer table (a few hundred entries), far short of a
/// vocabulary.
const SUMMARY_ARRAY_LIMIT: u64 = 4096;

/// Default alignment for the tensor-data section per the spec: used when
/// `general.alignment` is absent from the metadata.
const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Debug, Clone)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    /// Widens any scalar integer/bool variant to `u64` — used to read
    /// small config-style values (e.g. `general.alignment`,
    /// `<arch>.context_length`) without matching on every possible width.
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U8(v) => Some(*v as u64),
            GgufValue::U16(v) => Some(*v as u64),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::U64(v) => Some(*v),
            GgufValue::I8(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I16(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I32(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I64(v) if *v >= 0 => Some(*v as u64),
            GgufValue::Bool(v) => Some(*v as u64),
            _ => None,
        }
    }

    /// Renders the value for display. Arrays longer than `preview_limit`
    /// print only their first `preview_limit` elements plus a `... (N more)`
    /// marker instead of every element — metadata arrays like
    /// `tokenizer.ggml.tokens` routinely hold well over 100,000 entries,
    /// which would otherwise flood the terminal. Pass `usize::MAX` (as
    /// `--full` does) to disable truncation.
    pub fn display(&self, preview_limit: usize) -> String {
        match self {
            GgufValue::U8(v) => v.to_string(),
            GgufValue::I8(v) => v.to_string(),
            GgufValue::U16(v) => v.to_string(),
            GgufValue::I16(v) => v.to_string(),
            GgufValue::U32(v) => v.to_string(),
            GgufValue::I32(v) => v.to_string(),
            GgufValue::F32(v) => v.to_string(),
            GgufValue::Bool(v) => v.to_string(),
            GgufValue::U64(v) => v.to_string(),
            GgufValue::I64(v) => v.to_string(),
            GgufValue::F64(v) => v.to_string(),
            GgufValue::String(s) => format!("{s:?}"),
            GgufValue::Array(items) => {
                let shown = items.len().min(preview_limit);
                let mut parts: Vec<String> = items[..shown]
                    .iter()
                    .map(|v| v.display(preview_limit))
                    .collect();
                if shown < items.len() {
                    parts.push(format!("... ({} more)", items.len() - shown));
                }
                format!("[{}] ({} total)", parts.join(", "), items.len())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// Byte offset of this tensor's data, relative to the start of the
    /// tensor-data section (i.e. relative to [`GgufFile::data_offset`], not
    /// the start of the file).
    pub offset: u64,
}

impl TensorInfo {
    pub fn element_count(&self) -> u64 {
        self.dims.iter().product()
    }

    pub fn shape(&self) -> String {
        if self.dims.is_empty() {
            return "scalar".to_string();
        }
        self.dims
            .iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(" x ")
    }
}

#[derive(Debug)]
pub struct GgufFile {
    pub version: u32,
    pub metadata: Vec<(String, GgufValue)>,
    pub tensors: Vec<TensorInfo>,
    pub alignment: u64,
    /// Absolute byte offset in the file where tensor data begins, i.e. the
    /// end of the tensor-info table padded up to `alignment`.
    pub data_offset: u64,
}

impl GgufFile {
    pub fn open(path: &Path) -> Result<GgufFile> {
        Self::open_at(path, 0)
    }

    /// [`open`](Self::open) for a listing: the tensor table and every
    /// metadata value except the large arrays, which are skipped without
    /// being read into memory — and are then **absent** from
    /// [`metadata`](Self::metadata), not present and empty.
    ///
    /// A model's vocabulary is a quarter of a million strings in three
    /// arrays, and parsing them is nearly all of what opening a file costs:
    /// a directory of a couple of hundred models took six seconds to scan
    /// this way, twice per server start (`resolve_load_target`), so a
    /// coordinator's model swap spent most of its time listing models it
    /// was not about to load. A scan needs the architecture and the tensor
    /// types, and nothing an array of that size holds.
    pub fn open_summary(path: &Path) -> Result<GgufFile> {
        let file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        let len = file.metadata().ok().map(|m| m.len());
        Self::read_with(BufReader::new(file), Some(SUMMARY_ARRAY_LIMIT), len)
            .with_context(|| format!("failed to parse GGUF file {}", path.display()))
    }

    /// Reads the GGUF structure that begins `offset` bytes into `path`.
    ///
    /// `0` — [`open`](Self::open) — is the ordinary case: a `.gguf` file
    /// whose model starts where the file does. A non-zero offset is what a
    /// bundled `orangu-server` needs (`bundle::Bundle`): there the model is
    /// appended to the executable, so every structure the format defines sits
    /// at its usual place *relative to the segment*, not to the start of the
    /// file it happens to be carried in.
    ///
    /// [`data_offset`](Self::data_offset) stays relative to the segment for
    /// the same reason it is relative to the file here: the caller that knows
    /// where the segment starts is the one that adds `offset` back.
    pub fn open_at(path: &Path, offset: u64) -> Result<GgufFile> {
        let mut file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        if offset != 0 {
            file.seek(SeekFrom::Start(offset))
                .with_context(|| format!("failed to seek to {offset} in {}", path.display()))?;
        }
        let len = file.metadata().ok().map(|m| m.len().saturating_sub(offset));
        Self::read_with(BufReader::new(file), None, len)
            .with_context(|| format!("failed to parse GGUF file {}", path.display()))
    }

    /// Reads the GGUF structure from any byte stream, rather than from a file
    /// on this machine.
    ///
    /// The parser is strictly sequential and stops at the end of the tensor
    /// table, reading none of the tensor data behind it. A caller holding a
    /// network response can therefore hand it the response body and drop the
    /// connection the moment this returns, having transferred the few hundred
    /// kilobytes of header rather than the file — which may be hundreds of
    /// gigabytes. `orangu::model_download::RemoteModel::headers` is that
    /// caller, and it is why this exists; every path that has the file
    /// already goes through [`open`](Self::open) or
    /// [`open_at`](Self::open_at) instead.
    ///
    /// Wrap `inner` in a [`BufReader`] unless it is already buffered: the
    /// parser issues many small reads.
    pub fn read_from<R: Read>(inner: R) -> Result<GgufFile> {
        Self::read(Streamed(inner))
    }

    fn read<R: GgufSource>(inner: R) -> Result<GgufFile> {
        Self::read_with(inner, None, None)
    }

    /// The parser; `skip_arrays_over` is the summary mode's array limit —
    /// see [`open_summary`](Self::open_summary).
    ///
    /// `source_len` is the byte length of the GGUF (file minus segment start)
    /// when the caller knows it; it bounds the last tensor's extent.
    fn read_with<R: GgufSource>(
        inner: R,
        skip_arrays_over: Option<u64>,
        source_len: Option<u64>,
    ) -> Result<GgufFile> {
        let mut reader = Reader {
            inner,
            bytes_read: 0,
            pending_array: None,
        };

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        if &magic != MAGIC {
            bail!(
                "not a GGUF file: expected magic {:?}, found {:?}",
                MAGIC,
                magic
            );
        }

        let version = reader.read_u32()?;
        if version < MIN_SUPPORTED_VERSION {
            bail!(
                "GGUFv{version} is not supported (uses 32-bit tensor/metadata counts); only GGUFv{MIN_SUPPORTED_VERSION}+ is"
            );
        }

        let tensor_count = reader.read_u64()?;
        let metadata_kv_count = reader.read_u64()?;

        let mut metadata = Vec::with_capacity(metadata_kv_count.min(4096) as usize);
        for _ in 0..metadata_kv_count {
            let key = reader.read_string()?;
            let value_type = reader.read_u32()?;
            if let (Some(limit), 9) = (skip_arrays_over, value_type)
                && reader.skip_array_over(limit)?
            {
                continue;
            }
            let value = reader.read_value(value_type)?;
            metadata.push((key, value));
        }

        let mut tensors = Vec::with_capacity(tensor_count.min(65_536) as usize);
        for _ in 0..tensor_count {
            let name = reader.read_string()?;
            let n_dims = reader.read_u32()?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(reader.read_u64()?);
            }
            let ggml_type = reader.read_u32()?;
            let offset = reader.read_u64()?;
            tensors.push(TensorInfo {
                name,
                dims,
                ggml_type,
                offset,
            });
        }

        let alignment = metadata
            .iter()
            .find(|(key, _)| key == "general.alignment")
            .and_then(|(_, value)| value.as_u64())
            .filter(|&a| a > 0)
            .unwrap_or(DEFAULT_ALIGNMENT);

        let position = reader.bytes_read;
        let data_offset = position.div_ceil(alignment) * alignment;
        resolve_mxfp8(
            &mut tensors,
            alignment,
            source_len.map(|len| len.saturating_sub(data_offset)),
        );

        Ok(GgufFile {
            version,
            metadata,
            tensors,
            alignment,
            data_offset,
        })
    }

    /// Element counts per `ggml_type` across every tensor in this file. A
    /// multi-part model's shard files must have their totals summed together
    /// before picking one dominant type for the whole model — each shard's
    /// tensor-info table only covers the tensors stored in that shard.
    pub fn type_element_totals(&self) -> std::collections::HashMap<u32, u128> {
        let mut totals: std::collections::HashMap<u32, u128> = std::collections::HashMap::new();
        for tensor in &self.tensors {
            *totals.entry(tensor.ggml_type).or_default() += tensor.element_count() as u128;
        }
        totals
    }

    /// Whether this file is a multimodal projector ("mmproj") sidecar —
    /// a vision/audio encoder meant to be loaded alongside a base LLM's own
    /// checkpoint, not a standalone model in its own right. Identified the
    /// same way llama.cpp's own `clip.cpp` loader does: `general.architecture`
    /// is `"clip"`.
    pub fn is_clip_projector(&self) -> bool {
        self.metadata.iter().any(|(key, value)| {
            key == "general.architecture" && matches!(value, GgufValue::String(s) if s == "clip")
        })
    }
}

struct Reader<R> {
    inner: R,
    bytes_read: u64,
    /// An array header `skip_array_over` read and did not skip, for the
    /// `read_value` that follows it.
    pending_array: Option<(u32, u64)>,
}

/// What the parser reads from: bytes, and a way to skip some — by seeking
/// where the source is a file, by reading them into nothing where it is a
/// stream.
trait GgufSource: Read {
    fn skip(&mut self, n: u64) -> std::io::Result<()>;
}

impl GgufSource for BufReader<File> {
    fn skip(&mut self, n: u64) -> std::io::Result<()> {
        self.seek_relative(n as i64)
    }
}

impl<T: AsRef<[u8]>> GgufSource for std::io::Cursor<T> {
    fn skip(&mut self, n: u64) -> std::io::Result<()> {
        let remaining = (self.get_ref().as_ref().len() as u64).saturating_sub(self.position());
        if n > remaining {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("unexpected end of file while skipping {n} bytes"),
            ));
        }
        self.set_position(self.position() + n);
        Ok(())
    }
}

/// A source with no seek — a network body — skips by reading.
struct Streamed<R>(R);

impl<R: Read> Read for Streamed<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.0.read(buf)
    }
}

impl<R: Read> GgufSource for Streamed<R> {
    fn skip(&mut self, n: u64) -> std::io::Result<()> {
        let copied = std::io::copy(&mut (&mut self.0).take(n), &mut std::io::sink())?;
        if copied != n {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                format!("unexpected end of file while skipping {n} bytes"),
            ));
        }
        Ok(())
    }
}

impl<R: GgufSource> Reader<R> {
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        self.inner
            .read_exact(buf)
            .context("unexpected end of file")?;
        self.bytes_read += buf.len() as u64;
        Ok(())
    }

    fn read_u8(&mut self) -> Result<u8> {
        let mut buf = [0u8; 1];
        self.read_exact(&mut buf)?;
        Ok(buf[0])
    }

    fn read_i8(&mut self) -> Result<i8> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let mut buf = [0u8; 2];
        self.read_exact(&mut buf)?;
        Ok(u16::from_le_bytes(buf))
    }

    fn read_i16(&mut self) -> Result<i16> {
        let mut buf = [0u8; 2];
        self.read_exact(&mut buf)?;
        Ok(i16::from_le_bytes(buf))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(u32::from_le_bytes(buf))
    }

    fn read_i32(&mut self) -> Result<i32> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(i32::from_le_bytes(buf))
    }

    fn read_f32(&mut self) -> Result<f32> {
        let mut buf = [0u8; 4];
        self.read_exact(&mut buf)?;
        Ok(f32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(u64::from_le_bytes(buf))
    }

    fn read_i64(&mut self) -> Result<i64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(i64::from_le_bytes(buf))
    }

    fn read_f64(&mut self) -> Result<f64> {
        let mut buf = [0u8; 8];
        self.read_exact(&mut buf)?;
        Ok(f64::from_le_bytes(buf))
    }

    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_u8()? != 0)
    }

    /// Skips the bytes of one value of `value_type` without building it.
    fn skip_value(&mut self, value_type: u32) -> Result<()> {
        let fixed: u64 = match value_type {
            0 | 1 | 7 => 1,
            2 | 3 => 2,
            4..=6 => 4,
            10..=12 => 8,
            8 => {
                let len = self.read_u64()?;
                if len > MAX_STRING_BYTES {
                    bail!(
                        "string metadata value of {len} bytes exceeds the {MAX_STRING_BYTES}-byte limit"
                    );
                }
                len
            }
            9 => {
                let elem_type = self.read_u32()?;
                let len = self.read_u64()?;
                self.skip_array(elem_type, len)?;
                0
            }
            other => bail!("unknown GGUF metadata value type {other}"),
        };
        self.skip_bytes(fixed)
    }

    /// Skips `len` values of `elem_type`: fixed-size elements as one skip
    /// (a per-element seek is a system call each, and a vocabulary's score
    /// table is a quarter of a million of them), strings one at a time,
    /// since each carries its own length.
    fn skip_array(&mut self, elem_type: u32, len: u64) -> Result<()> {
        let fixed: Option<u64> = match elem_type {
            0 | 1 | 7 => Some(1),
            2 | 3 => Some(2),
            4..=6 => Some(4),
            10..=12 => Some(8),
            _ => None,
        };
        match fixed {
            Some(size) => self.skip_bytes(len.saturating_mul(size)),
            None => {
                for _ in 0..len {
                    self.skip_value(elem_type)?;
                }
                Ok(())
            }
        }
    }

    fn skip_bytes(&mut self, n: u64) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        self.inner
            .skip(n)
            .with_context(|| format!("skipping {n} bytes"))?;
        self.bytes_read += n;
        Ok(())
    }

    /// At an array value: reads its header and, when it holds more than
    /// `limit` elements, skips the rest and returns `true`; otherwise reads
    /// nothing further and returns `false`, leaving the header in
    /// `pending_array` for the `read_value` that follows.
    fn skip_array_over(&mut self, limit: u64) -> Result<bool> {
        let elem_type = self.read_u32()?;
        let len = self.read_u64()?;
        if len > MAX_ARRAY_ELEMENTS {
            bail!(
                "array metadata value of {len} elements exceeds the {MAX_ARRAY_ELEMENTS}-element limit"
            );
        }
        if len > limit {
            self.skip_array(elem_type, len)?;
            return Ok(true);
        }
        self.pending_array = Some((elem_type, len));
        Ok(false)
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()?;
        if len > MAX_STRING_BYTES {
            bail!("string metadata value of {len} bytes exceeds the {MAX_STRING_BYTES}-byte limit");
        }
        let mut buf = vec![0u8; len as usize];
        self.read_exact(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }

    fn read_value(&mut self, value_type: u32) -> Result<GgufValue> {
        match value_type {
            0 => Ok(GgufValue::U8(self.read_u8()?)),
            1 => Ok(GgufValue::I8(self.read_i8()?)),
            2 => Ok(GgufValue::U16(self.read_u16()?)),
            3 => Ok(GgufValue::I16(self.read_i16()?)),
            4 => Ok(GgufValue::U32(self.read_u32()?)),
            5 => Ok(GgufValue::I32(self.read_i32()?)),
            6 => Ok(GgufValue::F32(self.read_f32()?)),
            7 => Ok(GgufValue::Bool(self.read_bool()?)),
            8 => Ok(GgufValue::String(self.read_string()?)),
            9 => {
                // The header, unless `skip_array_over` already consumed it.
                let (elem_type, len) = match self.pending_array.take() {
                    Some(header) => header,
                    None => {
                        let elem_type = self.read_u32()?;
                        let len = self.read_u64()?;
                        if len > MAX_ARRAY_ELEMENTS {
                            bail!(
                                "array metadata value of {len} elements exceeds the {MAX_ARRAY_ELEMENTS}-element limit"
                            );
                        }
                        (elem_type, len)
                    }
                };
                let mut items = Vec::new();
                for _ in 0..len {
                    items.push(self.read_value(elem_type)?);
                }
                Ok(GgufValue::Array(items))
            }
            10 => Ok(GgufValue::U64(self.read_u64()?)),
            11 => Ok(GgufValue::I64(self.read_i64()?)),
            12 => Ok(GgufValue::F64(self.read_f64()?)),
            other => Err(anyhow!("unknown GGUF metadata value type {other}")),
        }
    }
}

/// `ggml_type` id -> canonical name, per
/// <https://github.com/ggml-org/ggml/blob/master/include/ggml.h>.
///
/// Ids that ggml has since **deleted** still get their historical name here
/// rather than a `None` placeholder, because real files carrying them are
/// still served (every `bartowski/*-GGUF` repo from 2024 has a
/// `Q4_0_4_4`/`Q4_0_4_8`/`Q4_0_8_8` upload). Printing `reserved(31)` for one
/// tells a user nothing; printing `Q4_0_4_4` and letting
/// [`removed_ggml_type`] explain what happened to it tells them what to
/// download instead. A retired id is never reused, so the name stays
/// unambiguous. Ids beyond the table are types added after this was
/// written.
///
/// The table runs past the end of ggml's own enum because ids 64 and up are
/// where a quantizer that ships types of its own puts them, by the
/// convention that leaves 42..63 free for ggml to grow into. Those ids
/// therefore have to be *named* here for a file carrying one to say
/// anything useful, and the gap below them is the reservation itself —
/// `None`, printed as `reserved(N)`, so a file that somehow carries one is
/// reported as a range ggml has not filled yet rather than as a type this
/// build is merely behind on.
const GGML_TYPE_NAMES: &[Option<&str>] = &[
    Some("F32"),        // 0
    Some("F16"),        // 1
    Some("Q4_0"),       // 2
    Some("Q4_1"),       // 3
    Some("Q4_2"),       // 4  (removed from ggml)
    Some("Q4_3"),       // 5  (removed from ggml)
    Some("Q5_0"),       // 6
    Some("Q5_1"),       // 7
    Some("Q8_0"),       // 8
    Some("Q8_1"),       // 9
    Some("Q2_K"),       // 10
    Some("Q3_K"),       // 11
    Some("Q4_K"),       // 12
    Some("Q5_K"),       // 13
    Some("Q6_K"),       // 14
    Some("Q8_K"),       // 15
    Some("IQ2_XXS"),    // 16
    Some("IQ2_XS"),     // 17
    Some("IQ3_XXS"),    // 18
    Some("IQ1_S"),      // 19
    Some("IQ4_NL"),     // 20
    Some("IQ3_S"),      // 21
    Some("IQ2_S"),      // 22
    Some("IQ4_XS"),     // 23
    Some("I8"),         // 24
    Some("I16"),        // 25
    Some("I32"),        // 26
    Some("I64"),        // 27
    Some("F64"),        // 28
    Some("IQ1_M"),      // 29
    Some("BF16"),       // 30
    Some("Q4_0_4_4"),   // 31 (removed from ggml)
    Some("Q4_0_4_8"),   // 32 (removed from ggml)
    Some("Q4_0_8_8"),   // 33 (removed from ggml)
    Some("TQ1_0"),      // 34
    Some("TQ2_0"),      // 35
    Some("IQ4_NL_4_4"), // 36 (removed from ggml)
    Some("IQ4_NL_4_8"), // 37 (removed from ggml)
    Some("IQ4_NL_8_8"), // 38 (removed from ggml)
    Some("MXFP4"),      // 39
    Some("NVFP4"),      // 40
    Some("Q1_0"),       // 41
    Some("Q2_0"),       // 42
    Some("F8_E4M3"),    // 43
    None,               // 44 \
    None,               // 45 |
    None,               // 46 |
    None,               // 47 |
    None,               // 48 |
    None,               // 49 |
    None,               // 50 |
    None,               // 51 |
    None,               // 52 > reserved for ggml itself
    None,               // 53 |
    None,               // 54 |
    None,               // 55 |
    None,               // 56 |
    None,               // 57 |
    None,               // 58 |
    None,               // 59 |
    None,               // 60 |
    None,               // 61 |
    None,               // 62 |
    None,               // 63 /
    Some("IQ1_XS"),     // 64
    Some("IQ1_XXS"),    // 65
    Some("IQ1_XXXS"),   // 66
];

/// `ggml_type` ids ggml itself has **deleted** *and* that this build cannot
/// read either.
///
/// "Removed upstream" and "unreadable here" are not the same set, which is
/// why this list is far narrower than the retired ids in
/// [`GGML_TYPE_NAMES`]. All six ARM-SIMD pre-repacked layouts (31–33 over
/// `Q4_0`, 36–38 over `IQ4_NL`) are retired upstream but *are* read here:
/// the packing is a lossless permutation that
/// `engine::quant::deinterleave_repack` undoes at load time. What remains
/// is `Q4_2`/`Q4_3`, which predate the K-quants and have no successor at
/// all — nothing to point a user at but their own source weights.
const REMOVED_GGML_TYPES: &[u32] = &[4, 5];

/// Whether `ggml_type` is one ggml has removed *and* this build cannot read
/// — see [`REMOVED_GGML_TYPES`] for why that is a narrower set than the
/// retired ids.
/// `GGML_TYPE_F32`, `_F16` and `_Q8_0` — the types [`GgufFile::read_tensor`]
/// can materialize as `f32`.
const TYPE_F32: u32 = 0;
const TYPE_F16: u32 = 1;
const TYPE_Q8_0: u32 = 8;

/// A `Q8_0` block: one `f16` scale followed by 32 `int8` weights.
const Q8_0_BLOCK: usize = 32;
const Q8_0_BYTES: usize = 2 + Q8_0_BLOCK;

impl GgufFile {
    /// Reads one tensor out of `path` and returns it as `f32`.
    ///
    /// Seeks straight to the tensor rather than reading the file, so pulling
    /// a single projection out of an eight-gigabyte model costs its own size
    /// and nothing more.
    ///
    /// Only `F32`, `F16` and `Q8_0` are handled. That is deliberately narrow:
    /// this exists to hand real weights to something that needs plain floats
    /// — [`crate::npu_ort`] compiling a layer, a test checking a kernel — not
    /// to become a second dequantizer beside the engine's own. An unsupported
    /// type is an error naming itself, never a silent zero.
    pub fn read_tensor(&self, path: &Path, name: &str) -> Result<Vec<f32>> {
        self.read_tensor_at(path, name, 0)
    }

    /// [`Self::read_tensor`] for a GGUF that begins `base` bytes into `path`.
    pub fn read_tensor_at(&self, path: &Path, name: &str, base: u64) -> Result<Vec<f32>> {
        let info = self
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| anyhow!("no tensor named `{name}` in {}", path.display()))?;

        let count = info.element_count() as usize;
        let bytes = match info.ggml_type {
            TYPE_F32 => count * 4,
            TYPE_F16 => count * 2,
            TYPE_Q8_0 => {
                if !count.is_multiple_of(Q8_0_BLOCK) {
                    bail!(
                        "`{name}` has {count} elements, not a whole number of {Q8_0_BLOCK}-element Q8_0 blocks"
                    );
                }
                count / Q8_0_BLOCK * Q8_0_BYTES
            }
            other => bail!(
                "`{name}` is {}, which this reader cannot dequantize (F32, F16 and Q8_0 only)",
                ggml_type_name(other)
            ),
        };

        let mut file =
            File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
        file.seek(SeekFrom::Start(base + self.data_offset + info.offset))
            .with_context(|| format!("failed to seek to `{name}`"))?;
        let mut raw = vec![0u8; bytes];
        file.read_exact(&mut raw)
            .with_context(|| format!("failed to read {bytes} bytes of `{name}`"))?;

        let mut out = vec![0.0f32; count];
        match info.ggml_type {
            TYPE_F32 => {
                for (value, chunk) in out.iter_mut().zip(raw.as_chunks::<4>().0) {
                    *value = f32::from_le_bytes(*chunk);
                }
            }
            TYPE_F16 => {
                for (value, chunk) in out.iter_mut().zip(raw.as_chunks::<2>().0) {
                    *value = half::f16::from_bits(u16::from_le_bytes(*chunk)).to_f32();
                }
            }
            TYPE_Q8_0 => {
                for (block, dst) in raw
                    .as_chunks::<Q8_0_BYTES>()
                    .0
                    .iter()
                    .zip(out.as_chunks_mut::<Q8_0_BLOCK>().0)
                {
                    let scale =
                        half::f16::from_bits(u16::from_le_bytes([block[0], block[1]])).to_f32();
                    for (value, quantized) in dst.iter_mut().zip(&block[2..]) {
                        *value = *quantized as i8 as f32 * scale;
                    }
                }
            }
            _ => unreachable!("guarded by the match above"),
        }
        Ok(out)
    }

    /// The `f32` scalar stored under `name`, if there is one.
    ///
    /// Some projectors ship an activation range beside each weight —
    /// `<layer>.input_min`, `<layer>.output_max` and so on — which is exactly
    /// the calibration a static quantizer would otherwise have to measure.
    pub fn read_scalar(&self, path: &Path, name: &str) -> Option<f32> {
        self.read_tensor(path, name).ok()?.first().copied()
    }
}

pub fn is_removed_ggml_type(ggml_type: u32) -> bool {
    REMOVED_GGML_TYPES.contains(&ggml_type)
}

/// Vendor-private `ggml_type` ids that sit *above* the contiguous table —
/// too far up for a run of `None`s to be readable, so they are named by id.
///
/// `PQ2_0`/`PTQ1_0` are Prism ML's ternary formats (`PrismML-Eng/llama.cpp`,
/// `ggml/include/ggml.h`): the same 2-bit and base-3 codecs as ggml's own
/// `Q2_0` (42) and `TQ1_0` (34), with one `f16` scale per **128** weights
/// instead of per 64 and per 256. They carry their own ids because stock
/// ggml would otherwise read a `PQ2_0` file as `Q2_0` and decode it wrong
/// without a word — a file carrying one names itself here for the same
/// reason. `engine::quant` reads both.
///
/// `MXFP8` is an id this reader assigns rather than one a file carries: see
/// [`TYPE_MXFP8`].
const PRIVATE_GGML_TYPE_NAMES: &[(u32, &str)] =
    &[(142, "PQ2_0"), (143, "PTQ1_0"), (TYPE_MXFP8, "MXFP8")];

/// `ggml_type` 43 as ggml defines it: one OCP FP8 E4M3 byte per element.
const TYPE_F8_E4M3: u32 = 43;

/// Elements under one `MXFP8` scale, and the bytes such a block takes.
const MXFP8_BLOCK: u64 = 32;
const MXFP8_BLOCK_BYTES: u64 = 1 + MXFP8_BLOCK;

/// OCP MXFP8: blocks of one E8M0 power-of-two scale followed by 32 FP8 E4M3
/// values, 33 bytes per 32 elements.
///
/// Some quantizers write this layout under id 43, which ggml reserves for
/// *unscaled* E4M3 — the same tensor shape in 32 fewer bytes per kilobyte.
/// The two are told apart by the bytes the tensor occupies in the file
/// ([`resolve_mxfp8`]), and a tensor found to carry the scales is given
/// this id, which no file uses, so that every later size and dequantize
/// step reads it as what it is.
pub const TYPE_MXFP8: u32 = 144;

/// Re-types each id-43 tensor whose stored bytes hold an E8M0 scale per 32
/// elements as [`TYPE_MXFP8`].
///
/// A tensor's extent is the distance to the next tensor's offset or, for
/// the last one, to `data_len` (the size of the tensor-data section, when
/// the caller knows it). Plain E4M3 ends within one alignment pad of its
/// element count; the scaled layout needs `n / 32` bytes more, so a tensor
/// counts as scaled only when that excess is itself at least one alignment
/// unit — below it the two cannot be told apart and the plain reading
/// stands.
fn resolve_mxfp8(tensors: &mut [TensorInfo], alignment: u64, data_len: Option<u64>) {
    let mut offsets: Vec<u64> = tensors.iter().map(|t| t.offset).collect();
    offsets.sort_unstable();
    offsets.dedup();
    for tensor in tensors.iter_mut() {
        let count = tensor.element_count();
        if tensor.ggml_type != TYPE_F8_E4M3
            || !count.is_multiple_of(MXFP8_BLOCK)
            || count / MXFP8_BLOCK < alignment
        {
            continue;
        }
        let end = match offsets.iter().find(|&&o| o > tensor.offset) {
            Some(&next) => Some(next),
            None => data_len,
        };
        let scaled = count / MXFP8_BLOCK * MXFP8_BLOCK_BYTES;
        if end.is_some_and(|end| end.saturating_sub(tensor.offset) >= scaled) {
            tensor.ggml_type = TYPE_MXFP8;
        }
    }
}

pub fn ggml_type_name(ggml_type: u32) -> String {
    if let Some((_, name)) = PRIVATE_GGML_TYPE_NAMES
        .iter()
        .find(|(id, _)| *id == ggml_type)
    {
        return name.to_string();
    }
    match GGML_TYPE_NAMES.get(ggml_type as usize) {
        Some(Some(name)) => name.to_string(),
        Some(None) => format!("reserved({ggml_type})"),
        None => format!("unknown({ggml_type})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn build_minimal_gguf() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes()); // version
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&2u64.to_le_bytes()); // metadata_kv_count

        // general.architecture = "llama"
        write_string(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes()); // STRING
        write_string(&mut buf, "llama");

        // tokenizer.ggml.tokens = ["a", "b", "c"] (array of strings)
        write_string(&mut buf, "tokenizer.ggml.tokens");
        buf.extend_from_slice(&9u32.to_le_bytes()); // ARRAY
        buf.extend_from_slice(&8u32.to_le_bytes()); // element type STRING
        buf.extend_from_slice(&3u64.to_le_bytes()); // 3 elements
        write_string(&mut buf, "a");
        write_string(&mut buf, "b");
        write_string(&mut buf, "c");

        // One tensor: "weight", shape [2, 4], type F32 (0), offset 0
        write_string(&mut buf, "weight");
        buf.extend_from_slice(&2u32.to_le_bytes()); // n_dims
        buf.extend_from_slice(&2u64.to_le_bytes());
        buf.extend_from_slice(&4u64.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // ggml_type F32
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset

        buf
    }

    fn write_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    /// Builds a complete GGUF — header *and* data section — in a temp file,
    /// so the seek-and-dequantize path is exercised the way it runs.
    fn write_gguf_with_tensor(
        dir: &std::path::Path,
        name: &str,
        dims: &[u64],
        ggml_type: u32,
        data: &[u8],
    ) -> std::path::PathBuf {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // one tensor
        buf.extend_from_slice(&0u64.to_le_bytes()); // no metadata

        write_string(&mut buf, name);
        buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            buf.extend_from_slice(&d.to_le_bytes());
        }
        buf.extend_from_slice(&ggml_type.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes()); // offset within the data

        // The data section starts at the next multiple of the alignment.
        while !buf.len().is_multiple_of(32) {
            buf.push(0);
        }
        buf.extend_from_slice(data);

        let path = dir.join(format!("{name}.gguf"));
        std::fs::write(&path, &buf).expect("writing the fixture");
        path
    }

    /// `Q8_0` is a scale per 32 weights; dequantizing is that scale times
    /// each `int8`, and this checks it against values computed by hand.
    #[test]
    fn reads_a_q8_0_tensor_back_as_f32() {
        let dir = std::env::temp_dir().join("orangu-gguf-q8-test");
        std::fs::create_dir_all(&dir).expect("temp dir");

        let scale = half::f16::from_f32(0.25);
        let mut data = scale.to_bits().to_le_bytes().to_vec();
        let quantized: Vec<i8> = (0..32).map(|i| (i as i8) - 16).collect();
        data.extend(quantized.iter().map(|q| *q as u8));

        let path = write_gguf_with_tensor(&dir, "w", &[32], 8, &data);
        let gguf = GgufFile::open(&path).expect("open");
        let values = gguf.read_tensor(&path, "w").expect("read");

        assert_eq!(values.len(), 32);
        for (value, q) in values.iter().zip(&quantized) {
            assert_eq!(*value, *q as f32 * 0.25);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// `F32` and `F16` come back unchanged and rounded respectively, and a
    /// scalar reads as one value — which is how a projector's calibration
    /// ranges are stored.
    #[test]
    fn reads_float_tensors_and_scalars() {
        let dir = std::env::temp_dir().join("orangu-gguf-float-test");
        std::fs::create_dir_all(&dir).expect("temp dir");

        let wanted = [1.5f32, -2.25, 0.0, 1024.0];
        let mut data = Vec::new();
        for v in wanted {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let path = write_gguf_with_tensor(&dir, "f32s", &[4], 0, &data);
        let gguf = GgufFile::open(&path).expect("open");
        assert_eq!(gguf.read_tensor(&path, "f32s").expect("read"), wanted);

        let mut halves = Vec::new();
        for v in wanted {
            halves.extend_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
        }
        let path = write_gguf_with_tensor(&dir, "f16s", &[4], 1, &halves);
        let gguf = GgufFile::open(&path).expect("open");
        assert_eq!(gguf.read_tensor(&path, "f16s").expect("read"), wanted);

        let path = write_gguf_with_tensor(&dir, "scalar", &[1], 0, &7.5f32.to_le_bytes());
        let gguf = GgufFile::open(&path).expect("open");
        assert_eq!(gguf.read_scalar(&path, "scalar"), Some(7.5));
        assert_eq!(gguf.read_scalar(&path, "absent"), None);

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A type this reader cannot dequantize is an error naming itself, and a
    /// missing tensor is an error rather than an empty vector — either one
    /// silently returning zeros would be indistinguishable from real weights
    /// that happen to be zero.
    #[test]
    fn an_unreadable_tensor_is_an_error() {
        let dir = std::env::temp_dir().join("orangu-gguf-error-test");
        std::fs::create_dir_all(&dir).expect("temp dir");

        // Q4_K (12), which this reader does not handle.
        let path = write_gguf_with_tensor(&dir, "q4k", &[256], 12, &[0u8; 144]);
        let gguf = GgufFile::open(&path).expect("open");
        let err = gguf.read_tensor(&path, "q4k").expect_err("should refuse");
        assert!(err.to_string().contains("Q4_K"), "{err}");

        assert!(gguf.read_tensor(&path, "nope").is_err());

        // A Q8_0 tensor whose length is not a whole number of blocks.
        let path = write_gguf_with_tensor(&dir, "ragged", &[20], 8, &[0u8; 34]);
        let gguf = GgufFile::open(&path).expect("open");
        assert!(gguf.read_tensor(&path, "ragged").is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The summary open skips an array longer than its limit — it is then
    /// absent, not present and empty — keeps every other value, and reads
    /// the tensor table behind the skipped bytes correctly; and its skipping
    /// covers strings (one length each) and fixed-size elements (one seek)
    /// alike. Built over both a file and an in-memory cursor, the two
    /// sources with a skip of their own.
    #[test]
    fn the_summary_open_skips_large_arrays_and_keeps_the_rest() {
        let limit = SUMMARY_ARRAY_LIMIT as usize;
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&1u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&5u64.to_le_bytes()); // metadata_kv_count
        write_string(&mut buf, "general.architecture");
        buf.extend_from_slice(&8u32.to_le_bytes());
        write_string(&mut buf, "clip");
        // A vocabulary-sized string array: skipped.
        write_string(&mut buf, "tokenizer.ggml.tokens");
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&((limit + 1) as u64).to_le_bytes());
        for i in 0..=limit {
            write_string(&mut buf, &format!("tok{i}"));
        }
        // A score table of the same length: skipped in one seek.
        write_string(&mut buf, "tokenizer.ggml.scores");
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&6u32.to_le_bytes()); // F32
        buf.extend_from_slice(&((limit + 1) as u64).to_le_bytes());
        for i in 0..=limit {
            buf.extend_from_slice(&(i as f32).to_le_bytes());
        }
        // A short array: kept.
        write_string(&mut buf, "small.array");
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes()); // U32
        buf.extend_from_slice(&3u64.to_le_bytes());
        for v in [7u32, 8, 9] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        write_string(&mut buf, "general.alignment");
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&32u32.to_le_bytes());
        write_string(&mut buf, "weight");
        buf.extend_from_slice(&1u32.to_le_bytes());
        buf.extend_from_slice(&6u64.to_le_bytes());
        buf.extend_from_slice(&1u32.to_le_bytes()); // F16
        buf.extend_from_slice(&0u64.to_le_bytes());

        let check = |file: &GgufFile| {
            let keys: Vec<&str> = file.metadata.iter().map(|(k, _)| k.as_str()).collect();
            assert_eq!(
                keys,
                ["general.architecture", "small.array", "general.alignment"]
            );
            assert!(file.is_clip_projector());
            assert!(matches!(
                &file.metadata[1].1,
                GgufValue::Array(items) if items.len() == 3
            ));
            assert_eq!(file.tensors.len(), 1);
            assert_eq!(file.tensors[0].name, "weight");
            assert_eq!(file.tensors[0].dims, vec![6]);
            assert_eq!(file.tensors[0].ggml_type, 1);
        };
        let from_cursor =
            GgufFile::read_with(Cursor::new(buf.clone()), Some(SUMMARY_ARRAY_LIMIT), None).unwrap();
        check(&from_cursor);

        let dir = std::env::temp_dir().join("orangu-gguf-summary-test");
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("summary.gguf");
        std::fs::write(&path, &buf).expect("write");
        let from_file = GgufFile::open_summary(&path).expect("open_summary");
        check(&from_file);
        // The full open still sees everything.
        let full = GgufFile::open(&path).expect("open");
        assert_eq!(full.metadata.len(), 5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parses_header_metadata_and_tensors() {
        let bytes = build_minimal_gguf();
        let file = GgufFile::read(Cursor::new(bytes)).unwrap();

        assert_eq!(file.version, 3);
        let architecture = file
            .metadata
            .iter()
            .find(|(key, _)| key == "general.architecture")
            .map(|(_, value)| value);
        assert!(matches!(architecture, Some(GgufValue::String(s)) if s == "llama"));
        assert_eq!(file.tensors.len(), 1);
        assert_eq!(file.tensors[0].name, "weight");
        assert_eq!(file.tensors[0].dims, vec![2, 4]);
        assert_eq!(file.tensors[0].element_count(), 8);
        assert_eq!(file.tensors[0].shape(), "2 x 4");
        assert_eq!(ggml_type_name(file.tensors[0].ggml_type), "F32");
        assert_eq!(file.alignment, 32);

        let tokens = file
            .metadata
            .iter()
            .find(|(key, _)| key == "tokenizer.ggml.tokens")
            .map(|(_, value)| value)
            .unwrap();
        assert!(matches!(tokens, GgufValue::Array(items) if items.len() == 3));
    }

    /// The property `orangu::model_download::RemoteModel::headers` is built
    /// on, and the reason planning a model before downloading it is cheap:
    /// the parser stops at the end of the tensor table and reads none of the
    /// tensor data behind it.
    ///
    /// Stated as a test because it is a promise made to a *network* caller,
    /// where it is the difference between transferring a few hundred
    /// kilobytes and transferring hundreds of gigabytes — and where nothing
    /// about a passing parse would reveal that the promise had been broken.
    /// A parser that read to EOF would still return the right `GgufFile`
    /// here; only the bytes left unread show the difference.
    #[test]
    fn read_from_stops_at_the_end_of_the_tensor_table() {
        let header = build_minimal_gguf();
        let mut bytes = header.clone();
        // Stand-in for the tensor data a real file carries behind its table.
        bytes.extend(std::iter::repeat_n(0xABu8, 4096));

        let mut cursor = Cursor::new(bytes);
        let file = GgufFile::read_from(&mut cursor).unwrap();
        assert_eq!(file.tensors.len(), 1);
        assert_eq!(
            cursor.position(),
            header.len() as u64,
            "the parser read past the tensor table into the tensor data"
        );
    }

    /// The removed ids must still name themselves — a real `bartowski`
    /// repo serves files carrying 31/32/33, and `list`'s `SUPPORTED` column
    /// prints this name, so `reserved(31)` there tells a user nothing about
    /// which file to fetch instead. Ids past the table stay `unknown(N)`:
    /// those are types newer than this build, not retired ones.
    #[test]
    fn removed_ggml_types_are_named_and_carry_their_replacement() {
        assert_eq!(ggml_type_name(31), "Q4_0_4_4");
        assert_eq!(ggml_type_name(33), "Q4_0_8_8");
        assert_eq!(ggml_type_name(36), "IQ4_NL_4_4");
        assert!(is_removed_ggml_type(4));
        assert!(is_removed_ggml_type(5));
        // 31-33 and 36-38 are retired upstream but readable here
        // (de-interleaved at load), so they must NOT be listed as
        // unreadable — doing so would reject a file this build can serve.
        for readable in [31, 32, 33, 36, 37, 38] {
            assert!(!is_removed_ggml_type(readable), "type {readable}");
        }
        // Live types must not be mistaken for retired ones.
        assert!(!is_removed_ggml_type(2));
        assert!(!is_removed_ggml_type(20));
        assert_eq!(ggml_type_name(255), "unknown(255)");
    }

    fn f8_tensor(name: &str, count: u64, offset: u64) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            dims: vec![256, count / 256],
            ggml_type: TYPE_F8_E4M3,
            offset,
        }
    }

    /// Id 43 is plain E4M3 when the tensor ends within an alignment pad of
    /// its element count, and MXFP8 when it spans the extra scale byte per
    /// 32 elements — whether the next tensor or the data's end bounds it.
    #[test]
    fn id_43_reads_as_mxfp8_only_when_its_extent_holds_the_scales() {
        let n = 256 * 64;
        let scaled = n / 32 * 33;
        let mut tensors = vec![
            f8_tensor("plain", n, 0),
            f8_tensor("scaled", n, n),
            f8_tensor("last_scaled", n, n + scaled),
        ];
        resolve_mxfp8(&mut tensors, 32, Some(n + 2 * scaled));
        let types: Vec<u32> = tensors.iter().map(|t| t.ggml_type).collect();
        assert_eq!(types, [TYPE_F8_E4M3, TYPE_MXFP8, TYPE_MXFP8]);

        // The last tensor with no known data length keeps the plain reading,
        // and so does one whose extent is only the plain size.
        let mut tensors = vec![f8_tensor("unknown_end", n, 0)];
        resolve_mxfp8(&mut tensors, 32, None);
        assert_eq!(tensors[0].ggml_type, TYPE_F8_E4M3);
        resolve_mxfp8(&mut tensors, 32, Some(n));
        assert_eq!(tensors[0].ggml_type, TYPE_F8_E4M3);
    }

    /// The three sub-`IQ1_S` types live at 64+, above a gap ggml has not
    /// filled. All three must name themselves, the gap must read as
    /// reserved rather than as a real type, and the last id ggml has
    /// actually used must still be the one it says it is — an off-by-one in
    /// the run of `None`s would shift every name after it.
    #[test]
    fn narrow_iq1_types_are_named_above_ggmls_reserved_range() {
        assert_eq!(ggml_type_name(42), "Q2_0");
        assert_eq!(ggml_type_name(43), "F8_E4M3");
        assert_eq!(ggml_type_name(44), "reserved(44)");
        assert_eq!(ggml_type_name(63), "reserved(63)");
        assert_eq!(ggml_type_name(64), "IQ1_XS");
        assert_eq!(ggml_type_name(65), "IQ1_XXS");
        assert_eq!(ggml_type_name(66), "IQ1_XXXS");
        assert_eq!(ggml_type_name(67), "unknown(67)");
        assert_eq!(ggml_type_name(TYPE_MXFP8), "MXFP8");
        for reserved in 43..64 {
            assert!(!is_removed_ggml_type(reserved), "type {reserved}");
        }
    }

    /// Prism's two ternary ids live far above the table. Both must name
    /// themselves — `list`'s `QUANT` column prints this for a
    /// `Ternary-Bonsai-2-27B-PTQ1_0.gguf` — and the ids around them must
    /// stay `unknown`, not silently inherit a neighbour's name.
    #[test]
    fn prism_ternary_types_are_named_by_id() {
        assert_eq!(ggml_type_name(142), "PQ2_0");
        assert_eq!(ggml_type_name(143), "PTQ1_0");
        assert_eq!(ggml_type_name(141), "unknown(141)");
        assert_eq!(ggml_type_name(145), "unknown(145)");
        assert!(!is_removed_ggml_type(142));
        assert!(!is_removed_ggml_type(143));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = build_minimal_gguf();
        bytes[0] = b'X';
        let err = GgufFile::read(Cursor::new(bytes)).unwrap_err();
        assert!(err.to_string().contains("not a GGUF file"), "{err}");
    }

    #[test]
    fn rejects_truncated_file() {
        let bytes = build_minimal_gguf();
        let truncated = &bytes[..bytes.len() - 10];
        assert!(GgufFile::read(Cursor::new(truncated.to_vec())).is_err());
    }

    #[test]
    fn array_display_truncates_long_arrays() {
        let value = GgufValue::Array(vec![
            GgufValue::U32(1),
            GgufValue::U32(2),
            GgufValue::U32(3),
        ]);
        assert_eq!(value.display(2), "[1, 2, ... (1 more)] (3 total)");
        assert_eq!(value.display(usize::MAX), "[1, 2, 3] (3 total)");
    }

    #[test]
    fn type_element_totals_weighs_by_element_count_not_tensor_count() {
        let file = GgufFile {
            version: 3,
            metadata: Vec::new(),
            tensors: vec![
                TensorInfo {
                    name: "norm".to_string(),
                    dims: vec![8],
                    ggml_type: 0, // F32, few elements
                    offset: 0,
                },
                TensorInfo {
                    name: "attn.weight".to_string(),
                    dims: vec![4096, 4096],
                    ggml_type: 12, // Q4_K, dominates by element count
                    offset: 0,
                },
            ],
            alignment: 32,
            data_offset: 0,
        };
        let totals = file.type_element_totals();
        let dominant = totals
            .into_iter()
            .max_by_key(|(_, total)| *total)
            .map(|(ty, _)| ty);
        assert_eq!(dominant, Some(12));
    }
}
