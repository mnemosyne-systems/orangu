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
        Self::read(BufReader::new(file))
            .with_context(|| format!("failed to parse GGUF file {}", path.display()))
    }

    fn read<R: Read>(inner: R) -> Result<GgufFile> {
        let mut reader = Reader {
            inner,
            bytes_read: 0,
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
}

impl<R: Read> Reader<R> {
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
                let elem_type = self.read_u32()?;
                let len = self.read_u64()?;
                if len > MAX_ARRAY_ELEMENTS {
                    bail!(
                        "array metadata value of {len} elements exceeds the {MAX_ARRAY_ELEMENTS}-element limit"
                    );
                }
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
    None,               // 43 \
    None,               // 44 |
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
pub fn is_removed_ggml_type(ggml_type: u32) -> bool {
    REMOVED_GGML_TYPES.contains(&ggml_type)
}

pub fn ggml_type_name(ggml_type: u32) -> String {
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

    /// The three sub-`IQ1_S` types live at 64+, above a gap ggml has not
    /// filled. All three must name themselves, the gap must read as
    /// reserved rather than as a real type, and the last id ggml has
    /// actually used must still be the one it says it is — an off-by-one in
    /// the run of `None`s would shift every name after it.
    #[test]
    fn narrow_iq1_types_are_named_above_ggmls_reserved_range() {
        assert_eq!(ggml_type_name(42), "Q2_0");
        assert_eq!(ggml_type_name(43), "reserved(43)");
        assert_eq!(ggml_type_name(63), "reserved(63)");
        assert_eq!(ggml_type_name(64), "IQ1_XS");
        assert_eq!(ggml_type_name(65), "IQ1_XXS");
        assert_eq!(ggml_type_name(66), "IQ1_XXXS");
        assert_eq!(ggml_type_name(67), "unknown(67)");
        for reserved in 43..64 {
            assert!(!is_removed_ggml_type(reserved), "type {reserved}");
        }
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
