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

//! Looking at a model file without loading it: `--dump` prints its
//! metadata and tensor table, `--hash` a content hash per tensor, and
//! `--compare` says where two files differ.
//!
//! `--compare` is what a conversion is checked with. Two files of the same
//! model at the same weight format must hold the same bytes in every
//! tensor; where the formats differ, the tensors are read back as floats
//! and the largest difference is reported instead. Metadata is compared
//! key by key, and a key present on one side only is named — a converter
//! that forgets a key produces a file that loads and answers slightly
//! wrong, which no tensor comparison would catch.

use anyhow::{Context, Result};
use memmap2::Mmap;
use orangu::gguf::{GgufFile, GgufValue, TensorInfo};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs::File, path::Path};

/// An open file and its mapped bytes.
struct Mapped {
    file: GgufFile,
    map: Mmap,
}

impl Mapped {
    fn open(path: &Path) -> Result<Self> {
        let file = GgufFile::open(path).with_context(|| format!("reading {}", path.display()))?;
        let handle = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        // Safety: opened read-only; this process never writes to it.
        let map =
            unsafe { Mmap::map(&handle) }.with_context(|| format!("mapping {}", path.display()))?;
        Ok(Self { file, map })
    }

    /// Every tensor's stored bytes, keyed by name.
    ///
    /// A tensor's extent is its type's size where this tool knows it, and
    /// otherwise runs to the next tensor (or the end of the file) — which
    /// includes the alignment padding, and that is zeros in every writer.
    fn tensor_bytes(&self) -> BTreeMap<&str, (&TensorInfo, &[u8])> {
        let base = self.file.data_offset as usize;
        let mut offsets: Vec<u64> = self.file.tensors.iter().map(|t| t.offset).collect();
        offsets.sort_unstable();
        let file_end = self.map.len().saturating_sub(base) as u64;
        self.file
            .tensors
            .iter()
            .map(|t| {
                let next = offsets
                    .iter()
                    .find(|&&o| o > t.offset)
                    .copied()
                    .unwrap_or(file_end);
                let len = orangu::dequantize::tensor_byte_size(t.ggml_type, t.element_count())
                    .unwrap_or(next - t.offset);
                let start = (base as u64 + t.offset) as usize;
                let end = (start + len as usize).min(self.map.len());
                (t.name.as_str(), (t, &self.map[start.min(end)..end]))
            })
            .collect()
    }
}

/// `--dump`: the header, every metadata key, and the tensor table.
pub fn dump(path: &Path) -> Result<()> {
    let mapped = Mapped::open(path)?;
    let file = &mapped.file;
    println!(
        "{}: GGUF v{}, {} metadata keys, {} tensors, alignment {}",
        path.display(),
        file.version,
        file.metadata.len(),
        file.tensors.len(),
        file.alignment
    );
    println!();
    for (key, value) in &file.metadata {
        println!("{key:<48} {}", value.display(8));
    }
    println!();
    for tensor in &file.tensors {
        println!(
            "{:<48} {:<8} {}",
            tensor.name,
            orangu::gguf::ggml_type_name(tensor.ggml_type),
            tensor.shape()
        );
    }
    Ok(())
}

/// `--hash`: a SHA-256 of each tensor's stored bytes, in file order.
pub fn hash(path: &Path) -> Result<()> {
    let mapped = Mapped::open(path)?;
    let bytes = mapped.tensor_bytes();
    let hashes: Vec<(String, String)> = mapped
        .file
        .tensors
        .par_iter()
        .map(|t| {
            (
                t.name.clone(),
                hex(&Sha256::digest(bytes[t.name.as_str()].1)),
            )
        })
        .collect();
    for (name, digest) in hashes {
        println!("{digest}  {name}");
    }
    Ok(())
}

/// What `--compare` found.
#[derive(Debug, Default, PartialEq)]
pub struct Comparison {
    /// Keys whose values differ, with both renderings.
    pub metadata_differs: Vec<(String, String, String)>,
    pub metadata_only_a: Vec<String>,
    pub metadata_only_b: Vec<String>,
    pub tensors_only_a: Vec<String>,
    pub tensors_only_b: Vec<String>,
    /// Same name, different shape.
    pub shape_differs: Vec<(String, String, String)>,
    /// Same name and type, different bytes.
    pub bytes_differ: Vec<String>,
    /// For each of [`Self::bytes_differ`] whose type reads back as floats,
    /// the largest absolute difference and the largest value.
    pub bytes_difference: BTreeMap<String, (f32, f32)>,
    /// Same name, different type: the largest absolute difference read
    /// back as floats, or `None` when one side's type cannot be read back.
    /// Against a quantized file this is the format's rounding when the
    /// tensors hold the same weights, and the size of the weights
    /// themselves when they do not.
    pub type_differs: Vec<TypeDiffers>,
    pub identical: usize,
}

impl Comparison {
    pub fn is_identical(&self) -> bool {
        self.metadata_differs.is_empty()
            && self.metadata_only_a.is_empty()
            && self.metadata_only_b.is_empty()
            && self.tensors_only_a.is_empty()
            && self.tensors_only_b.is_empty()
            && self.shape_differs.is_empty()
            && self.bytes_differ.is_empty()
            && self.type_differs.is_empty()
    }
}

/// One tensor stored at a different type on each side.
#[derive(Debug, PartialEq)]
pub struct TypeDiffers {
    pub name: String,
    pub left: String,
    pub right: String,
    /// The largest absolute difference read back as floats, and the
    /// largest absolute value on the left; `None` when either side cannot
    /// be read back.
    pub difference: Option<(f32, f32)>,
}

/// Metadata keys that say how and by what a file was written rather than
/// what the model is. They differ between any two writers and are not
/// reported.
const PROVENANCE_KEYS: [&str; 2] = ["general.quantization_version", "general.file_type"];

/// Compares two files.
pub fn compare_files(a: &Path, b: &Path) -> Result<Comparison> {
    let left = Mapped::open(a)?;
    let right = Mapped::open(b)?;
    let mut out = Comparison::default();

    let meta = |file: &GgufFile| -> BTreeMap<String, GgufValue> {
        file.metadata
            .iter()
            .filter(|(k, _)| !PROVENANCE_KEYS.contains(&k.as_str()))
            .cloned()
            .collect()
    };
    let (ma, mb) = (meta(&left.file), meta(&right.file));
    for (key, va) in &ma {
        match mb.get(key) {
            None => out.metadata_only_a.push(key.clone()),
            Some(vb) if !same_value(va, vb) => out.metadata_differs.push(match (va, vb) {
                (GgufValue::Array(x), GgufValue::Array(y)) => {
                    let at = x
                        .iter()
                        .zip(y)
                        .position(|(p, q)| !same_value(p, q))
                        .unwrap_or(x.len().min(y.len()));
                    let item =
                        |v: &[GgufValue]| v.get(at).map_or("(none)".to_string(), |e| e.display(4));
                    (
                        format!("{key}[{at}]"),
                        format!("{} (of {})", item(x), x.len()),
                        format!("{} (of {})", item(y), y.len()),
                    )
                }
                _ => (key.clone(), va.display(4), vb.display(4)),
            }),
            Some(_) => {}
        }
    }
    out.metadata_only_b = mb
        .keys()
        .filter(|k| !ma.contains_key(*k))
        .cloned()
        .collect();

    let (ta, tb) = (left.tensor_bytes(), right.tensor_bytes());
    out.tensors_only_a = ta
        .keys()
        .filter(|k| !tb.contains_key(*k))
        .map(|k| k.to_string())
        .collect();
    out.tensors_only_b = tb
        .keys()
        .filter(|k| !ta.contains_key(*k))
        .map(|k| k.to_string())
        .collect();

    enum Verdict {
        Same,
        Shape(String, String),
        Bytes(Option<(f32, f32)>),
        Type(String, String, Option<(f32, f32)>),
    }
    let difference =
        |ia: &orangu::gguf::TensorInfo, ba: &[u8], ib: &orangu::gguf::TensorInfo, bb: &[u8]| {
            let elements = ia.element_count() as usize;
            match (
                orangu::dequantize::dequantize(ia.ggml_type, ba, elements),
                orangu::dequantize::dequantize(ib.ggml_type, bb, elements),
            ) {
                (Ok(x), Ok(y)) => Some((
                    x.iter()
                        .zip(&y)
                        .map(|(p, q)| (p - q).abs())
                        .fold(0.0f32, f32::max),
                    x.iter().map(|v| v.abs()).fold(0.0f32, f32::max),
                )),
                _ => None,
            }
        };
    let verdicts: Vec<(String, Verdict)> = ta
        .par_iter()
        .filter_map(|(name, (ia, ba))| {
            let (ib, bb) = tb.get(name)?;
            let verdict = if ia.dims != ib.dims {
                Verdict::Shape(ia.shape(), ib.shape())
            } else if ia.ggml_type == ib.ggml_type {
                if ba == bb {
                    Verdict::Same
                } else {
                    Verdict::Bytes(difference(ia, ba, ib, bb))
                }
            } else {
                let diff = difference(ia, ba, ib, bb);
                Verdict::Type(
                    orangu::gguf::ggml_type_name(ia.ggml_type),
                    orangu::gguf::ggml_type_name(ib.ggml_type),
                    diff,
                )
            };
            Some((name.to_string(), verdict))
        })
        .collect();
    for (name, verdict) in verdicts {
        match verdict {
            Verdict::Same => out.identical += 1,
            Verdict::Shape(x, y) => out.shape_differs.push((name, x, y)),
            Verdict::Bytes(difference) => {
                if let Some(d) = difference {
                    out.bytes_difference.insert(name.clone(), d);
                }
                out.bytes_differ.push(name)
            }
            Verdict::Type(left, right, difference) => out.type_differs.push(TypeDiffers {
                name,
                left,
                right,
                difference,
            }),
        }
    }
    Ok(out)
}

/// Values compare equal across integer widths: one writer's `U32` context
/// length is another's `U64`, and both mean the same model.
fn same_value(a: &GgufValue, b: &GgufValue) -> bool {
    match (a.as_u64(), b.as_u64()) {
        (Some(x), Some(y)) => x == y,
        _ => a.display(usize::MAX) == b.display(usize::MAX),
    }
}

/// `--compare`: prints the differences, and fails when there are any.
pub fn compare(a: &Path, b: &Path) -> Result<()> {
    let found = compare_files(a, b)?;
    for (key, x, y) in &found.metadata_differs {
        println!("metadata  {key}: {x}  ≠  {y}");
    }
    for key in &found.metadata_only_a {
        println!("metadata  {key}: only in {}", a.display());
    }
    for key in &found.metadata_only_b {
        println!("metadata  {key}: only in {}", b.display());
    }
    for name in &found.tensors_only_a {
        println!("tensor    {name}: only in {}", a.display());
    }
    for name in &found.tensors_only_b {
        println!("tensor    {name}: only in {}", b.display());
    }
    for (name, x, y) in &found.shape_differs {
        println!("tensor    {name}: shape {x}  ≠  {y}");
    }
    for name in &found.bytes_differ {
        match found.bytes_difference.get(name) {
            Some((d, m)) => println!(
                "tensor    {name}: different bytes, largest difference {d:.3e} ({:.2}% of the largest value)",
                100.0 * d / m.max(f32::MIN_POSITIVE)
            ),
            None => println!("tensor    {name}: different bytes"),
        }
    }
    for TypeDiffers {
        name,
        left: x,
        right: y,
        difference,
    } in &found.type_differs
    {
        match difference {
            Some((d, m)) => println!(
                "tensor    {name}: {x} vs {y}, largest difference {d:.3e} ({:.2}% of the largest value)",
                100.0 * d / m.max(f32::MIN_POSITIVE)
            ),
            None => println!("tensor    {name}: {x} vs {y}"),
        }
    }
    println!(
        "{} tensors identical, {} differ",
        found.identical,
        found.bytes_differ.len()
            + found.type_differs.len()
            + found.shape_differs.len()
            + found.tensors_only_a.len()
            + found.tensors_only_b.len()
    );
    if !found.is_identical() {
        anyhow::bail!("the files differ");
    }
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant;
    use crate::write::{self, TensorPlan};

    fn write_file(path: &Path, meta: Vec<(String, GgufValue)>, values: &[f32], ggml_type: u32) {
        let plans = vec![TensorPlan {
            name: "w".into(),
            dims: vec![values.len() as u64],
            ggml_type,
        }];
        write::write(path, &meta, &plans, |_| Ok(values.to_vec()), &|_, _, _| {}).unwrap();
    }

    #[test]
    fn identical_files_compare_identical_and_a_changed_value_does_not() {
        let dir = tempfile::tempdir().unwrap();
        let meta = vec![("general.name".to_string(), GgufValue::String("m".into()))];
        let (a, b, c) = (
            dir.path().join("a.gguf"),
            dir.path().join("b.gguf"),
            dir.path().join("c.gguf"),
        );
        write_file(
            &a,
            meta.clone(),
            &[1.0, 2.0, 3.0, 4.0],
            quant::GGML_TYPE_F16,
        );
        write_file(
            &b,
            meta.clone(),
            &[1.0, 2.0, 3.0, 4.0],
            quant::GGML_TYPE_F16,
        );
        write_file(&c, meta, &[1.0, 2.0, 3.0, 5.0], quant::GGML_TYPE_F16);
        let same = compare_files(&a, &b).unwrap();
        assert!(same.is_identical(), "{same:?}");
        assert_eq!(same.identical, 1);
        let changed = compare_files(&a, &c).unwrap();
        assert_eq!(changed.bytes_differ, vec!["w".to_string()]);
    }

    #[test]
    fn a_missing_key_and_a_different_type_are_named() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (dir.path().join("a.gguf"), dir.path().join("b.gguf"));
        write_file(
            &a,
            vec![("general.name".to_string(), GgufValue::String("m".into()))],
            &[1.0, 2.0],
            quant::GGML_TYPE_F32,
        );
        write_file(&b, vec![], &[1.0, 2.5], quant::GGML_TYPE_F16);
        let found = compare_files(&a, &b).unwrap();
        assert_eq!(found.metadata_only_a, vec!["general.name".to_string()]);
        assert_eq!(found.type_differs.len(), 1);
        assert_eq!(found.type_differs[0].difference, Some((0.5, 2.0)));
    }

    #[test]
    fn integer_widths_do_not_count_as_a_difference() {
        assert!(same_value(&GgufValue::U32(7), &GgufValue::U64(7)));
        assert!(!same_value(&GgufValue::U32(7), &GgufValue::U32(8)));
    }
}
