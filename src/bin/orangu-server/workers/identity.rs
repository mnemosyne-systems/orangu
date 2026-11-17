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

//! Making sure every node of a tree has the same model installed, down to
//! the quantization.
//!
//! A parent sends its own [`ModelIdentity`] for the range it assigns; the
//! worker computes its own from the file it resolved, refuses the
//! assignment when they differ, and returns its identity in `AssignAck`,
//! which the parent checks again — a parent does not take a worker's word
//! for it.
//!
//! What is compared is the file's content, never its name: a renamed copy
//! is the same model, and two files both called `Q4_K_M` are not
//! necessarily the same weights.
//!
//! - [`ModelIdentity::header_hash`] covers the architecture's
//!   hyperparameters (every `<arch>.*` metadata key) and the whole tensor
//!   directory: each tensor's name, ggml type and shape. A different
//!   quantization — `Q4_K_M` against `Q4_K_S`, `Q8_0`, `F16` — stores
//!   tensors at different types, so it never shares this hash.
//! - [`ModelIdentity::range_hash`] covers the weights of the layers in
//!   question, sampled: the first, middle and last [`SAMPLE_BYTES`] of each
//!   of their tensors. Same layout, different weights — another release,
//!   another importance matrix — differ here. Sampling, not hashing every
//!   byte, keeps the check to a few megabytes of reading however large the
//!   model is.
//!
//! `general.*`, `tokenizer.*` and `quantize.*` metadata is left out: a name
//! or a URL can differ between two copies of the same weights, and only the
//! top-level node tokenizes.

use super::protocol::{ErrorCode, ModelIdentity, WorkerError};
use crate::engine::loader::{LoadedModel, block_index};
use orangu::gguf::GgufValue;
use sha2::{Digest, Sha256};
use std::ops::Range;
use std::path::Path;

/// Bytes hashed from each end, and from the middle, of every tensor in the
/// range.
pub const SAMPLE_BYTES: usize = 4096;

/// `loaded`'s identity for `layers`. `label` and `quant` only name it in
/// messages.
pub fn model_identity(
    loaded: &LoadedModel,
    label: &str,
    quant: &str,
    layers: Range<usize>,
) -> ModelIdentity {
    ModelIdentity {
        label: label.to_string(),
        quant: quant.to_string(),
        header_hash: header_hash(loaded),
        range_hash: range_hash(loaded, layers),
    }
}

/// The quantization of the file at `path`, as `list` shows it: the tag in
/// its name, or else the ggml type most of its weights are stored as.
pub fn quantization_of(path: &Path) -> String {
    orangu::gguf::GgufFile::open(path)
        .ok()
        .and_then(|gguf| orangu::model_spec::quantization_for_file(path, &gguf))
        .unwrap_or_else(|| "unknown".to_string())
}

/// Whether `loaded` has the hyperparameters and tensor directory `expected`
/// names — the same model, at least in shape — without hashing any weights.
pub fn same_layout(loaded: &LoadedModel, expected: &ModelIdentity) -> bool {
    header_hash(loaded) == expected.header_hash
}

/// [`ModelIdentity::header_hash`] of `loaded`.
pub fn header_hash(loaded: &LoadedModel) -> [u8; 32] {
    let mut hash = Sha256::new();
    let prefix = format!("{}.", loaded.config.architecture);
    hash.update(loaded.config.architecture.as_bytes());
    let mut keys: Vec<&(String, GgufValue)> = loaded
        .metadata
        .iter()
        .filter(|(key, _)| key.starts_with(&prefix))
        .collect();
    keys.sort_by(|a, b| a.0.cmp(&b.0));
    for (key, value) in keys {
        field(&mut hash, key.as_bytes());
        value_bytes(&mut hash, value);
    }
    for (name, ggml_type, dims) in loaded.tensor_directory() {
        field(&mut hash, name.as_bytes());
        hash.update(ggml_type.to_le_bytes());
        hash.update((dims.len() as u32).to_le_bytes());
        for d in dims {
            hash.update(d.to_le_bytes());
        }
    }
    hash.finalize().into()
}

fn range_hash(loaded: &LoadedModel, layers: Range<usize>) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update((layers.start as u64).to_le_bytes());
    hash.update((layers.end as u64).to_le_bytes());
    for (name, _, _) in loaded.tensor_directory() {
        if !block_index(name).is_some_and(|il| layers.contains(&il)) {
            continue;
        }
        let Some(data) = loaded.tensor_data(name) else {
            continue;
        };
        field(&mut hash, name.as_bytes());
        hash.update((data.len() as u64).to_le_bytes());
        for sample in samples(data) {
            hash.update(sample);
        }
    }
    hash.finalize().into()
}

/// The parts of a tensor's `data` [`ModelIdentity::range_hash`] reads: all
/// of it when small, else its first, middle and last [`SAMPLE_BYTES`].
fn samples(data: &[u8]) -> Vec<&[u8]> {
    if data.len() <= 3 * SAMPLE_BYTES {
        return vec![data];
    }
    let middle = data.len() / 2 - SAMPLE_BYTES / 2;
    vec![
        &data[..SAMPLE_BYTES],
        &data[middle..middle + SAMPLE_BYTES],
        &data[data.len() - SAMPLE_BYTES..],
    ]
}

/// Every slice of `loaded`'s mapped file that [`ModelIdentity::range_hash`]
/// reads for `layers`, as `(offset, len)` in the file: what a node fetching
/// the model in part fetches to check the weights it was sent against a
/// peer's hash before it needs them whole. A tensor that is not read
/// through the mapping (a rewritten repack) has none.
pub fn sampled_ranges(loaded: &LoadedModel, layers: Range<usize>) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    for (name, _, _) in loaded.tensor_directory() {
        if !block_index(name).is_some_and(|il| layers.contains(&il)) {
            continue;
        }
        let Some(data) = loaded.tensor_data(name) else {
            continue;
        };
        for sample in samples(data) {
            if let Some((_, at)) =
                crate::engine::page_cache::locate(sample.as_ptr() as usize, sample.len())
            {
                out.push((at, sample.len() as u64));
            }
        }
    }
    out.sort_unstable();
    out
}

/// A length-prefixed field, so no two sequences of fields hash alike.
fn field(hash: &mut Sha256, bytes: &[u8]) {
    hash.update((bytes.len() as u64).to_le_bytes());
    hash.update(bytes);
}

/// A metadata value in a fixed encoding of its own, rather than whatever
/// `Debug` prints this release.
fn value_bytes(hash: &mut Sha256, value: &GgufValue) {
    match value {
        GgufValue::U8(v) => hash.update([0, *v]),
        GgufValue::I8(v) => hash.update([1, *v as u8]),
        GgufValue::U16(v) => {
            hash.update([2]);
            hash.update(v.to_le_bytes());
        }
        GgufValue::I16(v) => {
            hash.update([3]);
            hash.update(v.to_le_bytes());
        }
        GgufValue::U32(v) => {
            hash.update([4]);
            hash.update(v.to_le_bytes());
        }
        GgufValue::I32(v) => {
            hash.update([5]);
            hash.update(v.to_le_bytes());
        }
        GgufValue::F32(v) => {
            hash.update([6]);
            hash.update(v.to_le_bytes());
        }
        GgufValue::Bool(v) => hash.update([7, u8::from(*v)]),
        GgufValue::String(v) => {
            hash.update([8]);
            field(hash, v.as_bytes());
        }
        GgufValue::U64(v) => {
            hash.update([10]);
            hash.update(v.to_le_bytes());
        }
        GgufValue::I64(v) => {
            hash.update([11]);
            hash.update(v.to_le_bytes());
        }
        GgufValue::F64(v) => {
            hash.update([12]);
            hash.update(v.to_le_bytes());
        }
        GgufValue::Array(items) => {
            hash.update([9]);
            hash.update((items.len() as u64).to_le_bytes());
            for item in items {
                value_bytes(hash, item);
            }
        }
    }
}

/// Refuses `actual` unless it is `expected`'s model, saying which way it
/// differs. `layers` is the range the two were computed for.
pub fn check(
    expected: &ModelIdentity,
    actual: &ModelIdentity,
    layers: &Range<usize>,
) -> Result<(), WorkerError> {
    if expected.header_hash != actual.header_hash {
        let message = if expected.quant != actual.quant {
            format!(
                "the parent serves {} ({}) but this node has {} ({}): a different quantization",
                expected.label, expected.quant, actual.label, actual.quant
            )
        } else {
            format!(
                "the parent serves {} ({}) but this node has {} ({}): the same quantization tag, \
                 but a different file layout",
                expected.label, expected.quant, actual.label, actual.quant
            )
        };
        return Err(WorkerError::new(ErrorCode::ModelMismatch, message));
    }
    if expected.range_hash != actual.range_hash {
        return Err(WorkerError::new(
            ErrorCode::ModelMismatch,
            format!(
                "the parent serves {} ({}) and this node has {} ({}) with the same layout, but the \
                 weights of layers {}..{} differ: another release or quantization run of the model",
                expected.label,
                expected.quant,
                actual.label,
                actual.quant,
                layers.start,
                layers.end
            ),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::workers::pipeline::fixture::{self, N_LAYER, Variant};

    fn identity(variant: Variant, layers: Range<usize>) -> ModelIdentity {
        let quant = if variant.f16_ffn_down { "F16" } else { "F32" };
        model_identity(&fixture::loaded(variant), "fixture", quant, layers)
    }

    /// The same file, loaded twice — as two nodes would — is the same
    /// model; so is a copy under another name.
    #[test]
    fn the_same_file_is_the_same_model() {
        let a = identity(Variant::default(), 1..3);
        let b = identity(Variant::default(), 1..3);
        assert_eq!(a, b);
        assert!(check(&a, &b, &(1..3)).is_ok());
        let renamed = ModelIdentity {
            label: "copy".to_string(),
            ..b
        };
        assert!(check(&a, &renamed, &(1..3)).is_ok());
    }

    /// Another quantization is refused whichever layers are compared, and
    /// the message names both quantizations.
    #[test]
    fn another_quantization_is_refused() {
        let quantized = Variant {
            f16_ffn_down: true,
            ..Variant::default()
        };
        for layers in [0..1, 2..N_LAYER] {
            let parent = identity(Variant::default(), layers.clone());
            let worker = identity(quantized, layers.clone());
            let err = check(&parent, &worker, &layers).unwrap_err();
            assert_eq!(err.code, ErrorCode::ModelMismatch);
            assert!(
                err.message.contains("F32") && err.message.contains("F16"),
                "{err}"
            );
            assert!(err.message.contains("different quantization"), "{err}");
        }
    }

    /// The same quantization tag on a different layout is still refused.
    #[test]
    fn a_different_layout_under_the_same_tag_is_refused() {
        let parent = identity(Variant::default(), 0..1);
        let worker = ModelIdentity {
            quant: parent.quant.clone(),
            ..identity(
                Variant {
                    f16_ffn_down: true,
                    ..Variant::default()
                },
                0..1,
            )
        };
        let err = check(&parent, &worker, &(0..1)).unwrap_err();
        assert!(err.message.contains("different file layout"), "{err}");
    }

    /// Different weights in the same layout are refused for a range that
    /// holds them, and — since a worker only runs its own range — accepted
    /// for one that does not.
    #[test]
    fn different_weights_are_refused_where_they_are_used() {
        let nudged = Variant {
            nudge_layer: Some(2),
            ..Variant::default()
        };
        let parent = identity(Variant::default(), 1..3);
        let worker = identity(nudged, 1..3);
        let err = check(&parent, &worker, &(1..3)).unwrap_err();
        assert!(err.message.contains("layers 1..3"), "{err}");
        let parent = identity(Variant::default(), 3..N_LAYER);
        let worker = identity(nudged, 3..N_LAYER);
        assert!(check(&parent, &worker, &(3..N_LAYER)).is_ok());
    }

    /// A range hash is for its range: the same file's identity for other
    /// layers does not stand in for it.
    #[test]
    fn an_identity_is_for_its_own_range() {
        let a = identity(Variant::default(), 1..3);
        let b = identity(Variant::default(), 1..4);
        assert!(check(&a, &b, &(1..3)).is_err());
    }

    /// Two quantizations of one real model are told apart, and the check
    /// stays cheap on a real file. Run with
    /// `ORANGU_TEST_MODEL_A=a.gguf ORANGU_TEST_MODEL_B=b.gguf cargo test
    /// --release --bin orangu-server real_quantizations -- --ignored
    /// --nocapture`, `a` and `b` two quantizations of one model.
    #[test]
    #[ignore]
    fn real_quantizations_are_told_apart() {
        let open = |var: &str| {
            let path = std::path::PathBuf::from(std::env::var(var).expect(var));
            let loaded = LoadedModel::open(&path).unwrap();
            let layers = 0..loaded.config.n_layer;
            let at = std::time::Instant::now();
            let identity = model_identity(&loaded, var, &quantization_of(&path), layers.clone());
            println!("{var}: {} in {:?}", identity.quant, at.elapsed());
            (identity, layers)
        };
        let (a, layers) = open("ORANGU_TEST_MODEL_A");
        let (again, _) = open("ORANGU_TEST_MODEL_A");
        assert!(check(&a, &again, &layers).is_ok());
        let (b, _) = open("ORANGU_TEST_MODEL_B");
        let err = check(&a, &b, &layers).unwrap_err();
        println!("{err}");
        assert_eq!(err.code, ErrorCode::ModelMismatch);
    }

    #[test]
    fn the_quantization_is_read_off_the_file() {
        let path = fixture::path(Variant::default());
        assert_eq!(quantization_of(&path), "F32");
        assert_eq!(quantization_of(Path::new("/nonexistent.gguf")), "unknown");
    }
}
