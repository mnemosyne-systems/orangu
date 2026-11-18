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

//! Importance matrices: how much each input column of each weight matrix
//! matters, measured as the mean squared activation that reached it over a
//! calibration text.
//!
//! Below about three bits a quantization has too few levels to fit every
//! weight well, and the fit has to choose which errors to accept; the
//! matrix tells it which columns' errors cost the most. Two file formats
//! carry one:
//!
//! - **GGUF**: per tensor `<name>.in_sum2`, the summed squares, one row per
//!   expert, and `<name>.counts`, the rows each expert saw.
//! - **The older binary format**: a count of entries, then per entry its
//!   name, its call count and its values, already scaled to the call count.
//!
//! Both read to the same thing: per tensor, one importance per input column
//! (per expert for a mixture), the summed square over the rows that were
//! counted. An expert no calibration row reached gets a flat 1 — no
//! information, so no preference.

use crate::write::TensorPlan;
use anyhow::{Context, Result, anyhow, bail};
use orangu::gguf::{GgufFile, GgufValue};
use std::{
    collections::HashMap,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

/// A loaded importance matrix.
#[derive(Debug, Default)]
pub struct Imatrix {
    /// Per tensor name, `columns × experts` importances, expert-major.
    pub entries: HashMap<String, Vec<f32>>,
    /// The calibration texts it was measured over.
    pub datasets: Vec<String>,
    /// How many chunks of text were run.
    pub chunks: u32,
}

impl Imatrix {
    /// The importances for `tensor`, if the matrix has them.
    pub fn get(&self, tensor: &str) -> Option<&[f32]> {
        self.entries.get(tensor).map(Vec::as_slice)
    }
}

/// Where `spec` names an importance matrix: a file, or a Hugging Face
/// repository whose matrix file is found in the models directory or
/// downloaded into it.
pub fn locate(spec: &Path) -> Result<PathBuf> {
    if spec.is_file() {
        return Ok(spec.to_path_buf());
    }
    let Some(repo) = spec.to_str().filter(|s| s.split('/').count() == 2) else {
        bail!("{} is not an importance matrix file", spec.display());
    };
    let models = crate::hf::models_dir().ok_or_else(|| {
        anyhow!(
            "{repo} is not a file here, and no models directory is configured to find or \
             download it in ([orangu-server].models in orangu-server.conf)"
        )
    })?;
    let folder = models.join(orangu::model_download::repo_folder_name(repo));
    let local = std::fs::read_to_string(folder.join("refs").join("main"))
        .ok()
        .map(|commit| folder.join("snapshots").join(commit.trim()))
        .and_then(|dir| matrix_in(&dir));
    if let Some(path) = local {
        return Ok(path);
    }
    println!(
        "downloading the importance matrix of {repo} into {}",
        models.display()
    );
    let dir = orangu::model_download::download_repo_files(&models, repo, &is_matrix_file, None)?;
    matrix_in(&dir).ok_or_else(|| anyhow!("{repo} publishes no importance matrix"))
}

/// Whether a repository file is an importance matrix, by the names
/// publishers give them.
fn is_matrix_file(name: &str) -> bool {
    let name = name.to_ascii_lowercase();
    !name.contains('/')
        && (name.ends_with(".imatrix")
            || (name.contains("imatrix")
                && [".gguf", ".dat", ".gguf_file"]
                    .iter()
                    .any(|e| name.ends_with(e))))
}

/// The first importance matrix file in `dir`, by name.
fn matrix_in(dir: &Path) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_str().is_some_and(is_matrix_file))
        .map(|e| e.path())
        .collect();
    found.sort();
    found.into_iter().next()
}

impl Imatrix {
    /// Checks every entry a planned tensor has against its shape — one set
    /// of column importances per expert, the rows split evenly between
    /// them — and returns how many planned tensors have one.
    pub fn check(&self, plans: &[TensorPlan]) -> Result<usize> {
        let mut matched = 0;
        for plan in plans {
            let Some(values) = self.get(&plan.name) else {
                continue;
            };
            let ncols = plan.ncols();
            let rows = plan.elements() / ncols.max(1);
            let experts = values.len() / ncols.max(1);
            if experts == 0 || values.len() != experts * ncols || !rows.is_multiple_of(experts) {
                bail!(
                    "the importance matrix has {} values for {}, which has rows of {ncols} \
                     — is it for another model?",
                    values.len(),
                    plan.name
                );
            }
            matched += 1;
        }
        Ok(matched)
    }

    /// The `quantize.imatrix.*` keys recording which matrix a file was
    /// fitted to.
    pub fn provenance(&self, path: &Path) -> Vec<(String, GgufValue)> {
        let file_name = |p: &str| {
            Path::new(p)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(p)
                .to_string()
        };
        let mut out = vec![(
            "quantize.imatrix.file".to_string(),
            GgufValue::String(file_name(&path.to_string_lossy())),
        )];
        if let Some(dataset) = self.datasets.first() {
            out.push((
                "quantize.imatrix.dataset".into(),
                GgufValue::String(file_name(dataset)),
            ));
        }
        out.push((
            "quantize.imatrix.entries_count".into(),
            GgufValue::I32(self.entries.len() as i32),
        ));
        if self.chunks > 0 {
            out.push((
                "quantize.imatrix.chunks_count".into(),
                GgufValue::I32(self.chunks as i32),
            ));
        }
        out
    }
}

/// Tokens per calibration chunk.
const CHUNK: usize = 512;

/// Measures an importance matrix for `model`, a GGUF file, over the
/// calibration text `text`, and returns where it was written: `output`, or
/// without one `<models>/orangu-converted/<model>.<text>.imatrix.gguf`
/// (beside `model` without a models directory), where a matrix newer than
/// both files is reused.
pub fn measure(model: &Path, text: &Path, output: Option<&Path>) -> Result<PathBuf> {
    if let Some(output) = output {
        run(model, text, output)?;
        return Ok(output.to_path_buf());
    }
    let stem = |p: &Path| {
        p.file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default()
    };
    let name = format!("{}.{}.imatrix.gguf", stem(model), stem(text));
    let dir = match crate::hf::models_dir() {
        Some(models) => models.join("orangu-converted"),
        None => model.parent().map(Path::to_path_buf).unwrap_or_default(),
    };
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let out = dir.join(name);
    let modified = |p: &Path| std::fs::metadata(p).and_then(|m| m.modified()).ok();
    if let Some(made) = modified(&out)
        && [model, text]
            .iter()
            .all(|p| modified(p).is_some_and(|t| t <= made))
    {
        eprintln!("importance matrix {} (measured before)", out.display());
        return Ok(out);
    }
    run(model, text, &out)?;
    Ok(out)
}

/// Measures into `out` through a partial file, so an interrupted run leaves
/// no matrix behind.
fn run(model: &Path, text: &Path, out: &Path) -> Result<()> {
    let partial = out.with_extension("gguf.partial");
    eprintln!(
        "measuring an importance matrix for {} over {}",
        model.display(),
        text.display()
    );
    let links = crate::jobs::claim_for("imatrix");
    if let Err(err) = crate::jobs::imatrix(model, text, &partial, CHUNK, links) {
        let _ = std::fs::remove_file(&partial);
        return Err(err);
    }
    std::fs::rename(&partial, out)
        .with_context(|| format!("moving the matrix to {}", out.display()))
}

/// An importance matrix as loaded for a conversion, with where it came
/// from.
pub struct Loaded {
    pub matrix: Imatrix,
    pub path: PathBuf,
}

impl Loaded {
    /// Locates and reads the matrix `spec` names ([`locate`]).
    pub fn open(spec: &Path) -> Result<Self> {
        let path = locate(spec)?;
        let matrix = load(&path)?;
        Ok(Self { matrix, path })
    }

    /// Checks the matrix against the planned tensors and, when any of them
    /// will be fitted to it, records it in `metadata`. Returns how many
    /// tensors are.
    pub fn apply(
        &self,
        plans: &[TensorPlan],
        metadata: &mut Vec<(String, GgufValue)>,
    ) -> Result<usize> {
        self.matrix.check(plans)?;
        let fitted = plans
            .iter()
            .filter(|p| {
                orangu::quantize::takes_importance(p.ggml_type)
                    && self.matrix.get(&p.name).is_some()
            })
            .count();
        println!(
            "importance  {} ({} entries, fits {fitted} of {} tensors)",
            self.path.display(),
            self.matrix.entries.len(),
            plans.len()
        );
        if fitted > 0 {
            metadata.retain(|(k, _)| !k.starts_with("quantize.imatrix."));
            metadata.extend(self.matrix.provenance(&self.path));
        }
        Ok(fitted)
    }

    /// The importances for a planned tensor, if it is fitted to them.
    pub fn for_plan(&self, plan: &TensorPlan) -> Option<&[f32]> {
        if orangu::quantize::takes_importance(plan.ggml_type) {
            self.matrix.get(&plan.name)
        } else {
            None
        }
    }
}

/// Whether `plan` cannot be written without importances: the lattice types
/// below three bits, and the two-bit K-quant of a `Q2_K_S` file. Without a
/// matrix those searches land on points that read back fine and answer
/// badly. The token embedding and the output are never held to it.
fn requires_importance(plan: &TensorPlan, ftype: crate::quant::Ftype) -> bool {
    use crate::quant::*;
    if matches!(
        plan.name.as_str(),
        "token_embd.weight" | "per_layer_token_embd.weight" | "output.weight"
    ) {
        return false;
    }
    match plan.ggml_type {
        GGML_TYPE_IQ3_XXS | GGML_TYPE_IQ2_XXS | GGML_TYPE_IQ2_XS | GGML_TYPE_IQ2_S
        | GGML_TYPE_IQ1_M | GGML_TYPE_IQ1_S => true,
        GGML_TYPE_Q2_K => ftype == Ftype::Q2KS,
        _ => false,
    }
}

/// Refuses a plan that writes a tensor [`requires_importance`] says needs
/// importances the matrix (if any) does not have, naming the first.
pub fn require(
    ftype: crate::quant::Ftype,
    plans: &[TensorPlan],
    matrix: Option<&Loaded>,
) -> Result<()> {
    for plan in plans {
        if !requires_importance(plan, ftype) {
            continue;
        }
        match matrix {
            None => bail!(
                "{} needs an importance matrix: {} is written as {}, which without one reads \
                 back fine and answers badly — pass --imatrix FILE or an HF repo that publishes one",
                ftype.name(),
                plan.name,
                crate::quant::type_name(plan.ggml_type)
            ),
            Some(m) if m.matrix.get(&plan.name).is_none() => bail!(
                "{} has no importances for {}, which {} writes as {}",
                m.path.display(),
                plan.name,
                ftype.name(),
                crate::quant::type_name(plan.ggml_type)
            ),
            Some(_) => {}
        }
    }
    Ok(())
}

/// Reads an importance matrix in either format.
pub fn load(path: &Path) -> Result<Imatrix> {
    let mut magic = [0u8; 4];
    File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .with_context(|| format!("reading {}", path.display()))?;
    if &magic == b"GGUF" {
        load_gguf(path)
    } else {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        load_binary(&bytes).with_context(|| format!("reading {}", path.display()))
    }
}

fn load_gguf(path: &Path) -> Result<Imatrix> {
    let file = GgufFile::open(path).with_context(|| format!("reading {}", path.display()))?;
    let meta = |key: &str| file.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v);
    let has = |key: &str| meta(key).is_some();
    if !(has("imatrix.datasets") && has("imatrix.chunk_count") && has("imatrix.chunk_size")) {
        bail!(
            "{} is a GGUF file without importance-matrix metadata",
            path.display()
        );
    }
    let datasets = match meta("imatrix.datasets") {
        Some(GgufValue::Array(items)) => items
            .iter()
            .filter_map(|v| match v {
                GgufValue::String(s) => Some(s.clone()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    };
    let chunks = meta("imatrix.chunk_count")
        .and_then(GgufValue::as_u64)
        .unwrap_or(0) as u32;

    let mut entries = HashMap::new();
    for tensor in &file.tensors {
        let Some(name) = tensor.name.strip_suffix(".in_sum2") else {
            continue;
        };
        let sums = file.read_tensor(path, &tensor.name)?;
        let counts = file
            .read_tensor(path, &format!("{name}.counts"))
            .with_context(|| format!("{name} has sums but no counts"))?;
        if counts.is_empty() || !sums.len().is_multiple_of(counts.len()) {
            bail!(
                "{name}: {} sums do not divide over {} counts",
                sums.len(),
                counts.len()
            );
        }
        let per = sums.len() / counts.len();
        let mut values = Vec::with_capacity(sums.len());
        for (expert, &count) in counts.iter().enumerate() {
            let count = count.round();
            let row = &sums[expert * per..(expert + 1) * per];
            if count > 0.0 {
                values.extend(row.iter().map(|s| s / count));
            } else {
                values.extend(std::iter::repeat_n(1.0, per));
            }
        }
        entries.insert(name.to_string(), values);
    }
    if entries.is_empty() {
        bail!("{} holds no importance entries", path.display());
    }
    Ok(Imatrix {
        entries,
        datasets,
        chunks,
    })
}

/// The older format. Its values are already the summed squares divided by
/// the rows counted and multiplied by the call count, so dividing by the
/// call count gives the same importances the GGUF format does.
fn load_binary(bytes: &[u8]) -> Result<Imatrix> {
    let mut at = 0usize;
    let mut take = |n: usize| -> Result<&[u8]> {
        let end = at
            .checked_add(n)
            .filter(|&e| e <= bytes.len())
            .ok_or_else(|| anyhow!("the file ends early"))?;
        let out = &bytes[at..end];
        at = end;
        Ok(out)
    };
    let i32_at = |b: &[u8]| i32::from_le_bytes(b.try_into().expect("four bytes"));
    let entries_count = i32_at(take(4)?);
    if entries_count < 1 {
        bail!("no importance entries");
    }
    let mut entries = HashMap::new();
    for _ in 0..entries_count {
        let len = i32_at(take(4)?);
        let len = usize::try_from(len).map_err(|_| anyhow!("a negative name length"))?;
        let name = String::from_utf8(take(len)?.to_vec()).context("a name that is not UTF-8")?;
        let calls = i32_at(take(4)?);
        let count = i32_at(take(4)?);
        let count = usize::try_from(count)
            .ok()
            .filter(|&c| c > 0)
            .ok_or_else(|| anyhow!("{name} has no values"))?;
        let raw = take(
            count
                .checked_mul(4)
                .ok_or_else(|| anyhow!("{name} is too large"))?,
        )?;
        let values: Vec<f32> = raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .map(|v| if calls > 0 { v / calls as f32 } else { v })
            .collect();
        entries.insert(name, values);
    }
    let mut chunks = 0;
    let mut datasets = Vec::new();
    if let Ok(b) = take(4) {
        chunks = i32_at(b).max(0) as u32;
        if let Ok(b) = take(4)
            && let Ok(len) = usize::try_from(i32_at(b))
            && let Ok(name) = take(len)
        {
            datasets.push(String::from_utf8_lossy(name).into_owned());
        }
    }
    Ok(Imatrix {
        entries,
        datasets,
        chunks,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::write;

    #[test]
    fn the_binary_format_is_divided_by_its_call_count() {
        let mut bytes = Vec::new();
        bytes.extend(1i32.to_le_bytes());
        bytes.extend(3i32.to_le_bytes());
        bytes.extend(b"w.x");
        bytes.extend(4i32.to_le_bytes()); // calls
        bytes.extend(2i32.to_le_bytes()); // values
        bytes.extend(8.0f32.to_le_bytes());
        bytes.extend(2.0f32.to_le_bytes());
        bytes.extend(10i32.to_le_bytes()); // chunks
        bytes.extend(4i32.to_le_bytes());
        bytes.extend(b"wiki");
        let matrix = load_binary(&bytes).unwrap();
        assert_eq!(matrix.get("w.x"), Some(&[2.0, 0.5][..]));
        assert_eq!(matrix.chunks, 10);
        assert_eq!(matrix.datasets, vec!["wiki".to_string()]);
        assert!(load_binary(&bytes[..10]).is_err());
    }

    #[test]
    fn the_gguf_format_is_divided_per_expert_and_an_unreached_expert_is_flat() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m.imatrix.gguf");
        let meta = vec![
            (
                "imatrix.datasets".to_string(),
                GgufValue::Array(vec![GgufValue::String("calib.txt".into())]),
            ),
            ("imatrix.chunk_count".to_string(), GgufValue::U32(5)),
            ("imatrix.chunk_size".to_string(), GgufValue::U32(512)),
        ];
        let plans = vec![
            TensorPlan {
                name: "e.in_sum2".into(),
                dims: vec![2, 2],
                ggml_type: crate::quant::GGML_TYPE_F32,
            },
            TensorPlan {
                name: "e.counts".into(),
                dims: vec![1, 2],
                ggml_type: crate::quant::GGML_TYPE_F32,
            },
        ];
        write::write(
            &path,
            &meta,
            &plans,
            |plan| {
                Ok(if plan.name == "e.in_sum2" {
                    vec![6.0, 3.0, 9.0, 9.0]
                } else {
                    vec![3.0, 0.0]
                })
            },
            &|_, _, _| {},
        )
        .unwrap();
        let matrix = load(&path).unwrap();
        assert_eq!(matrix.get("e"), Some(&[2.0, 1.0, 1.0, 1.0][..]));
        assert_eq!(matrix.chunks, 5);
        assert_eq!(matrix.datasets, vec!["calib.txt".to_string()]);
    }

    /// The lattice types below three bits are refused without importances
    /// for every tensor they write; the token embedding is exempt.
    #[test]
    fn the_low_lattice_types_need_a_matrix() {
        use crate::quant::{Ftype, GGML_TYPE_IQ2_XS, GGML_TYPE_Q2_K, GGML_TYPE_Q4_K};
        let plan = |name: &str, ggml_type| TensorPlan {
            name: name.into(),
            dims: vec![256, 4],
            ggml_type,
        };
        let embedding = [plan("token_embd.weight", GGML_TYPE_Q2_K)];
        assert!(require(Ftype::IQ2XS, &embedding, None).is_ok());
        let body = [plan("blk.0.attn_q.weight", GGML_TYPE_IQ2_XS)];
        let err = require(Ftype::IQ2XS, &body, None).unwrap_err().to_string();
        assert!(
            err.contains("--imatrix") && err.contains("blk.0.attn_q.weight"),
            "{err}"
        );
        let matrix = Loaded {
            matrix: Imatrix {
                entries: [("blk.0.attn_q.weight".to_string(), vec![1.0; 256])].into(),
                ..Default::default()
            },
            path: "m.imatrix".into(),
        };
        assert!(require(Ftype::IQ2XS, &body, Some(&matrix)).is_ok());
        let other = [plan("blk.0.attn_k.weight", GGML_TYPE_IQ2_XS)];
        assert!(require(Ftype::IQ2XS, &other, Some(&matrix)).is_err());
        // A Q2_K tensor needs one only in a Q2_K_S file.
        let q2 = [plan("blk.0.attn_q.weight", GGML_TYPE_Q2_K)];
        assert!(require(Ftype::Q2K, &q2, None).is_ok());
        assert!(require(Ftype::Q2KS, &q2, None).is_err());
        assert!(require(Ftype::Q4KM, &[plan("blk.0.x.weight", GGML_TYPE_Q4_K)], None).is_ok());
    }

    /// Per weighted type: its block size and the byte ranges holding its
    /// scales.
    fn layout(kind: u32) -> Option<(usize, Vec<usize>)> {
        use crate::quant::*;
        Some(match kind {
            GGML_TYPE_Q2_K => (84, (0..16).chain(80..84).collect()),
            GGML_TYPE_Q3_K => (110, (96..110).collect()),
            GGML_TYPE_Q4_K => (144, (0..16).collect()),
            GGML_TYPE_Q5_K => (176, (0..16).collect()),
            GGML_TYPE_Q6_K => (210, (192..210).collect()),
            GGML_TYPE_IQ4_NL => (18, (0..2).collect()),
            GGML_TYPE_IQ4_XS => (136, (0..8).collect()),
            GGML_TYPE_Q4_0 => (18, (0..2).collect()),
            GGML_TYPE_Q5_0 => (22, (0..2).collect()),
            GGML_TYPE_Q5_1 => (24, (0..4).collect()),
            GGML_TYPE_IQ2_S => (82, (0..2).chain(74..82).collect()),
            GGML_TYPE_IQ3_S => (110, (0..2).chain(106..110).collect()),
            GGML_TYPE_IQ2_XXS => (66, (0..2).chain((0..8).map(|ib| 2 + 8 * ib + 7)).collect()),
            GGML_TYPE_IQ2_XS => (74, (0..2).chain(66..74).collect()),
            GGML_TYPE_IQ3_XXS => (98, (0..2).chain(66..98).collect()),
            GGML_TYPE_IQ1_S => (50, (0..2).chain((0..8).map(|ib| 34 + 2 * ib + 1)).collect()),
            GGML_TYPE_IQ1_M => (56, (48..56).collect()),
            _ => return None,
        })
    }

    /// Encodes every weighted tensor of `repo`'s `tags` again from its
    /// `reference` file with the repo's own importance matrix, and checks
    /// the blocks against the published ones: at least `percent` of each
    /// type's blocks identical.
    ///
    /// `exact` asks for every byte. Otherwise the file came from a build
    /// whose compiler fused some multiply-adds, which moves a few blocks'
    /// `f16` scale by its last bit and takes a scale code with it; the fit
    /// itself must still agree, so no block may differ only in its quants
    /// (bar a few for the types that keep levels fitted against the
    /// unrounded scale).
    fn reproduce(repo: &str, reference: &str, tags: &[&str], types: usize, exact: bool) {
        use orangu::gguf::GgufFile;
        use std::collections::BTreeMap;
        let models = crate::hf::models_dir().expect("a models directory");
        // By file name in the repo's snapshot: a listing shows one entry
        // per distinct file, and a publisher's `_L` can be the same bytes
        // as its `_M`.
        let folder = models.join(orangu::model_download::repo_folder_name(repo));
        let commit = std::fs::read_to_string(folder.join("refs/main")).unwrap();
        let snapshot = folder.join("snapshots").join(commit.trim());
        let find = |tag: &str| {
            let suffix = format!("-{}.gguf", tag.to_ascii_lowercase());
            std::fs::read_dir(&snapshot)
                .unwrap()
                .filter_map(|e| e.ok().map(|e| e.path()))
                .find(|p| {
                    p.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.to_ascii_lowercase().ends_with(&suffix))
                })
                .unwrap_or_else(|| panic!("{repo} has no {tag} file in {}", snapshot.display()))
        };
        let matrix = load(&locate(Path::new(repo)).unwrap()).unwrap();
        let reference_path = find(reference);
        let reference = GgufFile::open(&reference_path).unwrap();
        let reference_file = std::fs::File::open(&reference_path).unwrap();
        // Safety: read-only, and nothing writes the file meanwhile.
        let reference_map = unsafe { memmap2::Mmap::map(&reference_file) }.unwrap();
        let quants: Vec<_> = tags
            .iter()
            .map(|tag| {
                let path = find(tag);
                let file = GgufFile::open(&path).unwrap();
                let map = std::fs::File::open(&path).unwrap();
                // Safety: read-only, and nothing writes the file meanwhile.
                (file, unsafe { memmap2::Mmap::map(&map) }.unwrap())
            })
            .collect();
        // (blocks, identical, quants only) per type, and (blocks,
        // identical) per type for the tensors the matrix does not cover,
        // which are encoded without one.
        let mut stats: BTreeMap<u32, (usize, usize, usize)> = BTreeMap::new();
        let mut plain: BTreeMap<u32, (usize, usize)> = BTreeMap::new();
        for source in &reference.tensors {
            let importance = matrix.get(&source.name);
            let mut values = None;
            for (quant, bytes) in &quants {
                let Some(tensor) = quant.tensors.iter().find(|t| t.name == source.name) else {
                    continue;
                };
                let kind = tensor.ggml_type;
                let Some((block, scales)) = layout(kind) else {
                    continue;
                };
                let values = values.get_or_insert_with(|| {
                    let elements = source.element_count() as usize;
                    let start = (reference.data_offset + source.offset) as usize;
                    let size =
                        orangu::dequantize::tensor_byte_size(source.ggml_type, elements as u64)
                            .unwrap() as usize;
                    orangu::dequantize::dequantize(
                        source.ggml_type,
                        &reference_map[start..start + size],
                        elements,
                    )
                    .unwrap()
                });
                let ncols = tensor.dims[0] as usize;
                let start = (quant.data_offset + tensor.offset) as usize;
                let Some(importance) = importance else {
                    let ours = orangu::quantize::encode(kind, values, ncols);
                    let theirs = &bytes[start..start + ours.len()];
                    let entry = plain.entry(kind).or_default();
                    for (o, t) in ours.chunks(block).zip(theirs.chunks(block)) {
                        entry.0 += 1;
                        entry.1 += usize::from(o == t);
                    }
                    continue;
                };
                let ours = orangu::quantize::encode_weighted(kind, values, ncols, importance);
                let theirs = &bytes[start..start + ours.len()];
                let entry = stats.entry(kind).or_default();
                for (o, t) in ours.chunks(block).zip(theirs.chunks(block)) {
                    entry.0 += 1;
                    if o == t {
                        entry.1 += 1;
                    } else if scales.iter().all(|&i| o[i] == t[i]) {
                        entry.2 += 1;
                    }
                }
            }
        }
        for (kind, (blocks, same, quants_only)) in &stats {
            eprintln!(
                "{}: {same}/{blocks} identical, {quants_only} quants only",
                orangu::quantize::type_name(*kind)
            );
        }
        for (kind, (blocks, same)) in &plain {
            let name = orangu::quantize::type_name(*kind);
            eprintln!("{name} without a matrix: {same}/{blocks} identical");
            // A file made before the matrix-free K-quants took their
            // present arithmetic carries other bytes, so only a file named
            // exact is held to them.
            assert!(
                !exact || same == blocks,
                "{name} without a matrix: only {same} of {blocks} blocks identical"
            );
        }
        assert_eq!(stats.len(), types, "not every weighted type was found");
        for (kind, (blocks, same, quants_only)) in stats {
            let name = orangu::quantize::type_name(kind);
            // The 32-wide, codebook and lattice types keep the levels fitted against
            // the unrounded scale, so a last-bit difference there can move a
            // level behind an identical stored scale.
            let unrequantized = [
                crate::quant::GGML_TYPE_IQ4_NL,
                crate::quant::GGML_TYPE_IQ4_XS,
                crate::quant::GGML_TYPE_Q4_0,
                crate::quant::GGML_TYPE_Q5_0,
                crate::quant::GGML_TYPE_Q5_1,
                crate::quant::GGML_TYPE_IQ1_S,
                crate::quant::GGML_TYPE_IQ2_XXS,
                crate::quant::GGML_TYPE_IQ2_XS,
                crate::quant::GGML_TYPE_IQ2_S,
                crate::quant::GGML_TYPE_IQ3_XXS,
                crate::quant::GGML_TYPE_IQ3_S,
            ];
            // IQ1_M's split search was rearranged on 2026-09-22 to build its
            // sums once per block; the same splits win except where the two
            // orders of arithmetic round a near-tie differently, and a file
            // made before then carries the older choice.
            let allowed = if exact {
                0
            } else if unrequantized.contains(&kind) {
                blocks / 1000
            } else if kind == crate::quant::GGML_TYPE_IQ1_M {
                blocks / 100
            } else {
                0
            };
            assert!(
                quants_only <= allowed,
                "{name}: {quants_only} blocks with the same scales encoded differently"
            );
            assert!(
                if exact {
                    same == blocks
                } else {
                    same * 100 >= blocks * 95
                },
                "{name}: only {same} of {blocks} blocks identical"
            );
        }
    }

    /// [`reproduce`] on bartowski's SmolLM2-360M-Instruct: Q3_K to Q6_K, and
    /// IQ4_NL, IQ4_XS, the 32-wide fallbacks and IQ3_S. Needs the F16, the
    /// K-quants and the matrix in the models directory:
    /// `cargo test -- --ignored published_imatrix`.
    #[test]
    #[ignore]
    fn published_imatrix_quants_are_reproduced_dense() {
        reproduce(
            "bartowski/SmolLM2-360M-Instruct-GGUF",
            "F16",
            &[
                "Q2_K", "Q3_K_S", "Q3_K_M", "Q4_K_S", "Q4_K_M", "Q5_K_S", "Q5_K_M", "Q6_K",
                "IQ4_XS", "Q4_0", "IQ3_M", "IQ3_XS",
            ],
            10,
            false,
        );
    }

    /// [`reproduce`] on bartowski's Ling-3.0-tiny, a mixture with
    /// per-expert importances, byte for byte: Q2_K to Q6_K, IQ4_NL, Q5_1,
    /// IQ2_S and IQ3_S.
    /// Needs the BF16, the Q2_K, Q4_K_M and IQ2_M; the matrix is fetched
    /// when missing.
    #[test]
    #[ignore]
    fn published_imatrix_quants_are_reproduced_mixture() {
        reproduce(
            "bartowski/Ling-3.0-tiny-GGUF",
            "BF16",
            &["Q2_K", "Q4_K_M", "IQ2_M"],
            9,
            true,
        );
    }

    /// [`reproduce`] on unsloth's Qwen3-0.6B Q4_K_M, byte for byte: Q4_K and
    /// Q6_K against the matrix, and the token embedding, which the matrix
    /// does not cover, at Q6_K without one. Needs the BF16 and the Q4_K_M.
    #[test]
    #[ignore]
    fn published_imatrix_quants_are_reproduced_with_plain_tensors() {
        reproduce("unsloth/Qwen3-0.6B-GGUF", "BF16", &["Q4_K_M"], 2, true);
    }

    /// [`reproduce`] on unsloth's Qwen3-0.6B, whose dynamic quantizations
    /// carry every lattice type: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S,
    /// IQ3_XXS and IQ3_S. Needs the BF16 and the four `UD-` files; the
    /// matrix is fetched when missing.
    #[test]
    #[ignore]
    fn published_imatrix_quants_are_reproduced_lattice() {
        reproduce(
            "unsloth/Qwen3-0.6B-GGUF",
            "BF16",
            &["UD-IQ1_S", "UD-IQ1_M", "UD-IQ2_XXS", "UD-IQ3_XXS"],
            9,
            false,
        );
    }
}
