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

//! The manifest: the whole build in one file.
//!
//! Every setting a run has lives here, and the training material lives
//! under `repositories`. Nothing is passed on the command line that the
//! file cannot say, which is what makes a build reproducible — the file
//! *is* the description of the model, and re-running it months later
//! against a newer binary produces the same thing.
//!
//! The smallest useful manifest is the material and nothing else, because
//! every other field has a default:
//!
//! ```json
//! { "repositories": [ { "url": "https://github.com/owner/repo", "license": "MIT" } ] }
//! ```
//!
//! and the full form says everything out loud:
//!
//! ```json
//! {
//!   "name": "orangu-code",
//!   "license": "Apache-2.0",
//!   "training_size": "2b",
//!   "context_size": "256k",
//!   "quantization": "bf16",
//!   "repositories": [
//!     { "url": "https://github.com/owner/repo", "license": "MIT" }
//!   ]
//! }
//! ```
//!
//! Unknown keys are an error rather than being ignored: a misspelled
//! `traning_size` that silently trained the default size for a week is a
//! worse outcome than a parse failure.
//!
//! **The licence field is not decoration.** A repository is trained on
//! only if its declared licence is one of the OSI-approved ones
//! (<https://opensource.org/license>, by SPDX identifier). Copyleft is
//! included in that — what a corpus licence decides is not whether a model
//! can be trained but what the trained weights may be published under, and
//! that is the manifest author's call to make.
//!
//! A licence that is *not* an OSI-approved identifier — an unrecognised
//! name, a source-available licence that is not open source, or a compound
//! expression like `MIT OR Apache-2.0`, where which one you are relying on
//! is a choice this tool will not make for you — is **excluded from the
//! corpus** and named in the run's output. Excluded rather than fatal,
//! because one odd entry in a list of hundreds should not stop a run; named
//! rather than dropped, because a corpus that is quietly smaller than the
//! manifest says is a training run nobody can reproduce.
//! `"allow_any_license": true` trains on them anyway, and sits in the same
//! file as the declarations it overrides.
//!
//! The declared licence is what is checked; this tool does not audit the
//! repository's own `LICENSE` file against it. Getting that claim right is
//! the manifest author's responsibility, which is exactly why the field is
//! mandatory: it makes the claim explicit and reviewable.

use crate::wikipedia::{self, Wikipedia};
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

/// One source to train from.
///
/// Not necessarily a repository, despite the name of the list it lives in:
/// `corpus::classify` decides what a `url` actually is — a Git remote, a
/// local repository, a plain directory of files, or an archive — and each
/// is brought into reach its own way.
#[derive(Debug, Clone, Deserialize)]
pub struct Repository {
    /// A Git remote, a local repository, a directory of files, or an
    /// archive (`.tar.gz`, `.tar.bz2`, `.tar`, `.zip`), local or remote.
    pub url: String,
    /// The SPDX identifier the source is published under.
    pub license: String,
    /// Optional branch/tag to clone instead of the remote's default head.
    /// Git sources only.
    #[serde(default)]
    pub branch: Option<String>,
}

impl Repository {
    /// The directory name this source is brought into, when it is one that
    /// needs bringing: the url's last two path segments joined by `__`, so
    /// `owner/repo` stays legible and two sources of the same name from
    /// different owners cannot collide.
    pub fn slug(&self) -> String {
        let trimmed = self
            .url
            .trim_end_matches('/')
            .trim_end_matches(".git")
            .trim_end_matches('/');
        let mut parts: Vec<&str> = trimmed
            .rsplit(['/', ':'])
            .filter(|p| !p.is_empty())
            .take(2)
            .collect();
        parts.reverse();
        let slug: String = parts
            .join("__")
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        if slug.is_empty() {
            "repo".to_string()
        } else {
            slug
        }
    }
}

/// The default vocabulary size — shared by every size, so a tokenizer
/// trained once carries across all of them.
fn default_vocab_size() -> usize {
    32768
}
fn default_training_size() -> String {
    "2b".to_string()
}
fn default_context_size() -> String {
    "256k".to_string()
}
fn default_quantization() -> String {
    "bf16".to_string()
}
fn default_name() -> String {
    "orangu".to_string()
}
fn default_sequence_length() -> usize {
    2048
}
fn default_batch() -> usize {
    4
}
fn default_epochs() -> f64 {
    1.0
}
fn default_seed() -> u64 {
    1
}
fn default_jobs() -> usize {
    4
}
fn default_log_every() -> u64 {
    60
}
fn default_eval_every() -> u64 {
    200
}
fn default_checkpoint_every() -> u64 {
    200
}
fn default_chat_template() -> String {
    crate::vocab::CHAT_TEMPLATE.to_string()
}
/// Corpus files larger than this are skipped: at this size a source file
/// is a minified bundle, a checked-in binary, or a generated table.
fn default_max_file_size() -> u64 {
    1 << 20
}
/// Corpus text the tokenizer is trained on. More than this stops changing
/// the merges and only costs time.
fn default_tokenizer_sample() -> u64 {
    256 << 20
}

/// A manifest: the model's identity, every setting of the run, and the
/// repositories it trains on.
///
/// Every field but `repositories` has a default, and those defaults are the
/// ones documented in the manual — a manifest that says nothing but its
/// material trains a 2B model at a 256k context and writes it as BF16.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// The model's name, for `general.name`.
    #[serde(default = "default_name")]
    pub name: String,
    /// The licence the *model* is published under, for `general.license`.
    /// Absent writes no such key rather than inventing one.
    #[serde(default)]
    pub license: Option<String>,
    /// Free text for `general.description`.
    #[serde(default)]
    pub description: Option<String>,

    /// `smoke`, `0.5b`, `1b` or `2b`.
    #[serde(default = "default_training_size")]
    pub training_size: String,
    /// The context length the finished model declares: `8192`, `8k`, `256k`.
    #[serde(default = "default_context_size")]
    pub context_size: String,
    /// The weight format written: `bf16` (the default), `q6_k`, `q4_k_m`,
    /// and the rest of `--list-quantizations`.
    #[serde(default = "default_quantization")]
    pub quantization: String,
    /// Tokens in a newly trained vocabulary.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    /// Tokens per training sequence.
    #[serde(default = "default_sequence_length")]
    pub sequence_length: usize,
    /// Sequences per optimizer step.
    #[serde(default = "default_batch")]
    pub batch: usize,
    /// Optimizer steps to run. Absent derives them from `epochs`.
    #[serde(default)]
    pub steps: Option<u64>,
    /// Passes over the corpus, when `steps` is absent.
    #[serde(default = "default_epochs")]
    pub epochs: f64,
    /// Peak learning rate. Absent takes the size's own.
    #[serde(default)]
    pub learning_rate: Option<f32>,
    /// Weight initialization and batch sampling.
    #[serde(default = "default_seed")]
    pub seed: u64,
    /// Seconds between progress lines; `0` prints only the last step.
    #[serde(default = "default_log_every")]
    pub log_every: u64,
    /// Steps between validation passes; `0` turns them off.
    #[serde(default = "default_eval_every")]
    pub eval_every: u64,
    /// Steps between checkpoints; `0` turns them off.
    #[serde(default = "default_checkpoint_every")]
    pub checkpoint_every: u64,
    /// Continue from the checkpoint in the work directory.
    #[serde(default)]
    pub resume: bool,
    /// Write the model from the checkpoint without training further.
    #[serde(default)]
    pub export_only: bool,

    /// Repositories cloned at once.
    #[serde(default = "default_jobs")]
    pub jobs: usize,
    /// Largest corpus file read, in bytes.
    #[serde(default = "default_max_file_size")]
    pub max_file_size: u64,
    /// Bytes of corpus text the tokenizer is trained on.
    #[serde(default = "default_tokenizer_sample")]
    pub tokenizer_sample: u64,
    /// Use the corpus already in the work directory; clone nothing.
    #[serde(default)]
    pub offline: bool,
    /// Retrain the tokenizer and repack the corpus even if both exist.
    #[serde(default)]
    pub rebuild: bool,
    /// Train on repositories whose licence is not an OSI-approved one —
    /// including ones with no recognised identifier at all.
    #[serde(default, alias = "allow_any_licence")]
    pub allow_any_license: bool,

    /// The Jinja2 template written as `tokenizer.chat_template`. Absent
    /// writes ChatML, which is what the vocabulary's special tokens are;
    /// `""` writes no template at all, leaving a file that only completion
    /// endpoints will serve.
    #[serde(default = "default_chat_template")]
    pub chat_template: String,

    /// Corpus, tokenizer, packed tokens and checkpoints. Absent derives
    /// `~/.orangu/gguf/<manifest name>`.
    #[serde(default)]
    pub work_dir: Option<PathBuf>,
    /// Where to write the model. Absent derives it from the name, size and
    /// weight format.
    #[serde(default)]
    pub output: Option<PathBuf>,

    /// English (or another language's) Wikipedia, as prose to train on
    /// beside the code. Absent leaves it out.
    #[serde(default)]
    pub wikipedia: Option<Wikipedia>,

    /// The training material. After parsing this holds the repositories
    /// that are actually trained on; anything excluded moves to
    /// [`Manifest::excluded`].
    pub repositories: Vec<Repository>,

    /// Repositories left out of the corpus because their licence is not an
    /// OSI-approved one. Reported by the run rather than dropped quietly.
    #[serde(skip)]
    pub excluded: Vec<Repository>,
}

impl Manifest {
    /// Reads and validates a manifest.
    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading the manifest {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("in the manifest {}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Self> {
        let value: serde_json::Value = serde_json::from_str(text)?;
        if value.is_array() {
            bail!(
                "the manifest is a bare list — every setting of a run lives in it too, \
                 so it has to be an object with the repositories under \"repositories\""
            );
        }
        let mut manifest: Manifest = serde_json::from_value(value)?;
        manifest.check_structure()?;
        manifest.exclude_unapproved();
        if manifest.repositories.is_empty() {
            bail!(
                "every repository was excluded for its licence:\n  {}\n\
                 Only OSI-approved licences (<https://opensource.org/license>) are trained \
                 on; set \"allow_any_license\": true to train on these anyway.",
                manifest
                    .excluded
                    .iter()
                    .map(|r| format!("{} ({})", r.url, r.license))
                    .collect::<Vec<_>>()
                    .join("\n  ")
            );
        }
        Ok(manifest)
    }

    /// Moves every repository whose licence is not OSI-approved out of the
    /// corpus and into [`Manifest::excluded`].
    ///
    /// Excluded, not fatal: one odd entry in a list of hundreds should not
    /// stop a run. It is *reported*, though — a corpus that quietly trained
    /// on a tenth less than it was told to is a training run nobody can
    /// reproduce.
    fn exclude_unapproved(&mut self) {
        if self.allow_any_license {
            return;
        }
        let (keep, drop): (Vec<Repository>, Vec<Repository>) = self
            .repositories
            .drain(..)
            .partition(|repo| permission(&repo.license) == Permission::Approved);
        self.repositories = keep;
        self.excluded = drop;
    }

    /// Everything that has to be true of a manifest before a run starts —
    /// checked here, at parse time, rather than hours in when the value is
    /// first used.
    fn check_structure(&self) -> Result<()> {
        if self.repositories.is_empty() {
            bail!("no repositories listed — there is nothing to train on");
        }
        if self.vocab_size == 0 || self.sequence_length == 0 || self.batch == 0 {
            bail!("vocab_size, sequence_length and batch have to be nonzero");
        }
        if self.epochs <= 0.0 && self.steps.is_none() {
            bail!("epochs has to be positive when steps is not given");
        }

        let mut seen: BTreeSet<String> = BTreeSet::new();
        for repo in &self.repositories {
            if repo.url.trim().is_empty() {
                bail!("a repository entry has an empty url");
            }
            if !seen.insert(repo.url.clone()) {
                bail!("{} is listed twice", repo.url);
            }
        }
        Ok(())
    }

    /// Every distinct corpus licence, sorted — written into the model file
    /// so the trained weights carry their own provenance.
    ///
    /// Wikipedia's own licence is in here when Wikipedia is part of the
    /// corpus. It is not a software licence and never went through the
    /// repository gate, but it is part of what the weights were trained on
    /// and so part of what the file has to record.
    pub fn licences(&self) -> Vec<String> {
        let mut set: BTreeSet<String> = self
            .repositories
            .iter()
            .map(|r| r.license.trim().to_string())
            .collect();
        if self.wikipedia.is_some() {
            set.insert(wikipedia::LICENSE.to_string());
        }
        set.into_iter().collect()
    }
}

/// What this tool will do with a repository under a given licence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Permission {
    /// An OSI-approved open source licence: trained on.
    Approved,
    /// Not an OSI-approved SPDX identifier — an unrecognised name, a
    /// source-available licence that is not open source, or a compound
    /// expression. Excluded from the corpus unless the manifest says
    /// otherwise.
    Unknown,
}

/// Every OSI-approved licence, by SPDX identifier, upper-cased for
/// comparison.
///
/// Taken from the SPDX licence list's own `isOsiApproved` flag, which is
/// the machine-readable form of <https://opensource.org/license>, and
/// including the identifiers SPDX has since deprecated (`GPL-3.0`,
/// `LGPL-2.1`, and the rest) because manifests and repository metadata in
/// the wild still use them.
///
/// Copyleft is *in* this list. What a corpus licence decides is not
/// whether a model can be trained — it is what the trained weights may
/// then be published under, and that is the manifest author's call to
/// make, not this tool's. Every licence the corpus declares is written
/// into the finished model's metadata, so the question can be answered
/// from the file rather than from a memory of the run.
const OSI_APPROVED: &[&str] = &[
    "0BSD",
    "AAL",
    "AFL-1.1",
    "AFL-1.2",
    "AFL-2.0",
    "AFL-2.1",
    "AFL-3.0",
    "AGPL-3.0",
    "AGPL-3.0-ONLY",
    "AGPL-3.0-OR-LATER",
    "ALGLIB-DOCUMENTATION",
    "APACHE-1.1",
    "APACHE-2.0",
    "APL-1.0",
    "APSL-1.0",
    "APSL-1.1",
    "APSL-1.2",
    "APSL-2.0",
    "ARTISTIC-1.0",
    "ARTISTIC-1.0-CL8",
    "ARTISTIC-1.0-PERL",
    "ARTISTIC-2.0",
    "BLUEOAK-1.0.0",
    "BSD-1-CLAUSE",
    "BSD-2-CLAUSE",
    "BSD-2-CLAUSE-PATENT",
    "BSD-3-CLAUSE",
    "BSD-3-CLAUSE-LBNL",
    "BSD-3-CLAUSE-OPEN-MPI",
    "BSD-ASK-TO-ENDORSE",
    "BSL-1.0",
    "CAL-1.0",
    "CAL-1.0-COMBINED-WORK-EXCEPTION",
    "CATOSL-1.1",
    "CDDL-1.0",
    "CDDL-1.1",
    "CECILL-2.1",
    "CERN-OHL-P-2.0",
    "CERN-OHL-S-2.0",
    "CERN-OHL-W-2.0",
    "CNRI-PYTHON",
    "CPAL-1.0",
    "CPL-1.0",
    "CUA-OPL-1.0",
    "ECL-1.0",
    "ECL-2.0",
    "EFL-1.0",
    "EFL-2.0",
    "ENTESSA",
    "EPL-1.0",
    "EPL-2.0",
    "EUDATAGRID",
    "EUPL-1.1",
    "EUPL-1.2",
    "FAIR",
    "FRAMEWORX-1.0",
    "GPL-2.0",
    "GPL-2.0+",
    "GPL-2.0-ONLY",
    "GPL-2.0-OR-LATER",
    "GPL-3.0",
    "GPL-3.0+",
    "GPL-3.0-ONLY",
    "GPL-3.0-OR-LATER",
    "GPL-3.0-WITH-GCC-EXCEPTION",
    "HPND",
    "ICU",
    "INTEL",
    "IPA",
    "IPL-1.0",
    "ISC",
    "JAM",
    "LGPL-2.0",
    "LGPL-2.0+",
    "LGPL-2.0-ONLY",
    "LGPL-2.0-OR-LATER",
    "LGPL-2.1",
    "LGPL-2.1+",
    "LGPL-2.1-ONLY",
    "LGPL-2.1-OR-LATER",
    "LGPL-3.0",
    "LGPL-3.0+",
    "LGPL-3.0-ONLY",
    "LGPL-3.0-OR-LATER",
    "LILIQ-P-1.1",
    "LILIQ-R-1.1",
    "LILIQ-RPLUS-1.1",
    "LPL-1.0",
    "LPL-1.02",
    "LPPL-1.3C",
    "MIROS",
    "MIT",
    "MIT-0",
    "MIT-MODERN-VARIANT",
    "MOTOSOTO",
    "MPL-1.0",
    "MPL-1.1",
    "MPL-2.0",
    "MPL-2.0-NO-COPYLEFT-EXCEPTION",
    "MS-PL",
    "MS-RL",
    "MULANPSL-2.0",
    "MULTICS",
    "NASA-1.3",
    "NAUMEN",
    "NCSA",
    "NGPL",
    "NOKIA",
    "NPOSL-3.0",
    "NTP",
    "OCLC-2.0",
    "OFL-1.1",
    "OFL-1.1-NO-RFN",
    "OFL-1.1-RFN",
    "OGTSL",
    "OLDAP-2.8",
    "OLFL-1.3",
    "OSC-1.0",
    "OSET-PL-2.1",
    "OSL-1.0",
    "OSL-2.0",
    "OSL-2.1",
    "OSL-3.0",
    "PHP-3.0",
    "PHP-3.01",
    "POSTGRESQL",
    "PYTHON-2.0",
    "QPL-1.0",
    "RPL-1.1",
    "RPL-1.5",
    "RPSL-1.0",
    "RSCPL",
    "SIMPL-2.0",
    "SISSL",
    "SLEEPYCAT",
    "SPL-1.0",
    "UCL-1.0",
    "UNICODE-3.0",
    "UNICODE-DFS-2016",
    "UNLICENSE",
    "UPL-1.0",
    "VSL-1.0",
    "W3C",
    "W3C-20150513",
    "WATCOM-1.0",
    "WORDNET",
    "WXWINDOWS",
    "XNET",
    "ZLIB",
    "ZPL-2.0",
    "ZPL-2.1",
];

/// Classifies a declared SPDX identifier.
///
/// A compound expression (`MIT OR Apache-2.0`) is not resolved here — which
/// of the two the corpus is being taken under is a choice, and this tool
/// does not make choices about somebody else's licensing. Name the one you
/// are relying on.
pub fn permission(spdx: &str) -> Permission {
    let upper = spdx.trim().to_ascii_uppercase();
    if upper.is_empty() {
        return Permission::Unknown;
    }
    if upper.contains(" OR ") || upper.contains(" AND ") || upper.contains(" WITH ") {
        return Permission::Unknown;
    }
    if OSI_APPROVED.contains(&upper.as_str()) {
        Permission::Approved
    } else {
        Permission::Unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str =
        r#"{"repositories": [{"url": "https://github.com/a/b", "license": "MIT"}]}"#;

    /// The whole point of the format: the material is enough, and every
    /// setting it does not mention has the documented default.
    #[test]
    fn a_manifest_of_material_alone_takes_every_default() {
        let m = Manifest::parse(MINIMAL).unwrap();
        assert_eq!(m.name, "orangu");
        assert_eq!(m.training_size, "2b");
        assert_eq!(m.context_size, "256k");
        assert_eq!(m.quantization, "bf16");
        assert_eq!(m.vocab_size, 32768);
        assert_eq!(m.sequence_length, 2048);
        assert_eq!(m.batch, 4);
        assert_eq!(m.epochs, 1.0);
        assert_eq!(m.seed, 1);
        assert_eq!(m.jobs, 4);
        assert_eq!(m.log_every, 60);
        assert_eq!(m.eval_every, 200);
        assert_eq!(m.checkpoint_every, 200);
        assert_eq!(m.max_file_size, 1 << 20);
        assert_eq!(m.tokenizer_sample, 256 << 20);
        assert_eq!(m.steps, None);
        assert_eq!(m.learning_rate, None);
        assert!(!m.resume && !m.offline && !m.rebuild && !m.export_only);
        assert!(!m.allow_any_license);
        assert!(m.excluded.is_empty());
        assert!(m.license.is_none() && m.work_dir.is_none() && m.output.is_none());
        assert!(m.wikipedia.is_none(), "Wikipedia is opt-in");
        assert_eq!(m.repositories.len(), 1);
        assert_eq!(m.repositories[0].slug(), "a__b");
    }

    /// Every setting has to actually be settable — a field that parses but
    /// is never read is the failure this catches.
    #[test]
    fn every_setting_can_be_given() {
        let m = Manifest::parse(
            r#"{
                 "name": "n", "license": "Apache-2.0", "description": "d",
                 "training_size": "1b", "context_size": "8k", "quantization": "q4_k_m",
                 "vocab_size": 16384, "sequence_length": 512, "batch": 2,
                 "steps": 1234, "epochs": 3.0, "learning_rate": 0.0004, "seed": 7,
                 "log_every": 5, "eval_every": 0, "checkpoint_every": 50,
                 "resume": true, "export_only": true,
                 "jobs": 8, "max_file_size": 4096, "tokenizer_sample": 8192,
                 "offline": true, "rebuild": true, "allow_any_license": true,
                 "work_dir": "/w", "output": "/o/m.gguf",
                 "repositories": [{"url": "u", "license": "GPL-3.0-or-later"}]
               }"#,
        )
        .unwrap();
        assert_eq!(m.name, "n");
        assert_eq!(m.license.as_deref(), Some("Apache-2.0"));
        assert_eq!(m.training_size, "1b");
        assert_eq!(m.context_size, "8k");
        assert_eq!(m.quantization, "q4_k_m");
        assert_eq!(m.vocab_size, 16384);
        assert_eq!(m.sequence_length, 512);
        assert_eq!(m.batch, 2);
        assert_eq!(m.steps, Some(1234));
        assert_eq!(m.epochs, 3.0);
        assert_eq!(m.learning_rate, Some(0.0004));
        assert_eq!(m.seed, 7);
        assert_eq!(m.log_every, 5);
        assert_eq!(m.eval_every, 0);
        assert_eq!(m.checkpoint_every, 50);
        assert!(m.resume && m.export_only && m.offline && m.rebuild);
        assert_eq!(m.jobs, 8);
        assert_eq!(m.max_file_size, 4096);
        assert_eq!(m.tokenizer_sample, 8192);
        assert_eq!(m.work_dir.as_deref(), Some(Path::new("/w")));
        assert_eq!(m.output.as_deref(), Some(Path::new("/o/m.gguf")));
    }

    /// Wikipedia is a corpus source like any other, and what it brings
    /// with it is its licence.
    #[test]
    fn wikipedia_joins_the_corpus_and_its_provenance() {
        let m = Manifest::parse(
            r#"{"wikipedia": {"language": "en"},
                "repositories": [{"url": "u", "license": "MIT"}]}"#,
        )
        .unwrap();
        let settings = m.wikipedia.as_ref().unwrap();
        assert_eq!(settings.language, "en");
        assert!(m.licences().contains(&"CC-BY-SA-4.0".to_string()));

        // An empty object is every default: English, a bounded amount.
        let m = Manifest::parse(
            r#"{"wikipedia": {}, "repositories": [{"url": "u", "license": "MIT"}]}"#,
        )
        .unwrap();
        assert_eq!(m.wikipedia.unwrap(), crate::wikipedia::Wikipedia::default());
    }

    #[test]
    fn repository_slugs_survive_every_url_shape() {
        let m = Manifest::parse(
            r#"{"repositories":
                 [{"url": "https://github.com/a/b.git", "license": "MIT"},
                  {"url": "git@github.com:c/d.git", "license": "ISC"}]}"#,
        )
        .unwrap();
        assert_eq!(m.repositories[0].slug(), "a__b");
        assert_eq!(m.repositories[1].slug(), "c__d");
        assert_eq!(m.licences(), vec!["ISC".to_string(), "MIT".to_string()]);
    }

    /// A misspelled setting must fail the parse. Ignoring it would train
    /// the default for a week and report success.
    #[test]
    fn an_unknown_setting_is_an_error_that_names_it() {
        let err = Manifest::parse(
            r#"{"traning_size": "1b", "repositories": [{"url": "u", "license": "MIT"}]}"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("traning_size"), "{err}");
    }

    /// The old shape — a bare list of repositories — can no longer carry a
    /// run's settings, so it is refused with the reason rather than a
    /// serde type error.
    #[test]
    fn a_bare_list_is_refused_with_the_reason() {
        let err = Manifest::parse(r#"[{"url": "u", "license": "MIT"}]"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("repositories"), "{err}");
    }

    /// Copyleft is open source, so it trains — the gate is OSI approval,
    /// not permissiveness.
    #[test]
    fn every_osi_approved_licence_is_trained_on() {
        for license in [
            "MIT",
            "Apache-2.0",
            "BSD-3-Clause",
            "PostgreSQL",
            "ISC",
            "0BSD",
            "GPL-2.0-only",
            "GPL-3.0-or-later",
            "LGPL-2.1-or-later",
            "AGPL-3.0-only",
            "EPL-2.0",
            "MPL-2.0",
            "CDDL-1.0",
            "EUPL-1.2",
            // Identifiers SPDX has deprecated but repositories still use.
            "GPL-3.0",
            "LGPL-2.1",
        ] {
            assert_eq!(
                permission(license),
                Permission::Approved,
                "{license} is OSI-approved"
            );
        }
    }

    /// What is not open source does not train.
    #[test]
    fn licences_that_are_not_osi_approved_are_not_trained_on() {
        for license in [
            "SSPL-1.0",
            "BUSL-1.1",
            "Elastic-2.0",
            "CC-BY-NC-4.0",
            "Weird-1.0",
            "",
            // A choice this tool does not make for the caller.
            "MIT OR Apache-2.0",
            "GPL-2.0 WITH Linux-syscall-note",
        ] {
            assert_eq!(
                permission(license),
                Permission::Unknown,
                "{license:?} is not an OSI-approved identifier"
            );
        }
    }

    /// An unapproved licence takes its repository out of the corpus and
    /// says so, rather than stopping a run that has eleven good ones.
    #[test]
    fn an_unapproved_licence_is_excluded_not_fatal() {
        let m = Manifest::parse(
            r#"{"repositories": [
                 {"url": "https://github.com/a/b", "license": "MIT"},
                 {"url": "https://github.com/c/d", "license": "SSPL-1.0"}]}"#,
        )
        .unwrap();
        assert_eq!(m.repositories.len(), 1);
        assert_eq!(m.repositories[0].url, "https://github.com/a/b");
        assert_eq!(m.excluded.len(), 1);
        assert_eq!(m.excluded[0].url, "https://github.com/c/d");
        // The provenance written into the model is what was trained on.
        assert_eq!(m.licences(), vec!["MIT".to_string()]);
    }

    /// Excluding everything leaves nothing to train on, and that *is*
    /// fatal — with the reason and the setting that overrides it.
    #[test]
    fn excluding_everything_is_an_error_that_names_the_override() {
        let err = Manifest::parse(r#"{"repositories": [{"url": "u", "license": "SSPL-1.0"}]}"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("SSPL-1.0"), "{err}");
        assert!(err.contains("allow_any_license"), "{err}");
    }

    /// The override lives in the same file as the declarations it
    /// overrides, under either spelling.
    #[test]
    fn the_override_is_a_setting_in_the_manifest() {
        for field in ["allow_any_license", "allow_any_licence"] {
            let json = format!(
                r#"{{"{field}": true, "repositories": [{{"url": "u", "license": "Weird-1.0"}}]}}"#
            );
            let m = Manifest::parse(&json).unwrap();
            assert_eq!(m.repositories.len(), 1, "{field}");
            assert!(m.excluded.is_empty(), "{field}");
        }
    }

    /// Copyleft is trainable material. The gate is OSI approval, not
    /// permissiveness — what a corpus licence decides is what the finished
    /// weights may be published under, and that is the manifest author's
    /// call. The licences are written into the model's metadata; nothing
    /// here refuses them.
    #[test]
    fn copyleft_is_kept_rather_than_refused() {
        let m = Manifest::parse(
            r#"{"repositories": [
                 {"url": "a", "license": "MIT"},
                 {"url": "b", "license": "GPL-2.0-only"},
                 {"url": "c", "license": "EPL-2.0"}]}"#,
        )
        .unwrap();
        assert_eq!(m.repositories.len(), 3);
        assert!(m.excluded.is_empty());
        assert!(m.licences().contains(&"GPL-2.0-only".to_string()));
    }

    #[test]
    fn a_duplicate_url_is_an_error() {
        let err = Manifest::parse(
            r#"{"repositories": [{"url": "u", "license": "MIT"}, {"url": "u", "license": "MIT"}]}"#,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("listed twice"), "{err}");
    }

    #[test]
    fn an_empty_manifest_is_an_error() {
        assert!(Manifest::parse(r#"{"repositories": []}"#).is_err());
    }

    /// Settings that cannot be zero are caught at parse time, not when the
    /// training loop divides by one of them.
    #[test]
    fn nonsensical_settings_are_caught_before_the_run() {
        let with = |field: &str, value: &str| {
            format!(r#"{{"{field}": {value}, "repositories": [{{"url": "u", "license": "MIT"}}]}}"#)
        };
        assert!(Manifest::parse(&with("vocab_size", "0")).is_err());
        assert!(Manifest::parse(&with("sequence_length", "0")).is_err());
        assert!(Manifest::parse(&with("batch", "0")).is_err());
        assert!(Manifest::parse(&with("epochs", "0")).is_err());
        // With explicit steps, epochs is never read.
        assert!(Manifest::parse(&with("steps", "10")).is_ok());
    }
}
