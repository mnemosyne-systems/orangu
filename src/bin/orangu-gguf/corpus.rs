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

//! Turning a manifest into text: bring each source into reach, then walk it
//! for files worth training on.
//!
//! **Not everything is a Git repository.** A source is whatever its `url`
//! turns out to be, decided by looking rather than by assuming
//! ([`classify`]): a remote to clone, a local repository to clone, a
//! directory of files that is already on disk and needs no copying at all,
//! or an archive to unpack. Guessing wrong here fails in an unhelpful way
//! — `git clone` against a directory of text files reports that it is not
//! a repository, which is true and unhelpful — so the check is explicit and
//! the classification is reported.
//!
//! Cloning is `git clone --depth 1` and nothing else — no dataset library,
//! no hosting API. A clone that is already on disk is left alone, so a run
//! interrupted halfway resumes by re-running the same command, and a corpus
//! can be inspected, edited, or deleted with ordinary tools between runs.
//!
//! Compressed files are read through: a `.gz` or `.bz2` beside the source
//! is decompressed on the way in, and what decides whether it is training
//! text is the name *underneath* the compression — `main.rs.gz` is Rust,
//! `image.png.gz` is still a picture.
//!
//! What gets walked is deliberately narrow. Generated trees (`node_modules`,
//! `target`, `dist`, `vendor`) are not representative source; large files are
//! usually minified bundles, checked-in binaries, or generated tables; and a
//! file that is not valid UTF-8 is not text. Each of those is skipped, and
//! the count of what was skipped is reported rather than hidden, because
//! "the corpus is a tenth the size I expected" is a question this stage
//! should be able to answer.

use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::{
    fs::{self, File},
    io::{BufReader, Read},
    path::{Component, Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};
use walkdir::WalkDir;

use crate::manifest::{Manifest, Repository};

/// Directory names never descended into: build output, dependency trees,
/// and version-control metadata.
const SKIP_DIRS: &[&str] = &[
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "out",
    "target",
    "third_party",
    "vendor",
];

/// File extensions kept. Source in the languages a coding model is for,
/// plus the prose that lives beside it — a model that has never read a
/// `README` cannot explain itself.
const KEEP_EXTENSIONS: &[&str] = &[
    "adoc", "bash", "c", "cc", "cpp", "cs", "css", "cxx", "go", "h", "hpp", "hs", "html", "java",
    "jsx", "js", "kt", "lua", "md", "ml", "mjs", "php", "pl", "py", "r", "rb", "rs", "rst",
    "scala", "sh", "sql", "swift", "toml", "ts", "tsx", "txt", "vim", "yaml", "yml", "zig", "zsh",
];

/// Extensionless files that are still source.
const KEEP_NAMES: &[&str] = &[
    "Makefile",
    "makefile",
    "Dockerfile",
    "CMakeLists.txt",
    "README",
    "LICENSE",
    "COPYING",
];

/// A directory to walk, and the largest file worth reading from it.
///
/// The size cap is a heuristic about *repositories*: past a megabyte a
/// source file is a minified bundle, a checked-in binary, or a generated
/// table, and none of those is representative text. It has no business
/// applying to text this tool wrote itself — a prose shard is megabytes of
/// exactly what the corpus is for — so a root can opt out of it, and the
/// prose root does.
#[derive(Debug, Clone)]
pub struct Root {
    pub path: PathBuf,
    pub max_file_size: Option<u64>,
}

impl Root {
    /// A cloned repository: the cap applies.
    pub fn repository(path: PathBuf, max_file_size: u64) -> Self {
        Root {
            path,
            max_file_size: Some(max_file_size),
        }
    }

    /// Text this tool wrote: every file is read, whatever its size.
    pub fn generated(path: PathBuf) -> Self {
        Root {
            path,
            max_file_size: None,
        }
    }
}

/// What the walk found and what it left behind.
#[derive(Debug, Default, Clone)]
pub struct ScanReport {
    pub kept: usize,
    pub skipped_extension: usize,
    pub skipped_large: usize,
}

/// The archive shapes a source can arrive as. Extension first, because a
/// remote URL is only a name until it is fetched.
const ARCHIVES: &[&str] = &[
    ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tbz", ".tar", ".zip",
];

/// What a manifest entry's `url` turns out to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// A Git repository, remote or on disk: cloned into the corpus.
    Git,
    /// A directory of files that is not a repository: read where it is,
    /// and not copied. A corpus directory can be tens of gigabytes, and
    /// duplicating it to look at it would be absurd.
    Directory,
    /// An archive: unpacked into the corpus.
    Archive,
}

/// Decides what a source is by looking at it.
///
/// A remote is classified by its name — there is nothing else to go on
/// until it is fetched — and a local path by what is actually there. A
/// path that does not exist is left to Git, whose error message ("does not
/// appear to be a git repository") is the right one for a mistyped remote.
pub fn classify(url: &str) -> Source {
    let trimmed = url.trim();
    let lower = trimmed.to_ascii_lowercase();
    let looks_remote = lower.contains("://") || (lower.contains('@') && lower.contains(':'));

    if ARCHIVES.iter().any(|ext| lower.ends_with(ext)) {
        return Source::Archive;
    }
    if looks_remote {
        return Source::Git;
    }

    let path = Path::new(trimmed);
    if path.join(".git").exists() {
        Source::Git
    } else if path.is_dir() {
        Source::Directory
    } else {
        Source::Git
    }
}

/// Brings every source in the manifest into reach under `dir`, returning
/// the roots to walk.
///
/// `jobs` sources are fetched at once. A failure is fatal: a corpus that
/// is quietly missing a tenth of its sources is a training run whose data
/// mix nobody can reproduce.
pub fn fetch_all(manifest: &Manifest, dir: &Path, jobs: usize) -> Result<Vec<PathBuf>> {
    fs::create_dir_all(dir)
        .with_context(|| format!("creating the corpus directory {}", dir.display()))?;

    let done = AtomicUsize::new(0);
    let total = manifest.repositories.len();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(jobs.max(1))
        .build()
        .context("building the fetch thread pool")?;

    let results: Vec<Result<PathBuf>> = pool.install(|| {
        manifest
            .repositories
            .par_iter()
            .map(|repo| {
                let source = classify(&repo.url);
                let dest = dir.join(repo.slug());
                let outcome = match source {
                    Source::Git => clone(repo, &dest),
                    Source::Directory => {
                        let path = PathBuf::from(repo.url.trim());
                        if !path.is_dir() {
                            bail!("{} is not a directory", repo.url);
                        }
                        Ok((path, "directory"))
                    }
                    Source::Archive => unpack(repo, dir, &dest),
                }?;
                let (path, what) = outcome;
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                println!("[{n}/{total}] {} — {what}", repo.slug());
                Ok(path)
            })
            .collect()
    });

    results.into_iter().collect()
}

/// `git clone --depth 1`, or nothing if the clone is already there.
fn clone(repo: &Repository, dest: &Path) -> Result<(PathBuf, &'static str)> {
    if dest.is_dir() {
        return Ok((dest.to_path_buf(), "already cloned"));
    }
    let mut cmd = Command::new("git");
    cmd.args(["clone", "--depth", "1", "--quiet"]);
    if let Some(branch) = &repo.branch {
        cmd.args(["--branch", branch]);
    }
    cmd.arg(&repo.url).arg(dest);
    let out = cmd
        .output()
        .with_context(|| format!("running git clone for {}", repo.url))?;
    if !out.status.success() {
        bail!(
            "git clone {} failed: {}",
            repo.url,
            String::from_utf8_lossy(&out.stderr).trim()
        );
    }
    Ok((dest.to_path_buf(), "cloned"))
}

/// Fetches an archive if it is remote, then unpacks it into `dest`.
fn unpack(repo: &Repository, dir: &Path, dest: &Path) -> Result<(PathBuf, &'static str)> {
    if dest.is_dir() {
        return Ok((dest.to_path_buf(), "already unpacked"));
    }
    let url = repo.url.trim();
    let lower = url.to_ascii_lowercase();

    // A remote archive is downloaded beside the corpus first, so an
    // interrupted unpack does not re-download it.
    let local: PathBuf = if lower.contains("://") {
        let name = url.rsplit('/').next().unwrap_or("archive");
        let downloaded = dir.join(format!("{}-{name}", repo.slug()));
        if !downloaded.exists() {
            let temporary = downloaded.with_extension("partial");
            // A client of its own rather than `reqwest::blocking::get`,
            // whose default 30-second timeout covers the *whole* request
            // and would abort any archive that takes longer than that to
            // arrive — which is most of them.
            let client = reqwest::blocking::Client::builder()
                .user_agent(concat!("orangu-gguf/", env!("CARGO_PKG_VERSION")))
                .connect_timeout(std::time::Duration::from_secs(30))
                .timeout(None)
                .build()
                .context("building the HTTP client")?;
            let mut response = client
                .get(url)
                .send()
                .and_then(|r| r.error_for_status())
                .with_context(|| format!("fetching {url}"))?;
            let mut out = File::create(&temporary)
                .with_context(|| format!("creating {}", temporary.display()))?;
            std::io::copy(&mut response, &mut out)?;
            drop(out);
            fs::rename(&temporary, &downloaded)?;
        }
        downloaded
    } else {
        PathBuf::from(url)
    };
    if !local.is_file() {
        bail!("{} is not a file", local.display());
    }

    let staging = dest.with_extension("unpacking");
    let _ = fs::remove_dir_all(&staging);
    fs::create_dir_all(&staging)?;
    let reader = BufReader::with_capacity(1 << 20, File::open(&local)?);

    if lower.ends_with(".zip") {
        let mut archive = zip::ZipArchive::new(File::open(&local)?)
            .with_context(|| format!("reading {}", local.display()))?;
        for i in 0..archive.len() {
            let mut entry = archive.by_index(i)?;
            if !entry.is_file() {
                continue;
            }
            let Some(name) = entry.enclosed_name() else {
                continue;
            };
            let Some(target) = safe_join(&staging, &name) else {
                continue;
            };
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut out = File::create(&target)?;
            std::io::copy(&mut entry, &mut out)?;
        }
    } else if lower.ends_with(".tar.gz") || lower.ends_with(".tgz") {
        unpack_tar(flate2::read::GzDecoder::new(reader), &staging)?;
    } else if lower.ends_with(".tar.bz2") || lower.ends_with(".tbz2") || lower.ends_with(".tbz") {
        unpack_tar(bzip2::read::BzDecoder::new(reader), &staging)?;
    } else if lower.ends_with(".tar") {
        unpack_tar(reader, &staging)?;
    } else {
        bail!("{url} is not an archive shape this tool unpacks");
    }

    fs::rename(&staging, dest)
        .with_context(|| format!("renaming {} into place", staging.display()))?;
    Ok((dest.to_path_buf(), "unpacked"))
}

fn unpack_tar(reader: impl Read, staging: &Path) -> Result<()> {
    let mut archive = tar::Archive::new(reader);
    for entry in archive.entries()? {
        let mut entry = entry?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry.path()?.into_owned();
        let Some(target) = safe_join(staging, &path) else {
            continue;
        };
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut out = File::create(&target)?;
        std::io::copy(&mut entry, &mut out)?;
    }
    Ok(())
}

/// Joins an archive entry's own path onto `root`, or `None` if it tries to
/// leave.
///
/// An archive is somebody else's data, and an entry named `../../.bashrc`
/// unpacks over a home directory unless something stops it. Absolute
/// paths, parent components and root prefixes are all dropped rather than
/// sanitized into something else.
fn safe_join(root: &Path, entry: &Path) -> Option<PathBuf> {
    let mut out = root.to_path_buf();
    let mut any = false;
    for component in entry.components() {
        match component {
            Component::Normal(part) => {
                out.push(part);
                any = true;
            }
            Component::CurDir => {}
            _ => return None,
        }
    }
    any.then_some(out)
}

/// Every file under `roots` worth reading, and what was left out.
/// Every file under `roots` worth reading, and what was left out.
pub fn scan(roots: &[Root]) -> (Vec<PathBuf>, ScanReport) {
    let mut files = Vec::new();
    let mut report = ScanReport::default();

    for root in roots {
        let max_file_bytes = root.max_file_size.unwrap_or(u64::MAX);
        let walk = WalkDir::new(&root.path)
            .follow_links(false)
            .into_iter()
            .filter_entry(|e| {
                !e.file_type().is_dir()
                    || !e
                        .file_name()
                        .to_str()
                        .is_some_and(|n| SKIP_DIRS.contains(&n))
            });
        for entry in walk.filter_map(|e| e.ok()) {
            if !entry.file_type().is_file() {
                continue;
            }
            if !wanted_name(entry.path()) {
                report.skipped_extension += 1;
                continue;
            }
            match entry.metadata() {
                Ok(md) if md.len() <= max_file_bytes && md.len() > 0 => {
                    report.kept += 1;
                    files.push(entry.into_path());
                }
                Ok(_) => report.skipped_large += 1,
                Err(_) => report.skipped_large += 1,
            }
        }
    }
    files.sort();
    (files, report)
}

/// The compression a file's name declares, if any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Compression {
    Gzip,
    Bzip2,
}

fn compression(name: &str) -> Option<Compression> {
    let lower = name.to_ascii_lowercase();
    if lower.ends_with(".gz") {
        Some(Compression::Gzip)
    } else if lower.ends_with(".bz2") {
        Some(Compression::Bzip2)
    } else {
        None
    }
}

/// A file's name with any compression suffix removed — what decides
/// whether it is training text. `main.rs.gz` is Rust; `image.png.gz` is
/// still a picture.
fn uncompressed_name(name: &str) -> &str {
    match compression(name) {
        Some(Compression::Gzip) => &name[..name.len() - 3],
        Some(Compression::Bzip2) => &name[..name.len() - 4],
        None => name,
    }
}

/// Whether a path's name marks it as text this tool trains on.
fn wanted_name(path: &Path) -> bool {
    let name = match path.file_name().and_then(|n| n.to_str()) {
        Some(n) => uncompressed_name(n),
        None => return false,
    };
    if KEEP_NAMES.contains(&name) {
        return true;
    }
    // A minified or bundled artefact is source-shaped but is not source.
    if name.contains(".min.") || name.contains(".bundle.") || name.ends_with(".lock") {
        return false;
    }
    Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|ext| KEEP_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
}

/// The most any one file expands to. A compressed file's size on disk says
/// nothing about what it expands to, and an archive from somewhere else is
/// not something to hand an unbounded allocation to.
const MAX_DECOMPRESSED: u64 = 64 << 20;

/// Reads one file as training text, or `None` if it is not UTF-8 text.
///
/// A stray NUL rules a file out even when the bytes happen to decode: it is
/// the reliable marker of a binary that was given a source extension.
pub fn read_document(path: &Path) -> Option<String> {
    let name = path.file_name()?.to_str()?;
    let bytes = match compression(name) {
        None => fs::read(path).ok()?,
        Some(kind) => {
            let file = BufReader::with_capacity(1 << 16, File::open(path).ok()?);
            let mut out = Vec::new();
            let read = match kind {
                Compression::Gzip => flate2::read::GzDecoder::new(file)
                    .take(MAX_DECOMPRESSED)
                    .read_to_end(&mut out),
                Compression::Bzip2 => bzip2::read::BzDecoder::new(file)
                    .take(MAX_DECOMPRESSED)
                    .read_to_end(&mut out),
            };
            read.ok()?;
            out
        }
    };
    if bytes.contains(&0) {
        return None;
    }
    String::from_utf8(bytes).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A source is what it turns out to be, not what its name suggests.
    #[test]
    fn sources_are_classified_by_looking() {
        let dir = tempfile::tempdir().unwrap();

        // A remote is a clone unless it names an archive.
        assert_eq!(classify("https://github.com/a/b"), Source::Git);
        assert_eq!(classify("https://github.com/a/b.git"), Source::Git);
        assert_eq!(classify("git@github.com:a/b.git"), Source::Git);
        assert_eq!(
            classify("https://example.com/corpus.tar.gz"),
            Source::Archive
        );
        assert_eq!(classify("https://example.com/corpus.zip"), Source::Archive);

        // A local repository is a clone; a plain directory is not.
        let repo = dir.path().join("repo");
        fs::create_dir_all(repo.join(".git")).unwrap();
        assert_eq!(classify(repo.to_str().unwrap()), Source::Git);

        let plain = dir.path().join("plain");
        fs::create_dir_all(&plain).unwrap();
        fs::write(plain.join("a.rs"), "fn main() {}").unwrap();
        assert_eq!(classify(plain.to_str().unwrap()), Source::Directory);

        // A local archive is one whatever it sits next to.
        let archive = dir.path().join("corpus.tar.bz2");
        fs::write(&archive, b"").unwrap();
        assert_eq!(classify(archive.to_str().unwrap()), Source::Archive);

        // A path that is not there is left for git to report on.
        assert_eq!(classify("/no/such/place"), Source::Git);
    }

    /// A directory of files is read where it is. Copying a corpus to look
    /// at it would double the disk for nothing.
    #[test]
    fn a_directory_source_is_read_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("prose");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("a.txt"), "hello").unwrap();

        let manifest = crate::manifest::Manifest::parse(&format!(
            r#"{{"repositories": [{{"url": {:?}, "license": "MIT"}}]}}"#,
            source.to_str().unwrap()
        ))
        .unwrap();
        let corpus = dir.path().join("corpus");
        let roots = fetch_all(&manifest, &corpus, 1).unwrap();
        assert_eq!(roots, vec![source.clone()], "read where it is");
        assert!(
            !corpus.join("prose").exists(),
            "nothing was copied into the corpus directory"
        );
    }

    /// An archive entry that tries to climb out of the directory is
    /// dropped, not sanitized into something else.
    #[test]
    fn an_archive_cannot_write_outside_its_directory() {
        let root = Path::new("/corpus/x");
        assert_eq!(
            safe_join(root, Path::new("src/main.rs")),
            Some(PathBuf::from("/corpus/x/src/main.rs"))
        );
        assert_eq!(safe_join(root, Path::new("../../.bashrc")), None);
        assert_eq!(safe_join(root, Path::new("/etc/passwd")), None);
        assert_eq!(safe_join(root, Path::new("")), None);
        assert_eq!(
            safe_join(root, Path::new("./a/./b.rs")),
            Some(PathBuf::from("/corpus/x/a/b.rs"))
        );
    }

    /// A compressed file is read through, and what decides whether it is
    /// training text is the name underneath the compression.
    #[test]
    fn compressed_files_are_read_through() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();

        let gz = dir.path().join("main.rs.gz");
        let mut encoder =
            flate2::write::GzEncoder::new(File::create(&gz).unwrap(), flate2::Compression::fast());
        encoder.write_all(b"fn main() {}").unwrap();
        encoder.finish().unwrap();

        assert!(wanted_name(&gz), "the name under .gz is Rust");
        assert_eq!(read_document(&gz).as_deref(), Some("fn main() {}"));

        assert!(!wanted_name(Path::new("logo.png.gz")), "still a picture");
        assert!(!wanted_name(Path::new("archive.tar.bz2")));
        assert!(wanted_name(Path::new("notes.md.bz2")));
        assert_eq!(uncompressed_name("main.rs.gz"), "main.rs");
        assert_eq!(uncompressed_name("main.rs"), "main.rs");
    }

    #[test]
    fn keeps_source_and_prose_and_drops_the_rest() {
        assert!(wanted_name(Path::new("a/b/main.rs")));
        assert!(wanted_name(Path::new("a/b/README.md")));
        assert!(wanted_name(Path::new("a/b/Makefile")));
        assert!(!wanted_name(Path::new("a/b/logo.png")));
        assert!(!wanted_name(Path::new("a/b/app.min.js")));
        assert!(!wanted_name(Path::new("a/b/Cargo.lock")));
    }

    #[test]
    fn a_binary_with_a_source_extension_is_not_text() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("fake.rs");
        fs::write(&path, b"fn main() {}\0garbage").unwrap();
        assert!(read_document(&path).is_none());
    }

    #[test]
    fn scan_skips_generated_trees_and_oversized_files() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("node_modules/pkg")).unwrap();
        fs::write(dir.path().join("node_modules/pkg/index.js"), "x").unwrap();
        fs::write(dir.path().join("keep.rs"), "fn main() {}").unwrap();
        fs::write(dir.path().join("huge.rs"), vec![b'x'; 4096]).unwrap();

        let (files, report) = scan(&[Root::repository(dir.path().to_path_buf(), 1024)]);
        assert_eq!(files.len(), 1, "{files:?}");
        assert!(files[0].ends_with("keep.rs"));
        assert_eq!(report.skipped_large, 1);
    }

    /// The size cap must not reach the prose this tool downloaded itself.
    /// A Wikipedia shard is tens of megabytes of exactly the text the
    /// corpus is for, and the cap silently threw it away.
    #[test]
    fn a_generated_root_ignores_the_size_cap() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("enwiki-0.txt"), vec![b'a'; 64 * 1024]).unwrap();

        let (capped, report) = scan(&[Root::repository(dir.path().to_path_buf(), 1024)]);
        assert!(capped.is_empty());
        assert_eq!(report.skipped_large, 1);

        let (kept, report) = scan(&[Root::generated(dir.path().to_path_buf())]);
        assert_eq!(kept.len(), 1);
        assert_eq!(report.skipped_large, 0);
    }
}
