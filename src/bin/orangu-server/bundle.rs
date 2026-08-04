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

//! One file that is both the server and the model it serves.
//!
//! `orangu-server bundle <model>` writes a **new** executable: this binary's
//! own program image, followed by the model's `.gguf` bytes, followed by a
//! manifest naming where they landed. The result is an ordinary executable —
//! the loader reads the program image at the front and never looks at what
//! follows it — that needs no models directory, no download, and (see
//! [`crate::config::bundled_configuration`]) no `orangu-server.conf`: one
//! download, `chmod +x`, run.
//!
//! ## Why appended rather than compiled in
//!
//! The obvious alternative — `include_bytes!` — would put a multi-gigabyte
//! array through `rustc` on every build, tie one binary to one model at
//! compile time, and make a bundle something only whoever can build the
//! project can produce. Appending instead means the bundle is made *from* a
//! finished binary, in seconds, by anyone who has one: `bundle` is a file
//! operation, not a build step.
//!
//! It also means a bundled binary can bundle again — [`base_length`] finds
//! the program image inside whatever it is given, so re-bundling swaps the
//! model rather than stacking a second copy on top of the first.
//!
//! ## Layout
//!
//! ```text
//! [ program image                       ]  base_len bytes, byte-identical
//! [ padding to a 4 KiB boundary         ]
//! [ shard 1 .gguf                       ]
//! [ padding, shard 2 .gguf, ...         ]  only a split model has these
//! [ manifest (JSON)                     ]
//! [ manifest_offset: u64                ]  ─┐
//! [ manifest_len:    u64                ]   ├ the 32-byte footer
//! [ MAGIC:           16 bytes           ]  ─┘
//! ```
//!
//! Every shard starts on a 4 KiB boundary so its tensor data keeps the same
//! alignment relative to a page that it has in a file of its own — the
//! mapping is of the executable, and a model should not read differently for
//! having been carried in one.
//!
//! The footer is fixed-size and last, so finding a bundle is one seek and a
//! 32-byte read regardless of how large the payload is, and a binary with no
//! bundle is ruled out just as cheaply.
//!
//! ## Platform note
//!
//! Appending leaves ELF and PE images valid as they are. A Mach-O one is
//! code-signed, and copying half of a signed binary out of another file does
//! not carry the signature over intact, so on macOS the program image is
//! re-signed ad-hoc (`codesign --sign -`) — **before** the payload is
//! appended to it, since `codesign` writes the new signature at the end of
//! the image it was pointed at and would otherwise write it straight over
//! the model. See [`write_bundle`].

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{IsTerminal, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::OnceLock,
};

use crate::config::Role;

/// The last 16 bytes of a bundled executable. Carries its own format number
/// so a future layout change is a different magic rather than a manifest
/// this build would misread.
const MAGIC: &[u8; 16] = b"ORANGU-BUNDLE-01";

/// `manifest_offset` (8) + `manifest_len` (8) + [`MAGIC`] (16).
const FOOTER_LEN: u64 = 32;

/// What each shard's start is padded up to — see the module doc.
const ALIGN: u64 = 4096;

/// A sanity bound on the manifest length read out of the footer, so a file
/// that merely happens to end in [`MAGIC`] cannot ask for an arbitrary
/// allocation before a byte of it has been shown to be JSON.
const MAX_MANIFEST_LEN: u64 = 1024 * 1024;

/// The model `bundle` offers when it is not told which to use — the one the
/// project ships against, and the answer an empty line at the prompt takes.
/// Fetched from Hugging Face if it isn't already in the models directory,
/// exactly as naming it on the command line would.
pub const DEFAULT_MODEL: &str = "unsloth/gemma-4-E2B-it-GGUF:Q4_K_M";

/// The reserved model spec meaning "the model inside this binary".
///
/// Only ever needed to name the embedded model *again* — a bundled server
/// already serves it without being asked. That happens on the way back from
/// a handover: the web console loads a different model, the new image cannot
/// load it, and the fallback (`reexec::FALLBACK_MODEL_VAR`) has to name what
/// was running before. Its label is a Hugging Face repo, which would send
/// the fallback to the network for a model that is already in the file it is
/// falling back into.
pub const EMBEDDED_SPEC: &str = "bundled";

/// A shard's byte range within the bundled executable.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct ShardEntry {
    /// The `.gguf` file name this shard was made from — diagnostics only;
    /// nothing resolves against it.
    name: String,
    offset: u64,
    len: u64,
}

/// The JSON record between the payload and the footer: everything a bundled
/// binary needs to know at startup, and everything `bundle` needs to know to
/// re-bundle.
#[derive(Serialize, Deserialize, Debug, Clone)]
struct Manifest {
    /// Length of the program image the payload was appended to — where a
    /// re-bundle truncates back to. Not derivable from the file itself: the
    /// padding before shard 1 belongs to neither part.
    base_len: u64,
    /// The model's id on the API (`/v1/models`, every response's `model`
    /// field) — the spec it was bundled from.
    model: String,
    /// The quantization the file is stored at, for the startup banner. Left
    /// out when `model` already names one, so the banner never reads
    /// `...:Q4_K_M:Q4_K_M`.
    quantization: Option<String>,
    /// The role this bundle serves in unless a flag or config says otherwise.
    role: String,
    /// Where this bundle listens unless a flag or config says otherwise —
    /// `bundle`'s own `--host`/`--port`/`--web`. Absent fields, and a bundle
    /// written before these were recorded at all, mean the built-in defaults
    /// (`config::bundled_configuration`), which is why every one of them is
    /// optional and skipped when unset rather than written as a null.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    host: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    port: Option<u16>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    web: Option<u16>,
    /// The `orangu-server` version that wrote the bundle. Always this
    /// binary's own, since it is this binary's program image being copied —
    /// recorded so `bundle` can report it.
    version: String,
    shards: Vec<ShardEntry>,
}

/// A model found inside the running executable.
#[derive(Debug, Clone)]
pub struct Bundle {
    /// The executable carrying it — what gets mapped to read the weights.
    pub exe: PathBuf,
    pub model: String,
    pub quantization: Option<String>,
    pub role: Role,
    /// Where this bundle listens unless this run says otherwise. See
    /// [`crate::config::BundledListen`].
    pub listen: crate::config::BundledListen,
    /// Where each shard's GGUF structure starts, in shard order — what
    /// `engine::loader::LoadedModel::open_bundled` maps against.
    pub shard_offsets: Vec<u64>,
    /// Total size of the embedded model, for the startup banner.
    pub bytes: u64,
}

/// The model inside *this* executable, or `None` for an ordinary build.
///
/// Read once and cached: it costs a seek and a small read, but `prepare` and
/// the banner both ask, and a bundled binary that grew a second answer part
/// way through startup would be worse than one that answered slowly.
///
/// A file whose footer matches but whose manifest does not parse is reported
/// on stderr and then treated as unbundled. It is a corrupt bundle, and the
/// alternative — refusing to start at all — would take away the one thing
/// that might still work (`orangu-server <model>` against a real file).
pub fn embedded() -> Option<&'static Bundle> {
    static EMBEDDED: OnceLock<Option<Bundle>> = OnceLock::new();
    EMBEDDED.get_or_init(load_embedded).as_ref()
}

fn load_embedded() -> Option<Bundle> {
    let exe = std::env::current_exe().ok()?;
    match read_manifest(&exe) {
        Ok(Some(manifest)) => match Role::parse(&manifest.role) {
            Ok(role) => Some(Bundle {
                bytes: manifest.shards.iter().map(|shard| shard.len).sum(),
                shard_offsets: manifest.shards.iter().map(|shard| shard.offset).collect(),
                exe,
                model: manifest.model,
                quantization: manifest.quantization,
                role,
                listen: crate::config::BundledListen {
                    host: manifest.host,
                    port: manifest.port,
                    web: manifest.web,
                },
            }),
            Err(err) => {
                eprintln!("warning: ignoring the embedded model: {err:#}");
                None
            }
        },
        Ok(None) => None,
        Err(err) => {
            eprintln!("warning: ignoring the embedded model: {err:#}");
            None
        }
    }
}

/// The manifest inside `path`, `Ok(None)` when it carries no bundle.
///
/// The distinction matters: "not a bundle" is the ordinary state of every
/// `orangu-server` ever built and must not be an error, while "says it is a
/// bundle and then isn't readable" is a broken file and must not be silent.
fn read_manifest(path: &Path) -> Result<Option<Manifest>> {
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let len = file
        .metadata()
        .with_context(|| format!("failed to stat {}", path.display()))?
        .len();
    if len < FOOTER_LEN {
        return Ok(None);
    }

    let mut footer = [0u8; FOOTER_LEN as usize];
    file.seek(SeekFrom::Start(len - FOOTER_LEN))?;
    file.read_exact(&mut footer)
        .with_context(|| format!("failed to read the footer of {}", path.display()))?;
    if &footer[16..] != MAGIC {
        return Ok(None);
    }

    let manifest_offset = u64::from_le_bytes(footer[..8].try_into().expect("8 bytes"));
    let manifest_len = u64::from_le_bytes(footer[8..16].try_into().expect("8 bytes"));
    if manifest_len == 0 || manifest_len > MAX_MANIFEST_LEN {
        bail!("{} declares a {manifest_len}-byte manifest", path.display());
    }
    if manifest_offset + manifest_len != len - FOOTER_LEN {
        bail!(
            "{} declares a manifest at {manifest_offset}..{} but is {len} bytes",
            path.display(),
            manifest_offset + manifest_len
        );
    }

    let mut json = vec![0u8; manifest_len as usize];
    file.seek(SeekFrom::Start(manifest_offset))?;
    file.read_exact(&mut json)
        .with_context(|| format!("failed to read the manifest of {}", path.display()))?;
    let manifest: Manifest = serde_json::from_slice(&json)
        .with_context(|| format!("failed to parse the manifest of {}", path.display()))?;

    for shard in &manifest.shards {
        if shard.offset < manifest.base_len || shard.offset + shard.len > manifest_offset {
            bail!(
                "{} places shard '{}' outside its own payload",
                path.display(),
                shard.name
            );
        }
    }
    if manifest.shards.is_empty() {
        bail!("{} carries a manifest but no model", path.display());
    }

    Ok(Some(manifest))
}

/// The length of the program image inside `path`: the whole file for an
/// ordinary binary, and everything before the payload for one that is
/// already a bundle — which is what makes re-bundling replace the model
/// rather than append a second one.
fn base_length(path: &Path) -> Result<u64> {
    if let Some(manifest) = read_manifest(path)? {
        return Ok(manifest.base_len);
    }
    Ok(std::fs::metadata(path)
        .with_context(|| format!("failed to stat {}", path.display()))?
        .len())
}

/// Everything the `bundle` subcommand was given, already resolved by the
/// caller — see `main::run_command`, which owns the CLI shape.
pub struct Request {
    /// Where a model spec is resolved against, and downloaded into.
    pub models_dir: PathBuf,
    /// The model to embed, or `None` to ask.
    pub model: Option<String>,
    /// The role flag that was given, or `None` to ask (and to fall back to
    /// [`Role::default`] when there is nobody to ask).
    pub role: Option<Role>,
    /// The address and ports to bake in, each `None` where no flag was
    /// given and the bundle should take the built-in default.
    pub listen: crate::config::BundledListen,
    /// Where to write. Defaults to [`default_output`].
    pub output: Option<PathBuf>,
    /// The executable to bundle. Defaults to this one — the flag exists for
    /// bundling a build for another platform, which cannot be run here to
    /// bundle itself.
    pub binary: Option<PathBuf>,
    /// Skip both prompts: the confirmation, and the role question when no
    /// role flag was given.
    pub yes: bool,
}

/// What a binary was built to run on, read out of its own header rather than
/// assumed from the machine doing the bundling.
///
/// The distinction is the whole reason this exists. `--binary` is for
/// bundling a build for *another* platform, and a bundle is a file somebody
/// downloads — so the one thing its name has to say is which machine it runs
/// on. Taking that from the host would put `x86_64` on the `aarch64` bundle
/// cross-built beside it, which is worse than saying nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
struct BinaryTarget {
    /// The architecture, spelled the way a Rust target triple spells it —
    /// `x86_64`, `aarch64`, and so on, matching `std::env::consts::ARCH`'s
    /// vocabulary so the name lines up with the target somebody built for.
    arch: String,
    /// Whether it is a PE image, and so wants a `.exe` suffix — which, like
    /// the architecture, follows the binary rather than the host.
    windows: bool,
}

impl BinaryTarget {
    /// This machine's own, for a binary whose header says nothing this
    /// recognizes. A wrong-but-plausible guess is the right failure here:
    /// the name is a label, not something anything resolves against, and
    /// `bundle` shouldn't refuse to run over an executable format it merely
    /// hasn't been taught to read.
    fn host() -> Self {
        Self {
            arch: std::env::consts::ARCH.to_string(),
            windows: cfg!(windows),
        }
    }
}

/// Reads `path`'s executable header far enough to name the architecture it
/// was built for. Falls back to [`BinaryTarget::host`] for anything
/// unrecognized, including an unreadable file — the caller is about to open
/// it properly anyway, and will report that failure with context this can't.
fn detect_target(path: &Path) -> BinaryTarget {
    let Ok(header) = read_prefix(path, 4096) else {
        return BinaryTarget::host();
    };
    elf_target(&header)
        .or_else(|| macho_target(&header))
        .or_else(|| pe_target(&header))
        .unwrap_or_else(BinaryTarget::host)
}

/// Up to `len` bytes from the front of `path` — fewer if the file is
/// shorter, which the header parsers below treat as "not this format".
fn read_prefix(path: &Path, len: usize) -> std::io::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let mut buffer = vec![0u8; len];
    let mut filled = 0;
    while filled < len {
        match file.read(&mut buffer[filled..])? {
            0 => break,
            read => filled += read,
        }
    }
    buffer.truncate(filled);
    Ok(buffer)
}

fn u16_at(bytes: &[u8], offset: usize, little_endian: bool) -> Option<u16> {
    let raw: [u8; 2] = bytes.get(offset..offset + 2)?.try_into().ok()?;
    Some(if little_endian {
        u16::from_le_bytes(raw)
    } else {
        u16::from_be_bytes(raw)
    })
}

fn u32_at(bytes: &[u8], offset: usize, little_endian: bool) -> Option<u32> {
    let raw: [u8; 4] = bytes.get(offset..offset + 4)?.try_into().ok()?;
    Some(if little_endian {
        u32::from_le_bytes(raw)
    } else {
        u32::from_be_bytes(raw)
    })
}

/// ELF (Linux, and every other ELF platform): `e_machine` at offset 18, in
/// the endianness `EI_DATA` (offset 5) declares. `EI_CLASS` (offset 4)
/// separates the 32- and 64-bit RISC-V machines, which share one
/// `e_machine`.
fn elf_target(header: &[u8]) -> Option<BinaryTarget> {
    if header.first_chunk::<4>()? != b"\x7fELF" {
        return None;
    }
    let little_endian = *header.get(5)? != 2;
    let sixty_four_bit = *header.get(4)? == 2;
    let arch = match u16_at(header, 18, little_endian)? {
        0x03 => "x86",
        0x3E => "x86_64",
        0x28 => "arm",
        0xB7 => "aarch64",
        0x08 => "mips",
        0x15 => "powerpc64",
        0x16 => "s390x",
        0xF3 if sixty_four_bit => "riscv64",
        0xF3 => "riscv32",
        0x2B => "sparc64",
        0x66 => "loongarch64",
        _ => return None,
    };
    Some(BinaryTarget {
        arch: arch.to_string(),
        windows: false,
    })
}

/// Mach-O (macOS): `cputype` at offset 4. A universal binary — several
/// architectures in one file, which is what `lipo` produces and what a Mac
/// release usually ships — is named `universal` rather than after whichever
/// slice happens to come first, since it genuinely runs on both.
fn macho_target(header: &[u8]) -> Option<BinaryTarget> {
    const FAT_MAGIC: u32 = 0xCAFE_BABE;
    const FAT_MAGIC_64: u32 = 0xCAFE_BABF;

    let magic = u32_at(header, 0, true)?;
    let little_endian = match magic {
        0xFEED_FACF | 0xFEED_FACE => true,
        0xCFFA_EDFE | 0xCEFA_EDFE => false,
        _ => {
            // The fat header is big-endian regardless of its slices'.
            let fat = u32_at(header, 0, false)?;
            if fat != FAT_MAGIC && fat != FAT_MAGIC_64 {
                return None;
            }
            return Some(BinaryTarget {
                arch: match u32_at(header, 4, false)? {
                    // One slice is not really a universal binary; fall
                    // through to naming its architecture below.
                    1 => return macho_slice(header),
                    _ => "universal".to_string(),
                },
                windows: false,
            });
        }
    };
    let arch = macho_arch(u32_at(header, 4, little_endian)?)?;
    Some(BinaryTarget {
        arch: arch.to_string(),
        windows: false,
    })
}

/// The architecture of a single-slice fat binary: `cputype` is the first
/// word of the one `fat_arch` record, which starts at offset 8.
fn macho_slice(header: &[u8]) -> Option<BinaryTarget> {
    Some(BinaryTarget {
        arch: macho_arch(u32_at(header, 8, false)?)?.to_string(),
        windows: false,
    })
}

fn macho_arch(cputype: u32) -> Option<&'static str> {
    match cputype {
        0x0100_000C => Some("aarch64"),
        0x0100_0007 => Some("x86_64"),
        0x0000_000C => Some("arm"),
        0x0000_0007 => Some("x86"),
        _ => None,
    }
}

/// PE (Windows): `MZ`, then the COFF header at the offset stored at `0x3C`,
/// whose `Machine` field is two bytes past the `PE\0\0` signature.
fn pe_target(header: &[u8]) -> Option<BinaryTarget> {
    if header.first_chunk::<2>()? != b"MZ" {
        return None;
    }
    let pe_offset = u32_at(header, 0x3C, true)? as usize;
    if header.get(pe_offset..pe_offset + 4)? != b"PE\0\0" {
        return None;
    }
    let arch = match u16_at(header, pe_offset + 4, true)? {
        0x8664 => "x86_64",
        0xAA64 => "aarch64",
        0x014C => "x86",
        0x01C4 => "arm",
        _ => return None,
    };
    Some(BinaryTarget {
        arch: arch.to_string(),
        windows: true,
    })
}

/// The default output name — `orangu-server-bundle-<arch>`, plus `.exe` for
/// a Windows binary.
///
/// The architecture is in the name because a bundle is a file that gets
/// copied around, and its one hard requirement is a machine that can run it:
/// a directory holding bundles for three platforms has to be readable, and
/// `orangu-server-bundle` three times over is not. It also keeps a
/// cross-bundling run from writing over the bundle it made a moment ago for
/// a different target.
///
/// Never the running binary's own name either way, so a `bundle` run in a
/// directory that happens to hold one cannot overwrite it.
fn default_output(target: &BinaryTarget) -> PathBuf {
    let suffix = if target.windows { ".exe" } else { "" };
    PathBuf::from(format!("orangu-server-bundle-{}{suffix}", target.arch))
}

pub fn run(request: Request) -> Result<()> {
    let Request {
        models_dir,
        model,
        role,
        listen,
        output,
        binary,
        yes,
    } = request;
    // Checked here rather than left for the target machine's `bind` to
    // reject: a bundle is built once and run somewhere else, so a typo in
    // the address belongs to the build, where there is somebody to tell.
    validate_host(listen.host.as_deref())?;

    let source = match binary {
        Some(path) => path,
        None => std::env::current_exe().context("failed to resolve this executable")?,
    };
    if !source.is_file() {
        bail!("{} is not a file", source.display());
    }

    let (model_path, label) = match model {
        Some(spec) => (
            orangu::model_spec::resolve_or_fetch_model(&models_dir, &spec)
                .with_context(|| format!("resolving model '{spec}'"))?,
            spec,
        ),
        None => select_model(&models_dir)?,
    };
    // Asked in the same order the ordinary interactive startup asks — model,
    // then role — so the two flows read alike. A `--yes` run, or one with no
    // terminal to ask on, takes the default rather than hanging.
    let role = match role {
        Some(role) => role,
        None if yes || !std::io::stdin().is_terminal() => Role::default(),
        None => {
            let role = crate::init::prompt_role("Role: ", Role::default())?;
            crate::init::echo_answer("Role: ", role.label());
            role
        }
    };

    let gguf = orangu::gguf::GgufFile::open(&model_path)?;
    // The same header check the web console's Load button runs before a
    // handover, for the same reason: a bundle that cannot be loaded is worth
    // catching now, not after a multi-gigabyte copy.
    crate::reexec::precheck(&model_path)
        .with_context(|| format!("{} cannot be served by this build", model_path.display()))?;
    let shards = crate::engine::loader::shard_paths(&model_path, &gguf)?;
    let quantization = (!crate::label_carries_tag(&label))
        .then(|| orangu::model_spec::quantization_for_file(&model_path, &gguf))
        .flatten();

    let target = detect_target(&source);
    let output = output.unwrap_or_else(|| default_output(&target));
    if same_file(&source, &output) {
        bail!(
            "refusing to bundle {} onto itself; pass a different --output",
            source.display()
        );
    }

    let base_len = base_length(&source)?;
    let model_bytes: u64 = shards
        .iter()
        .map(|shard| {
            std::fs::metadata(shard)
                .map(|meta| meta.len())
                .with_context(|| format!("failed to stat {}", shard.display()))
        })
        .sum::<Result<u64>>()?;

    let display = match &quantization {
        Some(quant) => format!("{label}:{quant}"),
        None => label.clone(),
    };
    println!("Model      {display} ({})", format_bytes(model_bytes));
    println!("Role       {}", role.label());
    println!("Listen     {}", describe_listen(&listen));
    // The architecture is named here, not only implied by the output name,
    // because it is read off the binary rather than assumed — and a
    // cross-bundling run where that came out wrong is much easier to notice
    // on its own line than in a file name.
    println!(
        "Binary     {} ({}, {})",
        source.display(),
        format_bytes(base_len),
        target.arch
    );
    println!(
        "Output     {} ({})",
        output.display(),
        format_bytes(base_len + model_bytes)
    );
    if !yes {
        let question = if output.exists() {
            format!("\nOverwrite {} with this bundle? [y/N]: ", output.display())
        } else {
            "\nWrite this bundle? [y/N]: ".to_string()
        };
        if !crate::confirm(&question)? {
            println!("Aborted. Nothing written.");
            return Ok(());
        }
    }

    let manifest = write_bundle(
        &source,
        base_len,
        &shards,
        &output,
        |image_len, shard_entries| Manifest {
            base_len: image_len,
            model: label.clone(),
            quantization: quantization.clone(),
            role: role.label().to_string(),
            host: listen.host.clone(),
            port: listen.port,
            web: listen.web,
            version: crate::VERSION.to_string(),
            shards: shard_entries,
        },
    )?;

    // Read back rather than trusted: everything above is what was *meant* to
    // be written, and the one question that matters is whether the file on
    // disk is a bundle this binary's own startup path would accept. On macOS
    // this is also what would catch a `codesign` that rewrote more of the
    // file than the program image it was pointed at.
    match read_manifest(&output)? {
        Some(written) if written.shards.len() == manifest.shards.len() => {}
        _ => bail!("{} was written but is not a valid bundle", output.display()),
    }

    println!(
        "Wrote {} ({})",
        output.display(),
        format_bytes(
            std::fs::metadata(&output)
                .map(|meta| meta.len())
                .unwrap_or(base_len + model_bytes)
        )
    );
    Ok(())
}

/// Copies the program image and every shard into `output`, then the manifest
/// `build` makes from where the shards actually landed, then the footer.
/// Returns the manifest that was written.
///
/// The image is written, made executable, and (on macOS) **signed** before a
/// single payload byte follows it. The order is deliberate and is the whole
/// reason this is one function rather than two: a Mach-O signature covers the
/// program image up to a recorded `codeLimit`, and `codesign` writes the new
/// signature at the end of that image — over anything already sitting there.
/// Signing first leaves the payload outside the signed range, where the
/// kernel never looks, instead of underneath it.
///
/// Signing can change the image's length, so `build` is handed the length the
/// image actually ended up with rather than the `base_len` asked for. That is
/// what a later re-bundle truncates back to, so the prefix it copies is the
/// signed image, still valid on its own.
fn write_bundle(
    source: &Path,
    base_len: u64,
    shards: &[PathBuf],
    output: &Path,
    build: impl FnOnce(u64, Vec<ShardEntry>) -> Manifest,
) -> Result<Manifest> {
    let total: u64 = base_len
        + shards
            .iter()
            .map(|shard| std::fs::metadata(shard).map(|meta| meta.len()).unwrap_or(0))
            .sum::<u64>();
    let mut progress = Progress::new(total);

    {
        let mut out = File::create(output)
            .with_context(|| format!("failed to create {}", output.display()))?;
        let mut input =
            File::open(source).with_context(|| format!("failed to open {}", source.display()))?;
        copy_exactly(&mut input, base_len, &mut out, &mut progress)
            .with_context(|| format!("failed to copy {}", source.display()))?;
        out.flush()
            .with_context(|| format!("failed to write {}", output.display()))?;
    }
    make_executable(output)?;
    sign(output);

    let image_len = std::fs::metadata(output)
        .with_context(|| format!("failed to stat {}", output.display()))?
        .len();
    let mut out = std::fs::OpenOptions::new()
        .append(true)
        .open(output)
        .with_context(|| format!("failed to reopen {}", output.display()))?;

    let mut position = image_len;
    let mut entries = Vec::with_capacity(shards.len());
    for shard in shards {
        position = pad(&mut out, position)?;
        let mut file =
            File::open(shard).with_context(|| format!("failed to open {}", shard.display()))?;
        let len = file
            .metadata()
            .with_context(|| format!("failed to stat {}", shard.display()))?
            .len();
        copy_exactly(&mut file, len, &mut out, &mut progress)
            .with_context(|| format!("failed to copy {}", shard.display()))?;
        entries.push(ShardEntry {
            name: shard
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_default(),
            offset: position,
            len,
        });
        position += len;
    }
    progress.finish();

    let manifest = build(image_len, entries);
    let json = serde_json::to_vec(&manifest).context("failed to encode the bundle manifest")?;
    let manifest_offset = position;
    out.write_all(&json)
        .with_context(|| format!("failed to write {}", output.display()))?;

    let mut footer = [0u8; FOOTER_LEN as usize];
    footer[..8].copy_from_slice(&manifest_offset.to_le_bytes());
    footer[8..16].copy_from_slice(&(json.len() as u64).to_le_bytes());
    footer[16..].copy_from_slice(MAGIC);
    out.write_all(&footer)
        .with_context(|| format!("failed to write {}", output.display()))?;
    out.flush()
        .with_context(|| format!("failed to write {}", output.display()))?;

    Ok(manifest)
}

/// Pads `out` up to the next [`ALIGN`] boundary and returns the position it
/// now sits at — where the next shard begins.
fn pad(out: &mut File, position: u64) -> Result<u64> {
    let padded = position.div_ceil(ALIGN) * ALIGN;
    if padded > position {
        out.write_all(&vec![0u8; (padded - position) as usize])
            .context("failed to write padding")?;
    }
    Ok(padded)
}

/// Copies exactly `len` bytes, failing rather than writing a short payload
/// the manifest would then describe wrongly — a shard that shrank between
/// being measured and being read is the case this catches.
fn copy_exactly(input: &mut File, len: u64, out: &mut File, progress: &mut Progress) -> Result<()> {
    const CHUNK: usize = 4 * 1024 * 1024;
    let mut buffer = vec![0u8; CHUNK];
    let mut remaining = len;
    while remaining > 0 {
        let want = remaining.min(CHUNK as u64) as usize;
        let read = input.read(&mut buffer[..want])?;
        if read == 0 {
            bail!("expected {len} bytes but the file ended {remaining} bytes early");
        }
        out.write_all(&buffer[..read])?;
        remaining -= read as u64;
        progress.advance(read as u64);
    }
    Ok(())
}

/// A one-line percentage for a copy measured in gigabytes. Silent when
/// stdout isn't a terminal — the carriage returns would be the only thing
/// that landed in a log — and updated only when the whole percent changes,
/// so a slow disk doesn't turn into thousands of lines.
struct Progress {
    total: u64,
    done: u64,
    shown: u64,
    enabled: bool,
}

impl Progress {
    fn new(total: u64) -> Self {
        Self {
            total,
            done: 0,
            shown: u64::MAX,
            enabled: total > 0 && std::io::stdout().is_terminal(),
        }
    }

    fn advance(&mut self, bytes: u64) {
        self.done += bytes;
        if !self.enabled {
            return;
        }
        let percent = self.done * 100 / self.total;
        if percent != self.shown {
            self.shown = percent;
            print!("\rWriting    {percent}%");
            let _ = std::io::stdout().flush();
        }
    }

    fn finish(&mut self) {
        if self.enabled {
            print!("\r\x1b[2K");
            let _ = std::io::stdout().flush();
        }
    }
}

/// Whether the two paths name the same file, so `bundle` cannot be asked to
/// read and write one at once. Compared after resolving symlinks; an output
/// that doesn't exist yet plainly isn't the source.
fn same_file(source: &Path, output: &Path) -> bool {
    match (std::fs::canonicalize(source), std::fs::canonicalize(output)) {
        (Ok(source), Ok(output)) => source == output,
        _ => false,
    }
}

#[cfg(unix)]
fn make_executable(path: &Path) -> Result<()> {
    use std::os::unix::fs::PermissionsExt;

    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
        .with_context(|| format!("failed to make {} executable", path.display()))
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> Result<()> {
    Ok(())
}

/// Re-signs a freshly copied Mach-O program image ad-hoc — see the module doc
/// and [`write_bundle`] for why this runs before the payload is appended.
/// Best-effort and noisy on failure rather than fatal: the file is written
/// and correct, and `codesign` is the one part of this that depends on the
/// developer tools being installed.
#[cfg(target_os = "macos")]
fn sign(path: &Path) {
    let result = std::process::Command::new("codesign")
        .args(["--force", "--sign", "-"])
        .arg(path)
        .status();
    match result {
        Ok(status) if status.success() => {}
        Ok(status) => eprintln!(
            "warning: codesign exited with {status}; macOS will refuse to run {} until it is \
             signed (codesign --force --sign - {})",
            path.display(),
            path.display()
        ),
        Err(err) => eprintln!(
            "warning: could not run codesign ({err}); macOS will refuse to run {} until it is \
             signed (codesign --force --sign - {})",
            path.display(),
            path.display()
        ),
    }
}

#[cfg(not(target_os = "macos"))]
fn sign(_path: &Path) {}

/// The model picker `bundle` shows when it wasn't told which model to embed:
/// the same table `orangu-server list` prints, the same prompt the ordinary
/// interactive startup uses, and [`DEFAULT_MODEL`] pre-selected as the
/// answer an empty line takes.
///
/// Unlike `main::select_model_interactively`, an empty models directory is
/// not an error here. Nothing has to be installed to bundle: the answer is a
/// spec, and a spec that names a Hugging Face repo is fetched — which for the
/// default is exactly what a first-time `bundle` should do.
fn select_model(models_dir: &Path) -> Result<(PathBuf, String)> {
    let groups = orangu::model_spec::scan_models_dir(models_dir)
        .map(|models| orangu::model_spec::group_models(&models))
        .unwrap_or_default();
    if !groups.is_empty() {
        print!(
            "{}",
            orangu::model_spec::format_groups(
                &groups,
                models_dir,
                &Default::default(),
                &crate::model_support(&groups),
                crate::dimming(orangu::model_spec::Dimming::Unsupported),
            )
        );
    }

    let answer = crate::init::prompt_model_nr(&groups, Some(DEFAULT_MODEL))?;
    crate::init::echo_answer("Model: ", &answer);

    if let Ok(nr) = answer.parse::<usize>() {
        let group = nr
            .checked_sub(1)
            .and_then(|index| groups.get(index))
            .ok_or_else(|| anyhow!("no model with NR {nr} ({} model(s) listed)", groups.len()))?;
        return Ok((group.representative_path.clone(), group.label.clone()));
    }
    let path = orangu::model_spec::resolve_or_fetch_model(models_dir, &answer)
        .with_context(|| format!("resolving model '{answer}'"))?;
    Ok((path, answer))
}

fn format_bytes(bytes: u64) -> String {
    orangu::format::format_bytes(bytes)
}

/// Rejects a `--host` that could never bind, while there is still somebody
/// to tell.
///
/// A bundle is built on one machine and run on another, so an address that
/// isn't one would otherwise surface as a bind failure on a machine whose
/// operator may have no idea what it was supposed to say. `all`/`*` and any
/// literal IP address are accepted; a hostname is not, matching what
/// `[orangu-server].host` itself accepts.
fn validate_host(host: Option<&str>) -> Result<()> {
    let Some(host) = host else {
        return Ok(());
    };
    let trimmed = host.trim();
    if trimmed.eq_ignore_ascii_case(crate::config::HOST_ALL)
        || trimmed == crate::config::HOST_ALL_ALIAS
        || trimmed.parse::<std::net::IpAddr>().is_ok()
    {
        return Ok(());
    }
    bail!(
        "invalid --host '{host}': expected '{}' (or '{}') for every interface, \
         or a literal address such as 0.0.0.0 or 127.0.0.1",
        crate::config::HOST_ALL,
        crate::config::HOST_ALL_ALIAS
    )
}

/// The `Listen` line of `bundle`'s summary: what the bundle will bind when
/// it is started with no config file and no flags. Spelled out in full — the
/// defaults included — because the whole question this line answers is "what
/// will this thing be reachable on", and half an answer doesn't.
fn describe_listen(listen: &crate::config::BundledListen) -> String {
    let host = listen
        .host
        .as_deref()
        .unwrap_or(crate::config::BUNDLED_HOST);
    let api = listen.port.unwrap_or_else(crate::config::default_port);
    let web = listen.web.unwrap_or_else(crate::config::bundled_web_port);
    let console = match web {
        0 => "console off".to_string(),
        web => format!("console {host}:{web}"),
    };
    format!("API {host}:{api}, {console}")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in program image and a stand-in model, bundled and read back
    /// — the round trip every bundled start depends on.
    fn bundle_into(dir: &Path, image: &[u8], model: &[u8]) -> PathBuf {
        let binary = dir.join("orangu-server");
        std::fs::write(&binary, image).unwrap();
        let shard = dir.join("model.gguf");
        std::fs::write(&shard, model).unwrap();
        let output = dir.join("bundled");

        write_bundle(
            &binary,
            image.len() as u64,
            &[shard],
            &output,
            |image_len, shards| Manifest {
                base_len: image_len,
                model: "user/model".to_string(),
                quantization: Some("Q4_K_M".to_string()),
                role: "code".to_string(),
                host: Some("0.0.0.0".to_string()),
                port: Some(9100),
                web: Some(9200),
                version: "1.1.0".to_string(),
                shards,
            },
        )
        .unwrap();
        output
    }

    #[test]
    fn a_bundle_reads_back_what_it_wrote() {
        let dir = tempfile::tempdir().unwrap();
        let image = vec![0xAAu8; 5000];
        let model = vec![0x42u8; 3000];
        let output = bundle_into(dir.path(), &image, &model);

        let manifest = read_manifest(&output).unwrap().expect("a bundle");
        assert_eq!(manifest.base_len, 5000);
        assert_eq!(manifest.model, "user/model");
        assert_eq!(manifest.quantization.as_deref(), Some("Q4_K_M"));
        assert_eq!(Role::parse(&manifest.role).unwrap(), Role::Code);
        assert_eq!(manifest.shards.len(), 1);
        assert_eq!(manifest.shards[0].len, 3000);
    }

    /// The program image has to come out byte-identical, and the model has to
    /// be readable at the offset the manifest names — those two together are
    /// the whole contract with the OS loader and with
    /// `LoadedModel::open_bundled`.
    #[test]
    fn the_image_is_untouched_and_the_model_is_where_it_says() {
        let dir = tempfile::tempdir().unwrap();
        let image = vec![0xAAu8; 5000];
        let model: Vec<u8> = (0..3000u32).map(|i| i as u8).collect();
        let output = bundle_into(dir.path(), &image, &model);

        let written = std::fs::read(&output).unwrap();
        assert_eq!(&written[..image.len()], &image[..]);

        let manifest = read_manifest(&output).unwrap().unwrap();
        let shard = &manifest.shards[0];
        let start = shard.offset as usize;
        assert_eq!(&written[start..start + shard.len as usize], &model[..]);
    }

    /// Every shard starts on a page boundary, so a bundled model's tensor
    /// data is aligned exactly as it would be in a file of its own.
    #[test]
    fn shards_start_on_a_page_boundary() {
        let dir = tempfile::tempdir().unwrap();
        // A deliberately unaligned image length: the padding is what has to
        // make up the difference.
        let output = bundle_into(dir.path(), &vec![0u8; 4097], &[1u8; 64]);

        let manifest = read_manifest(&output).unwrap().unwrap();
        assert_eq!(manifest.shards[0].offset % ALIGN, 0);
        assert!(manifest.shards[0].offset >= manifest.base_len);
    }

    /// Bundling a bundle replaces the model instead of stacking a second one
    /// behind the first — otherwise every re-bundle would carry every model
    /// ever bundled into it.
    #[test]
    fn re_bundling_truncates_back_to_the_program_image() {
        let dir = tempfile::tempdir().unwrap();
        let image = vec![0xAAu8; 5000];
        let first = bundle_into(dir.path(), &image, &vec![1u8; 3000]);

        assert_eq!(base_length(&first).unwrap(), 5000);

        let shard = dir.path().join("second.gguf");
        std::fs::write(&shard, vec![2u8; 100]).unwrap();
        let second = dir.path().join("second");
        write_bundle(&first, 5000, &[shard], &second, |image_len, shards| {
            Manifest {
                base_len: image_len,
                model: "user/second".to_string(),
                quantization: None,
                role: "all".to_string(),
                host: None,
                port: None,
                web: None,
                version: "1.1.0".to_string(),
                shards,
            }
        })
        .unwrap();

        let manifest = read_manifest(&second).unwrap().unwrap();
        assert_eq!(manifest.model, "user/second");
        assert_eq!(manifest.shards.len(), 1);
        assert_eq!(manifest.shards[0].len, 100);
        // The second bundle is the image plus one small model, not the image
        // plus both.
        assert!(std::fs::metadata(&second).unwrap().len() < 5000 + ALIGN + 1000);
    }

    /// An ordinary binary is not an error, it is simply not a bundle — every
    /// `orangu-server` ever built takes this path at startup.
    #[test]
    fn an_ordinary_binary_carries_no_bundle() {
        let dir = tempfile::tempdir().unwrap();
        let plain = dir.path().join("orangu-server");
        std::fs::write(&plain, vec![0u8; 5000]).unwrap();
        assert!(read_manifest(&plain).unwrap().is_none());

        // Too short to hold a footer at all.
        let tiny = dir.path().join("tiny");
        std::fs::write(&tiny, b"no").unwrap();
        assert!(read_manifest(&tiny).unwrap().is_none());
    }

    /// ...but one that claims to be a bundle and isn't readable has to say
    /// so. Silently starting as if there were no model would send the user
    /// to a config file they were told they didn't need.
    #[test]
    fn a_corrupt_bundle_is_an_error_not_a_shrug() {
        let dir = tempfile::tempdir().unwrap();
        let output = bundle_into(dir.path(), &vec![0xAAu8; 5000], &vec![1u8; 3000]);

        // Corrupt the manifest's JSON, leaving the footer intact.
        let mut bytes = std::fs::read(&output).unwrap();
        let manifest_offset = u64::from_le_bytes(
            bytes[bytes.len() - 32..bytes.len() - 24]
                .try_into()
                .unwrap(),
        );
        bytes[manifest_offset as usize] = b'!';
        std::fs::write(&output, &bytes).unwrap();

        let err = read_manifest(&output).unwrap_err();
        assert!(err.to_string().contains("manifest"), "{err:#}");
    }

    /// A 64-bit little-endian ELF header carrying `e_machine`.
    fn elf_header(e_machine: u16, sixty_four_bit: bool) -> Vec<u8> {
        let mut header = vec![0u8; 64];
        header[..4].copy_from_slice(b"\x7fELF");
        header[4] = if sixty_four_bit { 2 } else { 1 }; // EI_CLASS
        header[5] = 1; // EI_DATA: little-endian
        header[18..20].copy_from_slice(&e_machine.to_le_bytes());
        header
    }

    #[test]
    fn an_elf_binary_names_the_architecture_it_was_built_for() {
        for (machine, expected) in [(0x3Eu16, "x86_64"), (0xB7, "aarch64"), (0x28, "arm")] {
            let target = elf_target(&elf_header(machine, true)).expect("an ELF target");
            assert_eq!(target.arch, expected);
            assert!(!target.windows);
        }
        // RISC-V shares one `e_machine` across widths; `EI_CLASS` is what
        // separates them.
        assert_eq!(elf_target(&elf_header(0xF3, true)).unwrap().arch, "riscv64");
        assert_eq!(
            elf_target(&elf_header(0xF3, false)).unwrap().arch,
            "riscv32"
        );
    }

    #[test]
    fn a_macho_binary_names_its_cputype_and_a_fat_one_says_universal() {
        let mut arm64 = vec![0u8; 32];
        arm64[..4].copy_from_slice(&0xFEED_FACFu32.to_le_bytes());
        arm64[4..8].copy_from_slice(&0x0100_000Cu32.to_le_bytes());
        assert_eq!(macho_target(&arm64).unwrap().arch, "aarch64");

        // A universal binary really does run on both, so it is not named
        // after whichever slice happens to be first.
        let mut fat = vec![0u8; 32];
        fat[..4].copy_from_slice(&0xCAFE_BABEu32.to_be_bytes());
        fat[4..8].copy_from_slice(&2u32.to_be_bytes()); // nfat_arch
        assert_eq!(macho_target(&fat).unwrap().arch, "universal");

        // ...but a one-slice fat binary is just that architecture.
        let mut single = vec![0u8; 32];
        single[..4].copy_from_slice(&0xCAFE_BABEu32.to_be_bytes());
        single[4..8].copy_from_slice(&1u32.to_be_bytes());
        single[8..12].copy_from_slice(&0x0100_0007u32.to_be_bytes());
        assert_eq!(macho_target(&single).unwrap().arch, "x86_64");
    }

    /// A PE binary decides the `.exe` suffix as well as the architecture —
    /// both follow the binary being bundled, not the machine bundling it.
    #[test]
    fn a_pe_binary_is_named_with_an_exe_suffix() {
        let mut header = vec![0u8; 256];
        header[..2].copy_from_slice(b"MZ");
        header[0x3C..0x40].copy_from_slice(&0x80u32.to_le_bytes());
        header[0x80..0x84].copy_from_slice(b"PE\0\0");
        header[0x84..0x86].copy_from_slice(&0x8664u16.to_le_bytes());

        let target = pe_target(&header).expect("a PE target");
        assert_eq!(target.arch, "x86_64");
        assert!(target.windows);
        assert_eq!(
            default_output(&target),
            PathBuf::from("orangu-server-bundle-x86_64.exe")
        );
    }

    #[test]
    fn the_default_output_carries_the_architecture() {
        assert_eq!(
            default_output(&BinaryTarget {
                arch: "aarch64".to_string(),
                windows: false,
            }),
            PathBuf::from("orangu-server-bundle-aarch64")
        );
    }

    /// A format this doesn't read is not a failure: the name is a label, and
    /// falling back to the host's own architecture beats refusing to bundle.
    #[test]
    fn an_unrecognized_header_falls_back_to_this_machine() {
        let dir = tempfile::tempdir().unwrap();
        let odd = dir.path().join("something-else");
        std::fs::write(&odd, b"not an executable this knows").unwrap();

        assert_eq!(detect_target(&odd), BinaryTarget::host());
        assert_eq!(
            detect_target(&dir.path().join("missing")),
            BinaryTarget::host()
        );
    }

    /// The end-to-end version of the two above: whatever built *this* test
    /// binary is what `detect_target` reads back off it.
    #[test]
    fn this_very_binary_reports_the_architecture_it_was_built_for() {
        let exe = std::env::current_exe().expect("current exe");

        assert_eq!(detect_target(&exe).arch, std::env::consts::ARCH);
    }

    /// The address `bundle --host`/`--port`/`--web` was given travels in the
    /// bundle, the same way the role does — a bundle is started without a
    /// config file, so where it listens has to be decidable when it is built.
    #[test]
    fn the_address_a_bundle_was_built_with_travels_with_it() {
        let dir = tempfile::tempdir().unwrap();
        let output = bundle_into(dir.path(), &vec![0xAAu8; 5000], &[1u8; 3000]);

        let manifest = read_manifest(&output).unwrap().expect("a bundle");
        assert_eq!(manifest.host.as_deref(), Some("0.0.0.0"));
        assert_eq!(manifest.port, Some(9100));
        assert_eq!(manifest.web, Some(9200));
    }

    /// A bundle built before these were recorded — or built without them —
    /// has no such keys, and must keep exactly the behaviour it had:
    /// loopback on the built-in ports.
    #[test]
    fn a_manifest_without_an_address_reads_as_the_built_in_defaults() {
        let manifest: Manifest = serde_json::from_str(
            r#"{"base_len":5000,"model":"user/model","quantization":null,"role":"all",
                "version":"1.1.0",
                "shards":[{"name":"m.gguf","offset":8192,"len":10}]}"#,
        )
        .expect("a manifest with no address keys");
        assert_eq!(manifest.host, None);
        assert_eq!(manifest.port, None);
        assert_eq!(manifest.web, None);

        let conf = crate::config::bundled_configuration(
            PathBuf::new(),
            Role::All,
            &crate::config::BundledListen::default(),
        );
        assert_eq!(conf.host, crate::config::BUNDLED_HOST);
        assert_eq!(conf.port, 8100);
        assert_eq!(conf.web, 8200);
    }

    /// ...and one that does record an address comes up on it, console
    /// included: `bundle --host all` means "expose this bundle", not "expose
    /// half of it".
    #[test]
    fn a_recorded_address_is_what_a_bundle_binds() {
        let conf = crate::config::bundled_configuration(
            PathBuf::new(),
            Role::All,
            &crate::config::BundledListen {
                host: Some("0.0.0.0".to_string()),
                port: Some(9100),
                web: Some(9200),
            },
        );
        assert_eq!(conf.host, "0.0.0.0");
        assert_eq!(conf.web_host, "0.0.0.0");
        assert_eq!(conf.port, 9100);
        assert_eq!(conf.web, 9200);
        // Still not "explicit": a run-time `--host` may move both, which is
        // the point of being able to override a bundle's baked-in address.
        assert!(!conf.web_host_explicit);
    }

    /// A `--host` that could never bind is caught at build time. The bundle
    /// runs on a machine whose operator may have no idea what the address was
    /// supposed to say, so "connection refused over there" is the wrong place
    /// to find out.
    #[test]
    fn an_impossible_host_is_rejected_while_there_is_somebody_to_tell() {
        for good in [None, Some("all"), Some("*"), Some("0.0.0.0"), Some("::1")] {
            assert!(validate_host(good).is_ok(), "{good:?}");
        }
        for bad in ["localhost", "example.com", "0.0.0", "everything"] {
            let err = validate_host(Some(bad)).unwrap_err();
            assert!(err.to_string().contains(bad), "{err:#}");
        }
    }

    /// The summary line spells out what the bundle will actually be reachable
    /// on — defaults included, since half an answer to "where will this
    /// listen" is no answer.
    #[test]
    fn the_summary_names_the_address_in_full() {
        use crate::config::BundledListen;

        assert_eq!(
            describe_listen(&BundledListen::default()),
            "API 127.0.0.1:8100, console 127.0.0.1:8200"
        );
        assert_eq!(
            describe_listen(&BundledListen {
                host: Some("all".to_string()),
                port: None,
                web: None,
            }),
            "API all:8100, console all:8200"
        );
        assert_eq!(
            describe_listen(&BundledListen {
                host: None,
                port: Some(9100),
                web: Some(0),
            }),
            "API 127.0.0.1:9100, console off"
        );
    }

    /// A footer naming a range that doesn't line up with the file's own
    /// length is rejected before anything is allocated for it.
    #[test]
    fn a_footer_that_does_not_add_up_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let output = bundle_into(dir.path(), &vec![0xAAu8; 5000], &vec![1u8; 3000]);

        let mut bytes = std::fs::read(&output).unwrap();
        let len = bytes.len();
        bytes[len - 24..len - 16].copy_from_slice(&999u64.to_le_bytes());
        std::fs::write(&output, &bytes).unwrap();

        assert!(read_manifest(&output).is_err());
    }
}
