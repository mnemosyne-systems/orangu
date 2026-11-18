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

//! A Hugging Face tokenizer — `tokenizer.json`, `tokenizer_config.json`
//! and `config.json` — as the `tokenizer.*` keys of a GGUF file.
//!
//! Two kinds of BPE: byte-level ([`Style::ByteLevel`], a GPT-2 style
//! alphabet of 256 printable stand-ins for bytes), and SentencePiece-style
//! with byte fallback ([`Style::ByteFallback`], `▁` for a space and
//! `<0xNN>` tokens for bytes no merge covers). Everything here is read from the files, never from
//! running the tokenizer, and that shapes two rules:
//!
//! - **The pre-tokenizer is recognised by its definition.** How text is cut
//!   into words before merging is what `tokenizer.ggml.pre` names, and the
//!   server's tokenizer picks its split from that name. The definition in
//!   `tokenizer.json` is matched against [`PRE_TOKENIZERS`]; one that is not
//!   there is an error, because a guessed name gives a model that loads,
//!   tokenizes every prompt slightly wrong, and answers worse for it.
//! - **Special tokens follow the files in a fixed order of authority**:
//!   `tokenizer.json`'s post-processor for whether BOS/EOS are added, then
//!   `tokenizer_config.json`'s `add_*_token` and `*_token` entries, then
//!   `config.json`'s `*_token_id`. A token set by an earlier source is not
//!   overridden by a later one. `add_bos_token` and `add_eos_token` are
//!   always written: with no source saying otherwise the tokenizer adds
//!   neither, and a reader treats a missing `add_bos_token` as `true`.

use anyhow::{Context, Result, anyhow, bail};
use orangu::gguf::GgufValue;
use serde_json::Value;
use std::{collections::HashMap, path::Path};

/// `tokenizer.ggml.token_type` values.
const NORMAL: i32 = 1;
const CONTROL: i32 = 3;
const USER_DEFINED: i32 = 4;
const UNUSED: i32 = 5;
const BYTE: i32 = 6;

/// Which kind of BPE vocabulary a checkpoint carries, and how the
/// architecture wants it written.
#[derive(Debug, Clone, Copy)]
pub enum Style<'a> {
    /// GPT-2 style byte-level BPE, written as `tokenizer.ggml.model =
    /// "gpt2"` with a recognised pre-tokenizer.
    ByteLevel,
    /// BPE over `▁`-marked words with `<0xNN>` byte tokens, written under
    /// the architecture's tokenizer `model` name with a score per token.
    /// `visible` names special tokens the reader must show in the output
    /// rather than hide — a chat format's structural markers that a parser
    /// downstream reads.
    ByteFallback {
        model: &'a str,
        visible: &'a [&'a str],
    },
    /// A SentencePiece model (`tokenizer.model`), written as
    /// `tokenizer.ggml.model = "llama"` with each piece's own score and
    /// kind: the reader segments by score, so no merges are written.
    SentencePiece,
}

/// One piece of a SentencePiece model.
struct Piece {
    text: String,
    score: f32,
    /// The model's own kind, numbered as `tokenizer.ggml.token_type` is.
    kind: i32,
}

/// A protobuf varint at `at`, advancing past it.
fn varint(bytes: &[u8], at: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let b = *bytes
            .get(*at)
            .ok_or_else(|| anyhow!("tokenizer.model ends inside a number"))?;
        *at += 1;
        value |= u64::from(b & 0x7f) << shift;
        if b & 0x80 == 0 {
            return Ok(value);
        }
    }
    bail!("tokenizer.model has a number longer than 64 bits")
}

/// One protobuf field: its number, its wire type, the value of a varint or
/// fixed field, and the bytes of a length-delimited one.
type Field<'a> = (u64, u64, u64, &'a [u8]);

/// The fields of one protobuf message: (field number, wire type, the
/// value for a varint or fixed field, the bytes for a length-delimited one).
fn fields(bytes: &[u8]) -> Result<Vec<Field<'_>>> {
    let mut out = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let key = varint(bytes, &mut at)?;
        let (field, wire) = (key >> 3, key & 7);
        let take = |at: &mut usize, n: usize| -> Result<&[u8]> {
            let slice = bytes
                .get(*at..*at + n)
                .ok_or_else(|| anyhow!("tokenizer.model ends inside a field"))?;
            *at += n;
            Ok(slice)
        };
        match wire {
            0 => out.push((field, wire, varint(bytes, &mut at)?, &[][..])),
            1 => {
                let b = take(&mut at, 8)?;
                out.push((field, wire, u64::from_le_bytes(b.try_into()?), &[][..]));
            }
            2 => {
                let n = varint(bytes, &mut at)? as usize;
                out.push((field, wire, 0, take(&mut at, n)?));
            }
            5 => {
                let b = take(&mut at, 4)?;
                out.push((
                    field,
                    wire,
                    u64::from(u32::from_le_bytes(b.try_into()?)),
                    &[][..],
                ));
            }
            other => bail!("tokenizer.model has a field of wire type {other}"),
        }
    }
    Ok(out)
}

/// The pieces of a SentencePiece model file, in id order: field 1 of the
/// model, each with its text (1), score (2) and kind (3, normal when
/// absent).
fn sentencepiece(path: &Path) -> Result<Vec<Piece>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let mut pieces = Vec::new();
    for (field, wire, _, body) in fields(&bytes)? {
        if field != 1 || wire != 2 {
            continue;
        }
        let mut piece = Piece {
            text: String::new(),
            score: 0.0,
            kind: NORMAL,
        };
        for (f, w, value, b) in fields(body)? {
            match (f, w) {
                (1, 2) => piece.text = String::from_utf8(b.to_vec())?,
                (2, 5) => piece.score = f32::from_bits(value as u32),
                (3, 0) => piece.kind = value as i32,
                _ => {}
            }
        }
        pieces.push(piece);
    }
    if pieces.is_empty() {
        bail!("{} has no pieces", path.display());
    }
    Ok(pieces)
}

/// A SentencePiece vocabulary as the file carries it: `tokenizer.model`'s
/// pieces padded to `vocab_size`, then the tokens `added_tokens.json` and
/// `tokenizer_config.json` add over them — a special-looking or special one
/// as a control token, any other as user-defined text with its `▁` read as
/// a space. Tokens, scores and kinds.
fn sentencepiece_vocab(
    dir: &Path,
    tokenizer_config: &Value,
    vocab_size: usize,
) -> Result<(Vec<String>, Vec<f32>, Vec<i32>)> {
    let pieces = sentencepiece(&dir.join("tokenizer.model"))?;
    let mut tokens: Vec<String> = (0..vocab_size).map(|i| format!("[PAD{i}]")).collect();
    let mut scores = vec![-10000.0f32; vocab_size];
    let mut types = vec![UNUSED; vocab_size];
    for (id, piece) in pieces.into_iter().enumerate().take(vocab_size) {
        tokens[id] = piece.text;
        scores[id] = piece.score;
        types[id] = piece.kind;
    }
    if let Some(added) = read_json(&dir.join("added_tokens.json"))? {
        for (text, id) in added.as_object().into_iter().flatten() {
            let Some(id) = id.as_u64().map(|i| i as usize).filter(|&i| i < vocab_size) else {
                continue;
            };
            tokens[id] = text.clone();
            scores[id] = -1000.0;
            types[id] = USER_DEFINED;
        }
    }
    for (id, entry) in tokenizer_config["added_tokens_decoder"]
        .as_object()
        .into_iter()
        .flatten()
    {
        let (Ok(id), Some(text)) = (id.parse::<usize>(), entry["content"].as_str()) else {
            continue;
        };
        if id >= vocab_size {
            continue;
        }
        if entry["special"].as_bool().unwrap_or(false) || looks_special(text) {
            tokens[id] = text.to_string();
            types[id] = CONTROL;
        } else {
            tokens[id] = text.replace('\u{2581}', " ");
            types[id] = USER_DEFINED;
        }
        scores[id] = -1000.0;
    }
    Ok((tokens, scores, types))
}

/// Whether `text` is a byte-fallback token, `<0x00>` through `<0xFF>`.
fn is_byte_token(text: &str) -> bool {
    let b = text.as_bytes();
    b.len() == 6
        && text.starts_with("<0x")
        && b[5] == b'>'
        && b[3].is_ascii_hexdigit()
        && b[4].is_ascii_hexdigit()
}

/// The special-token kinds read from the files, in the order they are
/// looked up, with the key each is written under.
const SPECIAL: [(&str, &str); 6] = [
    ("bos", "tokenizer.ggml.bos_token_id"),
    ("eos", "tokenizer.ggml.eos_token_id"),
    ("unk", "tokenizer.ggml.unknown_token_id"),
    ("sep", "tokenizer.ggml.seperator_token_id"),
    ("pad", "tokenizer.ggml.padding_token_id"),
    ("mask", "tokenizer.ggml.mask_token_id"),
];

/// A pre-tokenizer definition, reduced to what decides the split.
#[derive(Debug, PartialEq, Eq)]
enum Step {
    /// `Split` on a regular expression.
    Split(String),
    /// `Digits`, one per token or runs.
    Digits(bool),
    /// `ByteLevel`, with its own GPT-2 regex or without.
    ByteLevel(bool),
}

/// Known pre-tokenizers: their steps, and the `tokenizer.ggml.pre` name
/// whose split the server applies for them.
const PRE_TOKENIZERS: &[(&str, &[StepSpec])] = &[
    ("gpt-2", &[StepSpec::ByteLevel(true)]),
    (
        "smollm",
        &[StepSpec::Digits(true), StepSpec::ByteLevel(true)],
    ),
    (
        "llama-bpe",
        &[StepSpec::Split(LLAMA3_REGEX), StepSpec::ByteLevel(false)],
    ),
    (
        "qwen2",
        &[StepSpec::Split(QWEN2_REGEX), StepSpec::ByteLevel(false)],
    ),
    (
        "qwen35",
        &[StepSpec::Split(QWEN35_REGEX), StepSpec::ByteLevel(false)],
    ),
    (
        "gpt-4o",
        &[StepSpec::Split(GPT4O_REGEX), StepSpec::ByteLevel(false)],
    ),
    (
        "tekken",
        &[StepSpec::Split(TEKKEN_REGEX), StepSpec::ByteLevel(false)],
    ),
    (
        "bailingmoe2",
        &[StepSpec::Split(BAILING_REGEX), StepSpec::ByteLevel(false)],
    ),
];

const GPT4O_REGEX: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const BAILING_REGEX: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";
const TEKKEN_REGEX: &str = r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const LLAMA3_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN2_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const QWEN35_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// [`Step`] in a form a `const` table can hold.
#[derive(Debug, Clone, Copy)]
enum StepSpec {
    Split(&'static str),
    Digits(bool),
    ByteLevel(bool),
}

impl StepSpec {
    fn owned(self) -> Step {
        match self {
            StepSpec::Split(r) => Step::Split(r.to_string()),
            StepSpec::Digits(d) => Step::Digits(d),
            StepSpec::ByteLevel(b) => Step::ByteLevel(b),
        }
    }
}

/// The steps of a `pre_tokenizer` definition, flattened out of any
/// `Sequence`.
fn steps(definition: &Value, out: &mut Vec<Step>) -> Result<()> {
    match definition["type"].as_str().unwrap_or("") {
        "Sequence" => {
            for inner in definition["pretokenizers"].as_array().into_iter().flatten() {
                steps(inner, out)?;
            }
        }
        "Split" => {
            let regex = definition["pattern"]["Regex"]
                .as_str()
                .or_else(|| definition["pattern"]["String"].as_str())
                .ok_or_else(|| anyhow!("a Split pre-tokenizer with no pattern"))?;
            out.push(Step::Split(regex.to_string()));
        }
        "Digits" => out.push(Step::Digits(
            definition["individual_digits"].as_bool().unwrap_or(false),
        )),
        "ByteLevel" => out.push(Step::ByteLevel(
            definition["use_regex"].as_bool().unwrap_or(true),
        )),
        other => bail!("the {other:?} pre-tokenizer is not one this converter recognises"),
    }
    Ok(())
}

/// The `tokenizer.ggml.pre` name for a `pre_tokenizer` definition.
fn pre_name(definition: &Value) -> Result<&'static str> {
    let mut found = Vec::new();
    steps(definition, &mut found)?;
    PRE_TOKENIZERS
        .iter()
        .find(|(_, spec)| {
            spec.len() == found.len() && spec.iter().zip(&found).all(|(s, f)| s.owned() == *f)
        })
        .map(|(name, _)| *name)
        .ok_or_else(|| {
            anyhow!(
                "this tokenizer's pre-tokenizer ({found:?}) is not one this converter recognises"
            )
        })
}

/// Whether an added token that is not marked special still reads as a
/// control token: `<|...|>`-shaped ones and a few known names.
fn looks_special(token: &str) -> bool {
    matches!(token, "<pad>" | "<mask>" | "<2mass>" | "[@BOS@]")
        || (token.starts_with("<|") && token.ends_with("|>"))
        || (token.starts_with("<｜") && token.ends_with("｜>"))
        || (token.starts_with("<unused") && token.ends_with('>'))
}

fn read_json(path: &Path) -> Result<Option<Value>> {
    if !path.is_file() {
        return Ok(None);
    }
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(Some(
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?,
    ))
}

/// The text of a `*_token` entry: a string, or an object with `content`.
fn token_text(entry: &Value) -> Option<&str> {
    entry.as_str().or_else(|| entry["content"].as_str())
}

/// The `tokenizer.*` keys for the checkpoint in `dir`.
///
/// `vocab_size` is the model's: a vocabulary shorter than the embedding is
/// padded with unused `[PAD<n>]` entries so every row has a token.
pub fn metadata(
    dir: &Path,
    config: &Value,
    vocab_size: usize,
    style: Style<'_>,
) -> Result<Vec<(String, GgufValue)>> {
    let tokenizer = read_json(&dir.join("tokenizer.json"))?
        .ok_or_else(|| anyhow!("{} has no tokenizer.json", dir.display()))?;
    let tokenizer_config = read_json(&dir.join("tokenizer_config.json"))?.unwrap_or(Value::Null);
    let model = &tokenizer["model"];
    let spm = matches!(style, Style::SentencePiece);
    if !spm && model["type"].as_str() != Some("BPE") {
        bail!(
            "the tokenizer is {:?}; only BPE converts so far",
            model["type"]
        );
    }
    let byte_fallback = model["byte_fallback"].as_bool().unwrap_or(false);
    match style {
        Style::ByteLevel if byte_fallback => {
            bail!("this tokenizer falls back to byte tokens, which this architecture does not read")
        }
        Style::ByteFallback { .. } if !byte_fallback => {
            bail!("this architecture's tokenizer falls back to byte tokens, and this one does not")
        }
        _ => {}
    }

    // Tokens by id.
    let mut by_id: HashMap<u64, String> = HashMap::new();
    for (text, id) in model["vocab"]
        .as_object()
        .ok_or_else(|| anyhow!("tokenizer.json has no vocab"))?
    {
        let id = id
            .as_u64()
            .ok_or_else(|| anyhow!("token {text:?} has no id"))?;
        by_id.insert(id, text.clone());
    }
    struct Added {
        special: bool,
    }
    let mut added: HashMap<u64, Added> = HashMap::new();
    let added_tokens = tokenizer["added_tokens"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    for entry in &added_tokens {
        let (Some(id), Some(content)) = (entry["id"].as_u64(), entry["content"].as_str()) else {
            continue;
        };
        by_id.insert(id, content.to_string());
        added.insert(
            id,
            Added {
                special: entry["special"].as_bool().unwrap_or(false),
            },
        );
    }
    // Tokens `tokenizer_config.json` adds that `tokenizer.json` lacks.
    for (id, entry) in tokenizer_config["added_tokens_decoder"]
        .as_object()
        .into_iter()
        .flatten()
    {
        let (Ok(id), Some(content)) = (id.parse::<u64>(), entry["content"].as_str()) else {
            continue;
        };
        if by_id.contains_key(&id) {
            continue;
        }
        by_id.insert(id, content.to_string());
        added.insert(
            id,
            Added {
                special: entry["special"].as_bool().unwrap_or(false),
            },
        );
    }
    // A SentencePiece vocabulary leaves out a token past its size, as its
    // own tooling does.
    if !spm
        && let Some(&max) = by_id.keys().max()
        && max as usize >= vocab_size
    {
        bail!("token id {max} is past the model's vocabulary of {vocab_size}");
    }
    let in_vocab: std::collections::HashSet<u64> = model["vocab"]
        .as_object()
        .into_iter()
        .flatten()
        .filter_map(|(_, id)| id.as_u64())
        .collect();
    let mut tokens = Vec::with_capacity(vocab_size);
    let mut types = Vec::with_capacity(vocab_size);
    for id in 0..vocab_size as u64 {
        if let Style::ByteFallback { visible, .. } = style {
            let Some(text) = by_id.get(&id) else {
                tokens.push(format!("[PAD{id}]"));
                types.push(UNUSED);
                continue;
            };
            let special = added.get(&id).is_some_and(|a| a.special);
            types.push(if visible.contains(&text.as_str()) {
                USER_DEFINED
            } else if is_byte_token(text) {
                BYTE
            } else if special {
                CONTROL
            } else if in_vocab.contains(&id) {
                NORMAL
            } else {
                USER_DEFINED
            });
            tokens.push(text.clone());
            continue;
        }
        match by_id.get(&id) {
            None => {
                tokens.push(format!("[PAD{id}]"));
                types.push(UNUSED);
            }
            Some(text) => match added.get(&id) {
                Some(a) if a.special || looks_special(text) => {
                    tokens.push(text.clone());
                    types.push(CONTROL);
                }
                Some(_) => {
                    tokens.push(text.replace('\u{2581}', " "));
                    types.push(USER_DEFINED);
                }
                None => {
                    tokens.push(text.clone());
                    types.push(NORMAL);
                }
            },
        }
    }

    // A SentencePiece model's own pieces, scores and kinds replace what
    // `tokenizer.json` says.
    let scores: Option<Vec<f32>> = if spm {
        let (t, s, k) = sentencepiece_vocab(dir, &tokenizer_config, vocab_size)?;
        tokens = t;
        types = k;
        Some(s)
    } else {
        None
    };

    // Merges, in either of the two spellings `tokenizer.json` uses. A pair
    // whose halves contain a space writes it as U+0120, the byte-level
    // alphabet's space, so the pair still splits on its one real space.
    let merges: Vec<String> = match model["merges"].as_array().filter(|_| !spm) {
        Some(list) => list
            .iter()
            .map(|m| match m {
                Value::String(s) => Ok(s.clone()),
                Value::Array(pair) if pair.len() == 2 => {
                    let half = |v: &Value| -> Result<String> {
                        Ok(v.as_str()
                            .ok_or_else(|| anyhow!("a merge half is not a string"))?
                            .replace(' ', "\u{0120}"))
                    };
                    Ok(format!("{} {}", half(&pair[0])?, half(&pair[1])?))
                }
                _ => bail!("tokenizer.json has a merge in an unknown form"),
            })
            .collect::<Result<_>>()?,
        None => Vec::new(),
    };

    let pre = match style {
        Style::ByteLevel => Some(pre_name(&tokenizer["pre_tokenizer"])?),
        Style::ByteFallback { .. } => None,
        Style::SentencePiece => Some("default"),
    };

    // Special tokens and whether they are added.
    let mut ids: HashMap<&str, u32> = HashMap::new();
    let mut add: Vec<(&str, bool)> = Vec::new();
    let set_add =
        |add: &mut Vec<(&str, bool)>, kind: &'static str, value: bool, force: bool| match add
            .iter_mut()
            .find(|(k, _)| *k == kind)
        {
            Some(entry) if force => entry.1 = value,
            Some(_) => {}
            None => add.push((kind, value)),
        };
    let content_id = |text: &str| -> Option<u32> {
        added_tokens
            .iter()
            .find(|a| a["content"].as_str() == Some(text))
            .and_then(|a| a["id"].as_u64())
            .map(|id| id as u32)
    };
    let mut config_tokens: HashMap<&str, String> = HashMap::new();
    for (kind, _) in SPECIAL.iter().chain([("cls", "")].iter()) {
        if let Some(text) = token_text(&tokenizer_config[format!("{kind}_token")]) {
            config_tokens.insert(kind, text.to_string());
        }
    }
    // A BERT-shaped config names CLS and SEP where BOS and EOS would be.
    if !config_tokens.contains_key("bos")
        && let Some(cls) = config_tokens.get("cls").cloned()
    {
        config_tokens.insert("bos", cls);
    }
    if !config_tokens.contains_key("eos")
        && let Some(sep) = config_tokens.get("sep").cloned()
    {
        config_tokens.insert("eos", sep);
    }
    let post = &tokenizer["post_processor"];
    let processors: Vec<&Value> = match post["processors"].as_array() {
        Some(list) => list.iter().collect(),
        None if post.is_object() => vec![post],
        None => vec![],
    };
    for processor in processors {
        match processor["type"].as_str().unwrap_or("") {
            "ByteLevel" => {
                set_add(&mut add, "bos", false, false);
                set_add(&mut add, "eos", false, false);
            }
            "TemplateProcessing" => {
                let single = processor["single"].as_array().cloned().unwrap_or_default();
                if single.len() > 1 {
                    if let Some(first) = single[0]["SpecialToken"]["id"].as_str() {
                        if !config_tokens.contains_key("bos") {
                            config_tokens.insert("bos", first.to_string());
                        }
                        let adds = config_tokens.get("bos").map(String::as_str) == Some(first)
                            || config_tokens.get("cls").map(String::as_str) == Some(first);
                        set_add(&mut add, "bos", adds, true);
                    }
                    if let Some(last) = single[single.len() - 1]["SpecialToken"]["id"].as_str() {
                        if !config_tokens.contains_key("eos") {
                            config_tokens.insert("eos", last.to_string());
                        }
                        let adds = config_tokens.get("eos").map(String::as_str) == Some(last);
                        set_add(&mut add, "eos", adds, true);
                    }
                }
            }
            _ => {}
        }
    }
    for (kind, _) in SPECIAL {
        if let Some(value) = tokenizer_config[format!("add_{kind}_token")].as_bool() {
            set_add(&mut add, kind, value, false);
        }
        if let Some(id) = config_tokens.get(kind).and_then(|t| content_id(t))
            && (id as usize) < vocab_size
        {
            ids.entry(kind).or_insert(id);
        }
    }
    // What the tokenizer does when nothing above says it adds a token is
    // add nothing, and the file says so: a reader that finds no
    // `add_bos_token` key assumes BOS is wanted.
    set_add(&mut add, "bos", false, false);
    set_add(&mut add, "eos", false, false);
    for (kind, _) in SPECIAL {
        let from_config = config[format!("{kind}_token_id")]
            .as_u64()
            .or_else(|| config["text_config"][format!("{kind}_token_id")].as_u64());
        if let Some(id) = from_config
            && (id as usize) < vocab_size
        {
            ids.entry(kind).or_insert(id as u32);
        }
    }

    // A turn that ends on a token other than the tokenizer's EOS says so
    // only in `generation_config.json`, whose first stop token is the one
    // the model writes. Without it the server generates past the end of the
    // turn.
    let mut eot = token_text(&tokenizer_config["eot_token"]).and_then(content_id);
    if eot.is_none()
        && let Some(generation) = read_json(&dir.join("generation_config.json"))?
    {
        let first = match &generation["eos_token_id"] {
            Value::Array(ids) => ids.first().and_then(Value::as_u64),
            other => other.as_u64(),
        };
        if let Some(first) = first
            && (first as usize) < vocab_size
            && ids.get("eos").is_some_and(|&eos| eos as u64 != first)
        {
            eot = Some(first as u32);
        }
    }

    let model_name = match style {
        Style::ByteLevel => "gpt2",
        Style::ByteFallback { model, .. } => model,
        Style::SentencePiece => "llama",
    };
    let count = tokens.len();
    let mut out: Vec<(String, GgufValue)> = vec![(
        "tokenizer.ggml.model".into(),
        GgufValue::String(model_name.into()),
    )];
    if let Some(pre) = pre {
        out.push(("tokenizer.ggml.pre".into(), GgufValue::String(pre.into())));
    }
    out.push((
        "tokenizer.ggml.tokens".into(),
        GgufValue::Array(tokens.into_iter().map(GgufValue::String).collect()),
    ));
    if let Some(scores) = scores {
        out.push((
            "tokenizer.ggml.scores".into(),
            GgufValue::Array(scores.into_iter().map(GgufValue::F32).collect()),
        ));
    } else if let Style::ByteFallback { .. } = style {
        // A BPE vocabulary ranks by its merges, not by scores; every token
        // carries the same one.
        out.push((
            "tokenizer.ggml.scores".into(),
            GgufValue::Array(vec![GgufValue::F32(-1000.0); count]),
        ));
    }
    out.push((
        "tokenizer.ggml.token_type".into(),
        GgufValue::Array(types.into_iter().map(GgufValue::I32).collect()),
    ));
    if !merges.is_empty() {
        out.push((
            "tokenizer.ggml.merges".into(),
            GgufValue::Array(merges.into_iter().map(GgufValue::String).collect()),
        ));
    }
    for (kind, key) in SPECIAL {
        if let Some(&id) = ids.get(kind) {
            out.push((key.into(), GgufValue::U32(id)));
        }
    }
    if let Some(eot) = eot {
        out.push(("tokenizer.ggml.eot_token_id".into(), GgufValue::U32(eot)));
    }
    for (kind, value) in add {
        out.push((
            format!("tokenizer.ggml.add_{kind}_token"),
            GgufValue::Bool(value),
        ));
    }
    if let Some(prefix) = tokenizer_config["add_prefix_space"].as_bool() {
        out.push((
            "tokenizer.ggml.add_space_prefix".into(),
            GgufValue::Bool(prefix),
        ));
    }
    if let Some(clean) = tokenizer_config["clean_up_tokenization_spaces"].as_bool() {
        out.push(("tokenizer.ggml.clean_spaces".into(), GgufValue::Bool(clean)));
    }
    if let Some(template) = chat_template(dir, &tokenizer_config)? {
        out.push((
            "tokenizer.chat_template".into(),
            GgufValue::String(template),
        ));
    }
    Ok(out)
}

/// The chat template: `chat_template.jinja` beside the checkpoint, then
/// `chat_template.json`, then `tokenizer_config.json`'s own entry.
fn chat_template(dir: &Path, tokenizer_config: &Value) -> Result<Option<String>> {
    let jinja = dir.join("chat_template.jinja");
    if jinja.is_file() {
        return Ok(Some(
            std::fs::read_to_string(&jinja)
                .with_context(|| format!("reading {}", jinja.display()))?,
        ));
    }
    if let Some(json) = read_json(&dir.join("chat_template.json"))?
        && let Some(t) = json["chat_template"].as_str()
    {
        return Ok(Some(t.to_string()));
    }
    Ok(tokenizer_config["chat_template"]
        .as_str()
        .map(str::to_string))
}

/// Overwrites (or adds) one key in a metadata list: an architecture's
/// reader that wants a tokenizer setting the checkpoint does not state.
pub fn set(metadata: &mut Vec<(String, GgufValue)>, key: &str, value: GgufValue) {
    match metadata.iter_mut().find(|(k, _)| k == key) {
        Some(entry) => entry.1 = value,
        None => metadata.push((key.to_string(), value)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One SentencePiece piece as protobuf: text (1), score (2), kind (3).
    fn piece(text: &str, score: f32, kind: Option<u8>) -> Vec<u8> {
        let mut body = vec![0x0a, text.len() as u8];
        body.extend(text.as_bytes());
        body.push(0x15);
        body.extend(score.to_le_bytes());
        if let Some(k) = kind {
            body.extend([0x18, k]);
        }
        let mut out = vec![0x0a, body.len() as u8];
        out.extend(body);
        out
    }

    #[test]
    fn a_sentencepiece_model_is_read_piece_by_piece() {
        let dir = tempfile::tempdir().unwrap();
        let mut model = piece("<unk>", 0.0, Some(2));
        model.extend(piece("\u{2581}the", -3.5, None));
        model.extend(piece("<0x41>", 0.0, Some(6)));
        // A field the reader skips: the trainer spec.
        model.extend([0x12, 0x02, 0x08, 0x01]);
        std::fs::write(dir.path().join("tokenizer.model"), &model).unwrap();
        let config = serde_json::json!({"added_tokens_decoder": {
            "3": {"content": "<start_of_turn>", "special": true},
            "4": {"content": "a\u{2581}b", "special": false}
        }});
        let (tokens, scores, types) = sentencepiece_vocab(dir.path(), &config, 6).unwrap();
        assert_eq!(
            tokens,
            [
                "<unk>",
                "\u{2581}the",
                "<0x41>",
                "<start_of_turn>",
                "a b",
                "[PAD5]"
            ]
        );
        assert_eq!(scores, [0.0, -3.5, 0.0, -1000.0, -1000.0, -10000.0]);
        assert_eq!(types, [2, NORMAL, BYTE, CONTROL, USER_DEFINED, UNUSED]);
    }
    use serde_json::json;

    fn checkpoint(dir: &Path, tokenizer: Value, tokenizer_config: Value) {
        std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
        std::fs::write(
            dir.join("tokenizer_config.json"),
            tokenizer_config.to_string(),
        )
        .unwrap();
    }

    fn get<'a>(meta: &'a [(String, GgufValue)], key: &str) -> Option<&'a GgufValue> {
        meta.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    fn tiny_tokenizer(pre: Value, merges: Value) -> Value {
        json!({
            "added_tokens": [
                {"id": 3, "content": "<|im_start|>", "special": true, "normalized": false},
                {"id": 4, "content": "<|plain|>", "special": false, "normalized": false},
                {"id": 5, "content": "my\u{2581}word", "special": false, "normalized": false}
            ],
            "pre_tokenizer": pre,
            "post_processor": {"type": "ByteLevel"},
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "ab": 2}, "merges": merges}
        })
    }

    #[test]
    fn tokens_types_padding_and_both_merge_spellings() {
        let dir = tempfile::tempdir().unwrap();
        checkpoint(
            dir.path(),
            tiny_tokenizer(
                json!({"type": "ByteLevel", "use_regex": true}),
                json!([["a", "b"]]),
            ),
            json!({"bos_token": "<|im_start|>", "chat_template": "{{ x }}"}),
        );
        let meta = metadata(dir.path(), &json!({"eos_token_id": 1}), 8, Style::ByteLevel).unwrap();
        let tokens = match get(&meta, "tokenizer.ggml.tokens") {
            Some(GgufValue::Array(t)) => t.clone(),
            other => panic!("{other:?}"),
        };
        assert_eq!(tokens.len(), 8);
        assert!(matches!(&tokens[5], GgufValue::String(s) if s == "my word"));
        assert!(matches!(&tokens[7], GgufValue::String(s) if s == "[PAD7]"));
        let types: Vec<i32> = match get(&meta, "tokenizer.ggml.token_type") {
            Some(GgufValue::Array(t)) => t
                .iter()
                .map(|v| match v {
                    GgufValue::I32(i) => *i,
                    _ => panic!(),
                })
                .collect(),
            other => panic!("{other:?}"),
        };
        // `<|plain|>` is not marked special but reads as a control token.
        assert_eq!(types, vec![1, 1, 1, 3, 3, 4, 5, 5]);
        assert!(
            matches!(get(&meta, "tokenizer.ggml.pre"), Some(GgufValue::String(s)) if s == "gpt-2")
        );
        assert!(matches!(
            get(&meta, "tokenizer.ggml.bos_token_id"),
            Some(GgufValue::U32(3))
        ));
        assert!(matches!(
            get(&meta, "tokenizer.ggml.eos_token_id"),
            Some(GgufValue::U32(1))
        ));
        assert!(matches!(
            get(&meta, "tokenizer.ggml.add_bos_token"),
            Some(GgufValue::Bool(false))
        ));
        assert!(
            matches!(get(&meta, "tokenizer.chat_template"), Some(GgufValue::String(s)) if s == "{{ x }}")
        );
        match get(&meta, "tokenizer.ggml.merges") {
            Some(GgufValue::Array(m)) => {
                assert!(matches!(&m[0], GgufValue::String(s) if s == "a b"))
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_merge_half_with_a_space_is_written_in_the_byte_alphabet() {
        let dir = tempfile::tempdir().unwrap();
        checkpoint(
            dir.path(),
            tiny_tokenizer(json!({"type": "ByteLevel"}), json!([["a b", "c"]])),
            json!({}),
        );
        let meta = metadata(dir.path(), &json!({}), 6, Style::ByteLevel).unwrap();
        match get(&meta, "tokenizer.ggml.merges") {
            Some(GgufValue::Array(m)) => {
                assert!(matches!(&m[0], GgufValue::String(s) if s == "a\u{0120}b c"))
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn pre_tokenizers_are_recognised_and_an_unknown_one_is_refused() {
        let smollm = json!({"type": "Sequence", "pretokenizers": [
            {"type": "Digits", "individual_digits": true},
            {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}
        ]});
        assert_eq!(pre_name(&smollm).unwrap(), "smollm");
        let llama3 = json!({"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": LLAMA3_REGEX}, "behavior": "Isolated"},
            {"type": "ByteLevel", "use_regex": false}
        ]});
        assert_eq!(pre_name(&llama3).unwrap(), "llama-bpe");
        let other = json!({"type": "Split", "pattern": {"Regex": "x+"}});
        assert!(pre_name(&other).is_err());
    }

    #[test]
    fn a_template_post_processor_decides_whether_bos_is_added() {
        let dir = tempfile::tempdir().unwrap();
        let mut tokenizer = tiny_tokenizer(json!({"type": "ByteLevel"}), json!(["a b"]));
        tokenizer["post_processor"] = json!({"type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<|im_start|>", "type_id": 0}},
                       {"Sequence": {"id": "A", "type_id": 0}}]});
        checkpoint(dir.path(), tokenizer, json!({"add_bos_token": false}));
        // ... and nothing said about EOS writes `false`, not nothing.
        let meta = metadata(dir.path(), &json!({}), 6, Style::ByteLevel).unwrap();
        // The post-processor outranks the config's flag.
        assert!(matches!(
            get(&meta, "tokenizer.ggml.add_bos_token"),
            Some(GgufValue::Bool(true))
        ));
        assert!(matches!(
            get(&meta, "tokenizer.ggml.bos_token_id"),
            Some(GgufValue::U32(3))
        ));
        assert!(matches!(
            get(&meta, "tokenizer.ggml.add_eos_token"),
            Some(GgufValue::Bool(false))
        ));
    }

    #[test]
    fn tokens_only_the_tokenizer_config_adds_fill_their_ids() {
        let dir = tempfile::tempdir().unwrap();
        checkpoint(
            dir.path(),
            tiny_tokenizer(json!({"type": "ByteLevel"}), json!(["a b"])),
            json!({"added_tokens_decoder": {
                "3": {"content": "<|other|>", "special": true},
                "6": {"content": "<|audio_start|>", "special": true}
            }}),
        );
        let meta = metadata(dir.path(), &json!({}), 8, Style::ByteLevel).unwrap();
        let Some(GgufValue::Array(tokens)) = get(&meta, "tokenizer.ggml.tokens") else {
            panic!()
        };
        // `tokenizer.json` wins where both name an id.
        assert!(matches!(&tokens[3], GgufValue::String(s) if s == "<|im_start|>"));
        assert!(matches!(&tokens[6], GgufValue::String(s) if s == "<|audio_start|>"));
        assert!(matches!(&tokens[7], GgufValue::String(s) if s == "[PAD7]"));
        let Some(GgufValue::Array(types)) = get(&meta, "tokenizer.ggml.token_type") else {
            panic!()
        };
        assert!(matches!(types[6], GgufValue::I32(3)));
    }

    #[test]
    fn a_first_stop_token_other_than_eos_is_the_end_of_turn() {
        let dir = tempfile::tempdir().unwrap();
        checkpoint(
            dir.path(),
            tiny_tokenizer(json!({"type": "ByteLevel"}), json!(["a b"])),
            json!({"eos_token": "<|im_start|>"}),
        );
        std::fs::write(
            dir.path().join("generation_config.json"),
            r#"{"eos_token_id": [4, 3]}"#,
        )
        .unwrap();
        let meta = metadata(dir.path(), &json!({}), 6, Style::ByteLevel).unwrap();
        assert!(matches!(
            get(&meta, "tokenizer.ggml.eot_token_id"),
            Some(GgufValue::U32(4))
        ));
        // The same token first, as most checkpoints have it, adds nothing.
        std::fs::write(
            dir.path().join("generation_config.json"),
            r#"{"eos_token_id": [3, 4]}"#,
        )
        .unwrap();
        let meta = metadata(dir.path(), &json!({}), 6, Style::ByteLevel).unwrap();
        assert!(get(&meta, "tokenizer.ggml.eot_token_id").is_none());
    }

    #[test]
    fn a_byte_fallback_vocabulary_has_byte_tokens_scores_and_visible_markers() {
        let dir = tempfile::tempdir().unwrap();
        let tokenizer = json!({
            "added_tokens": [
                {"id": 0, "content": "<pad>", "special": true},
                {"id": 3, "content": "<|turn>", "special": true},
                {"id": 4, "content": "<|channel>", "special": true}
            ],
            "pre_tokenizer": {"type": "Split", "pattern": {"String": " "}},
            "model": {"type": "BPE", "byte_fallback": true,
                      "vocab": {"<pad>": 0, "<0x0A>": 1, "\u{2581}a": 2, "<|turn>": 3, "<|channel>": 4},
                      "merges": [["\u{2581}", "a"]]}
        });
        checkpoint(dir.path(), tokenizer, json!({"eot_token": "<|turn>"}));
        let style = Style::ByteFallback {
            model: "gemma4",
            visible: &["<|channel>"],
        };
        let meta = metadata(dir.path(), &json!({}), 5, style).unwrap();
        assert!(
            matches!(get(&meta, "tokenizer.ggml.model"), Some(GgufValue::String(s)) if s == "gemma4")
        );
        assert!(get(&meta, "tokenizer.ggml.pre").is_none());
        let types: Vec<i32> = match get(&meta, "tokenizer.ggml.token_type") {
            Some(GgufValue::Array(t)) => t
                .iter()
                .map(|v| match v {
                    GgufValue::I32(i) => *i,
                    _ => panic!(),
                })
                .collect(),
            other => panic!("{other:?}"),
        };
        assert_eq!(types, vec![3, 6, 1, 3, 4]);
        assert!(
            matches!(get(&meta, "tokenizer.ggml.scores"), Some(GgufValue::Array(s)) if s.len() == 5)
        );
        assert!(matches!(
            get(&meta, "tokenizer.ggml.eot_token_id"),
            Some(GgufValue::U32(3))
        ));
        // A byte-level architecture refuses it.
        assert!(metadata(dir.path(), &json!({}), 5, Style::ByteLevel).is_err());
    }

    #[test]
    fn with_no_post_processor_neither_flag_is_left_out() {
        let dir = tempfile::tempdir().unwrap();
        let mut tokenizer = tiny_tokenizer(json!({"type": "ByteLevel"}), json!(["a b"]));
        tokenizer["post_processor"] = Value::Null;
        checkpoint(dir.path(), tokenizer, json!({}));
        let meta = metadata(dir.path(), &json!({}), 6, Style::ByteLevel).unwrap();
        assert!(matches!(
            get(&meta, "tokenizer.ggml.add_bos_token"),
            Some(GgufValue::Bool(false))
        ));
        assert!(matches!(
            get(&meta, "tokenizer.ggml.add_eos_token"),
            Some(GgufValue::Bool(false))
        ));
    }
}
