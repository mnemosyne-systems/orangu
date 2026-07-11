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

//! A from-scratch BPE tokenizer covering two real vocab shapes, dispatched
//! on `tokenizer.ggml.model`:
//!
//! - **`"gpt2"`** (every Llama3/Qwen2/Qwen3/Mistral/qwen35moe GGUF): the
//!   classic byte-level scheme — every byte maps to a printable-unicode
//!   alphabet (`byte_to_unicode_table`) before merges run, so a leading
//!   space rides along inside a token like `"Ġcapital"`.
//! - **`"gemma4"`** (confirmed against real upstream `llama.cpp` source,
//!   `src/llama-vocab.cpp`'s `tokenizer_pre == "gemma4"` branch, fetched
//!   and read directly): merges still come from `tokenizer.ggml.merges`
//!   (this is genuinely still BPE, not SentencePiece-unigram, despite the
//!   model-name-shaped `tokenizer.ggml.model` value), but every literal
//!   space in the input is escaped to `▁` (U+2581) *before* merging, merges
//!   run on raw UTF-8 codepoints rather than the byte-to-unicode alphabet,
//!   and the pre-tokenizer only splits on newlines. Getting this wrong
//!   doesn't just produce different-but-valid tokens: decoding a `"gpt2"`-
//!   shaped reverse mapping against `▁`-marked tokens silently drops every
//!   space (that character isn't in the byte-to-unicode alphabet), which is
//!   exactly the "no spaces between words" bug this module now avoids.
//!
//! - **`"llama"`** (the original Llama/Llama2 vocab shape, and — despite
//!   the name — `gemma-embedding`'s vocab too, e.g. `ggml-org/
//!   embeddinggemma-300M-GGUF`, confirmed directly: no `tokenizer.ggml.
//!   merges` key at all, only `tokenizer.ggml.scores`): **not** a Viterbi/
//!   unigram-LM search despite "SentencePiece-unigram" being the usual
//!   name for this vocab shape — real upstream `llama.cpp` (`src/llama-
//!   vocab.cpp`'s `llm_tokenizer_spm_session`, fetched and read directly)
//!   runs the *same* greedy adjacent-pair-merge loop as `"gemma4"`, just
//!   with two differences: a pair is mergeable whenever its concatenated
//!   string is itself a valid vocab token (no explicit merge-rule table),
//!   with priority = that token's own score (highest first) rather than a
//!   merge rank; and there's no pre-tokenizer word-splitting at all — the
//!   whole (space-escaped) text is fed through the merge loop as one
//!   unbroken run, letting `▁`-marked vocab tokens define word boundaries
//!   implicitly instead of a regex splitting them first.
//!
//! Not implemented: per-architecture pre-tokenizer regex variants beyond
//! gpt2/gemma4 (`tokenizer.ggml.pre`, e.g. `"llama3"` vs `"deepseek-
//! coder"` split text slightly differently around digits/whitespace). One
//! reasonable default pre-tokenizer regex (close to GPT-2's own) is used
//! for every `"gpt2"`-model vocab; see `doc/SERVER_ROADMAP.md`.

use anyhow::{Context, Result, anyhow};
use orangu::gguf::{GgufFile, GgufValue};
use regex::Regex;
use std::collections::HashMap;

/// GPT-2's own pre-tokenizer split pattern: contractions, then runs of
/// letters/digits/other-non-space (each optionally preceded by one space,
/// so the space stays attached to what follows), then any remaining
/// whitespace. GPT-2's original pattern distinguishes trailing whitespace
/// via a negative lookahead (`\s+(?!\S)`); the `regex` crate doesn't support
/// look-around, so that distinction is dropped here — the only observable
/// difference is how a run of 2+ consecutive spaces before a word splits
/// (bundled as one whitespace token here, rather than keeping the last
/// space attached to the following word).
const SPLIT_PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+";

/// SentencePiece's word-boundary marker (U+2581, "▁") — gemma4-family
/// vocabs escape every literal space to this before running BPE merges, and
/// unescape it back on the way out.
const SPM_SPACE: char = '\u{2581}';

pub struct Tokenizer {
    /// Byte-mapped-unicode (`VocabKind::Gpt2Byte`) or raw-UTF-8 (every other
    /// `VocabKind`) token string -> id.
    token_to_id: HashMap<String, u32>,
    id_to_token: Vec<String>,
    /// `(left, right)` -> merge rank; lower merges first.
    merge_ranks: HashMap<(String, String), usize>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    split_re: Regex,
    /// Control/special tokens (`tokenizer.ggml.token_type` == `CONTROL`),
    /// longest-string-first, so a literal occurrence in text (e.g. a chat
    /// template's `<|start_header_id|>`) is recognized as one atomic token
    /// instead of being run through BPE like ordinary text.
    special_tokens: Vec<(String, u32)>,
    pub bos_token: Option<u32>,
    pub eos_token: Option<u32>,
    /// `tokenizer.ggml.add_eos_token` — sentence-embedding models (e.g.
    /// `gemma-embedding`) set this to mark a trailing sentence-boundary
    /// token; causal decoder models generally don't. Only consulted by
    /// [`Tokenizer::encode_for_embedding`], never by [`Tokenizer::encode`]
    /// itself — a trailing EOS injected into a generation prompt would
    /// immediately end generation, which is never what a chat/completion
    /// caller wants.
    add_eos_token: bool,
    /// `tokenizer.ggml.add_bos_token` — only consulted by [`Tokenizer::
    /// encode_for_embedding`], same reasoning as `add_eos_token`: a chat/
    /// completion caller decides whether it wants BOS for itself (via
    /// [`Tokenizer::encode`]'s own `add_bos` parameter), independent of
    /// what this metadata says. Defaults to `true` when the key is absent
    /// — most decoder-LM GGUFs this engine has been tested against want
    /// BOS by default and simply don't bother setting the key; `qwen3vl`
    /// (`ADD_BOS_TOKEN` explicitly `false`) is the one real counter-
    /// example found so far, which is exactly why this isn't hardcoded.
    add_bos_token: bool,
    vocab_kind: VocabKind,
    /// Token id -> raw byte value, for `<0xXX>`-format byte-fallback tokens
    /// (`tokenizer.ggml.token_type == BYTE`). Only populated for a non-
    /// `Gpt2Byte` vocab — a byte-encoded vocab already represents every
    /// byte through its ordinary byte-to-unicode alphabet instead.
    byte_fallback: HashMap<u32, u8>,
    /// `tokenizer.ggml.scores` — `VocabKind::SpmUnigram`'s per-token merge
    /// priority (see [`Tokenizer::spm_merge_symbols`]). Empty for every
    /// other vocab kind.
    scores: Vec<f32>,
    /// `tokenizer.ggml.add_space_prefix` — `VocabKind::SpmUnigram` only:
    /// whether to prepend a literal space (before `▁`-escaping) to the
    /// first text segment, and to any segment immediately following a
    /// special token. Defaults to upstream `llama.cpp`'s own SPM-type
    /// default (`true`) when the key is absent, since that default only
    /// ever applies to a `VocabKind::SpmUnigram` vocab.
    add_space_prefix: bool,
}

/// Which of the two vocab-string encodings and matching merge/decode
/// algorithm this tokenizer uses — see the module doc comment for the full
/// story on each. Dispatched once, at load time, from `tokenizer.ggml.
/// model`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VocabKind {
    /// `"gpt2"` (Llama3/Qwen2/Qwen3/Mistral/qwen35moe): byte-to-unicode
    /// alphabet, GPT-2's own pre-tokenizer regex, merge-rank BPE.
    Gpt2Byte,
    /// `"gemma4"`: raw UTF-8 codepoints, `▁`-space-escaping, newline-only
    /// pre-split, merge-rank BPE, cross-token cleanup on decode.
    Gemma4Raw,
    /// `"llama"` (original Llama/Llama2, and `gemma-embedding`): raw UTF-8
    /// codepoints, `▁`-space-escaping, *no* pre-split, score-priority
    /// merge (no merge-rank table), *no* cross-token cleanup on decode.
    SpmUnigram,
}

/// llama.cpp's `LLAMA_TOKEN_TYPE_CONTROL`/`_BYTE` (`enum llama_token_type`
/// in `llama.h`).
const TOKEN_TYPE_CONTROL: i64 = 3;
const TOKEN_TYPE_BYTE: i64 = 6;

impl Tokenizer {
    pub fn from_gguf(gguf: &GgufFile) -> Result<Self> {
        let tokens = string_array(gguf, "tokenizer.ggml.tokens")
            .ok_or_else(|| anyhow!("GGUF file is missing tokenizer.ggml.tokens"))?;
        let merges = string_array(gguf, "tokenizer.ggml.merges").unwrap_or_default();
        let tokenizer_model = metadata_string(gguf, "tokenizer.ggml.model").unwrap_or_default();
        let vocab_kind = match tokenizer_model.as_str() {
            "gemma4" => VocabKind::Gemma4Raw,
            "llama" => VocabKind::SpmUnigram,
            _ => VocabKind::Gpt2Byte,
        };
        let scores = f32_array(gguf, "tokenizer.ggml.scores").unwrap_or_default();
        let add_space_prefix =
            metadata_u32(gguf, "tokenizer.ggml.add_space_prefix").is_none_or(|v| v != 0);

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (id, tok) in tokens.iter().enumerate() {
            token_to_id.insert(tok.clone(), id as u32);
        }

        let mut merge_ranks = HashMap::with_capacity(merges.len());
        for (rank, merge) in merges.iter().enumerate() {
            let Some((left, right)) = split_merge(merge) else {
                continue;
            };
            merge_ranks.insert((left.to_string(), right.to_string()), rank);
        }

        let byte_to_char = byte_to_unicode_table();
        let char_to_byte = byte_to_char
            .iter()
            .enumerate()
            .map(|(b, &c)| (c, b as u8))
            .collect();

        let bos_token = metadata_u32(gguf, "tokenizer.ggml.bos_token_id");
        let eos_token = metadata_u32(gguf, "tokenizer.ggml.eos_token_id");
        let add_eos_token = metadata_u32(gguf, "tokenizer.ggml.add_eos_token").unwrap_or(0) != 0;
        let add_bos_token =
            metadata_u32(gguf, "tokenizer.ggml.add_bos_token").is_none_or(|v| v != 0);

        let token_types = i64_array(gguf, "tokenizer.ggml.token_type").unwrap_or_default();
        let mut special_tokens: Vec<(String, u32)> = token_types
            .iter()
            .enumerate()
            .filter(|&(_, &ty)| ty == TOKEN_TYPE_CONTROL)
            .filter_map(|(id, _)| tokens.get(id).map(|tok| (tok.clone(), id as u32)))
            .collect();
        special_tokens.sort_by_key(|(tok, _)| std::cmp::Reverse(tok.len()));

        let mut byte_fallback = HashMap::new();
        if vocab_kind != VocabKind::Gpt2Byte {
            for (id, &ty) in token_types.iter().enumerate() {
                if ty == TOKEN_TYPE_BYTE
                    && let Some(tok) = tokens.get(id)
                    && let Some(byte) = parse_byte_fallback_token(tok)
                {
                    byte_fallback.insert(id as u32, byte);
                }
            }
        }

        Ok(Self {
            token_to_id,
            id_to_token: tokens,
            merge_ranks,
            byte_to_char,
            char_to_byte,
            split_re: Regex::new(SPLIT_PATTERN).context("building tokenizer split regex")?,
            special_tokens,
            bos_token,
            eos_token,
            add_eos_token,
            add_bos_token,
            vocab_kind,
            byte_fallback,
            scores,
            add_space_prefix,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_token.len()
    }

    /// Encodes `text` to token ids, prefixed with the model's BOS token if
    /// it has one (matching llama.cpp's default `add_bos` behavior for a
    /// chat/completion request).
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut ids = Vec::new();
        if add_bos && let Some(bos) = self.bos_token {
            ids.push(bos);
        }
        // `true` even when `add_bos` didn't actually push a BOS token —
        // matches upstream `llama.cpp`'s own `llm_tokenizer_spm_session`
        // (`bool is_prev_special = true; // prefix with space if first
        // token`), only consulted by `VocabKind::SpmUnigram`'s space-
        // prefix handling below.
        let mut is_prev_special = true;
        for segment in self.split_on_special_tokens(text) {
            match segment {
                Segment::Special(id) => {
                    ids.push(id);
                    is_prev_special = true;
                }
                Segment::Plain(text) => {
                    match self.vocab_kind {
                        VocabKind::Gpt2Byte => self.encode_plain_byte(text, &mut ids),
                        VocabKind::Gemma4Raw => self.encode_plain_raw(text, &mut ids),
                        VocabKind::SpmUnigram => {
                            self.encode_plain_spm(text, is_prev_special, &mut ids)
                        }
                    }
                    is_prev_special = false;
                }
            }
        }
        ids
    }

    /// Like [`Tokenizer::encode`], but with `add_bos` driven by this
    /// tokenizer's own `add_bos_token` metadata (not hardcoded — `qwen3vl`-
    /// embedding models explicitly set this `false`, unlike every other
    /// model tested so far) and with the model's EOS token additionally
    /// appended when `add_eos_token` says to. The embeddings-request path
    /// (`http::openai::pooled_embedding`) is the only caller; a trailing
    /// EOS has no place in a generation prompt.
    pub fn encode_for_embedding(&self, text: &str) -> Vec<u32> {
        let mut ids = self.encode(text, self.add_bos_token);
        if self.add_eos_token
            && let Some(eos) = self.eos_token
        {
            ids.push(eos);
        }
        ids
    }

    fn encode_plain_byte(&self, text: &str, ids: &mut Vec<u32>) {
        for word_match in self.split_re.find_iter(text) {
            let word = word_match.as_str();
            let symbols = self.bpe_merge_byte(word);
            for symbol in symbols {
                match self.token_to_id.get(&symbol) {
                    Some(&id) => ids.push(id),
                    // A symbol with no vocab entry (shouldn't happen for a
                    // byte-level vocab, which always has all 256 single
                    // bytes) falls back to its individual mapped bytes.
                    None => {
                        for ch in symbol.chars() {
                            if let Some(&id) = self.token_to_id.get(&ch.to_string()) {
                                ids.push(id);
                            }
                        }
                    }
                }
            }
        }
    }

    /// The `"gemma4"`-vocab encode path: escape every literal space to `▁`
    /// first (matching `llama_escape_whitespace`), then split only on
    /// newline runs and BPE-merge each word's raw UTF-8 codepoints —
    /// no byte-to-unicode alphabet involved at all.
    fn encode_plain_raw(&self, text: &str, ids: &mut Vec<u32>) {
        let escaped = text.replace(' ', &SPM_SPACE.to_string());
        for word in split_newline_runs(&escaped) {
            // Real llama.cpp's gemma4-specific fix (ref: llama.cpp#21343):
            // an all-newline run that's directly a vocab token is emitted
            // as-is, skipping the merge process (this vocab has multi-
            // newline tokens like "\n\n\n...\n" that BPE-merging the
            // ordinary way wouldn't necessarily reconstruct).
            if !word.is_empty()
                && word.chars().all(|c| c == '\n')
                && let Some(&id) = self.token_to_id.get(word)
            {
                ids.push(id);
                continue;
            }
            let symbols = self.bpe_merge_raw(word);
            for symbol in symbols {
                match self.token_to_id.get(&symbol) {
                    Some(&id) => ids.push(id),
                    // Non-byte-encoded BPE represents an unmatched symbol's
                    // raw bytes via `<0xXX>`-format fallback tokens rather
                    // than a byte-to-unicode alphabet.
                    None => {
                        for byte in symbol.bytes() {
                            if let Some(&id) = self.token_to_id.get(&byte_fallback_token_name(byte))
                            {
                                ids.push(id);
                            }
                        }
                    }
                }
            }
        }
    }

    /// The `"llama"`-vocab (`VocabKind::SpmUnigram`) encode path: unlike
    /// `Self::encode_plain_raw`, there is no pre-tokenizer word-splitting
    /// at all — the *whole* segment is escaped and fed through `Self::
    /// spm_merge_symbols` as one run (real upstream `llama.cpp` does the
    /// same: `llm_tokenizer_spm_session::tokenize` never splits its input).
    /// `is_prev_special` gates the leading-space prefix exactly as
    /// upstream's `is_prev_special`/`add_space_prefix` do.
    fn encode_plain_spm(&self, text: &str, is_prev_special: bool, ids: &mut Vec<u32>) {
        let mut escaped = String::new();
        if self.add_space_prefix && is_prev_special {
            escaped.push(' ');
        }
        escaped.push_str(text);
        let escaped = escaped.replace(' ', &SPM_SPACE.to_string());

        let symbols: Vec<String> = escaped.chars().map(|c| c.to_string()).collect();
        let symbols = self.spm_merge_symbols(symbols);
        for symbol in symbols {
            match self.token_to_id.get(&symbol) {
                Some(&id) => ids.push(id),
                // A leftover symbol that never merged into a vocab token
                // (e.g. a character outside this vocab entirely) falls
                // back to its raw bytes' `<0xXX>`-format tokens — the same
                // fallback `Self::encode_plain_raw` uses.
                None => {
                    for byte in symbol.bytes() {
                        if let Some(&id) = self.token_to_id.get(&byte_fallback_token_name(byte)) {
                            ids.push(id);
                        }
                    }
                }
            }
        }
    }

    /// Splits `text` around any literal occurrence of a control/special
    /// token's own string (e.g. a chat template's `<|start_header_id|>`),
    /// longest-match-first, so those bypass BPE entirely — matching
    /// llama.cpp's own special-token handling in `llama-vocab.cpp`.
    fn split_on_special_tokens<'a>(&self, text: &'a str) -> Vec<Segment<'a>> {
        if self.special_tokens.is_empty() {
            return vec![Segment::Plain(text)];
        }
        let mut segments = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // Earliest occurrence across every special token; on a tie
            // (two candidates starting at the same position) the longer one
            // wins, since `self.special_tokens` is sorted longest-first and
            // this scan keeps the first (not later) match at a given
            // earliest position.
            let earliest = self
                .special_tokens
                .iter()
                .filter_map(|(special, id)| {
                    rest.find(special.as_str()).map(|pos| (pos, special, *id))
                })
                .min_by_key(|(pos, special, _)| (*pos, std::cmp::Reverse(special.len())));

            let Some((pos, special, id)) = earliest else {
                segments.push(Segment::Plain(rest));
                break;
            };
            if pos > 0 {
                segments.push(Segment::Plain(&rest[..pos]));
            }
            segments.push(Segment::Special(id));
            rest = &rest[pos + special.len()..];
        }
        segments
    }

    /// Decodes token ids back to text. For a `VocabKind::Gpt2Byte`
    /// (`"gpt2"`) vocab: concatenates each token's mapped string, reversing
    /// the byte-to-unicode mapping. For `Gemma4Raw`/`SpmUnigram`:
    /// raw-byte-fallback tokens decode to their one raw byte, everything
    /// else decodes as literal UTF-8 text with `▁` unescaped back to a
    /// space. Unknown ids are
    /// skipped. Safe to call per-token during streaming (each token's
    /// bytes depend only on itself) — but see [`Tokenizer::
    /// clean_up_tokenization_spaces`] for the cross-token cleanup a
    /// `"gemma4"` vocab additionally needs once the *complete* text is
    /// known, exactly mirroring real llama.cpp's own split between a per-
    /// token `token_to_piece` (no cleanup) and a whole-sequence
    /// `detokenize` (cleanup applied once at the end).
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(&b) = self.byte_fallback.get(&id) {
                bytes.push(b);
                continue;
            }
            let Some(token) = self.id_to_token.get(id as usize) else {
                continue;
            };
            if self.vocab_kind == VocabKind::Gpt2Byte {
                for ch in token.chars() {
                    if let Some(&b) = self.char_to_byte.get(&ch) {
                        bytes.push(b);
                    }
                }
            } else {
                for ch in token.chars() {
                    if ch == SPM_SPACE {
                        bytes.push(b' ');
                    } else {
                        let mut buf = [0u8; 4];
                        bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Cross-token cleanup a `"gemma4"` vocab needs once the *complete*
    /// generated (or detokenized) text is known — real llama.cpp's own
    /// `clean_up_tokenization_spaces`-style pass (`llama_vocab::impl::
    /// detokenize`'s three passes, confirmed by reading that source
    /// directly): drops the space immediately before `? ! . ,`, collapses
    /// a lone apostrophe surrounded by spaces, and drops the space before
    /// `'s`/`'m`/`'re`/`'ve` contractions (deliberately *not* `'t`/`'d`/
    /// `'ll` — matching upstream's own, slightly inconsistent behavior
    /// exactly rather than a more "sensible" version). A no-op for
    /// `VocabKind::Gpt2Byte` (whose pre-tokenizer types this engine
    /// supports — `"gpt2"`-style Qwen2/Qwen3/Llama3/qwen35moe — don't use
    /// this pass, their tokens are already correctly spaced) *and* for
    /// `VocabKind::SpmUnigram` (real upstream sets `clean_spaces = false`
    /// for `LLAMA_VOCAB_TYPE_SPM`, confirmed directly against `src/
    /// llama-vocab.cpp` — this pass is `"gemma4"`-specific, not a general
    /// property of "not byte-encoded").
    pub fn clean_up_tokenization_spaces(&self, text: &str) -> String {
        if self.vocab_kind != VocabKind::Gemma4Raw {
            return text.to_string();
        }
        clean_spaces_postprocess(text)
    }

    pub fn token_text(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(String::as_str)
    }

    /// Standard GPT-2 BPE: map `word`'s bytes to the byte-unicode alphabet,
    /// then run [`Tokenizer::merge_symbols`].
    fn bpe_merge_byte(&self, word: &str) -> Vec<String> {
        let symbols: Vec<String> = word
            .bytes()
            .map(|b| self.byte_to_char[b as usize].to_string())
            .collect();
        self.merge_symbols(symbols)
    }

    /// The `"gemma4"`-vocab BPE variant: `word`'s raw UTF-8 codepoints are
    /// the initial symbols (no byte-to-unicode alphabet), then run
    /// [`Tokenizer::merge_symbols`] — the merge algorithm itself is
    /// identical either way, only the starting alphabet differs.
    fn bpe_merge_raw(&self, word: &str) -> Vec<String> {
        let symbols: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        self.merge_symbols(symbols)
    }

    /// Repeatedly merges the lowest-rank adjacent pair of `symbols` until
    /// none of the remaining pairs have a merge rule — the one merge
    /// algorithm shared by both vocab shapes; only the initial symbol
    /// alphabet differs between [`Tokenizer::bpe_merge_byte`] and
    /// [`Tokenizer::bpe_merge_raw`].
    fn merge_symbols(&self, mut symbols: Vec<String>) -> Vec<String> {
        if symbols.len() < 2 {
            return symbols;
        }

        loop {
            let mut best: Option<(usize, usize)> = None; // (rank, pair index)
            for i in 0..symbols.len() - 1 {
                if let Some(&rank) = self
                    .merge_ranks
                    .get(&(symbols[i].clone(), symbols[i + 1].clone()))
                    && best.is_none_or(|(best_rank, _)| rank < best_rank)
                {
                    best = Some((rank, i));
                }
            }
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols.splice(i..=i + 1, [merged]);
        }
        symbols
    }

    /// `VocabKind::SpmUnigram`'s merge algorithm — real upstream
    /// `llama.cpp`'s `llm_tokenizer_spm_session` (`src/llama-vocab.cpp`,
    /// fetched and read directly): the same greedy adjacent-pair-merge
    /// loop as [`Tokenizer::merge_symbols`], but a pair is mergeable
    /// whenever its *concatenated string is itself a valid vocab token*
    /// (this vocab has no `tokenizer.ggml.merges` table at all) rather
    /// than an explicit merge-rank lookup, and priority is that token's
    /// own *score* — highest first, ties broken by earliest position
    /// (matching upstream's `llm_bigram_spm::comparator` exactly: `(l.score
    /// < r.score) || (l.score == r.score && l.left > r.left)`, a max-heap
    /// on score with leftmost-wins on ties — the same outcome this method's
    /// "only replace `best` on strictly greater score" rescan produces,
    /// since earlier-found ties are never displaced).
    fn spm_merge_symbols(&self, mut symbols: Vec<String>) -> Vec<String> {
        if symbols.len() < 2 {
            return symbols;
        }

        loop {
            let mut best: Option<(f32, usize)> = None; // (score, pair index)
            for i in 0..symbols.len() - 1 {
                let merged = format!("{}{}", symbols[i], symbols[i + 1]);
                if let Some(&id) = self.token_to_id.get(&merged) {
                    let score = self.scores.get(id as usize).copied().unwrap_or(0.0);
                    if best.is_none_or(|(best_score, _)| score > best_score) {
                        best = Some((score, i));
                    }
                }
            }
            let Some((_, i)) = best else { break };
            let merged = format!("{}{}", symbols[i], symbols[i + 1]);
            symbols.splice(i..=i + 1, [merged]);
        }
        symbols
    }
}

enum Segment<'a> {
    Plain(&'a str),
    Special(u32),
}

fn i64_array(gguf: &GgufFile, key: &str) -> Option<Vec<i64>> {
    gguf.metadata.iter().find_map(|(k, v)| {
        (k == key).then_some(v).and_then(|v| match v {
            GgufValue::Array(items) => Some(
                items
                    .iter()
                    .map(|item| match item {
                        GgufValue::I8(v) => *v as i64,
                        GgufValue::I16(v) => *v as i64,
                        GgufValue::I32(v) => *v as i64,
                        GgufValue::I64(v) => *v,
                        GgufValue::U8(v) => *v as i64,
                        GgufValue::U16(v) => *v as i64,
                        GgufValue::U32(v) => *v as i64,
                        GgufValue::U64(v) => *v as i64,
                        _ => 0,
                    })
                    .collect(),
            ),
            _ => None,
        })
    })
}

fn f32_array(gguf: &GgufFile, key: &str) -> Option<Vec<f32>> {
    gguf.metadata.iter().find_map(|(k, v)| {
        (k == key).then_some(v).and_then(|v| match v {
            GgufValue::Array(items) => Some(
                items
                    .iter()
                    .map(|item| match item {
                        GgufValue::F32(v) => *v,
                        GgufValue::F64(v) => *v as f32,
                        _ => 0.0,
                    })
                    .collect(),
            ),
            _ => None,
        })
    })
}

fn string_array(gguf: &GgufFile, key: &str) -> Option<Vec<String>> {
    gguf.metadata.iter().find_map(|(k, v)| {
        (k == key).then_some(v).and_then(|v| match v {
            GgufValue::Array(items) => Some(
                items
                    .iter()
                    .filter_map(|item| match item {
                        GgufValue::String(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect(),
            ),
            _ => None,
        })
    })
}

fn metadata_u32(gguf: &GgufFile, key: &str) -> Option<u32> {
    gguf.metadata
        .iter()
        .find(|(k, _)| k == key)
        .and_then(|(_, v)| v.as_u64())
        .map(|v| v as u32)
}

fn metadata_string(gguf: &GgufFile, key: &str) -> Option<String> {
    gguf.metadata.iter().find_map(|(k, v)| {
        (k == key).then_some(v).and_then(|v| match v {
            GgufValue::String(s) => Some(s.clone()),
            _ => None,
        })
    })
}

/// Splits a `tokenizer.ggml.merges` entry into its `(left, right)` pieces —
/// matches real llama.cpp's own `word.find(' ', 1)` exactly (search for the
/// space starting one *byte* in, not from the very start), rather than the
/// simpler "first space anywhere" a plain `split_once(' ')` would use.
/// Observably identical for every merge in practice (a merge's `left` part
/// is never itself a bare, unescaped space), but this is what upstream
/// actually does, so it's what this mirrors. Searches raw bytes rather than
/// slicing `merge[1..]` directly: byte 1 of a merge starting with a multi-
/// byte character (e.g. a gemma4 merge starting with `▁`, itself 3 bytes)
/// isn't a char boundary, so a `&str` slice there would panic — but a
/// space is always a complete one-byte codepoint, so once a `b' '` is
/// found, splitting immediately before and after it is always safe.
fn split_merge(merge: &str) -> Option<(&str, &str)> {
    let bytes = merge.as_bytes();
    if bytes.len() < 2 {
        return None;
    }
    let pos = bytes[1..].iter().position(|&b| b == b' ')? + 1;
    Some((&merge[..pos], &merge[pos + 1..]))
}

/// Parses ggml's `<0xXX>` byte-fallback token spelling (two uppercase hex
/// digits) back to the raw byte it represents.
fn parse_byte_fallback_token(token: &str) -> Option<u8> {
    let hex = token.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(hex, 16).ok()
}

/// Formats a raw byte as ggml's `<0xXX>` byte-fallback token spelling (two
/// uppercase hex digits) — the encode-side inverse of
/// [`parse_byte_fallback_token`].
fn byte_fallback_token_name(byte: u8) -> String {
    format!("<0x{byte:02X}>")
}

/// The `"gemma4"`-vocab pre-tokenizer: splits into runs of non-newline
/// characters and runs of newlines (`"[^\n]+|[\n]+"` in real llama.cpp) —
/// hand-written rather than added as another `Regex` pattern, since a plain
/// "does this char equal '\n'" scan needs no regex engine at all.
fn split_newline_runs(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut in_newline_run: Option<bool> = None;
    for (i, ch) in text.char_indices() {
        let is_newline = ch == '\n';
        match in_newline_run {
            Some(prev) if prev != is_newline => {
                out.push(&text[start..i]);
                start = i;
                in_newline_run = Some(is_newline);
            }
            None => in_newline_run = Some(is_newline),
            _ => {}
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// Real llama.cpp's `clean_up_tokenization_spaces`-style post-processing
/// (`llama_vocab::impl::detokenize`'s three passes, read directly from
/// upstream source rather than guessed) — see [`Tokenizer::
/// clean_up_tokenization_spaces`] for what and why.
fn clean_spaces_postprocess(text: &str) -> String {
    let mut chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return text.to_string();
    }

    // Pass 1: drop the space immediately before ? ! . ,
    let mut out: Vec<char> = Vec::with_capacity(chars.len());
    out.push(chars[0]);
    for &x in &chars[1..] {
        if out.last() == Some(&' ') && matches!(x, '?' | '!' | '.' | ',') {
            out.pop();
        }
        out.push(x);
    }
    chars = out;

    // Pass 2: collapse a lone apostrophe surrounded by spaces (" ' " -> "'").
    let mut out: Vec<char> = Vec::with_capacity(chars.len());
    out.push(chars[0]);
    let mut i = 1;
    while i < chars.len() {
        let x = chars[i];
        if x == '\'' && i + 1 < chars.len() && out.last() == Some(&' ') && chars[i + 1] == ' ' {
            out.pop(); // drop the preceding space
            out.push(x);
            i += 2; // and the following one
            continue;
        }
        out.push(x);
        i += 1;
    }
    chars = out;

    // Pass 3: drop the space before 's/'m/'re/'ve contractions — but *not*
    // before 't/'d/'ll, matching upstream's own inconsistency exactly.
    let mut out: Vec<char> = Vec::with_capacity(chars.len());
    out.push(chars[0]);
    for i in 1..chars.len() {
        let x = chars[i];
        if x == '\'' && out.last() == Some(&' ') {
            let drop_space = match chars.get(i + 1..) {
                Some([c, ..]) if *c == 's' || *c == 'm' => true,
                Some([c1, c2, ..]) if (*c1 == 'r' || *c1 == 'v') && *c2 == 'e' => true,
                _ => false,
            };
            if drop_space {
                out.pop();
            }
        }
        out.push(x);
    }

    out.into_iter().collect()
}

/// The GPT-2 `bytes_to_unicode()` table: every byte value maps to a visible
/// unicode codepoint, so a BPE vocab (whose tokens are ordinary strings) can
/// represent arbitrary binary byte sequences. Printable ASCII/Latin-1 bytes
/// map to themselves; everything else (control characters, space, ...) maps
/// to a codepoint starting at 256 upward, in byte order.
fn byte_to_unicode_table() -> [char; 256] {
    let printable: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut table = [0u32; 256];
    let mut assigned = [false; 256];
    for &b in &printable {
        table[b as usize] = b;
        assigned[b as usize] = true;
    }
    let mut next_extra = 256u32;
    for (b, slot) in table.iter_mut().enumerate() {
        if !assigned[b] {
            *slot = next_extra;
            next_extra += 1;
        }
    }
    let mut out = ['\u{0}'; 256];
    for (b, &cp) in table.iter().enumerate() {
        out[b] = char::from_u32(cp).expect("byte_to_unicode codepoints are all valid");
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_gguf(tokens: &[&str], merges: &[&str], bos: u32, eos: u32) -> GgufFile {
        GgufFile {
            version: 3,
            metadata: vec![
                (
                    "tokenizer.ggml.tokens".to_string(),
                    GgufValue::Array(
                        tokens
                            .iter()
                            .map(|t| GgufValue::String(t.to_string()))
                            .collect(),
                    ),
                ),
                (
                    "tokenizer.ggml.merges".to_string(),
                    GgufValue::Array(
                        merges
                            .iter()
                            .map(|m| GgufValue::String(m.to_string()))
                            .collect(),
                    ),
                ),
                (
                    "tokenizer.ggml.bos_token_id".to_string(),
                    GgufValue::U32(bos),
                ),
                (
                    "tokenizer.ggml.eos_token_id".to_string(),
                    GgufValue::U32(eos),
                ),
            ],
            tensors: vec![],
            alignment: 32,
            data_offset: 0,
        }
    }

    /// A byte-level vocab with every single byte's mapped char as its own
    /// token, plus one merge combining 'h'+'i' into "hi" — the minimal
    /// vocab needed to exercise both the fallback (unmerged) path and an
    /// actual BPE merge.
    fn minimal_byte_vocab() -> GgufFile {
        let byte_to_char = byte_to_unicode_table();
        let mut tokens: Vec<String> = (0..256u32)
            .map(|b| byte_to_char[b as usize].to_string())
            .collect();
        tokens.push("hi".to_string());
        let owned_tokens: Vec<&str> = tokens.iter().map(String::as_str).collect();
        build_gguf(&owned_tokens, &["h i"], 1, 2)
    }

    #[test]
    fn encode_falls_back_to_individual_bytes_without_merges() {
        let gguf = build_gguf(&["a", "b", "c"], &[], 0, 1);
        // No merges and no BOS configured beyond what's requested — just
        // confirms construction succeeds and vocab_size is right.
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        assert_eq!(tok.vocab_size(), 3);
    }

    #[test]
    fn encode_applies_a_merge_rule() {
        let gguf = minimal_byte_vocab();
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        let ids = tok.encode("hi", false);
        let hi_id = tok.token_to_id["hi"];
        assert_eq!(ids, vec![hi_id]);
    }

    #[test]
    fn encode_recognizes_adjacent_literal_special_tokens_over_bpe() {
        let byte_to_char = byte_to_unicode_table();
        let mut tokens: Vec<String> = (0..256u32)
            .map(|b| byte_to_char[b as usize].to_string())
            .collect();
        let start_idx = tokens.len();
        tokens.push("<|a|>".to_string());
        let end_idx = tokens.len();
        tokens.push("<|b|>".to_string());
        let owned_tokens: Vec<&str> = tokens.iter().map(String::as_str).collect();
        let mut gguf = build_gguf(&owned_tokens, &[], 0, 0);
        let mut types = vec![1i32; owned_tokens.len()]; // NORMAL
        types[start_idx] = 3; // CONTROL
        types[end_idx] = 3; // CONTROL
        gguf.metadata.push((
            "tokenizer.ggml.token_type".to_string(),
            GgufValue::Array(types.into_iter().map(GgufValue::I32).collect()),
        ));

        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        // Two special tokens directly adjacent, no plain text between them
        // — the case that broke a naive longest-token-first scan.
        let ids = tok.encode("<|a|><|b|>", false);
        assert_eq!(ids, vec![start_idx as u32, end_idx as u32]);
    }

    #[test]
    fn encode_prefixes_bos_when_requested() {
        let gguf = minimal_byte_vocab();
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        let ids = tok.encode("hi", true);
        assert_eq!(ids[0], tok.bos_token.unwrap());
    }

    /// The bug this test is named for: `Tokenizer::encode_for_embedding`
    /// must honor this tokenizer's own `add_bos_token` metadata rather
    /// than hardcoding BOS on — real `qwen3vl`-embedding models set
    /// `tokenizer.ggml.add_bos_token = false` (confirmed directly against
    /// a real GGUF), and prepending a spurious BOS anyway produced a
    /// genuinely wrong embedding (cosine ~0.47 against real llama.cpp,
    /// not just float noise) before this was caught and fixed.
    #[test]
    fn encode_for_embedding_omits_bos_when_add_bos_token_is_false() {
        let mut gguf = minimal_byte_vocab();
        gguf.metadata.push((
            "tokenizer.ggml.add_bos_token".to_string(),
            GgufValue::Bool(false),
        ));
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        let ids = tok.encode_for_embedding("hi");
        assert_ne!(ids[0], tok.bos_token.unwrap());
    }

    /// The complementary case: `add_bos_token` absent (or explicitly
    /// `true`) still prepends BOS, matching every other model this engine
    /// has been tested against.
    #[test]
    fn encode_for_embedding_includes_bos_by_default() {
        let gguf = minimal_byte_vocab();
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        let ids = tok.encode_for_embedding("hi");
        assert_eq!(ids[0], tok.bos_token.unwrap());
    }

    #[test]
    fn decode_reverses_encode_for_ascii_text() {
        let gguf = minimal_byte_vocab();
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        let text = "hi there";
        let ids = tok.encode(text, false);
        assert_eq!(tok.decode(&ids), text);
    }

    #[test]
    fn missing_tokens_key_is_an_error() {
        let gguf = GgufFile {
            version: 3,
            metadata: vec![],
            tensors: vec![],
            alignment: 32,
            data_offset: 0,
        };
        assert!(Tokenizer::from_gguf(&gguf).is_err());
    }

    /// Like `build_gguf`, plus `tokenizer.ggml.model = "gemma4"` — the
    /// non-byte-encoded, space-escaped BPE vocab shape.
    fn build_gemma4_gguf(tokens: &[&str], merges: &[&str]) -> GgufFile {
        let mut gguf = build_gguf(tokens, merges, 0, 1);
        gguf.metadata.push((
            "tokenizer.ggml.model".to_string(),
            GgufValue::String("gemma4".to_string()),
        ));
        gguf
    }

    /// Regression test for a real bug caught by testing against a real
    /// downloaded gemma-4-E2B-it model, not just synthetic vocabs: sending
    /// any message through the web UI against a gemma4 model crashed the
    /// whole server on startup with "start byte index 1 is not a char
    /// boundary" — merges starting with `▁` (a 3-byte character) made
    /// `split_merge`'s original `merge[1..]` implementation panic.
    #[test]
    fn split_merge_does_not_panic_on_a_multibyte_first_character() {
        assert_eq!(split_merge("\u{2581} b"), Some(("\u{2581}", "b")));
    }

    /// The core bug this vocab shape's support fixes: without it, `▁`
    /// (SentencePiece's word-boundary marker, `tokenizer.ggml.model =
    /// "gemma4"`'s real spelling for a leading space) isn't in the byte-
    /// to-unicode alphabet a `"gpt2"`-shaped reverse mapping expects, so it
    /// gets silently dropped — producing exactly the "no spaces between
    /// words" a real gemma-4-E2B-it model's web UI responses showed before
    /// this was fixed.
    #[test]
    fn gemma4_escapes_spaces_and_merges_raw_codepoints() {
        let gguf = build_gemma4_gguf(&["a", "b", "\u{2581}", "\u{2581}b"], &["\u{2581} b"]);
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        assert_eq!(tok.vocab_kind, VocabKind::Gemma4Raw);

        let ids = tok.encode("a b", false);
        assert_eq!(
            ids,
            vec![tok.token_to_id["a"], tok.token_to_id["\u{2581}b"]]
        );
        assert_eq!(tok.decode(&ids), "a b");
    }

    #[test]
    fn gemma4_byte_fallback_roundtrips_an_unmapped_byte() {
        let mut gguf = build_gemma4_gguf(&["a", "<0x00>"], &[]);
        gguf.metadata.push((
            "tokenizer.ggml.token_type".to_string(),
            GgufValue::Array(vec![GgufValue::I32(1), GgufValue::I32(6)]), // NORMAL, BYTE
        ));
        let tok = Tokenizer::from_gguf(&gguf).unwrap();

        // A NUL byte has no single-codepoint vocab entry other than the
        // byte-fallback token, so it must round-trip through it.
        let text = "a\u{0}";
        let ids = tok.encode(text, false);
        assert_eq!(ids, vec![tok.token_to_id["a"], tok.token_to_id["<0x00>"]]);
        assert_eq!(tok.decode(&ids), text);
    }

    /// Like `build_gemma4_gguf`, but `tokenizer.ggml.model = "llama"`
    /// (`VocabKind::SpmUnigram`) with `scores` instead of `merges` — no
    /// `tokenizer.ggml.merges` key at all, matching a real GGUF like
    /// `ggml-org/embeddinggemma-300M-GGUF`'s (confirmed directly: it has
    /// none). `add_space_prefix` defaults to upstream's own SPM default
    /// (`true`) unless explicitly overridden.
    fn build_spm_gguf(tokens: &[&str], scores: &[f32], add_space_prefix: Option<bool>) -> GgufFile {
        let mut gguf = GgufFile {
            version: 3,
            metadata: vec![
                (
                    "tokenizer.ggml.tokens".to_string(),
                    GgufValue::Array(
                        tokens
                            .iter()
                            .map(|t| GgufValue::String(t.to_string()))
                            .collect(),
                    ),
                ),
                (
                    "tokenizer.ggml.scores".to_string(),
                    GgufValue::Array(scores.iter().map(|&s| GgufValue::F32(s)).collect()),
                ),
                ("tokenizer.ggml.bos_token_id".to_string(), GgufValue::U32(0)),
                ("tokenizer.ggml.eos_token_id".to_string(), GgufValue::U32(1)),
                (
                    "tokenizer.ggml.model".to_string(),
                    GgufValue::String("llama".to_string()),
                ),
            ],
            tensors: vec![],
            alignment: 32,
            data_offset: 0,
        };
        if let Some(add_space_prefix) = add_space_prefix {
            gguf.metadata.push((
                "tokenizer.ggml.add_space_prefix".to_string(),
                GgufValue::Bool(add_space_prefix),
            ));
        }
        gguf
    }

    /// `tokenizer.ggml.model = "llama"` with no `merges` key at all
    /// dispatches to `VocabKind::SpmUnigram`, not `Gpt2Byte` — the exact
    /// bug found (and fixed) against a real `embeddinggemma-300M` GGUF:
    /// without this, the vocab fell back to byte-encoded BPE with almost
    /// every merge missing, producing near-byte-level tokenization (43
    /// tokens instead of 11 for a 9-word sentence).
    #[test]
    fn tokenizer_model_llama_dispatches_to_spm_unigram() {
        let gguf = build_spm_gguf(&["a", "b", "\u{2581}", "\u{2581}b"], &[0.0; 4], Some(false));
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        assert_eq!(tok.vocab_kind, VocabKind::SpmUnigram);
    }

    /// The core of `Self::spm_merge_symbols`: a pair only merges when its
    /// concatenated string is itself a valid vocab token — no merge-rank
    /// table involved at all, unlike `Self::merge_symbols`. `"a\u{2581}"`
    /// (a + space) is *not* a vocab token here, so `('a','\u{2581}')`
    /// never merges even though it's the leftmost pair; only `('\u{2581}',
    /// 'b') -> "\u{2581}b"` does, and the leftover `'a'` symbol (with no
    /// direct vocab entry) falls through to `Self::spm_merge_symbols`'s
    /// caller, which byte-falls-back it via `<0x61>`.
    #[test]
    fn spm_merge_only_merges_pairs_that_are_valid_vocab_tokens() {
        let mut gguf = build_spm_gguf(
            &["<0x61>", "b", "\u{2581}", "\u{2581}b"],
            &[0.0, 0.0, 0.0, 1.0],
            Some(false),
        );
        gguf.metadata.push((
            "tokenizer.ggml.token_type".to_string(),
            GgufValue::Array(vec![
                GgufValue::I32(6), // BYTE
                GgufValue::I32(1), // NORMAL
                GgufValue::I32(1), // NORMAL
                GgufValue::I32(1), // NORMAL
            ]),
        ));
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        // symbols start as ['a', '▁', 'b']; "a▁" isn't a vocab token so
        // only "▁b" merges, leaving ['a', '▁b'] — 'a' has no direct vocab
        // entry (only a byte-fallback token), so it decomposes to <0x61>.
        let ids = tok.encode("a b", false);
        assert_eq!(
            ids,
            vec![tok.token_to_id["<0x61>"], tok.token_to_id["\u{2581}b"]]
        );
    }

    /// When two *different* valid merges are both available in the same
    /// pass, the higher-scoring one is chosen first — checked by giving
    /// `"ab"` a higher score than `"bc"` in a 3-symbol chain where both are
    /// simultaneously mergeable.
    #[test]
    fn spm_merge_prefers_higher_score_over_leftmost_when_both_valid() {
        // ids: 0=a 1=<pad> 2=b 3=c 4=ab 5=bc
        let gguf = build_spm_gguf(
            &["a", "<pad>", "b", "c", "ab", "bc"],
            &[0.0, 0.0, 0.0, 0.0, /* ab */ 5.0, /* bc */ 1.0],
            Some(false),
        );
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        // symbols start as ['a', 'b', 'c']; "ab" (score 5) beats "bc"
        // (score 1), so it merges first -> ['ab', 'c'] -> "abc" isn't a
        // token, so no further merge.
        let ids = tok.encode("abc", false);
        assert_eq!(ids, vec![tok.token_to_id["ab"], tok.token_to_id["c"]]);
    }

    /// `add_space_prefix = true` (upstream's own default for `VocabKind::
    /// SpmUnigram` when the GGUF doesn't override it) prepends a literal
    /// space — escaped to `▁` like every other space — to the very start
    /// of the text, so a bare `"b"` becomes `"▁b"`, not `"b"`.
    #[test]
    fn spm_add_space_prefix_escapes_a_leading_space_by_default() {
        let gguf = build_spm_gguf(&["b", "\u{2581}b"], &[0.0, 1.0], None);
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        let ids = tok.encode("b", false);
        assert_eq!(ids, vec![tok.token_to_id["\u{2581}b"]]);
    }

    /// A character with no vocab entry at all (and no possible merge)
    /// falls back to its raw UTF-8 bytes' `<0xXX>`-format tokens — same
    /// fallback convention `Self::encode_plain_raw` uses.
    #[test]
    fn spm_falls_back_to_byte_tokens_for_an_unknown_character() {
        let mut gguf = build_spm_gguf(&["a", "<0x00>"], &[0.0, 0.0], Some(false));
        gguf.metadata.push((
            "tokenizer.ggml.token_type".to_string(),
            GgufValue::Array(vec![GgufValue::I32(1), GgufValue::I32(6)]), // NORMAL, BYTE
        ));
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        let text = "a\u{0}";
        let ids = tok.encode(text, false);
        assert_eq!(ids, vec![tok.token_to_id["a"], tok.token_to_id["<0x00>"]]);
        assert_eq!(tok.decode(&ids), text);
    }

    /// `VocabKind::SpmUnigram` doesn't get `Gemma4Raw`'s punctuation
    /// cleanup pass — real upstream `llama.cpp` sets `clean_spaces = false`
    /// for `LLAMA_VOCAB_TYPE_SPM` specifically (confirmed directly against
    /// `src/llama-vocab.cpp`), unlike the `"gemma4"` BPE variant.
    #[test]
    fn clean_up_tokenization_spaces_is_a_no_op_for_spm_unigram() {
        let gguf = build_spm_gguf(&["a"], &[0.0], Some(false));
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        assert_eq!(
            tok.clean_up_tokenization_spaces("hi , there !"),
            "hi , there !"
        );
    }

    #[test]
    fn clean_up_tokenization_spaces_is_a_no_op_for_a_byte_encoded_vocab() {
        let gguf = minimal_byte_vocab();
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        assert_eq!(
            tok.clean_up_tokenization_spaces("hi , there !"),
            "hi , there !"
        );
    }

    #[test]
    fn clean_up_tokenization_spaces_drops_space_before_punctuation() {
        let gguf = build_gemma4_gguf(&["a"], &[]);
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        assert_eq!(
            tok.clean_up_tokenization_spaces("Hello , world !"),
            "Hello, world!"
        );
    }

    #[test]
    fn clean_up_tokenization_spaces_collapses_a_lone_apostrophe() {
        let gguf = build_gemma4_gguf(&["a"], &[]);
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        // Both the space before *and* the space after get dropped (real
        // llama.cpp's own `text[++i] = '\0'` consumes the trailing space
        // too, not just the leading one).
        assert_eq!(
            tok.clean_up_tokenization_spaces("rock ' n ' roll"),
            "rock'n'roll"
        );
    }

    #[test]
    fn clean_up_tokenization_spaces_drops_space_before_s_and_ve_but_not_t_or_ll() {
        let gguf = build_gemma4_gguf(&["a"], &[]);
        let tok = Tokenizer::from_gguf(&gguf).unwrap();
        // 's/'ve lose their preceding space; 't/'ll keep it — matching real
        // llama.cpp's own (slightly inconsistent) behavior exactly.
        assert_eq!(
            tok.clean_up_tokenization_spaces("it 's fine, we 've won"),
            "it's fine, we've won"
        );
        assert_eq!(
            tok.clean_up_tokenization_spaces("don 't stop, they 'll see"),
            "don 't stop, they 'll see"
        );
    }

    /// Cross-check against real llama.cpp's `/tokenize?add_special=false`
    /// for `ggml-org/embeddinggemma-300M-GGUF:Q8_0` — the exact model that
    /// exposed this whole vocab shape gap (`tokenizer.ggml.model =
    /// "llama"`, no `merges` key at all, so it needs `VocabKind::
    /// SpmUnigram`, not `Gpt2Byte`). Before that fix, this sentence
    /// encoded to 43 near-byte-level tokens; real llama.cpp gives exactly
    /// these 9. Run with `ORANGU_TEST_EMBEDDING_MODEL=/path/to/
    /// embeddinggemma-300M-Q8_0.gguf cargo test --release --bin
    /// orangu-server tokenizer -- --ignored`.
    #[test]
    #[ignore]
    fn embeddinggemma_tokenization_matches_real_llama_cpp() {
        let path =
            std::env::var("ORANGU_TEST_EMBEDDING_MODEL").expect("set ORANGU_TEST_EMBEDDING_MODEL");
        let gguf = GgufFile::open(std::path::Path::new(&path)).expect("open model");
        let tok = Tokenizer::from_gguf(&gguf).expect("build tokenizer");
        assert_eq!(tok.vocab_kind, VocabKind::SpmUnigram);

        let ids = tok.encode("The quick brown fox jumps over the lazy dog", false);
        assert_eq!(
            ids,
            vec![818, 3823, 8864, 37423, 38167, 1024, 506, 31770, 4799]
        );
    }

    /// Cross-check against real llama.cpp's `/tokenize` for
    /// `mradermacher/Qwen3-VL-Embedding-8B-GGUF:Q4_K_M` — a plain
    /// `VocabKind::Gpt2Byte` vocab (`tokenizer.ggml.model = "gpt2"`,
    /// `.pre = "qwen2"`), already-supported shape, included here mainly to
    /// pin `add_bos_token = false` (confirmed directly against the real
    /// GGUF) — `Tokenizer::encode_for_embedding` must *not* prepend BOS
    /// for this model, unlike every other tested model. Run with
    /// `ORANGU_TEST_QWEN3VL_MODEL=/path/to/Qwen3-VL-Embedding-8B.Q4_K_M
    /// .gguf cargo test --release --bin orangu-server tokenizer --
    /// --ignored`.
    #[test]
    #[ignore]
    fn qwen3vl_tokenization_matches_real_llama_cpp() {
        let path =
            std::env::var("ORANGU_TEST_QWEN3VL_MODEL").expect("set ORANGU_TEST_QWEN3VL_MODEL");
        let gguf = GgufFile::open(std::path::Path::new(&path)).expect("open model");
        let tok = Tokenizer::from_gguf(&gguf).expect("build tokenizer");
        assert_eq!(tok.vocab_kind, VocabKind::Gpt2Byte);
        assert!(!tok.add_bos_token);
        assert!(tok.add_eos_token);

        let ids = tok.encode("The quick brown fox jumps over the lazy dog", false);
        assert_eq!(
            ids,
            vec![785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562]
        );

        let embedding_ids = tok.encode_for_embedding("The quick brown fox jumps over the lazy dog");
        assert_eq!(
            embedding_ids,
            vec![
                785,
                3974,
                13876,
                38835,
                34208,
                916,
                279,
                15678,
                5562,
                tok.eos_token.unwrap()
            ]
        );
    }
}
