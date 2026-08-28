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

//! The tokenizer: a byte-level BPE vocabulary trained on the corpus, in the
//! exact shape a `"gpt2"`-model GGUF carries it — a `tokens` array in the
//! byte-to-unicode alphabet, and a `merges` array of space-joined pairs in
//! rank order.
//!
//! Two properties are load-bearing, and both are about the *reader* of the
//! file rather than this trainer:
//!
//! - **The pre-tokenizer split must be the one the inference side applies.**
//!   The file declares `tokenizer.ggml.pre = "gpt-2"`, whose split is the
//!   generic pattern in [`SPLIT_PATTERN`]; training on any other split would
//!   produce merges that the encoder can never reproduce, so every prompt
//!   would tokenize into pieces the model was never trained on. The pattern
//!   here is that pattern, character for character.
//! - **Every byte has a token.** The 256 base symbols come first, so there
//!   is no input this vocabulary cannot represent and no need for an unknown
//!   token.
//!
//! The specials sit at the end. `<|endoftext|>` is real — the packer writes
//! one between documents, so the model is trained to emit it and it works as
//! a stop token. The ChatML pair is *reserved*: present in the vocabulary so
//! a later instruction-tuning run has ids to train, and typed as control
//! tokens so they never appear in output, but a base model has no use for
//! them and this tool writes no chat template claiming otherwise.

use anyhow::{Context, Result, bail};
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use std::{cmp::Reverse, collections::BinaryHeap, fs, path::Path};

/// The generic GPT-2-shaped pre-tokenizer split, character for character
/// what a `"gpt2"` vocabulary with an unrecognised `tokenizer.ggml.pre` is
/// read back with.
pub const SPLIT_PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+";

/// What the split pattern's character classes are, for one ASCII byte.
///
/// `\p{L}` and `\p{N}` over ASCII are exactly the letters and the digits;
/// `\s` is `White_Space`, which over ASCII is `\t\n\v\f\r` and the space
/// — note the vertical tab, which `u8::is_ascii_whitespace` leaves out.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Letter,
    Number,
    Space,
    Other,
}

const fn ascii_classes() -> [Class; 128] {
    let mut table = [Class::Other; 128];
    let mut b = 0usize;
    while b < 128 {
        table[b] = if (b >= 'a' as usize && b <= 'z' as usize)
            || (b >= 'A' as usize && b <= 'Z' as usize)
        {
            Class::Letter
        } else if b >= '0' as usize && b <= '9' as usize {
            Class::Number
        } else if (b >= 0x09 && b <= 0x0D) || b == 0x20 {
            Class::Space
        } else {
            Class::Other
        };
        b += 1;
    }
    table
}

const ASCII_CLASS: [Class; 128] = ascii_classes();

/// The contraction alternatives, in the order [`SPLIT_PATTERN`] lists them.
/// The order is the whole of their meaning: the pattern is matched
/// leftmost-*first*, so the first branch that matches wins whether or not a
/// later one would match more.
const CONTRACTIONS: [&[u8]; 7] = [b"s", b"t", b"re", b"ve", b"m", b"ll", b"d"];

/// Applies [`SPLIT_PATTERN`] to text — by hand while the text is ASCII, and
/// by the pattern itself the moment it is not.
///
/// The pattern is one fixed expression run over every byte of the corpus,
/// twice: once training the vocabulary and once packing it. A DFA is a
/// general machine doing a specific job, and the specific job is small
/// enough to write out: four character classes and five alternatives.
///
/// What it must not do is disagree. The classes are only decidable by hand
/// for ASCII — outside it, `\p{L}` is a Unicode table and not a range — so
/// the scanner does not guess: the instant a match could involve a
/// non-ASCII byte it hands that one match back to the pattern, which is the
/// authority, and picks up after it. Text that is entirely ASCII never
/// touches the DFA; text that is not pays for exactly the matches that
/// touch it.
pub struct Splitter {
    pattern: regex::Regex,
}

impl Splitter {
    pub fn new() -> Result<Self> {
        Ok(Splitter {
            pattern: regex::Regex::new(SPLIT_PATTERN).context("compiling the split pattern")?,
        })
    }

    /// The pre-tokens of `text`, in order, covering every byte of it.
    pub fn split<'s, 't>(&'s self, text: &'t str) -> Pieces<'s, 't> {
        Pieces {
            splitter: self,
            text,
            at: 0,
        }
    }

    /// The compiled pattern, for the differential test that keeps the
    /// scanner honest.
    #[cfg(test)]
    pub fn pattern(&self) -> &regex::Regex {
        &self.pattern
    }

    /// The end of the match starting at `at`, or `None` when deciding it
    /// would mean classifying a non-ASCII character.
    fn ascii_match(text: &[u8], at: usize) -> Option<usize> {
        let first = *text.get(at)?;
        if first >= 0x80 {
            return None;
        }

        // `'s`, `'t`, `'re`, `'ve`, `'m`, `'ll`, `'d` — before everything,
        // because the pattern lists them before everything.
        if first == b'\'' {
            for tail in CONTRACTIONS {
                if text[at + 1..].starts_with(tail) {
                    return Some(at + 1 + tail.len());
                }
            }
        }

        // The three ` ?`-prefixed alternatives share a shape: an optional
        // single space, then a run of one class.
        let (start, leading_space) = if first == b' ' {
            (at + 1, true)
        } else {
            (at, false)
        };
        let after_space = text.get(start).copied();
        if leading_space {
            match after_space {
                // A space that begins a non-ASCII character: whether the
                // run continues is the pattern's business.
                Some(b) if b >= 0x80 => return None,
                // A space followed by anything but a space starts one of
                // the three; a space followed by a space is `\s+`, and a
                // trailing space is `\s+` too.
                Some(b) if ASCII_CLASS[b as usize] != Class::Space => {}
                _ => return run(text, at, Class::Space),
            }
        }

        let class = match after_space {
            Some(b) if b < 0x80 => ASCII_CLASS[b as usize],
            // Only reachable without a leading space, since that case
            // returned above.
            _ => return None,
        };
        run(text, start, class)
    }
}

/// Extends a run of one class from `from`, returning where it ends, or
/// `None` if it runs into a byte this cannot classify.
fn run(text: &[u8], from: usize, class: Class) -> Option<usize> {
    let mut at = from;
    while at < text.len() {
        let b = text[at];
        if b >= 0x80 {
            return None;
        }
        if ASCII_CLASS[b as usize] != class {
            break;
        }
        at += 1;
    }
    Some(at)
}

pub struct Pieces<'s, 't> {
    splitter: &'s Splitter,
    text: &'t str,
    at: usize,
}

impl<'t> Iterator for Pieces<'_, 't> {
    type Item = &'t str;

    fn next(&mut self) -> Option<&'t str> {
        if self.at >= self.text.len() {
            return None;
        }
        let bytes = self.text.as_bytes();
        if let Some(end) = Splitter::ascii_match(bytes, self.at) {
            let piece = &self.text[self.at..end];
            self.at = end;
            return Some(piece);
        }
        // Every character matches one alternative or another, so the
        // pattern always matches at `at` — never past it.
        let m = self.splitter.pattern.find_at(self.text, self.at)?;
        debug_assert_eq!(m.start(), self.at);
        self.at = m.end();
        Some(m.as_str())
    }
}

/// The `tokenizer.ggml.pre` this vocabulary is written with.
pub const PRE_TYPE: &str = "gpt-2";

/// End of document: written between documents by the packer, and the
/// model's stop token.
pub const END_OF_TEXT: &str = "<|endoftext|>";
/// Reserved for a later instruction-tuning stage — see the module comment.
pub const CHATML: [&str; 2] = ["<|im_start|>", "<|im_end|>"];

/// The ChatML template, written into the file as `tokenizer.chat_template`.
///
/// A base model has no idea what a conversation is, and this does not give
/// it one. What it gives is a file that every chat client can *talk to* —
/// without a template, a chat endpoint has no way to turn messages into a
/// prompt and refuses the request outright, which makes a freshly trained
/// model look broken when it is merely untuned. The tokens it names are in
/// the vocabulary from the start so that instruction tuning changes what the
/// model does with them, never the format of the file.
pub const CHAT_TEMPLATE: &str = concat!(
    "{% for message in messages %}",
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}",
    "{% endfor %}",
    "{% if add_generation_prompt %}",
    "{{ '<|im_start|>assistant\n' }}",
    "{% endif %}"
);

/// Upstream's `llama_token_type`: a normal vocabulary entry, and a control
/// token that is never produced from raw text.
const TOKEN_TYPE_NORMAL: i32 = 1;
const TOKEN_TYPE_CONTROL: i32 = 3;

/// Unique pre-tokens kept when counting the training sample. Beyond this
/// the rarest are dropped: the tail of a code corpus is base64 blobs and
/// minified identifiers, which cost memory in the merge loop and earn no
/// merges.
const MAX_UNIQUE_WORDS: usize = 4_000_000;

/// A trained vocabulary, in the form the model file carries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vocab {
    /// Token strings, indexed by id, in the byte-to-unicode alphabet.
    pub tokens: Vec<String>,
    /// `"<left> <right>"` per merge, in rank order.
    pub merges: Vec<String>,
    /// Per-token `tokenizer.ggml.token_type`.
    pub token_type: Vec<i32>,
    pub bos: u32,
    pub eos: u32,
}

/// The lookup tables [`Vocab::encode`] needs, built once.
pub struct Encoder<'a> {
    vocab: &'a Vocab,
    /// `(left, right) -> (rank, merged id)`.
    ranks: FxHashMap<(u32, u32), (u32, u32)>,
    split: Splitter,
    /// Byte to its token id, resolved once. Encoding a corpus is a byte at
    /// a time; doing it through a `char`, a UTF-8 buffer and a hash lookup
    /// per byte is a table lookup wearing three disguises.
    byte_to_id: [u32; 256],
    /// The inverse alphabet, for [`Encoder::decode`]. Built here rather
    /// than per call.
    char_to_byte: FxHashMap<char, u8>,
}

impl Vocab {
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// The id of a token by its text. Asked only for the specials, at export
    /// time, so a scan of the table is the whole implementation.
    pub fn id_of(&self, text: &str) -> Option<u32> {
        self.tokens.iter().position(|t| t == text).map(|i| i as u32)
    }

    pub fn encoder(&self) -> Result<Encoder<'_>> {
        let mut token_to_id: FxHashMap<&str, u32> =
            FxHashMap::with_capacity_and_hasher(self.tokens.len(), Default::default());
        for (id, token) in self.tokens.iter().enumerate() {
            token_to_id.insert(token.as_str(), id as u32);
        }
        let mut ranks: FxHashMap<(u32, u32), (u32, u32)> =
            FxHashMap::with_capacity_and_hasher(self.merges.len(), Default::default());
        for (rank, merge) in self.merges.iter().enumerate() {
            let (left, right) = merge
                .split_once(' ')
                .ok_or_else(|| anyhow::anyhow!("merge {rank} is not a pair: {merge:?}"))?;
            let (Some(&l), Some(&r)) = (token_to_id.get(left), token_to_id.get(right)) else {
                bail!("merge {rank} names a token that is not in the vocabulary: {merge:?}");
            };
            let joined = format!("{left}{right}");
            let Some(&merged) = token_to_id.get(joined.as_str()) else {
                bail!("merge {rank} produces {joined:?}, which is not in the vocabulary");
            };
            ranks.insert((l, r), (rank as u32, merged));
        }
        // Every byte has a token, so each of these resolves; a vocabulary
        // where one does not is not usable and says so here rather than
        // silently dropping bytes on the first document.
        let byte_to_char = byte_to_char_table();
        let mut byte_to_id = [0u32; 256];
        let mut char_to_byte: FxHashMap<char, u8> =
            FxHashMap::with_capacity_and_hasher(256, Default::default());
        let mut buffer = [0u8; 4];
        for (b, &ch) in byte_to_char.iter().enumerate() {
            let key = ch.encode_utf8(&mut buffer);
            let Some(&id) = token_to_id.get(key) else {
                bail!("the vocabulary has no token for byte {b}");
            };
            byte_to_id[b] = id;
            char_to_byte.insert(ch, b as u8);
        }

        Ok(Encoder {
            vocab: self,
            ranks,
            split: Splitter::new()?,
            byte_to_id,
            char_to_byte,
        })
    }

    /// Writes the vocabulary out, through a temporary file, for the same
    /// reason the packed tokens go that way: a later run reuses what is
    /// there, and half a tokenizer is worse than none.
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string(self)?;
        let temporary = path.with_extension("tmp");
        fs::write(&temporary, json).with_context(|| format!("writing {}", temporary.display()))?;
        fs::rename(&temporary, path)
            .with_context(|| format!("renaming {} into place", temporary.display()))?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        Ok(serde_json::from_str(&text)?)
    }
}

impl Encoder<'_> {
    /// Text to token ids, by the same route the inference side takes:
    /// pre-tokenize, map bytes into the alphabet, then merge by rank,
    /// lowest rank first and leftmost on ties.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3 + 1);
        let mut symbols: Vec<u32> = Vec::with_capacity(RESCAN_MAX);
        let mut merger = Merger::default();
        for piece in self.split.split(text) {
            let bytes = piece.as_bytes();
            if bytes.len() <= RESCAN_MAX {
                symbols.clear();
                symbols.extend(bytes.iter().map(|&b| self.byte_to_id[b as usize]));
                self.merge_by_rescanning(&mut symbols);
                out.extend_from_slice(&symbols);
            } else {
                merger.load(bytes, &self.byte_to_id);
                merger.merge(&self.ranks);
                merger.emit(&mut out);
            }
        }
        out
    }

    /// Merges by rescanning for the lowest-ranked adjacent pair. Quadratic
    /// in the pre-token's length, and the fastest thing there is at the
    /// length nearly every pre-token has: an array scan and a hash lookup
    /// per pair, against [`Merger`]'s heap traffic.
    fn merge_by_rescanning(&self, symbols: &mut Vec<u32>) {
        loop {
            let mut best: Option<(u32, usize, u32)> = None;
            for i in 0..symbols.len().saturating_sub(1) {
                if let Some(&(rank, merged)) = self.ranks.get(&(symbols[i], symbols[i + 1]))
                    && best.is_none_or(|(best_rank, _, _)| rank < best_rank)
                {
                    best = Some((rank, i, merged));
                }
            }
            let Some((_, at, merged)) = best else { return };
            symbols[at] = merged;
            symbols.remove(at + 1);
        }
    }

    pub fn eos(&self) -> u32 {
        self.vocab.eos
    }

    /// Token ids back to text — the inverse alphabet mapping, for the
    /// round-trip check that proves the vocabulary is usable.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(token) = self.vocab.tokens.get(id as usize) else {
                continue;
            };
            if self.vocab.token_type[id as usize] == TOKEN_TYPE_CONTROL {
                continue;
            }
            for ch in token.chars() {
                if let Some(&b) = self.char_to_byte.get(&ch) {
                    bytes.push(b);
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Pre-tokens up to this many bytes are merged by rescanning; longer ones
/// go through [`Merger`].
///
/// Measured, not guessed. Rescanning is quadratic and the heap is not, but
/// the heap costs a push and a pop per pair where the scan costs a compare,
/// and nearly every pre-token in a corpus is a word — short enough that the
/// quadratic one wins outright. The threshold is where the two curves
/// cross, and the point of having both is that neither the common case nor
/// the long tail pays for the other.
const RESCAN_MAX: usize = 24;

/// One pre-token's symbols as a doubly-linked list, with a heap of the
/// merges still available on it.
///
/// This is the path for the pre-tokens that are not words. A base64 blob, a
/// minified line, a URL: one of them is longer than a thousand words put
/// together, and quadratic in that length costs more on its own than the
/// thousand words do. A corpus has plenty of them.
///
/// So: the list gives O(1) removal, the heap gives the next merge without a
/// scan, and a merged node keeps the count of original symbols underneath
/// it so a heap entry that a later merge invalidated can be recognised and
/// dropped rather than kept up to date. The buffers are reused across every
/// pre-token in a document.
/// A merge the heap still holds: `(rank, left, right, merged id, and the
/// two nodes' combined width when it was pushed)`. The rank leads, so the
/// heap pops in merge-table order; the left index breaks ties leftmost, the
/// way rescanning did; and the width is what makes a stale entry
/// recognisable.
type Available = (u32, u32, u32, u32, u32);

#[derive(Default)]
struct Merger {
    /// Token id per node; a merged-away node keeps its id but zero units.
    id: Vec<u32>,
    prev: Vec<i32>,
    next: Vec<i32>,
    /// Original symbols under this node, and 0 once it has been merged
    /// away. This is what makes a stale heap entry recognisable.
    units: Vec<u32>,
    heap: BinaryHeap<Reverse<Available>>,
}

impl Merger {
    fn load(&mut self, bytes: &[u8], byte_to_id: &[u32; 256]) {
        self.id.clear();
        self.prev.clear();
        self.next.clear();
        self.units.clear();
        self.heap.clear();

        self.id
            .extend(bytes.iter().map(|&b| byte_to_id[b as usize]));
        let n = self.id.len();
        self.prev.extend((0..n).map(|i| i as i32 - 1));
        self.next
            .extend((0..n).map(|i| if i + 1 < n { i as i32 + 1 } else { -1 }));
        self.units.resize(n, 1);
    }

    fn push_bigram(&mut self, ranks: &FxHashMap<(u32, u32), (u32, u32)>, l: usize, r: usize) {
        if let Some(&(rank, merged)) = ranks.get(&(self.id[l], self.id[r])) {
            let size = self.units[l] + self.units[r];
            self.heap
                .push(Reverse((rank, l as u32, r as u32, merged, size)));
        }
    }

    /// Merges lowest rank first, leftmost on ties — the same order the
    /// rescanning version produced, and the order the merge table was
    /// trained in.
    fn merge(&mut self, ranks: &FxHashMap<(u32, u32), (u32, u32)>) {
        for l in 0..self.id.len().saturating_sub(1) {
            self.push_bigram(ranks, l, l + 1);
        }
        while let Some(Reverse((_rank, l, r, merged, size))) = self.heap.pop() {
            let (l, r) = (l as usize, r as usize);
            // Stale: either end already merged away, they are no longer
            // neighbours, or one of them has grown since this was pushed.
            if self.units[l] == 0
                || self.units[r] == 0
                || self.next[l] != r as i32
                || self.units[l] + self.units[r] != size
            {
                continue;
            }
            self.id[l] = merged;
            self.units[l] += self.units[r];
            self.units[r] = 0;
            self.next[l] = self.next[r];
            if self.next[r] >= 0 {
                self.prev[self.next[r] as usize] = l as i32;
            }
            if self.prev[l] >= 0 {
                let p = self.prev[l] as usize;
                self.push_bigram(ranks, p, l);
            }
            if self.next[l] >= 0 {
                let nx = self.next[l] as usize;
                self.push_bigram(ranks, l, nx);
            }
        }
    }

    /// Walks the surviving list from the head. Node 0 is always the head:
    /// the leftmost symbol is never the right-hand side of a merge.
    fn emit(&self, out: &mut Vec<u32>) {
        let mut at = if self.id.is_empty() { -1 } else { 0 };
        while at >= 0 {
            out.push(self.id[at as usize]);
            at = self.next[at as usize];
        }
    }
}

/// The byte-to-unicode alphabet: printable bytes stand for themselves, and
/// the 68 that do not (control characters, space, the C1 range) are lifted
/// into the private range starting at `U+0100` so every token string is
/// printable and round-trips through a JSON metadata array.
pub fn byte_to_char_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut assigned = [false; 256];
    for b in 0u32..256 {
        let direct =
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        if direct {
            table[b as usize] = char::from_u32(b).unwrap();
            assigned[b as usize] = true;
        }
    }
    let mut n = 0u32;
    for b in 0..256 {
        if !assigned[b] {
            table[b] = char::from_u32(256 + n).unwrap();
            n += 1;
        }
    }
    table
}

/// Trains a byte-level BPE vocabulary of `vocab_size` tokens (specials
/// included) over `documents`.
///
/// `progress` is called with the number of merges learned so far, at a
/// coarse interval, so a caller can show that a long run is moving.
pub fn train(
    documents: impl Iterator<Item = String>,
    vocab_size: usize,
    progress: &dyn Fn(usize, usize),
) -> Result<Vocab> {
    let specials: Vec<&str> = std::iter::once(END_OF_TEXT).chain(CHATML).collect();
    let base = 256 + specials.len();
    if vocab_size < base + 1 {
        bail!("a vocabulary of {vocab_size} has no room for 256 byte tokens and the specials");
    }

    let byte_to_char = byte_to_char_table();
    let split = Splitter::new()?;

    // Pass one: how often each pre-token occurs. Working in *strings* here
    // rather than symbol vectors keeps the table small — the corpus is
    // mostly the same few hundred thousand words over and over.
    let mut counts: FxHashMap<Vec<u8>, u64> = FxHashMap::default();
    for doc in documents {
        for piece in split.split(&doc) {
            bump(&mut counts, piece.as_bytes());
        }
        if counts.len() > MAX_UNIQUE_WORDS * 2 {
            prune(&mut counts, MAX_UNIQUE_WORDS);
        }
    }
    if counts.len() > MAX_UNIQUE_WORDS {
        prune(&mut counts, MAX_UNIQUE_WORDS);
    }
    if counts.is_empty() {
        bail!("the corpus produced no text to train a tokenizer on");
    }

    // Every byte is a token; ids 0..256 are those, in byte order.
    let mut tokens: Vec<String> = (0..256).map(|b| byte_to_char[b].to_string()).collect();

    let mut words: Vec<Vec<u32>> = Vec::with_capacity(counts.len());
    let mut word_counts: Vec<u64> = Vec::with_capacity(counts.len());
    for (word, count) in counts {
        words.push(word.iter().map(|&b| b as u32).collect());
        word_counts.push(count);
    }

    // Pair statistics, kept incrementally: how often each adjacent pair
    // occurs, and which words to revisit when it is merged.
    let mut pair_counts: FxHashMap<(u32, u32), i64> = FxHashMap::default();
    let mut pair_words: FxHashMap<(u32, u32), FxHashSet<u32>> = FxHashMap::default();
    for (w, symbols) in words.iter().enumerate() {
        for pair in symbols.windows(2) {
            let key = (pair[0], pair[1]);
            *pair_counts.entry(key).or_insert(0) += word_counts[w] as i64;
            pair_words.entry(key).or_default().insert(w as u32);
        }
    }

    let mut heap: BinaryHeap<(i64, Reverse<(u32, u32)>)> = pair_counts
        .iter()
        .map(|(&pair, &count)| (count, Reverse(pair)))
        .collect();

    let target_merges = vocab_size - base;
    let mut merges: Vec<String> = Vec::with_capacity(target_merges);

    while merges.len() < target_merges {
        // The heap holds stale entries by design — a pair's count changes
        // as other merges consume it, and rewriting the heap on every
        // change would cost more than re-pushing the current value here.
        let (count, Reverse(pair)) = loop {
            let Some(top) = heap.pop() else {
                break (0, Reverse((0, 0)));
            };
            match pair_counts.get(&top.1.0) {
                Some(&current) if current == top.0 && current > 0 => break top,
                _ => continue,
            }
        };
        if count <= 0 {
            // Nothing left to merge: the corpus is exhausted before the
            // requested vocabulary size. Better a smaller honest vocabulary
            // than padding it with tokens nothing produced.
            break;
        }

        let merged_id = tokens.len() as u32;
        let left = tokens[pair.0 as usize].clone();
        let right = tokens[pair.1 as usize].clone();
        tokens.push(format!("{left}{right}"));
        merges.push(format!("{left} {right}"));

        let affected: Vec<u32> = pair_words
            .get(&pair)
            .map(|set| set.iter().copied().collect())
            .unwrap_or_default();
        pair_words.remove(&pair);
        pair_counts.remove(&pair);

        let mut touched: FxHashSet<(u32, u32)> = FxHashSet::default();
        for w in affected {
            let idx = w as usize;
            if !contains_pair(&words[idx], pair) {
                continue;
            }
            let count = word_counts[idx] as i64;
            // Subtract the word's whole pair multiset, rewrite it, then add
            // the new one back. Doing it wholesale is what keeps a repeated
            // symbol (`aaa` merging `aa`) correct without a special case.
            for p in words[idx].windows(2) {
                let key = (p[0], p[1]);
                *pair_counts.entry(key).or_insert(0) -= count;
                touched.insert(key);
            }
            let mut rewritten = Vec::with_capacity(words[idx].len());
            let mut i = 0;
            while i < words[idx].len() {
                if i + 1 < words[idx].len()
                    && words[idx][i] == pair.0
                    && words[idx][i + 1] == pair.1
                {
                    rewritten.push(merged_id);
                    i += 2;
                } else {
                    rewritten.push(words[idx][i]);
                    i += 1;
                }
            }
            words[idx] = rewritten;
            for p in words[idx].windows(2) {
                let key = (p[0], p[1]);
                *pair_counts.entry(key).or_insert(0) += count;
                pair_words.entry(key).or_default().insert(w);
                touched.insert(key);
            }
        }
        for key in touched {
            if let Some(&current) = pair_counts.get(&key)
                && current > 0
            {
                heap.push((current, Reverse(key)));
            }
        }

        if merges.len().is_multiple_of(512) {
            progress(merges.len(), target_merges);
        }
    }
    progress(merges.len(), target_merges);

    let mut token_type = vec![TOKEN_TYPE_NORMAL; tokens.len()];
    let eot = tokens.len() as u32;
    for special in &specials {
        tokens.push((*special).to_string());
        token_type.push(TOKEN_TYPE_CONTROL);
    }

    Ok(Vocab {
        tokens,
        merges,
        token_type,
        bos: eot,
        eos: eot,
    })
}

/// Adds one occurrence of `word` to the pre-token counts.
///
/// The obvious `entry(word.to_vec())` allocates on every occurrence and
/// throws the allocation away on all but the first — and a corpus is the
/// same few hundred thousand words over and over, so nearly every
/// occurrence is a repeat. Looking first costs a second hash on the rare
/// miss and no allocation at all on the common hit.
#[inline]
fn bump(counts: &mut FxHashMap<Vec<u8>, u64>, word: &[u8]) {
    if let Some(count) = counts.get_mut(word) {
        *count += 1;
    } else {
        counts.insert(word.to_vec(), 1);
    }
}

/// Keeps the `keep` most frequent entries, dropping the tail.
fn prune(counts: &mut FxHashMap<Vec<u8>, u64>, keep: usize) {
    let mut frequencies: Vec<u64> = counts.values().copied().collect();
    if frequencies.len() <= keep {
        return;
    }
    frequencies.sort_unstable_by(|a, b| b.cmp(a));
    let cutoff = frequencies[keep];
    counts.retain(|_, &mut count| count > cutoff);
}

fn contains_pair(symbols: &[u32], pair: (u32, u32)) -> bool {
    symbols.windows(2).any(|w| w[0] == pair.0 && w[1] == pair.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trained(text: &str, size: usize) -> Vocab {
        train(std::iter::once(text.to_string()), size, &|_, _| {}).unwrap()
    }

    /// Merges every pre-token through [`Merger`], whatever its length —
    /// the path `encode` only takes for the long ones. Comparing the two
    /// on the same input is the only thing that keeps the threshold from
    /// being a place where the tokenizer quietly changes its answer.
    fn merge_all_through_the_heap(encoder: &Encoder, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut merger = Merger::default();
        for piece in encoder.split.split(text) {
            merger.load(piece.as_bytes(), &encoder.byte_to_id);
            merger.merge(&encoder.ranks);
            merger.emit(&mut out);
        }
        out
    }

    /// The linked-list merge has to produce the same ids as the rescanning
    /// one on every input, not just on words. The inputs that matter are
    /// the ones a corpus is full of and a hand-written test is not: long
    /// runs of one symbol, repeated pairs, and text long enough that the
    /// quadratic version is the slow one.
    #[test]
    fn both_merge_paths_produce_the_same_tokens() {
        let corpus = "fn main() { let mut total = 0; for value in values { total += value; } }\n\
             the quick brown fox jumps over the lazy dog, and the dog does not care\n\
             aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\n\
             ababababababababababababababababababab\n\
             https://example.com/a/very/long/path?with=query&and=more#fragment\n\
             SGVsbG8gd29ybGQsIHRoaXMgaXMgYSBiYXNlNjQgYmxvYiB0aGF0IGtlZXBzIGdvaW5n";
        let vocab = trained(corpus, 1200);
        let encoder = vocab.encoder().unwrap();

        let mut compared = 0;
        for line in corpus.lines() {
            let mixed = encoder.encode(line);
            let heaped = merge_all_through_the_heap(&encoder, line);
            assert_eq!(mixed, heaped, "disagreed on {line:?}");
            compared += mixed.len();
        }
        assert!(compared > 60, "only compared {compared} tokens");
    }

    /// A pre-token long enough that rescanning is visibly quadratic still
    /// has to come out right — and the repeated symbol is the case that
    /// breaks a merger which forgets that a node grew.
    #[test]
    fn a_long_run_of_one_symbol_merges_correctly() {
        let vocab = trained(&"a".repeat(4096), 400);
        let encoder = vocab.encoder().unwrap();
        let ids = encoder.encode(&"a".repeat(1000));
        assert_eq!(encoder.decode(&ids), "a".repeat(1000));

        assert_eq!(ids, merge_all_through_the_heap(&encoder, &"a".repeat(1000)));
    }

    /// Text that exercises every alternative of the split pattern and
    /// every way the scanner can meet a non-ASCII byte: inside a word, at
    /// the start of one, inside a symbol run, as whitespace, and as a
    /// character whose class ASCII cannot decide.
    const SPLIT_CASES: &str = "\
        fn main() { let x = 1; }\n\
        don't can't we're I've I'm you'll he'd 'x '' '\n\
        double  spaces\tand\ttabs\r\n\x0bvertical\x0cform\n\
        1234 5 007 x1 1x a1b2\n\
        !!! ??? ...:::  ---> <<>>\n\
        café naïve Straße résumé\n\
        \u{a0}nbsp\u{2009}thin\u{3000}ideographic\n\
        日本語のテキスト、句読点つき\n\
        Ω=Ω α+β \u{0301}combining\n\
        emoji 🙂🙂 and 🇩🇰 flags\n\
        mixed café1 2naïve x\u{0345}y\n\
        \u{5d0}\u{5b0}\u{5d1} hebrew points\n\
        trailing space ";

    /// The scanner and the pattern must produce the same pieces, byte for
    /// byte, on every input — a disagreement is not a slower tokenizer, it
    /// is a different one, and every prompt would then split into pieces
    /// the model was never trained on.
    #[test]
    fn the_scanner_splits_exactly_as_the_pattern_does() {
        let splitter = Splitter::new().unwrap();
        let mut compared = 0;
        for text in [
            SPLIT_CASES,
            "",
            " ",
            "'",
            "a",
            "\u{e9}",
            " \u{e9}",
            "'\u{e9}",
        ] {
            let scanned: Vec<&str> = splitter.split(text).collect();
            let matched: Vec<&str> = splitter
                .pattern()
                .find_iter(text)
                .map(|m| m.as_str())
                .collect();
            assert_eq!(scanned, matched, "disagreed on {text:?}");
            // Whatever it splits into, it must still be the whole input.
            assert_eq!(scanned.concat(), text);
            compared += scanned.len();
        }
        assert!(compared > 100, "only compared {compared} pieces");
    }

    /// The same comparison over whatever corpus is on this machine, rather
    /// than over text chosen by the person who wrote the scanner. Ignored
    /// by default because it needs a corpus; the point of it is to be run
    /// after any change to the scanner.
    ///
    ///   cargo test --release --bin orangu-gguf the_scanner_agrees_over_a_real_corpus -- --ignored --nocapture
    #[test]
    #[ignore]
    fn the_scanner_agrees_over_a_real_corpus() {
        let root = match std::env::var_os("HOME") {
            Some(home) => std::path::PathBuf::from(home).join(".orangu/gguf"),
            None => return,
        };
        let splitter = Splitter::new().unwrap();
        let mut files = 0usize;
        let mut bytes = 0usize;
        let mut pieces = 0usize;

        let mut stack = vec![root];
        while let Some(directory) = stack.pop() {
            let Ok(entries) = std::fs::read_dir(&directory) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                    continue;
                }
                if files >= 400 {
                    break;
                }
                let Ok(text) = std::fs::read_to_string(&path) else {
                    continue;
                };
                let text = &text[..text.len().min(1 << 20)];
                let scanned: Vec<&str> = splitter.split(text).collect();
                let matched: Vec<&str> = splitter
                    .pattern()
                    .find_iter(text)
                    .map(|m| m.as_str())
                    .collect();
                assert_eq!(scanned, matched, "disagreed on {}", path.display());
                files += 1;
                bytes += text.len();
                pieces += scanned.len();
            }
        }
        println!("{files} files, {bytes} bytes, {pieces} pieces, no disagreement");
        assert!(files > 0, "no corpus under ~/.orangu/gguf to check against");
    }

    /// Every position in the corpus is also a starting position, and the
    /// scanner has to be right at all of them — including the ones a whole
    /// document scan would never begin a match at.
    #[test]
    fn the_scanner_agrees_from_every_offset() {
        let splitter = Splitter::new().unwrap();
        for (at, _) in SPLIT_CASES.char_indices() {
            let tail = &SPLIT_CASES[at..];
            let scanned: Vec<&str> = splitter.split(tail).take(4).collect();
            let matched: Vec<&str> = splitter
                .pattern()
                .find_iter(tail)
                .take(4)
                .map(|m| m.as_str())
                .collect();
            assert_eq!(scanned, matched, "disagreed starting at byte {at}");
        }
    }

    #[test]
    fn the_alphabet_is_a_bijection_over_all_256_bytes() {
        let table = byte_to_char_table();
        let unique: FxHashSet<char> = table.iter().copied().collect();
        assert_eq!(unique.len(), 256);
        assert_eq!(table[b'a' as usize], 'a');
        assert_eq!(table[b' ' as usize], 'Ġ');
    }

    #[test]
    fn every_byte_has_a_token_and_the_specials_come_last() {
        let vocab = trained("hello hello world", 300);
        assert_eq!(vocab.tokens[0].chars().count(), 1);
        assert_eq!(vocab.tokens[vocab.tokens.len() - 3], END_OF_TEXT);
        assert_eq!(vocab.tokens[vocab.eos as usize], END_OF_TEXT);
        assert_eq!(vocab.token_type[vocab.eos as usize], TOKEN_TYPE_CONTROL);
    }

    /// The merges have to actually compress: a repeated word must end up as
    /// fewer tokens than it has bytes.
    #[test]
    fn frequent_words_become_single_tokens() {
        let text = "orangu orangu orangu orangu orangu orangu";
        let vocab = trained(text, 400);
        let encoder = vocab.encoder().unwrap();
        let ids = encoder.encode("orangu");
        assert_eq!(ids.len(), 1, "{:?}", ids);
    }

    /// Every merge must name tokens the vocabulary has, in rank order —
    /// this is what `Vocab::encoder` validates, and a reader of the file
    /// applies the same rule.
    #[test]
    fn the_merge_table_is_self_consistent() {
        let vocab = trained("fn main() { println!(\"hi\"); } fn main() {}", 512);
        assert!(vocab.encoder().is_ok());
    }

    #[test]
    fn encoding_round_trips_through_decoding() {
        let text = "fn main() {\n    let x = 1;\n}\n";
        let vocab = trained(text, 512);
        let encoder = vocab.encoder().unwrap();
        assert_eq!(encoder.decode(&encoder.encode(text)), text);
    }

    /// Bytes that never appeared in training still encode, because the base
    /// alphabet covers all 256 of them.
    #[test]
    fn unseen_bytes_still_encode() {
        let vocab = trained("aaaa bbbb", 300);
        let encoder = vocab.encoder().unwrap();
        let text = "\u{1F600} \u{00e9}";
        assert_eq!(encoder.decode(&encoder.encode(text)), text);
    }

    #[test]
    fn a_vocabulary_smaller_than_the_alphabet_is_refused() {
        assert!(train(std::iter::once("x".to_string()), 100, &|_, _| {}).is_err());
    }

    /// A corpus with nothing left to merge stops early rather than padding
    /// the vocabulary with tokens no text produced.
    #[test]
    fn a_tiny_corpus_stops_short_of_the_target() {
        let vocab = trained("ab", 4096);
        assert!(vocab.len() < 4096);
        assert!(vocab.encoder().is_ok());
    }
}
