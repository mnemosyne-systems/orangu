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

//! The prompt prefixes this model's clients keep reusing, remembered across
//! restarts and prefilled again in the background when the server comes up —
//! `[orangu-server].prefix_warmup`.
//!
//! Every `orangu` turn opens with the same ~1900 tokens — its system prompt
//! and tool definitions — and the paged KV pool shares those pages between
//! requests, so within one server's life only the first request pays for
//! them. That first request is the one a user notices: on the CIX P1 a
//! one-line prompt to a fresh server took 12.8 s where the same prompt once
//! the prefix is cached takes 4.3 (`doc/PERF-ALL.md`, task 4).
//!
//! So the server keeps, per model, the prefixes requests actually *reused*
//! — the cached span of a prompt the prefix cache served, [`MIN_TOKENS`] or
//! longer — with how often each was reused, and writes them as token ids to
//! `~/.orangu/server/<fingerprint>/warm-prefixes.json` whenever the set
//! changes. A few kilobytes, and independent of how the KV cache is laid
//! out: nothing but token ids is kept, and the fingerprint (architecture,
//! label, KV structure — the slot store's) ties them to this model. At the
//! next start each is prefilled once, most-reused first, as a prompt with no
//! answer, before or while the first requests arrive; its pages land in the
//! pool exactly as a client's would.
//!
//! What it costs: the warm-up occupies a slot for its prefill (~9 s for
//! 1900 tokens on this board), so a request arriving during it waits — never
//! longer than the same request would have prefilled cold. A prefix no
//! request reuses is never stored.

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::Instant;

use serde::{Deserialize, Serialize};

/// The shortest reused prefix worth remembering: shorter ones are cheap to
/// prefill again and would crowd out the long ones.
pub const MIN_TOKENS: usize = 256;

/// How many prefixes are kept — the most reused. A client's system prompt
/// and tools are one; a second client, or a second role's prompt, another.
const MAX_ENTRIES: usize = 4;

/// The longest prefix kept, in tokens: past this a prefix is a conversation,
/// not a client's fixed opening.
const MAX_TOKENS: usize = 16_384;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
struct Entry {
    /// How many requests reused this prefix.
    hits: u64,
    tokens: Vec<u32>,
}

#[derive(Debug, Serialize, Deserialize, Default)]
struct File {
    /// The model's fingerprint; a file under another model's is ignored.
    fingerprint: String,
    prefixes: Vec<Entry>,
}

struct State {
    path: PathBuf,
    file: File,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

/// Whether `[orangu-server].prefix_warmup` (or `ORANGU_PREFIX_WARMUP`, which
/// wins) leaves this on. Default on.
pub fn enabled(configured: bool) -> bool {
    match std::env::var("ORANGU_PREFIX_WARMUP") {
        Ok(v) => !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ),
        Err(_) => configured,
    }
}

/// Where this model's prefixes are kept.
pub fn path_for(fingerprint: &str) -> Option<PathBuf> {
    Some(
        home::home_dir()?
            .join(".orangu/server")
            .join(fingerprint)
            .join("warm-prefixes.json"),
    )
}

/// Opens the record for this model, and returns the prefixes to warm, most
/// reused first. A missing, unreadable or foreign file starts an empty one.
pub fn init(path: PathBuf, fingerprint: &str) -> Vec<Vec<u32>> {
    let file = load(&path, fingerprint);
    let prefixes = most_reused(&file);
    if let Ok(mut state) = STATE.lock() {
        *state = Some(State { path, file });
    }
    prefixes
}

/// The record at `path` when it is this model's, else an empty one.
fn load(path: &std::path::Path, fingerprint: &str) -> File {
    std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<File>(&bytes).ok())
        .filter(|f| f.fingerprint == fingerprint)
        .unwrap_or_else(|| File {
            fingerprint: fingerprint.to_string(),
            prefixes: Vec::new(),
        })
}

/// The kept prefixes, most reused first.
fn most_reused(file: &File) -> Vec<Vec<u32>> {
    let mut prefixes = file.prefixes.clone();
    prefixes.sort_by_key(|e| std::cmp::Reverse(e.hits));
    prefixes.into_iter().map(|e| e.tokens).collect()
}

/// A request reused `prefix` from the cache. Remembered when it is at least
/// [`MIN_TOKENS`] long; written out when the set of prefixes changes.
pub fn record(prefix: &[u32]) {
    if prefix.len() < MIN_TOKENS {
        return;
    }
    let prefix = &prefix[..prefix.len().min(MAX_TOKENS)];
    let Ok(mut guard) = STATE.lock() else {
        return;
    };
    let Some(state) = guard.as_mut() else {
        return;
    };
    if update(&mut state.file.prefixes, prefix) {
        save(state);
    }
}

/// [`record`]'s rule on the list alone: `true` when the kept set changed
/// (a prefix added, evicted or extended) rather than only a hit counted.
fn update(entries: &mut Vec<Entry>, prefix: &[u32]) -> bool {
    // The same opening seen again, possibly longer or shorter this time (a
    // page more or less of it cached): one entry, the longer span kept.
    if let Some(e) = entries
        .iter_mut()
        .find(|e| e.tokens.starts_with(prefix) || prefix.starts_with(&e.tokens))
    {
        e.hits += 1;
        if prefix.len() > e.tokens.len() {
            e.tokens = prefix.to_vec();
            return true;
        }
        return false;
    }
    entries.push(Entry {
        hits: 1,
        tokens: prefix.to_vec(),
    });
    if entries.len() > MAX_ENTRIES {
        entries.sort_by_key(|e| std::cmp::Reverse(e.hits));
        entries.truncate(MAX_ENTRIES);
    }
    true
}

/// Written through a temporary file and a rename, so a crash leaves the old
/// record or the new one, never half of either.
fn save(state: &State) {
    let Ok(json) = serde_json::to_vec(&state.file) else {
        return;
    };
    let write = || -> std::io::Result<()> {
        if let Some(dir) = state.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = state.path.with_extension("json.tmp");
        std::fs::write(&tmp, &json)?;
        std::fs::rename(&tmp, &state.path)
    };
    if let Err(err) = write() {
        log::warn!(
            "orangu-server: warm prefixes not saved to {}: {err}",
            state.path.display()
        );
    }
}

/// Prefills each of `prefixes` once through the engine, as a prompt with no
/// answer, and reports the result as an `[adapt]` line.
pub async fn warm_up(
    engine: std::sync::Arc<crate::engine::generate::Engine>,
    prefixes: Vec<Vec<u32>>,
) {
    if prefixes.is_empty() {
        crate::engine::adapt::note(
            "prefix warm-up",
            "nothing to warm",
            "no prefix has been reused on this model yet",
        );
        return;
    }
    let started = Instant::now();
    let mut warmed = 0usize;
    let mut tokens = 0usize;
    for prefix in prefixes {
        let n = prefix.len();
        let mut rx = engine
            .generate(crate::engine::generate::GenerateRequest {
                prompt_tokens: prefix,
                sampling: crate::engine::sampling::SamplingParams::default(),
                json_output: false,
                max_tokens: 0,
                stop_token_ids: Vec::new(),
                cache_prompt: true,
                id_slot: None,
                timings_per_token: false,
                role: None,
            })
            .await;
        let mut ok = false;
        while let Some(event) = rx.recv().await {
            match event {
                crate::engine::generate::StreamEvent::Done { .. } => {
                    ok = true;
                    break;
                }
                crate::engine::generate::StreamEvent::Error(_)
                | crate::engine::generate::StreamEvent::Overloaded => break,
                _ => {}
            }
        }
        if ok {
            warmed += 1;
            tokens += n;
        }
    }
    crate::engine::adapt::note(
        "prefix warm-up",
        format!(
            "{warmed} prefix(es), {tokens} tokens, prefilled in {:.1} s",
            started.elapsed().as_secs_f64()
        ),
        "reused by requests before the last restart (~/.orangu/server/<model>/warm-prefixes.json)",
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(n: usize, seed: u32) -> Vec<u32> {
        (0..n as u32)
            .map(|i| i.wrapping_mul(2654435761) ^ seed)
            .collect()
    }

    #[test]
    fn the_same_opening_is_one_entry_and_keeps_its_longest_span() {
        let mut entries = Vec::new();
        let long = tokens(1920, 7);
        assert!(update(&mut entries, &long[..1856]));
        // Seen again, a page longer: extended, still one entry.
        assert!(update(&mut entries, &long));
        // Seen again, a page shorter: counted, nothing to write.
        assert!(!update(&mut entries, &long[..1856]));
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].hits, 3);
        assert_eq!(entries[0].tokens.len(), 1920);
    }

    #[test]
    fn only_the_most_reused_are_kept() {
        let mut entries = Vec::new();
        for seed in 0..MAX_ENTRIES as u32 {
            let t = tokens(300, seed);
            for _ in 0..=seed {
                update(&mut entries, &t);
            }
        }
        // A new prefix seen once evicts the least reused (seed 0, one hit) —
        // or itself, tied with it; either way the most reused stay.
        update(&mut entries, &tokens(300, 99));
        assert_eq!(entries.len(), MAX_ENTRIES);
        let top = tokens(300, MAX_ENTRIES as u32 - 1);
        assert!(entries.iter().any(|e| e.tokens == top));
    }

    #[test]
    fn a_file_for_another_model_starts_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("warm-prefixes.json");
        std::fs::write(
            &path,
            serde_json::to_vec(&File {
                fingerprint: "other".into(),
                prefixes: vec![Entry {
                    hits: 3,
                    tokens: tokens(300, 1),
                }],
            })
            .unwrap(),
        )
        .unwrap();
        assert!(most_reused(&load(&path, "mine")).is_empty());
        std::fs::write(
            &path,
            serde_json::to_vec(&File {
                fingerprint: "mine".into(),
                prefixes: vec![Entry {
                    hits: 3,
                    tokens: tokens(300, 1),
                }],
            })
            .unwrap(),
        )
        .unwrap();
        assert_eq!(most_reused(&load(&path, "mine")), vec![tokens(300, 1)]);
    }
}
