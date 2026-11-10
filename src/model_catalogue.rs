// Copyright (C) 2026 The orangu community
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

//! A small curated catalogue of models this engine serves: each with the
//! quantization to fetch, the context it is sized for, the roles it suits,
//! and the drafter that pairs with it where there is one.
//!
//! `orangu-server suggest` lists the entries that fit the machine, each at
//! the largest context that fits up to its own, and
//! `orangu-server download` takes an entry's quantization when the request
//! names none (and its drafter with `--with-drafter`). The sizes are the
//! files' own, so a fit is weighed against what will actually be read.

/// A model that drafts for another — see [`CatalogueEntry::drafter`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Drafter {
    /// A smaller model of the same vocabulary, as `<user>/<model>:<quant>`
    /// — `[orangu-server].draft_model`.
    Model(&'static str),
    /// A multi-token-prediction head shipped in the entry's own repo, by its
    /// path there; the server attaches it from beside the model.
    MtpHead(&'static str),
}

/// One model of the catalogue.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CatalogueEntry {
    /// The Hugging Face repo, `<user>/<model>`.
    pub repo: &'static str,
    /// The quantization tag to fetch.
    pub quant: &'static str,
    /// The quantization's file size, all shards, in bytes.
    pub file_bytes: u64,
    /// The bytes that are not routed experts — the whole file for a dense
    /// model. What a card has to hold; the experts run from host memory.
    pub dense_bytes: u64,
    /// The KV cache a token takes on the card (`f16`), from the model's
    /// attention layers; `0` where it is estimated from the parameters.
    pub kv_bytes_per_token: u64,
    /// Parameters, in billions — what the context's memory is estimated
    /// from.
    pub params_billion: f64,
    /// The context, in tokens, it is sized for here.
    pub context: u64,
    /// The roles it suits, in `orangu.conf`'s names.
    pub roles: &'static [&'static str],
    /// The drafter that pairs with it, where there is one the server uses by
    /// default (a head it leaves off is not paired). Prompt lookup drafts
    /// for every model without being fetched.
    pub drafter: Option<Drafter>,
}

impl CatalogueEntry {
    /// `<user>/<model>:<quant>`, as `download` and `orangu.conf` take it.
    pub fn spec(&self) -> String {
        format!("{}:{}", self.repo, self.quant)
    }
}

const GB: u64 = 1_000_000_000;

/// The catalogue, smallest file first.
pub const CATALOGUE: &[CatalogueEntry] = &[
    CatalogueEntry {
        repo: "ggml-org/embeddinggemma-300M-GGUF",
        quant: "Q8_0",
        file_bytes: 333_590_944,
        dense_bytes: 333_590_944,
        kv_bytes_per_token: 24_576,
        params_billion: 0.3,
        context: 2048,
        roles: &["embeddings"],
        drafter: None,
    },
    CatalogueEntry {
        repo: "unsloth/Qwen3.5-0.8B-GGUF",
        quant: "Q4_K_M",
        file_bytes: 532_517_120,
        dense_bytes: 532_517_120,
        kv_bytes_per_token: 0,
        params_billion: 0.8,
        context: 32768,
        roles: &["explorer"],
        drafter: None,
    },
    CatalogueEntry {
        repo: "Qwen/Qwen3-Embedding-0.6B-GGUF",
        quant: "Q8_0",
        file_bytes: 639_150_592,
        dense_bytes: 639_150_592,
        kv_bytes_per_token: 114_688,
        params_billion: 0.6,
        context: 8192,
        roles: &["embeddings"],
        drafter: None,
    },
    CatalogueEntry {
        repo: "unsloth/Qwen3-1.7B-GGUF",
        quant: "Q4_K_M",
        file_bytes: 1_107_409_472,
        dense_bytes: 1_107_409_472,
        kv_bytes_per_token: 114_688,
        params_billion: 1.7,
        context: 32768,
        roles: &["explorer"],
        drafter: Some(Drafter::Model("unsloth/Qwen3-0.6B-GGUF:Q4_K_M")),
    },
    CatalogueEntry {
        repo: "bartowski/Llama-3.2-3B-Instruct-GGUF",
        quant: "Q4_K_M",
        file_bytes: 2_019_377_696,
        dense_bytes: 2_019_377_696,
        kv_bytes_per_token: 114_688,
        params_billion: 3.2,
        context: 32768,
        roles: &["all", "explorer"],
        drafter: Some(Drafter::Model(
            "bartowski/Llama-3.2-1B-Instruct-GGUF:Q4_K_M",
        )),
    },
    CatalogueEntry {
        repo: "unsloth/Qwen3-4B-GGUF",
        quant: "Q4_K_M",
        file_bytes: 2_497_281_312,
        dense_bytes: 2_497_281_312,
        kv_bytes_per_token: 147_456,
        params_billion: 4.0,
        context: 32768,
        roles: &["all", "code"],
        drafter: Some(Drafter::Model("unsloth/Qwen3-0.6B-GGUF:Q4_K_M")),
    },
    CatalogueEntry {
        repo: "unsloth/gemma-4-E2B-it-GGUF",
        quant: "Q4_K_M",
        file_bytes: 3_106_738_272,
        dense_bytes: 3_106_738_272,
        kv_bytes_per_token: 18_432,
        params_billion: 5.1,
        context: 32768,
        roles: &["all", "code", "review"],
        drafter: None,
    },
    CatalogueEntry {
        repo: "unsloth/gemma-4-E4B-it-GGUF",
        quant: "Q4_K_M",
        file_bytes: 4_977_171_584,
        dense_bytes: 4_977_171_584,
        kv_bytes_per_token: 57_344,
        params_billion: 8.0,
        context: 32768,
        roles: &["all", "code", "review"],
        drafter: None,
    },
    CatalogueEntry {
        repo: "bartowski/Ornith-1.5-9B-GGUF",
        quant: "Q4_K_M",
        file_bytes: 5_910_782_848,
        dense_bytes: 5_910_782_848,
        kv_bytes_per_token: 131_072,
        params_billion: 9.0,
        context: 32768,
        roles: &["code", "review"],
        drafter: None,
    },
    CatalogueEntry {
        repo: "unsloth/gemma-4-12b-it-GGUF",
        quant: "Q4_K_M",
        file_bytes: 7_121_861_440,
        dense_bytes: 7_121_861_440,
        kv_bytes_per_token: 0,
        params_billion: 12.0,
        context: 32768,
        roles: &["all", "code", "review"],
        drafter: None,
    },
    CatalogueEntry {
        repo: "unsloth/gemma-4-26B-A4B-it-GGUF",
        quant: "UD-Q4_K_XL",
        file_bytes: 17_010_980_576,
        dense_bytes: 2_578_676_856,
        kv_bytes_per_token: 225_280,
        params_billion: 26.0,
        context: 32768,
        roles: &["all", "code", "review"],
        drafter: None,
    },
    CatalogueEntry {
        repo: "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
        quant: "Q4_K_M",
        file_bytes: 18_556_689_568,
        dense_bytes: 997_554_176,
        kv_bytes_per_token: 98_304,
        params_billion: 30.5,
        context: 65536,
        roles: &["code"],
        drafter: None,
    },
    CatalogueEntry {
        repo: "unsloth/Qwen3.6-35B-A3B-GGUF",
        quant: "UD-Q4_K_M",
        file_bytes: 22_134_528_992,
        dense_bytes: 2_555_013_632,
        kv_bytes_per_token: 81_920,
        params_billion: 35.0,
        context: 65536,
        roles: &["all", "code", "review"],
        drafter: None,
    },
];

/// The catalogue's entry for `repo` (`<user>/<model>`, any case), if any.
pub fn entry_for(repo: &str) -> Option<&'static CatalogueEntry> {
    CATALOGUE
        .iter()
        .find(|e| e.repo.eq_ignore_ascii_case(repo.trim()))
}

/// `spec` with the catalogue's quantization added when it names a
/// catalogue repo and no quantization of its own; otherwise as given.
pub fn with_catalogue_quant(spec: &str) -> String {
    if spec.contains(':') {
        return spec.to_string();
    }
    entry_for(spec).map_or_else(|| spec.to_string(), CatalogueEntry::spec)
}

/// The size as `suggest` prints it: `2.0 GB`.
pub fn format_size(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / GB as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every entry is a repo with a quantization, a role `orangu.conf`
    /// knows, and a drafter that is not itself; the list is smallest first,
    /// and names each repo once.
    #[test]
    fn every_entry_is_well_formed() {
        const ROLES: &[&str] = &["all", "code", "review", "explorer", "embeddings"];
        let mut seen = std::collections::HashSet::new();
        for e in CATALOGUE {
            assert_eq!(e.repo.matches('/').count(), 1, "{}", e.repo);
            assert!(!e.quant.is_empty() && !e.quant.contains(':'), "{}", e.repo);
            assert!(e.file_bytes > 0 && e.params_billion > 0.0 && e.context >= 1024);
            assert!(
                e.dense_bytes > 0 && e.dense_bytes <= e.file_bytes,
                "{}",
                e.repo
            );
            assert!(
                !e.roles.is_empty() && e.roles.iter().all(|r| ROLES.contains(r)),
                "{}",
                e.repo
            );
            assert!(seen.insert(e.repo.to_ascii_lowercase()), "{} twice", e.repo);
            match e.drafter {
                Some(Drafter::Model(d)) => {
                    assert!(d.contains(':') && !d.starts_with(e.repo), "{}", e.repo)
                }
                Some(Drafter::MtpHead(path)) => {
                    assert!(path.ends_with(".gguf"), "{}", e.repo)
                }
                None => {}
            }
        }
        assert!(
            CATALOGUE
                .windows(2)
                .all(|w| w[0].file_bytes <= w[1].file_bytes),
            "the catalogue is smallest first"
        );
    }

    #[test]
    fn a_catalogue_repo_takes_its_quantization_unless_one_is_named() {
        assert_eq!(
            with_catalogue_quant("unsloth/Qwen3.6-35B-A3B-GGUF"),
            "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M"
        );
        assert_eq!(
            with_catalogue_quant("UNSLOTH/qwen3.6-35b-a3b-gguf"),
            "unsloth/Qwen3.6-35B-A3B-GGUF:UD-Q4_K_M"
        );
        assert_eq!(
            with_catalogue_quant("unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0"),
            "unsloth/Qwen3.6-35B-A3B-GGUF:Q8_0"
        );
        assert_eq!(
            with_catalogue_quant("someone/else-GGUF"),
            "someone/else-GGUF"
        );
    }
}
