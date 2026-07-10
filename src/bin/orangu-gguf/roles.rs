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

//! Interactive role wizard: launched when `orangu-gguf` is started with no
//! subcommand. Walks the same two choices `orangu.conf`/`orangu-coordinator.conf`
//! already ask of you by hand — which of the five conventional roles (`all`,
//! `code`, `review`, `explorer`, `embeddings`) and which downloaded model —
//! and turns them into a ready-to-run `llama-server` command line.
//!
//! The per-role flag sets in [`build_extra_args`] are not derived from
//! hardware heuristics — they mirror the hand-tuned, verified examples in
//! the manual's OpenAI-platform chapter (`doc/manual/en/73-openai.md`)
//! verbatim, which is the project's own canonical reference for running
//! llama.cpp well *with orangu specifically* (KV cache reuse, on-disk slot
//! persistence, etc.) If that chapter's examples change, this table should
//! change with them.

use crate::config::GgufConfiguration;
use crate::gguf::{GgufFile, GgufValue};
use crate::models::{self, ModelGroup};
use crate::prompt::prompt;
use crate::system::{self, CpuInfo};
use anyhow::{Context, Result};

/// The conventional roles `orangu.conf`/`orangu-coordinator.conf` document,
/// in the order they're listed there.
pub const ROLES: &[&str] = &["all", "code", "review", "explorer", "embeddings"];

struct RoleProfile {
    name: &'static str,
    /// Context size to request, before clamping to the model's own maximum
    /// (`<arch>.context_length`, when present).
    default_ctx_size: u64,
}

const ROLE_PROFILES: &[RoleProfile] = &[
    RoleProfile {
        name: "all",
        default_ctx_size: 262144,
    },
    RoleProfile {
        name: "code",
        default_ctx_size: 131072,
    },
    RoleProfile {
        name: "review",
        default_ctx_size: 262144,
    },
    RoleProfile {
        name: "explorer",
        default_ctx_size: 131072,
    },
    RoleProfile {
        name: "embeddings",
        default_ctx_size: 8192,
    },
];

/// Resolves a role the user typed: either its 1-based index into [`ROLES`]
/// (matching how `list`'s `NR` column works) or its name, case-insensitively.
fn resolve_role(input: &str) -> Option<&'static RoleProfile> {
    if let Ok(index) = input.parse::<usize>() {
        return index.checked_sub(1).and_then(|i| ROLE_PROFILES.get(i));
    }
    ROLE_PROFILES
        .iter()
        .find(|role| role.name.eq_ignore_ascii_case(input))
}

/// Resolves a model the user typed against an already-scanned group list:
/// either its 1-based `NR` or its `MODEL` label. Deliberately doesn't
/// re-scan the models directory (unlike [`models::resolve_show_target`]) —
/// the wizard already has `groups` in hand from listing them, and a second
/// full scan would cost real time against a large models directory for no
/// benefit.
fn find_group<'a>(groups: &'a [ModelGroup], input: &str) -> Option<&'a ModelGroup> {
    if let Ok(nr) = input.parse::<usize>() {
        return nr.checked_sub(1).and_then(|index| groups.get(index));
    }
    groups.iter().find(|group| group.label == input)
}

/// The model reference argument: `-hf <user>/<model>[:quant]` when `label`
/// is one (Hugging Face hub cache labels always contain a `/`, and no bare
/// GGUF filename ever does), otherwise `-m <path>` for a plain local file.
fn model_reference(label: &str, representative_path: &std::path::Path) -> String {
    if label.contains('/') {
        format!("-hf {label}")
    } else {
        format!("-m {}", representative_path.display())
    }
}

/// Reads `<arch>.<suffix>` from a GGUF file's metadata (e.g. `block_count`,
/// `context_length`, `pooling_type`) — the per-architecture hyperparameter
/// keys the GGUF spec namespaces under whatever `general.architecture` names.
fn architecture_metadata_u64(gguf: &GgufFile, suffix: &str) -> Option<u64> {
    let architecture = gguf.metadata.iter().find_map(|(key, value)| {
        (key == "general.architecture")
            .then_some(value)
            .and_then(|v| match v {
                GgufValue::String(s) => Some(s.as_str()),
                _ => None,
            })
    })?;
    let key = format!("{architecture}.{suffix}");
    gguf.metadata
        .iter()
        .find(|(k, _)| *k == key)
        .and_then(|(_, v)| v.as_u64())
}

/// Maps llama.cpp's `enum llama_pooling_type` (`include/llama.h`) to the
/// name its own `--pooling` flag expects.
fn pooling_type_name(value: u64) -> Option<&'static str> {
    match value {
        0 => Some("none"),
        1 => Some("mean"),
        2 => Some("cls"),
        3 => Some("last"),
        4 => Some("rank"),
        _ => None,
    }
}

/// A cheap textual approximation of whether the model's own chat template
/// (`tokenizer.chat_template`) supports llama-server's `--reasoning-preserve`
/// flag. llama.cpp's own probe (`jinja::caps_get`, `common/jinja/caps.cpp`)
/// actually *executes* the template against a synthetic conversation
/// carrying a reasoning trace and checks whether that trace survives in the
/// rendered output — reproducing that exactly would mean embedding a
/// Jinja-compatible template engine, out of proportion for this tool. That
/// probe's outcome only ever depends on whether the template references one
/// of three Jinja variables it sets beforehand (`caps_apply_preserve_reasoning`):
/// `preserve_thinking`, `clear_thinking`, `truncate_history_thinking`. A
/// template referencing none of them categorically can't honor the flag; one
/// that does is a strong, if not certain, signal that it does.
fn supports_reasoning_preserve(gguf: &GgufFile) -> bool {
    let Some(template) = gguf.metadata.iter().find_map(|(key, value)| {
        (key == "tokenizer.chat_template")
            .then_some(value)
            .and_then(|v| match v {
                GgufValue::String(s) => Some(s.as_str()),
                _ => None,
            })
    }) else {
        return false;
    };
    [
        "preserve_thinking",
        "clear_thinking",
        "truncate_history_thinking",
    ]
    .iter()
    .any(|needle| template.contains(needle))
}

/// The extra arguments for `role`, verbatim from `doc/manual/en/73-openai.md`'s
/// per-role example, substituting only `ctx_size` (clamped to the model's own
/// maximum by the caller), `threads` (the detected physical core count), and
/// — `embeddings` only — `pooling` (the model's own `pooling_type` metadata
/// when available; see [`build_command`]). `--port`, `-hf`/`-m`, and
/// `llama-server` itself are added separately by the caller — every role
/// example there uses a bare `--port 8100`.
fn build_extra_args(role: &str, ctx_size: u64, threads: usize, pooling: &str) -> Vec<String> {
    match role {
        "all" => vec![
            format!("--ctx-size {ctx_size}"),
            "-sm layer".to_string(),
            format!("-t {threads}"),
            "--webui-mcp-proxy".to_string(),
            "--fit on".to_string(),
            "--tools all".to_string(),
            "-b 2048".to_string(),
            "-ub 2048".to_string(),
            "--cache-reuse 256".to_string(),
            "--slot-save-path ~/.orangu/llama-slots".to_string(),
            "-fa on".to_string(),
            "-ctk q8_0".to_string(),
            "-ctv q8_0".to_string(),
        ],
        "code" => vec![
            format!("--ctx-size {ctx_size}"),
            format!("-t {threads}"),
            "--webui-mcp-proxy".to_string(),
            "--fit on".to_string(),
            "--image-min-tokens 1024".to_string(),
            "--tools all".to_string(),
            "-b 2048".to_string(),
            "-ub 2048".to_string(),
            "--cache-reuse 256".to_string(),
            "--slot-save-path ~/.orangu/llama-slots".to_string(),
            "-fa on".to_string(),
            "-ctk q8_0".to_string(),
            "-ctv q8_0".to_string(),
        ],
        "review" => vec![
            format!("--ctx-size {ctx_size}"),
            "-np 1".to_string(),
            "-fa on".to_string(),
            "-sm layer".to_string(),
            format!("-t {threads}"),
            "--webui-mcp-proxy".to_string(),
            "--fit on".to_string(),
            "--tools all".to_string(),
            "-b 2048".to_string(),
            "-ub 2048".to_string(),
            "--cache-reuse 256".to_string(),
            "--slot-save-path ~/.orangu/llama-slots".to_string(),
            "--reasoning-budget 0".to_string(),
            "--reasoning off".to_string(),
            "-ctk q8_0".to_string(),
            "-ctv q8_0".to_string(),
        ],
        "explorer" => vec![
            format!("--ctx-size {ctx_size}"),
            "-np 1".to_string(),
            "-fa on".to_string(),
            "-ctk q8_0".to_string(),
            "-ctv q8_0".to_string(),
            "-b 2048".to_string(),
            "-ub 2048".to_string(),
            "--cache-reuse 256".to_string(),
            "--slot-save-path ~/.orangu/llama-slots".to_string(),
            "--temp 0.7".to_string(),
            "--top-p 0.8".to_string(),
            "--top-k 20".to_string(),
            "--min-p 0".to_string(),
            "--jinja".to_string(),
            "--fit on".to_string(),
        ],
        "embeddings" => vec![
            "--embedding".to_string(),
            format!("--pooling {pooling}"),
            format!("--ctx-size {ctx_size}"),
            "-np 8".to_string(),
            "--kv-unified".to_string(),
            "-b 2048".to_string(),
            "-ub 2048".to_string(),
            "--fit on".to_string(),
        ],
        other => {
            unreachable!("resolve_role only ever returns a name from ROLE_PROFILES, got {other}")
        }
    }
}

struct RecommendedCommand {
    command: String,
    notes: Vec<String>,
}

fn build_command(
    role: &RoleProfile,
    label: &str,
    representative_path: &std::path::Path,
    gguf: &GgufFile,
    cpu: &CpuInfo,
    models_dir: &std::path::Path,
) -> RecommendedCommand {
    let context_length = architecture_metadata_u64(gguf, "context_length");
    let ctx_size = match context_length {
        Some(max) if max > 0 => role.default_ctx_size.min(max),
        _ => role.default_ctx_size,
    };
    let threads = cpu.physical_cores.unwrap_or(cpu.logical_cores);
    let pooling_metadata =
        architecture_metadata_u64(gguf, "pooling_type").and_then(pooling_type_name);
    // `mean` is the classical, most broadly-applicable pooling strategy for
    // sentence-embedding models, so it's the fallback when a model's own
    // metadata doesn't say — verified against a real GGUF where the
    // metadata *did* say (embeddinggemma-300M reports `pooling_type=mean`),
    // which is also the model this role's example was originally verified
    // against, so deriving from metadata rather than hard-coding either
    // value is both more correct and more general.
    let pooling = pooling_metadata.unwrap_or("mean");

    // `LLAMA_CACHE` is llama.cpp's own highest-priority override for where
    // `-hf` looks for (and downloads into) its Hugging Face hub cache —
    // pointing it at the configured `models` directory is what makes `-hf`
    // find a model `orangu-gguf download` already fetched there, instead of
    // llama.cpp falling back to its own default `~/.cache/huggingface/hub`.
    let mut parts = vec![
        format!("LLAMA_CACHE={}", models_dir.display()),
        "llama-server".to_string(),
        model_reference(label, representative_path),
        "--port 8100".to_string(),
    ];
    parts.extend(build_extra_args(role.name, ctx_size, threads, pooling));
    // Pointless (nothing to preserve) when the role's own flags already
    // disable reasoning outright, as `review`'s `--reasoning off` does.
    if supports_reasoning_preserve(gguf) && !parts.iter().any(|p| p == "--reasoning off") {
        parts.push("--reasoning-preserve".to_string());
    }

    let mut notes = Vec::new();
    if let Some(max) = context_length
        && ctx_size < role.default_ctx_size
    {
        notes.push(format!(
            "Context size capped at the model's own maximum ({max}), below the usual {} for {}.",
            role.default_ctx_size, role.name
        ));
    }
    if role.name == "embeddings" && pooling_metadata.is_none() {
        notes.push(format!(
            "No usable pooling_type metadata found on this model — defaulting to \
             --pooling {pooling}; check the model's card if results look off."
        ));
    }

    RecommendedCommand {
        command: parts.join(" "),
        notes,
    }
}

/// Runs the interactive wizard: prompts for a role, then a model (scanned
/// once up front — re-prompting on anything unrecognized rather than
/// aborting on the first bad entry), then prints a `llama-server` command
/// line tuned for that combination.
pub fn run_wizard(config: &GgufConfiguration) -> Result<()> {
    let summaries = models::scan_models_dir(&config.models)?;
    let groups = models::group_models(&summaries);
    if groups.is_empty() {
        anyhow::bail!(
            "No .gguf files found under {}; run 'orangu-gguf list' for details",
            config.models.display()
        );
    }

    println!("Roles");
    for (index, role) in ROLES.iter().enumerate() {
        println!("  {}  {role}", index + 1);
    }
    let role = loop {
        let input = prompt("\nSelect a role [1-5 or name]: ")?;
        match resolve_role(&input) {
            Some(role) => break role,
            None => println!(
                "'{input}' is not a known role; enter 1-5 or one of: {}.",
                ROLES.join(", ")
            ),
        }
    };

    println!();
    print!("{}", models::format_list(&summaries, &config.models));
    let group = loop {
        let input = prompt("\nSelect a model [NR or MODEL]: ")?;
        match find_group(&groups, &input) {
            Some(group) => break group,
            None => println!("'{input}' is not a listed NR or MODEL."),
        }
    };

    let gguf = GgufFile::open(&group.representative_path)
        .with_context(|| format!("failed to re-read {}", group.representative_path.display()))?;
    let cpu = system::detect_cpu();
    let recommended = build_command(
        role,
        &group.label,
        &group.representative_path,
        &gguf,
        &cpu,
        &config.models,
    );

    println!("\nRecommended command for role '{}':\n", role.name);
    println!("  {}\n", recommended.command);
    for note in &recommended.notes {
        println!("  - {note}");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolve_role_accepts_index_or_name_case_insensitively() {
        assert_eq!(resolve_role("1").map(|r| r.name), Some("all"));
        assert_eq!(resolve_role("5").map(|r| r.name), Some("embeddings"));
        assert_eq!(resolve_role("Review").map(|r| r.name), Some("review"));
        assert!(resolve_role("0").is_none());
        assert!(resolve_role("6").is_none());
        assert!(resolve_role("not-a-role").is_none());
    }

    #[test]
    fn model_reference_prefers_hf_when_the_label_names_a_repo() {
        assert_eq!(
            model_reference(
                "unsloth/Qwen3-Coder-Next-GGUF:Q4_K_M",
                std::path::Path::new("/x")
            ),
            "-hf unsloth/Qwen3-Coder-Next-GGUF:Q4_K_M"
        );
        assert_eq!(
            model_reference(
                "model-Q4_K_M",
                std::path::Path::new("/models/model-Q4_K_M.gguf")
            ),
            "-m /models/model-Q4_K_M.gguf"
        );
    }

    #[test]
    fn find_group_resolves_by_nr_or_label() {
        let groups = vec![
            ModelGroup {
                label: "org/a:Q4_K_M".to_string(),
                size_bytes: 1,
                quantization: None,
                errors: Vec::new(),
                representative_path: "/a.gguf".into(),
            },
            ModelGroup {
                label: "org/b:Q4_K_M".to_string(),
                size_bytes: 1,
                quantization: None,
                errors: Vec::new(),
                representative_path: "/b.gguf".into(),
            },
        ];
        assert_eq!(
            find_group(&groups, "1").map(|g| g.label.as_str()),
            Some("org/a:Q4_K_M")
        );
        assert_eq!(
            find_group(&groups, "org/b:Q4_K_M").map(|g| g.label.as_str()),
            Some("org/b:Q4_K_M")
        );
        assert!(find_group(&groups, "3").is_none());
        assert!(find_group(&groups, "nope").is_none());
    }

    #[test]
    fn pooling_type_name_matches_llama_pooling_type_enum() {
        assert_eq!(pooling_type_name(0), Some("none"));
        assert_eq!(pooling_type_name(1), Some("mean"));
        assert_eq!(pooling_type_name(2), Some("cls"));
        assert_eq!(pooling_type_name(3), Some("last"));
        assert_eq!(pooling_type_name(4), Some("rank"));
        assert_eq!(pooling_type_name(5), None);
    }

    #[test]
    fn supports_reasoning_preserve_detects_referenced_jinja_variables() {
        let with_clear_thinking = test_gguf(vec![(
            "tokenizer.chat_template",
            GgufValue::String("{% if not clear_thinking %}...{% endif %}".to_string()),
        )]);
        assert!(supports_reasoning_preserve(&with_clear_thinking));

        let with_preserve_thinking = test_gguf(vec![(
            "tokenizer.chat_template",
            GgufValue::String("{% if preserve_thinking %}...{% endif %}".to_string()),
        )]);
        assert!(supports_reasoning_preserve(&with_preserve_thinking));

        let unrelated_template = test_gguf(vec![(
            "tokenizer.chat_template",
            GgufValue::String("{{ messages[0].content }}".to_string()),
        )]);
        assert!(!supports_reasoning_preserve(&unrelated_template));

        let no_template = test_gguf(vec![]);
        assert!(!supports_reasoning_preserve(&no_template));
    }

    /// Each of these locks in `build_extra_args`'s output against
    /// `doc/manual/en/73-openai.md`'s own per-role example, verbatim (aside
    /// from the substituted `ctx_size`/`threads`) — if that chapter's
    /// examples change, this test should fail and prompt updating both.
    #[test]
    fn build_extra_args_matches_the_manuals_all_role_example() {
        assert_eq!(
            build_extra_args("all", 262144, 4, "").join(" "),
            "--ctx-size 262144 -sm layer -t 4 --webui-mcp-proxy --fit on --tools all \
             -b 2048 -ub 2048 --cache-reuse 256 --slot-save-path ~/.orangu/llama-slots \
             -fa on -ctk q8_0 -ctv q8_0"
        );
    }

    #[test]
    fn build_extra_args_matches_the_manuals_code_role_example() {
        assert_eq!(
            build_extra_args("code", 131072, 4, "").join(" "),
            "--ctx-size 131072 -t 4 --webui-mcp-proxy --fit on --image-min-tokens 1024 \
             --tools all -b 2048 -ub 2048 --cache-reuse 256 \
             --slot-save-path ~/.orangu/llama-slots -fa on -ctk q8_0 -ctv q8_0"
        );
    }

    #[test]
    fn build_extra_args_matches_the_manuals_review_role_example() {
        assert_eq!(
            build_extra_args("review", 262144, 4, "").join(" "),
            "--ctx-size 262144 -np 1 -fa on -sm layer -t 4 --webui-mcp-proxy --fit on \
             --tools all -b 2048 -ub 2048 --cache-reuse 256 \
             --slot-save-path ~/.orangu/llama-slots --reasoning-budget 0 --reasoning off \
             -ctk q8_0 -ctv q8_0"
        );
    }

    #[test]
    fn build_extra_args_matches_the_manuals_explorer_role_example() {
        assert_eq!(
            build_extra_args("explorer", 131072, 4, "").join(" "),
            "--ctx-size 131072 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 2048 \
             --cache-reuse 256 --slot-save-path ~/.orangu/llama-slots --temp 0.7 \
             --top-p 0.8 --top-k 20 --min-p 0 --jinja --fit on"
        );
    }

    #[test]
    fn build_extra_args_matches_the_manuals_embeddings_role_example() {
        assert_eq!(
            build_extra_args("embeddings", 8192, 4, "mean").join(" "),
            "--embedding --pooling mean --ctx-size 8192 -np 8 --kv-unified -b 2048 -ub 2048 --fit on"
        );
    }

    fn test_gguf(metadata: Vec<(&str, GgufValue)>) -> GgufFile {
        GgufFile {
            version: 3,
            metadata: metadata
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
            tensors: Vec::new(),
            alignment: 32,
            data_offset: 0,
        }
    }

    fn test_cpu(physical_cores: Option<usize>) -> CpuInfo {
        CpuInfo {
            brand: "Test".to_string(),
            vendor: String::new(),
            arch: "x86_64".to_string(),
            physical_cores,
            logical_cores: physical_cores.unwrap_or(4),
            frequency_mhz: 0,
            total_memory_bytes: 0,
            available_memory_bytes: 0,
        }
    }

    #[test]
    fn build_command_uses_pooling_from_model_metadata() {
        // Verified against the real embeddinggemma-300M GGUF: its own
        // metadata reports pooling_type=1 (mean), even though the manual's
        // embeddings example previously (incorrectly) recommended `last`.
        let gguf = test_gguf(vec![
            (
                "general.architecture",
                GgufValue::String("gemma-embedding".to_string()),
            ),
            ("gemma-embedding.pooling_type", GgufValue::U32(1)),
            ("gemma-embedding.context_length", GgufValue::U32(2048)),
        ]);
        let role = ROLE_PROFILES
            .iter()
            .find(|r| r.name == "embeddings")
            .unwrap();
        let cpu = test_cpu(Some(8));

        let recommended = build_command(
            role,
            "ggml-org/embeddinggemma-300M-GGUF:Q8_0",
            std::path::Path::new("/x"),
            &gguf,
            &cpu,
            std::path::Path::new("/models"),
        );

        assert!(recommended.command.contains("LLAMA_CACHE=/models"));
        assert!(recommended.command.contains("--pooling mean"));
        assert!(recommended.command.contains("--kv-unified"));
        assert!(!recommended.notes.iter().any(|n| n.contains("pooling_type")));
    }

    #[test]
    fn build_command_falls_back_to_mean_pooling_without_metadata() {
        let gguf = test_gguf(vec![(
            "general.architecture",
            GgufValue::String("unknown".to_string()),
        )]);
        let role = ROLE_PROFILES
            .iter()
            .find(|r| r.name == "embeddings")
            .unwrap();
        let cpu = test_cpu(Some(8));

        let recommended = build_command(
            role,
            "org/model:Q8_0",
            std::path::Path::new("/x"),
            &gguf,
            &cpu,
            std::path::Path::new("/models"),
        );

        assert!(recommended.command.contains("--pooling mean"));
        assert!(
            recommended
                .notes
                .iter()
                .any(|n| n.contains("No usable pooling_type"))
        );
    }

    #[test]
    fn build_command_adds_reasoning_preserve_when_supported() {
        let gguf = test_gguf(vec![(
            "tokenizer.chat_template",
            GgufValue::String("{% if not clear_thinking %}...{% endif %}".to_string()),
        )]);
        let role = ROLE_PROFILES.iter().find(|r| r.name == "code").unwrap();
        let cpu = test_cpu(Some(8));

        let recommended = build_command(
            role,
            "org/model",
            std::path::Path::new("/x"),
            &gguf,
            &cpu,
            std::path::Path::new("/models"),
        );

        assert!(recommended.command.contains("--reasoning-preserve"));
    }

    #[test]
    fn build_command_omits_reasoning_preserve_without_support() {
        let gguf = test_gguf(vec![]);
        let role = ROLE_PROFILES.iter().find(|r| r.name == "code").unwrap();
        let cpu = test_cpu(Some(8));

        let recommended = build_command(
            role,
            "org/model",
            std::path::Path::new("/x"),
            &gguf,
            &cpu,
            std::path::Path::new("/models"),
        );

        assert!(!recommended.command.contains("--reasoning-preserve"));
    }

    #[test]
    fn build_command_omits_reasoning_preserve_when_role_disables_reasoning() {
        // `review`'s own flags include `--reasoning off`; adding
        // `--reasoning-preserve` on top would be pointless (nothing to
        // preserve) and read as contradictory.
        let gguf = test_gguf(vec![(
            "tokenizer.chat_template",
            GgufValue::String("{% if not clear_thinking %}...{% endif %}".to_string()),
        )]);
        let role = ROLE_PROFILES.iter().find(|r| r.name == "review").unwrap();
        let cpu = test_cpu(Some(8));

        let recommended = build_command(
            role,
            "org/model",
            std::path::Path::new("/x"),
            &gguf,
            &cpu,
            std::path::Path::new("/models"),
        );

        assert!(!recommended.command.contains("--reasoning-preserve"));
    }
}
