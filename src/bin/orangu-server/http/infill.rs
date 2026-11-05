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

//! `POST /infill`: code completion at a cursor, for a model whose vocabulary
//! has fill-in-the-middle tokens.
//!
//! The prompt is the file around the cursor, framed by those tokens, with other
//! files in front of it when the client sent any:
//!
//! ```text
//! <repo>project\n<sep>other.rs\n…text…<sep>this.rs\n<pre>before<suf>after<mid>
//! ```
//!
//! The text is the user's, and is encoded without reading special tokens in it:
//! a file that mentions `<|fim_middle|>` must not be able to end its own prefix.

use axum::{
    Json,
    extract::State,
    http::{HeaderMap, StatusCode},
    response::IntoResponse,
};
use serde::Deserialize;
use std::sync::Arc;

use super::AppState;
use crate::engine::generate::GenerateRequest;
use crate::engine::text_stop::{InfillRule, StopRules};
use crate::engine::tokenizer::{FimTokens, Tokenizer};

/// How many tokens around the cursor a prompt holds. An editor asks for a
/// completion on nearly every keystroke, so this bounds what each one costs
/// whatever the file's size. The code nearest the cursor is what predicts the
/// completion.
const CURSOR_WINDOW: usize = 2048;

/// What stands between other files' snippets, for a vocabulary with no token
/// to name a file with.
const SNIPPET_RULE: &str = "\n\n--- snippet ---\n\n";

/// Another file, for context.
#[derive(Deserialize, Clone, Debug)]
pub struct ExtraChunk {
    #[serde(default)]
    pub filename: String,
    pub text: String,
}

#[derive(Deserialize)]
pub struct InfillRequest {
    input_prefix: String,
    input_suffix: String,
    /// Text the user typed after the prefix, which the completion continues.
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    input_extra: Vec<ExtraChunk>,
    /// The cursor's file, and the repository's name, in front of the cursor
    /// when `input_extra` puts other files there.
    #[serde(default)]
    filename: Option<String>,
    #[serde(default)]
    repo: Option<String>,
    /// A line indented less than this ends the completion. Taken from the
    /// cursor's own line when absent; `0` turns the test off.
    #[serde(default)]
    n_indent: Option<usize>,
    #[serde(default = "super::native::default_n_predict")]
    n_predict: usize,
    #[serde(default = "super::openai::default_cache_prompt")]
    cache_prompt: bool,
    /// How many of the first tokens a context shift keeps.
    #[serde(default)]
    n_keep: Option<usize>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    min_p: Option<f32>,
    #[serde(default)]
    repeat_penalty: Option<f32>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    stop: Option<super::openai::StopField>,
    #[serde(default)]
    stream: bool,
    #[serde(default)]
    id_slot: Option<usize>,
}

/// The text a prompt is built from.
struct Cursor<'a> {
    prefix: &'a str,
    suffix: &'a str,
    typed: &'a str,
    extra: &'a [ExtraChunk],
    filename: &'a str,
    repo: &'a str,
}

/// The other files' tokens, in the most compact form the vocabulary has.
fn extra_tokens(tokenizer: &Tokenizer, fim: &FimTokens, cursor: &Cursor<'_>) -> Vec<u32> {
    let mut tokens = Vec::new();
    if cursor.extra.is_empty() {
        return tokens;
    }
    if let Some(repo) = fim.repo {
        tokens.push(repo);
        tokens.extend(tokenizer.encode_literal(&format!("{}\n", cursor.repo)));
    }
    for chunk in cursor.extra {
        match fim.separator {
            Some(separator) => {
                tokens.push(separator);
                tokens.extend(tokenizer.encode_literal(&format!("{}\n", chunk.filename)));
            }
            None => tokens.extend(tokenizer.encode_literal(SNIPPET_RULE)),
        }
        tokens.extend(tokenizer.encode_literal(&chunk.text));
    }
    if let Some(separator) = fim.separator {
        tokens.push(separator);
        tokens.extend(tokenizer.encode_literal(&format!("{}\n", cursor.filename)));
    }
    tokens
}

/// `tokens` limited to its last `keep`.
fn tail(tokens: &[u32], keep: usize) -> &[u32] {
    &tokens[tokens.len().saturating_sub(keep)..]
}

/// The prompt's tokens. Three quarters of [`CURSOR_WINDOW`] goes to what stands
/// before the cursor, which predicts more, and what either side does not use the
/// other may; a side too long keeps the part nearest the cursor. Other files
/// take what the window, the reply and the same again for a safety margin leave
/// of the context, most recent last.
fn build_prompt(
    tokenizer: &Tokenizer,
    fim: &FimTokens,
    cursor: &Cursor<'_>,
    context: usize,
    n_predict: usize,
) -> Vec<u32> {
    let before = tokenizer.encode_literal(cursor.prefix);
    let typed = tokenizer.encode_literal(cursor.typed);
    let after = tokenizer.encode_literal(cursor.suffix);
    let framing = 3 + usize::from(tokenizer.wants_bos());
    let window = CURSOR_WINDOW
        .min(context.saturating_sub(n_predict))
        .saturating_sub(framing + typed.len());
    let before_share = window - window / 4;
    let after_take = after.len().min(window - before_share.min(before.len()));
    let before_take = before.len().min(window - after_take);
    let extra = extra_tokens(tokenizer, fim, cursor);
    let extra_room = context.saturating_sub(CURSOR_WINDOW + 2 * n_predict);

    let mut tokens = Vec::with_capacity(framing + before_take + typed.len() + after_take);
    if tokenizer.wants_bos()
        && let Some(bos) = tokenizer.bos_token
    {
        tokens.push(bos);
    }
    tokens.extend_from_slice(tail(&extra, extra_room));
    tokens.push(fim.prefix);
    tokens.extend_from_slice(tail(&before, before_take));
    tokens.extend_from_slice(&typed);
    tokens.push(fim.suffix);
    tokens.extend_from_slice(&after[..after_take]);
    tokens.push(fim.middle);
    tokens
}

/// The stop tokens of a completion: the model's own, and the tokens that open
/// another part of the frame, which mean it has finished the middle.
fn stop_tokens(tokenizer: &Tokenizer, fim: &FimTokens) -> Vec<u32> {
    let mut ids = tokenizer.stop_token_ids();
    for id in [fim.pad, fim.repo, fim.separator].into_iter().flatten() {
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    ids
}

pub async fn infill(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<InfillRequest>,
) -> axum::response::Response {
    if let Some(rejection) = super::audio::reject_text_endpoint(&state, "/infill") {
        return rejection;
    }
    if !state.engine.role.allows_generation() {
        return (
            StatusCode::NOT_IMPLEMENTED,
            format!(
                "this server is running in --{} mode; generation endpoints are disabled",
                state.engine.role.label()
            ),
        )
            .into_response();
    }
    let Some(fim) = state.engine.tokenizer.fim_tokens() else {
        return (
            StatusCode::NOT_IMPLEMENTED,
            "this model's vocabulary has no fill-in-the-middle tokens (prefix, suffix and \
             middle), so it cannot complete at a cursor; use /completion or \
             /v1/chat/completions",
        )
            .into_response();
    };
    if let Some(rejection) = super::openai::reject_unknown_slot(&state, req.id_slot) {
        return rejection;
    }
    let context = crate::engine::generate::kv_device_token_ceiling()
        .map_or(state.engine.model.config().n_ctx_train, |ceiling| {
            ceiling.min(state.engine.model.config().n_ctx_train)
        });
    let n_predict = req.n_predict;
    let role = super::openai::request_role(&state, &headers);
    let sampling = super::openai::sampling_for(
        role,
        super::openai::RequestSampling {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            min_p: req.min_p,
            repeat_penalty: req.repeat_penalty,
            seed: req.seed,
            ..Default::default()
        },
    );
    let rule = InfillRule::at_cursor(
        &format!("{}{}", req.input_prefix, req.prompt),
        &req.input_suffix,
        req.n_indent,
    );
    let stop = StopRules {
        strings: super::openai::StopField::strings(req.stop),
        infill: Some(rule),
    };
    // Tokenizing a whole file is not work for the async runtime.
    let tokenizer = state.engine.tokenizer.clone();
    let (input_prefix, input_suffix, typed, extra, filename, repo) = (
        req.input_prefix,
        req.input_suffix,
        req.prompt,
        req.input_extra,
        req.filename.unwrap_or_else(|| "file".to_string()),
        req.repo.unwrap_or_else(|| "project".to_string()),
    );
    let tokens = tokio::task::spawn_blocking(move || {
        build_prompt(
            &tokenizer,
            &fim,
            &Cursor {
                prefix: &input_prefix,
                suffix: &input_suffix,
                typed: &typed,
                extra: &extra,
                filename: &filename,
                repo: &repo,
            },
            context,
            n_predict,
        )
    })
    .await;
    let tokens = match tokens {
        Ok(tokens) => tokens,
        Err(err) => return (StatusCode::INTERNAL_SERVER_ERROR, err.to_string()).into_response(),
    };
    if let Some(rejection) = super::reject_oversized_context(&state, tokens.len()) {
        return rejection;
    }
    let stop_token_ids = stop_tokens(&state.engine.tokenizer, &fim);
    super::native::serve_generation(
        &state,
        GenerateRequest {
            prompt_tokens: tokens,
            sampling,
            max_tokens: n_predict,
            stop_token_ids,
            cache_prompt: req.cache_prompt,
            n_keep: req.n_keep,
            id_slot: req.id_slot,
            role: Some(role),
            stop,
            ..Default::default()
        },
        req.stream,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use orangu::gguf::{GgufFile, GgufValue};

    const PRE: u32 = 256;
    const SUF: u32 = 257;
    const MID: u32 = 258;
    const REPO: u32 = 259;
    const SEP: u32 = 260;
    const PAD: u32 = 261;

    /// One token per byte, then the frame's tokens, spelled in `specials`.
    fn tokenizer(specials: &[&str]) -> Tokenizer {
        let mut tokens: Vec<GgufValue> = (0..=255u8)
            .map(|b| GgufValue::String(format!("<0x{b:02X}>")))
            .collect();
        let mut types: Vec<GgufValue> = (0..256).map(|_| GgufValue::I32(6)).collect();
        for special in specials {
            tokens.push(GgufValue::String(special.to_string()));
            types.push(GgufValue::I32(3));
        }
        tokens.push(GgufValue::String("</s>".to_string()));
        types.push(GgufValue::I32(3));
        Tokenizer::from_gguf(&GgufFile {
            version: 3,
            metadata: vec![
                (
                    "tokenizer.ggml.tokens".to_string(),
                    GgufValue::Array(tokens),
                ),
                (
                    "tokenizer.ggml.model".to_string(),
                    GgufValue::String("llama".to_string()),
                ),
                (
                    "tokenizer.ggml.token_type".to_string(),
                    GgufValue::Array(types),
                ),
                (
                    "tokenizer.ggml.add_space_prefix".to_string(),
                    GgufValue::U32(0),
                ),
                (
                    "tokenizer.ggml.add_bos_token".to_string(),
                    GgufValue::U32(0),
                ),
            ],
            tensors: vec![],
            alignment: 32,
            data_offset: 0,
        })
        .expect("tokenizer")
    }

    fn fim_vocabulary() -> Tokenizer {
        tokenizer(&[
            "<|fim_prefix|>",
            "<|fim_suffix|>",
            "<|fim_middle|>",
            "<|repo_name|>",
            "<|file_sep|>",
            "<|fim_pad|>",
        ])
    }

    fn cursor<'a>(prefix: &'a str, suffix: &'a str) -> Cursor<'a> {
        Cursor {
            prefix,
            suffix,
            typed: "",
            extra: &[],
            filename: "this.rs",
            repo: "orangu",
        }
    }

    fn prompt(tokenizer: &Tokenizer, cursor: &Cursor<'_>, context: usize) -> Vec<u32> {
        build_prompt(
            tokenizer,
            &tokenizer.fim_tokens().unwrap(),
            cursor,
            context,
            16,
        )
    }

    #[test]
    fn a_vocabulary_declares_its_frame_by_its_spellings() {
        let fim = fim_vocabulary().fim_tokens().expect("has a frame");
        assert_eq!((fim.prefix, fim.suffix, fim.middle), (PRE, SUF, MID));
        assert_eq!(fim.repo, Some(REPO));
        assert_eq!(fim.separator, Some(SEP));
        assert_eq!(fim.pad, Some(PAD));
    }

    /// All three are what make a frame; a vocabulary with two has none.
    #[test]
    fn a_vocabulary_without_the_whole_frame_cannot_infill() {
        assert_eq!(tokenizer(&[]).fim_tokens(), None);
        assert_eq!(
            tokenizer(&["<|fim_prefix|>", "<|fim_suffix|>"]).fim_tokens(),
            None
        );
    }

    /// The metadata keys say which is which even where no spelling would.
    #[test]
    fn the_metadata_keys_win_over_the_spellings() {
        let mut tokens: Vec<GgufValue> = ["a", "b", "c", "<|fim_prefix|>"]
            .iter()
            .map(|t| GgufValue::String(t.to_string()))
            .collect();
        tokens.push(GgufValue::String("</s>".to_string()));
        let key = |name: &str, id: u32| (format!("tokenizer.ggml.{name}"), GgufValue::U32(id));
        let tokenizer = Tokenizer::from_gguf(&GgufFile {
            version: 3,
            metadata: vec![
                (
                    "tokenizer.ggml.tokens".to_string(),
                    GgufValue::Array(tokens),
                ),
                (
                    "tokenizer.ggml.model".to_string(),
                    GgufValue::String("llama".to_string()),
                ),
                key("fim_pre_token_id", 0),
                key("fim_suf_token_id", 1),
                key("fim_mid_token_id", 2),
            ],
            tensors: vec![],
            alignment: 32,
            data_offset: 0,
        })
        .unwrap();
        let fim = tokenizer.fim_tokens().unwrap();
        assert_eq!((fim.prefix, fim.suffix, fim.middle), (0, 1, 2));
    }

    #[test]
    fn the_prompt_frames_the_text_around_the_cursor() {
        let t = fim_vocabulary();
        assert_eq!(
            prompt(&t, &cursor("ab", "c"), 4096),
            [PRE, b'a' as u32, b'b' as u32, SUF, b'c' as u32, MID]
        );
    }

    /// What the user typed since the prefix belongs right after it.
    #[test]
    fn what_was_typed_continues_the_prefix() {
        let t = fim_vocabulary();
        let got = prompt(
            &t,
            &Cursor {
                typed: "x",
                ..cursor("a", "b")
            },
            4096,
        );
        assert_eq!(got, [PRE, b'a' as u32, b'x' as u32, SUF, b'b' as u32, MID]);
    }

    /// **A file must not be able to end its own prefix.** Encoded the way a
    /// template is, these would be the frame's own tokens, and the model would
    /// be told the cursor was somewhere else.
    #[test]
    fn a_file_that_spells_a_frame_token_stays_text() {
        let t = fim_vocabulary();
        let got = prompt(&t, &cursor("<|fim_middle|>", "<|fim_prefix|>"), 4096);
        assert_eq!(got.iter().filter(|&&id| id == MID).count(), 1);
        assert_eq!(got.iter().filter(|&&id| id == PRE).count(), 1);
        assert_eq!(*got.last().unwrap(), MID);
    }

    /// Too long a side keeps what is nearest the cursor: the end of the prefix
    /// and the start of the suffix.
    #[test]
    fn the_text_nearest_the_cursor_survives_a_small_context() {
        let t = fim_vocabulary();
        let before = "a".repeat(5000) + "END";
        let after = "START".to_string() + &"z".repeat(5000);
        let got = prompt(&t, &cursor(&before, &after), 4096);
        assert!(got.len() <= CURSOR_WINDOW, "{}", got.len());
        let pre = got.iter().position(|&id| id == PRE).unwrap();
        let suf = got.iter().position(|&id| id == SUF).unwrap();
        let text = |slice: &[u32]| slice.iter().map(|&b| b as u8 as char).collect::<String>();
        assert!(text(&got[pre + 1..suf]).ends_with("END"));
        assert!(text(&got[suf + 1..]).starts_with("START"));
        assert!(
            suf - pre > got.len() - suf,
            "the prefix gets the larger share"
        );
    }

    /// What one side leaves unused, the other may have.
    #[test]
    fn a_short_side_leaves_its_room_to_the_other() {
        let t = fim_vocabulary();
        let after = "z".repeat(5000);
        let got = prompt(&t, &cursor("a", &after), 4096);
        assert_eq!(got.len(), CURSOR_WINDOW);
        let before = "a".repeat(5000);
        let got = prompt(&t, &cursor(&before, "z"), 4096);
        assert_eq!(got.len(), CURSOR_WINDOW);
    }

    /// A context smaller than the window still has room for the reply.
    #[test]
    fn the_reply_always_has_room() {
        let t = fim_vocabulary();
        let before = "a".repeat(5000);
        let got = prompt(&t, &cursor(&before, &before), 64);
        assert!(got.len() + 16 <= 64, "{}", got.len());
        assert_eq!(*got.last().unwrap(), MID);
    }

    /// Other files go in front, each under its name, then the cursor's own.
    #[test]
    fn other_files_stand_in_front_of_the_cursor_under_their_names() {
        let t = fim_vocabulary();
        let extra = [ExtraChunk {
            filename: "o".to_string(),
            text: "y".to_string(),
        }];
        let got = prompt(
            &t,
            &Cursor {
                extra: &extra,
                filename: "t",
                repo: "r",
                ..cursor("a", "b")
            },
            8192,
        );
        let nl = b'\n' as u32;
        assert_eq!(
            got,
            [
                REPO,
                b'r' as u32,
                nl,
                SEP,
                b'o' as u32,
                nl,
                b'y' as u32,
                SEP,
                b't' as u32,
                nl,
                PRE,
                b'a' as u32,
                SUF,
                b'b' as u32,
                MID,
            ]
        );
    }

    /// Without a token to name a file, a plain rule keeps them apart.
    #[test]
    fn a_vocabulary_with_no_file_token_rules_snippets_off() {
        let t = tokenizer(&["<PRE>", "<SUF>", "<MID>"]);
        let extra = [ExtraChunk {
            filename: "o".to_string(),
            text: "y".to_string(),
        }];
        let got = prompt(
            &t,
            &Cursor {
                extra: &extra,
                ..cursor("a", "b")
            },
            8192,
        );
        let rule = t.encode_literal(SNIPPET_RULE);
        assert_eq!(&got[..rule.len()], &rule[..]);
        assert_eq!(got[rule.len()], t.encode_literal("y")[0]);
        assert_eq!(got[rule.len() + 1], PRE);
    }

    /// The extra files take what the context leaves, most recent last, and the
    /// cursor's window is never cut for them.
    #[test]
    fn other_files_only_take_what_the_context_leaves() {
        let t = fim_vocabulary();
        let extra = [ExtraChunk {
            filename: "o".to_string(),
            text: "y".repeat(6000),
        }];
        let at = |context| {
            let got = prompt(
                &t,
                &Cursor {
                    extra: &extra,
                    ..cursor("a", "b")
                },
                context,
            );
            got.iter().filter(|&&id| id == b'y' as u32).count()
        };
        assert_eq!(at(CURSOR_WINDOW), 0);
        assert_eq!(at(16384), 6000);
        assert!(at(4096) > 1000 && at(4096) < 6000);
    }

    /// A model writing one of the frame's other tokens has finished the middle.
    #[test]
    fn the_frame_tokens_end_a_completion() {
        let t = fim_vocabulary();
        let ids = stop_tokens(&t, &t.fim_tokens().unwrap());
        for id in [PAD, REPO, SEP] {
            assert!(ids.contains(&id), "{id}: {ids:?}");
        }
        assert!(!ids.contains(&MID));
    }
}
