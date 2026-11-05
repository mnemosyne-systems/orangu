//! Fuzz targets for what this server parses from outside: see
//! `src/fuzzing.rs` for how they run. The GGUF reader's own target is beside it
//! in the library.

use crate::engine::chat_template::{ChatMessage, ChatTemplate};
use crate::engine::text_stop::{InfillRule, StopRules, TextGate};
use crate::engine::tokenizer::Tokenizer;
use crate::engine::{json_schema, tool_calls, tool_learn};
use crate::fuzzing::{GgufBytes, Rng, fuzz};
use orangu::gguf::GgufFile;
use serde_json::{Value, json};
use std::sync::Arc;

fn lossy(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

/// Text in pieces of a size the input picks, the way a stream arrives.
fn chunks(text: &str, rng: &mut Rng) -> Vec<String> {
    let mut pieces = Vec::new();
    let mut piece = String::new();
    for c in text.chars() {
        piece.push(c);
        if rng.below(4) == 0 {
            pieces.push(std::mem::take(&mut piece));
        }
    }
    if !piece.is_empty() {
        pieces.push(piece);
    }
    pieces
}

const SAMPLE: &str = "fn main() {\n    let x = 2024; // don't\n\n\tprintln!(\"{x}\");\n}\n  \
                      caf\u{e9} e\u{301} \u{4f60}\u{597d} 12345 <|im_start|>user\n";

/// A `gpt2`-shaped vocabulary covering every byte, in the GGUF form a file has.
fn vocabulary_file(pre: &str) -> Vec<u8> {
    let mut tokens: Vec<String> = bytes_to_unicode();
    tokens.push("he".to_string());
    tokens.push("<|im_start|>".to_string());
    let types: Vec<u32> = (0..tokens.len() as u32)
        .map(|i| if i as usize == tokens.len() - 1 { 3 } else { 1 })
        .collect();
    let mut file = GgufBytes::new(0, 6);
    file.kv_string("tokenizer.ggml.model", "gpt2")
        .kv_string("tokenizer.ggml.pre", pre)
        .kv_strings("tokenizer.ggml.tokens", &tokens)
        .kv_strings("tokenizer.ggml.merges", &["h e".to_string()])
        .kv_u32("tokenizer.ggml.bos_token_id", 256);
    file.string("tokenizer.ggml.token_type").u32(9).u32(4);
    file.u64(types.len() as u64);
    for ty in types {
        file.u32(ty);
    }
    file.0
}

fn bytes_to_unicode() -> Vec<String> {
    crate::engine::tokenizer::byte_to_unicode_table()
        .iter()
        .map(char::to_string)
        .collect()
}

const PRE_TYPES: &[&str] = &[
    "gpt-2",
    "qwen2",
    "qwen35",
    "smollm",
    "llama-bpe",
    "gpt-4o",
    "tekken",
    "tekken-num3",
    "bailingmoe2",
    "joyai-llm",
    "glm4",
];

/// Whatever a damaged vocabulary file does, loading it and using what loads
/// must come back as a result.
#[test]
fn fuzz_tokenizer_file() {
    let seeds = PRE_TYPES.iter().map(|pre| vocabulary_file(pre)).collect();
    fuzz(
        "tokenizer_file",
        seeds,
        &[b"tokenizer.ggml.", b"<|im_start|>", b"gemma4", b"llama"],
        1 << 20,
        |input| {
            let Ok(gguf) = GgufFile::read_from(std::io::Cursor::new(input)) else {
                return;
            };
            let Ok(tokenizer) = Tokenizer::from_gguf(&gguf) else {
                return;
            };
            let ids = tokenizer.encode(SAMPLE, true);
            let _ = tokenizer.decode(&ids);
            let _ = tokenizer.encode_for_embedding(SAMPLE);
            let _ = tokenizer.fim_tokens();
        },
    );
}

/// What a client sends goes through whichever split the model's file names,
/// and every byte of a vocabulary that has them all comes back.
#[test]
fn fuzz_tokenizer_text() {
    let tokenizers: Vec<Tokenizer> = PRE_TYPES
        .iter()
        .map(|pre| {
            let file = vocabulary_file(pre);
            let gguf = GgufFile::read_from(std::io::Cursor::new(file)).unwrap();
            Tokenizer::from_gguf(&gguf).unwrap()
        })
        .collect();
    let seeds = [
        SAMPLE,
        "\n    x",
        "a  b\t\tc\r\n\r\n  ",
        "12345678 3.14 1,000",
        "\u{4f60}\u{597d}abc\u{4e2d}\u{6587}123\u{30c6}\u{30b9}\u{30c8}",
        "\u{1f600}\u{200d}\u{1f468} \u{0301}\u{0301}",
        "'s 'T 're 'LL",
    ];
    fuzz(
        "tokenizer_text",
        seeds.iter().map(|s| s.as_bytes().to_vec()).collect(),
        &[
            b"<|im_start|>",
            b"\n",
            b"    ",
            b"\r\n",
            b"'s",
            b"\xf0\x9f\x98\x80",
            b"123",
        ],
        4096,
        |input| {
            let text = lossy(input);
            for tokenizer in &tokenizers {
                let literal = tokenizer.encode_literal(&text);
                assert_eq!(tokenizer.decode(&literal), text, "lost text");
                let _ = tokenizer.encode(&text, true);
            }
        },
    );
}

fn template_seeds() -> Vec<Vec<u8>> {
    let mut seeds: Vec<Vec<u8>> = [
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}\
         {% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}",
        "{%- if tools %}{{ tools | tojson }}{% endif -%}{% for m in messages %}\
         {% if m.tool_calls %}{% for c in m.tool_calls %}<tool_call>\n{\"name\": \
         \"{{ c.function.name }}\", \"arguments\": {{ c.function.arguments | tojson }}}\n\
         </tool_call>{% endfor %}{% else %}{{ m.content }}{% endif %}{% endfor %}",
        "{{ bos_token }}{% for m in messages %}{{ raise_exception('no') if m.role == 'x' }}\
         {{ m.get('content', '').strip() }}{% endfor %}{{ strftime_now('%Y') }}",
        "{% macro f(n) %}{{ n }}{% if n %}{{ f(n - 1) }}{% endif %}{% endmacro %}{{ f(3) }}",
        "{% set ns = namespace(a=1) %}{% for i in range(3) %}{% set ns.a = ns.a * 2 %}{% endfor %}\
         {{ ns.a }}{{ 'x' * 3 }}{{ messages | length }}",
    ]
    .iter()
    .map(|s| s.as_bytes().to_vec())
    .collect();
    // Real templates, where a run has them: `ORANGU_FUZZ_TEMPLATES=<dir>`.
    if let Some(dir) = std::env::var_os("ORANGU_FUZZ_TEMPLATES")
        && let Ok(entries) = std::fs::read_dir(dir)
    {
        seeds.extend(
            entries
                .flatten()
                .filter_map(|e| std::fs::read(e.path()).ok()),
        );
    }
    seeds
}

/// A template comes in a model file and runs on every request: it must not be
/// able to take the server down, or hold it for long.
#[test]
fn fuzz_chat_template() {
    let tools = json!([{"type": "function", "function": {
        "name": "read", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    }}]);
    let mut call = ChatMessage::text("assistant", "");
    call.tool_calls = Some(json!([{
        "id": "1", "type": "function", "function": {"name": "read", "arguments": "{}"},
    }]));
    let messages = [
        ChatMessage::text("system", "be brief"),
        ChatMessage::text("user", "hi"),
        call,
        ChatMessage::text("tool", "ok"),
    ];
    fuzz(
        "chat_template",
        template_seeds(),
        &[
            b"{% for i in range(99999) %}",
            b"{% endfor %}",
            b"{{ 'x' * 999999 }}",
            b"{% macro f(n) %}{{ f(n) }}{% endmacro %}{{ f(1) }}",
            b"{% if tools %}",
            b"{% endif %}",
            b"| tojson",
            b"messages",
            b"tool_calls",
        ],
        1 << 16,
        |input| {
            let template = ChatTemplate::new(lossy(input));
            let _ = template.render(&messages, true, "<s>", "</s>", Default::default());
            let _ = template.render_with_tools(
                &messages,
                true,
                "<s>",
                "</s>",
                Default::default(),
                Some(&tools),
            );
            if let Some(calls) = template.learned_calls() {
                let mut reader = tool_learn::LearnedReader::new(calls);
                let _ = reader.push("<tool_call>x");
                let _ = reader.finish();
            }
        },
    );
}

const SCHEMAS: &[&str] = &[
    r#"{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"array","items":{"type":"integer"},"minItems":1}},"required":["a"],"additionalProperties":false}"#,
    r##"{"$defs":{"n":{"anyOf":[{"type":"null"},{"$ref":"#/$defs/n"}]}},"$ref":"#/$defs/n"}"##,
    r#"{"oneOf":[{"enum":[1,"x",null]},{"const":{"k":[1,2]}},{"type":"number"},{"type":"boolean"}]}"#,
    r##"{"type":"array","items":{"$ref":"#"}}"##,
];

/// A schema comes in a request and is compiled before anything else looks at
/// it, and what it holds a reply to has to be answerable at every byte.
#[test]
fn fuzz_json_schema() {
    let mut seeds: Vec<Vec<u8>> = SCHEMAS.iter().map(|s| s.as_bytes().to_vec()).collect();
    seeds.push(b"[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]]".to_vec());
    fuzz(
        "json_schema",
        seeds,
        &[
            b"{\"type\":\"object\"",
            b"\"properties\":{",
            b"\"items\":{\"items\":{\"items\":{",
            b"\"anyOf\":[{",
            b"\"$ref\":\"#\"",
            b"\"$defs\"",
            b"\"required\":[\"a\"]",
            b"\"enum\":[",
            b"\"minItems\":99999999",
            b"\"maxItems\":99999999",
            b"\"minimum\":1e308",
        ],
        1 << 16,
        |input| {
            let Ok(schema) = serde_json::from_slice::<Value>(input) else {
                return;
            };
            let value = json!({"a": "x", "b": [1, 2], "c": null});
            let _ = json_schema::validate(&schema, &value);
            let Ok(compiled) = json_schema::compile(&schema) else {
                return;
            };
            let mut prefix = json_schema::SchemaPrefix::new(Arc::new(compiled));
            for byte in br#"{"a": "x\n", "b": [1, -2.5e3], "c": null, "d": [[[]]]}"# {
                let _ = prefix.allows(std::slice::from_ref(byte));
                let _ = prefix.is_complete();
                if !prefix.push_bytes(std::slice::from_ref(byte)) {
                    break;
                }
            }
        },
    );
}

const CALLS: &[&str] = &[
    "text <tool_call>\n{\"name\": \"read\", \"arguments\": {\"q\": \"x\"}}\n</tool_call> after",
    "<|tool_call>call:read{q:<|\"|>x<|\"|>,n:1}<tool_call|>",
    "<function=read><parameter=q>x</parameter></function>",
    "<tool_call>read<arg_key>q</arg_key><arg_value>x</arg_value></tool_call>",
    "[TOOL_CALLS]read[ARGS]{\"q\": 1}",
    "<|tool_calls_section_begin|><|tool_call_begin|>functions.read:0<|tool_call_argument_begin|>{\"q\":1}<|tool_call_end|><|tool_calls_section_end|>",
    "<think>x</think><tool_call>{\"name\":\"a\"",
    "<minimax:tool_call><invoke name=\"read\"><parameter name=\"q\">x</parameter></invoke></minimax:tool_call>",
];

/// A model's output is untrusted too: it is what a tool call is read from,
/// a piece at a time.
#[test]
fn fuzz_tool_calls() {
    let learned_templates = [
        "{% for m in messages %}{% for c in m.tool_calls %}<<CALL>>{\"name\": \"{{ c.function.name }}\", \
         \"arguments\": {{ c.function.arguments | tojson }}}<</CALL>>{% endfor %}{% endfor %}",
        "{% for m in messages %}{% for c in m.tool_calls %}[[{{ c.function.name }}|{{ c.function.arguments | tojson }}]]{% endfor %}{% endfor %}",
    ];
    let learned: Vec<_> = learned_templates
        .iter()
        .filter_map(|t| ChatTemplate::new(t.to_string()).learned_calls())
        .collect();
    fuzz(
        "tool_calls",
        CALLS.iter().map(|s| s.as_bytes().to_vec()).collect(),
        &[
            b"<tool_call>",
            b"</tool_call>",
            b"<|tool_call>",
            b"<tool_call|>",
            b"call:",
            b"<|\"|>",
            b"{\"name\":",
            b"\"arguments\":",
            b"<function=",
            b"<parameter=",
            b"[TOOL_CALLS]",
            b"<<CALL>>",
            b"\\u",
        ],
        1 << 14,
        |input| {
            let text = lossy(input);
            let mut rng = Rng::new(input.len() as u64);
            let split = tool_calls::split(&text);
            let _ = split.has_calls();
            let _ = tool_calls::close_trailing(&text);
            let _ = tool_calls::may_be_partial(&text);
            if let Some((at, opener)) = tool_calls::first_opener(&text) {
                let span = &text[at + opener.len()..];
                let _ = tool_calls::call_name(opener, span);
                let mut stream = tool_calls::ArgumentStream::new(opener);
                let mut so_far = String::new();
                for piece in chunks(span, &mut rng) {
                    so_far.push_str(&piece);
                    let _ = stream.advance(&so_far);
                }
                let _ = stream.finish(&so_far, "{}");
            }
            let _ = tool_calls::last_opener(&text);
            for calls in &learned {
                let mut reader = tool_learn::LearnedReader::new(calls.clone());
                let mut read = String::new();
                for piece in chunks(&text, &mut rng) {
                    read.push_str(&reader.push(&piece));
                }
                read.push_str(&reader.finish());
                let _ = tool_calls::split(&read);
            }
        },
    );
}

/// However a reply arrives in pieces, the stop rules let the same text out.
#[test]
fn fuzz_text_stop() {
    fuzz(
        "text_stop",
        vec![
            b"stop\nhello world STOP more\n  indented\n    deeper\nback\n".to_vec(),
            "caf\u{e9} \u{4f60}\u{597d} STO\u{1f600}"
                .as_bytes()
                .to_vec(),
            b"\n\n\n   \n".to_vec(),
        ],
        &[b"STOP", b"\n", b"    ", b"\xf0\x9f\x98\x80", b"after"],
        1 << 14,
        |input| {
            let text = lossy(input);
            let mut rng = Rng::new(input.len() as u64);
            let rules = [
                StopRules {
                    strings: vec!["STOP".to_string(), "\u{e9} ".to_string()],
                    infill: None,
                },
                StopRules {
                    strings: Vec::new(),
                    infill: Some(InfillRule::at_cursor("  x", "after\n", Some(2))),
                },
                StopRules {
                    strings: vec!["\n\n".to_string()],
                    infill: Some(InfillRule::at_cursor("", "", None)),
                },
            ];
            for rule in &rules {
                let run = |pieces: &[String]| {
                    let mut gate = TextGate::new(rule);
                    let mut out = String::new();
                    for piece in pieces {
                        let passed = gate.push(piece);
                        out.push_str(&passed.text);
                        if passed.stop {
                            return out;
                        }
                    }
                    out.push_str(&gate.finish());
                    out
                };
                let pieces = chunks(&text, &mut rng);
                assert_eq!(run(&pieces), run(std::slice::from_ref(&text)), "{pieces:?}");
            }
        },
    );
}
