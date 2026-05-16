\newpage

# Configuration

`orangu` uses an INI configuration file.

## `[orangu]`

The main section selects the default profile and client-wide limits:

```ini
[orangu]
model = gemma-4-E4B-it-GGUF
timeout = 1800
max_tool_rounds = 10
```

| Key | Required | Description |
| :-- | :-- | :-- |
| `model` | Yes, if multiple profiles exist | Default profile name |
| `timeout` | No | Request timeout in seconds. The default is `1800` |
| `max_tool_rounds` | No | Maximum tool-calling turns before the client aborts the prompt |

## Model profiles

Each profile is a named section:

```ini
[gemma-4-E4B-it-GGUF]
provider = llama.cpp
endpoint = http://localhost:8100/v1
model = ggml-org/gemma-4-E4B-it-GGUF
```

| Key | Required | Description |
| :-- | :-- | :-- |
| `provider` | Yes | `llama.cpp` or `openai` |
| `endpoint` | Yes | OpenAI-compatible API URL |
| `model` | Yes | Model identifier used in chat completion requests |
| `api_key` | No | Bearer token for authenticated endpoints |
| `api_key_env` | No | Environment variable containing the bearer token |

- The endpoint may be configured either with or without `/v1`
- The client normalizes the endpoint internally before calling `/v1/chat/completions`
- Slow responses will display a blinking `Thinking` placeholder in the output area while the model is working

## Sample file

The distributed sample lives at:

```text
doc/etc/orangu.conf
```

It ships with llama.cpp-style profiles and a 30-minute timeout suitable for local tool-calling workloads.
