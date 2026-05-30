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
| `quotes` | No | Quote set shown while the model is thinking. Defaults to `none`. Options: `none`, `star_trek`, `star_wars`, `marco_pierre_white`, `gordon_ramsay`, `calvin_and_hobbes`, `all` |
| `width` | No | Virtual terminal width in characters. Controls the layout canvas for `/show_file` output. Defaults to `512` |
| `banner` | No | Horizontal placement of the banner. Defaults to `left`. Options: `left`, `center`, `right` |
| `feedback` | No | Show a green or red dot in the output window after each command to indicate success or failure. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `auto_rebase` | No | Automatically rebase the branch before `/pull_request` if it is behind the base. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |
| `auto_squash` | No | Automatically squash commits before `/pull_request` if more than one commit is ahead of the base. Defaults to `off`. Options: `on`, `true`, `1`, `off`, `false`, `0` |

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
- Set `feedback = on` in `[orangu]` to show a green or red dot in the output window after each command completes

## Sample file

The distributed sample lives at:

```text
doc/etc/orangu.conf
```

It ships with llama.cpp-style profiles and a 30-minute timeout suitable for local tool-calling workloads.
