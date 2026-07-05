# Local LLM

`orangu` is designed to talk directly to a local llama.cpp server using its OpenAI-compatible API.

## Example configuration

```ini
[orangu]
server = main-server
model = ggml-org/gemma-4-E4B-it-GGUF
timeout = 1800
max_tool_rounds = 10

[main-server]
provider = llama.cpp
endpoint = http://localhost:8100/v1
model = ggml-org/gemma-4-E4B-it-GGUF
```

## Quick verification

Check the server:

```sh
curl http://localhost:8100/v1/models
```

Run the client:

```sh
orangu --config ./orangu.conf
```

Once connected, run `/information` inside `orangu` to see which OpenAI and llama.cpp-native endpoints the server exposes (models, `/health`, `/props`, `/slots`, `/metrics`, embeddings) and whether each is enabled.

## Notes

- The endpoint may be configured as either the server root or the `/v1` path.
- Tool-calling prompts can be slow on local models, so a larger timeout is recommended.
- The local tools run against the current workspace and can edit files on disk.
- If you start `llama-server` with `--api-key <key>`, set `api_key = <key>` in the server section. The key is sent on every request, so the `/v1/models` health probe keeps working.
