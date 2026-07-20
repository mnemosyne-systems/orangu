\newpage

## Serving models per role

`orangu-server` is the inference engine `orangu` connects to. It loads a GGUF
model and serves an OpenAI-compatible API; the *Inference server* chapter covers
its full command set, host/port/backend configuration, and model-inventory
subcommands. This chapter lists a good starting model for each role and how to
serve it.

The model argument resolves the same way `orangu-server show`/`download` do: an
existing local `.gguf` path, an `NR`/`MODEL` label under the configured `models`
directory, or a `<user>/<model>[:quant]` Hugging Face repo (fetched on first
use). The role flag (`--all`/`--code`/`--review`/`--explorer`/`--embedding`,
mutually exclusive, `--all` by default) selects how the server presents itself;
give each running server its own port in `orangu-server.conf` when you run more
than one at a time.

`role = all`

```sh
orangu-server --all unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL
```

`role = code`

```sh
orangu-server --code yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF
```

`role = review`

For the fastest reviews, disable thinking with `--reasoning-budget 0 --reasoning off`:

```sh
orangu-server --review unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL \
              --reasoning-budget 0 --reasoning off
```

`role = explorer`

```sh
orangu-server --explorer bartowski/gemma-4-12B-it-GGUF
```

Host, port, web-UI port, GPU backend, and the `models` directory are set in the
`[orangu-server]` section of `orangu-server.conf` (see the *Inference server*
chapter) rather than as per-run flags. Session KV-cache persistence across tab
park/close/quit and reload is handled by the server's own default slot behavior —
no flag to enable, no directory to create in advance.

### Embedding model

Semantic `/search` needs a server serving an embedding model. Start one with the
`--embedding` role flag:

```sh
orangu-server --embedding ggml-org/embeddinggemma-300M-GGUF
```

Give it its own port if you run it alongside a chat server, and its own config
section with `role = embeddings` (see the *Configuration* chapter). `orangu`
probes it at startup and enables `/search` when it responds; if the probe fails
it prints the reason (connection refused, timed out, or an error status) so you
can tell why, rather than a silent "not detected". The model is fetched from
Hugging Face on first use — wait for the server's "listening" line **before**
starting `orangu`, since a server still downloading or loading the model will not
yet accept connections and the probe will report it unreachable.

The cached vectors are specific to the embedding model that produced them, and
are keyed by the endpoint you configured. If you restart the embedding server
with a **different model** (or point `role = embeddings` at a different endpoint),
the cache no longer matches — delete the workspace's `embeddings/` subdirectory
under `~/.orangu/workspace/<hash>/` and run `/search` again to re-index.
Restarting with the **same** model reuses the cache and only re-embeds files that
changed.
