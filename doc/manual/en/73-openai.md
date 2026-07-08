\newpage

## OpenAI platform

### llama.cpp

[**llama.cpp**][llama] is an OpenAI compatible platform.

Please, look at their documentation for further information.

**Building**

Clone the repository

```sh
git clone https://github.com/ggml-org/llama.cpp.git
```

```sh
cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DCMAKE_INSTALL_PREFIX=/usr/local/ -DGGML_VULKAN=1 -B build
cmake --build build --config Debug -j 12

cd build
su
make install
```

for example for an AMD/Vulkan based platform.

**Running**

`role = all`

```sh
llama-server -hf unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL \
             --port 8100 \
             --ctx-size 262144 \
             -sm layer \
             -t 4 \
             --webui-mcp-proxy \
             --fit on \
             --tools all \
             -b 2048 \
             -ub 2048 \
             --cache-reuse 256 \
             --slot-save-path ~/.orangu/llama-slots \
             -fa on \
             -ctk q8_0 \
             -ctv q8_0
```

`role = code`

```sh
llama-server -hf yuxinlu1/gemma-4-12B-coder-fable5-composer2.5-v1-GGUF \
             --port 8100 \
             --ctx-size 131072 \
             -t 4 \
             --webui-mcp-proxy \
             --fit on \
             --image-min-tokens 1024 \
             --tools all \
             -b 2048 \
             -ub 2048 \
             --cache-reuse 256 \
             --slot-save-path ~/.orangu/llama-slots \
             -fa on \
             -ctk q8_0 \
             -ctv q8_0
```

`role = review`

```sh
llama-server -hf unsloth/gemma-4-26B-A4B-it-qat-GGUF:UD-Q4_K_XL \
             --port 8100 \
             --ctx-size 262144 \
             -np 1 \
             -fa on \
             -sm layer \
             -t 4 \
             --webui-mcp-proxy \
             --fit on \
             --tools all \
             -b 2048 \
             -ub 2048 \
             --cache-reuse 256 \
             --slot-save-path ~/.orangu/llama-slots \
             --reasoning-budget 0 \
             --reasoning off \
             -ctk q8_0 \
             -ctv q8_0
```

`role = explorer`

```sh
llama-server -hf bartowski/gemma-4-12B-it-GGUF \
             --port 8100 \
             --ctx-size 131072 \
             -np 1 \
             -fa on \
             -ctk q8_0 \
             -ctv q8_0 \
             -b 2048 \
             -ub 2048 \
             --cache-reuse 256 \
             --slot-save-path ~/.orangu/llama-slots \
             --temp 0.7 \
             --top-p 0.8 \
             --top-k 20 \
             --min-p 0 \
             --jinja \
             --fit on
```

`--slot-save-path PATH` turns on llama.cpp's slot save/restore endpoints; `orangu`
uses them automatically when present to persist a session's KV cache to disk on
tab park/close/quit and reload it on tab activate/resume, avoiding a full
re-prefill of the conversation so far. **Create the directory before starting
the server** — llama-server exits immediately with "not a directory" if `PATH`
does not already exist:

```sh
mkdir -p ~/.orangu/llama-slots
```

The flag is optional: without it, `orangu` detects the server doesn't support
slot persistence (one informational notice, not an error) and behaves exactly
as before. Combined with `--cache-reuse`, above, both layers of `orangu`'s KV
cache cooperation are then active — in-memory reuse across requests within a
session, and on-disk persistence across tab switches and restarts.

Embedding model

Semantic `/search` needs a server serving an embedding model. Start one with
`--embedding` (switches the server to the embeddings endpoint), `--pooling last`
(the pooling strategy the model expects), `-np N` (the number of requests the
server processes in parallel), and a physical batch size (`-b`/`-ub`) large
enough for embedding requests that batch several chunks together:

```sh
llama-server -hf ggml-org/embeddinggemma-300M-GGUF \
             --port 8100 \
             --embedding \
             --pooling last \
             --ctx-size 8192 \
             -np 8 \
             -b 2048 \
             -ub 2048 \
             --fit on
```

`8100` matches the port used throughout this chapter's other examples. If you
are running this alongside one of them (a chat server and the embeddings server
at the same time), give the embeddings server its own free port instead (e.g.
`8300`) so the two do not collide.

`-np N` sets how many embedding requests the server handles at the same time.
`/search` uploads several files at once (up to eight), so matching `-np` to that
(`-np 8`) lets those requests run truly in parallel and makes the first index
build markedly faster; with `-np 1` they queue and the upload is effectively
sequential.

`orangu` keeps each embedding request within a conservative token budget on its
own side, but that budget is still shared across every one of the `-np` slots
processing requests at the same moment — so with several requests in flight, the
server's default physical batch size (`-b`/`--batch-size`, `-ub`/`--ubatch-size`,
512 tokens) can still be too small. Raising both to `2048`, as above, gives enough
headroom for `-np 8` requests to run at once without hitting "input is too large
to process".

Give this server its own section with `role = embeddings` (see the Configuration
chapter). orangu probes it at startup and enables `/search` when it responds; if
the probe fails it prints the reason (connection refused, timed out, or an error
status) so you can tell why, rather than a silent "not detected". `-hf` downloads
the model from Hugging Face the first time it is used — start `llama-server` and
wait for its "server is listening" line **before** starting orangu, since a
server that is still downloading or loading the model will not yet accept
connections and the probe will report it unreachable.

The cached vectors are specific to the embedding model that produced them, and
are keyed by the endpoint you configured. If you restart the embedding server
with a **different model** (or point `role = embeddings` at a different endpoint),
the cache no longer matches — delete the workspace's `embeddings/` subdirectory
under `~/.orangu/workspace/<hash>/` and run `/search` again to re-index.
Restarting with the **same** model reuses the cache and only re-embeds files that
changed.
