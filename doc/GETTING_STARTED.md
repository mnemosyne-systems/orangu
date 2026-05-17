# Getting started

## 1. Start llama.cpp

Run a local `llama-server` instance with an OpenAI-compatible endpoint.

```sh
llama-server \
  --model /path/to/model.gguf \
  --port 8100 \
  --ctx-size 8192
```

## 2. Create a client configuration

Start from the sample file:

```sh
cp doc/etc/orangu.conf ./orangu.conf
```

Adjust the model name and endpoint if needed.

## 3. Run the client

```sh
cargo run --bin orangu -- --config ./orangu.conf
```

Or with an installed binary:

```sh
orangu --config ./orangu.conf
```

## 4. Try a few commands

- `/help`
- `/connect`
- `/disconnect`
- `/list_models`
- `/list_files`
- `/show_file README.md`
- `/tools`
- `/model`
- `/reload`
- `/open_file README.md`

Then try a natural-language request such as:

```text
list files
```

Built-in commands also accept natural-language forms, for example:

```text
open README.md
list models
list files
show help
```

Lines whose first non-whitespace character is `#` stay local and are not sent to the model. Lines whose first non-whitespace character is `\` are ignored.
