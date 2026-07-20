\newpage

# Introduction

[**orangu**][orangu] is a local, workspace-aware, tool-driven coding environment.

More than a client, **orangu is a complete, self-contained AI coding stack** — three
cooperating programs written end to end in Rust: the coding environment
(`orangu`), an on-demand model manager (`orangu-coordinator`, see the *Coordinator*
chapter), and a native, pure-Rust GGUF inference server (`orangu-server`, see the
*Inference server* chapter) that implements the transformer forward pass itself,
with no dependency on llama.cpp/ggml's compiled code and no Python. Every layer
speaks the OpenAI-compatible API. See *A complete stack* below.

**orangu** is named after the [Orangutan](https://en.wikipedia.org/wiki/Orangutan) - the smartest ape.

![orangu terminal interface](images/orangu-terminal.png)

## Features

* A complete Rust stack — the `orangu` editor, the `orangu-coordinator` model manager, and the native `orangu-server` GGUF inference engine, with no llama.cpp, ggml, or Python dependency
* OpenAI-compatible chat completions served by the built-in `orangu-server` — fully local, no Internet connection required after setup
* Interactive code review (`/review`) and LLM-driven auto review (`/auto_review`) of the changes on your branch, with a category-grouped report you can export or post to an issue
* Local file reading and editing
* Workspace-aware Git and forge tools (commit, rebase, push, pull requests, comments) for the whole change-and-review loop
* URL fetching for external knowledge
* Shell command execution inside the workspace
* Model switching and runtime server target control
* PDF export of the console or a review report (`/export`)
* Persistent history, shell-style editing, and a terminal status banner
* Built-in offline manual (`/manual`) with full-text search

## A complete stack

Most local-AI setups are a patchwork: one tool for the editor, a separate engine
a separate engine for inference, and glue to manage which model is loaded.
orangu is the whole stack in one project — three cooperating programs, each
speaking the OpenAI-compatible API to the next:

```
                 you
                  │
        ┌─────────▼──────────┐   orangu — the coding environment: terminal UI,
        │       orangu       │   local + Git/forge tools, code review, knowledge
        └─────────┬──────────┘   graph, semantic search, compression
                  │  OpenAI-compatible HTTP
        ┌─────────▼──────────┐   orangu-coordinator — optional on-demand model
        │ orangu-coordinator │   manager: starts/stops the server and swaps the
        └─────────┬──────────┘   model per role (review, chat, embeddings)
                  │  OpenAI-compatible HTTP
        ┌─────────▼──────────┐   orangu-server — native pure-Rust GGUF engine:
        │    orangu-server   │   loading, tokenization, the transformer forward
        └─────────┬──────────┘   pass, sampling, scheduling, GGUF inventory
                  │
     CPU · Vulkan · CUDA · ROCm · OpenCL
```

* **`orangu`** — the workspace-aware coding environment you drive (this manual's
  main subject): the terminal UI, local and Git/forge tools, `/review` and
  `/auto_review`, the knowledge graph, semantic `/search`, and the
  context-compression engine.
* **`orangu-coordinator`** — an optional companion HTTP proxy that starts and
  stops `orangu-server` on demand and swaps to whichever model each request
  needs, so a single-GPU machine can use a different model per role without ever
  running more than one server at once. See the *Coordinator* chapter.
* **`orangu-server`** — *is* the inference engine: GGUF loading, tokenization,
  the transformer forward pass, sampling, and request scheduling implemented
  directly in Rust with no dependency on llama.cpp/ggml's compiled code, running
  on CPU or GPU (Vulkan, CUDA, ROCm, OpenCL). It also serves as the machine's
  GGUF inventory. See the *Inference server* chapter.

Because every layer talks to the next over the OpenAI-compatible API, the pieces
stay cleanly separated, yet they ship and run as one. The result is a fully
local, fully private, single-language AI coding stack — no Python, no llama.cpp,
no cloud.

## Community

Contributions to [**orangu**][orangu] are managed on [GitHub][orangu]

* [Ask a question][ask]
* [Raise an issue][issue]
* [Feature request][request]
* [Code submission][submission]

Contributions are most welcome!

Please, consult our [Code of Conduct][conduct] policies for interacting in our
community.

Consider giving the project a [star][star] on
[GitHub][orangu] if you find it useful. And, feel free to follow
the project on [X][twitter] as well.
