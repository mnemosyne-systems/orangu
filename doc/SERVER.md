# orangu-server

`orangu-server` loads a GGUF model and serves it over HTTP, with an
OpenAI-compatible API, its own native endpoints and a web console. It is the
inference engine itself: GGUF loading, tokenization, the forward pass, sampling
and request scheduling are all implemented here in Rust. It serves text, chat
and embedding models, plus text-to-image, text-to-video and text-to-speech
models, for twenty-four servable architecture families.

It is also the machine's model inventory tool. `system`, `suggest`, `list`,
`show`, `plan`, `download`, `delete` and `refresh` answer the questions that
matter when getting, choosing and cleaning up a model, and need no model
loaded.

`orangu-coordinator` ([COORDINATOR.md](COORDINATOR.md)) sits in front of the
server on machines that keep one model resident at a time, starting and
stopping it on demand.

This page is a map for developers. The [manual](manual/en/46-server.md) is the
full documentation: each topic is documented once there, and the links below go
to it.

## Quick start

```sh
orangu-server unsloth/gemma-4-E2B-it-GGUF
```

The model can be a local `.gguf` path, an `NR` or `MODEL` from
`orangu-server list`, or a `<user>/<model>[:quant]` Hugging Face repo, which is
fetched first if it isn't already on disk. With no model argument, the server
lists what is on disk and asks.

The API listens on `http://127.0.0.1:8100` by default:

```sh
curl -s localhost:8100/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello"}]}'
```

More in the manual's
[Quick start](manual/en/46-server.md#quick-start).

## Where to read

| Topic | Section |
| :--- | :--- |
| Fetching, sizing, listing and deleting models | [GGUF inventory](manual/en/46-server.md#gguf-inventory) |
| The server and a model as one file | [Bundling](manual/en/46-server.md#bundling-the-server-and-a-model-as-one-file) |
| The container image | [Container image](manual/en/46-server.md#container-image-the-whole-stack) |
| `orangu-server.conf`, every key | [Configuration](manual/en/46-server.md#configuration) |
| Prometheus and readiness | [Monitoring](manual/en/46-server.md#monitoring-metrics-and-ready) |
| Conversations longer than the context | [Context shift](manual/en/46-server.md#context-shift) |
| Draft models and speculative decoding | [Speculative decoding](manual/en/46-server.md#speculative-decoding-draft_model) |
| Workers on other machines | [The `[workers]` section](manual/en/46-server.md#the-workers-section) |
| What the file tools may touch | [Workspace](manual/en/46-server.md#workspace) |
| `all`, `code`, `review` and the other roles | [Roles](manual/en/46-server.md#roles) |
| Vulkan, Metal, CUDA, ROCm, OpenCL, NPU | [GPU backend](manual/en/46-server.md#gpu-backend) |
| The web console | [Web UI](manual/en/46-server.md#web-ui) |
| Pictures, clips and speech | [Image](manual/en/46-server.md#image-generation), [video](manual/en/46-server.md#video-generation) and [speech](manual/en/46-server.md#speech-generation) generation |
| Changing the loaded model | [Model management](manual/en/46-server.md#model-management) |
| Deleting old chat sessions | [Session management](manual/en/46-server.md#session-management) |
| Stopping the server | [Shutting it down](manual/en/46-server.md#shutting-it-down) |
| Which architectures run, and how | [Scope](manual/en/46-server.md#scope) |

The engine's internals are in [ENGINE.md](ENGINE.md) and the
[Inference server internals](manual/en/78-server.md) chapter.

## Endpoint reference

Every endpoint is documented in one place, field by field, with the rules
(bearer token, queue `503`, TLS) that apply to all of them:
[the HTTP endpoints chapter](manual/en/80-http.md#endpoints-at-a-glance).
