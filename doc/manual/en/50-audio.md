\newpage

# Speech generation

`orangu-server` speaks text aloud. Point it at a speech model and every
chat turn in the web console, and every `/v1/audio/speech` or
`/v1/chat/completions` request, is answered with the text spoken: 24 kHz
mono speech, as WAV, FLAC or raw PCM, in a voice from a voice pack. This
chapter is the practical side: what to download, how to start the
server, and what the web console's settings do. The *Inference server*
chapter's **Speech generation** section has the reference for every key,
and the *HTTP endpoints* chapter the API.

## The model, in three files

The speech model is **Qwen3-TTS**, published as GGUF in two sizes, 1.7B
and 0.6B. The Base checkpoints are the ones to serve, and `Q8_0` is the
quantization to take:

```sh
orangu-server download cstr/qwen3-tts-1.7b-base-GGUF:Q8_0
```

That is the model. It carries three parts:

- **The talker**, a Qwen3 decoder that reads the text and writes the
  speech one frame at a time, as the first of sixteen codebooks of a
  12 Hz speech codec.
- **The code predictor**, which fills in the other fifteen codebooks of
  each frame.
- **A speaker encoder**, which this server does not use.

Two more kinds of file make it speak, and `download` fetches both after
the model, each skipped when the models directory already holds one:

- **The speech codec**: `qwen3-tts-tokenizer-12hz.gguf` from
  `cstr/qwen3-tts-tokenizer-12hz-GGUF`. It turns each frame's sixteen
  codes, twelve and a half frames a second, into 24 kHz mono sound. The
  server takes the largest codec it finds, so the `F16` file is preferred
  over a `q8_0` one beside it. One codec serves both model sizes.
- **A voice pack**: from `cstr/qwen3-tts-voices-GGUF`,
  `qwen3-tts-voice-1.7b-designed.gguf` for the 1.7B model and
  `qwen3-tts-voice-0.6b-designed.gguf` for the 0.6B. A pack is who
  speaks: a speaker embedding, and a short reference recording as its
  transcript and its codec codes. The two sizes need packs of their own
  width; `download` fetches the one that fits, when none of that width is
  there.

The server finds the codec and the voice packs by what is inside the
file, not by its name. A file in another directory is named in the
configuration instead (`codec`, `voices`).

`list` shows the model as `Yes (qwen3tts)`, and the codec and the voice
packs as `Companion (qwen3tts)`: they are part of a speech model and are
never served on their own.

## Starting the server

The speech model is served in its own role, `audio`, and `--audio` is
the flag that asks for it:

```sh
orangu-server --audio
```

That prints the model table with every model that is not a speech model
greyed out, so the pick is a speech model. A speech model given on the
command line, or named in the configuration, comes up in the `audio`
role without the flag. The server then reports the codec and the voice
packs it found:

```
qwen3tts codec …/qwen3-tts-tokenizer-12hz.gguf
qwen3tts voice pack …/qwen3-tts-voice-1.7b-designed.gguf
```

and the voices they hold. A voice that a pack calls `default` is named
after its file instead, the file name without `qwen3-tts-voice-`: the
1.7B pack's voice is `1.7b-designed`. A pack made for the other model
size is skipped, with a line in the log saying so.

With the model named in the configuration, `--daemon` starts on it
without the table; `orangu-server -i` writes such a configuration, and
answers `role: audio` on its own when the model is a speech model:

```ini
[orangu-server]
models = ~/.cache/huggingface/hub
model = cstr/qwen3-tts-1.7b-base-GGUF:Q8_0

[web]
port = 8200
```

Nothing else is needed. The defaults are the first voice loaded, the
language detected from the text, WAV, and at most two minutes of speech
a reply. The keys are in the *Inference server* chapter's **Speech
generation** section: `audio_voice`, `audio_language`, `audio_format`,
`audio_temperature`, `audio_top_k`, `audio_repetition_penalty` and
`audio_max_seconds`, plus `codec` and `voices`.

The picker will not serve a speech model in any other role (`--code` is
an error that says so), nor a language model under `--audio`. `bundle`
refuses a speech model, which needs its codec and voice packs beside it.

## How the voice is made

The voice is cloned in context. The server gives the talker the voice
pack's reference transcript followed by the text to speak, laid over the
reference recording's frames, and the talker carries on from there in
the same voice, saying the new text. The codec then turns the frames
into sound.

A clip ends when the model ends it. Two limits stop one that runs on: the
longest reply (`audio_max_seconds`, two minutes by default), and a guard
of eight seconds plus 1.2 seconds for every token of the text.
A clip cut short is said so in the log, and in the console's caption.

The server logs one line a clip:

```
orangu-server: [audio] 4.6 s of speech (58 frames, voice 1.7b-designed, language auto, seed …) in … s — prompt (… rows) … s, speaking … s, decoding … s
```

## In the web console

Open the console (`http://<host>:8200`). The topbar names the model, and
the gear at its right is **Settings**.

### Downloading from the console

**Settings › Models** is the model manager. Type
`cstr/qwen3-tts-1.7b-base-GGUF:Q8_0` into the download box and the model
arrives first, then the codec and the voice pack, with progress for each
file. When the downloads finish, the row's **Load** button restarts the
server on the model, and the console reconnects on its own.

### Speech settings

**Settings › Audio** holds what every reply gets. Each row has an **(i)**
that explains it on hover. On a model that is not a speech model the tab
is greyed, with *Not a speech model*.

| setting | choices | |
| :-- | :-- | :-- |
| **Voice** | the voices loaded | The pack's description is under it |
| **Language** | Auto, or one of Chinese, English, French, German, Italian, Japanese, Korean, Portuguese, Russian, Spanish | Auto lets the model tell from the text |
| **Format** | WAV, FLAC | How the reply is kept beside the session. FLAC is smaller and sounds the same |
| **Temperature** | 0 to 2 | How freely the model reads the text: lower is steadier, and `0` always takes the likeliest |
| **Longest reply** | seconds, up to 600 | A reply also stops when the model ends it, which is normally long before this |

**Reset** puts the form back to the configuration file's values, and the
dialog's **Save** makes the form the server's defaults until it
restarts, as on the Video pane.

### Speaking a reply

Type the text and send it. While it is made the progress reads
*Reading the text*, *Speaking — 3 s so far*, then *Decoding*. The reply
is the text spoken, in the browser's own audio player, with a download
button under it, and a caption with the length, the voice, the language
and the seed:

> *4.6 s, voice 1.7b-designed, language auto, seed 1234567*

A reply the length limit cut ends its caption with *— cut at the length
limit*. Replies are kept beside the session, as WAV, or FLAC when that is
the format, and come back through **History**.

Stop ends the reply at the next frame, and so does closing the tab.

## Through the API

Everything the console does, a client can:

- `POST /v1/audio/speech` is OpenAI's speech API. It takes `input` and
  `voice` (a name, or `{"id": name}`), `response_format` (`wav`, `flac`
  or `pcm`), plus the local knobs `language`, `temperature`, `top_k`,
  `repetition_penalty`, `max_seconds` and `seed`. The answer is the
  audio itself, as `audio/wav`, `audio/flac` or `audio/pcm` (bare
  16-bit little-endian mono at 24 kHz, as OpenAI's), with headers that
  name the voice, the seed, the frames, the sample rate and whether the
  model ended the clip.
- `GET /v1/audio/voices` lists the voices, the languages and the formats,
  and names the default voice.
- `POST /v1/chat/completions` on the same server speaks the last user
  message and answers in OpenAI's audio-output shape: the message's
  `audio` holds the clip as base64, its format, its transcript, the voice
  and its length.
- `GET /props` reports the codec, the voice packs, the voices and the
  defaults, and `POST /props` sets the defaults. The Audio pane's Save
  calls it.

```sh
orangu-server download cstr/qwen3-tts-1.7b-base-GGUF:Q8_0
orangu-server --audio
curl -s http://localhost:8100/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"input": "Hello from orangu.", "voice": "1.7b-designed"}' -o hello.wav
```

The text endpoints that have no speech to give — `/v1/completions`,
`/completion`, `/v1/embeddings` and `/v1/messages` — answer `501` on a
speech model. All of it is documented field by field in the *HTTP
endpoints* chapter.

## What to expect, and what not

- **One clip at a time.** A second request waits for the first.
- **The speech pipeline runs on the CPU.** The talker, the code predictor
  and the codec run on the CPU whatever `backend` and `device` say, and
  none of a language model's startup — placement, probes, the KV pool,
  the warm-up — applies.
- **The clip is sent whole.** It is not streamed as it is spoken:
  `stream_format: "sse"` is refused, and a streamed chat request gets the
  whole clip in one chunk.
- **Voices come from voice packs.** Cloning a voice from your own
  recording needs the codec's encoder and the speaker encoder, which this
  server does not run.
- **No voice design.** A Base model speaks in the voice it is given and
  cannot be steered by a description, so `instructions` is refused.
- **One speed.** `speed` other than `1.0` is refused.
- **WAV, FLAC and PCM only.** `mp3`, `opus` and `aac` are refused, with
  the formats that are supported named.
