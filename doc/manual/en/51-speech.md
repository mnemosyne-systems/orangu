\newpage

# Speech recognition

orangu turns speech into text with **Whisper**, OpenAI's speech
recognizer, run on orangu's own engine: the spectrogram, the encoder and
the decoder are all orangu's code, and nothing beyond the models
directory is needed. Whisper hears close to a hundred languages and
detects which one is spoken. This chapter covers which models to use,
how to convert one, and how to transcribe with it.

Speech recognition is new. Today a model converts and a WAV file
transcribes from the command line; the `/v1/audio/transcriptions`
endpoint, push-to-talk in the terminal (Ctrl-S) and a microphone button
in the web console are being built on top of it.

## The models

Any Hugging Face checkpoint of `WhisperForConditionalGeneration`
converts. These are the ones to choose from:

| Model | Parameters | F16 file | Mel bins | Notes |
|---|---|---|---|---|
| `openai/whisper-tiny` | 39M | 77 MB | 80 | Fast; English only in practice |
| `openai/whisper-base` | 74M | 145 MB | 80 | |
| `openai/whisper-small` | 244M | 485 MB | 80 | |
| `openai/whisper-medium` | 769M | 1.5 GB | 80 | |
| `openai/whisper-large-v3` | 1.55B | 3.1 GB | 128 | The most accurate |
| `openai/whisper-large-v3-turbo` | 809M | 1.6 GB | 128 | **Recommended**: large-v3's encoder, a four-layer decoder |

`whisper-large-v3-turbo` is the model to take for anything but English:
it is nearly as accurate as `large-v3` and decodes several times faster.
The small models are useful for trying things out, but they lose most
languages other than English.

The English-only checkpoints (`whisper-tiny.en` and the rest) and
distilled ones (`distil-whisper/*`) have the same architecture and
convert, but have not been tested yet.

## Converting a model

Whisper is published as safetensors, not GGUF, so a model is converted
once with `orangu-gguf`. Download the checkpoint's `config.json`,
`generation_config.json`, `tokenizer.json`, `tokenizer_config.json` and
`model.safetensors` into one directory, or let `orangu-gguf` fetch them
by naming the repository:

```sh
orangu-gguf -m openai/whisper-large-v3-turbo -q f16 -o whisper-large-v3-turbo-f16.gguf
```

`f16` keeps the weights as they are published; the convolutions and the
position tables are small, and quantizing them is not worth it.

## Transcribing a file

```sh
orangu-server transcribe whisper-large-v3-turbo-f16.gguf recording.wav
```

The text goes to standard output, and the detected language and the
timings to standard error:

```
Write a hello world example in Python
language en, 3.5s of audio; load 0.49s, mel 0.05s, encode 57.07s, decode 1.89s (7 tokens)
```

`--language` names the spoken language (`en`, `ar`, `de`, ...) instead
of detecting it. Detection listens to the first thirty seconds, so name
the language when a recording starts with silence or music.

The file is a WAV: 8, 16, 24 or 32-bit integer or 32 and 64-bit float
samples, any number of channels (mixed to mono) and any sample rate
(resampled to Whisper's 16 kHz).

## Languages

Whisper's languages are the ones its tokenizer lists, from `en` to
`yue`. These are the ones tested with orangu so far, each by saying
"Write a Hello World example in Python":

| Language | Code | tiny | large-v3-turbo |
|---|---|---|---|
| English | `en` | Correct | Correct |
| Arabic | `ar` | Not usable | Correct; detected without `--language` |

A language not in the table is very likely to work with
`large-v3-turbo`, but has not been checked yet.

## What to expect, and what not

- **Thirty-second windows.** Whisper reads sound thirty seconds at a
  time. A longer recording is cut into back-to-back windows and the
  texts joined, so a word that falls on a boundary can be cut in two.
- **The CPU only, for now.** The encoder runs on the CPU and takes about
  a minute per thirty seconds with `large-v3-turbo`; the decoder is
  quick. The GPU backends come next.
- **No timestamps yet**, and no `srt` or `vtt` output.
- **WAV only.** FLAC, MP3 and the browser's formats are not read yet.
