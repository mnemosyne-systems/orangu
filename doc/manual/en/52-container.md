\newpage

# Container image

`contrib/docker` builds one Podman or Docker image holding the whole serving
stack: `orangu-coordinator` on port 9000, the `orangu-server` it starts on
demand, and the models it serves. The image is built from a coordinator
configuration, and inside it the coordinator works exactly as it does on
your machine: ordinary chat, `/review`, the explorer and `/search` each reach
their own profile, and the coordinator swaps models as they are needed. See
the *Coordinator* chapter for how profiles and routing work.

Point `orangu` on any machine at port 9000 and it has every model the image
was built with, with nothing to install beside the container.

## What you need

- `make`.
- Podman 4.4 or newer, or Docker with BuildKit (the default since Docker
  23.0).
- An `orangu-server` on the build machine that can see your models — the one
  that lists them in `orangu-server list`. The build uses its
  `show --path` to find each model's files.
- The models, already downloaded (`orangu-server download`).

On Windows, GNU make (for example `winget install ezwinports.make`) works from
PowerShell, cmd or Git Bash. Its recipes are POSIX shell, so the Makefile runs
them with the `sh` that ships with Git for Windows, found next to `git.exe` on
the `PATH`.

## Quick start

```sh
cd contrib/docker
make CONFIG=~/.orangu/orangu-coordinator.conf    # build orangu:rocky10
make run                                         # http://localhost:9000/v1
```

`make` with no `CONFIG` uses the coordinator's own configuration, found the
way `orangu-coordinator` finds it: `orangu-coordinator.conf` in
`contrib/docker`, then `~/.orangu/orangu-coordinator.conf`.

## The coordinator configuration

The image always runs `orangu-coordinator`, so it is always built from an
`orangu-coordinator.conf`: the one you already use, or one written for the
image. Every profile in it goes into the image.

```ini
[orangu-coordinator]
host = all
port = 9000
models = ~/.cache/huggingface/hub

[all]
model = unsloth/gemma-4-E2B-it-GGUF:Q4_K_M

[code]
model = unsloth/gemma-4-E2B-it-GGUF:Q4_K_M

[review]
model = unsloth/gemma-4-E2B-it-GGUF:Q4_K_M

[explorer]
model = unsloth/gemma-4-E2B-it-GGUF:Q4_K_M

[embeddings]
model = ggml-org/embeddinggemma-300M-GGUF:Q8_0
```

- `models` is the directory the models are in **on the build machine**. Each
  `model =` is resolved against it, so the labels are the ones
  `orangu-server list` shows.
- A model used by several profiles is copied once.
- An image with a single model is a configuration with a single `[all]`
  profile.

`CONFIG` takes any path: absolute, or relative to `contrib/docker`, where
`make` runs. A leading `~` is expanded, also from PowerShell and cmd, which
leave it alone. The file is read only while building; the image carries its
own copy, and nothing at run time refers back to it.

### What changes in the image's copy

The configuration is copied as written, except where the container needs it
to differ:

| Key | In the image | Why |
| :-- | :-- | :-- |
| `[orangu-coordinator].host` | `all` | Loopback would be unreachable from outside the container |
| `[orangu-coordinator].models` | `/opt/orangu/models` | Where the models are copied to |
| `log_type`, `log_path` | removed | Output goes to `podman logs` / `docker logs`; a host path does not exist in the container |
| a profile's `web` | removed | A web console would let a click load or download another model into the container |

Everything else — ports, `backend`, `slots`, `startup_timeout`,
`idle_timeout`, `shutdown_token` — is kept as it is.

Each model keeps its path relative to the models directory. For a Hugging
Face cache that is `models--<user>--<repo>/snapshots/<revision>/<file>`, and
that path is what a model's label (`user/repo:QUANT`) is read from — so every
`model =` works unchanged inside the image, and so does `/model <label>`. A
model that lies outside the models directory is copied to
`/opt/orangu/models/<file>`, and the profiles naming it are pointed there.

A multi-token-prediction draft head that `download` put in `MTP/` beside a
model is copied with it, to the place the server looks for it.

## Building

```sh
make check  CONFIG=...      # seconds: the distro, the platform, the engine
make stage  CONFIG=...      # copy the models into build/, print the summary
make        CONFIG=...      # check, stage, then build the image
```

`make` runs all three steps; the first two exist on their own so a problem
shows before gigabytes are copied or minutes are spent compiling. Staging
prints what goes in:

```text
Profile    all, code, review, explorer -> unsloth/gemma-4-E2B-it-GGUF:Q4_K_M (2.89 GiB)
Profile    embeddings -> ggml-org/embeddinggemma-300M-GGUF:Q8_0 (318.14 MiB)
Models     3.20 GiB in all
```

and `build/orangu-coordinator.conf` is the configuration the image will run.

`orangu-server` and `orangu-coordinator` are compiled from the checkout in
the first stage of the build, so the image matches the sources, unreleased
changes included. The first build compiles everything and takes a while;
later ones reuse Cargo's cache and recompile only what changed.

### Choosing the base: `DISTRO`

| `DISTRO` | Base image | Platforms |
| :-- | :-- | :-- |
| `rocky10` (default) | `rockylinux/rockylinux:10` | amd64, arm64, and others |
| `debian12` | `debian:bookworm-slim` | amd64, arm64 |
| `archlinux` | `archlinux:latest` | amd64 only |

```sh
make DISTRO=debian12 CONFIG=...     # builds orangu:debian12
```

Each distro has its own `Dockerfile.<DISTRO>`. The binaries are compiled once,
on Debian 12, and copied into whichever base is chosen: Debian 12's C library
is older than Rocky 10's and Arch's, so they run on all three, and building a
second distro reuses the first one's compile.

### Podman or Docker

`ENGINE=auto`, the default, uses Podman if `podman version` answers and
Docker otherwise. `ENGINE=podman` or `ENGINE=docker` forces one. The
`version` check fails both when a tool is not installed and when it is not
running — Docker Desktop stopped, or Podman's VM not started — so `make
check` reports either before anything is copied.

### Variables

| Variable | Default | Meaning |
| :-- | :-- | :-- |
| `CONFIG` | the coordinator's own | The `orangu-coordinator.conf` the image runs |
| `DISTRO` | `rocky10` | `rocky10`, `debian12` or `archlinux` |
| `IMAGE` | `orangu:<DISTRO>` | The image name |
| `ENGINE` | `auto` | `auto`, `podman` or `docker` |
| `PLATFORM` | this machine | For example `linux/arm64` |
| `PORT` | `9000` | The host port `make run` publishes |
| `ORANGU_SERVER` | `orangu-server` | The `orangu-server` that resolves the models |

`make list` runs `orangu-server list`, for the labels to put in `model =`.

## Running the container

```sh
make run                                  # in the foreground; Ctrl+C stops it
podman run -d --name orangu -p 9000:9000 -v orangu-data:/home/orangu/.orangu orangu:rocky10
docker run -d --name orangu -p 9000:9000 -v orangu-data:/home/orangu/.orangu orangu:rocky10
docker logs -f orangu                     # profiles at startup, then every model swap
```

The image runs as the unprivileged user `orangu`, and has two volumes:
`/home/orangu/.orangu`, for sessions and the coordinator's state, and
`/workspace`. Only port 9000 is published; the servers inside are started by
the coordinator and reached through it. With port 9000 already taken, use
`make run PORT=9100` or `-p 9100:9000`.

Podman lists an image built as `orangu:rocky10` as
`localhost/orangu:rocky10`. Rootless Podman maps the container's user, so add
`--userns=keep-id` when mounting a host folder, and `:Z` on SELinux systems
such as Fedora and RHEL:

```sh
podman run -d -p 9000:9000 --userns=keep-id \
    -v ~/orangu-data:/home/orangu/.orangu:Z localhost/orangu:rocky10
```

On Windows and macOS, Podman runs in a VM: run `podman machine start` first,
and make sure its disk has room for the models.

The coordinator runs one server at a time, so the machine running the image
needs memory for the largest model, not for all of them together.

## Connecting orangu

Give `orangu` a configuration that points at the container. It can live
anywhere, and is passed with `-c`:

```ini
[orangu]
server = container
model = unsloth/gemma-4-E2B-it-GGUF:Q4_K_M

[container]
endpoint = http://localhost:9000/v1
```

```sh
orangu -c orangu-container.conf -p "Reply with one word: ready"   # one-shot check
orangu -c orangu-container.conf                                   # the TUI
```

Compared with a configuration for a local `orangu-server`:

| In `orangu.conf` | Local `orangu-server` | The container |
| :-- | :-- | :-- |
| Section name | `[main-server]` | `[container]`, or any name |
| `endpoint` port | `8100`, the server | `9000`, the coordinator |
| Server sections | one per role, each with `role =` | one, without `role =` |
| `model` | the model the server loaded | one of the image's profile models |

- **The section name is only a label.** `[orangu].server` picks a section by
  name; `main-server` is what `orangu --init` writes, and nothing depends on
  it.
- **Port 9000, not 8100.** The client talks to the coordinator only.
- **One section is enough.** The client detects the coordinator, ignores
  `role =`, and leaves the choice of model for each task to it.
- **`model`** should be one of the image's models. A model the image does not
  have falls back to the `all` profile; nothing is downloaded into the
  container.
- **A separate file** leaves `~/.orangu/orangu.conf` pointing at your local
  server. An `orangu.conf` in the project directory works too: `orangu` reads
  it before the one in `~/.orangu`.

From another machine, use the host's address — `http://<host>:9000/v1` — with
port 9000 open in its firewall.

## How the build works

```text
make check   ->  the distro exists, the platform suits it, an engine answers
make stage   ->  stage.sh: the models and the configuration into build/
build        ->  podman/docker build -f Dockerfile.<DISTRO> --build-context stage=build/
```

| File | Role |
| :-- | :-- |
| `Makefile` | Reads the variables, picks the engine, runs the steps |
| `stage.sh` | Collects the models and the coordinator configuration into `build/` |
| `Dockerfile.<DISTRO>` | Compiles the binaries, then assembles the image |
| `Dockerfile.<DISTRO>.dockerignore` | What the compile does not see |

`stage.sh` reads the configuration's `models` directory and every distinct
`model =`, resolves each with `orangu-server show <model> --path`, copies the
files with their relative paths — a hard link when the models are on the same
file system as the checkout, a copy otherwise — and writes the adjusted
configuration. It also handles what Windows adds: `C:\` paths from
`orangu-server.exe`, CRLF line ends, and `$HOME` in Git for Windows' `sh`.

The build has two inputs: the repository, for the compile, and `build/`,
passed as a separate `stage` build context, for the models and the
configuration. The models never pass through the repository's context, so
they neither slow the compile nor invalidate its cache.

The compile's cache follows the files it is given. Its first step copies the
repository, so any change to a file it copies makes the compile run again —
only the orangu crate itself, since Cargo's `target/` is a cache mount that
survives. The `.dockerignore` files keep `target/`, `.git`, models and
`contrib/docker` out of that copy, so editing a Dockerfile, `stage.sh` or the
configuration recompiles nothing. Docker reads `<Dockerfile>.dockerignore`
beside the Dockerfile it is given, hence one per distro, all alike; Podman is
handed the same file with `--ignorefile`.

## Troubleshooting

| Message | Cause and fix |
| :-- | :-- |
| `no coordinator config` | Pass `CONFIG=`, or create `~/.orangu/orangu-coordinator.conf` (`orangu-coordinator --init`) |
| `CONFIG=... does not exist` | A wrong path; a relative one starts from `contrib/docker` |
| `could not resolve '<model>'` | The label is not in `orangu-server list`, or `models =` names another directory |
| `... is a picture model` | Picture generators need a text encoder and VAE beside them, which cannot be staged |
| `neither podman nor docker answers` | Start Docker Desktop, or `podman machine start` |
| `no Dockerfile.<name>` | `DISTRO` is not one of `rocky10`, `debian12`, `archlinux` |
| `the archlinux base image is linux/amd64 only` | Use another `DISTRO` for `linux/arm64` |
| The build stops with `EOF` while sending the models | The engine ran out of memory; give Docker Desktop or the Podman VM more |
| `port is already allocated` | Something else uses 9000: `PORT=9100`, or `-p 9100:9000` |

A failed build keeps `build/`; `make clean` removes it.

## Limits

- **CPU only.** The images carry no GPU runtime.
- **Picture models** (`qwen_image`, `qwen_image_2_1`) are refused.
- **Disk space.** Building needs room for the models about three times: the
  copy in `build/` (free when it is a hard link), the context the engine
  receives, and the image.
- **Licensing.** The image contains the models' weights; check that their
  licenses allow redistribution before pushing it anywhere public.
