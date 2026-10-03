# orangu in a container

One image with the whole serving stack: `orangu-coordinator` on port 9000 in
front of `orangu-server`, and one model from `orangu-server list` inside it.
Point `orangu` on any machine at it and you're done:

```ini
[container]
endpoint = http://<host>:9000/v1
```

## Build

You need `make`, Docker (with BuildKit, the default since 23.0) or Podman 4.4+,
and an `orangu-server` on this machine that can see your models.

On Windows, GNU make (e.g. `winget install ezwinports.make`) works from
PowerShell, cmd, or Git Bash. The recipes are POSIX shell, so the Makefile
runs them with the `sh` that ships with Git for Windows, found next to
`git.exe` on your PATH.

Copying the model into `build/` is a hard link when the model and this
checkout are on the same drive, and a full copy otherwise.

```sh
cd contrib/docker
make list                 # the models on this machine, with their NR
make MODEL=3              # build orangu:latest with model NR 3
make run                  # start it on http://localhost:9000/v1
```

`MODEL` takes anything `orangu-server show` accepts: an NR, a MODEL name from
`list` (with `:QUANT` when the repo has several), or a path. The Makefile asks
`orangu-server show <MODEL> --path` for the model's file(s) — every shard of a
split model — and copies them into the image.

`orangu-server` and `orangu-coordinator` are compiled from this checkout in
the first stage of the build, so the image matches your sources, unreleased
changes included. The first build compiles everything; later ones reuse
Cargo's cache and recompile only what changed.

| Variable        | Default         | Meaning                                                    |
| --------------- | --------------- | ---------------------------------------------------------- |
| `MODEL`         |                 | NR, MODEL or path from `orangu-server list`                |
| `MODEL_FILES`   |                 | The `.gguf` file(s) instead of `MODEL`, space-separated    |
| `IMAGE`         | `orangu:latest` | The image name                                             |
| `ENGINE`        | `docker`        | `docker` or `podman`                                       |
| `PLATFORM`      | this machine    | e.g. `linux/arm64`                                         |
| `PORT`          | `9000`          | The host port `make run` publishes                         |
| `ORANGU_SERVER` | `orangu-server` | The `orangu-server` used to resolve `MODEL`                |

`MODEL_FILES` is for an `orangu-server` that predates `show --path`, or a
model it doesn't list. Paths with spaces have to go through `MODEL`.

## What's in the image

```text
/opt/orangu/bin/orangu-coordinator     the entrypoint, port 9000
/opt/orangu/bin/orangu-server          started by the coordinator on demand
/opt/orangu/models/<model>.gguf        the model, in a layer of its own
/opt/orangu/orangu-coordinator.conf    one profile, role all, that model
```

- **One profile, role `all`.** It takes every request, whatever model name
  the client asks for. To change it, edit `orangu-coordinator.conf.in`.
- **No model switching.** The server runs without a web console, so nothing
  can load or download another model into the container.
- **The model has its own layer**, before the binaries. Rebuilding after a
  code change reuses it rather than copying gigabytes again.
- **Unprivileged.** Everything runs as user `orangu`. Two volumes:
  `/home/orangu/.orangu` (sessions and the coordinator's state) and
  `/workspace`.
- **CPU only.** The base is `debian:bookworm-slim`.

## Running it yourself

```sh
docker run --rm -it -p 9000:9000 -v orangu-data:/home/orangu/.orangu orangu:latest
```

With Podman, an image built as `orangu:latest` is listed as
`localhost/orangu:latest`. Rootless Podman maps the container's user, so add
`--userns=keep-id` when mounting a host folder, and `:Z` on SELinux systems:

```sh
podman run --rm -it -p 9000:9000 --userns=keep-id \
    -v ~/orangu-data:/home/orangu/.orangu:Z localhost/orangu:latest
```

On Windows and macOS, Podman runs in a VM: `podman machine start` first, and
make sure its disk has room for the model.

## Disk space

While building you need room for the model about three times: the copy in
`build/` (a hard link when the model is on the same filesystem, which costs
nothing), the build context the engine receives, and the image itself.
`build/` is removed after a successful build; `make clean` removes it after a
failed one.

## Licensing

The image contains the model's weights. Check that its license allows
redistribution before pushing the image anywhere public.
