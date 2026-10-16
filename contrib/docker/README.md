# orangu in a container

One image with the whole serving stack: `orangu-coordinator` on port 9000 in
front of `orangu-server`, and every model a coordinator configuration names.
Inside, the coordinator switches models by task as it does on your machine —
`/review` gets the review model, the explorer its own, `/search` the
embeddings model.

The full guide is the **Container image** chapter of the manual,
[`doc/manual/en/50-container.md`](../../doc/manual/en/50-container.md) (also
`/manual` in `orangu`). This is the short version.

## Build and run

You need `make`, Podman 4.4+ or Docker with BuildKit, and an `orangu-server`
that lists your models.

```sh
cd contrib/docker
make CONFIG=~/.orangu/orangu-coordinator.conf    # build orangu:rocky10
make run                                         # http://localhost:9000/v1
```

`CONFIG` is the `orangu-coordinator.conf` the image runs; every profile in
it goes in. Without it, the coordinator's own configuration is used. A single
model is a configuration with one `[all]` profile:

```ini
[orangu-coordinator]
models = ~/.cache/huggingface/hub

[all]
model = unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `CONFIG` | the coordinator's own | The `orangu-coordinator.conf` the image runs |
| `DISTRO` | `rocky10` | `rocky10`, `debian12` or `archlinux` |
| `IMAGE` | `orangu:<DISTRO>` | The image name |
| `ENGINE` | `auto` | `auto` (Podman if it answers, else Docker), `podman`, `docker` |
| `PLATFORM` | this machine | e.g. `linux/arm64` (not with `archlinux`) |
| `PORT` | `9000` | The host port `make run` publishes |
| `ORANGU_SERVER` | `orangu-server` | The `orangu-server` that resolves the models |

Other targets: `make list` (the model labels), `make check` (distro,
platform and engine, in seconds), `make stage` (copy the models into `build/`
and print what would go in, without building), `make clean`.

## Connect orangu

```ini
# orangu-container.conf
[orangu]
server = container
model = unsloth/gemma-4-E2B-it-GGUF:Q4_K_M

[container]
endpoint = http://localhost:9000/v1
```

```sh
orangu -c orangu-container.conf
```

One server section, on the coordinator's port 9000, without `role =`: the
client detects the coordinator and leaves the model for each task to it.

## Files

| File | Role |
| --- | --- |
| `Makefile` | Reads the variables, picks the engine, runs check, stage and build |
| `stage.sh` | Copies the models and writes the image's coordinator configuration into `build/` |
| `Dockerfile.<DISTRO>` | Compiles the binaries on Debian 12, then assembles the image on that base |
| `Dockerfile.<DISTRO>.dockerignore` | Keeps `target/`, `.git`, models and `contrib/docker` out of the compile |
