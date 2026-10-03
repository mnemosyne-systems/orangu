# syntax=docker/dockerfile:1
#
# The orangu stack in one image: orangu-coordinator on port 9000 in front of
# orangu-server, serving one model chosen from `orangu-server list`.
#
# Built by the Makefile next to this file, which stages the model and the
# coordinator config and passes them in as the `stage` build context:
#
#   make -C contrib/docker MODEL=1
#
# The build context is the repository root; Dockerfile.dockerignore keeps
# target/, .git/ and the staged model out of it.

ARG RUST_VERSION=1
ARG DEBIAN_RELEASE=bookworm

# --- Build: orangu-server and orangu-coordinator from this checkout ---------
FROM docker.io/library/rust:${RUST_VERSION}-${DEBIAN_RELEASE} AS build

# There is no .git in the build context, so the commit the binaries report
# (orangu::build_info) is handed in instead — see src/build.rs.
ARG ORANGU_BUILD_COMMIT=unknown
ENV ORANGU_BUILD_COMMIT=${ORANGU_BUILD_COMMIT}

WORKDIR /src
COPY . .
# The registry and target directory are cache mounts, so a rebuild after a
# source change recompiles only what changed. target/ is not in the image,
# hence the copy out of it.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/src/target \
    cargo build --release --locked --bin orangu-server --bin orangu-coordinator \
    && mkdir -p /out \
    && cp target/release/orangu-server target/release/orangu-coordinator /out/

# --- Runtime -----------------------------------------------------------------
FROM docker.io/library/debian:${DEBIAN_RELEASE}-slim

# An unprivileged user, and the directories that become volumes, created and
# owned by it before VOLUME declares them — a volume starts as a copy of what
# the image has there, and a root-owned one would be unwritable.
#
# ca-certificates: the HTTP client both binaries build at startup loads the
# system's root certificates, and refuses to start without any.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --home-dir /home/orangu orangu \
    && mkdir -p /home/orangu/.orangu /workspace \
    && chown orangu:orangu /home/orangu/.orangu /workspace

# The model first, in a layer of its own: it is by far the largest, and
# changes least often. A rebuild after a code change reuses it.
COPY --from=stage models/ /opt/orangu/models/

# The coordinator starts orangu-server from next to itself.
COPY --from=build /out/orangu-server /out/orangu-coordinator /opt/orangu/bin/
COPY --from=stage orangu-coordinator.conf /opt/orangu/orangu-coordinator.conf

USER orangu
WORKDIR /workspace
VOLUME ["/home/orangu/.orangu", "/workspace"]
EXPOSE 9000
ENTRYPOINT ["/opt/orangu/bin/orangu-coordinator", "--config", "/opt/orangu/orangu-coordinator.conf"]
