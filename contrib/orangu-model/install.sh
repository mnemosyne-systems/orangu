#!/bin/sh
# Copyright (C) 2026 The orangu community
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Stage 4: put the finished models where `orangu-server` looks for them.
#
# Reads the models directory out of `orangu-server.conf` — the same file and
# the same key the server itself reads, so there is nothing here to keep in
# step with it — and installs each `*.gguf` into it in Hugging Face's hub
# cache layout:
#
#   <models>/models--<org>--<model>-<size>-GGUF/
#       refs/main                       the snapshot's revision
#       blobs/<sha256>                  the file, named by its contents
#       snapshots/<rev>/<file>.gguf     a symlink to the blob
#
# That layout is not decoration. `orangu-server list` finds any `.gguf`
# anywhere under the models directory, but it only shows a model as
# `<org>/<name>:<QUANT>` — the form `--model` and `download` accept — when
# it can read an org and a name out of a `models--<org>--<name>` directory
# on the path. Installed flat, the same file lists under its bare filename
# and cannot be named as a spec at all.
#
# One repository per *training size*, named `<model>-<size>-GGUF`, with every
# quantization of that size inside it — which is the shape a published GGUF
# repository has, and which makes `list` show the three files as three rows
# of one model rather than as three unrelated models. A `smoke` build and a
# `2b` build are different models and get different repositories.
#
# Usage:
#   ./install.sh                    every *.gguf here and in ./smoke/
#   ./install.sh a.gguf b.gguf      just those
#
# Environment:
#   ORANGU_CONF        configuration file (default: the server's own search)
#   ORANGU_MODEL_ORG   organisation part of the repository id
#                      (default: mnemosynesystems, this project's own)
#   ORANGU_MODEL_NAME  model part of it (default: orangu)

. "$(dirname "$0")/common.sh"

# The organisation these models are published under. It is a real one, and
# that is deliberate rather than incidental: `list` checks a repository it
# recognises against Hugging Face, comparing the *blob* names — which are
# content hashes, and which Hugging Face's own LFS object ids are too. So a
# model installed here under the name it is published under compares
# correctly against the published copy, and says nothing at all until there
# is one.
ORG="${ORANGU_MODEL_ORG:-mnemosynesystems}"
MODEL="${ORANGU_MODEL_NAME:-orangu}"

# ---------------------------------------------------------------- the config
#
# The server looks for `./orangu-server.conf` first and `~/.orangu/` second;
# this looks in the same order, for the same reason — a checkout with its own
# configuration should not silently install into the home one.
if [ -n "${ORANGU_CONF:-}" ]; then
    CONF="$ORANGU_CONF"
elif [ -f ./orangu-server.conf ]; then
    CONF="./orangu-server.conf"
elif [ -f "$HOME/.orangu/orangu-server.conf" ]; then
    CONF="$HOME/.orangu/orangu-server.conf"
else
    echo "error: no orangu-server.conf in . or ~/.orangu" >&2
    echo "       run 'orangu-server -i' to write one, or set ORANGU_CONF" >&2
    exit 1
fi

# `models` from the `[orangu-server]` section, honouring full-line and
# inline comments the way the server's own parser does. A key of the same
# name in another section is not this one.
MODELS=$(awk '
    /^[[:space:]]*[#;]/ { next }
    /^[[:space:]]*\[/ {
        section = $0
        sub(/^[[:space:]]*\[/, "", section)
        sub(/\][[:space:]]*$/, "", section)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", section)
        next
    }
    section == "orangu-server" && /=/ {
        key = $0
        sub(/=.*/, "", key)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
        if (key != "models") next
        value = $0
        sub(/^[^=]*=/, "", value)
        sub(/[[:space:]][#;].*$/, "", value)
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
        print value
        exit
    }
' "$CONF")

if [ -z "$MODELS" ]; then
    echo "error: [orangu-server].models is not set in $CONF" >&2
    exit 1
fi

# A leading `~` is the one path expansion the server does, so it is the one
# this does too.
case "$MODELS" in
    "~") MODELS="$HOME" ;;
    "~/"*) MODELS="$HOME/${MODELS#~/}" ;;
esac

if [ ! -d "$MODELS" ]; then
    echo "error: models directory $MODELS does not exist (from $CONF)" >&2
    exit 1
fi

echo "config      $CONF"
echo "models      $MODELS"
echo "repository  $ORG/$MODEL-<size>-GGUF"
echo

# ----------------------------------------------------------------- the files
if [ $# -gt 0 ]; then
    FILES="$*"
else
    FILES=""
    for candidate in ./*.gguf ./smoke/*.gguf; do
        [ -f "$candidate" ] || continue
        FILES="$FILES $candidate"
    done
fi

if [ -z "$(printf '%s' "$FILES" | tr -d ' ')" ]; then
    echo "error: no *.gguf here or in ./smoke/ — run a build stage first" >&2
    exit 1
fi

# The repository a file belongs to.
#
# `orangu-gguf` names what it writes `<manifest name>-<size>-<FORMAT>.gguf`,
# so the size is the last field once the format is off the end. The format
# is the run of capitals, digits and underscores after the final `-` or `.`,
# which is the same tag `list` reads back out of the filename for its QUANT
# column — so stripping exactly that leaves exactly the size, and `0.5b`
# survives it because the tag has to be upper case.
#
# `orangu-code-smoke-BF16.gguf` and `orangu-code-smoke-Q4_K_M.gguf` are
# therefore two files of `orangu-smoke-GGUF`.
size_of() {
    stem=$(basename "$1" .gguf)
    base=$(printf '%s' "$stem" | sed -E 's/[-.][A-Z0-9_]+$//')
    [ -n "$base" ] || base="$stem"
    # The last `-`-separated field, or the whole name if there is only one.
    printf '%s' "${base##*-}"
}

# One name per line: this is read through `sort -u`, and a `printf` without
# the newline runs every name into one.
repo_of() {
    printf '%s-%s-GGUF\n' "$MODEL" "$(size_of "$1")"
}

# Which repositories this run touches, once each.
REPOS=$(for file in $FILES; do repo_of "$file"; done | sort -u)

STAGED=$(mktemp -d)
trap 'rm -rf "$STAGED"' EXIT

for repo in $REPOS; do
    dir="$MODELS/models--$ORG--$repo"

    # A directory this script did not create is not its to reorganise, and
    # since the organisation above is a real one this is not hypothetical:
    # `orangu-server download mnemosynesystems/...` writes the same path.
    # Rewriting `snapshots/` under a repository the download cache owns
    # would lose whatever it had put there.
    if [ -d "$dir" ] && [ ! -f "$dir/.orangu-installed" ]; then
        echo "error: $dir exists and was not installed by this script" >&2
        echo "       it looks like a downloaded copy of the same repository." >&2
        echo "       Remove it, or set ORANGU_MODEL_ORG to install elsewhere." >&2
        exit 1
    fi

    # Content-address every file first, so the revision can be derived from
    # what is actually in the repository rather than from the clock. Two
    # installs of an unchanged model land on the same revision and do
    # nothing; a retrained model gets a new one, which is what a revision is
    # for.
    manifest="$STAGED/$repo.manifest"
    : > "$manifest"
    for file in $FILES; do
        [ "$(repo_of "$file")" = "$repo" ] || continue
        sha=$(sha256sum "$file" | cut -d' ' -f1)
        printf '%s %s %s\n' "$(basename "$file")" "$sha" "$file" >> "$manifest"
    done
    sort -o "$manifest" "$manifest"

    rev=$(cut -d' ' -f1,2 "$manifest" | sha256sum | cut -c1-40)

    mkdir -p "$dir/blobs" "$dir/snapshots/$rev" "$dir/refs"
    : > "$dir/.orangu-installed"

    while read -r name sha source; do
        blob="$dir/blobs/$sha"
        if [ ! -f "$blob" ]; then
            # Into place under a temporary name: an interrupted copy that
            # kept the blob's name would be a corrupt file that every later
            # run treats as already installed, because the name is the only
            # thing checked.
            cp "$source" "$blob.partial"
            mv "$blob.partial" "$blob"
        fi
        ln -sfn "../../blobs/$sha" "$dir/snapshots/$rev/$name"
        echo "  $ORG/$repo:$(printf '%s' "${name%.gguf}" | sed -E 's/.*[-.]([A-Z0-9_]+)$/\1/')  <- $source"
    done < "$manifest"

    printf '%s\n' "$rev" > "$dir/refs/main"

    # Older revisions of a repository this script owns, and any blob nothing
    # points at any more. Left in place they would list as extra rows of a
    # model that no longer exists.
    for old in "$dir/snapshots/"*; do
        [ -d "$old" ] || continue
        # Not `[ ... ] && continue`: under `set -e` a false test there is a
        # failed statement, and the script would exit on the revision it
        # just wrote.
        if [ "$(basename "$old")" != "$rev" ]; then
            echo "  (removing previous revision $(basename "$old"))"
            rm -rf "$old"
        fi
    done
    for blob in "$dir/blobs/"*; do
        [ -f "$blob" ] || continue
        if ! cut -d' ' -f2 "$manifest" | grep -qxF "$(basename "$blob")"; then
            rm -f "$blob"
        fi
    done
done

echo
echo "Installed. They are now in the list the server serves from:"
echo
echo "  orangu-server list"
echo
echo "and can be served by name, without a path:"
echo
for repo in $REPOS; do
    echo "  orangu-server --model $ORG/$repo:Q4_K_M"
done
echo
echo "Until the same model is published to https://huggingface.co/$ORG,"
echo "'list' finds nothing to compare against and says nothing about updates."
echo "Once it is, it compares the file contents and marks the row (Refresh)"
echo "if what is published differs from what is installed here."
