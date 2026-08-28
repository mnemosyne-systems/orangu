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
# Shared by every stage. Sourced, never run.
#
# There is deliberately almost nothing here, and nothing at all that a run
# is configured by. Every setting — the size, the context, the weight
# format, the corpus, the schedule — lives in the manifest, not in these
# scripts and not in an environment variable that has to be kept in step
# with it. What is left is finding the binary and finding the file one
# stage hands to the next.

set -eu

KIT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Resolves the binary: the PATH first, then a build tree above this
# directory — so the kit works from an installed release and from a
# checkout, neither of which needs configuring.
if command -v orangu-gguf > /dev/null 2>&1; then
    ORANGU_GGUF="orangu-gguf"
elif [ -x "${KIT_DIR}/../../target/release/orangu-gguf" ]; then
    ORANGU_GGUF="${KIT_DIR}/../../target/release/orangu-gguf"
else
    echo "error: orangu-gguf is not on the PATH" >&2
    echo "       install it, or build it: cargo build --release --bin orangu-gguf" >&2
    exit 1
fi

# The BF16 file the quantization stages read: the one named on the command
# line, or the single `*-BF16.gguf` in this directory.
#
# Found rather than constructed, because its name comes from the manifest —
# rebuilding it here would mean keeping a copy of the manifest's `name` and
# `training_size` in the shell, which is exactly the duplication the
# manifest format exists to remove. The smoke stage keeps its own output in
# `smoke/`, so it never makes this ambiguous.
find_bf16() {
    if [ $# -gt 0 ] && [ -n "$1" ]; then
        echo "$1"
        return 0
    fi
    found=""
    count=0
    for file in ./*-BF16.gguf; do
        [ -f "$file" ] || continue
        found="$file"
        count=$((count + 1))
    done
    if [ "$count" -eq 0 ]; then
        echo "error: no *-BF16.gguf here — run ./10-bf16.sh first," >&2
        echo "       or name the file: $0 path/to/model-BF16.gguf" >&2
        return 1
    fi
    if [ "$count" -gt 1 ]; then
        echo "error: several *-BF16.gguf here — name the one you mean:" >&2
        echo "       $0 path/to/model-BF16.gguf" >&2
        return 1
    fi
    echo "$found"
}
