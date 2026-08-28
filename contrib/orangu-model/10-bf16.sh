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
# Stage 1: the training run. Random weights in, a BF16 GGUF out.
#
# Everything about the run is in `corpus.json` — the corpus, the size, the
# context length, the weight format, the schedule. Edit that file, not this
# script. Anything passed here is handed straight to `orangu-gguf`, so a
# one-off override still works:
#
#   ./10-bf16.sh -ts 0.5b
#
# This is the long one. `2b` is weeks of continuous CPU compute and wants
# ~32 GB of memory for the weights, their gradients and the optimizer's two
# moments; the corpus clone alone is several gigabytes of disk.
#
# It is safe to interrupt: a checkpoint is written every 200 steps, and
# `"resume": true` in the manifest continues from it.

. "$(dirname "$0")/common.sh"

# A source whose licence is not an OSI-approved one is left out of the
# corpus rather than stopping the run, so the only licence failure that
# reaches here is every source being excluded. Say where the answer lives —
# in the manifest, beside the declarations it overrides.
if ! "$ORANGU_GGUF" "${KIT_DIR}/corpus.json" "$@"; then
    status=$?
    echo >&2
    echo "hint: if sources were excluded for their licence, the answer is a line" >&2
    echo "      in the manifest:  \"allow_any_license\": true" >&2
    echo "      See this directory's README." >&2
    exit $status
fi
