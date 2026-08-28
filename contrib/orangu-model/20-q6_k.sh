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
# Stage 2: BF16 to Q6_K.
#
# The largest quantization worth writing: indistinguishable from BF16 on
# anything you can measure, at under half the size. Quantizing always reads
# the BF16 file, never another quantization — rounding twice is worse than
# rounding once from the original, and `orangu-gguf` refuses to do it.
#
#   ./20-q6_k.sh [path/to/model-BF16.gguf]

. "$(dirname "$0")/common.sh"

INPUT="$(find_bf16 "${1:-}")"
OUTPUT="$(echo "$INPUT" | sed 's/-BF16\.gguf$//')-Q6_K.gguf"

"$ORANGU_GGUF" --model "$INPUT" --quantization q6_k --output "$OUTPUT"
