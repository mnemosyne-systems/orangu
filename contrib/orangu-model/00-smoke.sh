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
# Stage 0: prove the whole pipeline works here, in minutes.
#
# `corpus-smoke.json` is a four-repository corpus of about 20 MB at the
# `smoke` size — clone, tokenizer, packing, training, and a written GGUF
# that `orangu-server` loads and generates from. It is a *pipeline* test,
# not a model: 200 steps on 20 MB teaches nothing, and the output is meant
# to be gibberish. What it proves is that every stage runs on this machine,
# which is worth knowing before committing days to stage 1.
#
# Everything lands in ./smoke/, so it never collides with the real run's
# files or its work directory.

. "$(dirname "$0")/common.sh"

mkdir -p smoke
BF16="smoke/orangu-code-smoke-BF16.gguf"

"$ORANGU_GGUF" "${KIT_DIR}/corpus-smoke.json" --output "$BF16" "$@"

"$ORANGU_GGUF" --model "$BF16" --quantization q6_k \
    --output "smoke/orangu-code-smoke-Q6_K.gguf"
"$ORANGU_GGUF" --model "$BF16" --quantization q4_k_m \
    --output "smoke/orangu-code-smoke-Q4_K_M.gguf"

echo "Serve it to confirm the file loads and generates:"
echo
echo "  orangu-server ${BF16} --port 8100"
echo
echo "The web console is then on http://localhost:8200. Expect nonsense out"
echo "of it: this model has ~8M parameters and saw 200 steps, so what you"
echo "are checking is that a prompt goes in and tokens come back, not what"
echo "they say. Then run ./10-bf16.sh for the real thing."
