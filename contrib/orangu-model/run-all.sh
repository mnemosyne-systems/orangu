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
# Every build stage in order: smoke test, train, then both quantizations.
#
# Installing is not one of them. `./install.sh` writes outside this
# directory — into whatever models directory `orangu-server.conf` names —
# and it would pick up the smoke model as well as the real one, so it is
# left as a thing you ask for.
#
# The smoke test runs first on purpose. It costs minutes and it is the only
# thing standing between a typo in the manifest and finding out about it a
# week into the training run. To leave it out — or to run any other stage
# on its own — invoke that stage directly rather than this script.

. "$(dirname "$0")/common.sh"

"${KIT_DIR}/00-smoke.sh"
"${KIT_DIR}/10-bf16.sh"
"${KIT_DIR}/20-q6_k.sh"
"${KIT_DIR}/30-q4_k_m.sh"

ls -la ./*.gguf

echo
echo "To make orangu-server able to see and serve these:"
echo
echo "  ./install.sh"

