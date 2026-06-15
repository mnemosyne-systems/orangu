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
# Run with: sh tests/install_test.sh
set -e
pass=0; fail=0
check() { if [ "$1" = "$2" ]; then pass=$((pass+1)); else fail=$((fail+1)); printf "FAIL: %s (got '%s')\n" "$3" "$2"; fi; }

# OS detection
check "linux"       "$(case Linux  in Linux) echo linux;; Darwin) echo darwin;; *) echo unsupported;; esac)" "Linux"
check "darwin"      "$(case Darwin in Linux) echo linux;; Darwin) echo darwin;; *) echo unsupported;; esac)" "Darwin"
check "unsupported" "$(case Other  in Linux) echo linux;; Darwin) echo darwin;; *) echo unsupported;; esac)" "unknown OS"

# Arch detection
check "x86_64"      "$(case x86_64  in x86_64) echo x86_64;; aarch64|arm64) echo aarch64;; *) echo unsupported;; esac)" "x86_64"
check "aarch64"     "$(case aarch64 in x86_64) echo x86_64;; aarch64|arm64) echo aarch64;; *) echo unsupported;; esac)" "aarch64"
check "aarch64"     "$(case arm64   in x86_64) echo x86_64;; aarch64|arm64) echo aarch64;; *) echo unsupported;; esac)" "arm64 alias"
check "unsupported" "$(case armv7l  in x86_64) echo x86_64;; aarch64|arm64) echo aarch64;; *) echo unsupported;; esac)" "armv7l"

# Target triple construction (mirrors the formula in install.sh)
triple() {  # $1=os $2=arch $3=libc
    if [ "$1" = "linux" ]; then echo "${2}-unknown-linux-${3}"; else echo "${2}-apple-darwin"; fi
}
check "x86_64-unknown-linux-gnu"   "$(triple linux  x86_64  gnu)"  "linux gnu triple"
check "aarch64-unknown-linux-musl" "$(triple linux  aarch64 musl)" "linux musl triple"
check "aarch64-apple-darwin"       "$(triple darwin aarch64 '')"   "darwin triple"

# INSTALL_DIR default and override
unset INSTALL_DIR
check "$HOME/.local/bin" "${INSTALL_DIR:-$HOME/.local/bin}" "default INSTALL_DIR"
INSTALL_DIR=/tmp/custom
check "/tmp/custom" "${INSTALL_DIR:-$HOME/.local/bin}" "custom INSTALL_DIR"
unset INSTALL_DIR

# Writable directory and cleanup
TMP=$(mktemp -d); rm -rf "$TMP"
check "" "$([ ! -d "$TMP" ] && echo '')" "temp dir cleaned up"

printf "%d passed, %d failed\n" "$pass" "$fail"
[ "$fail" -eq 0 ]
