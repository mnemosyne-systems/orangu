#!/bin/sh
# Stages what an orangu image carries into one folder, the `stage` build
# context the Dockerfiles copy from:
#
#   <stage>/models/...                    every model the profiles use
#   <stage>/orangu-coordinator.conf       the coordinator config, for /opt/orangu
#
# Run by the Makefile (`make stage`); the variables are the Makefile's:
#
#   CONFIG          the orangu-coordinator.conf the image runs: every profile
#                   in it, each model resolved against its `models` directory.
#                   Unset, the coordinator's own lookup is used.
#   ORANGU_SERVER   the orangu-server that resolves models (show --path)
#
# A model keeps its path relative to the models directory — for a Hugging
# Face cache that is `models--<user>--<repo>/snapshots/<rev>/<file>` — because
# that path is what its label (`user/repo:QUANT`) is read from. The config's
# `model =` values therefore work unchanged inside the image. A model outside
# the models directory goes to /opt/orangu/models/<file>, and the profiles
# naming it are pointed there.

set -eu

STAGE=${1:?usage: stage.sh <stage-dir>}
CONFIG=${CONFIG:-}
ORANGU_SERVER=${ORANGU_SERVER:-orangu-server}
IMAGE_MODELS=/opt/orangu/models

die() {
    echo "error: $*" >&2
    exit 1
}

# Forward slashes and no carriage return: orangu-server.exe prints Windows
# paths, and a config written on Windows has CRLF line ends.
normalize() {
    printf '%s' "$1" | tr -d '\r' | tr '\134' '/'
}

# Hard link when the model is on the same filesystem as the stage — free —
# and a copy otherwise. -L/-L: a Hugging Face cache on Linux names files by
# symlinks into blobs/.
place() {
    mkdir -p "$(dirname "$2")"
    [ -e "$2" ] && return 0
    ln -L "$1" "$2" 2>/dev/null || cp -L "$1" "$2"
}

# A path a native Windows program can open, from Git for Windows' sh, whose
# $HOME is /c/Users/...; unchanged everywhere else.
native() {
    if command -v cygpath > /dev/null 2>&1; then cygpath -m "$1"; else printf '%s' "$1"; fi
}

# orangu-server, resolving against the config's models directory rather than
# the one orangu-server's own config names.
resolve() {
    "$ORANGU_SERVER" --config "$SERVER_CONF" "$@"
}

human() {
    awk -v b="$1" 'BEGIN {
        split("B KiB MiB GiB TiB", u); i = 1
        while (b >= 1024 && i < 5) { b /= 1024; i++ }
        printf (i == 1 ? "%d %s" : "%.2f %s"), b, u[i]
    }'
}

# --- Which config ------------------------------------------------------------
# CONFIG, or where orangu-coordinator itself looks: ./orangu-coordinator.conf,
# then ~/.orangu/orangu-coordinator.conf.
if [ -n "$CONFIG" ]; then
    # A leading ~ is expanded here: PowerShell and cmd pass it through as is.
    case $CONFIG in
        "~"|"~"/*|"~"\\*) CONFIG=$HOME${CONFIG#"~"} ;;
    esac
    [ -f "$CONFIG" ] || die "CONFIG=$CONFIG does not exist"
    SOURCE=$CONFIG
elif [ -f ./orangu-coordinator.conf ]; then
    SOURCE=./orangu-coordinator.conf
elif [ -f "$HOME/.orangu/orangu-coordinator.conf" ]; then
    SOURCE=$HOME/.orangu/orangu-coordinator.conf
else
    die "no coordinator config: set CONFIG to an orangu-coordinator.conf, \
or create ~/.orangu/orangu-coordinator.conf (orangu-coordinator --init)"
fi

rm -rf "$STAGE"
mkdir -p "$STAGE/models"
TMP="$STAGE/.work"
mkdir -p "$TMP"

# --- The models it names -----------------------------------------------------
# The models directory, and every model (one per line, deduplicated, in
# first-use order), read from the config's INI.
echo "Config     $SOURCE"
tr -d '\r' < "$SOURCE" > "$TMP/source.conf"
MODELS_DIR=$(awk -F= '
    /^[ \t]*\[/ { section = $0; gsub(/[][ \t]/, "", section); next }
    section == "orangu-coordinator" && $1 ~ /^[ \t]*models[ \t]*$/ {
        value = substr($0, index($0, "=") + 1)
        gsub(/^[ \t]+|[ \t]+$/, "", value); print value
    }' "$TMP/source.conf")
[ -n "$MODELS_DIR" ] || die "$SOURCE has no [orangu-coordinator] models = ..."
case $MODELS_DIR in
    "~"/*) MODELS_DIR=$(native "$HOME")/${MODELS_DIR#"~/"} ;;
esac
awk -F= '
    /^[ \t]*[#;]/ { next }
    /^[ \t]*\[/ { section = $0; gsub(/[][ \t]/, "", section); next }
    section != "orangu-coordinator" && $1 ~ /^[ \t]*model[ \t]*$/ {
        value = substr($0, index($0, "=") + 1)
        gsub(/^[ \t]+|[ \t]+$/, "", value)
        if (value != "" && !seen[value]++) print value
    }' "$TMP/source.conf" > "$TMP/models.txt"
[ -s "$TMP/models.txt" ] || die "$SOURCE has no profiles with a model"
printf '[orangu-server]\nmodels = %s\n' "$MODELS_DIR" > "$TMP/server.conf"
SERVER_CONF=$(native "$TMP/server.conf")
MODELS_DIR=$(normalize "$MODELS_DIR")
MODELS_DIR=${MODELS_DIR%/}

# --- Copy every model --------------------------------------------------------
# rewrites.txt: "<model as configured>\t<new value>", for models outside the
# models directory, which lose the path their label was read from.
: > "$TMP/rewrites.txt"
: > "$TMP/summary.txt"
TOTAL=0
while IFS= read -r model; do
    # An image model needs a text encoder and VAE beside it that
    # `show --path` doesn't list; refused rather than shipped broken.
    if resolve show "$model" 2>/dev/null | grep -q 'general.architecture *= *"qwen_image'; then
        die "$model is a picture model; its companion files can't be staged, \
so it can't go in an image"
    fi
    resolve show "$model" --path > "$TMP/files.txt" \
        || die "could not resolve '$model'; check it with 'orangu-server show $model'"

    first=
    bytes=0
    while IFS= read -r file; do
        file=$(normalize "$file")
        [ -n "$file" ] || continue
        [ -f "$file" ] || die "'$model' resolved to $file, which does not exist"
        # The path inside the image's models directory.
        case $file in
            */models--*) rel=models--${file#*/models--} ;;
            "$MODELS_DIR"/*) rel=${file#"$MODELS_DIR"/} ;;
            *) rel=$(basename "$file") ;;
        esac
        echo "Model      $file"
        place "$file" "$STAGE/models/$rel"
        size=$(wc -c < "$file" | tr -d ' ')
        bytes=$((bytes + size))
        [ -n "$first" ] || first=$rel
    done < "$TMP/files.txt"
    [ -n "$first" ] || die "'$model' resolved to no files"
    TOTAL=$((TOTAL + bytes))

    # A file that landed at the top of the models directory is named by its
    # path in the image, which always resolves.
    case $first in
        */*) ;;
        *) printf '%s\t%s\n' "$model" "$IMAGE_MODELS/$first" >> "$TMP/rewrites.txt" ;;
    esac

    # A multi-token-prediction draft head that `download` put in MTP/ beside
    # the model (or beside its per-quant folder): copied to the same place,
    # where the server looks. Without it the image serves correctly, only
    # slower.
    src_dir=$(dirname "$(normalize "$(head -n 1 "$TMP/files.txt")")")
    rel_dir=$(dirname "$first")
    for pair in "$src_dir/MTP|$rel_dir/MTP" "$src_dir/../MTP|$(dirname "$rel_dir")/MTP"; do
        heads=${pair%%|*}
        target=${pair#*|}
        case $target in ../*|..) continue ;; esac
        ls "$heads"/mtp-*.gguf > /dev/null 2>&1 || continue
        for head in "$heads"/mtp-*.gguf; do
            echo "MTP head   $head"
            place "$head" "$STAGE/models/$target/$(basename "$head")"
        done
        break
    done

    printf '%s\t%s\n' "$model" "$bytes" >> "$TMP/summary.txt"
done < "$TMP/models.txt"

# --- The coordinator config --------------------------------------------------
# The config as written, made to fit the container: [orangu-coordinator]
# listens on every interface, reads models from /opt/orangu/models, and logs
# to the console (docker/podman logs) — a log_path on the host means nothing
# in there. Profiles lose `web`: a console would let a click load or download
# another model into the container. A model that moved is pointed at its new
# path.
SOURCE_NAME=$SOURCE awk -F= -v models="$IMAGE_MODELS" -v rewrites="$TMP/rewrites.txt" '
    BEGIN {
        while ((getline line < rewrites) > 0) {
            split(line, kv, "\t"); moved[kv[1]] = kv[2]
        }
        print "# Generated by contrib/docker/stage.sh from " ENVIRON["SOURCE_NAME"] "."
        print "# The original, made to fit the container: see contrib/docker/README.md."
        print ""
    }
    # Blank lines are held back until the next line that is not blank, so a
    # key added at the end of a section lands inside it, not after the gap
    # before the next one.
    function out(text) {
        for (; blanks > 0; blanks--) print ""
        print text
    }
    function finish() {
        if (section == "orangu-coordinator") {
            if (!host) out("host = all")
            if (!dir) out("models = " models)
        }
    }
    /^[ \t]*$/ { blanks++; next }
    /^[ \t]*\[/ {
        held = blanks; blanks = 0
        finish()
        blanks += held
        section = $0; gsub(/[][ \t]/, "", section)
        out($0); next
    }
    /^[ \t]*[#;]/ || !/=/ { out($0); next }
    {
        key = $1; gsub(/^[ \t]+|[ \t]+$/, "", key)
        value = substr($0, index($0, "=") + 1)
        gsub(/^[ \t]+|[ \t]+$/, "", value)
    }
    section == "orangu-coordinator" && key == "host" { out("host = all"); host = 1; next }
    section == "orangu-coordinator" && key == "models" { out("models = " models); dir = 1; next }
    section == "orangu-coordinator" && (key == "log_type" || key == "log_path") { next }
    section != "orangu-coordinator" && key == "web" { next }
    section != "orangu-coordinator" && key == "model" && (value in moved) {
        out("model = " moved[value]); next
    }
    { out($0) }
    END { blanks = 0; finish() }
' "$TMP/source.conf" > "$STAGE/orangu-coordinator.conf"

# --- Summary -----------------------------------------------------------------
echo
while IFS="$(printf '\t')" read -r model bytes; do
    profiles=$(awk -F= -v want="$model" '
        /^[ \t]*\[/ { section = $0; gsub(/[][ \t]/, "", section); next }
        section != "orangu-coordinator" && $1 ~ /^[ \t]*model[ \t]*$/ {
            value = substr($0, index($0, "=") + 1)
            gsub(/^[ \t]+|[ \t]+$/, "", value)
            if (value == want) names = names (names ? ", " : "") section
        }
        END { print names }' "$TMP/source.conf")
    printf 'Profile    %s -> %s (%s)\n' "$profiles" "$model" "$(human "$bytes")"
done < "$TMP/summary.txt"
printf 'Models     %s in all\n' "$(human "$TOTAL")"
rm -rf "$TMP"
