#!/usr/bin/env bash
# Build Stockfish 11 and its knight/bishop-preferring variants (Stage 3).
#
# Every folder engines/sf11*/ with an engine.toml gets its own binary in its
# bin/: the base build (engines/sf11, unmodified) and each variant, patched
# by patch_sf11.py from the manifest's [variant] table.
#
#   module load GCC/10.2.0                      # your cluster's module name
#   bash engines/stockfish11/build_linux.sh     # all sf11* engines
#   bash engines/stockfish11/build_linux.sh sf11_b050 sf11_n050   # just these
#
# Settings (environment variables):
#   ARCH   instruction set (default x86-64-bmi2, which runs on the cluster
#          nodes; see `make help` in Stockfish 11's src/)
#   JOBS   parallel make jobs (default 4)
#
# Stockfish 11 has a hand-crafted evaluation, so there is no network to
# download; the build needs git, make, a C++11 compiler and Python 3.11+.
# Builds skip profile-guided optimisation, which affects speed only.
#
# Afterwards (repository root):
#   python code/setup_engines.py check --engine sf11_b050     # etc.
#   python code/setup_engines.py record                       # then commit engines.lock.json
set -euo pipefail

ARCH="${ARCH:-x86-64-bmi2}"
JOBS="${JOBS:-4}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINES="$(cd "$HERE/.." && pwd)"
WORK="${TMPDIR:-/tmp}/stockfish11-build-$$"
PYTHON="${PYTHON:-python3}"

if [[ $# -gt 0 ]]; then
    KEYS=("$@")
else
    KEYS=()
    for manifest in "$ENGINES"/sf11*/engine.toml; do
        KEYS+=("$(basename "$(dirname "$manifest")")")
    done
fi

echo "Building Stockfish 11 (${KEYS[*]}) for ${ARCH} with $(g++ --version | head -n 1)"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT
git clone --quiet --depth 1 --branch sf_11 https://github.com/official-stockfish/Stockfish.git "$WORK/pristine"

for key in "${KEYS[@]}"; do
    manifest="$ENGINES/$key/engine.toml"
    [[ -f "$manifest" ]] || { echo "no manifest at $manifest" >&2; exit 1; }
    echo
    echo "== $key"
    rm -rf "$WORK/$key"
    cp -r "$WORK/pristine" "$WORK/$key"
    if grep -q '^\[variant\]' "$manifest"; then
        "$PYTHON" "$HERE/patch_sf11.py" "$WORK/$key/src" "$manifest" > "$WORK/$key/variant.json"
        cat "$WORK/$key/variant.json"
    else
        echo '{"key": "'"$key"'", "unmodified": true}' > "$WORK/$key/variant.json"
        echo "unmodified Stockfish 11"
    fi
    (cd "$WORK/$key/src" && make -j"$JOBS" build ARCH="$ARCH" COMP=gcc > "$WORK/$key/build.log" 2>&1) \
        || { tail -n 30 "$WORK/$key/build.log"; exit 1; }
    mkdir -p "$ENGINES/$key/bin"
    cp "$WORK/$key/src/stockfish" "$ENGINES/$key/bin/stockfish"
    cp "$WORK/$key/variant.json" "$ENGINES/$key/bin/variant.json"
    printf 'uci\nquit\n' | "$ENGINES/$key/bin/stockfish" | grep -m 1 "id name" || true
    # bench's node count identifies the build: the base build matches the
    # official Stockfish 11 bench, and each variant has its own.
    "$ENGINES/$key/bin/stockfish" bench 2>&1 | grep -E "Nodes searched" || true
done
echo
echo "Installed: ${KEYS[*]}"
