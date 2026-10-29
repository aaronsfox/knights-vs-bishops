#!/usr/bin/env bash
# Build Stockfish from source and install it as engines/stockfish/bin/stockfish.
#
# Usually not needed: the universal Linux release binary works on most
# clusters. Build from source if that binary fails (e.g. a GLIBC error).
#
#   module load GCC/10.2.0                      # your cluster's module name
#   bash engines/stockfish/build_linux.sh
#
# Settings (environment variables):
#   SF_VERSION  release tag                     (default sf_19)
#   ARCH        instruction set, see `make help` (default x86-64-bmi2, which
#               runs on the cluster nodes and the Zen 4 PC; x86-64-vnni512 or
#               x86-64-avx512 may be faster on Cascade Lake)
#   PGO=1       profile-guided build: faster, but runs the new binary during
#               the build, so the build machine must support ARCH
#
# Needs git, make, a C++17 compiler and internet access (the build downloads
# Stockfish's network), so run it on a login node.
set -euo pipefail

VERSION="${SF_VERSION:-sf_19}"
ARCH="${ARCH:-x86-64-bmi2}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${TMPDIR:-/tmp}/stockfish-build-${VERSION}"

echo "Building Stockfish ${VERSION} for ${ARCH} with $(g++ --version | head -n 1)"
rm -rf "$WORK"
git clone --depth 1 --branch "$VERSION" https://github.com/official-stockfish/Stockfish.git "$WORK"
cd "$WORK/src"

if [[ "${PGO:-0}" == "1" ]]; then
    make -j4 profile-build ARCH="$ARCH" COMP=gcc
else
    make -j4 build ARCH="$ARCH" COMP=gcc
fi

mkdir -p "$HERE/bin"
cp stockfish "$HERE/bin/stockfish"
echo "Installed $HERE/bin/stockfish"
printf 'uci\nquit\n' | "$HERE/bin/stockfish" | grep -m 1 "id name" || true
