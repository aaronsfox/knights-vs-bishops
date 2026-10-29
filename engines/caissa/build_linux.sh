#!/usr/bin/env bash
# Build Caissa from source and install it as engines/caissa/bin/caissa.
# Caissa publishes Windows binaries only, so Linux needs this build.
#
#   module load GCC/10.2.0                      # GCC 10+ (C++20); your module name
#   bash engines/caissa/build_linux.sh
#
# Settings (environment variables):
#   CAISSA_VERSION  release tag                 (default 2.0)
#   ARCH            makefile target: bmi2, avx2, avx2-vnni, avx512, sse4, sse2
#                   (default bmi2; avoid avx512icl on Cascade Lake). Match the
#                   Windows release file: bmi2 <-> caissa-2.0-x64-avx2-bmi2.exe,
#                   avx512 <-> caissa-2.0-x64-avx512.exe
#   CC              C++ compiler                (default g++; Caissa's makefile
#                   uses CC for C++)
#
# Needs git, make, a C++20 compiler and internet access (the build downloads
# Caissa's network from the Caissa-Nets repository and embeds it), so run it
# on a login node.
set -euo pipefail

VERSION="${CAISSA_VERSION:-2.0}"
ARCH="${ARCH:-bmi2}"
COMPILER="${CC:-g++}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${TMPDIR:-/tmp}/caissa-build-${VERSION}"

echo "Building Caissa ${VERSION} for ${ARCH} with $($COMPILER --version | head -n 1)"
rm -rf "$WORK"
git clone --depth 1 --branch "$VERSION" https://github.com/Witek902/Caissa.git "$WORK"
cd "$WORK/src"

make -j4 "$ARCH" CC="$COMPILER"

# The makefile names its output caissa-<version>-x64-<target>.
BUILT=$(ls -t caissa-*-x64-"$ARCH" 2>/dev/null | head -n 1 || true)
if [[ -z "$BUILT" ]]; then
    echo "Build finished but no caissa-*-x64-${ARCH} file was found in $WORK/src" >&2
    exit 1
fi

mkdir -p "$HERE/bin"
cp "$BUILT" "$HERE/bin/caissa"
chmod +x "$HERE/bin/caissa"
echo "Installed $HERE/bin/caissa (from $BUILT)"
printf 'uci\nquit\n' | "$HERE/bin/caissa" | grep -m 1 "id name" || true
