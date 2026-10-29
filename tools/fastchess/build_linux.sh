#!/usr/bin/env bash
# Build fastchess (the match runner for all engine games) and install it as
# tools/fastchess/bin/fastchess.
#
#   module load GCC/10.2.0                      # C++17 compiler; your module name
#   bash tools/fastchess/build_linux.sh
#
# Settings (environment variables):
#   FASTCHESS_VERSION  release tag              (default v1.8.2-alpha)
#   CXX                C++ compiler              (default g++)
#
# The binary uses portable x86-64 code (`build=release`), so it runs on every
# node whatever CPU the login node has. fastchess's release build links
# everything statically, but a statically linked glibc older than 2.34 (most
# clusters) crashes as soon as a thread starts (exit -11, segmentation
# fault). So glibc stays dynamic and only the C++ runtime is linked
# statically: the binary needs no compiler module loaded at run time.
# Needs git and make; run it on a login node.
set -euo pipefail

VERSION="${FASTCHESS_VERSION:-v1.8.2-alpha}"
COMPILER="${CXX:-g++}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${TMPDIR:-/tmp}/fastchess-build-${VERSION}"

echo "Building fastchess ${VERSION} with $($COMPILER --version | head -n 1)"
rm -rf "$WORK"
git clone --depth 1 --branch "$VERSION" https://github.com/Disservin/fastchess.git "$WORK"
cd "$WORK"
make -j4 build=release CXX="$COMPILER" \
    LDFLAGS="-pthread -Wl,--no-as-needed -lstdc++fs -static-libgcc -static-libstdc++"

mkdir -p "$HERE/bin"
cp fastchess "$HERE/bin/fastchess"
chmod +x "$HERE/bin/fastchess"
echo "Installed $HERE/bin/fastchess"
"$HERE/bin/fastchess" -version

# Smoke test: two very fast games from a removal position (White without its
# c1 bishop, Black without its b8 knight), with whichever Stockfish build is
# installed. fastchess crashes here (exit -11) if the binary can't start threads.
SF=""
for candidate in "$HERE/../../engines/stockfish/bin/stockfish" "$HERE/../../engines/sf11/bin/stockfish"; do
    [[ -x "$candidate" ]] && { SF="$candidate"; break; }
done
if [[ -n "$SF" ]]; then
    TEST="$(mktemp -d)"
    echo "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RN1QKBNR w KQkq - 0 1" > "$TEST/start.epd"
    (cd "$TEST" && "$HERE/bin/fastchess" -variant standard \
        -openings file=start.epd format=epd \
        -engine cmd="$SF" name=a -engine cmd="$SF" name=b \
        -each tc=0.5+0.01 proto=uci option.Threads=1 option.Hash=16 \
        -rounds 1 -repeat -concurrency 2 -pgnout file=test.pgn > /dev/null) \
        || { echo "fastchess smoke test FAILED (exit $?)"; exit 1; }
    echo "Smoke test: $(grep -c '^\[Result ' "$TEST/test.pgn") of 2 games played"
    rm -rf "$TEST"
else
    echo "No Stockfish build installed; skipped the smoke test"
fi
