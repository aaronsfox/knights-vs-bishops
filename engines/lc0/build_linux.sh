#!/usr/bin/env bash
# Build Lc0 on Linux and copy it into engines/lc0/bin/.
#
# GPU build (production; CUDA backend, no cuDNN):
#   module load GCC/10.2.0 CUDA/12.2.2      # your cluster's module names
#   LC0_CUDA_CC=80 bash engines/lc0/build_linux.sh
# LC0_CUDA_CC is the GPU's compute capability without the dot: 80 = A100
# (8.0), 70 = V100, 90 = H100. Setting it means the build doesn't need a GPU,
# so it can run on a login node.
#
# CPU-only build (BLAS backend):
#   bash engines/lc0/build_linux.sh
#
# Needs: git, meson and ninja (`pip install meson ninja`), a C++17 compiler,
# OpenBLAS for the CPU backend, and the CUDA toolkit (nvcc) for the GPU build.
set -euo pipefail

VERSION="${LC0_VERSION:-v0.32.1}"
CUDA_CC="${LC0_CUDA_CC:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK="${TMPDIR:-/tmp}/lc0-build-${VERSION}"

# By default Lc0 compiles its CPU code for the build machine's own processor.
# The login node and the compute nodes may have different CPUs (Intel login
# nodes, AMD EPYC in DGX A100 nodes), so build portable x86-64 code instead.
PORTABLE=(-Dnative_arch=false -Dispc_native_only=false)

if [[ -n "$CUDA_CC" ]]; then
    command -v nvcc >/dev/null || { echo "nvcc not found: load the CUDA module first" >&2; exit 1; }
    # plain_cuda = CUDA backend without cuDNN. native_cuda=false + cc_cuda
    # compiles for the named GPU generation instead of detecting a local GPU.
    OPTIONS=(-Dplain_cuda=true -Dcudnn=false -Dnative_cuda=false "-Dcc_cuda=${CUDA_CC}"
             -Dopencl=false -Dgtest=false "${PORTABLE[@]}")
    echo "Building Lc0 ${VERSION} with CUDA for compute capability ${CUDA_CC}"
else
    OPTIONS=(-Dcudnn=false -Dplain_cuda=false -Dopencl=false -Dgtest=false "${PORTABLE[@]}")
    echo "Building Lc0 ${VERSION} for CPU only"
fi

rm -rf "$WORK"
git clone --depth 1 --branch "$VERSION" --recurse-submodules \
    https://github.com/LeelaChessZero/lc0.git "$WORK"
cd "$WORK"

# Lc0's build.sh passes extra arguments to meson. If meson reports an unknown
# option, list this version's option names with `meson configure build/release`
# and adjust OPTIONS above.
./build.sh "${OPTIONS[@]}"

mkdir -p "$HERE/bin"
cp build/release/lc0 "$HERE/bin/lc0"
echo "Installed $HERE/bin/lc0"
"$HERE/bin/lc0" --help | head -n 3 || true
