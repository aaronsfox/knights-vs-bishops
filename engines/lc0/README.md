# Lc0 (Leela Chess Zero)

Version pinned for this study: **v0.32.1**.

Lc0 needs two files: the engine binary and a neural network. Neither is
committed.

**Hardware.** Production runs use an NVIDIA GPU on the Linux cluster (the
`dgx` partition's A100s) with the full-precision `cuda` backend. On a single
CPU core Lc0 manages only about 16 nodes per second with this network, which
makes searches of 10⁵–10⁶ nodes impractical. Searches are limited by node
count, so the GPU changes how long a search takes, not what it covers. On
Windows (no NVIDIA GPU) the CPU build is used for development and testing
only.

## Settings (`engine.toml`)

- `Threads = 1`: one search thread, so the tree search is reproducible.
- `SmartPruningFactor = 0`: search exactly the node budget rather than
  stopping once the best move can't change.
- `Backend = "cuda"` on Linux only: the fp32 CUDA backend without cuDNN.
  (`cuda-fp16` and `cuda-auto` would use half precision on an A100.)
- `MinibatchSize = 256`: a fixed batch size, so the search gathers positions
  for the network the same way every run.

Rerun a few sample positions on the GPU to confirm results reproduce exactly.

## Install on the Linux cluster (GPU build)

The build doesn't need a GPU, so run it on a login node (it downloads the
source). It needs `meson` and `ninja` (`pip install meson ninja` in the
project environment), a C++17 compiler and the CUDA toolkit:

```bash
module load GCC/10.2.0 CUDA/12.2.2          # your cluster's module names
conda activate kvb
LC0_CUDA_CC=80 bash engines/lc0/build_linux.sh
```

`LC0_CUDA_CC` is the GPU's compute capability without the dot: 80 for the
A100, 70 for a V100, 90 for an H100. The script installs `bin/lc0`. The CUDA
module must also be loaded when Lc0 runs; `jobs/evaluate_sp_lc0_gpu.slurm`
does this.

Without `LC0_CUDA_CC` the script builds a CPU-only version (BLAS backend,
needs OpenBLAS).

## Install on Windows (development)

Download `lc0-v0.32.1-windows-cpu-dnnl.zip` (or `-cpu-openblas`) from the
[releases page](https://github.com/LeelaChessZero/lc0/releases/tag/v0.32.1)
and put `lc0.exe` and its DLLs in `bin/`. The `Backend` setting applies on
Linux only, so the Windows CPU build uses its default backend.

## Install the network

Download the network named in `engine.toml` from
<https://lczero.org/play/networks/bestnets/> into `engines/lc0/networks/`.
Use the identical file on every machine (compare `sha256sum`): if you
decompress the `.pb.gz` download on one machine, do the same on the others.

## Verify

On a GPU node (`srun --partition=dgx --gres=gpu:1 --pty bash`) with the CUDA
module loaded:

```bash
python code/setup_engines.py check  --engine lc0
python code/setup_engines.py smoke  --engine lc0
python code/setup_engines.py record --engine lc0
```
