# Caissa

Version pinned for this study: **2.0** (release tag `2.0`, September 2026,
[release page](https://github.com/Witek902/Caissa/releases/tag/2.0)).

Caissa's networks are trained on its own self-play games, and it reports
win/draw/loss probabilities (`UCI_ShowWDL`). Version 2.0 moves to a
multilayer network (wider first layer, pairwise activation, output
subnetworks chosen by piece count), fixes several FEN-parsing issues and a
`hashfull` reporting bug, and removes the `UCI_AnalyseMode` option. This
study doesn't use that option.

## Install

The binary goes in `engines/caissa/bin/` (git ignores that folder).

**Windows (development).** Download the release build for your CPU from the
[2.0 release page](https://github.com/Witek902/Caissa/releases/tag/2.0) and
rename it to `bin/caissa.exe`. The release files are
`caissa-2.0-x64-<build>.exe` for `avx512icl`, `avx512`, `avx2-bmi2`, `avx2`,
`sse4-popcnt` and `sse2`. Most modern CPUs run `avx2-bmi2`; `avx512icl` needs
an Intel Ice Lake (or newer) or AMD Zen 4 CPU.

**Linux cluster (production data).** The 2.0 release has Windows builds only,
so build from source with `build_linux.sh`. It needs `make`, GCC 10+ (C++20)
and internet access, because the build downloads Caissa's network from the
Caissa-Nets repository and embeds it, so run it on a login node. No CMake is
needed:

```bash
module load GCC/10.2.0                          # GCC 10+; your cluster's module name
bash engines/caissa/build_linux.sh              # ARCH=bmi2 by default
```

Match the Windows release file to the Linux target: the default `bmi2` with
`caissa-2.0-x64-avx2-bmi2.exe`, or `ARCH=avx512` with
`caissa-2.0-x64-avx512.exe`. Avoid `avx512icl` on Cascade Lake nodes. If
GCC 10 rejects some C++20 code, use a newer compiler (e.g. conda's
`gxx_linux-64`) with `CC=x86_64-conda-linux-gnu-g++`. If the network download
fails, get the `.pnn` file named in the error from the Caissa-Nets releases,
put it in the build's `data/neuralNets/` folder and rerun `make` there.

## Use the same build type everywhere

Pick one instruction-set build (for example `bmi2`) that runs on both your PC
and the cluster's compute nodes, and use it on both. Different builds use
different vector code paths, which can change results slightly. Caissa's
`bench` command prints a node count: if it matches on both machines, the
builds search identically.

## Verify

```bash
python code/setup_engines.py check  --engine caissa
python code/setup_engines.py smoke  --engine caissa
python code/setup_engines.py record --engine caissa
```
