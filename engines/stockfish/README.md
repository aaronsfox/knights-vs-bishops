# Stockfish

Version pinned for this study: **19** (check `engine.toml`).

## Install

The binary goes in `engines/stockfish/bin/` (git ignores that folder).

Stockfish 19 publishes one **universal** build per platform instead of one
build per instruction set. A universal build picks the fastest code path the
CPU supports when it starts, so the same download suits every machine.

**Windows (development).** From the
[GitHub releases](https://github.com/official-stockfish/Stockfish/releases),
download `stockfish-windows-x86-64-universal.zip`, unzip it and copy the
executable to `bin/stockfish.exe`.

**Linux cluster (production data).** Download
`stockfish-linux-x86-64-universal.tar.gz`, then:

```bash
tar -xzf stockfish-linux-x86-64-universal.tar.gz
ls stockfish/                       # find the executable's name
cp stockfish/<executable> <repo>/engines/stockfish/bin/stockfish
chmod +x <repo>/engines/stockfish/bin/stockfish
```

If the archive holds files the executable needs alongside it (for example a
separate network file), copy them into `bin/` too and record them in
`engine.toml` so their hashes are checked.

If the binary fails with a `GLIBC` version error, the cluster's system
libraries are too old for the release build. Build from source instead
(login node; the build downloads Stockfish's network):

```bash
module load GCC/10.2.0                          # your cluster's module name
bash engines/stockfish/build_linux.sh           # ARCH=x86-64-bmi2 by default
```

See the comments at the top of `build_linux.sh` for the settings (`ARCH`,
`SF_VERSION`, `PGO`).

## Check that PC and cluster search identically

Because a universal build may run a different internal code path on your PC
(AMD Zen 4) than on the cluster (Intel Cascade Lake), confirm once that they
agree. Stockfish's `bench` prints a node count that should be identical on
any CPU for the same version:

```bash
./bin/stockfish bench 2>&1 | grep -E "Nodes searched|Nodes/second"
```

Run it on your PC and on a compute node: "Nodes searched" must match; only
"Nodes/second" should differ.

## Verify

```bash
python code/setup_engines.py check  --engine stockfish
python code/setup_engines.py smoke  --engine stockfish
python code/setup_engines.py record --engine stockfish   # then commit engines/engines.lock.json
```
