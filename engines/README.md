# Engines

Each engine has its own folder:

| Folder | Engine | Shown as | Search | Used in |
| --- | --- | --- | --- | --- |
| `stockfish/` | Stockfish 19 | Stockfish | Alpha-beta, NNUE | Stage 2, book, ladder rungs |
| `caissa/` | Caissa 2.0 | Caissa | Alpha-beta, NNUE (own self-play data) | Stage 2 |
| `lc0/` | Lc0 v0.32.1 (GPU) | Lc0 | Neural MCTS | Stage 2 (key cells), evaluation |
| `sf11/` | Stockfish 11, unmodified | Stockfish 11 | Alpha-beta, hand-crafted | Stage 3 baseline |
| `sf11_n020/` | Stockfish 11, knight +0.2 pawns relative to bishop | Aron Knightzowitsch | " | Stage 3: likes knights |
| `sf11_n050/` | Stockfish 11, knight +0.5 | Hikaru Knightamura | " | Stage 3: loves knights |
| `sf11_b020/` | Stockfish 11, bishop +0.2 pawns relative to knight | Bishy Anand | " | Stage 3: likes bishops |
| `sf11_b050/` | Stockfish 11, bishop +0.5 | Bobby Bischer | " | Stage 3: loves bishops |
| `sf11_pair200/` | Stockfish 11, bishop-pair term x 2 | Magnus Pairlsen | " | Stage 3 (optional) |

`stockfish11/` is not an engine: it holds the build script and patcher that
make all the `sf11*` binaries (see `stockfish11/README.md`).

In each engine folder:

- `engine.toml` is the manifest: version, display name, binary path per
  platform, the UCI options every search uses, any network file, and (for
  the Stockfish 11 variants) the `[variant]` settings the patcher applies.
- `bin/` holds the binary and `networks/` any network file. Git ignores both,
  so binaries never enter the repository.

`engines.lock.json` records the SHA-256 of every binary and network, per
platform. Production runs refuse to start if the files on disk don't match it,
so every result can be traced to the exact engine build that produced it.

## Installing on the Linux cluster

**Stockfish 19, Caissa 2.0 and Lc0** are the Chess960 project's builds. The
lock file already holds their Linux hashes, so the quickest route is to copy
them from the Chess960 clone (they then match without re-recording):

```bash
for e in stockfish caissa lc0; do cp ~/chess960-analysis/engines/$e/bin/* engines/$e/bin/; done
cp ~/chess960-analysis/engines/lc0/networks/*.pb engines/lc0/networks/
cp ~/chess960-analysis/tools/fastchess/bin/fastchess tools/fastchess/bin/
python code/setup_engines.py check --engine stockfish     # lock: matches
```

Otherwise build them with each folder's `build_linux.sh` (login node; see
the Chess960 notes in each `README.md`):

```bash
module load GCC/10.2.0
bash engines/stockfish/build_linux.sh
bash engines/caissa/build_linux.sh
module load CUDA/12.2.2 && LC0_CUDA_CC=80 bash engines/lc0/build_linux.sh
```

**Stockfish 11 and its variants** (login node, a couple of minutes each):

```bash
module load GCC/10.2.0
bash engines/stockfish11/build_linux.sh       # sf11 and every sf11_* variant
python code/setup_engines.py check
python code/setup_engines.py smoke --engine sf11_b050
python code/setup_engines.py record           # then commit engines.lock.json
```

## Adding or upgrading an engine

1. Install the binary as the engine's README describes.
2. Update `version` in `engine.toml`.
3. Run `python code/setup_engines.py check --engine <name>` and
   `python code/setup_engines.py smoke --engine <name>`.
4. Run `python code/setup_engines.py record --engine <name>` and commit the
   updated `engines.lock.json`.

To add another Stockfish 11 variant, copy a `sf11_*` folder, change its
`[variant]` values and `display_name`, and build it with
`bash engines/stockfish11/build_linux.sh <folder>`.

Don't upgrade an engine partway through a production run: results from two
builds would be mixed.
