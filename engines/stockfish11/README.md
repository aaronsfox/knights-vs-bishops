# Stockfish 11 and the knight/bishop variants (Stage 3)

Stockfish 11 is the last release with a hand-crafted evaluation, so its
piece values are explicit constants that can be changed. Stage 3 plays each
variant against the unmodified build (`engines/sf11`).

| Engine | Shown as | Bishop - knight shift | Bishop-pair term |
| --- | --- | --- | --- |
| `sf11` | Stockfish 11 | 0 | x 1 |
| `sf11_n020` | Aron Knightzowitsch | -0.2 pawns (likes knights) | x 1 |
| `sf11_n050` | Hikaru Knightamura | -0.5 pawns (loves knights) | x 1 |
| `sf11_b020` | Bishy Anand | +0.2 pawns (likes bishops) | x 1 |
| `sf11_b050` | Bobby Bischer | +0.5 pawns (loves bishops) | x 1 |
| `sf11_pair200` | Magnus Pairlsen | 0 | x 2 (optional) |

## What the patch changes

`patch_sf11.py` reads each manifest's `[variant]` table and edits the
Stockfish 11 source before building:

- **Piece values** (`src/types.h`). Stockfish 11 has
  KnightValueMg/Eg = 781/854 and BishopValueMg/Eg = 825/915, with
  PawnValueMg/Eg = 128/213 (so the bishop is already ~0.34 of a middlegame
  pawn and ~0.29 of an endgame pawn above the knight, before the pair). A
  shift of s pawns moves the bishop up by s/2 and the knight down by s/2, in
  each phase's pawn units (s = 0.5: +/-32 middlegame, +/-53 endgame), so the
  total of one bishop and one knight is unchanged.
- **Bishop pair** (`src/material.cpp`). The pair term QuadraticOurs[0][0]
  (1438, divided by 16, so about 90 internal units) is multiplied by
  `bishop_pair_scale`.
- **Name** (`src/misc.cpp`). The variant's key goes into its UCI id name,
  e.g. "Stockfish 11 sf11_b050 64 BMI2", so every PGN records the build.

Stockfish uses these constants everywhere (evaluation, static exchange
evaluation for move ordering and pruning, endgame recognition, game phase),
so a variant believes its values throughout, not just in its evaluation. The
patcher checks every original constant first and refuses to patch a tree
that doesn't match (another version, or one already patched); the knight and
bishop values are kept different, since Stockfish tells some endgames apart
by material value.

## Build

```bash
module load GCC/10.2.0
bash engines/stockfish11/build_linux.sh                # all sf11* engines
bash engines/stockfish11/build_linux.sh sf11_b050      # one
```

Each build prints its UCI id and `bench` node count. The unmodified build
must give Stockfish 11's official bench, **5156767** nodes; each variant has
its own (in the local test build: sf11_b020 4918486, sf11_b050 5058181,
sf11_n020 4801640, sf11_n050 4772929, sf11_pair200 4598686; these should
match on the cluster for the same compiler flags). `bin/variant.json`
records the values each binary was built with.

Then `python code/setup_engines.py record` and commit `engines.lock.json`.
Stockfish 11 reports centipawns only (no WDL), which is fine here: Stage 3
measures variants by game results.
