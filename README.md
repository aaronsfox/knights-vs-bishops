# Knights vs bishops

Are bishops worth more than knights, and when? Standard piece values treat
them as equal, while strong players and engines lean towards the bishop.
Kaufman's database study found an unpaired bishop and knight equal, with the
bishop's edge coming from the bishop pair. This project separates the single
piece from the pair in three steps about decisions during play, which build
one argument:

- **Step 1: human play** (to build: `lichess_games.py`, ported from the
  Chess960 project's Lichess reader; `human_events.py`; `analyse_human.py`).
  Rated Lichess blitz, rapid and classical games by rating band: which piece
  players give up in bishop-for-knight exchanges, whether keeping the bishop
  or the pair pays off, and how each piece's endgames are played against
  their exact Syzygy verdict (`code/tablebase.py`).
- **Step 2: preference engines** (`code/analyse_bias.py`). Stockfish 11
  builds that value the knight or bishop more (Aron Knightzowitsch, Hikaru
  Knightamura, Bishy Anand, Bobby Bischer, and Magnus Pairlsen for the pair)
  play the unmodified build: does acting on a preference win games?
- **Step 3: start-position value** (`code/evaluate_sp.py`,
  `code/selfplay.py`, `code/analyse_removal.py`). The standard position with
  minor pieces removed (the 52-cell removal matrix, plus the standard
  position and pawn-odds cells), evaluated up a node ladder and played out
  in engine self-play with Stockfish 19, Caissa 2.0 and Lc0: the controlled
  estimate, in pawns.

The pipeline, statistics and figure style come from the Chess960 fairness
project (`aaronsfox/chess960-analysis`); its results are not used. The
research and analysis plans, and the design decisions the code depends on,
are in `docs/README.md`.

## Repository layout

```
engines/            one folder per engine: manifest (engine.toml), bin/ (not committed)
  stockfish/ caissa/ lc0/            Step 3 engines (Chess960 builds)
  sf11/ sf11_n020/ sf11_n050/ sf11_b020/ sf11_b050/ sf11_pair200/   Step 2
  stockfish11/      build script and patcher for every sf11* engine
  engines.lock.json SHA-256 of every installed engine binary and network
config/             settings: positions, books, matches, tablebases
code/
  positions.py      the starting positions, colour-flipped twins and comparisons
  utils.py          shared helpers: engine setup and lock file, metrics, IO
  setup_engines.py  check, fingerprint and smoke-test the engines
  evaluate_sp.py    screening pass: evaluate positions at a node ladder
  opening_book_generator.py  balanced opening lines from each position
  selfplay.py       engine games with fastchess (self-play, variant vs base)
  tablebase.py      Syzygy probing (exact endgame verdicts for Step 1)
  stats_core.py     Dirichlet posteriors with a design-effect correction
  analyse_bias.py     Question 2: variants against Stockfish 11
  analyse_removal.py  Question 3: comparisons, designs, pawn equivalents
  plot_style.py     shared figure style (from Chess960)
  plot_boards.py    board diagrams of positions, book lines or any FEN
  task_positions.py splits a position list into SLURM array tasks
tools/fastchess/    build script for fastchess; bin/ not committed
jobs/               SLURM job scripts
tests/              unit tests (pytest)
docs/               research and analysis plans
results/            outputs (production folders committed, test runs not)
```

## Positions

Every starting position is the standard array with pieces removed, White to
move, castling intact. A position's id lists the removed squares, White's
first: `start`, `c1-b8` (White without the c1 bishop, Black without the b8
knight), `b1-c1-c8-f8` (White keeps BN, Black keeps NN), `f2` (pawn odds).
Groups: `control`, `1v1` (16), `2v2` (36), `pawn_odds` (4), `removal`
(the 52 removal cells), `key` (29 cells that get the top-up batch) and
`all` (57). Any
`--positions` option takes ids, groups or `@file`
(e.g. `@config/sample_positions.txt`, 14 positions spanning every design
for quick checks).

**Colour balance.** In self-play both sides are the same engine, so
swapping the engine copies doesn't move the material: in `c1-b8` the bishop
side is always Black. Each position is therefore pooled with its
**colour-flipped twin** (`b1-c8`) into a **comparison**, scored from the
**focal side** (the side with more bishops, or with the extra pawn). This
cancels White's first-move edge. Positions that are their own twin (`start`,
`c1-c8`, ...) are mirrored controls that measure White's edge. In Step 2
(variant vs base) swapping colours does balance the variant's colour.

```python
from positions import get_position
p = get_position("c1-b8")
p.design, p.twin_id, p.comparison_id   # ('BBN v BNN', 'b1-c8', 'b1-c8')
```

## Setup

Requires Python 3.11+.

```bash
conda env create -f environment.yml
conda activate kvb
python -m pytest
```

or `pip install chess numpy pandas pyarrow scipy pytest matplotlib cairosvg`.
Then install the engines and fastchess (`engines/README.md`; on the cluster
the Step 3 engines and fastchess can be copied from the Chess960 clone,
whose hashes are already in the lock file):

```bash
bash engines/stockfish11/build_linux.sh       # Stockfish 11 and the variants
python code/setup_engines.py check
python code/setup_engines.py smoke            # legal moves on all 57 positions?
python code/setup_engines.py record           # write hashes; commit engines.lock.json
```

## Running the pipeline

Everything is resumable (finished positions are skipped) and writes atomic
files; on the cluster use the SLURM scripts in `jobs/` (create `logs/` first).
Print the `--array` range for any position list with
`python code/task_positions.py <positions> --array <per_task>`.

**1. Screening pass** (Question 3a): node-limited evaluations.

```bash
sbatch --export=ALL,ENGINE=stockfish jobs/evaluate_sp_array.slurm   # 57 tasks, one position each
sbatch --export=ALL,ENGINE=caissa    jobs/evaluate_sp_array.slurm
sbatch jobs/evaluate_sp_lc0_gpu.slurm
python code/evaluate_sp.py collect
```

**2. Opening book**: 50 balanced lines per position, generated from each
position by Stockfish 19 (standard books don't apply with pieces missing).

```bash
sbatch jobs/opening_book_array.slurm           # 15 tasks of 4 positions
python code/opening_book_generator.py collect
```

**3. Self-play** (Step 3): a first batch of 500 games per position, then a
top-up batch for the key cells, sized from the first batch's draw rate and
design effect (the example adds 1,000 games per position).

```bash
sbatch --export=ALL,PLAYER=stockfish jobs/selfplay_array.slurm
sbatch --array=0-7 --time=03:00:00 \
    --export=ALL,PLAYER=stockfish,POSITIONS=key,BATCH=b2,GAMES_PER_LINE=20 jobs/selfplay_array.slurm
python code/selfplay.py collect
python code/analyse_removal.py
```

**4. Preference engines** (Step 2): the base build against itself first
(Gate 2), then each variant against it, e.g. 3,000 games from the standard
position.

```bash
sbatch --array=0 --time=03:00:00 \
    --export=ALL,PLAYER=sf11,POSITIONS=start,GAMES_PER_LINE=60 jobs/selfplay_array.slurm   # base vs itself
for v in sf11_n020 sf11_n050 sf11_b020 sf11_b050 sf11_pair200; do
  sbatch --array=0 --time=03:00:00 \
      --export=ALL,PLAYER=$v,OPPONENT=sf11,POSITIONS=start,GAMES_PER_LINE=60 jobs/selfplay_array.slurm
done
python code/selfplay.py collect
python code/analyse_bias.py
```

**5. Human games** (Step 1): to build. The Lichess reader from the Chess960
project (`lichess960.py`) will be ported to the monthly standard exports
(`lichess_db_standard_rated_YYYY-MM.pgn.zst`): streamed or downloaded on the
login node, filtered on headers, sampled to balance time controls and rating
bands, then replayed by an event extractor (`human_events.py`). Endgames
reached with 6 pieces or fewer get their exact verdict from the 3-6 piece
Syzygy tables (path in `config/tablebase.toml`). See the analysis plan.

## Outputs

| Folder | Contents |
| --- | --- |
| `results/evaluate_sp/` | node-ladder evaluations (`tables/evaluate_sp_final.parquet`, convergence) |
| `results/opening_book/` | per-position lines and pools; `book/opening_book_lines.parquet` |
| `results/selfplay/raw/<match>/` | one gzipped PGN per position (and batch); `<match>` is the player or `<player>__vs__<opponent>` |
| `results/selfplay/tables/` | games, positions and lines tables from `selfplay.py collect` |
| `results/selfplay/analysis/` | `removal_*` and `bias_*` tables and summaries |
| `results/tablebase/` | sampled positions and per-set summaries |
| `results/figures/` | figures in the `plot_style.py` style |

## Statistics

As in Chess960: in the engine games (Steps 2 and 3), each comparison's W/D/L counts get a Dirichlet(½ + (W, D, L)
/ DEFF) posterior, with DEFF = 1 + (m − 1)·ICC for games sharing an opening
line (m is Kish's weighted mean when batches give lines different numbers of
games). 20,000 seeded Monte Carlo draws give every derived quantity: E,
ΔElo, decisive rate, pawn equivalents (ΔElo divided by the same player's
pawn-odds ΔElo, draw by draw), differences between designs and the
additivity check. A Bayesian bootstrap over lines is the sensitivity check.
Every cell is reported and ranked; probability thresholds are for
transparency only.

## Checks built in

- Engine files are hash-checked against the lock file before production runs.
- Every game is matched to its position (file name and FEN header) and book line.
- `analyse_removal.py` reports losses on time, copy balance, book drift, the
  ICC, and the share of games that repeat move for move (warning above 5%).
- The base Stockfish 11 build reproduces the official bench (5156767 nodes).
