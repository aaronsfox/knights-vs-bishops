"""Exact endgame value from Syzygy tables (optional; not in the current plan).

The current plan judges endgames by the outcomes of real games (Step 1,
analysis 1c). This sampling mode, which draws uniform random positions for
swapped material pairs, is kept as an optional theoretical baseline.


For each material set (config/tablebase.toml) sample legal positions
uniformly, probe the tablebases for the result with best play (WDL) and the
distance to zeroing (DTZ), and record pawn features. Swapped pairs compare
the same material with the minor piece changing sides, e.g. KBP v KN (the
bishop side has the extra pawn) against KNP v KB (the knight side has it).

The **strong side** is always White in the sampled position before the side
to move is randomised: the side with the extra pawn(s), or, in the direct
contests (KBP v KNP, KRB v KRN), the side with the bishop.

Subcommands
-----------
run      Sample and probe one or more material sets; one Parquet file per
         set under raw/ (resumable: finished sets are skipped).
collect  Combine into tables: every sampled position, and per set the strong
         side's win/draw/loss rates with Jeffreys intervals, DTZ of wins, and the
         swapped-pair differences.

Examples
--------
    python code/tablebase.py run --sets KBPvKN,KNPvKB --samples 100000
    python code/tablebase.py run                        # every set in the config
    python code/tablebase.py collect

Uniform sampling is over legal placements with the side to move random and
positions where the side not to move is in check excluded; no castling or en
passant. 5- and 6-piece tables fit on disk (~150 GB); 7-piece sets need the
7-piece tables locally (~17 TB), since sampling at this scale is far beyond
what the Lichess tablebase API allows.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.syzygy
import numpy as np
import pandas as pd
from scipy import stats

from utils import CONFIG_DIR, REPO_ROOT, load_toml, setup_logging, utc_now, write_parquet_atomic

log = logging.getLogger("kvb.tablebase")
DEFAULT_CONFIG = CONFIG_DIR / "tablebase.toml"

PIECES = {"K": chess.KING, "Q": chess.QUEEN, "R": chess.ROOK, "B": chess.BISHOP,
          "N": chess.KNIGHT, "P": chess.PAWN}


# ---------------------------------------------------------------------------
# Material sets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MaterialSet:
    """e.g. "KBPvKN": White (the strong side) K+B+P against Black K+N."""

    name: str

    @property
    def sides(self) -> tuple[str, str]:
        white, black = self.name.split("v")
        if not (white.startswith("K") and black.startswith("K")):
            raise ValueError(f"bad material set {self.name!r}")
        return white, black

    @property
    def n_pieces(self) -> int:
        return sum(len(s) for s in self.sides)

    def pieces(self) -> list[tuple[chess.Color, chess.PieceType]]:
        white, black = self.sides
        return ([(chess.WHITE, PIECES[c]) for c in white]
                + [(chess.BLACK, PIECES[c]) for c in black])

    @property
    def minor(self) -> str:
        """The strong side's minor piece ('B' or 'N')."""
        white = self.sides[0]
        return "B" if "B" in white else "N"


def sample_position(ms: MaterialSet, rng: random.Random, max_tries: int = 10_000) -> chess.Board:
    """A uniformly random legal position with this material (rejection sampling)."""
    pieces = ms.pieces()
    for _ in range(max_tries):
        board = chess.Board(None)
        squares = rng.sample(range(64), len(pieces))
        ok = True
        for (colour, pt), sq in zip(pieces, squares):
            if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                ok = False
                break
            board.set_piece_at(sq, chess.Piece(pt, colour))
        if not ok:
            continue
        board.turn = rng.random() < 0.5
        board.castling_rights = chess.BB_EMPTY
        board.ep_square = None
        if board.is_valid():
            return board
    raise RuntimeError(f"no legal position found for {ms.name}")


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def is_passed(board: chess.Board, sq: chess.Square, colour: chess.Color) -> bool:
    """No enemy pawn ahead on the same or adjacent files."""
    f, r = chess.square_file(sq), chess.square_rank(sq)
    ahead = range(r + 1, 8) if colour == chess.WHITE else range(0, r)
    for ef in (f - 1, f, f + 1):
        if 0 <= ef < 8:
            for er in ahead:
                p = board.piece_at(chess.square(ef, er))
                if p and p.piece_type == chess.PAWN and p.color != colour:
                    return False
    return True


def square_is_light(sq: chess.Square) -> bool:
    return (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 1


def features(board: chess.Board, strong: chess.Color = chess.WHITE) -> dict[str, Any]:
    """Pawn features from the strong side's view (as recorded for endgames reached in Question 1c)."""
    pawns = list(board.pieces(chess.PAWN, chess.WHITE)) + list(board.pieces(chess.PAWN, chess.BLACK))
    own = list(board.pieces(chess.PAWN, strong))
    files = {chess.square_file(s) for s in pawns}
    bishops = list(board.pieces(chess.BISHOP, strong))
    out: dict[str, Any] = {
        "pawns_total": len(pawns),
        "pawns_one_wing": bool(files) and (max(files) <= 3 or min(files) >= 4),
        "strong_passed_pawns": sum(is_passed(board, s, strong) for s in own),
        "strong_rook_pawns": sum(chess.square_file(s) in (0, 7) for s in own),
        "strong_most_advanced_rank": (max((chess.square_rank(s) if strong else 7 - chess.square_rank(s))
                                          for s in own) + 1) if own else 0,
        "wrong_bishop": None, "bishop_pawn_same_colour_share": None,
    }
    if bishops and own:
        b_light = square_is_light(bishops[0])
        same = [square_is_light(s) == b_light for s in own]
        out["bishop_pawn_same_colour_share"] = float(np.mean(same))
        # The classic wrong bishop: all own pawns on one rook file, and the
        # bishop can't cover that file's promotion square.
        rook_files = {chess.square_file(s) for s in own}
        if rook_files in ({0}, {7}):
            promo = chess.square(rook_files.pop(), 7 if strong == chess.WHITE else 0)
            out["wrong_bishop"] = square_is_light(promo) != b_light
    return out


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def probe(tb: chess.syzygy.Tablebase, board: chess.Board) -> tuple[int, int | None]:
    """(WDL, DTZ) from White's (the strong side's) view. WDL is +2 win,
    +1 cursed win, 0 draw, -1 blessed loss, -2 loss (side to move's view in
    Syzygy, converted here)."""
    wdl = tb.probe_wdl(board)
    dtz = tb.probe_dtz(board)
    if board.turn == chess.BLACK:
        wdl, dtz = -wdl, -dtz
    return wdl, dtz


def run_set(tb: chess.syzygy.Tablebase, ms: MaterialSet, n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(f"{seed}:{ms.name}")
    rows = []
    for i in range(n):
        board = sample_position(ms, rng)
        wdl, dtz = probe(tb, board)
        rows.append({"material": ms.name, "sample": i, "fen": board.fen(),
                     "white_to_move": board.turn == chess.WHITE, "wdl": wdl, "dtz": dtz,
                     **features(board)})
    return pd.DataFrame(rows)


def raw_path(out_dir: Path, name: str) -> Path:
    return out_dir / "raw" / f"{name}.parquet"


def output_dir(config: dict[str, Any], override: str | None) -> Path:
    path = Path(override or config["output"]["dir"])
    return path if path.is_absolute() else REPO_ROOT / path


def cmd_run(args: argparse.Namespace) -> int:
    config = load_toml(Path(args.config))
    out_dir = output_dir(config, args.output_dir)
    names = args.sets.split(",") if args.sets else [n for pair in config["pairs"] for n in pair["sets"]]
    names = list(dict.fromkeys(names))
    n = args.samples or int(config["sampling"]["samples_per_set"])
    seed = int(config["sampling"]["seed"])
    tb_path = Path(args.tablebases or config["tablebases"]["path"])
    todo = [s for s in names if args.overwrite or not raw_path(out_dir, s).exists()]
    log.info("%d material sets (%d already done), %d samples each, tables %s",
             len(todo), len(names) - len(todo), n, tb_path)
    if not todo:
        return 0
    tb = chess.syzygy.open_tablebase(str(tb_path))
    try:
        for name in todo:
            start = time.perf_counter()
            try:
                df = run_set(tb, MaterialSet(name), n, seed)
            except (KeyError, chess.syzygy.MissingTableError) as exc:
                log.error("%s: missing tables (%s); skipped", name, exc)
                continue
            df["finished_utc"] = utc_now()
            write_parquet_atomic(df, raw_path(out_dir, name))
            w = (df.wdl > 0).mean()
            log.info("%s: %d positions, strong side wins %.1f%% (%.0fs)", name, len(df), 100 * w,
                     time.perf_counter() - start)
    finally:
        tb.close()
    return 0


# ---------------------------------------------------------------------------
# Collect: win rates and DTZ
# ---------------------------------------------------------------------------

def jeffreys(k: int, n: int) -> tuple[float, float, float]:
    """Proportion with its Jeffreys 95% interval."""
    lo, hi = stats.beta.ppf([0.025, 0.975], k + 0.5, n - k + 0.5)
    return k / n if n else np.nan, float(lo), float(hi)


def set_summary(df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    win, draw, loss = int((df.wdl > 0).sum()), int((df.wdl == 0).sum()), int((df.wdl < 0).sum())
    # Cursed wins (+1) are wins with best play that the 50-move rule turns into draws.
    row = {"positions": n, "cursed_wins": int((df.wdl == 1).sum())}
    for name, k in (("win", win), ("draw", draw), ("loss", loss)):
        p, lo, hi = jeffreys(k, n)
        row.update({f"{name}_rate": p, f"{name}_lo": lo, f"{name}_hi": hi})
    wins = df[df.wdl == 2].dtz.abs()
    row.update({"dtz_win_median": float(wins.median()) if len(wins) else np.nan,
                "dtz_win_q25": float(wins.quantile(.25)) if len(wins) else np.nan,
                "dtz_win_q75": float(wins.quantile(.75)) if len(wins) else np.nan})
    return row


def cmd_collect(args: argparse.Namespace) -> int:
    config = load_toml(Path(args.config))
    out_dir = output_dir(config, args.output_dir)
    files = sorted((out_dir / "raw").glob("*.parquet"))
    if not files:
        log.error("no sampled sets under %s", out_dir / "raw")
        return 1
    allpos = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    sets = pd.DataFrame([{"material": m, "strong_minor": MaterialSet(m).minor,
                          "pieces": MaterialSet(m).n_pieces, **set_summary(g)}
                         for m, g in allpos.groupby("material")])
    pairs = []
    for pair in config["pairs"]:
        if len(pair["sets"]) != 2:
            continue        # a direct contest: one set, reported in tablebase_sets.csv
        a, b = pair["sets"]
        if a in set(sets.material) and b in set(sets.material):
            ra, rb = (sets.set_index("material").loc[x] for x in (a, b))
            diff = ra.win_rate - rb.win_rate
            se = np.sqrt(ra.win_rate * (1 - ra.win_rate) / ra.positions
                         + rb.win_rate * (1 - rb.win_rate) / rb.positions)
            pairs.append({"pair": pair["name"], "bishop_set": a, "knight_set": b,
                          "bishop_win_rate": ra.win_rate, "knight_win_rate": rb.win_rate,
                          "difference": diff, "difference_lo": diff - 1.96 * se,
                          "difference_hi": diff + 1.96 * se})
    tables = out_dir / "tables"
    write_parquet_atomic(allpos, tables / "tablebase_positions.parquet")
    sets.to_csv(tables / "tablebase_sets.csv", index=False, float_format="%.6g")
    pd.DataFrame(pairs).to_csv(tables / "tablebase_pairs.csv", index=False, float_format="%.6g")
    for r in sets.itertuples():
        log.info("%-10s %6d positions: win %.1f%% [%.1f, %.1f], draw %.1f%%, DTZ of wins median %s",
                 r.material, r.positions, 100 * r.win_rate, 100 * r.win_lo, 100 * r.win_hi,
                 100 * r.draw_rate, r.dtz_win_median)
    (tables / "tablebase_pairs.json").write_text(json.dumps(pairs, indent=2, default=float))
    log.info("wrote %s", tables)
    return 0


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=str(DEFAULT_CONFIG))
    common.add_argument("--output-dir", help="override [output] dir in the config")
    common.add_argument("-v", "--verbose", action="store_true")
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", parents=[common], help="sample and probe material sets")
    r.add_argument("--sets", help="comma-separated, e.g. KBPvKN,KNPvKB (default: all in config)")
    r.add_argument("--samples", type=int, help="override samples_per_set")
    r.add_argument("--tablebases", help="override [tablebases] path")
    r.add_argument("--overwrite", action="store_true")
    r.set_defaults(func=cmd_run)
    c = sub.add_parser("collect", parents=[common], help="combine into tables")
    c.set_defaults(func=cmd_collect)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
