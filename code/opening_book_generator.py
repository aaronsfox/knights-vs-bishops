"""opening_book_generator: build the balanced opening book for self-play.

For each starting position of the study (positions.py) the book holds a
fixed number of short opening lines (default 50 lines of 6 plies, three moves
each). Standard opening books don't apply once pieces are removed, so the
lines are generated from each position. They give self-play games varied,
realistic openings without tilting the position towards either side (each
position keeps its own imbalance; the lines leave it where it started):

1. **Near-best moves only.** At each ply a MultiPV search lists the moves
   within ``move_margin_cp`` of the best move.
2. **Random walks.** Lines are built by choosing uniformly among those moves,
   so they spread over different first moves instead of following one line.
3. **Equal treatment of both sides.** Each side's total loss over its own
   moves is capped (``max_side_loss_cp``), and White's and Black's totals must
   be close (``max_side_loss_gap_cp``).
4. **Deeper check.** Every line in the pool has its final position, and
   the starting position, re-evaluated with a deeper single-line search
   (``check_nodes``). The shallow MultiPV scores are good enough to pick
   plausible moves but too noisy to balance on.
5. **No drift.** A line's final evaluation (deeper check) must be within
   ``max_shift_cp`` of the starting position's.
6. **Balanced selection.** From the acceptable lines, the book takes lines
   from different first moves in turn, choosing each time the line that keeps
   the average shift in *expected score* (deeper check, Stockfish's WDL)
   closest to zero; it then swaps lines in and out while that brings the
   average closer to zero, keeping every first move represented. Expected score is the scale game results are on;
   balancing centipawns instead can still tilt the expected result in
   unbalanced positions, because the WDL curve is not straight. Positions with fewer
   good lines than needed repeat lines, so every position has the same number.

Searches are fresh, single-threaded and node-limited and the random choices
are seeded per position, so the same engine build regenerates the identical
book.

Subcommands
-----------
build    Generate lines for positions (one Parquet file per position;
         resumable: finished positions are skipped).
reselect Redo the line selection for built positions from their saved pools
         (no engine searches), e.g. after improving the selection.
collect  Combine the per-position files into the book: a table of every
         line, a per-position summary, and PGN and EPD files for fastchess.

Examples
--------
    python code/opening_book_generator.py build --positions all
    python code/opening_book_generator.py build --positions start --output-dir results/test_book
    python code/opening_book_generator.py collect
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import random
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn
import pandas as pd

from utils import (
    CONFIG_DIR,
    all_positions,
    REPO_ROOT,
    EngineSetupError,
    expected_score_from_wdl,
    get_position,
    load_engine_config,
    load_toml,
    open_engine,
    parse_position_spec,
    platform_info,
    setup_logging,
    utc_now,
    write_parquet_atomic,
)

log = logging.getLogger("kvb.opening_book")

DEFAULT_CONFIG = CONFIG_DIR / "opening_book.toml"
MATE_CP = 100_000


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BookSettings:
    plies: int
    lines_per_position: int
    seed: int
    candidate_nodes: int
    multipv: int
    check_nodes: int
    move_margin_cp: int
    max_side_loss_cp: int
    max_side_loss_gap_cp: int
    max_shift_cp: int
    pool_factor: float
    max_walks: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "BookSettings":
        b, s, lim, smp = config["book"], config["search"], config["limits"], config["sampling"]
        return cls(
            plies=int(b["plies"]), lines_per_position=int(b["lines_per_position"]),
            seed=int(b["seed"]), candidate_nodes=int(s["candidate_nodes"]),
            multipv=int(s["multipv"]), check_nodes=int(s.get("check_nodes", 0)),
            move_margin_cp=int(lim["move_margin_cp"]),
            max_side_loss_cp=int(lim["max_side_loss_cp"]),
            max_side_loss_gap_cp=int(lim["max_side_loss_gap_cp"]),
            max_shift_cp=int(lim["max_shift_cp"]),
            pool_factor=float(smp["pool_factor"]), max_walks=int(smp["max_walks"]),
        )


def output_dir(config: dict[str, Any], override: str | None) -> Path:
    path = Path(override or config["output"]["dir"])
    return path if path.is_absolute() else REPO_ROOT / path


def raw_path(out_dir: Path, engine_key: str, pos_id: str) -> Path:
    return out_dir / "raw" / engine_key / f"{pos_id}.parquet"


def pool_path(out_dir: Path, engine_key: str, pos_id: str) -> Path:
    return out_dir / "raw" / engine_key / "pool" / f"{pos_id}.parquet"


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    move: chess.Move
    cp_mover: int      # score after the move, from the moving side's view
    cp_white: int      # the same score from White's view


def _cp(score: chess.engine.PovScore, color: chess.Color) -> int:
    return score.pov(color).score(mate_score=MATE_CP)


def candidate_moves(engine: chess.engine.SimpleEngine, board: chess.Board,
                    nodes: int, multipv: int) -> list[Candidate]:
    """Fresh MultiPV search; returns the moves best first.

    A node-limited search can stop partway through a depth, leaving the line
    being searched with an aspiration-window bound instead of an exact score.
    Such a line's score is replaced by that move's last exact score.
    """
    latest: dict[int, chess.engine.InfoDict] = {}
    last_exact: dict[chess.Move, chess.engine.PovScore] = {}
    with engine.analysis(board, chess.engine.Limit(nodes=nodes), multipv=multipv,
                         game=object()) as analysis:
        for info in analysis:
            if "score" not in info or not info.get("pv"):
                continue
            latest[info.get("multipv", 1)] = info
            if not (info.get("lowerbound") or info.get("upperbound")):
                last_exact[info["pv"][0]] = info["score"]
        analysis.wait()
    out: dict[chess.Move, Candidate] = {}
    for info in latest.values():
        move = info["pv"][0]
        score = info["score"]
        if info.get("lowerbound") or info.get("upperbound"):
            score = last_exact.get(move, score)
        out[move] = Candidate(move, _cp(score, board.turn), _cp(score, chess.WHITE))
    if not out:
        raise RuntimeError(f"no scored lines for {board.fen()}")
    return sorted(out.values(), key=lambda c: -c.cp_mover)


def evaluate_exact(engine: chess.engine.SimpleEngine, board: chess.Board,
                   nodes: int) -> dict[str, Any]:
    """Single-line search; returns the last exact score (and WDL) from White's view."""
    last: chess.engine.InfoDict | None = None
    exact: chess.engine.InfoDict | None = None
    with engine.analysis(board, chess.engine.Limit(nodes=nodes), game=object()) as analysis:
        for info in analysis:
            if "score" not in info:
                continue
            last = info
            if not (info.get("lowerbound") or info.get("upperbound")):
                exact = info
        analysis.wait()
    info = exact or last
    if info is None:
        raise RuntimeError(f"no scored lines for {board.fen()}")
    wdl = info.get("wdl")
    e = math.nan
    if wdl is not None:
        w = wdl.white()
        e = expected_score_from_wdl(w.wins, w.draws, w.losses)
    return {"cp": _cp(info["score"], chess.WHITE), "expected": e, "depth": info.get("depth")}


# ---------------------------------------------------------------------------
# Building lines
# ---------------------------------------------------------------------------

@dataclass
class Line:
    moves: tuple[chess.Move, ...]
    cp_end: int            # final evaluation from White's view (candidate search)
    white_loss: int
    black_loss: int
    shift_cp: float = 0.0  # final minus starting evaluation, centipawns (deeper check)
    shift: float = 0.0     # final minus starting expected score (deeper check), balanced to ~0

    @property
    def first_move(self) -> chess.Move:
        return self.moves[0]


def random_walks(root: chess.Board, search, s: BookSettings, rng: random.Random
                 ) -> tuple[int, dict[tuple[chess.Move, ...], Line], int]:
    """Build distinct lines by random walks from ``root``.

    ``search(board)`` returns the candidate moves (cached by the caller).
    Returns the root evaluation (White's view), the distinct lines, and the
    number of walks made.
    """
    root_cands = search(root)
    cp_root = root_cands[0].cp_white
    target = math.ceil(s.pool_factor * s.lines_per_position)
    pool: dict[tuple[chess.Move, ...], Line] = {}
    walks = 0
    while len(pool) < target and walks < s.max_walks:
        walks += 1
        board = root.copy(stack=False)
        loss = {chess.WHITE: 0, chess.BLACK: 0}
        moves: list[chess.Move] = []
        cp_end = cp_root
        for _ in range(s.plies):
            if board.is_game_over():
                break
            cands = search(board)
            best = cands[0].cp_mover
            side = board.turn
            options = [c for c in cands
                       if best - c.cp_mover <= s.move_margin_cp
                       and loss[side] + best - c.cp_mover <= s.max_side_loss_cp]
            choice = rng.choice(options) if options else cands[0]
            loss[side] += best - choice.cp_mover
            moves.append(choice.move)
            cp_end = choice.cp_white
            board.push(choice.move)
        key = tuple(moves)
        if len(key) == s.plies and key not in pool:
            pool[key] = Line(key, cp_end, loss[chess.WHITE], loss[chess.BLACK])
    return cp_root, pool, walks


def acceptable(line: Line, s: BookSettings) -> bool:
    return (abs(line.white_loss - line.black_loss) <= s.max_side_loss_gap_cp
            and abs(line.shift_cp) <= s.max_shift_cp)


BALANCE_TOL = 1e-4   # stop improving once |mean shift| is below this


def select_lines(pool: list[Line], k: int, rng: random.Random) -> list[Line]:
    """Choose ``k`` lines from the acceptable pool.

    1. Rotation: take lines from each first move in turn, each time the line
       that keeps the running mean ``shift`` closest to zero. This spreads the
       book over the different first moves.
    2. Refinement: then swap a chosen line for an unchosen one whenever that
       brings the mean shift closer to zero, as long as every first move in
       the selection keeps at least one line. The rotation alone can be
       forced into poorly balanced lines when one first move's lines all lean
       the same way.

    If the pool has fewer than ``k`` lines, all are used and then repeated in
    the same order.
    """
    if not pool:
        return []
    groups: dict[chess.Move, list[Line]] = defaultdict(list)
    for line in pool:
        groups[line.first_move].append(line)
    order = sorted(groups, key=lambda m: m.uci())
    rng.shuffle(order)
    n = min(k, len(pool))
    chosen: list[Line] = []
    total = 0.0
    while len(chosen) < n:
        for first in order:
            if len(chosen) >= n or not groups[first]:
                continue
            cands = groups[first]
            rng.shuffle(cands)  # random tie-break
            best = min(cands, key=lambda ln: abs(total + ln.shift))
            cands.remove(best)
            chosen.append(best)
            total += best.shift

    # Refinement by best-improvement swaps.
    unchosen = [ln for first in order for ln in groups[first]]
    while unchosen and abs(total) / n > BALANCE_TOL:
        counts: dict[chess.Move, int] = defaultdict(int)
        for ln in chosen:
            counts[ln.first_move] += 1
        best_gain, best_pair = 0.0, None
        for i, out in enumerate(chosen):
            for j, inn in enumerate(unchosen):
                if counts[out.first_move] == 1 and inn.first_move != out.first_move:
                    continue  # would drop a first move from the book
                gain = abs(total) - abs(total - out.shift + inn.shift)
                if gain > best_gain + 1e-12:
                    best_gain, best_pair = gain, (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        out, inn = chosen[i], unchosen[j]
        chosen[i], unchosen[j] = inn, out
        total += inn.shift - out.shift

    distinct = len(chosen)
    i = 0
    while len(chosen) < k:
        chosen.append(chosen[i % distinct])
        i += 1
    return chosen


def selection_rng(s: "BookSettings", pos_id: str) -> random.Random:
    """The selection has its own seed, so rebuilding the book from saved pools
    (reselect) gives exactly what a fresh build would."""
    return random.Random(f"{s.seed}:{pos_id}:select")


def build_position(engine: chess.engine.SimpleEngine, pos_id: str,
                   s: BookSettings) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    position = get_position(pos_id)
    root = position.board()
    rng = random.Random(f"{s.seed}:{pos_id}")
    cache: dict[str, list[Candidate]] = {}
    n_searches = 0

    def search(board: chess.Board) -> list[Candidate]:
        nonlocal n_searches
        key = board.fen()
        if key not in cache:
            cache[key] = candidate_moves(engine, board, s.candidate_nodes, s.multipv)
            n_searches += 1
        return cache[key]

    start = time.perf_counter()
    cp_root, pool, walks = random_walks(root, search, s, rng)

    # Deeper check of the starting position and every pool line's final position.
    root_check = evaluate_exact(engine, root, s.check_nodes) if s.check_nodes else None
    end_boards: dict[tuple[chess.Move, ...], chess.Board] = {}
    checks: dict[tuple[chess.Move, ...], dict[str, Any]] = {}
    for key, line in pool.items():
        board = root.copy(stack=False)
        for move in key:
            board.push(move)
        end_boards[key] = board
        if root_check:
            checks[key] = evaluate_exact(engine, board, s.check_nodes)
            line.shift_cp = checks[key]["cp"] - root_check["cp"]
            # Balance on expected score, the scale game results are on. Equal
            # centipawn drifts either way don't cancel in expected score when
            # the position is unbalanced (the WDL curve is not straight), so
            # balancing centipawns can still tilt the expected result.
            e_shift = checks[key]["expected"] - root_check["expected"]
            line.shift = e_shift if not math.isnan(e_shift) else line.shift_cp
        else:
            line.shift_cp = line.shift = line.cp_end - cp_root

    ok = [ln for ln in pool.values() if acceptable(ln, s)]
    n_gap = sum(abs(ln.white_loss - ln.black_loss) > s.max_side_loss_gap_cp for ln in pool.values())
    n_shift = sum(abs(ln.shift_cp) > s.max_shift_cp for ln in pool.values())
    pool_first_moves = len({ln.first_move for ln in pool.values()})
    pool_shifts = pd.Series([ln.shift_cp for ln in pool.values()], dtype=float)
    used_fallback = False
    if not ok:
        # Nothing meets every limit: use the lines that drift least.
        used_fallback = True
        ok = sorted(pool.values(), key=lambda ln: abs(ln.shift_cp))[:s.lines_per_position]
    chosen = select_lines(ok, s.lines_per_position, selection_rng(s, pos_id))
    if not chosen:
        raise RuntimeError(f"position {pos_id}: no lines of {s.plies} plies found")

    rows = []
    seen: set[tuple[chess.Move, ...]] = set()
    for idx, line in enumerate(chosen):
        board = root.copy(stack=False)
        san = []
        for move in line.moves:
            san.append(board.san(move))
            board.push(move)
        chk = checks.get(line.moves)
        rows.append({
            "position_id": pos_id,
            "design": position.design,
            "start_fen": position.fen,
            "line_index": idx,
            "repeat": line.moves in seen,
            "moves_uci": " ".join(m.uci() for m in line.moves),
            "moves_san": " ".join(san),
            "first_move": line.first_move.uci(),
            "end_fen": end_boards[line.moves].fen(),
            "cp_root": cp_root,
            "cp_end": line.cp_end,
            "shift_cp": line.cp_end - cp_root,
            "white_loss_cp": line.white_loss,
            "black_loss_cp": line.black_loss,
            "cp_root_check": root_check["cp"] if root_check else None,
            "expected_root_check": root_check["expected"] if root_check else math.nan,
            "cp_end_check": chk["cp"] if chk else None,
            "expected_end_check": chk["expected"] if chk else math.nan,
            "depth_end_check": chk["depth"] if chk else None,
        })
        seen.add(line.moves)
    df = pd.DataFrame(rows)
    if root_check:
        df["shift_cp_check"] = df["cp_end_check"] - df["cp_root_check"]
        df["shift_expected_check"] = df["expected_end_check"] - df["expected_root_check"]
    info = {
        "walks": walks, "pool_size": len(pool), "pool_first_moves": pool_first_moves,
        "rejected_gap": n_gap, "rejected_shift": n_shift, "acceptable": len(ok),
        "pool_shift_mean": round(float(pool_shifts.mean()), 1),
        "pool_shift_sd": round(float(pool_shifts.std()), 1),
        "distinct_selected": len(seen), "first_moves": df["first_move"].nunique(),
        "fallback": used_fallback, "candidate_searches": n_searches,
        "check_searches": len(checks),
        "seconds": round(time.perf_counter() - start, 1),
    }
    for key, value in info.items():
        df[key] = value

    # Every pool line with its checks, so the selection can be re-examined
    # (or redone with other limits) without repeating any search.
    n_chosen: dict[tuple[chess.Move, ...], int] = defaultdict(int)
    for line in chosen:
        n_chosen[line.moves] += 1
    ok_keys = {ln.moves for ln in ok}
    pool_rows = []
    for key, line in pool.items():
        chk = checks.get(key)
        pool_rows.append({
            "position_id": pos_id,
            "moves_uci": " ".join(m.uci() for m in key),
            "first_move": line.first_move.uci(),
            "cp_end": line.cp_end,
            "white_loss_cp": line.white_loss,
            "black_loss_cp": line.black_loss,
            "cp_end_check": chk["cp"] if chk else None,
            "expected_end_check": chk["expected"] if chk else math.nan,
            "depth_end_check": chk["depth"] if chk else None,
            "shift_cp_check": line.shift_cp,
            "shift_balance": line.shift,
            "acceptable": key in ok_keys,
            "times_selected": n_chosen.get(key, 0),
        })
    pool_df = pd.DataFrame(pool_rows)
    if root_check:
        pool_df["cp_root_check"] = root_check["cp"]
        pool_df["expected_root_check"] = root_check["expected"]
    return df, pool_df, info


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_build(args: argparse.Namespace) -> int:
    config = load_toml(Path(args.config))
    s = BookSettings.from_config(config)
    out_dir = output_dir(config, args.output_dir)
    engine_key = config["book"]["engine"]
    cfg = load_engine_config(engine_key)
    overrides = config.get("uci_overrides", {})
    cfg = dataclasses.replace(cfg, uci_options={**cfg.uci_options, **overrides})

    pos_ids = parse_position_spec(args.positions)
    todo = [p for p in pos_ids if args.overwrite or not raw_path(out_dir, engine_key, p).exists()]
    log.info("book engine %s %s | %d plies, %d lines | %d positions (%d already done)",
             cfg.name, cfg.version, s.plies, s.lines_per_position, len(todo),
             len(pos_ids) - len(todo))
    if not todo:
        return 0
    try:
        engine, meta = open_engine(cfg, strict_lock=not args.allow_unlocked)
    except EngineSetupError as exc:
        log.error("%s", exc)
        return 2

    run_id = uuid.uuid4().hex[:12]
    run_meta = {"run_id": run_id, **platform_info()}
    meta_path = out_dir / "runs" / f"{run_id}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({**meta, **run_meta, "settings": dataclasses.asdict(s),
                                     "positions": todo, "started": utc_now()},
                                    indent=2, default=str))
    constant = {
        "engine": meta["engine"], "engine_version": meta["engine_version"],
        "engine_id_name": meta["engine_id_name"], "binary_sha256": meta["binary_sha256"],
        "settings_json": json.dumps(dataclasses.asdict(s), sort_keys=True),
        "run_id": run_id, "hostname": run_meta["hostname"], "cpu": run_meta["cpu"],
    }
    try:
        for n, pos_id in enumerate(todo, 1):
            df, pool_df, info = build_position(engine, pos_id, s)
            for key, value in constant.items():
                df[key] = value
            df["finished_utc"] = utc_now()
            pool_df["run_id"] = run_id
            write_parquet_atomic(pool_df, pool_path(out_dir, engine_key, pos_id))
            write_parquet_atomic(df, raw_path(out_dir, engine_key, pos_id))
            mean_shift = df["shift_cp_check"].mean() if "shift_cp_check" in df else df["shift_cp"].mean()
            mean_e = df["shift_expected_check"].mean() if "shift_expected_check" in df else math.nan
            log.info("[%d/%d] %s %s: %d distinct lines (pool %d, %d first moves), "
                     "mean shift %+.1f cp / %+.4f E%s, %d candidate searches, %.0fs",
                     n, len(todo), pos_id, get_position(pos_id).design,
                     info["distinct_selected"], info["pool_size"], info["first_moves"],
                     mean_shift, mean_e, " (FALLBACK)" if info["fallback"] else "",
                     info["candidate_searches"], info["seconds"])
    finally:
        engine.quit()
    log.info("done")
    return 0


def reselect_position(s: BookSettings, old: pd.DataFrame, pool_df: pd.DataFrame
                      ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Redo one position's selection from its saved pool (no engine searches)."""
    pos_id = str(old["position_id"].iloc[0])
    position = get_position(pos_id)
    root = position.board()
    pool_lines: list[Line] = []
    for r in pool_df.itertuples(index=False):
        moves = tuple(chess.Move.from_uci(u) for u in r.moves_uci.split())
        pool_lines.append(Line(moves, int(r.cp_end), int(r.white_loss_cp), int(r.black_loss_cp),
                               shift_cp=float(r.shift_cp_check), shift=float(r.shift_balance)))
    ok = [ln for ln in pool_lines if acceptable(ln, s)]
    used_fallback = False
    if not ok:
        used_fallback = True
        ok = sorted(pool_lines, key=lambda ln: abs(ln.shift_cp))[:s.lines_per_position]
    chosen = select_lines(ok, s.lines_per_position, selection_rng(s, pos_id))

    by_moves = pool_df.set_index("moves_uci")
    old_depth = old.drop_duplicates("moves_uci").set_index("moves_uci")["depth_end_check"]
    first = old.iloc[0]
    rows, seen = [], set()
    for idx, line in enumerate(chosen):
        board = root.copy(stack=False)
        san = []
        for move in line.moves:
            san.append(board.san(move))
            board.push(move)
        key = " ".join(m.uci() for m in line.moves)
        pr = by_moves.loc[key]
        depth = pr["depth_end_check"] if "depth_end_check" in by_moves.columns else old_depth.get(key)
        rows.append({
            "position_id": pos_id,
            "design": position.design,
            "start_fen": position.fen,
            "line_index": idx,
            "repeat": line.moves in seen,
            "moves_uci": key,
            "moves_san": " ".join(san),
            "first_move": line.first_move.uci(),
            "end_fen": board.fen(),
            "cp_root": first["cp_root"],
            "cp_end": line.cp_end,
            "shift_cp": line.cp_end - first["cp_root"],
            "white_loss_cp": line.white_loss,
            "black_loss_cp": line.black_loss,
            "cp_root_check": first["cp_root_check"],
            "expected_root_check": first["expected_root_check"],
            "cp_end_check": pr["cp_end_check"],
            "expected_end_check": pr["expected_end_check"],
            "depth_end_check": depth,
        })
        seen.add(line.moves)
    df = pd.DataFrame(rows)
    df["shift_cp_check"] = df["cp_end_check"] - df["cp_root_check"]
    df["shift_expected_check"] = df["expected_end_check"] - df["expected_root_check"]
    info = {
        "acceptable": len(ok), "distinct_selected": len(seen),
        "first_moves": df["first_move"].nunique(), "fallback": used_fallback,
    }
    # Search statistics and provenance carry over from the build.
    keep = ["walks", "pool_size", "pool_first_moves", "rejected_gap", "rejected_shift",
            "pool_shift_mean", "pool_shift_sd", "candidate_searches", "check_searches",
            "seconds", "engine", "engine_version", "engine_id_name", "binary_sha256",
            "settings_json", "run_id", "hostname", "cpu", "finished_utc"]
    for col in keep:
        if col in old.columns:
            df[col] = first[col]
    for k_, v in info.items():
        df[k_] = v
    df["selection"] = "rotation+swap"
    df["reselected_utc"] = utc_now()

    counts: dict[str, int] = defaultdict(int)
    for line in chosen:
        counts[" ".join(m.uci() for m in line.moves)] += 1
    ok_keys = {" ".join(m.uci() for m in ln.moves) for ln in ok}
    pool_df = pool_df.copy()
    pool_df["acceptable"] = pool_df["moves_uci"].isin(ok_keys)
    pool_df["times_selected"] = pool_df["moves_uci"].map(lambda m: counts.get(m, 0))
    return df, pool_df, info


def cmd_reselect(args: argparse.Namespace) -> int:
    """Redo the line selection for built positions from their saved pools."""
    config = load_toml(Path(args.config))
    s = BookSettings.from_config(config)
    out_dir = output_dir(config, args.output_dir)
    engine_key = config["book"]["engine"]
    pos_ids = parse_position_spec(args.positions)
    current = json.dumps(dataclasses.asdict(s), sort_keys=True)
    done = changed = 0
    for pos_id in pos_ids:
        raw, pool = raw_path(out_dir, engine_key, pos_id), pool_path(out_dir, engine_key, pos_id)
        if not raw.exists() or not pool.exists():
            continue
        old, pool_df = pd.read_parquet(raw), pd.read_parquet(pool)
        if "settings_json" in old and old["settings_json"].iloc[0] != current:
            log.warning("%s was built with different settings than %s; the current limits "
                        "are applied to its saved pool", pos_id, args.config)
        before = old["shift_expected_check"].mean() if "shift_expected_check" in old else math.nan
        df, pool_df, info = reselect_position(s, old, pool_df)
        after = df["shift_expected_check"].mean()
        write_parquet_atomic(pool_df, pool)
        write_parquet_atomic(df, raw)
        done += 1
        if set(df["moves_uci"]) != set(old["moves_uci"]):
            changed += 1
        if args.verbose or abs(before) > 0.002:
            log.info("%s: mean drift %+.4f -> %+.4f E (%d first moves)",
                     pos_id, before, after, info["first_moves"])
    log.info("reselected %d positions (%d with a changed selection); now run collect", done, changed)
    return 0


def write_pgn(lines: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in lines.itertuples(index=False):
            board = chess.Board(row.start_fen)
            game = chess.pgn.Game()
            game.setup(board)
            game.headers["Event"] = "Knights vs bishops self-play opening book"
            game.headers["PositionId"] = str(row.position_id)
            game.headers["LineIndex"] = str(row.line_index)
            game.headers["Result"] = "*"
            node = game
            for uci in row.moves_uci.split():
                node = node.add_variation(chess.Move.from_uci(uci))
            print(game, file=fh, end="\n\n")
    tmp.replace(path)


def write_epd(lines: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in lines.itertuples(index=False):
            board = chess.Board(row.end_fen)
            fh.write(board.epd(id=f"{row.position_id}_l{row.line_index:02d}") + "\n")
    tmp.replace(path)


def cmd_collect(args: argparse.Namespace) -> int:
    config = load_toml(Path(args.config))
    out_dir = output_dir(config, args.output_dir)
    engine_key = config["book"]["engine"]
    files = sorted((out_dir / "raw" / engine_key).glob("*.parquet"))
    if not files:
        log.error("no book files in %s", out_dir / "raw" / engine_key)
        return 1
    lines = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    order = {pid: i for i, pid in enumerate(p.id for p in all_positions())}
    lines = (lines.assign(_order=lines.position_id.map(order))
             .sort_values(["_order", "line_index"]).drop(columns="_order").reset_index(drop=True))
    k = int(config["book"]["lines_per_position"])
    counts = lines.groupby("position_id").size()
    if (counts != k).any():
        log.warning("positions without exactly %d lines: %s", k, counts[counts != k].to_dict())

    agg = {
        "design": ("design", "first"),
        "lines": ("line_index", "size"),
        "distinct_lines": ("distinct_selected", "first"),
        "pool_size": ("pool_size", "first"),
        "first_moves": ("first_moves", "first"),
        "fallback": ("fallback", "first"),
        "cp_root": ("cp_root", "first"),
        "mean_shift_cp": ("shift_cp", "mean"),
        "mean_white_loss_cp": ("white_loss_cp", "mean"),
        "mean_black_loss_cp": ("black_loss_cp", "mean"),
        "seconds": ("seconds", "first"),
    }
    if "shift_cp_check" in lines:
        agg.update({
            "cp_root_check": ("cp_root_check", "first"),
            "mean_shift_cp_check": ("shift_cp_check", "mean"),
            "sd_shift_cp_check": ("shift_cp_check", "std"),
            "expected_root_check": ("expected_root_check", "first"),
            "mean_shift_expected_check": ("shift_expected_check", "mean"),
        })
    summary = lines.groupby("position_id", sort=False).agg(**agg).reset_index()

    book_dir = out_dir / "book"
    write_parquet_atomic(lines, book_dir / "opening_book_lines.parquet")
    lines.drop(columns=["settings_json"]).to_csv(book_dir / "opening_book_lines.csv", index=False)
    summary.to_csv(book_dir / "opening_book_summary.csv", index=False)
    write_pgn(lines, book_dir / "opening_book.pgn")
    write_epd(lines, book_dir / "opening_book.epd")
    log.info("wrote %s: %d positions, %d lines", book_dir, len(summary), len(lines))
    log.info("distinct lines per position: min %d, median %d; positions with repeats: %d; "
             "fallbacks: %d", summary.distinct_lines.min(), summary.distinct_lines.median(),
             int((summary.distinct_lines < k).sum()), int(summary.fallback.sum()))
    log.info("mean shift per position (cp): mean %+.2f, largest |shift| %.1f",
             summary.mean_shift_cp.mean(), summary.mean_shift_cp.abs().max())
    if "mean_shift_cp_check" in summary:
        log.info("deeper check, mean shift per position (cp): mean %+.2f, largest |shift| %.1f",
                 summary.mean_shift_cp_check.mean(), summary.mean_shift_cp_check.abs().max())
        log.info("deeper check, mean shift per position (expected score, the balanced "
                 "quantity): mean %+.4f, largest |shift| %.4f",
                 summary.mean_shift_expected_check.mean(),
                 summary.mean_shift_expected_check.abs().max())
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=str(DEFAULT_CONFIG))
    common.add_argument("--output-dir", help="override [output] dir in the config")
    common.add_argument("-v", "--verbose", action="store_true")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", parents=[common], help="generate lines for positions")
    p.add_argument("--positions", default="all")
    p.add_argument("--overwrite", action="store_true", help="redo finished positions")
    p.add_argument("--allow-unlocked", action="store_true",
                   help="run without a lock-file entry (testing only)")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("reselect", parents=[common],
                       help="redo the line selection from saved pools (no searches)")
    p.add_argument("--positions", default="all")
    p.set_defaults(func=cmd_reselect)

    p = sub.add_parser("collect", parents=[common], help="combine into the book files")
    p.set_defaults(func=cmd_collect)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
