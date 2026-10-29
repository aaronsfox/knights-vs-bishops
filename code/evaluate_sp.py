"""evaluate_sp: evaluate the study's starting positions (the screening pass).

Evaluates starting positions (positions.py: the standard position with
minor pieces or pawns removed) with one engine at a ladder of node
budgets. Every budget is a separate, fresh, single-threaded search, so each
value is exactly what an N-node search returns and reruns are reproducible.

Subcommands
-----------
run      Evaluate positions with one engine; writes one Parquet file per
         position (resumable: finished positions are skipped).
collect  Combine the per-position files into tidy tables: every info line,
         one evaluation row per search, and convergence metrics per position.

The evaluation of a search is its last line with an exact score. When the
node limit stops a search partway through a depth, the engine's last line is
often an aspiration-window bound (the true score is only known to be above or
below it), so it isn't used as the evaluation. The best move is always the
engine's own `bestmove`, and the last line is kept alongside as `last_*`.

Examples
--------
    python code/evaluate_sp.py run --engine stockfish --positions all
    python code/evaluate_sp.py run --engine stockfish --positions start,c1-b8 --multipv 5
    python code/evaluate_sp.py collect
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import chess
import chess.engine
import pandas as pd

from utils import (
    CONFIG_DIR,
    REPO_ROOT,
    EngineSetupError,
    elo_from_expected,
    expected_score_from_wdl,
    get_position,
    load_engine_config,
    load_toml,
    open_engine,
    parse_position_spec,
    platform_info,
    setup_logging,
    tail_drift,
    total_variation,
    utc_now,
    write_parquet_atomic,
)

log = logging.getLogger("kvb.evaluate_sp")

DEFAULT_CONFIG = CONFIG_DIR / "evaluate_sp.toml"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict[str, Any]:
    return load_toml(path)


def output_dir(config: dict[str, Any], override: str | None) -> Path:
    path = Path(override or config["output"]["dir"])
    return path if path.is_absolute() else REPO_ROOT / path


def budgets_for(config: dict[str, Any], engine_key: str, multipv: int,
                override: list[int] | None) -> list[int]:
    """Node budgets for this engine. MultiPV > 1 runs at the top budget only."""
    budgets = sorted(override or config["budgets"][engine_key])
    if multipv > 1 and config.get("multipv", {}).get("top_budget_only", True):
        return budgets[-1:]
    return budgets


def raw_path(out_dir: Path, engine_key: str, multipv: int, pos_id: str) -> Path:
    return out_dir / "raw" / engine_key / f"mpv{multipv}" / f"{pos_id}.parquet"


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------

def info_to_row(info: chess.engine.InfoDict) -> dict[str, Any]:
    """Flatten one UCI info line. Scores and WDL are from White's view
    (White is to move in every starting position)."""
    score = info["score"].white()
    wdl = info.get("wdl")
    w = d = l = None
    expected = math.nan
    if wdl is not None:
        wdl_white = wdl.white()
        w, d, l = wdl_white.wins, wdl_white.draws, wdl_white.losses
        expected = expected_score_from_wdl(w, d, l)
    return {
        "multipv_index": info.get("multipv", 1),
        "depth": info.get("depth"),
        "seldepth": info.get("seldepth"),
        "nodes": info.get("nodes"),
        "time_s": info.get("time"),
        "nps": info.get("nps"),
        "hashfull": info.get("hashfull"),
        "cp_white": score.score(),
        "mate_white": score.mate(),
        "wdl_w": w,
        "wdl_d": d,
        "wdl_l": l,
        "expected_score": expected,
        # Aspiration-window results are bounds, not exact scores. Engines may
        # print them only after some wall-clock time, so their presence can
        # vary between identical runs; the final result does not.
        "bound": "lower" if info.get("lowerbound") else "upper" if info.get("upperbound") else "exact",
        "pv": " ".join(move.uci() for move in info.get("pv", [])),
    }


def evaluation_row(search_rows: list[dict[str, Any]], pv_index: int = 1) -> dict[str, Any]:
    """The line that gives a search's evaluation for one principal variation:
    the last exact-score line, or the last line if none is exact."""
    lines = [r for r in search_rows if r["multipv_index"] == pv_index]
    exact = [r for r in lines if r["bound"] == "exact"]
    return (exact or lines)[-1]


def search(engine: chess.engine.SimpleEngine, board: chess.Board, nodes: int,
           multipv: int) -> tuple[list[dict[str, Any]], str | None, float]:
    """Run one fresh node-limited search and return every scored info line.

    Passing a new ``game`` object makes python-chess send ``ucinewgame``, so
    hash tables and search trees from earlier searches are cleared.

    With one thread and a node limit the search is deterministic: rerunning
    gives the same final score, WDL and principal variation. Intermediate
    lines can differ slightly between runs because some engines only print
    them once a search has run for a few seconds (see ``bound``).
    """
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    with engine.analysis(board, chess.engine.Limit(nodes=nodes), multipv=multipv,
                         game=object(), info=chess.engine.INFO_ALL) as analysis:
        for info in analysis:
            if "score" in info:
                rows.append(info_to_row(info))
        best = analysis.wait()
    wall = time.perf_counter() - start

    # Flag the last line reported for each principal variation. It may be an
    # aspiration-window bound; `collect` takes the evaluation from the last
    # exact line instead (see evaluation_row).
    last: dict[int, int] = {}
    for i, row in enumerate(rows):
        last[row["multipv_index"]] = i
    for i, row in enumerate(rows):
        row["is_final"] = i in last.values()
    bestmove = best.move.uci() if best.move else None
    if best.move is not None and not board.is_legal(best.move):
        raise RuntimeError(f"engine returned illegal move {bestmove} in {board.fen()}")
    return rows, bestmove, wall


def evaluate_position(engine: chess.engine.SimpleEngine, pos_id: str,
                      budgets: list[int], multipv: int) -> pd.DataFrame:
    position = get_position(pos_id)
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        search_rows, bestmove, wall = search(engine, position.board(), budget, multipv)
        if not search_rows:
            raise RuntimeError(f"no scored info lines for position {pos_id} at {budget} nodes")
        for row in search_rows:
            row.update(budget_nodes=budget, bestmove=bestmove, search_wall_s=wall)
        rows.extend(search_rows)
        final = evaluation_row(search_rows)
        log.info("  %-12s %11d nodes  depth %3s  cp %6s  E %s  %.1fs",
                 pos_id, budget, final["depth"], final["cp_white"],
                 "  n/a" if math.isnan(final["expected_score"]) else f"{final['expected_score']:.4f}",
                 wall)
    df = pd.DataFrame(rows)
    df.insert(0, "position_id", pos_id)
    df.insert(1, "design", position.design)
    df.insert(2, "fen", position.fen)
    df.insert(3, "multipv_setting", multipv)
    return df


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    out_dir = output_dir(config, args.output_dir)
    cfg = load_engine_config(args.engine)
    budgets = budgets_for(config, args.engine, args.multipv, args.budgets)
    pos_ids = parse_position_spec(args.positions)

    todo = [p for p in pos_ids
            if args.overwrite or not raw_path(out_dir, args.engine, args.multipv, p).exists()]
    log.info("engine %s %s | MultiPV %d | budgets %s | %d positions (%d already done)",
             cfg.name, cfg.version, args.multipv, budgets, len(todo), len(pos_ids) - len(todo))
    if not todo:
        return 0

    try:
        engine, meta = open_engine(cfg, strict_lock=not args.allow_unlocked)
    except EngineSetupError as exc:
        log.error("%s", exc)
        return 2
    if not meta["version_matches_id"]:
        log.warning("engine reports '%s' but engine.toml pins version %s",
                    meta["engine_id_name"], cfg.version)

    run_meta = {
        "run_id": uuid.uuid4().hex[:12],
        "config_file": str(Path(args.config).resolve().relative_to(REPO_ROOT))
        if Path(args.config).resolve().is_relative_to(REPO_ROOT) else str(args.config),
        **platform_info(),
    }
    constant = {
        "engine": meta["engine"],
        "engine_name": meta["engine_name"],
        "engine_version": meta["engine_version"],
        "engine_id_name": meta["engine_id_name"],
        "binary_sha256": meta["binary_sha256"],
        "network_sha256": meta["network_sha256"],
        "uci_options_json": json.dumps(meta["uci_options_applied"], sort_keys=True),
        **run_meta,
    }
    meta_path = out_dir / "runs" / f"{run_meta['run_id']}.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({**meta, **run_meta, "budgets": budgets,
                                     "multipv": args.multipv, "positions": todo,
                                     "started": utc_now()}, indent=2, default=str))

    try:
        for n, pos_id in enumerate(todo, 1):
            log.info("[%d/%d] position %s (%s)", n, len(todo), pos_id,
                     get_position(pos_id).design)
            df = evaluate_position(engine, pos_id, budgets, args.multipv)
            for key, value in constant.items():
                df[key] = value
            df["finished_utc"] = utc_now()
            write_parquet_atomic(df, raw_path(out_dir, args.engine, args.multipv, pos_id))
    finally:
        engine.quit()
    log.info("done")
    return 0


SEARCH_KEYS = ["engine", "position_id", "multipv_setting", "budget_nodes", "multipv_index"]

SUMMARY_COLS = [
    "engine", "engine_version", "position_id", "design", "fen",
    "multipv_setting", "multipv_index", "budget_nodes", "depth", "seldepth",
    "nodes", "time_s", "bound", "cp_white", "mate_white", "wdl_w", "wdl_d", "wdl_l",
    "expected_score", "pv", "bestmove", "search_wall_s",
]

LAST_COLS = ["depth", "nodes", "bound", "cp_white", "mate_white", "expected_score"]


def evaluation_table(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per search and principal variation: the evaluation line.

    The evaluation is the last line with an exact score; if a search has none,
    its last line is used and ``eval_source`` says so. The search's actual last
    line (often a bound) is kept in the ``last_*`` columns.
    """
    raw = raw.reset_index(drop=True)
    raw["_order"] = range(len(raw))   # lines are stored in the order printed
    last = raw.loc[raw["is_final"], SEARCH_KEYS + LAST_COLS]
    last = last.rename(columns={c: f"last_{c}" for c in LAST_COLS})

    exact = raw[raw["bound"] == "exact"].sort_values("_order")
    chosen = exact.groupby(SEARCH_KEYS, sort=False).tail(1).assign(eval_source="last_exact")
    have = chosen.set_index(SEARCH_KEYS).index
    rest = raw.loc[raw["is_final"]]
    rest = rest[~rest.set_index(SEARCH_KEYS).index.isin(have)].assign(eval_source="last_line_bound")

    summary = pd.concat([chosen, rest])[SUMMARY_COLS + ["eval_source"]]
    summary = summary.merge(last, on=SEARCH_KEYS, how="left")
    summary["elo_equivalent"] = summary["expected_score"].map(elo_from_expected)
    return summary.sort_values(SEARCH_KEYS).reset_index(drop=True)


def convergence_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Per engine and position: values at the top budget plus convergence
    measures across the budget ladder (MultiPV = 1 searches only)."""
    primary = summary[(summary.multipv_setting == 1) & (summary.multipv_index == 1)]
    records = []
    for (engine, pos_id), grp in primary.groupby(["engine", "position_id"], sort=True):
        grp = grp.sort_values("budget_nodes")
        e = grp["expected_score"].tolist()
        cp = grp["cp_white"].astype(float).tolist()
        top = grp.iloc[-1]
        records.append({
            "engine": engine,
            "position_id": pos_id,
            "design": top["design"],
            "n_budgets": len(grp),
            "top_budget_nodes": int(top["budget_nodes"]),
            "top_depth": top["depth"],
            "top_cp_white": top["cp_white"],
            "top_expected_score": top["expected_score"],
            "top_elo_equivalent": elo_from_expected(top["expected_score"]),
            "top_draw_prob": (top["wdl_d"] / 1000.0) if pd.notna(top["wdl_d"]) else math.nan,
            "tv_expected": total_variation(e),
            "tail_drift_expected": tail_drift(e),
            "tv_cp": total_variation(cp),
            "tail_drift_cp": tail_drift(cp),
        })
    return pd.DataFrame(records)


def cmd_collect(args: argparse.Namespace) -> int:
    config = load_config(Path(args.config))
    out_dir = output_dir(config, args.output_dir)
    pattern = f"raw/{args.engine or '*'}/mpv*/*.parquet"
    files = sorted(out_dir.glob(pattern))
    if not files:
        log.error("no result files match %s", out_dir / pattern)
        return 1
    raw = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    log.info("read %d files, %d info lines", len(files), len(raw))

    # Keep only the budgets in the current config, so searches from an older
    # ladder (e.g. 1e9 nodes) don't mix into the tables.
    if not args.all_budgets:
        allowed = {engine: set(b) for engine, b in config["budgets"].items()}
        keep = [
            engine not in allowed or budget in allowed[engine]
            for engine, budget in zip(raw["engine"], raw["budget_nodes"])
        ]
        dropped = raw.loc[[not k for k in keep]]
        if len(dropped):
            for (engine, budget), n in dropped.groupby(["engine", "budget_nodes"]).size().items():
                log.info("ignoring %d info lines for %s at %d nodes (not in config)", n, engine, budget)
        raw = raw.loc[keep].reset_index(drop=True)

    summary = evaluation_table(raw)
    n_fallback = int((summary["eval_source"] != "last_exact").sum())
    if n_fallback:
        log.warning("%d searches have no exact-score line; their evaluation is the "
                    "last (bound) line (eval_source = last_line_bound)", n_fallback)
    convergence = convergence_table(summary)

    tables = out_dir / "tables"
    write_parquet_atomic(raw, tables / "evaluate_sp_info_lines.parquet")
    write_parquet_atomic(summary, tables / "evaluate_sp_final.parquet")
    summary.to_csv(tables / "evaluate_sp_final.csv", index=False)
    convergence.to_csv(tables / "evaluate_sp_convergence.csv", index=False)
    log.info("wrote %s (%d searches, %d engine-position pairs)",
             tables, len(summary), len(convergence))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=str(DEFAULT_CONFIG), help="study config (TOML)")
    common.add_argument("--output-dir", help="override the config's output directory")
    common.add_argument("-v", "--verbose", action="store_true")

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", parents=[common], help="evaluate positions with one engine")
    run.add_argument("--engine", required=True, help="folder name under engines/")
    run.add_argument("--positions", default="all",
                     help='ids or groups, e.g. "all", "2v2", "start,c1-b8", or "@file"')
    run.add_argument("--multipv", type=int, default=1)
    run.add_argument("--budgets", type=int, nargs="+",
                     help="override the config's node budgets (testing)")
    run.add_argument("--overwrite", action="store_true", help="redo finished positions")
    run.add_argument("--allow-unlocked", action="store_true",
                     help="run without a lock-file entry (development only)")
    run.set_defaults(func=cmd_run)

    collect = sub.add_parser("collect", parents=[common], help="combine results into tables")
    collect.add_argument("--engine", help="only this engine (default: all)")
    collect.add_argument("--all-budgets", action="store_true",
                         help="keep searches at budgets no longer in the config")
    collect.set_defaults(func=cmd_collect)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
