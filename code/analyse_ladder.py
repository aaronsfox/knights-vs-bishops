"""Question 3: does the value of the minor pieces change with strength?

Question 2 repeated down a strength ladder (Maia-3 rungs, Stockfish at fixed
node budgets or UCI_Elo settings, and the full-strength engines):

* **3a. Calibrate the rungs:** a Bradley-Terry (Elo) scale fitted to games
  between players (matches with an opponent, from the standard position),
  anchored at the full-strength reference player (default stockfish = 0).
  Rungs are reported on this measured scale, not on their nominal settings.
* **3b. Question 2 at each rung:** the key comparisons and designs, analysed
  exactly as in analyse_removal.py (each rung has its own pawn-odds games,
  so effects are in that rung's pawn equivalents).
* **3c. Effect vs strength:** each design's value in pawns against measured
  strength, with a weighted regression (weights 1/SE^2) per design.

    python code/analyse_ladder.py --players stockfish_elo1500,stockfish_elo2000,stockfish
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import analyse_removal as ar
import stats_core as sc
from utils import REPO_ROOT

OUT_TABLES = REPO_ROOT / "results" / "selfplay" / "analysis"
OUT_FIGURES = REPO_ROOT / "results" / "figures"


def calibration(tables: Path, players: list[str], anchor: str) -> pd.DataFrame:
    """Bradley-Terry ratings from player-vs-opponent games among ``players``."""
    games = pd.read_parquet(tables / "selfplay_games.parquet").dropna(subset=["score_white"])
    g = games[(games.opponent.fillna("") != "") & games.player.isin(players)
              & games.opponent.isin(players)]
    if g.empty:
        return pd.DataFrame(columns=["player", "elo", "elo_se"])
    g = g.assign(points=np.where(g.white_engine == g.player, g.score_white, 1 - g.score_white))
    pairs = (g.groupby(["player", "opponent"]).agg(points_a=("points", "sum"), games=("points", "size"))
             .reset_index().rename(columns={"player": "a", "opponent": "b"}))
    # Half a point in one extra game per pairing keeps a 100% or 0% score finite.
    pairs = pairs.assign(points_a=pairs.points_a + 0.5, games=pairs.games + 1)
    return sc.bradley_terry(pairs, anchor)


def effect_vs_strength(designs: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if "pawns" not in designs or ratings.empty:
        return pd.DataFrame()
    d = designs.merge(ratings.rename(columns={"player": "match"}), on="match")
    for design, g in d[d.key_design].groupby("design"):
        g = g.dropna(subset=["pawns", "elo"])
        if len(g) < 3:
            continue
        se = (g.pawns_hi - g.pawns_lo) / (2 * 1.96)
        w = 1 / np.maximum(se, 1e-6) ** 2
        X = np.column_stack([np.ones(len(g)), g.elo / 100])
        beta = np.linalg.solve(X.T @ (X * w.to_numpy()[:, None]), X.T @ (w * g.pawns))
        cov = np.linalg.inv(X.T @ (X * w.to_numpy()[:, None]))
        rows.append({"design": design, "players": len(g), "intercept_pawns_at_anchor": beta[0],
                     "slope_pawns_per_100_elo": beta[1], "slope_se": float(np.sqrt(cov[1, 1]))})
    return pd.DataFrame(rows)


def fig_effect_vs_strength(designs: pd.DataFrame, ratings: pd.DataFrame, out_dir: Path) -> list[Path]:
    from plot_style import MUTED_TEXT, apply_style, bold_tick_labels
    plt = apply_style()
    d = designs.merge(ratings.rename(columns={"player": "match"}), on="match")
    d = d[d.key_design].dropna(subset=["pawns"])
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    colours = {"BN v NN": "#104485", "BB v BN": "#f17f03", "BB v NN": "#418c47"}
    for design, g in d.groupby("design"):
        g = g.sort_values("elo")
        c = colours.get(design, "#444444")
        ax.vlines(g.elo, g.pawns_lo, g.pawns_hi, color=c, lw=0.9, alpha=0.7)
        ax.plot(g.elo, g.pawns, color=c, marker="o", ms=4)
    ax.axhline(0, color=MUTED_TEXT, lw=0.7, ls=(0, (1, 2)))
    ax.set_xlabel("Measured strength (Elo, reference = 0)")
    ax.set_ylabel("Value to the focal side (pawns)")
    bold_tick_labels(ax)
    return ar.save(fig, "ladder_effect_vs_strength", out_dir)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tables", default=str(ar.TABLES))
    p.add_argument("--book", default=str(ar.BOOK))
    p.add_argument("--players", required=True, help="comma-separated rungs and engines")
    p.add_argument("--anchor", default="stockfish", help="player fixed at 0 Elo")
    p.add_argument("--tables-dir", default=str(OUT_TABLES))
    p.add_argument("--figures-dir", default=str(OUT_FIGURES))
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)
    players = args.players.split(",")
    tables = Path(args.tables)
    book = pd.read_parquet(args.book) if Path(args.book).exists() else None
    out = ar.run(ar.load_games(tables, players), book, Path(args.tables_dir), None, prefix="ladder")
    ratings = calibration(tables, players, args.anchor) if args.anchor in players else pd.DataFrame()
    trend = effect_vs_strength(out["designs"], ratings)
    ratings.to_csv(Path(args.tables_dir) / "ladder_ratings.csv", index=False, float_format="%.6g")
    trend.to_csv(Path(args.tables_dir) / "ladder_effect_vs_strength.csv", index=False,
                 float_format="%.6g")
    print("Measured strength (Bradley-Terry):")
    print(ratings.to_string(index=False) if len(ratings) else "  no games between players yet")
    if len(trend):
        print(trend.to_string(index=False))
    if not args.no_figures and len(ratings) and "pawns" in out["designs"]:
        fig_effect_vs_strength(out["designs"], ratings, Path(args.figures_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
