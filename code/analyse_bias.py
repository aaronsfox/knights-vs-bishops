"""Question 2 (Step 2): does preferring one piece win games? (manipulated engines)

Each Stockfish 11 variant plays the unmodified build (engines/sf11) from the
standard position, colours swapped in pairs over the opening book. A variant
scoring above 0.5 gains from its preference; below 0.5 it loses from it.

* **2a. Variant vs base:** the variant's W/D/L, with a Dirichlet posterior
  (stats_core.py; DEFF from the ICC of games within book lines), giving its
  E and Delta_Elo with 95% intervals, the decisive rate and P(Delta_Elo > 0).
* **2b. Likes vs loves:** within each direction (knight, bishop), the
  stronger preference minus the milder one, from independent posterior draws.

Gate 2 (analysis plan): the base build against itself should score ~0 Elo;
if such a match exists (sf11 vs sf11, or sf11 self-play) it is reported too.

    python code/analyse_bias.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import stats_core as sc
from utils import REPO_ROOT, load_engine_config

TABLES = REPO_ROOT / "results" / "selfplay" / "tables"
OUT_TABLES = REPO_ROOT / "results" / "selfplay" / "analysis"
OUT_FIGURES = REPO_ROOT / "results" / "figures"
BASE = "sf11"
DIRECTIONS = {"knight": ("sf11_n020", "sf11_n050"), "bishop": ("sf11_b020", "sf11_b050")}
LINE = ["position_id", "batch", "line_index"]


def display_name(key: str) -> str:
    try:
        return load_engine_config(key).display_name
    except Exception:  # noqa: BLE001 (a key without a manifest, e.g. a named player)
        return key


def load_games(tables: Path = TABLES, base: str = BASE) -> pd.DataFrame:
    games = pd.read_parquet(tables / "selfplay_games.parquet").dropna(subset=["score_white"])
    variants = games[games.opponent == base]
    # Gate 2: the base build's self-play, from the positions the variants play.
    gate = games[(games.player == base) & (games.opponent.fillna("") == "")
                 & games.position_id.isin(variants.position_id.unique())]
    g = pd.concat([variants, gate]).copy()
    # The player's score: self-play of the base build is scored for copy "_a".
    player_white = (g.white_engine == g.player) & ((g.opponent != "") | g.white_player.str.endswith("_a"))
    g["player_score"] = np.where(player_white, g.score_white, 1 - g.score_white)
    g["player_is_white"] = player_white
    return g


def variant_posteriors(games: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows, draws = [], {}
    for match, g in games.groupby("match"):
        icc = sc.icc_one_way(g, "player_score", cluster=["batch", "line_index"],
                             stratum=["position_id"])
        deff = sc.design_effect(icc, g.groupby(LINE).size())
        w, d, l = sc.wdl(g.player_score)
        p = sc.posterior_draws(w, d, l, deff, sc.rng_for(match))
        player = g.player.iloc[0]
        draws[player if g.opponent.iloc[0] else f"{player} (self-play)"] = sc.elo(p[:, 0] + p[:, 1] / 2)
        rows.append({"match": match, "player": player, "display_name": display_name(player),
                     "opponent": g.opponent.iloc[0] or f"{player} (self-play)",
                     "games": len(g), "wins": w, "draws": d, "losses": l,
                     "white_games": int(g.player_is_white.sum()),
                     "losses_on_time": int((g.termination == "time forfeit").sum()),
                     "icc": icc, "deff": deff, **sc.summarise_draws(p)})
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("delta_elo", ascending=False).reset_index(drop=True)
    return out, draws


def likes_vs_loves(draws: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for direction, (likes, loves) in DIRECTIONS.items():
        if likes in draws and loves in draws:
            diff = draws[loves] - draws[likes]
            lo, hi = sc.interval(diff)
            rows.append({"direction": direction, "likes": likes, "loves": loves,
                         "loves_minus_likes_delta_elo": float(np.median(diff)),
                         "lo": lo, "hi": hi, "p_loves_better": float((diff > 0).mean())})
    return pd.DataFrame(rows)


def fig_variants(post: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Each variant's Delta_Elo against the base build, 95% intervals, by name."""
    from plot_style import METRICS, MUTED_TEXT, apply_style, bold_tick_labels
    plt = apply_style()
    g = post[post.opponent == BASE].sort_values("delta_elo")
    fig, ax = plt.subplots(figsize=(3.5, 0.32 * max(len(g), 1) + 0.8))
    y = np.arange(len(g))
    ax.hlines(y, g.delta_elo_lo, g.delta_elo_hi, color="#104485", lw=1.2)
    ax.scatter(g.delta_elo, y, s=18, color="#104485", zorder=3, lw=0)
    ax.axvline(0, color=MUTED_TEXT, lw=0.7, ls=(0, (1, 2)))
    ax.set_yticks(y, g.display_name)
    ax.set_xlabel(f"{METRICS['delta_elo'].bold_symbol} against Stockfish 11")
    bold_tick_labels(ax)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / "bias_variants.pdf", out_dir / "bias_variants.png"]
    for p in paths:
        fig.savefig(p, metadata={"CreationDate": None} if p.suffix == ".pdf" else None)
    plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tables", default=str(TABLES))
    p.add_argument("--tables-dir", default=str(OUT_TABLES))
    p.add_argument("--figures-dir", default=str(OUT_FIGURES))
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)
    games = load_games(Path(args.tables))
    if games.empty:
        print(f"no games against {BASE} in {args.tables}")
        return 1
    post, draws = variant_posteriors(games)
    lvl = likes_vs_loves(draws)
    out = Path(args.tables_dir)
    out.mkdir(parents=True, exist_ok=True)
    post.to_csv(out / "bias_variants.csv", index=False, float_format="%.6g")
    lvl.to_csv(out / "bias_likes_vs_loves.csv", index=False, float_format="%.6g")
    lines = ["Question 2: variants against Stockfish 11", ""]
    for r in post.itertuples():
        lines.append(f"  {r.display_name:22s} ({r.player}) vs {r.opponent}: {r.games} games, "
                     f"W {r.wins} D {r.draws} L {r.losses} | Delta_Elo {r.delta_elo:+.1f} "
                     f"[{r.delta_elo_lo:+.1f}, {r.delta_elo_hi:+.1f}], P(>0) {r.p_positive:.2f}")
    for r in lvl.itertuples():
        lines.append(f"  {r.direction}: loves - likes {r.loves_minus_likes_delta_elo:+.1f} Elo "
                     f"[{r.lo:+.1f}, {r.hi:+.1f}]")
    text = "\n".join(lines) + "\n"
    (out / "bias_summary.txt").write_text(text)
    print(text)
    if not args.no_figures and (post.opponent == BASE).any():
        fig_variants(post, Path(args.figures_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
