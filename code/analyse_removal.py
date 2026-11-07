"""Question 3 (Step 3): start-position value of the minor pieces (the removal matrix).

Reads the ``selfplay.py collect`` tables (self-play matches only) and follows
the analysis plan:

* **Quality checks and design effect** per match: games per position, line
  matching, losses on time, the two engine copies' White scores, book drift,
  and the ICC of games within book lines, giving each comparison's DEFF.
* **Comparisons** (positions.py): each position is pooled with its
  colour-flipped twin and scored from the focal side (the side with more
  bishops, or with the extra pawn). Each comparison gets a Dirichlet
  posterior (stats_core.py), giving the focal side's E and Delta_Elo with 95%
  intervals, the decisive rate and the focal side's share of decisive games.
  Mirrored controls (positions that are their own twin) give White's edge.
* **3b. Cell estimates and pawn equivalents.** Every comparison, plus each
  design's mean over its comparisons (BN v NN = single bishop v knight,
  BB v BN = the pair over one bishop, BB v NN = both), in Elo and in pawn
  equivalents: Delta_Elo / Delta_Elo of the pawn-odds comparisons for the
  same player, draw by draw. White's edge in each position, and the
  colour interaction (focal side as White minus as Black).
* **3c. Additivity:** Delta(BB v NN) - [Delta(BB v BN) + Delta(BN v NN)].
* **3d. Which square:** the comparison table, ranked within each design.
* **3e. Evaluation vs outcome** (when evaluate_sp tables exist): outcomes
  regressed on the same engine's top-budget evaluation, with the correlation
  corrected for game noise.
* **3f. Decisiveness:** decisive rate against |Delta_Elo|.
* **3g. Agreement** between engines: Spearman's rho with
  a bootstrap interval, corrected for noise, and Bland-Altman.

Everything is ranked and reported in full; counts at probability thresholds
are for transparency only (analysis plan: rank, don't select).

    python code/analyse_removal.py
    python code/analyse_removal.py --matches stockfish,caissa
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import stats_core as sc
from positions import all_positions, get_position
from utils import REPO_ROOT

TABLES = REPO_ROOT / "results" / "selfplay" / "tables"
BOOK = REPO_ROOT / "results" / "opening_book" / "book" / "opening_book_lines.parquet"
EVALS = REPO_ROOT / "results" / "evaluate_sp" / "tables" / "evaluate_sp_final.parquet"
OUT_TABLES = REPO_ROOT / "results" / "selfplay" / "analysis"
OUT_FIGURES = REPO_ROOT / "results" / "figures"

KEY_DESIGNS = ("BN v NN", "BB v BN", "BB v NN")
PAWN_COMPARISONS = ("f7", "b7")      # comparison ids of the pawn-odds pairs
THRESHOLDS = (0.80, 0.95)
LINE = ["position_id", "batch", "line_index"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_games(tables: Path = TABLES, matches: list[str] | None = None) -> pd.DataFrame:
    games = pd.read_parquet(tables / "selfplay_games.parquet")
    games = games[games.opponent.fillna("") == ""]            # self-play only
    if matches:
        games = games[games.match.isin(matches)]
    return add_focal(games.dropna(subset=["score_white"]))


def add_focal(games: pd.DataFrame) -> pd.DataFrame:
    """Comparison, design and the focal side's score for every game."""
    info = pd.DataFrame([{
        "position_id": p.id, "comparison_id": p.comparison_id, "design": p.design,
        "group": p.group, "mirrored_control": p.is_mirrored_control,
        "focal": None if p.focal is None else ("white" if p.focal else "black"),
    } for p in all_positions()])
    g = games.merge(info, on="position_id", how="left")
    g["focal_is_white"] = g.focal != "black"        # controls: White's view
    g["focal_score"] = np.where(g.focal_is_white, g.score_white, 1 - g.score_white)
    return g


# ---------------------------------------------------------------------------
# Quality checks and the design effect
# ---------------------------------------------------------------------------

def match_icc(g: pd.DataFrame) -> float:
    return sc.icc_one_way(g, "score_white", cluster=["batch", "line_index"],
                          stratum=["position_id"])


def quality_checks(games: pd.DataFrame, book: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    rows, iccs = [], {}
    for match, g in games.groupby("match"):
        per_pos = g.groupby("position_id").size()
        icc = match_icc(g)
        iccs[match] = icc
        sizes = g.groupby(LINE).size()
        copies = g.groupby("white_player").score_white.mean()
        row = {
            "match": match, "positions": len(per_pos), "games": len(g),
            "games_per_position_min": int(per_pos.min()), "games_per_position_max": int(per_pos.max()),
            "unmatched_lines": int((g.line_index < 0).sum()),
            "fen_mismatches": int((~g.fen_matches_position.astype(bool)).sum())
            if "fen_matches_position" in g else 0,
            "losses_on_time": int((g.termination == "time forfeit").sum()),
            "abnormal_terminations": int((~g.termination.isin(["normal", "adjudication"])).sum()),
            "draw_rate": float((g.score_white == 0.5).mean()), "mean_plies": float(g.plies.mean()),
            # Share of games that repeat an earlier game of the same line move
            # for move (a deterministic player); these add no information.
            "repeated_game_share": (1 - g.drop_duplicates(LINE + ["moves_hash"]).shape[0] / len(g))
            if "moves_hash" in g else np.nan,
            "copy_white_scores": "; ".join(f"{k} {v:.4f}" for k, v in copies.items()),
            "icc": icc, "kish_games_per_line": sc.kish_cluster_size(sizes),
            "deff_typical": sc.design_effect(icc, sizes),
        }
        if book is not None and "shift_expected_check" in book:
            drift = book.groupby("position_id").shift_expected_check.mean()
            row["max_abs_position_mean_drift"] = float(drift.reindex(per_pos.index).abs().max())
            m = g.merge(book[["position_id", "line_index", "shift_expected_check"]],
                        on=["position_id", "line_index"], how="inner")
            if len(m) > 2:
                m["res"] = m.score_white - m.groupby("position_id").score_white.transform("mean")
                m["shift_c"] = (m.shift_expected_check
                                - m.groupby("position_id").shift_expected_check.transform("mean"))
                if m.shift_c.std() > 0:
                    fit = stats.linregress(m.shift_c, m.res)
                    row.update(book_shift_slope=fit.slope, book_shift_slope_se=fit.stderr)
                    m["adj"] = m.score_white - fit.slope * m.shift_c
                    row["icc_book_adjusted"] = sc.icc_one_way(
                        m, "adj", cluster=["batch", "line_index"], stratum=["position_id"])
        rows.append(row)
    return pd.DataFrame(rows), iccs


# ---------------------------------------------------------------------------
# Posteriors
# ---------------------------------------------------------------------------

def comparison_posteriors(games: pd.DataFrame, iccs: dict[str, float]
                          ) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    """Per match and comparison: the focal side's posterior summaries.
    Returns the table and the Delta_Elo draws keyed by (match, comparison)."""
    rows, draws = [], {}
    for (match, cid), g in games.groupby(["match", "comparison_id"], sort=False):
        w, d, l = sc.wdl(g.focal_score)
        deff = sc.design_effect(iccs[match], g.groupby(LINE).size())
        p = sc.posterior_draws(w, d, l, deff, sc.rng_for(match, cid))
        draws[(match, cid)] = sc.elo(p[:, 0] + p[:, 1] / 2)
        first = g.iloc[0]
        rows.append({
            "match": match, "comparison_id": cid, "design": first.design, "group": first.group,
            "mirrored_control": bool(first.mirrored_control),
            "positions": " ".join(sorted(g.position_id.unique())),
            "games": len(g), "focal_wins": w, "draws": d, "focal_losses": l, "deff": deff,
            "escore_observed": float(g.focal_score.mean()),
            **sc.summarise_draws(p),
        })
    out = pd.DataFrame(rows)
    out["rank_in_design"] = (out.groupby(["match", "design"]).delta_elo
                             .rank(ascending=False, method="min").astype(int))
    for t in THRESHOLDS:
        out[f"favours_focal_{int(t * 100):03d}"] = out.p_positive >= t
    return out, draws


def position_posteriors(games: pd.DataFrame, iccs: dict[str, float]) -> pd.DataFrame:
    """Per match and position: White's score (White's edge in that position)."""
    rows = []
    for (match, pid), g in games.groupby(["match", "position_id"], sort=False):
        w, d, l = sc.wdl(g.score_white)
        deff = sc.design_effect(iccs[match], g.groupby(LINE).size())
        p = sc.posterior_draws(w, d, l, deff, sc.rng_for(match, pid, "white"))
        first = g.iloc[0]
        rows.append({"match": match, "position_id": pid, "design": first.design,
                     "comparison_id": first.comparison_id, "focal": first.focal,
                     "games": len(g), "white_wins": w, "draws": d, "black_wins": l, "deff": deff,
                     **sc.summarise_draws(p, prefix="white_")})
    return pd.DataFrame(rows)


def colour_interaction(games: pd.DataFrame, iccs: dict[str, float]) -> pd.DataFrame:
    """Per comparison with two positions: the focal side's Delta_Elo when it
    is White minus when it is Black (does the piece matter more with the
    first move?)."""
    rows = []
    for (match, cid), g in games[~games.mirrored_control].groupby(["match", "comparison_id"]):
        parts = {}
        for is_white, h in g.groupby("focal_is_white"):
            w, d, l = sc.wdl(h.focal_score)
            deff = sc.design_effect(iccs[match], h.groupby(LINE).size())
            p = sc.posterior_draws(w, d, l, deff, sc.rng_for(match, cid, "colour", int(is_white)))
            parts[bool(is_white)] = sc.elo(p[:, 0] + p[:, 1] / 2)
        if len(parts) < 2:
            continue
        diff = parts[True] - parts[False]
        lo, hi = sc.interval(diff)
        rows.append({"match": match, "comparison_id": cid, "design": g.design.iloc[0],
                     "focal_white_delta_elo": float(np.median(parts[True])),
                     "focal_black_delta_elo": float(np.median(parts[False])),
                     "difference": float(np.median(diff)), "difference_lo": lo,
                     "difference_hi": hi})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3b, 3c: designs, pawn equivalents, additivity
# ---------------------------------------------------------------------------

def pawn_draws(comp: pd.DataFrame, draws: dict) -> dict[str, np.ndarray]:
    """Per match: Delta_Elo draws of one pawn (mean of the pawn-odds comparisons)."""
    out = {}
    for match in comp.match.unique():
        parts = [draws[(match, c)] for c in PAWN_COMPARISONS if (match, c) in draws]
        if parts:
            out[match] = np.mean(parts, axis=0)
    return out


def design_draws(comp: pd.DataFrame, draws: dict) -> dict[tuple[str, str], np.ndarray]:
    """Per match and design: the mean Delta_Elo over its comparisons (equal
    weights), excluding mirrored controls, which measure White's edge."""
    out = {}
    use = comp[~comp.mirrored_control & comp.group.isin(["1v1", "2v2"])]
    for (match, design), g in use.groupby(["match", "design"]):
        out[(match, design)] = np.mean([draws[(match, c)] for c in g.comparison_id], axis=0)
    return out


def summarise(x: np.ndarray, prefix: str) -> dict[str, float]:
    lo, hi = sc.interval(x)
    return {prefix: float(np.median(x)), f"{prefix}_lo": lo, f"{prefix}_hi": hi,
            f"p_{prefix}_positive": float((x > 0).mean())}


def design_summary(comp: pd.DataFrame, draws: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    pawns = pawn_draws(comp, draws)
    dd = design_draws(comp, draws)
    rows = []
    for (match, design), x in dd.items():
        row = {"match": match, "design": design, "key_design": design in KEY_DESIGNS,
               "comparisons": int(((comp.match == match) & (comp.design == design)
                                   & ~comp.mirrored_control).sum()),
               **summarise(x, "delta_elo")}
        if match in pawns:
            row.update(summarise(x / pawns[match], "pawns"))
        rows.append(row)
    ref = [{"match": m, **summarise(x, "one_pawn_delta_elo")} for m, x in pawns.items()]
    add = []
    for match in comp.match.unique():
        need = [(match, d) for d in ("BB v NN", "BB v BN", "BN v NN")]
        if all(k in dd for k in need):
            gap = dd[need[0]] - dd[need[1]] - dd[need[2]]
            r = {"match": match, **summarise(gap, "additivity_gap_delta_elo")}
            if match in pawns:
                r.update(summarise(gap / pawns[match], "additivity_gap_pawns"))
            add.append(r)
    out = pd.DataFrame(rows, columns=None if rows else ["match", "design", "key_design",
                                                         "comparisons", "delta_elo",
                                                         "delta_elo_lo", "delta_elo_hi"])
    if len(out):
        out = out.sort_values(["match", "key_design", "delta_elo"], ascending=[True, False, False])
    designs = out.merge(pd.DataFrame(ref), on="match", how="left") if ref and len(out) else out
    return designs, pd.DataFrame(add)


# ---------------------------------------------------------------------------
# 3e, 3f, 3g
# ---------------------------------------------------------------------------

def evaluation_vs_outcome(comp: pd.DataFrame, evals_path: Path = EVALS) -> pd.DataFrame:
    """Outcome Delta_Elo regressed on the same engine's top-budget evaluation
    (focal side's view, mean over the comparison's positions)."""
    if not evals_path.exists():
        return pd.DataFrame()
    ev = pd.read_parquet(evals_path)
    ev = ev[(ev.multipv_setting == 1) & (ev.multipv_index == 1)]
    top = ev.loc[ev.groupby(["engine", "position_id"]).budget_nodes.idxmax()]
    rows = []
    for match, g in comp[~comp.mirrored_control].groupby("match"):
        t = top[top.engine == match].set_index("position_id")
        if t.empty or t.expected_score.isna().all():
            continue
        xs = []
        for r in g.itertuples():
            vals = []
            for pid in r.positions.split():
                p = get_position(pid)
                e = t.expected_score.get(pid, np.nan)
                vals.append(sc.elo(e if p.focal != False else 1 - e))  # noqa: E712
            xs.append(np.nanmean(vals))
        x, y = np.asarray(xs, float), g.delta_elo.to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 3:
            continue
        f = stats.linregress(x[ok], y[ok])
        rel = sc.reliability(g.delta_elo.to_numpy()[ok], g.delta_elo_sd.to_numpy()[ok])
        rows.append({"match": match, "comparisons": int(ok.sum()), "slope": f.slope,
                     "slope_se": f.stderr, "intercept": f.intercept, "r2": f.rvalue ** 2,
                     "pearson": f.rvalue, "spearman": stats.spearmanr(x[ok], y[ok]).statistic,
                     "outcome_reliability": rel,
                     "pearson_corrected": f.rvalue / np.sqrt(rel) if rel > 0 else np.nan})
    return pd.DataFrame(rows)


def decisiveness(comp: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    fits, predictors = [], {}
    for match, g in comp.groupby("match"):
        x = g.delta_elo.abs().to_numpy()
        y = 1 - g.draws.to_numpy() / g.games.to_numpy()
        n = g.games.to_numpy() / g.deff.to_numpy()
        if len(g) < 4 or y.min() == y.max():
            continue
        beta, se, predict = sc.weighted_logistic(x, y, n)
        predictors[match] = predict
        fits.append({"match": match, "b0": beta[0], "b1": beta[1], "b2": beta[2],
                     "b0_se": se[0], "b1_se": se[1], "b2_se": se[2],
                     "decisive_at_0": float(predict(0.0)[0]),
                     "decisive_at_50": float(predict(50.0)[0]),
                     "decisive_at_100": float(predict(100.0)[0]),
                     "spearman_decisive_abs_delta": stats.spearmanr(x, y).statistic})
    return pd.DataFrame(fits), predictors


def agreement(comp: pd.DataFrame) -> pd.DataFrame:
    """Pairwise agreement between matches on the comparisons' Delta_Elo."""
    rows = []
    use = comp[~comp.mirrored_control]
    matches = sorted(use.match.unique())
    rng = np.random.default_rng(sc.SEED)
    for i, a in enumerate(matches):
        for b in matches[i + 1:]:
            A = use[use.match == a].set_index("comparison_id")
            B = use[use.match == b].set_index("comparison_id").reindex(A.index).dropna(subset=["delta_elo"])
            A = A.reindex(B.index)
            if len(A) < 4:
                continue
            x, y = A.delta_elo.to_numpy(), B.delta_elo.to_numpy()
            idx = rng.integers(0, len(x), size=(sc.N_BOOT, len(x)))
            boots = [stats.spearmanr(x[j], y[j]).statistic for j in idx]
            rel_a = sc.reliability(x, A.delta_elo_sd.to_numpy())
            rel_b = sc.reliability(y, B.delta_elo_sd.to_numpy())
            rho = stats.spearmanr(x, y).statistic
            diff, mean = y - x, (x + y) / 2
            ba = stats.linregress(mean, diff)
            rows.append({"match_a": a, "match_b": b, "comparisons": len(x), "spearman": rho,
                         "spearman_lo": np.nanpercentile(boots, 2.5),
                         "spearman_hi": np.nanpercentile(boots, 97.5),
                         "reliability_a": rel_a, "reliability_b": rel_b,
                         "spearman_corrected": rho / np.sqrt(rel_a * rel_b)
                         if rel_a > 0 and rel_b > 0 else np.nan,
                         "mean_difference_b_minus_a": diff.mean(),
                         "loa_low": diff.mean() - 1.96 * diff.std(ddof=1),
                         "loa_high": diff.mean() + 1.96 * diff.std(ddof=1),
                         "difference_on_mean_slope": ba.slope})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures (one plot per file, no legends; style from plot_style.py)
# ---------------------------------------------------------------------------

def save(fig, name: str, out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.pdf", out_dir / f"{name}.png"]
    for p in paths:
        fig.savefig(p, metadata={"CreationDate": None} if p.suffix == ".pdf" else None)
    plt.close(fig)
    return paths


def style_for(match: str):
    from plot_style import ENGINES, EngineStyle
    base = match.split("_n")[0] if match not in ENGINES else match
    return ENGINES.get(match) or ENGINES.get(base) or EngineStyle(match, "#444444", "o", "-")


def fig_comparisons_ranked(comp: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Every comparison's focal-side Delta_Elo with its 95% interval, ranked
    within each match; key designs filled, others open."""
    from plot_style import METRICS, MUTED_TEXT, apply_style, bold_tick_labels
    plt = apply_style()
    use = comp[~comp.mirrored_control & comp.group.isin(["1v1", "2v2"])]
    matches = sorted(use.match.unique())
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    width = 0.8 / max(len(matches), 1)
    for j, match in enumerate(matches):
        s = style_for(match)
        g = use[use.match == match].sort_values("delta_elo", ascending=False).reset_index(drop=True)
        x = np.arange(1, len(g) + 1) + (j - (len(matches) - 1) / 2) * width
        ax.vlines(x, g.delta_elo_lo, g.delta_elo_hi, color=s.colour, lw=0.8, alpha=0.6)
        key = g.design.isin(KEY_DESIGNS).to_numpy()
        ax.scatter(x[key], g.delta_elo[key], s=12, color=s.colour, marker=s.marker, zorder=3, lw=0)
        ax.scatter(x[~key], g.delta_elo[~key], s=12, facecolor="white", edgecolor=s.colour,
                   marker=s.marker, zorder=3, lw=0.8)
    ax.axhline(0, color=MUTED_TEXT, lw=0.7, ls=(0, (1, 2)))
    ax.set_xlabel("Comparison rank")
    ax.set_ylabel(f"Focal side {METRICS['delta_elo'].bold_symbol}")
    bold_tick_labels(ax)
    return save(fig, "removal_comparisons_ranked", out_dir)


def fig_designs(designs: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Each design's mean value in pawn equivalents (Elo if no pawn-odds
    games), with 95% intervals; one marker per match."""
    from plot_style import MUTED_TEXT, apply_style, bold_tick_labels
    plt = apply_style()
    col = "pawns" if "pawns" in designs and designs.pawns.notna().any() else "delta_elo"
    order = [d for d in designs.sort_values(["key_design", col], ascending=[False, False])
             .design.unique()]
    matches = sorted(designs.match.unique())
    fig, ax = plt.subplots(figsize=(3.5, 0.3 * len(order) + 0.8))
    for j, match in enumerate(matches):
        s = style_for(match)
        g = designs[designs.match == match].set_index("design").reindex(order)
        y = np.arange(len(order)) + (j - (len(matches) - 1) / 2) * 0.2
        ax.hlines(y, g[f"{col}_lo"], g[f"{col}_hi"], color=s.colour, lw=1.0)
        ax.scatter(g[col], y, s=14, color=s.colour, marker=s.marker, zorder=3, lw=0)
    ax.axvline(0, color=MUTED_TEXT, lw=0.7, ls=(0, (1, 2)))
    ax.set_yticks(np.arange(len(order)), order)
    ax.invert_yaxis()
    ax.set_xlabel("Value to the focal side (pawns)" if col == "pawns"
                  else "Value to the focal side (Elo)")
    bold_tick_labels(ax)
    return save(fig, "removal_designs", out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def summary_text(qc, designs, add, agree) -> str:
    L = ["Question 3: start-position value (removal matrix)", ""]
    for r in qc.itertuples():
        L.append(f"{r.match}: {r.games} games over {r.positions} positions, draws "
                 f"{100 * r.draw_rate:.1f}%, ICC {r.icc:.3f}, DEFF ~{r.deff_typical:.2f}, "
                 f"losses on time {r.losses_on_time}, unmatched lines {r.unmatched_lines}, "
                 f"repeated games {100 * r.repeated_game_share:.0f}%")
        if r.repeated_game_share > 0.05:
            L.append("    WARNING: many games repeat move for move (a deterministic player, e.g. a "
                     "fixed node budget); use 2 games per line with more lines")
    L += ["", "Designs (mean over comparisons; focal side's value)"]
    for r in designs.itertuples():
        pawns = (f" | {r.pawns:+.2f} pawns [{r.pawns_lo:+.2f}, {r.pawns_hi:+.2f}]"
                 if hasattr(r, "pawns") and pd.notna(getattr(r, "pawns", np.nan)) else "")
        L.append(f"  {r.match:16s} {r.design:10s} Delta_Elo {r.delta_elo:+6.1f} "
                 f"[{r.delta_elo_lo:+.1f}, {r.delta_elo_hi:+.1f}]{pawns}")
    if len(add):
        L += ["", "Additivity: Delta(BB v NN) - Delta(BB v BN) - Delta(BN v NN)"]
        for r in add.itertuples():
            L.append(f"  {r.match}: {r.additivity_gap_delta_elo:+.1f} Elo "
                     f"[{r.additivity_gap_delta_elo_lo:+.1f}, {r.additivity_gap_delta_elo_hi:+.1f}]")
    if len(agree):
        L += ["", "Agreement between players (comparisons' Delta_Elo)"]
        for r in agree.itertuples():
            L.append(f"  {r.match_a} vs {r.match_b}: Spearman {r.spearman:.2f} "
                     f"[{r.spearman_lo:.2f}, {r.spearman_hi:.2f}], corrected {r.spearman_corrected:.2f}")
    return "\n".join(L) + "\n"


def run(games: pd.DataFrame, book: pd.DataFrame | None, tables_dir: Path, figures_dir: Path | None,
        evals_path: Path = EVALS, prefix: str = "removal") -> dict[str, pd.DataFrame]:
    qc, iccs = quality_checks(games, book)
    comp, draws = comparison_posteriors(games, iccs)
    pos = position_posteriors(games, iccs)
    colour = colour_interaction(games, iccs)
    designs, add = design_summary(comp, draws)
    reg = evaluation_vs_outcome(comp, evals_path)
    dfit, _ = decisiveness(comp)
    agree = agreement(comp)
    out = {"quality_checks": qc, "comparisons": comp, "positions": pos,
           "colour_interaction": colour, "designs": designs, "additivity": add,
           "evaluation_vs_outcome": reg, "decisive_fit": dfit, "agreement": agree}
    tables_dir.mkdir(parents=True, exist_ok=True)
    for name, t in out.items():
        t.to_csv(tables_dir / f"{prefix}_{name}.csv", index=False, float_format="%.6g")
    text = summary_text(qc, designs, add, agree)
    (tables_dir / f"{prefix}_summary.txt").write_text(text)
    print(text)
    if figures_dir is not None and len(comp):
        fig_comparisons_ranked(comp, figures_dir)
        if len(designs):
            fig_designs(designs, figures_dir)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tables", default=str(TABLES), help="selfplay collect tables folder")
    p.add_argument("--book", default=str(BOOK))
    p.add_argument("--evaluations", default=str(EVALS))
    p.add_argument("--matches", help="comma-separated self-play matches (default: all)")
    p.add_argument("--tables-dir", default=str(OUT_TABLES))
    p.add_argument("--figures-dir", default=str(OUT_FIGURES))
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)
    games = load_games(Path(args.tables), args.matches.split(",") if args.matches else None)
    book_path = Path(args.book)
    book = (pd.read_parquet(book_path, columns=["position_id", "line_index", "shift_expected_check"])
            if book_path.exists() else None)
    run(games, book, Path(args.tables_dir), None if args.no_figures else Path(args.figures_dir),
        Path(args.evaluations))
    return 0


if __name__ == "__main__":
    sys.exit(main())
