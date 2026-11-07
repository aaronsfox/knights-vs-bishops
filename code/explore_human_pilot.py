"""Exploratory look at bishop and knight events in human games (Step 1 pilot).

Reads the games table from lichess_games.py and the event tables from
human_events.py (same --tag), applies the inclusion rules, and writes a
summary of what the data can show. Descriptive only: no engine evaluations
are needed (Lichess's own are summarised where present), and outcomes are
compared with the score expected from the two ratings and White's edge.

Sections
--------
1. Data: inclusion flow, games by rating band, terminations, clock and
   evaluation coverage.
2. Survival of the minor pieces: Kaplan-Meier curves for the original
   bishops and knights (by type, origin square and bishop colour), censored at
   the end of the game; which piece type is captured first; what captures it.
3. Exchanges: level minor-for-minor exchanges per game; for bishop-for-knight
   exchanges, whether the player who started it gave the bishop (BxN) or the
   knight (NxB), when, whether it broke a bishop pair, and whether the
   stronger player kept the bishop.
4. Exchange opportunities: how often a defended knight is taken by a bishop
   when that's possible, against a defended bishop taken by a knight.
5. Outcomes of imbalances: for each minor-piece design (B v N, BB v BN, ...)
   and for a clean extra pawn, the score of the side with more bishops (or
   the pawn) against the score expected from ratings; the bishop's edge in
   "pawns" is its residual over the pawn-up residual.
6. Endgames: minor-piece endgames reached, and the bishop side's residual in
   B v N endgames by pawn structure.

    python code/explore_human_pilot.py --tag pilot
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize

from plot_style import MUTED_TEXT, apply_style
from utils import REPO_ROOT, setup_logging

log = logging.getLogger("kvb.pilot")

OUT_DIR = REPO_ROOT / "results" / "lichess"
FIG_DIR = REPO_ROOT / "results" / "figures"
BAD_TERMINATIONS = {"Abandoned", "Rules infraction"}
PROVISIONAL_CHANGE = 30
MIN_PLIES = 10
BAND_EDGES = [0, 1800, 2000, 2200, 2400, 4000]
BAND_LABELS = ["<1800", "1800-1999", "2000-2199", "2200-2399", "2400+"]
PHASES = [0, 20, 40, 60, 80, 1000]
PHASE_LABELS = ["moves 1-10", "11-20", "21-30", "31-40", "41+"]
COLOURS = {"B": "#104485", "N": "#f17f03"}


# ---------------------------------------------------------------------------
# Loading and inclusion
# ---------------------------------------------------------------------------

def load(out_dir: Path, tag: str | None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    suffix = f"_{tag}" if tag else ""
    raw = sorted((out_dir / "games").glob(f"lichess_*{suffix}.parquet")) or \
        sorted((out_dir / "raw").glob(f"lichess_*{suffix}.parquet"))
    if not raw:
        raise SystemExit(f"no games under {out_dir / 'games'} or {out_dir / 'raw'} for tag {tag!r}"
                         " (run: python code/lichess_games.py headers --tag ...)")
    cols = ["month", "game_id", "white_id", "black_id", "white_elo", "black_elo",
            "white_rating_diff", "black_rating_diff", "white_title", "black_title", "score_white",
            "category", "termination", "plies", "has_clock", "has_eval", "white_berserk",
            "black_berserk", "tournament"]
    games = pd.concat([pd.read_parquet(f, columns=cols) for f in raw], ignore_index=True)
    events = {}
    for t in ("pieces", "exchanges", "opportunities", "spells", "endgames", "games"):
        files = sorted((out_dir / "events").glob(f"{t}_*{suffix}.parquet"))
        events[t] = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    return games, events


def include(games: pd.DataFrame, parsed: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    g = games.merge(parsed[["game_id", "parse_error"]], on="game_id", how="left")
    big = (g.white_rating_diff.abs() > PROVISIONAL_CHANGE) | (g.black_rating_diff.abs() > PROVISIONAL_CHANGE)
    steps = [
        ("no bot accounts", ~((g.white_title == "BOT") | (g.black_title == "BOT"))),
        ("not abandoned / rules infraction", ~g.termination.isin(BAD_TERMINATIONS)),
        (f"at least {MIN_PLIES} plies", g.plies >= MIN_PLIES),
        ("no berserk", ~(g.white_berserk | g.black_berserk)),
        ("both ratings present", g.white_elo.notna() & g.black_elo.notna()),
        (f"no provisional rating (|change| <= {PROVISIONAL_CHANGE})", ~big.fillna(False)),
        ("moves replayed without error", g.parse_error.fillna("missing") == ""),
    ]
    flow = [("parsed", len(g))]
    keep = pd.Series(True, index=g.index)
    for label, mask in steps:
        keep &= mask.fillna(False).astype(bool)
        flow.append((label, int(keep.sum())))
    g = g[keep].copy()
    g["mean_rating"] = (g.white_elo + g.black_elo) / 2
    g["band"] = pd.cut(g.mean_rating, BAND_EDGES, labels=BAND_LABELS, right=False)
    return g, flow


def white_edge(g: pd.DataFrame) -> float:
    """White's edge in Elo at equal ratings: the shift that makes the mean
    rating-expected score match White's mean score."""
    d = (g.white_elo - g.black_elo).to_numpy(float)
    target = g.score_white.mean()
    f = lambda a: np.mean(1 / (1 + 10 ** (-(d + a) / 400))) - target  # noqa: E731
    return float(optimize.brentq(f, -400, 400))


def focal_frame(ev: pd.DataFrame, g: pd.DataFrame, edge: float) -> pd.DataFrame:
    """Join events to games and add the focal side's score and the score
    expected from ratings and colour."""
    x = ev.merge(g[["game_id", "white_elo", "black_elo", "score_white", "band", "category"]],
                 on="game_id", how="inner")
    white = x.focal == "white"
    x["score"] = np.where(white, x.score_white, 1 - x.score_white)
    diff = np.where(white, x.white_elo - x.black_elo + edge, x.black_elo - x.white_elo - edge)
    x["expected"] = 1 / (1 + 10 ** (-diff / 400))
    x["residual"] = x.score - x.expected
    x["focal_rating_gap"] = np.where(white, x.white_elo - x.black_elo, x.black_elo - x.white_elo)
    return x


def residual_row(x: pd.DataFrame, label: str) -> dict:
    n = len(x)
    if n == 0:
        return {"group": label, "n": 0}
    se = x.residual.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return {"group": label, "n": n, "score": x.score.mean(), "expected": x.expected.mean(),
            "residual_pp": 100 * x.residual.mean(), "residual_lo": 100 * (x.residual.mean() - 1.96 * se),
            "residual_hi": 100 * (x.residual.mean() + 1.96 * se),
            "elo_approx": 695 * x.residual.mean(), "focal_rating_gap": x.focal_rating_gap.mean(),
            "draw_rate": (x.score == 0.5).mean()}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

K_ELO = 400 / np.log(10)


def lichess_winp(cp: np.ndarray) -> np.ndarray:
    """Lichess's expected score from a centipawn evaluation (clipped at +/-10 pawns)."""
    return 1 / (1 + np.exp(-0.00368208 * np.clip(cp, -1000, 1000)))


def frac_logit(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fractional logit (score in [0, 1]) with sandwich standard errors."""
    def nll(b: np.ndarray) -> float:
        p = np.clip(1 / (1 + np.exp(-(X @ b))), 1e-9, 1 - 1e-9)
        return -float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))
    b = optimize.minimize(nll, np.zeros(X.shape[1]), method="BFGS").x
    p = 1 / (1 + np.exp(-(X @ b)))
    H = (X * (p * (1 - p))[:, None]).T @ X
    u = X * (y - p)[:, None]
    Hi = np.linalg.inv(H)
    return b, np.sqrt(np.diag(Hi @ (u.T @ u) @ Hi))


def adjusted_effect(x: pd.DataFrame, edge: float, label: str, eval_col: str = "eval_cp") -> dict:
    """The focal side's edge in Elo beyond (1) the two ratings and White's edge,
    and (2) also the Lichess evaluation in `eval_col` (games that have one)."""
    gap = (x.focal_rating_gap + np.where(x.focal == "white", edge, -edge)).to_numpy(float) / K_ELO
    row: dict = {"group": label, "n": len(x)}
    if len(x) >= 20:
        b, se = frac_logit(x.score.to_numpy(float), np.column_stack([np.ones(len(x)), gap]))
        row.update(elo_rating_adj=b[0] * K_ELO, elo_rating_adj_se=se[0] * K_ELO)
    ev = np.where(x.focal == "white", x[eval_col], -x[eval_col])
    m = ~np.isnan(ev.astype(float))
    row["n_eval"] = int(m.sum())
    if m.sum() >= 20:
        w = lichess_winp(ev[m].astype(float))
        X = np.column_stack([np.ones(m.sum()), gap[m], np.log(w / (1 - w))])
        b, se = frac_logit(x.score.to_numpy(float)[m], X)
        row.update(elo_eval_adj=b[0] * K_ELO, elo_eval_adj_se=se[0] * K_ELO,
                   eval_coef=b[2], median_eval_cp=float(np.median(ev[m])))
    return row


# ---------------------------------------------------------------------------
# Survival
# ---------------------------------------------------------------------------

def kaplan_meier(durations: np.ndarray, events: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Survival S(t) on `grid` (plies)."""
    order = np.argsort(durations)
    d, e = durations[order], events[order]
    times = np.unique(d[e])
    s, surv = 1.0, {}
    n = len(d)
    idx = 0
    at_risk = n
    for t in times:
        while idx < n and d[idx] < t:
            at_risk -= 1
            idx += 1
        deaths = np.sum((d == t) & e)
        s *= 1 - deaths / at_risk
        surv[t] = s
    out, cur, ti = [], 1.0, 0
    keys = sorted(surv)
    for t in grid:
        while ti < len(keys) and keys[ti] <= t:
            cur = surv[keys[ti]]
            ti += 1
        out.append(cur)
    return np.array(out)


def survival_tables(pieces: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    p = pieces.copy()
    p["duration"] = np.where(p.survived, p.plies, p.captured_ply).astype(float)
    p["event"] = ~p.survived
    grid = np.arange(0, 201)
    curves = []
    groups = [("type", "B", p.piece == "B"), ("type", "N", p.piece == "N")]
    for o in sorted(p.origin.unique()):
        groups.append(("origin", o, p.origin == o))
    for bc in ("light", "dark"):
        groups.append(("bishop colour", bc, p.bishop_colour == bc))
    for kind, label, m in groups:
        s = kaplan_meier(p.duration[m].to_numpy(), p.event[m].to_numpy(), grid)
        curves.append(pd.DataFrame({"grouping": kind, "group": label, "ply": grid, "survival": s}))
    curves = pd.concat(curves, ignore_index=True)
    at = curves[curves.ply.isin([20, 40, 60, 80])].pivot_table(
        index=["grouping", "group"], columns="ply", values="survival").reset_index()
    at.columns = [f"surv_ply{c}" if isinstance(c, (int, np.integer)) else c for c in at.columns]
    return curves, at


def fig_survival(curves: pd.DataFrame, n_games: int, out: Path) -> list[Path]:
    plt = apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for t in ("B", "N"):
        c = curves[(curves.grouping == "type") & (curves.group == t)]
        axes[0].plot(c.ply / 2, c.survival, color=COLOURS[t], label={"B": "Bishops", "N": "Knights"}[t])
    styles = {"c1": "-", "f1": "--", "c8": "-", "f8": "--", "b1": "-", "g1": "--", "b8": "-", "g8": "--"}
    for o, ls in styles.items():
        c = curves[(curves.grouping == "origin") & (curves.group == o)]
        if len(c):
            col = COLOURS["B"] if o[0] in "cf" else COLOURS["N"]
            axes[1].plot(c.ply / 2, c.survival, color=col, ls=ls, lw=1, alpha=0.9 if o[1] == "1" else 0.5,
                         label=o)
    for ax, title in zip(axes, ("By piece", "By starting square")):
        ax.set_xlabel("Move")
        ax.set_title(title)
        ax.set_xlim(0, 80)
    axes[0].set_ylabel("Still on the board")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False, ncol=2, fontsize=8)
    fig.text(0.99, 0.01, f"{n_games:,} games", ha="right", color=MUTED_TEXT, fontsize=8)
    fig.tight_layout()
    paths = []
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out / f"human_pilot_survival.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def fig_outcomes(rows: pd.DataFrame, out: Path) -> list[Path]:
    plt = apply_style()
    r = rows[rows.n >= 30].iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(r) + 1.2))
    y = np.arange(len(r))
    ax.errorbar(r.residual_pp, y, xerr=[r.residual_pp - r.residual_lo, r.residual_hi - r.residual_pp],
                fmt="o", color=COLOURS["B"], ms=4, capsize=2)
    ax.axvline(0, color=MUTED_TEXT, lw=0.8)
    ax.set_yticks(y, [f"{g} (n={n:,})" for g, n in zip(r.group, r.n)])
    ax.set_xlabel("Score above rating expectation (percentage points)")
    fig.tight_layout()
    paths = []
    for ext in ("pdf", "png"):
        p = out / f"human_pilot_outcomes.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pct(x: float) -> str:
    return "-" if pd.isna(x) else f"{100 * x:.1f}%"


def run(out_dir: Path, tag: str | None, fig_dir: Path) -> str:
    games, ev = load(out_dir, tag)
    g, flow = include(games, ev["games"])
    ids = set(g.game_id)
    edge = white_edge(g)
    L = [f"Human-play pilot{' (' + tag + ')' if tag else ''}", "=" * 40, "",
         "1. DATA", "Inclusion flow: " + " -> ".join(f"{k} {v:,}" for k, v in flow),
         "Games by rating band: " + ", ".join(f"{k} {v:,}" for k, v in g.band.value_counts().sort_index().items() if v),
         "Categories: " + ", ".join(f"{k} {v:,}" for k, v in g.category.value_counts().items()),
         f"Results: White {pct((g.score_white == 1).mean())}, draws {pct((g.score_white == 0.5).mean())}, "
         f"Black {pct((g.score_white == 0).mean())}; White's edge at equal ratings {edge:+.0f} Elo",
         f"Clocks {pct(g.has_clock.mean())}; Lichess evaluations {pct(g.has_eval.mean())}; "
         f"tournament games {pct(g.tournament.mean())}",
         "Terminations: " + ", ".join(f"{k} {v:,}" for k, v in g.termination.value_counts().items())]

    tables: dict[str, pd.DataFrame] = {}

    # 2. Survival
    pcs = ev["pieces"][ev["pieces"].game_id.isin(ids)]
    curves, at = survival_tables(pcs)
    tables["survival_at"] = at
    first = pcs[~pcs.survived].sort_values(["game_id", "captured_ply"]).groupby("game_id").head(1)
    by = pcs[~pcs.survived].groupby("piece").captured_by.value_counts(normalize=True).unstack().fillna(0)
    L += ["", "2. SURVIVAL OF THE ORIGINAL MINOR PIECES",
          "Share still on the board (Kaplan-Meier) after 10 / 20 / 30 / 40 moves:"]
    for r in at.itertuples():
        L.append(f"  {r.grouping:13s} {r.group:6s} " + " / ".join(
            pct(getattr(r, f"surv_ply{k}", np.nan)) for k in (20, 40, 60, 80)))
    L.append(f"First minor piece captured in a game: bishop {pct((first.piece == 'B').mean())}, "
             f"knight {pct((first.piece == 'N').mean())} (of {len(first):,} games with a minor captured)")
    L.append("Captured by (share of each type's captures): ")
    L += ["  " + line for line in by.mul(100).round(1).to_string().splitlines()]

    # 3. Exchanges
    ex = ev["exchanges"][ev["exchanges"].game_id.isin(ids)].merge(
        g[["game_id", "white_elo", "black_elo", "band"]], on="game_id")
    bn = ex[ex.kind == "B for N"].copy()
    L += ["", "3. LEVEL MINOR-FOR-MINOR EXCHANGES",
          "Per 100 games: " + ", ".join(f"{k} {100 * v / len(g):.1f}" for k, v in ex.kind.value_counts().items())]
    if len(bn):
        bn["phase"] = pd.cut(bn.ply, PHASES, labels=PHASE_LABELS, right=True)
        keeper_elo = np.where(bn.bishop_giver == "white", bn.black_elo, bn.white_elo)
        giver_elo = np.where(bn.bishop_giver == "white", bn.white_elo, bn.black_elo)
        bn["keeper_stronger"] = keeper_elo > giver_elo
        L.append(f"Bishop-for-knight exchanges: {len(bn):,} in {bn.game_id.nunique():,} games. The player who "
                 f"started it gave the bishop (BxN) in {pct(bn.initiator_gave_bishop.mean())}, "
                 f"the knight (NxB) in {pct(1 - bn.initiator_gave_bishop.mean())}; "
                 f"broke a bishop pair {pct(bn.pair_broken.mean())}; "
                 f"the higher-rated player kept the bishop {pct(bn.keeper_stronger[keeper_elo != giver_elo].mean())}")
        ph = bn.groupby("phase", observed=False).agg(n=("ply", "size"), started_with_BxN=("initiator_gave_bishop", "mean"),
                                                     broke_pair=("pair_broken", "mean"))
        L += ["By when it happened:"] + ["  " + s for s in ph.round(3).to_string().splitlines()]
        bd = bn.groupby("band", observed=True).agg(n=("ply", "size"), started_with_BxN=("initiator_gave_bishop", "mean"),
                                                   keeper_stronger=("keeper_stronger", "mean"))
        L += ["By rating band:"] + ["  " + s for s in bd.round(3).to_string().splitlines()]
        tables["exchanges_by_phase"] = ph.reset_index()

    # 4. Opportunities, as episodes: consecutive turns of the same player with
    # the capture available (a pin can offer BxN for many moves in a row).
    op = ev["opportunities"][ev["opportunities"].game_id.isin(ids)].copy()
    if len(op):
        op = op.sort_values(["game_id", "side", "ply"])
        rows = []
        L += ["", "4. EXCHANGE OPPORTUNITIES (a defended minor can be taken by the other type)",
              "An episode = consecutive turns of one player with that capture available; taken = played it."]
        for kind, col in (("BxN", "bxn_available"), ("NxB", "nxb_available")):
            o = op[op[col]].copy()
            o["episode"] = (o.groupby(["game_id", "side"]).ply.diff() != 2).cumsum()
            e = o.groupby("episode").agg(taken=("played", lambda v: (v == kind).any()),
                                          turns=("ply", "size"), start=("ply", "first"))
            e["phase"] = pd.cut(e.start, PHASES, labels=PHASE_LABELS, right=True)
            L.append(f"{kind}: {len(e):,} episodes, taken in {pct(e.taken.mean())} "
                     f"(median {e.turns.median():.0f} turn(s) available)")
            t = e.groupby("phase", observed=False).agg(episodes=("taken", "size"), taken=("taken", "mean"))
            t.insert(0, "capture", kind)
            rows.append(t.reset_index())
        t = pd.concat(rows, ignore_index=True)
        tables["opportunities_by_phase"] = t
        L += ["By when the episode began:"] + ["  " + s for s in t.round(3).to_string(index=False).splitlines()]

    # 5. Outcomes of imbalances
    sp = ev["spells"][ev["spells"].game_id.isin(ids)]
    out_rows = []
    if len(sp):
        sp = sp.sort_values("start_ply").groupby(["game_id", "kind", "design"]).head(1)
        x = focal_frame(sp, g, edge)
        pawn = x[x.kind == "pawn up"]
        pu = residual_row(pawn, "Clean pawn up")
        out_rows.append(pu)
        mi = x[x.kind == "minor imbalance"]
        for d, sub in mi.groupby("design"):
            if len(sub) >= 30:
                out_rows.append(residual_row(sub, d))
                out_rows.append(residual_row(sub[sub.pawn_diff == 0], f"{d}, pawns level"))
        o = pd.DataFrame(out_rows)
        if len(o) and pu.get("n", 0):
            o["pawns_equiv"] = o.residual_pp / pu["residual_pp"]
        tables["outcomes"] = o
        L += ["", "5. OUTCOMES OF IMBALANCES (first spell of each kind per game; side with more bishops, or the pawn)",
              "Residual = score minus the score expected from both ratings and White's edge; Elo ~ 695 x residual."]
        if len(o):
            L += ["  " + s for s in o.round(3).to_string(index=False).splitlines()]
        # Adjusted models: ratings + White's edge, then also the Lichess
        # evaluation at the start of the spell (which already prices in the
        # engine's view of the pieces, so that column is the edge beyond it).
        adj = [adjusted_effect(pawn, edge, "Clean pawn up")]
        adj += [adjusted_effect(sub[sub.pawn_diff == 0], edge, f"{d}, pawns level")
                for d, sub in mi.groupby("design") if (sub.pawn_diff == 0).sum() >= 30]
        a = pd.DataFrame(adj)
        tables["outcomes_adjusted"] = a
        L += ["Fractional-logit edge of the focal side in Elo: adjusted for ratings and White's edge; and also "
              "for the Lichess evaluation at the start (the edge beyond what the engine already sees):"]
        L += ["  " + s for s in a.round(1).to_string(index=False).splitlines()]

    # 5b. Payoff of the bishop-for-knight decision: the first B-for-N exchange
    # in each game, scored for the player who kept the bishop, adjusted for the
    # evaluation just before the exchange began.
    exch = ev["exchanges"][ev["exchanges"].game_id.isin(ids)]
    if len(exch):
        first = exch.sort_values("ply").groupby("game_id").head(1)
        bn = first[first.kind == "B for N"].assign(
            focal=lambda d: np.where(d.bishop_giver == "white", "black", "white"))
        same = first[first.kind != "B for N"].assign(focal=lambda d: d.initiator)
        rows = []
        for label, d in (("Kept the bishop (first B-for-N exchange)", bn),
                         ("  ... and so the pair", bn[bn.pair_broken == True]),  # noqa: E712
                         ("  ... no pair involved", bn[bn.pair_broken == False]),  # noqa: E712
                         ("  ... in moves 1-10", bn[bn.ply <= 20]),
                         ("  ... after move 10", bn[bn.ply > 20]),
                         ("Control: started a same-type swap", same)):
            if len(d):
                x = focal_frame(d, g, edge)
                rows.append({**residual_row(x, label), **adjusted_effect(x, edge, label)})
        t = pd.DataFrame(rows)
        tables["exchange_payoff"] = t
        L += ["", "5b. PAYOFF OF THE BISHOP-FOR-KNIGHT DECISION (first minor exchange per game)",
              "elo_eval_adj = edge beyond ratings and the evaluation just BEFORE the exchange: the total effect "
              "of ending up with the bishop."]
        L += ["  " + s for s in t[["group", "n", "residual_pp", "elo_rating_adj", "elo_rating_adj_se", "n_eval",
                                    "elo_eval_adj", "elo_eval_adj_se", "median_eval_cp"]].round(1)
              .to_string(index=False).splitlines()]

    # 6. Endgames
    eg = ev["endgames"][ev["endgames"].game_id.isin(ids)]
    if len(eg):
        x = focal_frame(eg, g, edge)
        L += ["", "6. MINOR-PIECE ENDGAMES REACHED",
              "Per 100 games: " + ", ".join(f"{k} {100 * v / len(g):.1f}" for k, v in eg.type.value_counts().items())]
        bvn = x[x.type == "B v N"]
        rows = [residual_row(bvn, "Endgame " + "B v N, all"),
                residual_row(bvn[bvn.pawn_diff == 0], "Endgame " + "B v N, pawns level"),
                residual_row(bvn[(bvn.pawn_diff == 0) & bvn.pawns_both_wings], "Endgame " + "B v N, level, pawns on both wings"),
                residual_row(bvn[(bvn.pawn_diff == 0) & ~bvn.pawns_both_wings], "Endgame " + "B v N, level, one wing"),
                residual_row(bvn[(bvn.pawn_diff == 0) & (bvn.rooks_each == 1)], "Endgame " + "B v N, level, with rooks"),
                residual_row(bvn[(bvn.pawn_diff == 0) & (bvn.rooks_each == 0)], "Endgame " + "B v N, level, no rooks")]
        e = pd.DataFrame(rows)
        tables["endgames"] = e
        L += ["Bishop side's residual in B v N endgames:"] + ["  " + s for s in e.round(3).to_string(index=False).splitlines()]
        out_rows += [r for r in rows if r.get("n", 0)]

    tables_dir = out_dir / "analysis"
    tables_dir.mkdir(parents=True, exist_ok=True)
    sfx = f"_{tag}" if tag else ""
    for k, v in tables.items():
        v.to_csv(tables_dir / f"human_pilot_{k}{sfx}.csv", index=False, float_format="%.6g")
    curves.to_csv(tables_dir / f"human_pilot_survival_curves{sfx}.csv", index=False, float_format="%.6g")
    fig_survival(curves, len(g), fig_dir)
    if out_rows:
        fig_outcomes(pd.DataFrame(out_rows), fig_dir)
    text = "\n".join(L) + "\n"
    (tables_dir / f"human_pilot_summary{sfx}.txt").write_text(text)
    return text


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-dir", default=str(OUT_DIR))
    p.add_argument("--figures-dir", default=str(FIG_DIR))
    p.add_argument("--tag")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    setup_logging(args.verbose)
    logging.getLogger("fontTools").setLevel(logging.WARNING)   # quiet PDF font subsetting
    print(run(Path(args.output_dir), args.tag, Path(args.figures_dir)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
