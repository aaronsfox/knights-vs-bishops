"""Tests for the shared statistics (stats_core.py)."""

import numpy as np
import pandas as pd
import pytest

import stats_core as sc


def chess960_icc(games: pd.DataFrame, m: int) -> float:
    """The Chess960 estimator (equal games per line), for comparison."""
    g = games.copy()
    g["res"] = g.y - g.groupby("pos").y.transform("mean")
    lm = g.groupby(["pos", "line"]).res.agg(["mean", "var"])
    n_pos = g.pos.nunique()
    msb = m * (lm["mean"] ** 2).sum() / (len(lm) - n_pos)
    msw = lm["var"].mean()
    return float((msb - msw) / (msb + (m - 1) * msw))


def clustered(rng, n_pos=20, n_lines=50, m=10, sd_line=0.2):
    rows = []
    for pos in range(n_pos):
        base = rng.uniform(0.3, 0.7)
        for line in range(n_lines):
            p = np.clip(base + rng.normal(0, sd_line), 0.01, 0.99)
            for _ in range(m):
                rows.append({"pos": pos, "line": line, "y": float(rng.random() < p)})
    return pd.DataFrame(rows)


def test_icc_matches_chess960_with_equal_lines():
    g = clustered(np.random.default_rng(1))
    assert sc.icc_one_way(g, "y", ["line"], ["pos"]) == pytest.approx(chess960_icc(g, 10))


def test_icc_with_unequal_lines_is_close():
    rng = np.random.default_rng(2)
    g = clustered(rng)
    drop = g.groupby(["pos", "line"]).cumcount() >= np.where(g.line % 2 == 0, 4, 10)
    icc_full = sc.icc_one_way(g, "y", ["line"], ["pos"])
    icc_part = sc.icc_one_way(g[~drop], "y", ["line"], ["pos"])
    assert 0.05 < icc_full < 0.3
    assert icc_part == pytest.approx(icc_full, abs=0.05)


def test_design_effect():
    assert sc.kish_cluster_size([10] * 5) == 10
    assert sc.kish_cluster_size([2, 6]) == pytest.approx(40 / 8)
    assert sc.design_effect(0.07, [10] * 50) == pytest.approx(1.63)
    assert sc.design_effect(float("nan"), [10]) == 1.0
    assert sc.design_effect(-0.1, [10]) == 1.0          # never below 1


def test_posterior_summary():
    p = sc.posterior_draws(60, 30, 10, 1.0, sc.rng_for("x"))
    s = sc.summarise_draws(p)
    assert s["escore"] == pytest.approx(0.75, abs=0.01)
    assert s["delta_elo_lo"] < s["delta_elo"] < s["delta_elo_hi"]
    assert s["p_positive"] > 0.99
    wide = sc.summarise_draws(sc.posterior_draws(60, 30, 10, 3.0, sc.rng_for("x")))
    assert wide["escore_sd"] > s["escore_sd"]          # DEFF widens the posterior


def test_rng_is_reproducible():
    assert sc.rng_for("a", 1).random() == sc.rng_for("a", 1).random()
    assert sc.rng_for("a", 1).random() != sc.rng_for("a", 2).random()
