"""Statistics shared by the analyses (the Chess960 self-play framework).

Each set of games sharing a starting position is summarised by its win/draw/
loss counts, modelled as Dirichlet-multinomial: the posterior is
Dirichlet(1/2 + (W, D, L) / DEFF), a Jeffreys prior with counts deflated by
the design effect for correlation among games sharing an opening line
(Kish, 1965; Mosimann, 1962; Brier, 1980; Rao & Scott, 1981):

    DEFF = 1 + (m - 1) * ICC

with m the number of games per line. When lines have different numbers of
games (e.g. key cells topped up with a second batch), m is Kish's weighted
mean cluster size, sum(m_i^2) / sum(m_i). The ICC is a pooled one-way ANOVA
of games within lines, within positions. Every derived quantity comes from
Monte Carlo draws of the posteriors, seeded reproducibly per key (crc32 of
the key strings, since Python's hash() changes between processes). A Bayesian
bootstrap over lines is the sensitivity check.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
from scipy import optimize

PRIOR = 0.5            # Jeffreys prior, Dirichlet(1/2, 1/2, 1/2)
N_DRAWS = 20_000
N_BOOT = 2_000
SEED = 1999


def elo(e):
    """Elo-equivalent of an expected score: 400 log10(E / (1 - E))."""
    e = np.clip(e, 1e-9, 1 - 1e-9)
    return 400.0 * np.log10(e / (1.0 - e))


def rng_for(*keys) -> np.random.Generator:
    """Reproducible generator for a set of keys (ints or strings)."""
    return np.random.default_rng([SEED, *[k if isinstance(k, (int, np.integer))
                                          else zlib.crc32(str(k).encode()) for k in keys]])


def wdl(scores: pd.Series) -> tuple[int, int, int]:
    """(wins, draws, losses) from a series of scores (1, 0.5, 0)."""
    return int((scores == 1).sum()), int((scores == 0.5).sum()), int((scores == 0).sum())


# ---------------------------------------------------------------------------
# Clustering: ICC and design effect
# ---------------------------------------------------------------------------

def icc_one_way(df: pd.DataFrame, value: str, cluster: list[str], stratum: list[str]) -> float:
    """Pooled one-way ANOVA intraclass correlation of ``value`` within
    clusters (book lines), within strata (positions), allowing unequal
    cluster sizes.

    Stratum means are removed first, so differences between positions don't
    count as clustering. With equal cluster sizes m this is the Chess960
    estimator: (MSB - MSW) / (MSB + (m - 1) MSW)."""
    d = df[[*stratum, *cluster, value]].dropna().copy()
    d["res"] = d[value] - d.groupby(stratum)[value].transform("mean")
    g = d.groupby(stratum + cluster)["res"]
    n_i = g.size().to_numpy(float)
    mean_i = g.mean().to_numpy(float)
    k = len(n_i)
    n_strata = d.groupby(stratum).ngroups
    N = n_i.sum()
    ss_between = (n_i * mean_i ** 2).sum()
    ss_within = ((d["res"] - g.transform("mean")) ** 2).sum()
    if k - n_strata <= 0 or N - k <= 0:
        return float("nan")
    msb = ss_between / (k - n_strata)
    msw = ss_within / (N - k)
    # Effective cluster size for unequal sizes, within strata.
    sizes = d.groupby(stratum + cluster).size()
    per_stratum = sizes.groupby(level=list(range(len(stratum)))).apply(
        lambda s: (s ** 2).sum() / s.sum()).sum()
    n0 = (N - per_stratum) / (k - n_strata)
    return float((msb - msw) / (msb + (n0 - 1) * msw))


def kish_cluster_size(sizes) -> float:
    """Weighted mean cluster size sum(m^2) / sum(m) (= m when all equal)."""
    s = np.asarray(sizes, dtype=float)
    return float((s ** 2).sum() / s.sum()) if s.sum() else float("nan")


def design_effect(icc: float, cluster_sizes) -> float:
    if not np.isfinite(icc):
        return 1.0
    return float(max(1.0, 1 + (kish_cluster_size(cluster_sizes) - 1) * icc))


# ---------------------------------------------------------------------------
# Posteriors
# ---------------------------------------------------------------------------

def posterior_draws(w: float, d: float, l: float, deff: float, rng: np.random.Generator,
                    n: int = N_DRAWS) -> np.ndarray:
    """Draws of (p_win, p_draw, p_loss) from Dirichlet(1/2 + (W, D, L)/DEFF)."""
    alpha = PRIOR + np.array([w, d, l], dtype=float) / deff
    return rng.dirichlet(alpha, n)


def interval(x: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))


def summarise_draws(p: np.ndarray, prefix: str = "") -> dict[str, float]:
    """E, Delta_Elo, decisive rate and the share of decisive games won, with
    95% intervals, from Dirichlet draws."""
    e = p[:, 0] + p[:, 1] / 2
    de = elo(e)
    dec = 1 - p[:, 1]
    share = p[:, 0] / (p[:, 0] + p[:, 2])
    out = {}
    for name, x, centre in (("escore", e, np.mean), ("delta_elo", de, np.median),
                            ("decisive", dec, np.mean), ("share_decisive_won", share, np.mean)):
        lo, hi = interval(x)
        out[f"{prefix}{name}"] = float(centre(x))
        out[f"{prefix}{name}_lo"] = lo
        out[f"{prefix}{name}_hi"] = hi
        out[f"{prefix}{name}_sd"] = float(x.std())
    out[f"{prefix}p_positive"] = float((de > 0).mean())
    return out


def bayesian_bootstrap_escore(line_scores: pd.DataFrame, rng: np.random.Generator,
                              n: int = N_BOOT) -> np.ndarray:
    """Bayesian bootstrap over lines: draws of the expected score, weighting
    each line by a Dirichlet(1, ..., 1) weight times its number of games.
    ``line_scores`` has columns ``games`` and ``points``."""
    games = line_scores["games"].to_numpy(float)
    points = line_scores["points"].to_numpy(float)
    w = rng.dirichlet(np.ones(len(games)), n) * games
    return (w @ (points / games)) / w.sum(axis=1)


def reliability(means: np.ndarray, sds: np.ndarray) -> float:
    """1 - mean posterior variance / variance of posterior means."""
    means, sds = np.asarray(means, float), np.asarray(sds, float)
    return float(1 - (sds ** 2).mean() / means.var(ddof=1)) if len(means) > 1 else float("nan")


# ---------------------------------------------------------------------------
# Regression helpers (scipy only, as in Chess960)
# ---------------------------------------------------------------------------

def weighted_logistic(x: np.ndarray, y: np.ndarray, n: np.ndarray, degree: int = 2):
    """Binomial logistic regression of proportion y on polynomial(x), with
    trials n (effective sample sizes). Returns coefficients, their SEs and a
    prediction function."""
    X = np.vander(x, degree + 1, increasing=True)

    def nll(beta):
        eta = X @ beta
        return -(n * (y * eta - np.logaddexp(0, eta))).sum()

    def grad(beta):
        p = 1 / (1 + np.exp(-(X @ beta)))
        return -X.T @ (n * (y - p))

    beta0 = np.zeros(X.shape[1])
    ybar = np.clip(y.mean(), 1e-6, 1 - 1e-6)
    beta0[0] = np.log(ybar / (1 - ybar))
    res = optimize.minimize(nll, beta0, jac=grad, method="BFGS")
    p = 1 / (1 + np.exp(-(X @ res.x)))
    cov = np.linalg.pinv((X * (n * p * (1 - p))[:, None]).T @ X)

    def predict(xn):
        return 1 / (1 + np.exp(-(np.vander(np.atleast_1d(xn), degree + 1, increasing=True) @ res.x)))

    return res.x, np.sqrt(np.diag(cov)), predict
