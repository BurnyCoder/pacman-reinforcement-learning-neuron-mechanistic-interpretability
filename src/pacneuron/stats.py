"""Statistics: game-cluster bootstrap, paired sign-flip permutation test, Holm and Benjamini-Hochberg corrections,
and the intraclass correlation of games that share a start state.

Local: small numpy functions with no project dependencies (unit-tested on synthetic data in tests/test_stats.py).
Global: steps within one game are strongly autocorrelated, so every uncertainty estimate resamples whole games
(cluster bootstrap); paired steering comparisons use per-game differences between arms with the same game seed.
References: Holm (1979) step-down FWER control (https://www.jstor.org/stable/4615733); Benjamini & Hochberg (1995)
FDR (https://doi.org/10.1111/j.2517-6161.1995.tb02031.x); Efron & Tibshirani (1993) percentile bootstrap; resampling
whole games = cluster bootstrap (Field & Welsh 2007, https://doi.org/10.1111/j.1467-9868.2007.00593.x); sign-flip
permutation tests for paired designs (Good, "Permutation Tests", 2005).
"""

from __future__ import annotations

import numpy as np


def bootstrap_weights(n_games: int, n_boot: int, seed: int = 0) -> np.ndarray:
    """[n_boot, n_games] multinomial resampling counts: resample games with replacement."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_games, np.full(n_games, 1.0 / n_games), size=n_boot).astype(np.float64)


def mean_ci(per_game: np.ndarray, n_boot: int = 10000, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """Mean over games and its percentile game-bootstrap CI."""
    x = np.asarray(per_game, float)
    w = bootstrap_weights(len(x), n_boot, seed)
    boots = w @ x / len(x)
    lo, hi = np.quantile(boots, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_signflip_p(diffs: np.ndarray, sign: int = 1, n_perm: int = 10000, seed: int = 0) -> float:
    """One-sided p-value that the mean paired difference is in direction `sign` (+1 or -1).

    Under H0 (no effect) each game's difference is symmetric around 0, so random sign flips give the null
    distribution of the mean. p = (1 + #{null >= observed}) / (1 + n_perm), in the predicted direction (the +1 keeps
    random-permutation p-values valid; Phipson & Smyth 2010, https://arxiv.org/abs/1603.05766). Its floor is
    1 / (n_perm + 1), i.e. 0.0001 for 10,000 permutations.
    """
    d = sign * np.asarray(diffs, float)
    rng = np.random.default_rng(seed)
    flips = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (flips * d).mean(1)
    return float((1 + np.sum(null >= d.mean())) / (1 + n_perm))


def holm(pvals) -> np.ndarray:
    """Holm step-down adjusted p-values (FWER)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def bh(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (FDR)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        running = min(running, p[idx] * m / (rank + 1))
        adj[idx] = running
    return adj


def icc_oneway(values: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    """(ICC, design effect) of per-game values clustered in groups (e.g. games sharing a start state).

    One-way ANOVA estimator ICC(1) = (MSB - MSW) / (MSB + (n0 - 1) MSW) (Shrout & Fleiss 1979,
    https://doi.org/10.1037/0033-2909.86.2.420), with the unbalanced-design group size n0 = (N - Σn_i²/N) / (k - 1)
    (Snedecor & Cochran), clipped at 0. Design effect 1 + (m - 1) ICC (Kish 1965, "Survey Sampling") with the
    size-weighted mean group size m = Σn_i²/N: the variance inflation of a mean over clustered games.
    """
    y, g = np.asarray(values, float), np.unique(np.asarray(groups), return_inverse=True)[1]
    n, k, N = np.bincount(g), g.max() + 1, len(y)
    mu = np.bincount(g, y) / n
    msb = (n * (mu - y.mean()) ** 2).sum() / (k - 1)
    msw = ((y - mu[g]) ** 2).sum() / (N - k)
    n0 = (N - (n ** 2).sum() / N) / (k - 1)
    icc = max(float((msb - msw) / (msb + (n0 - 1) * msw)), 0.0)
    return icc, 1 + ((n ** 2).sum() / N - 1) * icc


def paired_signflip_p_two_sided(diffs: np.ndarray, n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sided sign-flip p-value for mean paired difference != 0 (used for off-target metrics)."""
    d = np.asarray(diffs, float)
    rng = np.random.default_rng(seed)
    null = (rng.choice([-1.0, 1.0], size=(n_perm, len(d))) * d).mean(1)
    return float((1 + np.sum(np.abs(null) >= abs(d.mean()))) / (1 + n_perm))
