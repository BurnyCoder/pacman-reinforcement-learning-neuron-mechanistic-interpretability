"""Neuron audit: how well does each unit's activation follow one ground-truth feature? (Goodfire-style "one job")

Local: vectorised over all units at once.
  - `BinModel`: per-feature 20-quantile bins fitted on one set of steps.
  - `r2_single`: out-of-sample R² of "unit ~ bin means of feature f", for every (unit, feature), with
    per-game sufficient statistics so the R² can be recomputed under a game bootstrap.
  - `r2_ceiling`: the same R² for HistGradientBoosting on all features (how much of the unit is explainable at all).
  - `clean_mask`: unit is clean if R²_all >= 0.3 and best single-feature R² / R²_all >= 0.7 (preregistration §5).
  - control activation sets (random orthonormal directions, non-negative unit mixtures).
Global: phases.audit uses these on discovery split-halves (exploratory labelling, candidate search) and phases.h1
on discovery -> held-out (confirmatory H1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from sklearn.ensemble import HistGradientBoostingRegressor

from pacneuron.features import CATEGORICAL

N_BINS = 20
CLEAN_R2_ALL, CLEAN_RATIO = 0.3, 0.7
ALIVE_FRAC = 0.001


@dataclass
class BinModel:
    """Bins per feature fitted on the "fit" steps.

    Discrete features (categorical, or <= 20 distinct values) use their fitted values as bins (unseen values go to
    the nearest one); continuous features use up to 19 inner 5%-quantile edges (<= 20 bins).
    """

    edges: dict[str, np.ndarray]
    discrete: dict[str, bool]

    @classmethod
    def fit(cls, feats: dict[str, np.ndarray]) -> BinModel:
        edges, discrete = {}, {}
        for k, v in feats.items():
            u = np.unique(v)
            discrete[k] = k in CATEGORICAL or len(u) <= N_BINS
            edges[k] = u if discrete[k] else np.unique(np.quantile(v, np.linspace(0, 1, N_BINS + 1)[1:-1]))
        return cls(edges, discrete)

    def index(self, k: str, v: np.ndarray) -> np.ndarray:
        e = self.edges[k]
        if self.discrete[k]:
            i = np.searchsorted(e, v).clip(0, len(e) - 1)
            left = (i - 1).clip(0)
            return np.where(np.abs(e[left] - v) < np.abs(e[i] - v), left, i)  # nearest fitted value
        return np.searchsorted(e, v, side="right")

    def n_bins(self, k: str) -> int:
        return len(self.edges[k]) + (0 if self.discrete[k] else 1)


def _onehot(idx: np.ndarray, n: int) -> csr_matrix:
    """[n, N] sparse indicator: row b has ones at the steps in bin b (fast group sums via matrix products)."""
    return csr_matrix((np.ones(len(idx), np.float32), (idx, np.arange(len(idx)))), shape=(n, len(idx)))


@dataclass
class R2Stats:
    """Per-game sufficient statistics for out-of-sample R², so R² can be recomputed for any game reweighting.

    sse[..., G, U]: summed squared prediction errors per game; sy, syy [G, U]: sums of y, y²; n [G]: steps.
    R² = 1 - SSE / (Σy² - (Σy)²/n) over the (re)weighted games.
    """

    sse: np.ndarray
    sy: np.ndarray
    syy: np.ndarray
    n: np.ndarray

    def r2(self, w: np.ndarray | None = None) -> np.ndarray:
        """R² for game weights w [G] (None = all games once). Returns shape sse.shape without the G axis."""
        w = np.ones(len(self.n)) if w is None else w
        sse = np.tensordot(w, self.sse, axes=([0], [-2]))
        sy, syy, n = w @ self.sy, w @ self.syy, w @ self.n
        sst = syy - sy ** 2 / n
        return 1.0 - sse / np.maximum(sst, 1e-12)


def _game_sums(game: np.ndarray, n_games: int, x: np.ndarray) -> np.ndarray:
    return np.asarray(_onehot(game, n_games) @ x)


def r2_single(A_fit, F_fit, A_ev, F_ev, game_ev, n_games) -> tuple[R2Stats, list[str]]:
    """Out-of-sample R² of each unit against bin means of each single feature. sse shape [F, G, U]."""
    names = list(F_fit)
    bm = BinModel.fit(F_fit)
    sse = np.zeros((len(names), n_games, A_ev.shape[1]), np.float64)
    gmean = A_fit.mean(0)
    for i, k in enumerate(names):
        nb = bm.n_bins(k)
        bf = bm.index(k, F_fit[k])
        cnt = np.bincount(bf, minlength=nb).astype(np.float64)
        sums = np.asarray(_onehot(bf, nb) @ A_fit, np.float64)
        means = np.where(cnt[:, None] > 0, sums / np.maximum(cnt, 1)[:, None], gmean)  # empty bin -> global mean
        resid = A_ev - means[bm.index(k, F_ev[k])]
        sse[i] = _game_sums(game_ev, n_games, (resid ** 2).astype(np.float32))
    sy = _game_sums(game_ev, n_games, A_ev)
    syy = _game_sums(game_ev, n_games, (A_ev ** 2).astype(np.float32))
    n = np.bincount(game_ev, minlength=n_games).astype(np.float64)
    return R2Stats(sse, sy, syy, n), names


def _fit_one(X_fit, y_fit, X_ev):
    # Histogram gradient boosting (scikit-learn's LightGBM-style estimator, Ke et al. 2017,
    # https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree); defaults
    # include early stopping on a 10% validation split above 10,000 samples
    # (https://scikit-learn.org/stable/modules/ensemble.html#histogram-based-gradient-boosting).
    m = HistGradientBoostingRegressor(max_iter=200, random_state=0)
    return m.fit(X_fit, y_fit).predict(X_ev).astype(np.float32)


def r2_ceiling(A_fit, F_fit, A_ev, F_ev, game_ev, n_games, max_fit: int = 50000, n_jobs: int = 16,
               seed: int = 0) -> R2Stats:
    """Out-of-sample R² of HistGradientBoosting(all features) per unit (preregistered ceiling). sse [G, U]."""
    names = list(F_fit)
    X_fit = np.stack([F_fit[k] for k in names], 1)
    X_ev = np.stack([F_ev[k] for k in names], 1)
    rng = np.random.default_rng(seed)
    sub = rng.choice(len(X_fit), size=min(max_fit, len(X_fit)), replace=False)
    preds = Parallel(n_jobs=n_jobs)(delayed(_fit_one)(X_fit[sub], A_fit[sub, u], X_ev) for u in range(A_fit.shape[1]))
    resid = A_ev - np.stack(preds, 1)
    sse = _game_sums(game_ev, n_games, (resid ** 2).astype(np.float32))
    return R2Stats(sse, _game_sums(game_ev, n_games, A_ev), _game_sums(game_ev, n_games, (A_ev ** 2).astype(np.float32)),
                   np.bincount(game_ev, minlength=n_games).astype(np.float64))


def clean_mask(r2_single_fu: np.ndarray, r2_all_u: np.ndarray, thr_all: float = CLEAN_R2_ALL,
               thr_ratio: float = CLEAN_RATIO) -> np.ndarray:
    """[U] bool: explainable at all (R²_all >= thr_all) and mostly by one feature (best R²₁ / R²_all >= thr_ratio)."""
    best = r2_single_fu.max(0)
    return (r2_all_u >= thr_all) & (best >= thr_ratio * r2_all_u)


def alive(A: np.ndarray, frac: float = ALIVE_FRAC) -> np.ndarray:
    """[U] bool: post-ReLU activation > 0 on at least `frac` of steps."""
    return (A > 0).mean(0) >= frac


def random_directions(n_units: int, seed: int = 0) -> np.ndarray:
    """[U, U] random orthonormal basis (QR of a Gaussian matrix, sign-fixed): control (a) projections.

    Fixing the signs by diag(R) makes Q Haar-distributed (Mezzadri 2007, https://arxiv.org/abs/math-ph/0609050).
    """
    q, r = np.linalg.qr(np.random.default_rng(seed).normal(size=(n_units, n_units)))
    return (q * np.sign(np.diag(r))).astype(np.float32)


def nonneg_mixtures(alive_idx: np.ndarray, n_units: int, n_out: int = 512, seed: int = 0) -> np.ndarray:
    """[U, n_out] mixing matrix: each output mixes 2-8 random alive units with Dirichlet(1) weights (control b)."""
    rng = np.random.default_rng(seed)
    W = np.zeros((n_units, n_out), np.float32)
    for j in range(n_out):
        k = rng.integers(2, 9)
        W[rng.choice(alive_idx, size=k, replace=False), j] = rng.dirichlet(np.ones(k))
    return W


def top_examples(a: np.ndarray, k: int = 16) -> np.ndarray:
    """Indices of the k largest values (descending)."""
    idx = np.argpartition(-a, min(k, len(a) - 1))[:k]
    return idx[np.argsort(-a[idx])]


def quantile_band_examples(a: np.ndarray, rng: np.random.Generator, per_band: int = 4) -> dict[str, list[int]]:
    """Goodfire-style bands of the activation distribution: top, 1-5%, 10-50%, bottom (indices into a)."""
    order = np.argsort(-a)
    n = len(a)
    bands = {"top": order[:per_band], "1-5%": order[int(0.01 * n):int(0.05 * n)],
             "10-50%": order[int(0.10 * n):int(0.50 * n)], "bottom": order[-per_band:]}
    return {b: [int(i) for i in (ix if len(ix) <= per_band else rng.choice(ix, per_band, replace=False))]
            for b, ix in bands.items()}
