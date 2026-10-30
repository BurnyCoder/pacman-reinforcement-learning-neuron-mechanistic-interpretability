"""Planted-neuron tests: the audit must recover known single-feature units and reject mixed ones."""

import numpy as np

from pacneuron.audit import (BinModel, alive, clean_mask, nonneg_mixtures, quantile_band_examples, r2_ceiling,
                             r2_single, random_directions, top_examples)


def _synthetic(n=6000, games=20, seed=0):
    rng = np.random.default_rng(seed)
    F = {"dist": rng.uniform(0, 30, n).astype(np.float32), "flag": (rng.random(n) < 0.3).astype(np.float32),
         "heading": rng.integers(0, 4, n).astype(np.float32), "noise": rng.normal(size=n).astype(np.float32)}
    A = np.stack([
        np.maximum(0, 20 - F["dist"]) + rng.normal(0, 0.5, n),  # 0: "ghost near" unit (one job: dist)
        3 * F["flag"] + rng.normal(0, 0.3, n),  # 1: flag detector
        F["dist"] / 10 + 3 * F["flag"] + (F["heading"] == 2) * 2 + rng.normal(0, 0.3, n),  # 2: mixed
        rng.normal(0, 1, n),  # 3: pure noise
    ], 1).astype(np.float32)
    game = np.repeat(np.arange(games), n // games)
    return A, F, game


def test_planted_units_recovered():
    A, F, game = _synthetic()
    fit, ev = game % 2 == 0, game % 2 == 1  # split-half by game, as in the discovery audit
    sub = lambda d, m: {k: v[m] for k, v in d.items()}
    ge = game[ev] // 2  # re-index eval games 0..G-1
    s1, names = r2_single(A[fit], sub(F, fit), A[ev], sub(F, ev), ge, ge.max() + 1)
    sall = r2_ceiling(A[fit], sub(F, fit), A[ev], sub(F, ev), ge, ge.max() + 1, n_jobs=1)
    r1, rall = s1.r2(), sall.r2()
    assert names[r1[:, 0].argmax()] == "dist" and r1[:, 0].max() > 0.9
    assert names[r1[:, 1].argmax()] == "flag" and r1[:, 1].max() > 0.9
    assert rall[2] > 0.8 and r1[:, 2].max() < 0.7 * rall[2]  # mixed: explainable but not by one feature
    assert rall[3] < 0.05
    np.testing.assert_array_equal(clean_mask(r1, rall), [True, True, False, False])
    # bootstrap reweighting with all-ones weights reproduces the point estimate
    np.testing.assert_allclose(s1.r2(np.ones(ge.max() + 1)), r1)


def test_bins_and_controls():
    A, F, _ = _synthetic()
    bm = BinModel.fit(F)
    assert bm.discrete["heading"] and bm.discrete["flag"] and not bm.discrete["dist"]
    assert bm.index("dist", F["dist"]).max() < bm.n_bins("dist")
    assert bm.index("heading", np.array([2.4, 7.0])).tolist() == [2, 3]  # nearest fitted category
    Q = random_directions(8)
    np.testing.assert_allclose(Q.T @ Q, np.eye(8), atol=1e-5)
    W = nonneg_mixtures(np.arange(8), 8, n_out=5)
    assert (W >= 0).all() and np.allclose(W.sum(0), 1) and ((W > 0).sum(0) >= 2).all()
    assert alive(np.maximum(A, 0)).all()
    a = A[:, 0]
    assert a[top_examples(a, 5)].min() >= np.sort(a)[-5]
    bands = quantile_band_examples(a, np.random.default_rng(0))
    assert set(bands) == {"top", "1-5%", "10-50%", "bottom"}
