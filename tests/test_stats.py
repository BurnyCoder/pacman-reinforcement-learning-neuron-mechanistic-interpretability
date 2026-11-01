"""Statistics on synthetic data with known answers."""

import numpy as np

from pacneuron.stats import bh, holm, mean_ci, paired_signflip_p


def test_holm_and_bh_known_values():
    p = [0.01, 0.04, 0.03, 0.2]
    np.testing.assert_allclose(holm(p), [0.04, 0.09, 0.09, 0.2])
    np.testing.assert_allclose(bh(p), [0.04, 0.0533333, 0.0533333, 0.2], rtol=1e-5)


def test_signflip_detects_effect_and_controls_false_positives():
    rng = np.random.default_rng(1)
    assert paired_signflip_p(rng.normal(0.5, 1, 60), sign=1) < 0.01
    assert paired_signflip_p(rng.normal(0.5, 1, 60), sign=-1) > 0.9
    # under the null, p-values are roughly uniform: ~5% below 0.05
    ps = [paired_signflip_p(rng.normal(0, 1, 40), n_perm=999, seed=s) for s in range(200)]
    assert 0.01 < np.mean(np.array(ps) < 0.05) < 0.12


def test_bootstrap_ci_coverage():
    rng = np.random.default_rng(2)
    hits = 0
    for s in range(200):
        x = rng.normal(1.0, 2.0, 50)
        _, lo, hi = mean_ci(x, n_boot=2000, seed=s)
        hits += lo <= 1.0 <= hi
    assert 0.88 < hits / 200 < 0.99


def test_icc_oneway_known_structure():
    from pacneuron.stats import icc_oneway
    rng = np.random.default_rng(4)
    g = np.repeat(np.arange(4), [10, 12, 13, 15])
    icc, deff = icc_oneway(rng.normal(0, 1, len(g)), g)  # no group effect -> ICC ~ 0, design effect ~ 1
    assert icc < 0.15 and deff < 2.5
    icc, deff = icc_oneway(10.0 * g + rng.normal(0, 0.01, len(g)), g)  # pure group effect -> ICC ~ 1
    assert icc > 0.99
    np.testing.assert_allclose(deff, 1 + ((np.bincount(g) ** 2).sum() / len(g) - 1) * icc)


def test_two_sided_signflip():
    from pacneuron.stats import paired_signflip_p_two_sided
    rng = np.random.default_rng(3)
    assert paired_signflip_p_two_sided(rng.normal(-0.6, 1, 60)) < 0.01  # detects either direction
    ps = [paired_signflip_p_two_sided(rng.normal(0, 1, 40), n_perm=999, seed=s) for s in range(200)]
    assert 0.01 < np.mean(np.array(ps) < 0.05) < 0.12
