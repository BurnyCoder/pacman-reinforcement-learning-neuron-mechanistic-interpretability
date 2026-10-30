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


def test_two_sided_signflip():
    from pacneuron.stats import paired_signflip_p_two_sided
    rng = np.random.default_rng(3)
    assert paired_signflip_p_two_sided(rng.normal(-0.6, 1, 60)) < 0.01  # detects either direction
    ps = [paired_signflip_p_two_sided(rng.normal(0, 1, 40), n_perm=999, seed=s) for s in range(200)]
    assert 0.01 < np.mean(np.array(ps) < 0.05) < 0.12
