"""Fidelity gate: our Windows env + model must reproduce real Sample Factory exactly.

The fixture tests/fixtures/sf_golden.npz is produced by scripts/sf_golden_wsl.py with sample-factory 2.0.3,
gym 0.26.2, ale-py 0.8.1 (the training-era stack) from SF's own make_atari_env and create_actor_critic.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from pacneuron.env import Game
from pacneuron.hub import checkpoint_path
from pacneuron.model import load_sf_checkpoint

FIX = Path(__file__).parent / "fixtures" / "sf_golden.npz"
pytestmark = pytest.mark.skipif(not FIX.exists(), reason="golden fixture missing (run scripts/sf_golden_wsl.sh)")


@pytest.fixture(scope="module")
def golden():
    return dict(np.load(FIX))


def test_observations_bit_exact(golden):
    """Same seed + same actions -> identical stacked 84x84 frames at every step, incl. life-loss resets."""
    game = Game(int(golden["seed"]))
    for t, a in enumerate(golden["actions"]):
        np.testing.assert_array_equal(game.obs, golden["obs"][t], err_msg=f"obs differ at step {t}")
        _, life_lost, game_over = game.step(int(a))
        assert life_lost == bool(golden["life_lost"][t]), f"life_lost differs at step {t}"
        assert game_over == bool(golden["game_over"][t]), f"game_over differs at step {t}"
        if game_over:
            break


def test_logits_and_values_match(golden):
    """Our pure-torch forward == SF forward on the golden observations (float32 tolerance)."""
    model = load_sf_checkpoint(checkpoint_path("1111"))
    with torch.no_grad():
        out = model(torch.as_tensor(golden["obs"]))
    np.testing.assert_allclose(out["logits"].numpy(), golden["logits"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(out["value"].numpy(), golden["values"].reshape(-1), atol=1e-5, rtol=1e-5)
