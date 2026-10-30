"""Shared fixtures: the seed-1111 agent and a short agent-played game with ground truth per step."""

import numpy as np
import pytest
import torch

from pacneuron.env import Game
from pacneuron.hub import checkpoint_path
from pacneuron.model import load_sf_checkpoint


@pytest.fixture(scope="session")
def model():
    return load_sf_checkpoint(checkpoint_path("1111"))


@pytest.fixture(scope="session")
def played(model):
    """1400 agent steps of game seed 0 (argmax actions, CPU): obs, RGB, RAM per step + actions and rewards."""
    game, out = Game(0, sticky=0.0), {"obs": [], "rgb": [], "ram": [], "action": [], "reward": [], "life_lost": []}
    for _ in range(1400):
        with torch.no_grad():
            a = int(model(torch.as_tensor(game.obs[None]))["logits"].argmax())
        out["obs"].append(game.obs)
        out["rgb"].append(game.info["rgb"])
        out["ram"].append(game.info["ram"])
        info, life_lost, over = game.step(a)
        out["action"].append(a)
        out["reward"].append(info["raw_reward"])
        out["life_lost"].append(life_lost)
        if over:
            break
    return {k: np.asarray(v) for k, v in out.items()}
