"""Project configuration: paths, model ids, and fixed experimental constants.

Local: reads `.env` (python-dotenv) for machine-specific values (data dir, HF cache, workers).
Global: every other module imports constants from here so seeds/splits/model revisions are
defined exactly once (needed for the preregistration to be checkable).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Repository root = two levels above this file (src/pacneuron/config.py -> repo/).
REPO_ROOT = Path(__file__).resolve().parents[2]
# Load .env before anything reads os.environ (HF_HOME must be set before huggingface_hub import).
load_dotenv(REPO_ROOT / ".env")

# Bulk data (checkpoints, activation shards, videos) lives outside git; default inside repo/data.
DATA_DIR = Path(os.environ.get("DATA_DIR", REPO_ROOT / "data"))
# Small, committed, machine-readable results that all reports/figures are generated from.
RESULTS_DIR = REPO_ROOT / "results"
# Generated figures that README/paper/site embed.
FIGURES_DIR = REPO_ROOT / "figures"
# Timestamped run logs (gitignored).
LOGS_DIR = REPO_ROOT / "logs"
# Parallel env worker processes for rollouts (ALE emulation is the bottleneck, not the GPU).
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "12"))
# Torch device for the policy network.
DEVICE = os.environ.get("DEVICE", "cuda")


@dataclass(frozen=True)
class ModelSpec:
    """One pretrained agent on the Hugging Face Hub.

    repo_id/filename identify the Sample Factory checkpoint; revision pins the exact git commit
    of the HF repo so results are reproducible even if the repo changes later.
    """

    name: str
    repo_id: str
    filename: str
    revision: str
    reported_mean_reward: float
    reported_std_reward: float


# Sample Factory 2.0 APPO agents trained 2B env steps on MsPacmanNoFrameskip-v4 (reward only).
# Source: https://huggingface.co/edbeeching/atari_2B_atari_mspacman_1111 (cfg.json, README.md).
# cfg.json has load_checkpoint_kind="latest", so the model-card reward refers to the latest checkpoint.
# Revisions are the HF git commit SHAs at the time of this study (pinned for reproducibility).
# All three runs end with the same "latest" checkpoint name (policy version 976624, 2,000,125,952 env steps).
_LATEST = "checkpoint_p0/checkpoint_000976624_2000125952.pth"
MODELS: dict[str, ModelSpec] = {
    "1111": ModelSpec("appo_1111", "edbeeching/atari_2B_atari_mspacman_1111",
                      _LATEST, "07c1243fc96930ca3e632bb4574246140f63d54b", 6288.00, 1954.58),
    "2222": ModelSpec("appo_2222", "edbeeching/atari_2B_atari_mspacman_2222",
                      _LATEST, "4cbe1d6c811280fffea83e7b5130ac55465fa5b5", 9056.50, 2544.32),
    "3333": ModelSpec("appo_3333", "edbeeching/atari_2B_atari_mspacman_3333",
                      _LATEST, "1364c94752de18e6d52b419e73681802de270b05", 6808.00, 1183.09),
}

# Environment id used at training time (Sample Factory sf_examples/atari/atari_utils.py, AtariSpec).
ENV_ID = "MsPacmanNoFrameskip-v4"
# Number of discrete actions in the minimal Ms. Pac-Man action set (NOOP, UP, RIGHT, LEFT, DOWN, 4 diagonals).
N_ACTIONS = 9

# Preregistered episode splits (policy-RNG seeds). Discovery is used for labelling, screening and
# candidate selection; held-out is only read by confirmatory phases after the prereg commit.
DISCOVERY_SEEDS = tuple(range(0, 100))
# ALE sticky-action probability. The preregistered stochasticity check (results/stochasticity_appo_1111_nosticky.json)
# found same-start games stay identical for a median of 1230 agent steps (> 200 threshold), so the preregistered
# fallback applies: every game in the study uses repeat_action_probability = 0.25 (Machado et al. 2018 recommendation,
# https://arxiv.org/abs/1709.06009). The fidelity tests still run without sticky actions (training setting).
STICKY = 0.25
HELDOUT_SEEDS = tuple(range(10000, 10100))
