"""Closed-loop steering arms: play games with an intervention spec, save compact records, compute metrics.

Local: a spec is a JSON-able dict {"site", "units", "value", "mode", "cell", "gate"}; `edit_factory(spec)` turns it
into a per-step edit for rollout.run_games (gates read the frightened timer from RAM each step).
Global: used for discovery pilots and for the preregistered held-out arms (baseline, candidates, null units).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pacneuron import data
from pacneuron.metrics import game_metrics
from pacneuron.model import Intervention
from pacneuron.ram import frightened_timer
from pacneuron.rollout import run_games

KEEP = ("ram", "maze", "action", "raw_reward", "life_lost", "game_over", "logits")


def edit_factory(spec: dict | None):
    """Spec -> rollout edit factory (None = unsteered baseline)."""
    if spec is None:
        return None

    def make(rams: np.ndarray, _steps: np.ndarray):
        gate = spec.get("gate", "none")
        rows = None
        if gate != "none":
            fr = frightened_timer(rams) > 0
            rows = torch.as_tensor(fr if gate == "frightened" else ~fr)
        cell = tuple(spec["cell"]) if spec.get("cell") is not None else None
        return Intervention(spec["site"], list(spec["units"]), float(spec["value"]), spec.get("mode", "set"), cell, rows)
    return make


def run_arm(model, seeds, spec: dict | None, out_dir: Path, desc: str) -> None:
    """Play `seeds` under `spec`; write compressed per-game records (+ maze bank) to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def save(rec):
        np.savez_compressed(out_dir / f"seed_{rec['seed']:06d}.npz", seed=rec["seed"], **{k: rec[k] for k in KEEP})
    _, bank = run_games(model, seeds, edit_factory=edit_factory(spec), on_game=save, desc=desc)
    data.save_bank(out_dir, bank)


def arm_metrics(out_dir: Path, **metric_kwargs) -> dict[int, dict]:
    """{game seed: metric dict} for every recorded game of an arm."""
    bank = data.load_bank(out_dir)
    return {g["seed"]: game_metrics(g, bank.mazes, **metric_kwargs) for g in data.load_games(out_dir, KEEP)}
