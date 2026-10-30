"""Re-render recorded states (RGB frames) by deterministic replay instead of storing every frame.

Local: `capture_frames(directory, requests)` replays each requested game once (in parallel processes) with its
recorded actions and returns {(seed, t): rgb}. Replay is exact (tests/test_ground_truth.py), including sticky
actions, because ALE's RNG is seeded by the game seed.
Global: used for top-activating example grids (microscope site, README, paper) and steering videos.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import numpy as np

from pacneuron.config import NUM_WORKERS


def _replay_one(args):
    path, steps, size = args
    from PIL import Image

    from pacneuron.env import replay  # imported in the child (spawn) to keep the parent light
    with np.load(path) as z:
        seed, actions = int(z["seed"]), z["action"]
    out = replay(seed, actions, capture=set(steps))
    shrink = (lambda f: f) if size is None else (
        lambda f: np.asarray(Image.fromarray(f[:172]).resize(size, Image.LANCZOS)))  # playfield only, downsampled
    return {(seed, t): shrink(out[t]["rgb"]) for t in steps if t in out}


def capture_frames(directory: Path, requests, size: tuple[int, int] | None = None) -> dict[tuple[int, int], np.ndarray]:
    """requests: iterable of (seed, t). Returns {(seed, t): RGB frame} (full 210x160, or playfield resized to size)."""
    by_game: dict[int, set] = {}
    for seed, t in requests:
        by_game.setdefault(int(seed), set()).add(int(t))
    jobs = [(directory / f"seed_{s:06d}.npz", sorted(ts), size) for s, ts in by_game.items()]
    frames: dict = {}
    with mp.get_context("spawn").Pool(min(NUM_WORKERS, max(len(jobs), 1))) as pool:
        for part in pool.imap_unordered(_replay_one, jobs):
            frames.update(part)
    return frames


def grid(frames: list[np.ndarray], cols: int = 4, pad: int = 2, crop: int = 172) -> np.ndarray:
    """Tile RGB frames (playfield rows only) into one image with a white gutter."""
    fs = [f[:crop] for f in frames]
    h, w = fs[0].shape[:2]
    rows = (len(fs) + cols - 1) // cols
    img = np.full((rows * (h + pad) + pad, cols * (w + pad) + pad, 3), 255, np.uint8)
    for k, f in enumerate(fs):
        r, c = divmod(k, cols)
        img[pad + r * (h + pad): pad + r * (h + pad) + h, pad + c * (w + pad): pad + c * (w + pad) + w] = f
    return img
