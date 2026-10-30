"""On-disk storage of recorded games and loading them as concatenated, in-control step tables.

Local: one uncompressed .npz per game under DATA_DIR/rollouts/<model>/<split>/, plus mazes.pkl (the MazeBank
whose ids the games' "maze" arrays refer to).
Global: collection phases write here; audit/screen/steering analyses read whole splits through `load_split`.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from pacneuron import maze as M
from pacneuron.config import DATA_DIR
from pacneuron.features import compute_features, in_control


def split_dir(model: str, split: str) -> Path:
    d = DATA_DIR / "rollouts" / model / split
    d.mkdir(parents=True, exist_ok=True)
    return d


def saver(directory: Path):
    """Return an `on_game` callback that writes each finished game to <directory>/seed_<seed>.npz."""
    def _save(rec: dict) -> None:
        np.savez(directory / f"seed_{rec['seed']:06d}.npz", **rec)
    return _save


def save_bank(directory: Path, bank: M.MazeBank) -> None:
    with open(directory / "mazes.pkl", "wb") as fh:
        pickle.dump(bank, fh)


def load_bank(directory: Path) -> M.MazeBank:
    with open(directory / "mazes.pkl", "rb") as fh:
        return pickle.load(fh)


def load_games(directory: Path, keys: tuple[str, ...] | None = None) -> list[dict]:
    """Load every game in a split directory (sorted by seed); `keys` restricts which arrays are read."""
    games = []
    for f in sorted(directory.glob("seed_*.npz")):
        with np.load(f) as z:
            games.append({k: z[k] for k in (keys or z.files) if k in z.files} | {"seed": int(z["seed"])})
    return games


def load_split(directory: Path, acts: tuple[str, ...] = ("fc_pre", "conv3", "aux_fc_pre")) -> dict:
    """Concatenate all games of a split into step tables restricted to in-control steps.

    Returns {"features": {name: [N]}, "game": [N] game index, "t": [N] step index, "seeds": [G],
    "action", "logits", "value", "ram", plus every requested activation array}.
    """
    bank = load_bank(directory)
    base = ("ram", "maze", "raw_reward", "life_lost", "game_over", "action", "logits", "value")
    games = load_games(directory, base + acts)
    out: dict = {"features": {}, "game": [], "t": [], "seeds": np.array([g["seed"] for g in games])}
    cols: dict[str, list] = {}
    for gi, g in enumerate(games):
        feats = compute_features(g, bank.mazes)
        keep = in_control(g["ram"])
        for k, v in feats.items():
            cols.setdefault(("f", k), []).append(v[keep])
        for k in ("action", "logits", "value", "ram", "maze") + tuple(a for a in acts if a in g):
            cols.setdefault(("a", k), []).append(g[k][keep])
        out["game"].append(np.full(keep.sum(), gi))
        out["t"].append(np.flatnonzero(keep))
    for (kind, k), parts in cols.items():
        arr = np.concatenate(parts)
        if kind == "f":
            out["features"][k] = arr
        else:
            out[k] = arr
    out["game"], out["t"] = np.concatenate(out["game"]), np.concatenate(out["t"])
    return out
