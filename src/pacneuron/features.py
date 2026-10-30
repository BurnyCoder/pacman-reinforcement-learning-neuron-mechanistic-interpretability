"""Ground-truth feature library used to label neurons (all from emulator RAM + the static maze graph).

Local: `compute_features(rec, mazes)` turns one recorded game (RAM per step, maze id per step, transition info)
into a dict name -> float32 array [T]. Directional features are one-hot over (up, right, left, down).
Global: audit.py correlates every neuron with every feature; metrics.py reuses the same quantities for
behavioural metrics of steering experiments. Agent outputs (value, entropy, action) are deliberately NOT here:
they are linear read-outs of the fc layer and would trivially "explain" fc neurons (they are reported separately).

Feature families (FAMILY maps name -> family, used for per-family trained-vs-random comparisons):
  player   : position, heading, open directions at the current tile, junction
  danger   : maze (BFS) / Euclidean distance and first-step direction to the nearest non-edible ghost
  edible   : distance and direction to the nearest edible ghost, frightened flag and timer
  food     : distance and direction to the nearest pellet / power pill, counts remaining
  game     : maze id, imminent life loss, rewards, ghost-eaten events
"""

from __future__ import annotations

import numpy as np

from pacneuron import maze as M
from pacneuron import ram as R

DIRS = M.DIRS
DANGER_RADIUS = 5  # lattice steps; "ghost nearby" threshold for the count feature
SOON = 10  # agent steps; horizon for "life lost soon"


def _onehot_dirs(prefix: str, steps: np.ndarray) -> dict[str, np.ndarray]:
    return {f"{prefix}_{d}": steps[:, k].astype(np.float32) for k, d in enumerate(DIRS)}


def compute_features(rec: dict, mazes: list[M.Maze]) -> dict[str, np.ndarray]:
    """rec: {"ram": [T,128] uint8, "maze": [T] int, "raw_reward": [T], "life_lost": [T] bool}.

    Feature value at index t describes the state the agent acts on at decision t (raw_reward[t] and
    life_lost[t] describe the transition caused by action t).
    """
    ram, mz_id = rec["ram"], rec["maze"]
    T = len(ram)
    pxy, gxy = R.player_xy(ram), R.ghosts_xy(ram)
    pnode = M.node_of(pxy[:, 0], pxy[:, 1])
    gnode = M.node_of(gxy[..., 0], gxy[..., 1])  # [T, 4]
    active, edible = R.ghosts_active(ram), R.ghost_edible(ram)
    danger = active & ~edible
    pel = R.pellets(ram).reshape(T, -1)
    pp = R.power_pills(ram)
    pp_node = M.node_of(R.POWER_PILL_XY[:, 0] - 2.5, R.POWER_PILL_XY[:, 1] - 1.5)  # pill 4x7 -> sprite-centre convention

    f: dict[str, np.ndarray] = {}
    f["player_x"], f["player_y"] = (pxy[:, 0] + 4).astype(np.float32), (pxy[:, 1] + 4.5).astype(np.float32)
    f["heading"] = R.heading(ram).astype(np.float32)
    open_dirs = np.zeros((T, 4), bool)
    d_danger = np.full((T, 4), M.UNREACHABLE, np.float32)
    d_edible = np.full((T, 4), M.UNREACHABLE, np.float32)
    dir_danger, dir_edible = np.zeros((T, 4), bool), np.zeros((T, 4), bool)
    d_pellet = np.full(T, M.UNREACHABLE, np.float32)
    dir_pellet, dir_power = np.zeros((T, 4), bool), np.zeros((T, 4), bool)
    d_power = np.full(T, M.UNREACHABLE, np.float32)
    eu_danger = np.full(T, 250.0, np.float32)
    for t in range(T):
        mz = mazes[mz_id[t]]
        p = pnode[t]
        open_dirs[t] = mz.nbr.reshape(-1, 4)[p] >= 0
        dist = mz.dist[p]
        for g in range(4):
            if danger[t, g]:
                d_danger[t, g] = dist[gnode[t, g]]
            elif active[t, g] and edible[t, g]:
                d_edible[t, g] = dist[gnode[t, g]]
        if danger[t].any():
            g = int(np.argmin(np.where(danger[t], d_danger[t], np.inf)))
            dir_danger[t] = mz.first_steps(p, gnode[t, g])
            eu_danger[t] = np.hypot(*(gxy[t, danger[t]] - pxy[t]).T).min()
        if (active[t] & edible[t]).any():
            dir_edible[t] = mz.first_steps(p, gnode[t, int(np.argmin(d_edible[t]))])
        if pel[t].any():
            cand = np.flatnonzero(pel[t])
            k = cand[np.argmin(dist[cand])]
            d_pellet[t], dir_pellet[t] = dist[k], mz.first_steps(p, k)
        if pp[t].any():
            cand = pp_node[pp[t]]
            k = cand[np.argmin(dist[cand])]
            d_power[t], dir_power[t] = dist[k], mz.first_steps(p, k)

    f.update(_onehot_dirs("open", open_dirs))
    f["junction"] = (open_dirs.sum(1) >= 3).astype(np.float32)
    ds = np.sort(d_danger, 1)
    f["d_danger1"], f["d_danger2"] = ds[:, 0], ds[:, 1]
    f["eu_danger1"] = eu_danger
    f["n_danger_near"] = (d_danger <= DANGER_RADIUS).sum(1).astype(np.float32)
    f.update(_onehot_dirs("dir_danger", dir_danger))
    f["d_edible1"] = d_edible.min(1)
    f.update(_onehot_dirs("dir_edible", dir_edible))
    f["frightened"] = (R.frightened_timer(ram) > 0).astype(np.float32)
    f["fright_timer"] = R.frightened_timer(ram).astype(np.float32)
    f["n_edible"] = (active & edible).sum(1).astype(np.float32)
    f["d_pellet"] = d_pellet
    f.update(_onehot_dirs("dir_pellet", dir_pellet))
    f["n_pellets"] = pel.sum(1).astype(np.float32)
    f["d_power"] = d_power
    f.update(_onehot_dirs("dir_power", dir_power))
    f["n_power"] = pp.sum(1).astype(np.float32)
    f["maze"] = np.asarray(ram[:, 0], np.float32)
    lost = np.asarray(rec["life_lost"], bool) | np.asarray(rec.get("game_over", np.zeros(T, bool)), bool)
    soon = np.zeros(T, bool)
    for t in np.flatnonzero(lost):
        soon[max(0, t - SOON + 1): t + 1] = True
    f["life_lost_soon"] = soon.astype(np.float32)
    f["reward_next"] = np.asarray(rec["raw_reward"], np.float32)
    eaten = np.zeros(T, bool)
    eaten[:-1] = (edible[:-1] & ~edible[1:] & (R.frightened_timer(ram[1:]) > 0)[:, None]).any(1)
    f["ghost_eaten_next"] = eaten.astype(np.float32)
    return f


def in_control(ram: np.ndarray) -> np.ndarray:
    """[T] bool: Ms. Pac-Man moved between t-1 and t or between t and t+1 (excludes death animation,
    ready screen and level transitions, during which she is frozen)."""
    xy = R.player_xy(ram)
    moved = np.zeros(len(ram), bool)
    step = (xy[1:] != xy[:-1]).any(1)
    moved[1:] |= step
    moved[:-1] |= step
    return moved


FAMILY = {
    **{k: "player" for k in ["player_x", "player_y", "heading", "junction"] + [f"open_{d}" for d in DIRS]},
    **{k: "danger" for k in ["d_danger1", "d_danger2", "eu_danger1", "n_danger_near"] + [f"dir_danger_{d}" for d in DIRS]},
    **{k: "edible" for k in ["d_edible1", "frightened", "fright_timer", "n_edible"] + [f"dir_edible_{d}" for d in DIRS]},
    **{k: "food" for k in ["d_pellet", "n_pellets", "d_power", "n_power"]
       + [f"dir_pellet_{d}" for d in DIRS] + [f"dir_power_{d}" for d in DIRS]},
    **{k: "game" for k in ["maze", "life_lost_soon", "reward_next", "ghost_eaten_next"]},
}
CATEGORICAL = {"heading", "maze"}  # treated as categories (not ordinal) by the audit


# ------------------------------------------------------------------------------------------------------------
# Spatial ground truth for conv3 channels. conv3 unit (i, j) sees obs pixels [8i, 8i+35] x [8j, 8j+35] of the 84x84
# input (conv1 k8 s4 -> conv2 k4 s2 -> conv3 k3 s1); we call its central 16 px [8i+10, 8i+25] the cell "core".
# Screen (160x210) -> obs (84x84) scaling follows the 84x84 INTER_AREA resize in env.py.
# ------------------------------------------------------------------------------------------------------------
CELL_TYPES = ("danger", "edible", "player", "pellet", "power")


def screen_to_cells(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """[..., 7, 7] bool: which conv3 cell cores contain the screen point (x, y)."""
    ox, oy = np.asarray(x) * 84 / 160, np.asarray(y) * 84 / 210
    lo = 8 * np.arange(7) + 10
    inx = (ox[..., None] >= lo) & (ox[..., None] <= lo + 15)
    iny = (oy[..., None] >= lo) & (oy[..., None] <= lo + 15)
    return iny[..., :, None] & inx[..., None, :]


def cell_objects(ram: np.ndarray) -> dict[str, np.ndarray]:
    """Per step, which conv3 cells contain each object type: {type: [T, 7, 7] bool} (pellet: count)."""
    T = len(ram)
    pxy, gxy = R.player_xy(ram), R.ghosts_xy(ram)
    active, edible = R.ghosts_active(ram), R.ghost_edible(ram)
    out = {"player": screen_to_cells(pxy[:, 0] + 4, pxy[:, 1] + 4.5)}
    g = screen_to_cells(gxy[..., 0] + 4.5, gxy[..., 1] + 5)  # [T, 4, 7, 7]
    out["danger"] = (g & (active & ~edible)[..., None, None]).any(1)
    out["edible"] = (g & (active & edible)[..., None, None]).any(1)
    pel = R.pellets(ram)  # [T, 14, 18]
    pc = screen_to_cells(M.PELLET_X[None, :] + 2.0, (7 + 12 * np.arange(14))[:, None] + 1.0)  # [14, 18, 7, 7]
    out["pellet"] = np.einsum("tij,ijab->tab", pel.astype(np.float32), pc.astype(np.float32))
    pp = R.power_pills(ram)  # [T, 4]
    ppc = screen_to_cells(R.POWER_PILL_XY[:, 0] + 2.0, R.POWER_PILL_XY[:, 1] + 3.5)  # [4, 7, 7]
    out["power"] = np.einsum("tk,kab->tab", pp.astype(np.float32), ppc.astype(np.float32)) > 0
    return out
