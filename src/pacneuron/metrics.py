"""Per-game behavioural metrics for steering experiments (preregistration §6), computed from RAM ground truth.

Local: `game_metrics(rec, mazes, target=None, trigger=None)` returns a dict of scalars for one recorded game.
Global: steer.py computes these for every arm (baseline, candidates, null units) and stats.py compares arms
game-by-game (paired by seed).
"""

from __future__ import annotations

import numpy as np

from pacneuron import maze as M
from pacneuron import ram as R
from pacneuron.features import compute_features, in_control

# Directions (up, right, left, down) contained in each action id (0 NOOP ... 8 DOWNLEFT).
ACTION_DIRS = np.array([
    [0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
    [1, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]], bool)
HORIZON = 8  # agent steps over which "approach" is measured


def cell_target_node(cell: tuple[int, int], mz: M.Maze) -> int:
    """Main-component lattice node nearest (in screen px) to the centre of conv3 cell (i, j)'s receptive field."""
    i, j = cell
    cx, cy = (8 * j + 17.5) * 160 / 84, (8 * i + 17.5) * 210 / 84  # RF centre: obs -> screen
    nodes = np.flatnonzero(mz.open.ravel())
    main = nodes[(mz.dist[nodes][:, nodes] < M.UNREACHABLE).sum(1) > len(nodes) // 2]
    r, c = np.divmod(main, M.COLS)
    d = np.hypot(M.PELLET_X[c] + 2 - cx, 7 + 12 * r + 1 - cy)
    return int(main[np.argmin(d)])


def approach_rate(rec: dict, mazes: list, target_cell: tuple[int, int]) -> tuple[float, float]:
    """(P(BFS distance to the cell's target node shrinks within HORIZON steps), fraction of time within 3 steps).

    Evaluated over in-control steps with a positive distance; distances use the maze of the current step.
    """
    ram, mz_id = rec["ram"], rec["maze"]
    xy = R.player_xy(ram)
    node = M.node_of(xy[:, 0], xy[:, 1])
    d = np.full(len(ram), np.nan)
    for mid in np.unique(mz_id):  # target node depends only on the maze layout: compute once per layout
        sel = mz_id == mid
        d[sel] = mazes[mid].dist[node[sel], cell_target_node(target_cell, mazes[mid])]
    d[d >= M.UNREACHABLE] = np.nan
    ok = in_control(ram)
    now, later = d[:-HORIZON], d[HORIZON:]
    valid = ok[:-HORIZON] & (now > 0) & ~np.isnan(now) & ~np.isnan(later)
    rate = float(np.mean(later[valid] < now[valid])) if valid.any() else np.nan
    near = float(np.mean(d[ok & ~np.isnan(d)] <= 3)) if (ok & ~np.isnan(d)).any() else np.nan
    return rate, near


def game_metrics(rec: dict, mazes: list, target_cell=None, trigger: str | None = None,
                 action_set: list[int] | None = None) -> dict:
    """Scalar behaviour summary of one game (see preregistration §6 for definitions)."""
    ram = rec["ram"]
    ctrl = in_control(ram)
    n_ctrl = max(int(ctrl.sum()), 1)
    edible, timer = R.ghost_edible(ram), R.frightened_timer(ram)
    ghosts = int((edible[:-1] & ~edible[1:] & (timer[1:] > 0)[:, None]).sum())
    pp = R.power_pills(ram)
    pills = int((pp[:-1] & ~pp[1:]).sum())
    pel = R.pellets(ram).reshape(len(ram), -1).sum(1)
    pellets = int(np.clip(pel[:-1] - pel[1:], 0, None).sum())
    deaths = int(np.sum(rec["life_lost"]) + np.sum(rec["game_over"]))
    m = {"return": float(np.sum(rec["raw_reward"])), "steps": len(ram), "in_control": n_ctrl,
         "deaths_per_1k": 1000 * deaths / n_ctrl, "pellets_per_1k": 1000 * pellets / n_ctrl,
         "ghosts_eaten": ghosts, "pills_eaten": pills, "ghosts_per_pill": ghosts / pills if pills else np.nan}
    if target_cell is not None:
        m["approach"], m["near_target"] = approach_rate(rec, mazes, tuple(target_cell))
    if trigger is not None and action_set is not None:
        f = compute_features(rec, mazes)[trigger] > 0
        sel = f & ctrl
        m["action_given_trigger"] = float(np.isin(rec["action"][sel], action_set).mean()) if sel.any() else np.nan
    return m
