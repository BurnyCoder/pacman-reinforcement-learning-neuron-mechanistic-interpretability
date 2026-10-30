"""Environment worker processes (deliberately torch-free so each spawned process stays small).

Local: `Pool(n, sticky)` starts n processes; each runs `_worker`, which owns several `env.Game`s and steps them
on request, returning observations plus ground truth (RAM, frightened-blue pixel counts, maze lattice).
Global: rollout.run_games drives a Pool; Windows requires the spawn start method, which re-imports this module
(not rollout.py/torch) in every child.
"""

from __future__ import annotations

import multiprocessing as mp

import numpy as np

from pacneuron import maze as M
from pacneuron.env import Game
from pacneuron.ram import FRIGHTENED_RGB, ghosts_xy


def _blue_pixels(rgb: np.ndarray, ram: np.ndarray) -> np.ndarray:
    """Count frightened-blue pixels inside each ghost's sprite box (pixel-level validation of the RAM edible bit)."""
    out = np.zeros(4, np.uint8)
    for g, (x, y) in enumerate(ghosts_xy(ram)):
        win = rgb[max(y - 1, 0):y + 12, max(x - 1, 0):x + 12]
        out[g] = min(int((win == FRIGHTENED_RGB).all(-1).sum()), 255)
    return out


def _worker(conn, sticky: float) -> None:
    """Worker loop over several games. Message: list of (slot, cmd, arg) with cmd 'reset' (arg=seed) or
    'step' (arg=action); reply: list of (slot, state). ('close', None) ends the process."""
    games: dict[int, Game] = {}
    nbr_cache: dict[int, tuple] = {}  # slot -> (wall colour, maze id byte, neighbour table)

    def state(slot, trans, force):
        game, info = games[slot], games[slot].info
        rgb, ram = info["rgb"], info["ram"]
        sig = (tuple(rgb[1, 80]), int(ram[0]))
        # the maze only changes at level transitions: re-extract the lattice on reset/life loss or a wall-colour
        # / maze-byte change (extraction costs ~1.7 ms, a game step ~1.2 ms)
        if force or slot not in nbr_cache or nbr_cache[slot][0] != sig:
            nbr_cache[slot] = (sig, M.lattice(rgb).astype(np.int16))
        return {"obs": game.obs, "ram": ram, "blue": _blue_pixels(rgb, ram), "nbr": nbr_cache[slot][1],
                "trans": trans}

    while True:
        msg = conn.recv()
        if msg == ("close", None):
            conn.close()
            return
        out = []
        for slot, cmd, arg in msg:
            if cmd == "reset":
                if slot in games:
                    games[slot].close()
                games[slot] = Game(arg, sticky)
                out.append((slot, state(slot, None, True)))
            else:
                info, life_lost, over = games[slot].step(arg)
                out.append((slot, state(slot, (info["raw_reward"], life_lost, over), life_lost)))
        conn.send(out)


class Pool:
    """Fixed pool of worker processes (spawn context: required on Windows)."""

    def __init__(self, n: int, sticky: float):
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for _ in range(n):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(child, sticky), daemon=True)
            p.start()
            self.conns.append(parent)
            self.procs.append(p)

    def close(self):
        for c in self.conns:
            c.send(("close", None))
        for p in self.procs:
            p.join(timeout=5)
