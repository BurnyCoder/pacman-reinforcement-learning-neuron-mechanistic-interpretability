"""Play many full games in parallel worker processes with batched GPU inference and optional interventions.

Local: `run_games(model, seeds, ...)` returns one record dict per game (RAM, maze id, rewards, actions, logits,
values, and optionally conv3 / fc activations). Each worker owns one `env.Game`; the main process batches the
active games' observations through the network, samples actions with a per-game RNG, and sends them back.
Global: the same function produces discovery/held-out audit data (record_acts=True) and every steering arm
(edit_fn != None). Common random numbers: the action-sampling RNG of game `seed` is seeded by `seed` alone, so
baseline and steered arms of the same seed share noop starts and random draws until their actions diverge.
"""

from __future__ import annotations

import multiprocessing as mp
from typing import Callable

import numpy as np
import torch

from pacneuron import maze as M
from pacneuron.config import DEVICE, NUM_WORKERS
from pacneuron.env import Game
from pacneuron.logs import get_logger
from pacneuron.model import EditFn, SFActorCritic
from pacneuron.ram import FRIGHTENED_RGB, ghosts_xy

log = get_logger(__name__)

# edit_factory(ram_batch [B,128] uint8, step_batch [B]) -> EditFn or None; lets an intervention depend on game state.
EditFactory = Callable[[np.ndarray, np.ndarray], EditFn | None]


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


class _Pool:
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


def run_games(model: SFActorCritic, seeds, edit_factory: EditFactory | None = None, record_acts: bool = False,
              sticky: float = 0.0, n_workers: int = NUM_WORKERS, games_per_worker: int = 3,
              bank: M.MazeBank | None = None, desc: str = "rollout") -> tuple[list[dict], M.MazeBank]:
    """Play one full game per seed; returns (records in seed order, MazeBank whose ids the records' "maze" use).

    Runs n_workers processes x games_per_worker concurrent games. Pass the same `bank` across calls so maze ids
    stay consistent within an experiment.
    """
    seeds = list(seeds)
    bank = bank or M.MazeBank()
    n_slots = min(n_workers * games_per_worker, len(seeds))
    n_workers = min(n_workers, n_slots)
    pool = _Pool(n_workers, sticky)
    queue = list(enumerate(seeds))
    slots: dict[int, dict] = {}  # slot -> running game buffers; slot s lives on worker s % n_workers
    records: dict[int, dict] = {}
    pending: dict[int, list] = {w: [] for w in range(n_workers)}  # commands to send per worker this round

    def start(slot):
        k, seed = queue.pop(0)
        pending[slot % n_workers].append((slot, "reset", seed))
        slots[slot] = {"k": k, "seed": seed, "rng": np.random.default_rng(seed), "buf": {}, "state": None}

    def push(buf, key, val):
        buf.setdefault(key, []).append(val)

    for slot in range(n_slots):
        start(slot)
    try:
        while slots:
            # 1) send this round's commands, 2) receive every worker's states (workers step in parallel)
            for w in range(n_workers):
                if pending[w]:
                    pool.conns[w].send(pending[w])
            replies = [pool.conns[w].recv() for w in range(n_workers) if pending[w]]
            pending = {w: [] for w in range(n_workers)}
            for reply in replies:
                for slot, st in reply:
                    st["maze_id"] = bank.add(st["nbr"])
                    s = slots[slot]
                    if st["trans"] is not None:
                        r, life_lost, over = st["trans"]
                        push(s["buf"], "raw_reward", r)
                        push(s["buf"], "life_lost", life_lost)
                        push(s["buf"], "game_over", over)
                        if over:
                            rec = {k: np.asarray(v) for k, v in s["buf"].items()}
                            rec["seed"] = s["seed"]
                            records[s["k"]] = rec
                            log.info(f"[{desc}] game seed={rec['seed']} score={rec['raw_reward'].sum():.0f} "
                                     f"steps={len(rec['action'])} ({len(records)}/{len(seeds)})")
                            del slots[slot]
                            if queue:
                                start(slot)
                            continue
                    s["state"] = st
            # 3) batched forward pass over all games that have a fresh state
            ready = sorted(sl for sl in slots if slots[sl]["state"] is not None)
            if not ready:
                continue
            obs = torch.as_tensor(np.stack([slots[sl]["state"]["obs"] for sl in ready]), device=DEVICE)
            rams = np.stack([slots[sl]["state"]["ram"] for sl in ready])
            steps = np.array([len(slots[sl]["buf"].get("action", [])) for sl in ready])
            edit = edit_factory(rams, steps) if edit_factory else None
            with torch.no_grad():
                out = model(obs, edit=edit)
            probs = torch.softmax(out["logits"].double(), -1).cpu().numpy()
            logits = out["logits"].float().cpu().numpy()
            value = out["value"].float().cpu().numpy()
            if record_acts:
                c3 = out["conv3"].half().cpu().numpy()
                fcp = out["fc_pre"].half().cpu().numpy()
            for b, sl in enumerate(ready):
                s, buf = slots[sl], slots[sl]["buf"]
                # inverse-CDF sampling with the game's own RNG (same as Categorical sampling, reproducible per seed)
                a = int(min(np.searchsorted(np.cumsum(probs[b]), s["rng"].random()), len(probs[b]) - 1))
                push(buf, "ram", s["state"]["ram"])
                push(buf, "blue", s["state"]["blue"])
                push(buf, "maze", s["state"]["maze_id"])
                push(buf, "action", a)
                push(buf, "logits", logits[b])
                push(buf, "value", value[b])
                if record_acts:
                    push(buf, "conv3", c3[b])
                    push(buf, "fc_pre", fcp[b])
                pending[sl % n_workers].append((sl, "step", a))
                s["state"] = None
    finally:
        pool.close()
    return [records[k] for k in range(len(seeds))], bank
