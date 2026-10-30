"""Play many full games in parallel worker processes with batched GPU inference and optional interventions.

Local: `run_games(model, seeds, ...)` returns one record dict per game (RAM, maze id, rewards, actions, logits,
values, and optionally conv3 / fc activations). Each worker owns one `env.Game`; the main process batches the
active games' observations through the network, samples actions with a per-game RNG, and sends them back.
Global: the same function produces discovery/held-out audit data (record_acts=True) and every steering arm
(edit_fn != None). Common random numbers: the action-sampling RNG of game `seed` is seeded by `seed` alone, so
baseline and steered arms of the same seed share noop starts and random draws until their actions diverge.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from pacneuron import maze as M
from pacneuron.config import DEVICE, NUM_WORKERS, STICKY
from pacneuron.logs import get_logger
from pacneuron.model import EditFn, SFActorCritic
from pacneuron.worker import Pool

log = get_logger(__name__)

# edit_factory(ram_batch [B,128] uint8, step_batch [B], job_batch [B]) -> EditFn or None. Lets an intervention
# depend on game state and, via job indices (positions in `seeds`), differ between games of one batched run.
EditFactory = Callable[[np.ndarray, np.ndarray, np.ndarray], EditFn | None]


def run_games(model: SFActorCritic, seeds, edit_factory: EditFactory | None = None, record_acts: bool = False,
              sticky: float = STICKY, n_workers: int = NUM_WORKERS, games_per_worker: int = 3,
              bank: M.MazeBank | None = None, desc: str = "rollout", aux_model: SFActorCritic | None = None,
              on_game: Callable[[dict], None] | None = None) -> tuple[list[dict], M.MazeBank]:
    """Play one full game per seed; returns (records in seed order, MazeBank whose ids the records' "maze" use).

    Runs n_workers processes x games_per_worker concurrent games. Pass the same `bank` across calls so maze ids
    stay consistent within an experiment. `aux_model` (e.g. the untrained control network) is evaluated on the
    same observations and its fc pre-activations recorded as "aux_fc_pre"; it never acts. If `on_game` is given,
    each finished record is handed to it (e.g. saved to disk) instead of being kept in memory.
    """
    seeds = list(seeds)
    bank = bank or M.MazeBank()
    n_slots = min(n_workers * games_per_worker, len(seeds))
    n_workers = min(n_workers, n_slots)
    pool = Pool(n_workers, sticky)
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
                            rec["seed"], rec["job"] = s["seed"], s["k"]
                            if on_game is not None:
                                on_game(rec)
                                rec = {"seed": rec["seed"], "score": float(rec["raw_reward"].sum()),
                                       "steps": len(rec["action"])}
                            records[s["k"]] = rec
                            log.info(f"[{desc}] game seed={s['seed']} score={sum(s['buf']['raw_reward']):.0f} "
                                     f"steps={len(s['buf']['action'])} ({len(records)}/{len(seeds)})")
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
            jobs = np.array([slots[sl]["k"] for sl in ready])
            edit = edit_factory(rams, steps, jobs) if edit_factory else None
            with torch.no_grad():
                out = model(obs, edit=edit)
            probs = torch.softmax(out["logits"].double(), -1).cpu().numpy()
            logits = out["logits"].float().cpu().numpy()
            value = out["value"].float().cpu().numpy()
            if record_acts:
                c3 = out["conv3"].half().cpu().numpy()
                fcp = out["fc_pre"].half().cpu().numpy()
                if aux_model is not None:
                    with torch.no_grad():
                        aux = aux_model(obs)["fc_pre"].half().cpu().numpy()
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
                    if aux_model is not None:
                        push(buf, "aux_fc_pre", aux[b])
                pending[sl % n_workers].append((sl, "step", a))
                s["state"] = None
    finally:
        pool.close()
    return [records[k] for k in range(len(seeds))], bank
