"""Open-loop screens on recorded discovery states (exact, fast, no environment needed).

Local:
- `patch_screen`: write each conv3 channel's per-cell dose into one 7x7 cell, then measure how the policy's
  probability of moving toward that cell's maze location changes ("attraction" > 0 = steered toward,
  < 0 = repelled). Exact rank-1 shortcut: fc_pre' = fc_pre + (v - x_old) * W_fc[:, flat index], since conv3 feeds
  the fc layer linearly.
- `ablation_screen`: zero each fc unit (logits shift by -h_n * W_pi[:, n]) and measure the change in
  P(move toward the nearest edible ghost / dangerous ghost / pellet) when that target exists.
Global: these screens only rank candidates. Behavioural claims require closed-loop steering (steer.py) on
held-out games.
"""

from __future__ import annotations

import numpy as np
import torch

from pacneuron import maze as M
from pacneuron import ram as R
from pacneuron.metrics import ACTION_DIRS, cell_target_node


def _toward_masks(S: dict, mazes: list, idx: np.ndarray) -> np.ndarray:
    """[N, 49, 9] bool: which actions move from the player's node toward each conv3 cell's target node."""
    ram = S["ram"][idx]
    xy = R.player_xy(ram)
    node = M.node_of(xy[:, 0], xy[:, 1])
    mz_ids = S["maze"][idx]
    out = np.zeros((len(idx), 49, 9), bool)
    targets = {m: [cell_target_node((c // 7, c % 7), mazes[m]) for c in range(49)] for m in np.unique(mz_ids)}
    for n in range(len(idx)):
        mz = mazes[mz_ids[n]]
        for c, tgt in enumerate(targets[mz_ids[n]]):
            dirs = mz.first_steps(int(node[n]), tgt)
            out[n, c] = (ACTION_DIRS & dirs).any(1)
    return out


@torch.no_grad()
def patch_screen(model, S: dict, mazes: list, doses: np.ndarray, n_states: int = 4000, seed: int = 0,
                 device: str = "cuda") -> dict:
    """Attraction effect of every (channel, cell) patch. doses: [64, 7, 7] value written at each cell.

    Returns {"attraction": [64,7,7] mean dP(toward cell), "attraction_junction": same at junction states,
    "kl": [64,7,7] mean KL(patched || base)}.
    """
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(S["game"]), size=min(n_states, len(S["game"])), replace=False))
    toward = torch.as_tensor(_toward_masks(S, mazes, idx), device=device, dtype=torch.float32)  # [N,49,9]
    has = toward.any(-1)  # target reachable and not already there
    junc = torch.as_tensor(S["features"]["junction"][idx] > 0, device=device)
    c3 = torch.as_tensor(S["conv3"][idx].astype(np.float32), device=device).flatten(1)  # [N, 3136] (C,H,W order)
    Wfc, bfc = model.fc.weight, model.fc.bias
    pre = c3 @ Wfc.T + bfc
    p0 = torch.softmax(model.policy(torch.relu(pre)), -1)
    att = torch.zeros(64, 49, device=device)
    att_j = torch.zeros(64, 49, device=device)
    kl = torch.zeros(64, 49, device=device)
    for c in range(64):
        for cell in range(49):
            k = c * 49 + cell  # flat index of (channel c, row cell//7, col cell%7)
            v = float(doses[c, cell // 7, cell % 7])
            p1 = torch.softmax(model.policy(torch.relu(pre + (v - c3[:, k:k + 1]) * Wfc[:, k])), -1)
            dt = ((p1 - p0) * toward[:, cell]).sum(-1)
            m = has[:, cell]
            att[c, cell] = dt[m].mean() if m.any() else 0
            att_j[c, cell] = dt[m & junc].mean() if (m & junc).any() else 0
            kl[c, cell] = (p1 * (torch.log(p1 + 1e-9) - torch.log(p0 + 1e-9))).sum(-1).mean()
    r = lambda x: x.view(64, 7, 7).cpu().numpy()
    return {"attraction": r(att), "attraction_junction": r(att_j), "kl": r(kl)}


@torch.no_grad()
def ablation_screen(model, S: dict, mazes: list, n_states: int = 20000, seed: int = 0, device: str = "cuda") -> dict:
    """Effect of zero-ablating each fc unit on P(move toward target) for edible ghosts, dangerous ghosts, pellets."""
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(S["game"]), size=min(n_states, len(S["game"])), replace=False))
    F = S["features"]
    masks = {}
    for tgt in ("edible", "danger", "pellet"):
        dirs = np.stack([F[f"dir_{tgt}_{d}"][idx] for d in M.DIRS], 1).astype(bool)  # [N,4]
        masks[tgt] = torch.as_tensor((ACTION_DIRS[None] & dirs[:, None]).any(-1), device=device, dtype=torch.float32)
    h = torch.relu(torch.as_tensor(S["fc_pre"][idx].astype(np.float32), device=device))
    logits = model.policy(h)
    p0 = torch.softmax(logits, -1)
    W = model.policy.weight  # [9, 512]
    out = {t: np.zeros(512) for t in masks}
    for n in range(512):
        p1 = torch.softmax(logits - h[:, n:n + 1] * W[:, n], -1)
        for t, m in masks.items():
            sel = m.any(-1)
            out[t][n] = float((((p1 - p0) * m).sum(-1))[sel].mean())
    return out
