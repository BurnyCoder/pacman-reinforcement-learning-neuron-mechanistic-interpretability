"""Attribution for individual units: where in the input and in which upstream units their activity comes from.

Local (all exact or model-based, no gradients):
- `fc_upstream`: an fc unit's pre-activation is exactly W_fc[n] · flatten(conv3) + b, so the contribution of each
  conv3 channel on a state is the sum of W_fc[n, channel block] * conv3[channel]; averaged over top states.
- `conv3_upstream`: a conv3 unit (c, i, j) pre-activation is exactly sum_k sum_ab W3[c,k,a,b] * conv2[k, i+a, j+b] + b3[c],
  so each conv2 channel's contribution is exact too.
- `occlusion`: replace a 12x12 patch (stride 6) of all four stacked frames with the normaliser's mean frame and
  record the drop in the unit's activation (Zeiler & Fergus 2014, https://arxiv.org/abs/1311.2901); averaged over
  the unit's top states -> a 13x13 "where it looks" map.
- `stratified_auroc`: AUROC of a unit for a binary label within strata (frightened x 3x3 player region), weighted
  by stratum size, so that confounds shared by the label and the strata cannot drive the score.
- `unique_r2`: drop-one-feature R² loss of the gradient-boosting ceiling (how much the feature adds beyond all
  other features).
Global: phases.attribution runs these for every frozen/derived steering candidate of an agent.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score

PATCH, STRIDE = 12, 6
GRID = (84 - PATCH) // STRIDE + 1  # 13 patch positions per axis


def fc_upstream(model, c3: np.ndarray, unit: int, top: int = 5) -> list[tuple[int, float]]:
    """Mean exact contribution of each conv3 channel to fc unit `unit` over states c3 [N,64,7,7]; top channels."""
    w = model.fc.weight[unit].detach().cpu().numpy().reshape(64, 49)
    contrib = (c3.reshape(len(c3), 64, 49).astype(np.float32) * w).sum(-1).mean(0)
    order = np.argsort(-np.abs(contrib))[:top]
    return [(int(k), float(contrib[k])) for k in order]


@torch.no_grad()
def conv3_upstream(model, obs: np.ndarray, channel: int, cell: tuple[int, int], top: int = 5) -> list[tuple[int, float]]:
    """Mean exact contribution of each conv2 channel to conv3 unit (channel, cell) over observations obs [N,4,84,84]."""
    dev = next(model.parameters()).device
    c2 = model(torch.as_tensor(obs, device=dev))["conv2"]  # [N,64,9,9]
    i, j = cell
    patch = c2[:, :, i:i + 3, j:j + 3]  # receptive field of conv3 (k=3, s=1) in conv2
    w = model.conv3.weight[channel]  # [64,3,3]
    contrib = (patch * w).sum((-1, -2)).mean(0).cpu().numpy()
    order = np.argsort(-np.abs(contrib))[:top]
    return [(int(k), float(contrib[k])) for k in order]


@torch.no_grad()
def occlusion(model, obs: np.ndarray, read) -> np.ndarray:
    """[13,13] mean drop of `read(out)` (a [N] activation) when a 12x12 patch of the input is set to the mean frame."""
    dev = next(model.parameters()).device
    x = torch.as_tensor(obs, device=dev)
    base = read(model(x))
    fill = (model.obs_mean.float() * 255).round().clamp(0, 255).to(torch.uint8)  # normaliser mean = "nothing here"
    out = np.zeros((GRID, GRID))
    for a in range(GRID):
        for b in range(GRID):
            xo = x.clone()
            ys, xs = slice(a * STRIDE, a * STRIDE + PATCH), slice(b * STRIDE, b * STRIDE + PATCH)
            xo[:, :, ys, xs] = fill[:, ys, xs]
            out[a, b] = float((base - read(model(xo))).mean())
    return out


def stratified_auroc(act: np.ndarray, label: np.ndarray, strata: np.ndarray, min_n: int = 30) -> float:
    """Size-weighted mean AUROC of `act` for binary `label` within strata that contain both classes."""
    vals, wts = [], []
    for s in np.unique(strata):
        m = strata == s
        y = label[m] > 0
        if y.sum() >= min_n and (~y).sum() >= min_n:
            vals.append(roc_auc_score(y, act[m]))
            wts.append(m.sum())
    return float(np.average(vals, weights=wts)) if vals else float("nan")


def unique_r2(act: np.ndarray, feats: dict[str, np.ndarray], game: np.ndarray, drop: list[str],
              max_fit: int = 40000, seed: int = 0) -> dict[str, float]:
    """Split-half (even/odd games) R² loss of HistGradientBoosting when each feature in `drop` is removed."""
    names = list(feats)
    X = np.stack([feats[k] for k in names], 1)
    fit, ev = game % 2 == 0, game % 2 == 1
    rng = np.random.default_rng(seed)
    sub = rng.choice(np.flatnonzero(fit), size=min(max_fit, int(fit.sum())), replace=False)

    def r2(cols):
        m = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(X[sub][:, cols], act[sub])
        resid = act[ev] - m.predict(X[ev][:, cols])
        return 1 - (resid ** 2).sum() / ((act[ev] - act[ev].mean()) ** 2).sum()

    full = r2(list(range(len(names))))
    out = {"r2_all": float(full)}
    for f in drop:
        out[f] = float(full - r2([k for k, n in enumerate(names) if n != f]))
    return out
