"""Analysis phases called by pipeline.py (each writes results/<phase>_<model>.json and bulk arrays to DATA_DIR).

Local: stochasticity (do games diverge?), audit (exploratory neuron labelling on discovery split-halves).
Global: the audit is the "microscope" (Goodfire-style per-neuron poles, labels, attribution); its labels feed the
patch screen and candidate selection; confirmatory statistics live in later phases on held-out games.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import rankdata

from pacneuron import audit as AU
from pacneuron import data
from pacneuron.config import DATA_DIR, MODELS
from pacneuron.features import CELL_TYPES, FAMILY, cell_objects
from pacneuron.logs import get_logger
from pacneuron.pipeline import _model, write_result
from pacneuron.stats import bootstrap_weights

log = get_logger("pacneuron.phases")
ACTIONS = ["NOOP", "UP", "RIGHT", "LEFT", "DOWN", "UPRIGHT", "UPLEFT", "DOWNRIGHT", "DOWNLEFT"]


def stochasticity(seed: str) -> None:
    """Preregistered check (§3): for discovery games starting from the same state, when do their RAMs diverge?"""
    games = data.load_games(data.split_dir(MODELS[seed].name, "discovery"), ("ram",))
    groups: dict[bytes, list] = {}
    for g in games:
        groups.setdefault(g["ram"][0].tobytes(), []).append(g["ram"])
    div = []
    for rams in groups.values():
        for a in range(len(rams)):
            for b in range(a + 1, len(rams)):
                n = min(len(rams[a]), len(rams[b]))
                diff = np.flatnonzero((rams[a][:n] != rams[b][:n]).any(1))
                div.append(int(diff[0]) if len(diff) else n)
    med = float(np.median(div)) if div else float("nan")
    write_result(f"stochasticity_{MODELS[seed].name}", {
        "distinct_start_states": len(groups), "pairs": len(div), "divergence_steps": div,
        "median_divergence": med, "sticky_fallback_triggered": bool(med > 200)})
    log.info(f"{len(groups)} distinct start states, {len(div)} same-start pairs, median divergence {med} steps")


def _cell_auroc(act: np.ndarray, labels: dict[str, np.ndarray], min_pos: int = 30) -> dict[str, float]:
    """Within-cell AUROC of one channel's activation for each object type, averaged over cells (location-controlled).

    act: [N, 7, 7]; labels[type]: [N, 7, 7] bool. AUROC per cell via the Mann-Whitney rank formula (ties averaged).
    """
    out = {}
    ranks = np.empty_like(act, dtype=np.float64)
    for i in range(7):
        for j in range(7):
            ranks[:, i, j] = rankdata(act[:, i, j])
    for typ, lab in labels.items():
        vals, wts = [], []
        for i in range(7):
            for j in range(7):
                y = lab[:, i, j] > 0
                npos, nneg = y.sum(), (~y).sum()
                if npos >= min_pos and nneg >= min_pos:
                    vals.append((ranks[y, i, j].sum() - npos * (npos + 1) / 2) / (npos * nneg))
                    wts.append(npos)
        out[typ] = float(np.average(vals, weights=wts)) if vals else float("nan")
    return out


def audit(seed: str, n_boot: int = 2000) -> None:
    """Exploratory discovery audit of all fc units (+ controls) and conv3 channels, using split-half games."""
    name = MODELS[seed].name
    S = data.load_split(data.split_dir(name, "discovery"))
    model = _model(seed)
    F, game = S["features"], S["game"]
    fit, ev = game % 2 == 0, game % 2 == 1
    ge = game[ev] // 2
    n_ev = int(ge.max()) + 1
    sub = lambda d, m: {k: v[m] for k, v in d.items()}
    Ff, Fe = sub(F, fit), sub(F, ev)

    fc_pre = S["fc_pre"].astype(np.float32)
    fc = np.maximum(fc_pre, 0)
    alive = AU.alive(fc)
    alive_idx = np.flatnonzero(alive)
    ctrl = np.maximum(S["aux_fc_pre"].astype(np.float32), 0)
    sets = {
        "agent": fc[:, alive],
        "random_dirs": fc @ AU.random_directions(fc.shape[1]),
        "nonneg_mix": fc @ AU.nonneg_mixtures(alive_idx, fc.shape[1]),
        "untrained": ctrl[:, AU.alive(ctrl)],
    }
    r2, summary = {}, {}
    for k, A in sets.items():
        log.info(f"audit {name}: R² for set {k} ({A.shape[1]} units)")
        s1, names = AU.r2_single(A[fit], Ff, A[ev], Fe, ge, n_ev)
        sall = AU.r2_ceiling(A[fit], Ff, A[ev], Fe, ge, n_ev)
        r2[k] = (s1, sall)
    W = bootstrap_weights(n_ev, n_boot)
    clean_boot = {}
    for k, (s1, sall) in r2.items():
        clean_boot[k] = np.array([AU.clean_mask(s1.r2(w), sall.r2(w)).mean() for w in W])
        summary[k] = {"units": int(s1.sse.shape[-1]), "clean_frac": float(AU.clean_mask(s1.r2(), sall.r2()).mean()),
                      "median_r2_all": float(np.median(sall.r2())), "median_best_r2": float(np.median(s1.r2().max(0)))}
    for k in ("random_dirs", "nonneg_mix", "untrained"):
        d = clean_boot["agent"] - clean_boot[k]
        summary["agent"][f"diff_vs_{k}"] = [float(d.mean()), *map(float, np.quantile(d, [0.025, 0.975]))]

    # per-unit table (agent fc, all 512 units; dead units flagged)
    s1, sall = r2["agent"]
    R1, RA = s1.r2(), sall.r2()
    Wpi = model.policy.weight.detach().cpu().numpy()
    wv = model.critic.weight.detach().cpu().numpy()[0]
    clean = AU.clean_mask(R1, RA)
    units = []
    rng = np.random.default_rng(0)
    for n in range(fc.shape[1]):
        u = {"unit": n, "alive": bool(alive[n]), "frac_active": float((fc[:, n] > 0).mean()),
             "p99": float(np.quantile(fc[fc[:, n] > 0, n], 0.99)) if alive[n] else 0.0,
             "max": float(fc[:, n].max()), "dla": Wpi[:, n].tolist(), "w_value": float(wv[n]),
             "top": [[int(S["seeds"][game[i]]), int(S["t"][i])] for i in AU.top_examples(fc[:, n])],
             "neg": [[int(S["seeds"][game[i]]), int(S["t"][i])] for i in AU.top_examples(-fc_pre[:, n])],
             "bands": {b: [[int(S["seeds"][game[i]]), int(S["t"][i])] for i in ix]
                       for b, ix in AU.quantile_band_examples(fc[:, n], rng).items()}}
        if alive[n]:
            col = int(np.searchsorted(alive_idx, n))
            order = np.argsort(-R1[:, col])
            best, second = names[order[0]], names[order[1]]
            sign = np.sign(np.corrcoef(fc[fit, n], Ff[best])[0, 1]) if Ff[best].std() > 0 else 0.0
            u |= {"best": best, "family": FAMILY[best], "r2_best": float(R1[order[0], col]), "second": second,
                  "r2_second": float(R1[order[1], col]), "r2_all": float(RA[col]), "clean": bool(clean[col]),
                  "sign": float(sign), "r2_by_feature": {nm: float(R1[i, col]) for i, nm in enumerate(names)}}
        units.append(u)

    # conv3 channels: within-cell AUROC for object types (location-controlled) + per-cell dose (p99 when active)
    log.info(f"audit {name}: conv3 channel AUROCs")
    c3 = S["conv3"]
    objs = cell_objects(S["ram"])
    pick = rng.choice(len(c3), size=min(60000, len(c3)), replace=False)
    labs = {k: v[pick] for k, v in objs.items()}
    channels = []
    for c in range(c3.shape[1]):
        a = c3[:, c].astype(np.float32)
        act = a > 0
        p99 = np.where(act.sum(0) > 20, np.nanquantile(np.where(act, a, np.nan), 0.99, axis=0), 0.0)
        channels.append({"channel": c, "frac_active": float(act.mean()), "auroc": _cell_auroc(a[pick], labs),
                         "p99_cell": np.nan_to_num(p99).round(4).tolist(), "max": float(a.max())})
    write_result(f"audit_{name}", {"model": name, "features": names, "summary": summary, "fc_units": units,
                                   "conv3_channels": channels, "n_steps": int(len(game)), "n_games": int(game.max() + 1),
                                   "actions": ACTIONS})
    out = DATA_DIR / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"audit_{name}.npz", **{f"{k}_r2_single": v[0].r2() for k, v in r2.items()},
             **{f"{k}_r2_all": v[1].r2() for k, v in r2.items()}, **{f"{k}_clean_boot": v for k, v in clean_boot.items()})
    log.info(f"audit {name}: summary {summary}")


def screen(seed: str) -> None:
    """Open-loop screens (discovery only): conv3 cell patches (attraction/repulsion) and fc ablations."""
    import json

    from pacneuron import screen as SC
    from pacneuron.config import RESULTS_DIR
    name = MODELS[seed].name
    d = data.split_dir(name, "discovery")
    S = data.load_split(d, acts=("fc_pre", "conv3"))
    mazes = data.load_bank(d).mazes
    audit_res = json.loads((RESULTS_DIR / f"audit_{name}.json").read_text())
    doses = np.array([ch["p99_cell"] for ch in audit_res["conv3_channels"]], np.float32)  # [64, 7, 7]
    model = _model(seed)
    log.info(f"screen {name}: conv3 patch screen over 64x49 patches")
    ps = SC.patch_screen(model, S, mazes, doses)
    log.info(f"screen {name}: fc ablation screen")
    ab = SC.ablation_screen(model, S, mazes)
    out = DATA_DIR / "analysis"
    np.savez(out / f"screen_{name}.npz", **ps, **{f"ablate_{k}": v for k, v in ab.items()})
    att = ps["attraction_junction"]
    order = np.argsort(att.ravel())
    pick = lambda ix: [{"channel": int(i // 49), "cell": [int(i % 49 // 7), int(i % 7)],
                        "attraction_junction": float(att.ravel()[i]), "attraction": float(ps["attraction"].ravel()[i]),
                        "dose": float(doses.ravel()[i])} for i in ix]
    per_channel = {"mean_attraction_junction": att.mean((1, 2)).tolist(),
                   "mean_abs_attraction_junction": np.abs(att).mean((1, 2)).tolist()}
    write_result(f"screen_{name}", {
        "model": name, "most_repelling": pick(order[:20]), "most_attracting": pick(order[::-1][:20]),
        "per_channel": per_channel,
        "fc_ablation": {k: {"most_negative": [[int(n), float(v[n])] for n in np.argsort(v)[:15]],
                            "most_positive": [[int(n), float(v[n])] for n in np.argsort(v)[::-1][:15]]}
                        for k, v in ab.items()}})
