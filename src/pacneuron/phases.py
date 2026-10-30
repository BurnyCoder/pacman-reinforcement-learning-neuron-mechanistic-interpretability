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
                      "ci": np.quantile(clean_boot[k], [0.025, 0.975]).tolist(),
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


def _paired_effect(base: dict, arm: dict, metric: str, sign: int) -> dict:
    """Paired (same game seed) difference of a metric between an arm and the baseline, with CI and p-value."""
    from pacneuron.stats import mean_ci, paired_signflip_p
    seeds = [s for s in arm if s in base and np.isfinite(arm[s].get(metric, np.nan))
             and np.isfinite(base[s].get(metric, np.nan))]
    d = np.array([arm[s][metric] - base[s][metric] for s in seeds])
    if len(d) < 3:
        return {"n": len(d)}
    m, lo, hi = mean_ci(d)
    return {"n": len(d), "delta": m, "ci": [lo, hi], "p_one_sided": paired_signflip_p(d, sign),
            "base_mean": float(np.mean([base[s][metric] for s in seeds])),
            "arm_mean": float(np.mean([arm[s][metric] for s in seeds]))}


def pilot(seed: str, n_games: int = 30) -> None:
    """Exploratory closed-loop pilots on DISCOVERY seeds for the specs in results/pilot_specs_<model>.json.

    Each spec: {"name", "spec": intervention, "metric": name, "sign": +1/-1, "metric_kwargs": {...}}. The baseline is
    the discovery collection itself (same seeds, same sticky-action RNG, unsteered) -> paired comparison.
    """
    import json

    from pacneuron import steer
    from pacneuron.config import DISCOVERY_SEEDS, RESULTS_DIR
    name = MODELS[seed].name
    specs = json.loads((RESULTS_DIR / f"pilot_specs_{name}.json").read_text())
    model = _model(seed)
    seeds = DISCOVERY_SEEDS[:n_games]
    base_dir = data.split_dir(name, "discovery")
    results = []
    for c in specs:
        arm_dir = DATA_DIR / "steer" / name / "pilot" / c["name"]
        if not (arm_dir / "mazes.pkl").exists():
            steer.run_arm(model, seeds, c["spec"], arm_dir, desc=f"pilot/{c['name']}")
        kw = c.get("metric_kwargs", {})
        arm = steer.arm_metrics(arm_dir, **kw)
        base = {s: m for s, m in steer.arm_metrics(base_dir, **kw).items() if s in arm}
        eff = {mt: _paired_effect(base, arm, mt, c["sign"] if mt == c["metric"] else 1)
               for mt in [c["metric"], "return", "deaths_per_1k", "pellets_per_1k", "ghosts_per_pill"]}
        results.append({"name": c["name"], "spec": c["spec"], "metric": c["metric"], "sign": c["sign"], "effects": eff})
        log.info(f"pilot {c['name']}: {c['metric']} delta={eff[c['metric']].get('delta')} ci={eff[c['metric']].get('ci')}")
    write_result(f"pilot_{name}", {"model": name, "split": "discovery", "n_games": n_games, "candidates": results})


def _start_groups(S: dict) -> np.ndarray:
    """[G] id of each game's start state (identical first in-control RAM): robustness clusters for games that
    share a start (they follow the same trajectory until sticky actions/sampling make them diverge)."""
    keys: dict[bytes, int] = {}
    ids = [keys.setdefault(S["ram"][np.flatnonzero(S["game"] == gi)[0]].tobytes(), len(keys))
           for gi in range(int(S["game"].max()) + 1)]
    return np.array(ids)


def h1(seed: str, n_boot: int = 10000) -> None:
    """Confirmatory H1 (preregistration §5): fit on all discovery games, score on held-out games."""
    from pacneuron.prereg import require_frozen
    from pacneuron.stats import bootstrap_weights
    lock = require_frozen()
    name = MODELS[seed].name
    Sf = data.load_split(data.split_dir(name, "discovery"), acts=("fc_pre", "aux_fc_pre"))
    Se = data.load_split(data.split_dir(name, "heldout"), acts=("fc_pre", "aux_fc_pre"))
    relu = lambda x: np.maximum(x.astype(np.float32), 0)
    fc_f, fc_e, ct_f, ct_e = relu(Sf["fc_pre"]), relu(Se["fc_pre"]), relu(Sf["aux_fc_pre"]), relu(Se["aux_fc_pre"])
    alive, alive_c = AU.alive(fc_f), AU.alive(ct_f)
    Q, Wmix = AU.random_directions(fc_f.shape[1]), AU.nonneg_mixtures(np.flatnonzero(alive), fc_f.shape[1])
    sets = {"agent": (fc_f[:, alive], fc_e[:, alive]), "random_dirs": (fc_f @ Q, fc_e @ Q),
            "nonneg_mix": (fc_f @ Wmix, fc_e @ Wmix), "untrained": (ct_f[:, alive_c], ct_e[:, alive_c])}
    ge, n_ev = Se["game"], int(Se["game"].max()) + 1
    stats = {}
    for k, (Af, Ae) in sets.items():
        log.info(f"h1 {name}: {k} ({Af.shape[1]} units)")
        stats[k] = (AU.r2_single(Af, Sf["features"], Ae, Se["features"], ge, n_ev)[0],
                    AU.r2_ceiling(Af, Sf["features"], Ae, Se["features"], ge, n_ev))
    W = bootstrap_weights(n_ev, n_boot, seed=1)
    groups = _start_groups(Se)
    Wg = bootstrap_weights(int(groups.max()) + 1, 2000, seed=2)[:, groups]  # resample start-state clusters
    out = {"lock": lock, "n_heldout_games": n_ev, "n_start_groups": int(groups.max()) + 1, "sets": {}}
    boots = {}
    for k, (s1, sa) in stats.items():
        boots[k] = np.array([AU.clean_mask(s1.r2(w), sa.r2(w)).mean() for w in W])
        bg = np.array([AU.clean_mask(s1.r2(w), sa.r2(w)).mean() for w in Wg])
        r1, ra = s1.r2(), sa.r2()
        sweep = {f"{ta:.1f}_{tr:.1f}": float(AU.clean_mask(r1, ra, ta, tr).mean())
                 for ta in (0.1, 0.2, 0.3, 0.4, 0.5) for tr in (0.5, 0.6, 0.7, 0.8, 0.9)}
        out["sets"][k] = {"units": int(r1.shape[1]), "clean_frac": float(AU.clean_mask(r1, ra).mean()),
                          "ci": np.quantile(boots[k], [0.025, 0.975]).tolist(), "ci_start_groups":
                          np.quantile(bg, [0.025, 0.975]).tolist(), "threshold_sweep": sweep,
                          "median_r2_all": float(np.median(ra)), "median_best_r2": float(np.median(r1.max(0))),
                          "r2_all": ra.tolist(), "best_r2": r1.max(0).tolist()}
    for k in ("random_dirs", "nonneg_mix", "untrained"):
        d = boots["agent"] - boots[k]
        out["sets"]["agent"][f"diff_vs_{k}"] = {"mean": float(d.mean()), "ci": np.quantile(d, [0.025, 0.975]).tolist()}
    ci = out["sets"]["agent"]["diff_vs_random_dirs"]["ci"]
    out["H1_supported"] = bool(ci[0] > 0)
    write_result(f"h1_{name}", out)


def confirm(seed: str) -> None:
    """Confirmatory steering (preregistration §6) for the frozen candidates on held-out games."""
    import json

    from pacneuron import steer
    from pacneuron.config import HELDOUT_SEEDS, RESULTS_DIR
    from pacneuron.prereg import require_frozen
    from pacneuron.stats import holm
    name = MODELS[seed].name
    derived = RESULTS_DIR / f"candidates_{name}.json"  # replication agents: rule-derived candidates (H3)
    lock = require_frozen([f"results/{derived.name}"] if derived.exists() else [])
    frozen = json.loads((derived if derived.exists() else RESULTS_DIR / "candidates.json").read_text())
    cands = [c for c in frozen["candidates"] if c.get("model", "appo_1111") == name]
    model = _model(seed)
    root = DATA_DIR / "steer" / name / "heldout"
    base_dir = data.split_dir(name, "heldout")  # unsteered held-out collection (seeds 10000-10099) = paired baseline
    if not (base_dir / "mazes.pkl").exists():
        raise RuntimeError("run --phase collect-heldout first (it is the baseline arm)")
    # play every missing arm (candidates + their null units) in one batched worker pool
    todo = []
    for c in cands:
        if not ((root / c["name"]) / "mazes.pkl").exists():
            todo.append((root / c["name"], c["spec"], list(HELDOUT_SEEDS[:frozen["n_candidate"]])))
        for k, ns in enumerate(c["null_specs"]):
            nd = root / f"null_{c['null_group']}_{k:03d}"
            if not (nd / "mazes.pkl").exists() and nd not in [t[0] for t in todo]:
                todo.append((nd, ns, list(HELDOUT_SEEDS[:frozen["n_null"]])))
    if todo:
        log.info(f"confirm {name}: playing {len(todo)} arms ({sum(len(t[2]) for t in todo)} games) in one pool")
        steer.run_arms(model, todo, desc=f"heldout/{name}")
    results = []
    for c in cands:
        kw = c.get("metric_kwargs", {})
        arm_dir = root / c["name"]
        base = steer.arm_metrics(base_dir, **kw)
        arm = steer.arm_metrics(arm_dir, **kw)
        eff = {mt: _paired_effect(base, arm, mt, c["sign"] if mt == c["metric"] else 1)
               for mt in [c["metric"], "return", "deaths_per_1k", "pellets_per_1k", "ghosts_per_pill"]}
        null_deltas = []
        for k, ns in enumerate(c["null_specs"]):
            nd = root / f"null_{c['null_group']}_{k:03d}"
            nm = steer.arm_metrics(nd, **kw)
            e = _paired_effect(base, nm, c["metric"], c["sign"])
            if "delta" in e:
                null_deltas.append(e["delta"])
        signed = c["sign"] * np.array(null_deltas)
        thr = float(np.quantile(signed, 0.95)) if len(signed) else float("nan")
        sec = None
        if "secondary_metric" in c:
            sm = c["secondary_metric"]
            sec = _paired_effect(steer.arm_metrics(base_dir, **sm["metric_kwargs"]),
                                 steer.arm_metrics(arm_dir, **sm["metric_kwargs"]), sm["metric"], sm["sign"])
        results.append({"name": c["name"], "label": c.get("label", c["name"]), "spec": c["spec"], "metric": c["metric"],
                        "sign": c["sign"], "effects": eff, "secondary": sec, "null_deltas": null_deltas,
                        "null_q95_signed": thr, "specific": bool(c["sign"] * eff[c["metric"]].get("delta", np.nan) > thr)})
    p = [r["effects"][r["metric"]].get("p_one_sided", 1.0) for r in results]
    for r, padj in zip(results, holm(p) if p else []):
        e = r["effects"][r["metric"]]
        r["p_holm"] = float(padj)
        r["passes_H2"] = bool(padj < 0.05 and r["sign"] * e.get("delta", 0) > 0 and r["specific"])
    pc = frozen.get("positive_control")
    pc_res = None
    if pc and name == "appo_1111":
        pc_dir = root / pc["name"]
        if not (pc_dir / "mazes.pkl").exists():
            steer.run_arm(model, HELDOUT_SEEDS[:frozen["n_candidate"]], pc["spec"], pc_dir, desc=f"heldout/{pc['name']}")
        pc_res = {"label": pc["label"], "metric": pc["metric"],
                  "effect": _paired_effect(steer.arm_metrics(base_dir, **pc["metric_kwargs"]),
                                           steer.arm_metrics(pc_dir, **pc["metric_kwargs"]), pc["metric"], pc["sign"])}
    write_result(f"confirm_{name}", {"lock": lock, "candidates": results, "positive_control": pc_res,
                                    "H2_supported": any(r["passes_H2"] for r in results)})


def fidelity(seed: str) -> None:
    """Fidelity numbers vs the golden Sample Factory trajectory (tests/test_fidelity.py asserts the same)."""
    from pacneuron.config import REPO_ROOT
    from pacneuron.env import Game
    g = dict(np.load(REPO_ROOT / "tests" / "fixtures" / "sf_golden.npz"))
    game, mismatched = Game(int(g["seed"]), sticky=0.0), 0
    for t, a in enumerate(g["actions"]):
        mismatched += int(not np.array_equal(game.obs, g["obs"][t]))
        _, _, over = game.step(int(a))
        if over:
            break
    model = _model(seed).cpu()
    with torch.no_grad():
        out = model(torch.as_tensor(g["obs"]))
    write_result("fidelity", {"golden_steps": int(len(g["obs"])), "obs_mismatched_steps": mismatched,
                              "life_losses": int(g["life_lost"].sum()), "game_over": bool(g["game_over"].any()),
                              "max_abs_logit_diff": float(np.abs(out["logits"].numpy() - g["logits"]).max()),
                              "max_abs_value_diff": float(np.abs(out["value"].numpy() - g["values"].reshape(-1)).max()),
                              "reference_stack": "sample-factory 2.0.3, gym 0.26.2, ale-py 0.8.1, torch 1.13.1 (WSL)"})


def ground_truth(seed: str) -> None:
    """Validation of the RAM ground truth on the discovery games (pixels vs RAM)."""
    from pacneuron import maze as M
    from pacneuron import ram as R
    from pacneuron.features import in_control
    d = data.split_dir(MODELS[seed].name, "discovery")
    games = data.load_games(d, ("ram", "blue", "maze"))
    agree_edible, agree_timer, eat_hits, eat_total, ctrl, n = 0, 0, 0, 0, 0, 0
    for g in games:
        ram = g["ram"]
        ed = R.ghost_edible(ram) & R.ghosts_active(ram)
        agree_edible += int(((g["blue"] >= 4) == ed).sum())
        agree_timer += int(((R.frightened_timer(ram) > 0) == R.ghost_edible(ram).any(1)).sum())
        pel, xy = R.pellets(ram), R.player_xy(ram)
        node = M.node_of(xy[:, 0], xy[:, 1])
        for t in range(1, len(ram)):
            for i, j in zip(*np.where(pel[t - 1] & ~pel[t])):
                eat_total += 1
                eat_hits += int(i * M.COLS + j in (node[t - 1], node[t]))
        ctrl += int(in_control(ram).sum())
        n += len(ram)
    write_result(f"ground_truth_{MODELS[seed].name}", {
        "steps": n, "games": len(games), "edible_bit_vs_blue_pixels_agreement": agree_edible / (4 * n),
        "timer_vs_any_edible_agreement": agree_timer / n, "pellet_eaten_at_player_node": eat_hits / eat_total,
        "pellet_events": eat_total, "in_control_fraction": ctrl / n, "maze_layouts": len(data.load_bank(d).mazes)})


def conjunctions(seed: str) -> None:
    """EXPLORATORY (post hoc): are fc units explained by conjunctions (place x heading ...) better than single features?"""
    name = MODELS[seed].name
    S = data.load_split(data.split_dir(name, "discovery"), acts=("fc_pre",))
    F, g = S["features"], S["game"]
    fc = np.maximum(S["fc_pre"].astype(np.float32), 0)
    A = fc[:, AU.alive(fc)]
    fit, ev = g % 2 == 0, g % 2 == 1

    def r2_bins(idx):
        nb = int(idx.max()) + 1
        cnt = np.bincount(idx[fit], minlength=nb).astype(np.float64)
        s = np.asarray(AU._onehot(idx[fit], nb) @ A[fit], np.float64)
        mu = np.where(cnt[:, None] > 0, s / np.maximum(cnt, 1)[:, None], A[fit].mean(0))
        y = A[ev]
        return 1 - ((y - mu[idx[ev]]) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)

    from pacneuron import maze as M
    from pacneuron import ram as R
    xy = R.player_xy(S["ram"])
    node = M.node_of(xy[:, 0], xy[:, 1])
    head = F["heading"].astype(int)
    fr = F["frightened"].astype(int)
    q = lambda v, n: np.searchsorted(np.unique(np.quantile(v, np.linspace(0, 1, n + 1)[1:-1])), v, side="right")
    dirs = (F["dir_danger_up"] + 2 * F["dir_danger_right"] + 4 * F["dir_danger_left"] + 8 * F["dir_danger_down"]).astype(int)
    res = {"place (lattice node)": r2_bins(node), "place x heading": r2_bins(node * 4 + head),
           "place x frightened": r2_bins(node * 2 + fr), "pellets left (20 bins) x frightened": r2_bins(q(F["n_pellets"], 20) * 2 + fr),
           "danger distance (20 bins) x danger direction": r2_bins(q(F["d_danger1"], 20) * 16 + dirs)}
    out = {k: {"median_r2": float(np.median(v)), "frac_gt_0.3": float(np.mean(v > 0.3)), "frac_gt_0.5": float(np.mean(v > 0.5))}
           for k, v in res.items()}
    write_result(f"conjunctions_{name}", {"exploratory": True, "units": int(A.shape[1]), "models": out})


def report(seed: str) -> None:
    """Regenerate all tables, inline numbers and figures from results/*.json."""
    from pacneuron.report import generate
    generate()
    log.info("report: tables, numbers and figures regenerated")


def site(seed: str) -> None:
    """Export the microscope data (frames + JSON) for the GitHub Pages site."""
    from pacneuron.export import export_site
    export_site(seed)


def conv3_states(seed: str) -> None:
    """EXPLORATORY: AUROC of each conv3 channel's spatial-mean activation for global game states (discovery)."""
    from sklearn.metrics import roc_auc_score
    name = MODELS[seed].name
    S = data.load_split(data.split_dir(name, "discovery"), acts=("conv3",))
    idx = np.random.default_rng(0).choice(len(S["game"]), size=min(60000, len(S["game"])), replace=False)
    m = S["conv3"][idx].astype(np.float32).mean((2, 3))
    out = {}
    for f in ("frightened", "n_danger_near", "life_lost_soon"):
        y = S["features"][f][idx] > 0
        out[f] = {"positive_rate": float(y.mean()), "auroc": [float(roc_auc_score(y, m[:, c])) for c in range(m.shape[1])]}
    write_result(f"conv3_states_{name}", {"exploratory": True, "n_steps": int(len(idx)), "states": out})


def videos(seed: str, game_seed: int = 10000, n_steps: int = 500) -> None:
    """Side-by-side replays (baseline | steered, same game seed) for every frozen candidate of this model."""
    import json

    from pacneuron.config import FIGURES_DIR, REPO_ROOT, RESULTS_DIR
    from pacneuron.render import capture_frames
    from pacneuron.viz import side_by_side_video
    name = MODELS[seed].name
    conf = json.loads((RESULTS_DIR / f"confirm_{name}.json").read_text())
    base_dir = data.split_dir(name, "heldout")
    steps = [(game_seed, t) for t in range(n_steps)]
    base = capture_frames(base_dir, steps)
    for c in conf["candidates"]:
        arm = capture_frames(DATA_DIR / "steer" / name / "heldout" / c["name"], steps)
        left = [base[k] for k in steps if k in base]
        right = [arm[k] for k in steps if k in arm]
        box = None
        if c["spec"].get("cell") is not None:
            i, j = c["spec"]["cell"]
            box = ((8 * j + 10) * 160 / 84, (8 * i + 10) * 210 / 84, (8 * j + 26) * 160 / 84, (8 * i + 26) * 210 / 84)
        out = FIGURES_DIR / f"video_{name}_{c['name']}"
        side_by_side_video(left, right, out, labels=("baseline", c["name"]), box=box)
        media = REPO_ROOT / "site" / "media"
        media.mkdir(parents=True, exist_ok=True)
        (media / f"{name}_{c['name']}.gif").write_bytes(out.with_suffix(".gif").read_bytes())
        log.info(f"videos: {out}.mp4/.gif ({min(len(left), len(right))} steps)")


def derive_candidates(seed: str) -> None:
    """H3: apply the frozen replication rules (results/candidates.json) to a new agent's DISCOVERY results.

    Writes results/candidates_<model>.json; it must be committed before that agent's held-out phases run.
    """
    import json

    from pacneuron.config import RESULTS_DIR
    from pacneuron.features import FAMILY  # noqa: F401  (documents that labels come from the same feature set)
    name = MODELS[seed].name
    frozen = json.loads((RESULTS_DIR / "candidates.json").read_text())
    audit_res = json.loads((RESULTS_DIR / f"audit_{name}.json").read_text())
    scr = json.loads((RESULTS_DIR / f"screen_{name}.json").read_text())
    units = {u["unit"]: u for u in audit_res["fc_units"]}
    # rule 1: most negative fc ablation-screen effect on P(toward nearest edible ghost)
    n = int(scr["fc_ablation"]["edible"]["most_negative"][0][0])
    dirs = {"up": [1, 5, 6], "right": [2, 5, 7], "left": [3, 6, 8], "down": [4, 7, 8]}
    best_dir = max(dirs, key=lambda d: units[n]["r2_by_feature"][f"dir_edible_{d}"])
    alive = [u["unit"] for u in audit_res["fc_units"] if u["alive"] and u["unit"] != n]
    null_fc = sorted(np.random.default_rng(0).choice(alive, 99, replace=False).tolist())
    # rule 2: most repelling (channel, cell) of the open-loop patch screen at the p99 dose
    top = scr["most_repelling"][0]
    ch, cell = int(top["channel"]), [int(x) for x in top["cell"]]
    p99 = lambda c: float(audit_res["conv3_channels"][c]["p99_cell"][cell[0]][cell[1]])
    cands = [{"name": f"fc{n}_ablate", "rule": "fc453_ablate", "label": f"fc neuron {n} zero-ablated", "model": name,
              "spec": {"site": "fc", "units": [n], "value": 0.0}, "metric": "ghosts_per_pill", "sign": -1,
              "secondary_metric": {"metric": "action_given_trigger", "sign": -1,
                                   "metric_kwargs": {"trigger": f"dir_edible_{best_dir}", "action_set": dirs[best_dir]}},
              "null_group": "fc_ablate", "null_specs": [{"site": "fc", "units": [int(u)], "value": 0.0} for u in null_fc]},
             {"name": f"c3_ch{ch}_cell{cell[0]}{cell[1]}", "rule": "c3_ch18_cell56",
              "label": f"conv3 channel {ch} set to its p99 at cell ({cell[0]}, {cell[1]})", "model": name,
              "spec": {"site": "conv3", "units": [ch], "cell": cell, "value": p99(ch)}, "metric": "approach", "sign": -1,
              "metric_kwargs": {"target_cell": cell}, "null_group": f"c3_cell{cell[0]}{cell[1]}",
              "null_specs": [{"site": "conv3", "units": [c], "cell": cell, "value": p99(c)} for c in range(64) if c != ch]}]
    out = {k: frozen[k] for k in ("n_baseline", "n_candidate", "n_null")} | {
        "derived_by": "replication_rules in results/candidates.json (no human choice)", "candidates": cands}
    (RESULTS_DIR / f"candidates_{name}.json").write_text(json.dumps(out, indent=1))
    log.info(f"derive_candidates {name}: {[c['label'] for c in cands]}")


def publish_hf(seed: str) -> None:
    """Upload derived artefacts to the HF dataset and update the study's HF collection."""
    from pacneuron.hf_publish import publish
    url = publish()
    write_result("hf_publish", {"collection": url, "dataset": "https://huggingface.co/datasets/BurnyCoder/mspacman-appo-neuron-atlas"})
    log.info(f"published: {url}")


def occupancy(seed: str) -> None:
    """DESCRIPTIVE figure data: where Ms. Pac-Man spends in-control steps, baseline vs each frozen conv3 candidate.

    Counts player positions on the 14x18 lattice over the candidate's held-out games (paired seeds), normalised to
    a probability map; writes results/occupancy_<model>.json (maps + per-cell difference) for the figure.
    """
    import json

    from pacneuron import maze as M
    from pacneuron import ram as R
    from pacneuron.config import RESULTS_DIR
    from pacneuron.features import in_control
    name = MODELS[seed].name
    conf = json.loads((RESULTS_DIR / f"confirm_{name}.json").read_text())
    out = {}
    for c in conf["candidates"]:
        if c["spec"]["site"] != "conv3":
            continue
        arm_dir = DATA_DIR / "steer" / name / "heldout" / c["name"]
        seeds = {g["seed"] for g in data.load_games(arm_dir, ("ram",))}
        maps = {}
        for label, d in (("baseline", data.split_dir(name, "heldout")), ("steered", arm_dir)):
            occ = np.zeros(M.ROWS * M.COLS)
            for g in data.load_games(d, ("ram",)):
                if g["seed"] in seeds:
                    xy = R.player_xy(g["ram"][in_control(g["ram"])])
                    occ += np.bincount(M.node_of(xy[:, 0], xy[:, 1]), minlength=occ.size)
            maps[label] = (occ / occ.sum()).reshape(M.ROWS, M.COLS).tolist()
        out[c["name"]] = {"cell": c["spec"]["cell"], **maps}
    write_result(f"occupancy_{name}", {"descriptive": True, "candidates": out})


def maze_frame(seed: str) -> None:
    """Render one reference frame (discovery game 0, step 5) by replay; background for maze figures."""
    from pacneuron.config import FIGURES_DIR
    from pacneuron.render import capture_frames
    f = capture_frames(data.split_dir(MODELS[seed].name, "discovery"), [(0, 5)])[(0, 5)]
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FIGURES_DIR / "maze_frame.npy", f)
