"""Regenerate every results table, inline number and figure from results/*.json (no hand-typed numbers).

Local: builds markdown tables (block markers `<!-- BEGIN:<key> -->...<!-- END:<key> -->`) and inline numbers
(`<!--n:<key>-->value<!--/n-->`) and PNG figures, then rewrites README.md, reports/*.md and paper/paper.md.
Global: `uv run python -m pacneuron.pipeline --phase report` is the single command that refreshes all numbers;
analysis phases only write JSON.
"""

from __future__ import annotations

import json
import re

import numpy as np

from pacneuron import viz
from pacneuron.config import FIGURES_DIR, MODELS, REPO_ROOT, RESULTS_DIR

NAMES = {"agent": "trained agent (fc units)", "random_dirs": "random directions (control a)",
         "nonneg_mix": "non-negative unit mixtures (control b)", "untrained": "untrained network (control c)"}


def _targets():
    return [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "reports").glob("*.md")), REPO_ROOT / "paper" / "paper.md",
            *sorted((REPO_ROOT / "docs").glob("*.md"))]


def _load(name: str):
    f = RESULTS_DIR / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def _f(x, d=3):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def _pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def inject(text: str, key: str, body: str) -> str:
    pat = re.compile(rf"(<!-- BEGIN:{re.escape(key)} -->)(.*?)(<!-- END:{re.escape(key)} -->)", re.S)
    return pat.sub(lambda m: f"{m.group(1)}\n{body.strip()}\n{m.group(3)}", text)


def inject_inline(text: str, nums: dict[str, str]) -> str:
    """Replace inline markers `<!--n:key-->old<!--/n-->` with freshly computed values (unknown keys untouched)."""
    return re.sub(r"(<!--n:([\w.:-]+)-->)(.*?)(<!--/n-->)",
                  lambda m: f"{m.group(1)}{nums.get(m.group(2), m.group(3))}{m.group(4)}", text)


# ------------------------------------------------------------------------------------------------- tables
def models_table() -> str:
    from pacneuron.model import SFActorCritic
    n_params = sum(p.numel() for p in SFActorCritic().parameters())
    rows = ["| agent | architecture | params | algorithm and objective | training data | discovery games (sticky p=0.25) | "
            "mean score ± SD | model-card score (no sticky) |", "|---|---|---|---|---|---|---|---|"]
    for spec in MODELS.values():
        r = _load(f"collect_{spec.name}_discovery")
        if r:
            s = np.array(r["scores"])
            rows.append(f"| [{spec.name}](https://huggingface.co/{spec.repo_id}) | Nature-CNN: conv 32/64/64 -> fc 512 "
                        f"(ReLU) -> linear policy (9) + value | {n_params:,} | APPO (Sample Factory 2.0), maximise "
                        f"clipped game reward, gamma 0.99, no imitation | 2B env steps of interaction with "
                        f"MsPacmanNoFrameskip-v4 | {r['games']} | {s.mean():.0f} ± {s.std(ddof=1):.0f} | "
                        f"{spec.reported_mean_reward:.0f} ± {spec.reported_std_reward:.0f} |")
    return "\n".join(rows)


def clean_table(src: dict, label: str) -> str:
    rows = [f"| unit set ({label}) | units | one-job fraction [95% CI] | median R²_all | median best single R² |",
            "|---|---|---|---|---|"]
    for k, nm in NAMES.items():
        if k in src:
            v = src[k]
            ci = v.get("ci")
            cis = f" [{_pct(ci[0])}, {_pct(ci[1])}]" if ci else ""
            rows.append(f"| {nm} | {v['units']} | {_pct(v['clean_frac'])}{cis} | {_f(v['median_r2_all'])} | "
                        f"{_f(v['median_best_r2'])} |")
    return "\n".join(rows)


def steering_table(conf: dict) -> str:
    rows = ["| intervention | primary metric (predicted sign) | baseline → steered | Δ [95% CI] | Holm p | above null 95th pct? | H2 |",
            "|---|---|---|---|---|---|---|"]
    for r in conf["candidates"]:
        e = r["effects"][r["metric"]]
        ci = e.get("ci", [None, None])
        rows.append(f"| {r['label']} | {r['metric']} ({'+' if r['sign'] > 0 else '−'}) | {_f(e.get('base_mean'))} → "
                    f"{_f(e.get('arm_mean'))} | {_f(e.get('delta'))} [{_f(ci[0])}, {_f(ci[1])}] | {_f(r.get('p_holm'), 4)} | "
                    f"{'yes' if r['specific'] else 'no'} (null q95 {_f(r['null_q95_signed'])}) | "
                    f"{'pass' if r.get('passes_H2') else 'fail'} |")
    pc = conf.get("positive_control")
    if pc and "delta" in pc["effect"]:
        e = pc["effect"]
        rows.append(f"| {pc['label']} (positive control) | {pc['metric']} (−) | {_f(e['base_mean'])} → {_f(e['arm_mean'])} | "
                    f"{_f(e['delta'])} [{_f(e['ci'][0])}, {_f(e['ci'][1])}] | – | – | – |")
    return "\n".join(rows)


def offtarget_table(conf: dict) -> str:
    rows = ["| intervention | Δ return | Δ deaths / 1k steps | Δ pellets / 1k steps | Δ ghosts per pill |", "|---|---|---|---|---|"]
    for r in conf["candidates"]:
        e = r["effects"]
        cell = lambda k, d=2, e=e: f"{_f(e[k].get('delta'), d)} [{_f(e[k].get('ci', [None])[0], d)}, {_f(e[k].get('ci', [None, None])[1], d)}]"
        rows.append(f"| {r['label']} | {cell('return', 0)} | {cell('deaths_per_1k')} | {cell('pellets_per_1k')} | {cell('ghosts_per_pill')} |")
    return "\n".join(rows)


def fidelity_table(r: dict) -> str:
    return "\n".join(["| check | value |", "|---|---|",
                      f"| golden trajectory steps (incl. {r['life_losses']} life losses and game over) | {r['golden_steps']} |",
                      f"| steps with any observation pixel mismatch | {r['obs_mismatched_steps']} |",
                      f"| max abs. logit difference | {r['max_abs_logit_diff']:.1e} |",
                      f"| max abs. value difference | {r['max_abs_value_diff']:.1e} |",
                      f"| reference stack | {r['reference_stack']} |"])


def ground_truth_table(r: dict) -> str:
    return "\n".join(["| validation (discovery games, sticky actions) | value |", "|---|---|",
                      f"| agent steps checked | {r['steps']} ({r['games']} games) |",
                      f"| RAM edible bit (bit 7 of ram[1+i]) vs blue sprite pixels, per ghost-step | {_pct(r['edible_bit_vs_blue_pixels_agreement'])} |",
                      f"| frightened timer ram[116] > 0 vs any ghost edible | {_pct(r['timer_vs_any_edible_agreement'])} |",
                      f"| eaten pellets at Ms. Pac-Man's lattice node ({r['pellet_events']} events) | {_pct(r['pellet_eaten_at_player_node'])} |",
                      f"| steps in control (moving; excludes death, ready and level transitions) | {_pct(r['in_control_fraction'])} |",
                      f"| distinct maze layouts seen | {r['maze_layouts']} |"])


def conjunction_table(r: dict) -> str:
    rows = ["| explanatory variable(s) (exploratory, post hoc) | median R² | units with R² > 0.3 | units with R² > 0.5 |",
            "|---|---|---|---|"]
    for k, v in r["models"].items():
        rows.append(f"| {k} | {_f(v['median_r2'])} | {_pct(v['frac_gt_0.3'])} | {_pct(v['frac_gt_0.5'])} |")
    return "\n".join(rows)


def pilot_table(r: dict) -> str:
    rows = ["| intervention (discovery pilot, paired games) | targeted metric | baseline → steered | Δ [95% CI] | Δ return | Δ deaths / 1k |",
            "|---|---|---|---|---|---|"]
    for c in r["candidates"]:
        e, ret, dth = c["effects"][c["metric"]], c["effects"]["return"], c["effects"]["deaths_per_1k"]
        rows.append(f"| `{c['name']}` | {c['metric']} | {_f(e.get('base_mean'))} → {_f(e.get('arm_mean'))} | "
                    f"{_f(e.get('delta'))} [{_f(e['ci'][0])}, {_f(e['ci'][1])}] | {_f(ret.get('delta'), 0)} | {_f(dth.get('delta'), 2)} |")
    return "\n".join(rows)


def screen_table(r: dict) -> str:
    rows = ["| rank | conv3 channel @ cell | Δ P(move toward cell) at junctions | dose (p99) |", "|---|---|---|---|"]
    for k, x in enumerate(r["most_repelling"][:5] + r["most_attracting"][:5]):
        rows.append(f"| {'repel' if k < 5 else 'attract'} {k % 5 + 1} | ch {x['channel']} @ ({x['cell'][0]}, {x['cell'][1]}) | "
                    f"{100 * x['attraction_junction']:+.1f} pp | {x['dose']:.1f} |")
    return "\n".join(rows)


# ------------------------------------------------------------------------------------------------- driver
def generate() -> None:
    blocks: dict[str, str] = {"models": models_table()}
    nums: dict[str, str] = {}
    if hf := _load("hf_publish"):
        nums["hf_collection"], nums["hf_dataset"] = hf["collection"], hf["dataset"]
    if fid := _load("fidelity"):
        blocks["fidelity"] = fidelity_table(fid)
        nums["fid_logit"] = f"{fid['max_abs_logit_diff']:.1e}"
        nums["fid_steps"] = str(fid["golden_steps"])
    for spec in MODELS.values():
        name = spec.name
        if gt := _load(f"ground_truth_{name}"):
            blocks[f"ground_truth_{name}"] = ground_truth_table(gt)
            nums[f"{name}:edible_agree"] = _pct(gt["edible_bit_vs_blue_pixels_agreement"])
        if cj := _load(f"conjunctions_{name}"):
            blocks[f"conjunctions_{name}"] = conjunction_table(cj)
            nums[f"{name}:place_heading_median"] = _f(cj["models"]["place x heading"]["median_r2"], 2)
            nums[f"{name}:place_heading_frac"] = _pct(cj["models"]["place x heading"]["frac_gt_0.3"])
        if cs := _load(f"conv3_states_{name}"):
            for f, v in cs["states"].items():
                au = np.array(v["auroc"])
                c = int(np.argmax(np.abs(au - 0.5)))
                nums[f"{name}:c3_{f}"] = f"channel {c} (AUROC {au[c]:.2f})"
                nums[f"{name}:c3_ch20_{f}"] = f"{au[20]:.2f}"
        if pl := _load(f"pilot_{name}"):
            blocks[f"pilot_{name}"] = pilot_table(pl)
            for c in pl["candidates"]:
                for mt, e in c["effects"].items():
                    if "delta" in e:
                        nums[f"{name}:pilot:{c['name']}:{mt}"] = f"{_f(e['base_mean'])} → {_f(e['arm_mean'])} (Δ {_f(e['delta'])} [{_f(e['ci'][0])}, {_f(e['ci'][1])}])"
        if ds := _load(f"dose_screen_{name}"):
            for lab, v in ds["doses"].items():
                nums[f"{name}:dose:{lab}"] = f"{100 * v['strongest_repel']:+.1f} pp"
        if sc := _load(f"screen_{name}"):
            blocks[f"screen_{name}"] = screen_table(sc)
        st, st0 = _load(f"stochasticity_{name}"), _load(f"stochasticity_{name}_nosticky")
        if st and st0:
            nums[f"{name}:div_nosticky"] = f"{st0['median_divergence']:.0f}"
            nums[f"{name}:div_sticky"] = f"{st['median_divergence']:.0f}"
        co, co0 = _load(f"collect_{name}_discovery"), _load(f"collect_{name}_discovery_nosticky")
        if co:
            nums[f"{name}:score_sticky"] = f"{np.mean(co['scores']):.0f}"
        if co0:
            nums[f"{name}:score_nosticky"] = f"{np.mean(co0['scores']):.0f}"
        if audit := _load(f"audit_{name}"):
            al = [u for u in audit["fc_units"] if u["alive"]]
            nums[f"{name}:alive"] = str(len(al))
            nums[f"{name}:dead"] = str(len(audit["fc_units"]) - len(al))
            obj = [(v, k, c["channel"]) for c in audit["conv3_channels"] for k, v in c["auroc"].items() if v == v]
            v, k, c = max(obj)
            nums[f"{name}:c3_best_obj"] = f"{v:.2f} (channel {c}, {k})"
            nums[f"{name}:clean_disc"] = str(sum(u["clean"] for u in al))
            nums[f"{name}:median_best_disc"] = _f(audit["summary"]["agent"]["median_best_r2"], 2)
            nums[f"{name}:median_all_disc"] = _f(audit["summary"]["agent"]["median_r2_all"], 2)
            blocks[f"audit_{name}"] = clean_table(audit["summary"], "discovery split-half, exploratory")
            viz.clean_fraction_bars(audit["summary"], FIGURES_DIR / f"clean_fraction_discovery_{name}.png",
                                    f"{name}: one-job neurons vs controls (discovery, exploratory)")
        if h1 := _load(f"h1_{name}"):
            for k in NAMES:
                nums[f"{name}:h1_{k}"] = _pct(h1["sets"][k]["clean_frac"])
            dv = h1["sets"]["agent"]["diff_vs_random_dirs"]
            nums[f"{name}:h1_diff"] = f"{_pct(dv['mean'])} [{_pct(dv['ci'][0])}, {_pct(dv['ci'][1])}]"
            nums[f"{name}:h1_verdict"] = "supported" if h1["H1_supported"] else "not supported"
            blocks[f"h1_{name}"] = clean_table(h1["sets"], "held-out, confirmatory") + \
                f"\n\nH1 test (agent minus random directions): {nums[f'{name}:h1_diff']}; H1 {nums[f'{name}:h1_verdict']}."
            viz.clean_fraction_bars(h1["sets"], FIGURES_DIR / f"clean_fraction_{name}.png",
                                    f"{name}: one-job neurons vs controls (held-out)")
            viz.r2_scatter({"trained agent": (np.array(h1["sets"]["agent"]["r2_all"]), np.array(h1["sets"]["agent"]["best_r2"])),
                            "random directions": (np.array(h1["sets"]["random_dirs"]["r2_all"]),
                                                  np.array(h1["sets"]["random_dirs"]["best_r2"]))},
                           FIGURES_DIR / f"r2_scatter_{name}.png", f"{name}: is each unit explained by one feature?")
        if (occ := _load(f"occupancy_{name}")) and (frame_f := RESULTS_DIR.parent / "figures" / "maze_frame.npy").exists():
            for cname, o in occ["candidates"].items():
                viz.occupancy_maps(np.load(frame_f), np.array(o["baseline"]), np.array(o["steered"]), tuple(o["cell"]),
                                   FIGURES_DIR / f"occupancy_{name}_{cname}.png",
                                   f"{name}: where the agent goes, with and without {cname} (held-out, outlined: patched cell)")
        if conf := _load(f"confirm_{name}"):
            blocks[f"steer_{name}"] = steering_table(conf)
            blocks[f"offtarget_{name}"] = offtarget_table(conf)
            nums[f"{name}:h2_verdict"] = "supported" if conf["H2_supported"] else "not supported"
            for r in conf["candidates"]:
                e = r["effects"][r["metric"]]
                if "delta" not in e:
                    continue
                key = f"{name}:{r['name']}"
                nums[f"{key}:base"], nums[f"{key}:arm"] = _f(e["base_mean"]), _f(e["arm_mean"])
                nums[f"{key}:delta"] = f"{_f(e['delta'])} [{_f(e['ci'][0])}, {_f(e['ci'][1])}]"
                nums[f"{key}:p"] = _f(r.get("p_holm"), 4)
                nums[f"{key}:pass"] = "passes" if r.get("passes_H2") else "fails"
                nd = np.array(r["null_deltas"], float)
                if len(nd):
                    nums[f"{key}:null_mean"] = _f(float(nd.mean()))
                    nums[f"{key}:null_n"] = str(len(nd))
                    nums[f"{key}:null_ge"] = str(int((r["sign"] * nd >= r["sign"] * e["delta"]).sum()))
                for k2 in ("return", "deaths_per_1k", "pellets_per_1k"):
                    x = r["effects"][k2]
                    if "delta" in x:
                        nums[f"{key}:{k2}"] = f"{_f(x['delta'], 1 if k2 != 'return' else 0)} [{_f(x['ci'][0], 1 if k2 != 'return' else 0)}, {_f(x['ci'][1], 1 if k2 != 'return' else 0)}]"
                if r.get("secondary") and "delta" in r["secondary"]:
                    s2 = r["secondary"]
                    nums[f"{key}:sec"] = f"{_f(s2['base_mean'])} → {_f(s2['arm_mean'])} (Δ {_f(s2['delta'])} [{_f(s2['ci'][0])}, {_f(s2['ci'][1])}])"
                viz.effect_vs_null([{"label": r["label"], "delta": e["delta"], "ci": e["ci"], "null": r["null_deltas"]}],
                                   FIGURES_DIR / f"steer_{name}_{r['name']}.png", f"{name}: {r['label']}",
                                   f"Δ {r['metric']} (steered − baseline, paired by game seed)")
    # H3 (preregistration §7): a frozen rule replicates if the rule-selected unit passes criteria 1-2 in BOTH
    # replication agents; within-agent Holm p-values are additionally Holm-corrected over the two agents.
    from pacneuron.stats import holm
    rules = {"fc453_ablate": "zero-ablate the fc unit most needed for approaching edible ghosts",
             "c3_ch18_cell56": "set the most repelling conv3 (channel, cell) to its p99"}
    rows = ["| rule (frozen on agent 1111) | agent | selected unit | primary metric Δ [95% CI] | Holm p (within agent) | "
            "null units at least as strong | passes |", "|---|---|---|---|---|---|---|"]
    verdict = {}
    for rule, desc in rules.items():
        ps, ok = [], []
        for a in ("appo_2222", "appo_3333"):
            conf = _load(f"confirm_{a}")
            if not conf:
                continue
            r = next(c for c, cf in zip(conf["candidates"], _load(f"candidates_{a}")["candidates"]) if cf.get("rule") == rule)
            e = r["effects"][r["metric"]]
            nd = np.array(r["null_deltas"], float)
            ps.append(r["p_holm"])
            ok.append(bool(r["passes_H2"]))
            rows.append(f"| {desc} | {a} | {r['label']} | {_f(e['delta'])} [{_f(e['ci'][0])}, {_f(e['ci'][1])}] | "
                        f"{_f(r['p_holm'], 4)} | {int((r['sign'] * nd >= r['sign'] * e['delta']).sum())} of {len(nd)} | "
                        f"{'yes' if r['passes_H2'] else 'no'} |")
        if len(ok) == 2:
            verdict[rule] = all(ok) and bool((holm(ps) < 0.05).all())
            nums[f"h3:{rule}"] = "replicates" if verdict[rule] else "does not replicate"
    blocks["h3"] = "\n".join(rows)
    for path in _targets():
        if path.exists():
            text = path.read_text(encoding="utf-8")
            new = text
            for key, body in blocks.items():
                new = inject(new, key, body)
            new = inject_inline(new, nums)
            if new != text:
                path.write_text(new, encoding="utf-8")
    (RESULTS_DIR / "report_numbers.json").write_text(json.dumps(nums, indent=1, ensure_ascii=False), encoding="utf-8")
