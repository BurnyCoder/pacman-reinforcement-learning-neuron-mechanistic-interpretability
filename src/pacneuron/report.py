"""Regenerate every results table and figure from results/*.json (no hand-typed numbers).

Local: builds markdown tables and PNG figures, then replaces the text between
`<!-- BEGIN:<key> -->` and `<!-- END:<key> -->` markers in README.md, reports/*.md and paper/paper.md.
Global: `uv run python -m pacneuron.pipeline --phase report` is the single command that refreshes all
numbers; the pipeline's analysis phases only write JSON.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from pacneuron import viz
from pacneuron.config import FIGURES_DIR, MODELS, REPO_ROOT, RESULTS_DIR

TARGETS = [REPO_ROOT / "README.md", *sorted((REPO_ROOT / "reports").glob("*.md")), REPO_ROOT / "paper" / "paper.md"]
NAMES = {"agent": "trained agent (fc units)", "random_dirs": "random directions (control a)",
         "nonneg_mix": "non-negative unit mixtures (control b)", "untrained": "untrained network (control c)"}


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


def models_table() -> str:
    rows = ["| agent | algorithm | training | games (sticky p=0.25) | mean score ± SD | model card score | in-control steps |",
            "|---|---|---|---|---|---|---|"]
    for seed, spec in MODELS.items():
        r = _load(f"collect_{spec.name}_discovery")
        if not r:
            continue
        s = np.array(r["scores"])
        rows.append(f"| [{spec.name}](https://huggingface.co/{spec.repo_id}) | APPO (Sample Factory 2.0), reward only | "
                    f"2B env steps | {r['games']} | {s.mean():.0f} ± {s.std(ddof=1):.0f} | "
                    f"{spec.reported_mean_reward:.0f} ± {spec.reported_std_reward:.0f} (no sticky) | {sum(r['steps'])} |")
    return "\n".join(rows)


def clean_table(src: dict, label: str) -> str:
    rows = [f"| unit set ({label}) | units | one-job fraction [95% CI] | median R²_all | median best single R² |",
            "|---|---|---|---|---|"]
    for k, nm in NAMES.items():
        if k not in src:
            continue
        v = src[k]
        ci = v.get("ci")
        cis = f" [{_pct(ci[0])}, {_pct(ci[1])}]" if ci else ""
        rows.append(f"| {nm} | {v['units']} | {_pct(v['clean_frac'])}{cis} | {_f(v['median_r2_all'])} | "
                    f"{_f(v['median_best_r2'])} |")
    return "\n".join(rows)


def steering_table(conf: dict) -> str:
    rows = ["| intervention | primary metric | baseline → steered | Δ [95% CI] | Holm p | > 95% of null? | H2 |",
            "|---|---|---|---|---|---|---|"]
    for r in conf["candidates"]:
        e = r["effects"][r["metric"]]
        rows.append(f"| {r['label']} | {r['metric']} | {_f(e.get('base_mean'))} → {_f(e.get('arm_mean'))} | "
                    f"{_f(e.get('delta'))} [{_f(e.get('ci', [None])[0])}, {_f(e.get('ci', [None, None])[1])}] | "
                    f"{_f(r.get('p_holm'), 4)} | {'yes' if r['specific'] else 'no'} | "
                    f"{'pass' if r.get('passes_H2') else 'fail'} |")
    return "\n".join(rows)


def generate() -> None:
    blocks: dict[str, str] = {"models": models_table()}
    for seed, spec in MODELS.items():
        name = spec.name
        audit = _load(f"audit_{name}")
        if audit:
            blocks[f"audit_{name}"] = clean_table(audit["summary"], "discovery split-half, exploratory")
            viz.clean_fraction_bars(audit["summary"], FIGURES_DIR / f"clean_fraction_discovery_{name}.png",
                                    f"{name}: one-job neurons vs controls (discovery, exploratory)")
        h1 = _load(f"h1_{name}")
        if h1:
            blocks[f"h1_{name}"] = clean_table(h1["sets"], "held-out, confirmatory") + \
                f"\n\nH1 (agent − random directions): {_pct(h1['sets']['agent']['diff_vs_random_dirs']['mean'])} " \
                f"[{_pct(h1['sets']['agent']['diff_vs_random_dirs']['ci'][0])}, " \
                f"{_pct(h1['sets']['agent']['diff_vs_random_dirs']['ci'][1])}] -> " \
                f"{'supported' if h1['H1_supported'] else 'not supported'}."
            viz.clean_fraction_bars(h1["sets"], FIGURES_DIR / f"clean_fraction_{name}.png",
                                    f"{name}: one-job neurons vs controls (held-out)")
            viz.r2_scatter({"trained agent": (np.array(h1["sets"]["agent"]["r2_all"]), np.array(h1["sets"]["agent"]["best_r2"])),
                            "random directions": (np.array(h1["sets"]["random_dirs"]["r2_all"]),
                                                  np.array(h1["sets"]["random_dirs"]["best_r2"]))},
                           FIGURES_DIR / f"r2_scatter_{name}.png", f"{name}: is each unit explained by one feature?")
        conf = _load(f"confirm_{name}")
        if conf:
            blocks[f"steer_{name}"] = steering_table(conf)
            for r in conf["candidates"]:
                e = r["effects"][r["metric"]]
                if "delta" in e:
                    viz.effect_vs_null([{"label": r["label"], "delta": e["delta"], "ci": e["ci"], "null": r["null_deltas"]}],
                                       FIGURES_DIR / f"steer_{name}_{r['name']}.png", f"{name}: {r['label']}",
                                       f"Δ {r['metric']} (steered − baseline, paired by game seed)")
    for path in TARGETS:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            new = text
            for key, body in blocks.items():
                new = inject(new, key, body)
            if new != text:
                path.write_text(new, encoding="utf-8")
