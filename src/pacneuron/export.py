"""Export the neuron microscope for GitHub Pages: per-unit JSON + top-activating frame grids (WebP).

Local: reads results/audit_<model>.json (+ screen/confirm results when present), re-renders the referenced frames by
deterministic replay, writes site/data/<model>.json and site/img/<model>/*.webp.
Global: site/index.html (static, no build step) loads these files; the Pages workflow publishes site/.
"""

from __future__ import annotations

import json

import numpy as np
from PIL import Image

from pacneuron import data
from pacneuron.config import MODELS, REPO_ROOT, RESULTS_DIR
from pacneuron.logs import get_logger
from pacneuron.render import capture_frames, grid

log = get_logger(__name__)
SITE = REPO_ROOT / "site"
THUMB = (80, 86)  # playfield (160x172) at half resolution


def _save_grid(frames, path, cols=4):
    if frames:
        Image.fromarray(grid(frames, cols=cols, crop=THUMB[1])).save(path, "WEBP", quality=80)


def export_site(seed: str) -> None:
    name = MODELS[seed].name
    audit = json.loads((RESULTS_DIR / f"audit_{name}.json").read_text())
    extra = {}
    for k in ("screen", "confirm", "h1", "pilot"):
        f = RESULTS_DIR / f"{k}_{name}.json"
        if f.exists():
            extra[k] = json.loads(f.read_text())
    img_dir = SITE / "img" / name
    img_dir.mkdir(parents=True, exist_ok=True)
    units = [u for u in audit["fc_units"] if u["alive"]]
    req = set()
    for u in units:
        req |= {tuple(x) for x in u["top"]} | {tuple(x) for x in u["neg"][:8]}
        req |= {tuple(x) for b in u["bands"].values() for x in b}
    log.info(f"export {name}: rendering {len(req)} frames by replay")
    frames = capture_frames(data.split_dir(name, "discovery"), req, size=THUMB)
    out_units = []
    for u in units:
        n = u["unit"]
        _save_grid([frames[tuple(x)] for x in u["top"] if tuple(x) in frames], img_dir / f"fc{n:03d}_top.webp")
        _save_grid([frames[tuple(x)] for x in u["neg"][:8] if tuple(x) in frames], img_dir / f"fc{n:03d}_neg.webp")
        bands = [frames[tuple(x)] for b in ("top", "1-5%", "10-50%", "bottom") for x in u["bands"][b] if tuple(x) in frames]
        _save_grid(bands, img_dir / f"fc{n:03d}_bands.webp")
        top_feats = sorted(u["r2_by_feature"].items(), key=lambda kv: -kv[1])[:8]
        out_units.append({k: u[k] for k in ("unit", "frac_active", "p99", "max", "dla", "w_value", "best", "family",
                                             "r2_best", "second", "r2_second", "r2_all", "clean", "sign")}
                         | {"top_features": top_feats})
    payload = {"model": name, "actions": audit["actions"], "summary": audit["summary"], "fc_units": out_units,
               "dead_units": [u["unit"] for u in audit["fc_units"] if not u["alive"]],
               "conv3_channels": [{k: c[k] for k in ("channel", "frac_active", "auroc", "max")}
                                  for c in audit["conv3_channels"]],
               "results": {k: _slim(v) for k, v in extra.items()}}
    (SITE / "data").mkdir(parents=True, exist_ok=True)
    (SITE / "data" / f"{name}.json").write_text(json.dumps(payload, default=float))
    log.info(f"export {name}: {len(out_units)} units -> site/data/{name}.json")


def _slim(obj):
    """Drop bulky arrays (per-unit R² vectors, null lists) from result files before shipping them to the browser."""
    if isinstance(obj, dict):
        return {k: _slim(v) for k, v in obj.items() if k not in ("r2_all", "best_r2", "divergence_steps")}
    if isinstance(obj, list):
        return [_slim(v) for v in obj] if len(obj) < 400 else []
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj
