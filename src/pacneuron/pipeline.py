"""Single entry point that runs the study as named phases.

Usage: uv run python -m pacneuron.pipeline --phase <name> [--seed 1111]
Phases (in order): collect-discovery, stochasticity, audit, screen, freeze-check, collect-heldout, h1, steer,
replicate, report, site. `all` runs the phases that need no human decision in between.
Global: each phase reads inputs from DATA_DIR / results and writes small JSON summaries to results/, from which
`pacneuron.report` regenerates every table and figure.
"""

from __future__ import annotations

import argparse
import json
import time

from pacneuron.config import DISCOVERY_SEEDS, HELDOUT_SEEDS, MODELS, RESULTS_DIR
from pacneuron.logs import get_logger

log = get_logger("pacneuron.pipeline")


def _model(seed: str):
    from pacneuron.config import DEVICE
    from pacneuron.hub import checkpoint_path
    from pacneuron.model import load_sf_checkpoint
    return load_sf_checkpoint(checkpoint_path(seed), DEVICE)


def write_result(name: str, obj) -> None:
    """Persist a phase summary to results/<name>.json (the only source for report numbers)."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))
    log.info(f"wrote results/{name}.json")


def collect(seed: str, split: str) -> None:
    """Play the split's games with activations recorded (agent + untrained control network on the same states)."""
    from pacneuron import data
    from pacneuron.model import orthogonal_init_like
    from pacneuron.rollout import run_games
    if split == "heldout":
        from pacneuron.prereg import require_frozen
        require_frozen()
    seeds = DISCOVERY_SEEDS if split == "discovery" else HELDOUT_SEEDS
    model = _model(seed)
    d = data.split_dir(MODELS[seed].name, split)
    t0 = time.time()
    recs, bank = run_games(model, seeds, record_acts=True, aux_model=orthogonal_init_like(model, seed=0),
                           on_game=data.saver(d), desc=f"{seed}/{split}")
    data.save_bank(d, bank)
    scores = [r["score"] for r in recs]
    write_result(f"collect_{MODELS[seed].name}_{split}", {
        "model": MODELS[seed].name, "split": split, "games": len(recs), "seeds": [r["seed"] for r in recs],
        "scores": scores, "steps": [r["steps"] for r in recs], "mazes": len(bank.mazes),
        "seconds": time.time() - t0})


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", required=True)
    p.add_argument("--seed", default="1111", choices=list(MODELS))
    a = p.parse_args(argv)
    log.info(f"phase={a.phase} seed={a.seed}")
    if a.phase == "collect-discovery":
        collect(a.seed, "discovery")
    elif a.phase == "collect-heldout":
        collect(a.seed, "heldout")
    else:
        from pacneuron import phases
        getattr(phases, a.phase.replace("-", "_"))(a.seed)


if __name__ == "__main__":
    main()
