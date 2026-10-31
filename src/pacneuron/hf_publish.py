"""Publish the study's derived artefacts to the Hugging Face Hub and gather everything in one collection.

Local: uploads results/*.json, site/data/*.json and DATA_DIR/analysis/*.npz (unit R² matrices, screens) to the
dataset repo `BurnyCoder/mspacman-appo-neuron-atlas` with a dataset card, then creates or updates the collection
"Ms. Pac-Man RL neuron interpretability" containing the three pretrained agents and the dataset.
API: huggingface_hub HfApi.create_repo / upload_folder / create_collection / add_collection_item
(https://huggingface.co/docs/huggingface_hub/guides/collections, https://huggingface.co/docs/huggingface_hub/guides/upload).
Global: the pretrained agents are not ours (Edward Beeching, Sample Factory); they are linked, not re-uploaded.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from pacneuron.config import DATA_DIR, MODELS, REPO_ROOT, RESULTS_DIR  # isort: skip  (loads .env first)
from huggingface_hub import HfApi  # noqa: E402

DATASET = "BurnyCoder/mspacman-appo-neuron-atlas"
COLLECTION = "Ms. Pac-Man RL neuron interpretability"
GITHUB = "https://github.com/BurnyCoder/pacman-reinforcement-learning-neuron-mechanistic-interpretability"

CARD = f"""---
license: mit
tags: [interpretability, reinforcement-learning, atari, ms-pacman, mechanistic-interpretability]
---
# Ms. Pac-Man APPO neuron atlas

Derived artefacts of a preregistered neuron-level interpretability and steering study of reward-trained
(no imitation learning) Sample Factory APPO agents playing Atari Ms. Pac-Man.

- `results/*.json`: the analysis outputs from which the study's report generator (`pacneuron.report`) builds
  every results table and inline number in the README, reports and paper:
  - fidelity, ground-truth validation;
  - audits (per fc unit: best feature, R², action votes, top-activating states);
  - screens, pilots, confirmatory H1/H2 tests, replication, exploratory claim checks.

  Numbers inside the frozen preregistration and its addendum were written by hand before the results existed.
- `site_data/*.json`: per-unit data behind the interactive microscope (alive fc units only).
- `analysis/*.npz`: unit × feature R² matrices for the agent and controls from the exploratory discovery
  split-half audit (not the held-out H1 test), bootstrap draws, open-loop screen maps.

Top-activating frames are not stored here (the GitHub repository ships them as WebP grids under `site/img`). They
can be re-rendered by deterministic replay from (game seed, step), which needs the recorded action sequences: run
the collection phases of the code at {GITHUB} to regenerate them. Agents:
{chr(10).join(f"- https://huggingface.co/{m.repo_id}" for m in MODELS.values())}
"""


def publish() -> str:
    api = HfApi()
    api.create_repo(DATASET, repo_type="dataset", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "README.md").write_text(CARD, encoding="utf-8")
        for sub, src, pattern in (("results", RESULTS_DIR, "*.json"), ("site_data", REPO_ROOT / "site" / "data", "*.json"),
                                  ("analysis", DATA_DIR / "analysis", "*.npz")):
            (root / sub).mkdir()
            for f in src.glob(pattern):
                shutil.copy(f, root / sub / f.name)
        api.upload_folder(folder_path=str(root), repo_id=DATASET, repo_type="dataset",
                          commit_message="Update neuron atlas from pacneuron pipeline")
    # collection descriptions are limited in length; the GitHub link goes into the dataset item's note instead
    col = api.create_collection(COLLECTION, description="Neuron-level interpretability and steering of "
                                "reward-trained (no imitation) Ms. Pac-Man RL agents.", exists_ok=True)
    for m in MODELS.values():
        api.add_collection_item(col.slug, m.repo_id, "model", note="Pretrained APPO agent analysed in the study",
                                exists_ok=True)
    api.add_collection_item(col.slug, DATASET, "dataset", note=f"Derived results, unit atlas and analysis arrays. "
                            f"Code, paper and preregistration: {GITHUB}", exists_ok=True)
    return f"https://huggingface.co/collections/{col.slug}"
