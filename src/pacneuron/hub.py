"""Download pinned pretrained checkpoints from the Hugging Face Hub.

Local: thin wrapper over `huggingface_hub.hf_hub_download` (docs:
https://huggingface.co/docs/huggingface_hub/guides/download) with the revision pinned in config.
Global: model.py loads the returned file; nothing else touches the network.
"""

from __future__ import annotations

from pathlib import Path

# config must be imported first: it loads .env, and huggingface_hub reads HF_HOME at import time.
from pacneuron.config import MODELS, ModelSpec  # isort: skip
from huggingface_hub import hf_hub_download  # noqa: E402


def checkpoint_path(seed: str, filename: str | None = None) -> Path:
    """Return a local path to the checkpoint of agent `seed` ("1111"/"2222"/"3333").

    `filename` overrides the default latest checkpoint (e.g. a milestone file for training dynamics).
    The file is cached under HF_HOME, so repeated calls do not re-download.
    """
    spec: ModelSpec = MODELS[seed]
    return Path(hf_hub_download(repo_id=spec.repo_id, filename=filename or spec.filename, revision=spec.revision))
