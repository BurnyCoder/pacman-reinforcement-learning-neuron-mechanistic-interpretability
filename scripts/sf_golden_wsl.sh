#!/usr/bin/env bash
# Build the reference Sample Factory environment in WSL and write tests/fixtures/sf_golden.npz.
# Run from Windows:  wsl -e bash -lc "bash /mnt/c/<repo>/scripts/sf_golden_wsl.sh /mnt/c/<repo>"
set -euo pipefail
REPO="$1"                                   # repo path as seen from WSL (/mnt/c/...)
WORK="$HOME/projects/mspacman-sf-golden"    # scratch venv lives in WSL's home, outside the repo
mkdir -p "$WORK" && cd "$WORK"
uv venv --python 3.10 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
# Era-matched stack: SF 2.0.3 requires gym>=0.26.1,<1.0 and torch<2.0; ale-py 0.8.1 ships with gym 0.26.2's atari extra.
uv pip install "sample-factory==2.0.3" "gym[atari,accept-rom-license]==0.26.2" "ale-py==0.8.1" \
  "torch==1.13.1" --extra-index-url https://download.pytorch.org/whl/cpu "numpy<2" opencv-python huggingface_hub
python - <<'EOF'
from huggingface_hub import hf_hub_download
for f in ["checkpoint_p0/checkpoint_000976624_2000125952.pth", "cfg.json"]:
    print(hf_hub_download("edbeeching/atari_2B_atari_mspacman_1111", f,
                          revision="07c1243fc96930ca3e632bb4574246140f63d54b", local_dir="ckpt"))
EOF
mkdir -p "$REPO/tests/fixtures"
python "$REPO/scripts/sf_golden_wsl.py" ckpt/checkpoint_p0/checkpoint_000976624_2000125952.pth ckpt/cfg.json \
  "$REPO/tests/fixtures/sf_golden.npz"
python -c "import importlib.metadata as m; print({p: m.version(p) for p in ['sample-factory', 'gym', 'ale-py', 'torch', 'opencv-python', 'numpy']})"
