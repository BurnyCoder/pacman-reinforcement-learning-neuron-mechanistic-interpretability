# Notes for coding agents

- Environment: `uv sync`; run everything with `uv run`. Windows needs the spawn-safe `if __name__ == "__main__"`
  guard for anything that calls `rollout.run_games` (it uses worker processes).
- Never change `env.py` wrappers or `model.py` normalisation without re-running `tests/test_fidelity.py`. They
  reproduce Sample Factory's observations bit-exactly and its logits to about 1e-5 on CPU, and gymnasium>=1.0's grayscale
  wrapper would silently break this. The test is CPU-only. On CUDA, PyTorch's default cuDNN TF32 convolutions change
  logits by up to about 0.045 (`results/fidelity.json`); `aa-control` checks that this leaves the seed pairing intact.
- Bulk data (activations, checkpoints, videos) goes to `DATA_DIR` from `.env`, never into git. Small result
  JSON goes to `results/`, and every number in a results table or inline marker must be generated from it
  (`pacneuron.report`). Videos (`videos` phase) and `paper/paper.pdf` (`scripts/build_paper.py`) are built separately.
- Never edit `reports/preregistration.md`, `reports/preregistration_addendum.md` or `results/candidates.json`: they
  are frozen. Corrections go to "Errata and undeclared deviations" in `reports/00_summary.md`.
- Held-out seeds (10000-10099) may only be read after the preregistration addendum is committed.
  `pacneuron.prereg.require_frozen_for` enforces this in `collect-heldout`, `h1`, `confirm` and `aa-control`; for
  agents 2222/3333 it also requires their committed `results/candidates_<agent>.json`.
  `offtarget`, `occupancy`, `specificity` and `videos` read only the arms that `confirm` wrote. The lock checks that the frozen
  files have no uncommitted diff against HEAD. It records SHA-256 hashes of their working-tree bytes, so the hashes
  depend on line endings.
- RAM uint8 arithmetic underflows: always go through `ram.py` helpers, which cast to int32.
- `ram[116]`: only its low six bits are the frightened countdown (`ram.frightened_ticks`). `ram.frightened_timer`
  keeps the raw byte because the frozen `fright_timer` feature used it.
- `STICKY=0` (environment variable) exists only to reproduce the preregistered no-sticky stochasticity check. It
  writes `*_nosticky` rollouts and results, and the pipeline refuses every other phase in that mode.
