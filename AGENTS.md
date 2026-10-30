# Notes for coding agents

- Environment: `uv sync`; run everything with `uv run`. Windows needs the spawn-safe `if __name__ == "__main__"`
  guard for anything that calls `rollout.run_games` (it uses worker processes).
- Never change `env.py` wrappers or `model.py` normalisation without re-running `tests/test_fidelity.py`. They
  reproduce Sample Factory bit-exactly, and gymnasium>=1.0's grayscale wrapper would silently break this.
- Bulk data (activations, checkpoints, videos) goes to `DATA_DIR` from `.env`, never into git. Small result
  JSON goes to `results/`, and every report number must be generated from it (`pacneuron.report`).
- Held-out seeds (10000-10099) may only be read by confirmatory phases, after the preregistration addendum is
  committed (`pacneuron.prereg` enforces this).
- RAM uint8 arithmetic underflows: always go through `ram.py` helpers, which cast to int32.
