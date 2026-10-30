# Pipeline reference

All phases run through one entry point. A comma-separated list runs phases in order:

```bash
uv run python -m pacneuron.pipeline --phase <phase>[,<phase>...] --seed {1111|2222|3333}
```

| phase | reads | writes | notes |
|---|---|---|---|
| `fidelity` | `tests/fixtures/sf_golden.npz` | `results/fidelity.json` | compares with real Sample Factory |
| `collect-discovery` | HF checkpoint | `DATA_DIR/rollouts/<agent>/discovery/seed_*.npz`, `mazes.pkl`; `results/collect_<agent>_discovery.json` | 100 games, activations + untrained-net activations |
| `stochasticity` | discovery games | `results/stochasticity_<agent>.json` | preregistered divergence check |
| `ground-truth` | discovery games | `results/ground_truth_<agent>.json` | RAM vs pixel validation |
| `audit` | discovery games | `results/audit_<agent>.json`, `DATA_DIR/analysis/audit_<agent>.npz` | exploratory split-half audit, all fc units + controls, conv3 channels |
| `conjunctions` | discovery games | `results/conjunctions_<agent>.json` | exploratory place × heading etc. |
| `conv3-states` | discovery games | `results/conv3_states_<agent>.json` | exploratory global-state AUROCs |
| `screen` | discovery games, audit | `results/screen_<agent>.json` | open-loop conv3 patch + fc ablation screens |
| `pilot` | `results/pilot_specs_<agent>.json` | `DATA_DIR/steer/<agent>/pilot/*`, `results/pilot_<agent>.json` | discovery seeds 0–29 |
| `derive-candidates` | audit, screen, frozen rules | `results/candidates_<agent>.json` | replication agents only; commit before held-out |
| `collect-heldout` | frozen files (lock) | `DATA_DIR/rollouts/<agent>/heldout/*` | refuses to run unless the preregistration lock holds |
| `h1` | discovery + held-out | `results/h1_<agent>.json` | confirmatory H1 |
| `confirm` | frozen candidates, held-out baseline | `DATA_DIR/steer/<agent>/heldout/*`, `results/confirm_<agent>.json` | confirmatory H2/H3 incl. null arms |
| `videos` | confirm arms | `figures/video_<agent>_*.mp4/.gif`, `site/media/` | side-by-side replays |
| `report` | `results/*.json` | tables, inline numbers and figures in README, reports, paper | the only source of numbers |
| `site` | audit + other results | `site/data/<agent>.json`, `site/img/<agent>/*.webp` | microscope export |

## Data formats

- **Game record** (`seed_XXXXXX.npz`), one row per agent decision `t`:
  - `ram` [T,128] uint8: the state acted on;
  - `maze` [T] (an index into `mazes.pkl`);
  - `blue` [T,4]: frightened-blue pixels per ghost;
  - `action` [T];
  - `logits` [T,9];
  - `value` [T];
  - `raw_reward` [T]: the unclipped reward of transition t;
  - `life_lost` [T], `game_over` [T];
  - optionally `conv3` [T,64,7,7] fp16, `fc_pre` [T,512] fp16 and `aux_fc_pre` [T,512] fp16 (the untrained
    control network).
- **Replay.** `env.replay(seed, actions)` reproduces any state exactly, including sticky actions, so frames are
  never stored.
- **Intervention spec** (JSON): `{"site": "fc"|"conv3", "units": [i], "value": v, "mode": "set"|"add",
  "cell": [r, c] | null, "gate": "none"|"frightened"|"not_frightened"}`.

## Extending

- **New agent.** Add a `ModelSpec` to `config.MODELS` with a pinned HF revision. If its checkpoint keys differ,
  extend `model._SF_KEYMAP`, then re-run the fidelity test against a new golden file.
- **New feature.** Add it to `features.compute_features` and `FAMILY`; the audit picks it up automatically.
- **New metric.** Add it to `metrics.game_metrics`; candidates refer to metrics by name.
