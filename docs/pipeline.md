# Pipeline reference

All phases run through one entry point. A comma-separated list runs phases in order:

```bash
uv run python -m pacneuron.pipeline --phase <phase>[,<phase>...] --seed {1111|2222|3333}
```

| phase | reads | writes | notes |
|---|---|---|---|
| `fidelity` | `tests/fixtures/sf_golden.npz` | `results/fidelity.json` | compares with real Sample Factory on CPU; also records the GPU (TF32) logit difference if CUDA is available |
| `collect-discovery` | HF checkpoint | `DATA_DIR/rollouts/<agent>/discovery/seed_*.npz`, `mazes.pkl`; `results/collect_<agent>_discovery.json` | 100 games, activations + untrained-net activations; with `STICKY=0` writes `discovery_nosticky` / `*_nosticky.json` instead |
| `stochasticity` | discovery games | `results/stochasticity_<agent>.json` | preregistered divergence check (`*_nosticky` with `STICKY=0`) |
| `ground-truth` | discovery games | `results/ground_truth_<agent>.json` | RAM vs pixel validation, edible-bit disagreement breakdown, `ram[116]` countdown dynamics |
| `audit` | discovery games | `results/audit_<agent>.json`, `DATA_DIR/analysis/audit_<agent>.npz` | exploratory split-half audit, all fc units + controls, conv3 channels |
| `conjunctions` | discovery games | `results/conjunctions_<agent>.json` | exploratory place × heading etc. |
| `conv3-states` | discovery games | `results/conv3_states_<agent>.json` | exploratory global-state AUROCs |
| `screen` | discovery games, audit | `results/screen_<agent>.json` | open-loop conv3 patch + fc ablation screens |
| `pilot` | `results/pilot_specs_<agent>.json` | `DATA_DIR/steer/<agent>/pilot/*`, `results/pilot_<agent>.json` | discovery seeds 0–29 |
| `derive-candidates` | audit, screen, frozen rules | `results/candidates_<agent>.json` | replication agents only; commit before held-out |
| `collect-heldout` | frozen files (lock) | `DATA_DIR/rollouts/<agent>/heldout/*` | refuses to run unless the preregistration lock holds |
| `h1` | discovery + held-out | `results/h1_<agent>.json` | confirmatory H1 |
| `confirm` | frozen candidates, held-out baseline | `DATA_DIR/steer/<agent>/heldout/*`, `results/confirm_<agent>.json` | confirmatory H2/H3 incl. null arms |
| `offtarget` | confirm arms | `results/offtarget_<agent>.json` | §6 criterion 3: two-sided tests + BH |
| `attribution` | discovery games, candidates | `results/attribution_<agent>.json` | upstream contributions, occlusion, stratified AUROC, unique R² |
| `occupancy` | confirm arms | `results/occupancy_<agent>.json` | where the agent goes with/without conv3 patches |
| `specificity` | confirm arms, `DATA_DIR/analysis/screen_<agent>.npz` | `results/specificity_<agent>.json` | post hoc (added by the 2026-11-01 audit): approach toward all 49 cell targets, region occupancy, open-loop KL rank, damage vs nulls and navigation for conv3 candidates; trigger-context rates and a null for the fc secondary metric; start-cluster ICC for every candidate |
| `aa-control` | frozen files (lock), held-out baseline | `DATA_DIR/steer/<agent>/heldout/aa_noop/*`, `results/aa_control_<agent>.json` | exploratory: unsteered replay of 50 held-out seeds; checks that the seed pairing holds |
| `claim-checks` | discovery games, audit arrays, attribution | `results/claim_checks_<agent>.json` | exploratory numbers behind prose claims: best-feature counts, candidates' frightened share and occlusion-peak distances, masked `ram[116]` R² |
| `dose-screen` | discovery games, audit | `results/dose_screen_<agent>.json` | exploratory dose-response of the patch screen |
| `maze-frame` | discovery game 0 | `figures/maze_frame.npy` | background frame for maze figures |
| `publish-hf` | results, site data, analysis arrays | HF dataset + collection, `results/hf_publish.json` | needs a HF token (`HF_TOKEN_PATH`) |
| `videos` | confirm arms | `figures/video_<agent>_*.mp4/.gif`, `site/media/` | side-by-side replays |
| `report` | `results/*.json` | tables, inline numbers and static figures in README, reports, paper, docs | the only source of marked numbers; the frozen preregistration files are hand-written and never rewritten |
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
  - optionally `conv3` [T,64,7,7] fp16 (post-ReLU), `fc_pre` [T,512] fp16 (pre-ReLU) and `aux_fc_pre` [T,512] fp16
    (the untrained control network, pre-ReLU).
- **Replay.** `env.replay(seed, actions)` reproduces any state exactly, including sticky actions, so frames are
  never stored.
- **Intervention spec** (JSON): `{"site": "fc"|"conv3", "units": [i], "value": v, "mode": "set"|"add",
  "cell": [r, c] | null, "gate": "none"|"frightened"|"not_frightened"}`.

## Extending

- **New agent.** Add a `ModelSpec` to `config.MODELS` with a pinned HF revision. If its checkpoint keys differ,
  extend `model._SF_KEYMAP`, then re-run the fidelity test against a new golden file.
- **New feature.** Add it to `features.compute_features` and `FAMILY`; the audit picks it up automatically.
- **New metric.** Add it to `metrics.game_metrics`; candidates refer to metrics by name.
