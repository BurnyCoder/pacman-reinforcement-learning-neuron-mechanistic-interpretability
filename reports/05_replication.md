# Experiment 5: Replication on independently trained agents (seeds 2222 and 3333; preregistered H3)

## Question
Seed 1111's steering effects came from specific units: fc 453, and conv3 channel 18 at one cell. Units have no
correspondence across independently trained networks. So we ask a different question: does the *same selection
rule*, applied to another agent's own discovery data, find a unit whose steering has the same effect?

## Method
- Each agent goes through the identical pipeline:
  - discovery collection (seeds 0–99, sticky actions 0.25);
  - stochasticity check;
  - audit;
  - open-loop screens.
- `derive_candidates` then applies the rules frozen in `results/candidates.json` **without human choice**:
  - **fc ablation rule:** the alive fc unit whose zero-ablation most reduces P(move toward the nearest edible
    ghost) in the open-loop screen. Primary metric: ghosts eaten per power pill (−). Secondary metric: the
    matching edible-ghost direction and action set.
  - **conv3 patch rule:** the (channel, cell) with the most negative open-loop attraction at the p99 dose.
    Primary metric: approach rate to that cell's target node (−).
- The derived file `results/candidates_<agent>.json` is committed, and only then are the agent's held-out arms
  run, with the same nulls and tests as seed 1111.

## Results

### Agent 2222
<!-- BEGIN:models -->
| agent | architecture | params | algorithm and objective | training data | discovery games (sticky p=0.25) | mean score ± SD | model-card score (no sticky) |
|---|---|---|---|---|---|---|---|
| [appo_1111](https://huggingface.co/edbeeching/atari_2B_atari_mspacman_1111) | Nature-CNN: conv 32/64/64 -> fc 512 (ReLU) -> linear policy (9) + value | 1,689,258 | APPO (Sample Factory 2.0), maximise clipped game reward, gamma 0.99, no imitation | 2B env steps of interaction with MsPacmanNoFrameskip-v4 | 100 | 4082 ± 2032 | 6288 ± 1955 |
| [appo_2222](https://huggingface.co/edbeeching/atari_2B_atari_mspacman_2222) | Nature-CNN: conv 32/64/64 -> fc 512 (ReLU) -> linear policy (9) + value | 1,689,258 | APPO (Sample Factory 2.0), maximise clipped game reward, gamma 0.99, no imitation | 2B env steps of interaction with MsPacmanNoFrameskip-v4 | 100 | 5161 ± 2056 | 9056 ± 2544 |
| [appo_3333](https://huggingface.co/edbeeching/atari_2B_atari_mspacman_3333) | Nature-CNN: conv 32/64/64 -> fc 512 (ReLU) -> linear policy (9) + value | 1,689,258 | APPO (Sample Factory 2.0), maximise clipped game reward, gamma 0.99, no imitation | 2B env steps of interaction with MsPacmanNoFrameskip-v4 | 100 | 5056 ± 2358 | 6808 ± 1183 |
<!-- END:models -->

One-job audit (discovery, exploratory):
<!-- BEGIN:audit_appo_2222 -->
| unit set (discovery split-half, exploratory) | units | one-job fraction [95% CI] | median R²_all | median best single R² |
|---|---|---|---|---|
| trained agent (fc units) | 380 | 1.1% [0.8%, 1.6%] | 0.703 | 0.024 |
| random directions (control a) | 512 | 0.8% [0.2%, 1.4%] | 0.762 | 0.180 |
| non-negative unit mixtures (control b) | 512 | 1.2% [1.2%, 1.8%] | 0.756 | 0.043 |
| untrained network (control c) | 500 | 9.2% [5.0%, 12.0%] | 0.540 | 0.083 |
<!-- END:audit_appo_2222 -->

Steering (held-out, confirmatory):
<!-- BEGIN:steer_appo_2222 -->
<!-- END:steer_appo_2222 -->

### Agent 3333
One-job audit (discovery, exploratory):
<!-- BEGIN:audit_appo_3333 -->
| unit set (discovery split-half, exploratory) | units | one-job fraction [95% CI] | median R²_all | median best single R² |
|---|---|---|---|---|
| trained agent (fc units) | 420 | 0.0% [0.0%, 0.5%] | 0.682 | 0.025 |
| random directions (control a) | 512 | 0.0% [0.0%, 0.0%] | 0.753 | 0.184 |
| non-negative unit mixtures (control b) | 512 | 0.0% [0.0%, 0.4%] | 0.718 | 0.052 |
| untrained network (control c) | 501 | 11.6% [7.8%, 13.8%] | 0.526 | 0.093 |
<!-- END:audit_appo_3333 -->

Steering (held-out, confirmatory):
<!-- BEGIN:steer_appo_3333 -->
<!-- END:steer_appo_3333 -->

## Interpretation
Filled in after the runs. A rule that replicates across seeds shows that the *function* (e.g. a single fc unit
gating ghost hunting) is a reliable product of training, not an accident of one network.
