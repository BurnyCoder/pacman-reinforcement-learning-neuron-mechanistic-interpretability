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
<!-- END:models -->

One-job audit (discovery, exploratory):
<!-- BEGIN:audit_appo_2222 -->
<!-- END:audit_appo_2222 -->

Steering (held-out, confirmatory):
<!-- BEGIN:steer_appo_2222 -->
<!-- END:steer_appo_2222 -->

### Agent 3333
One-job audit (discovery, exploratory):
<!-- BEGIN:audit_appo_3333 -->
<!-- END:audit_appo_3333 -->

Steering (held-out, confirmatory):
<!-- BEGIN:steer_appo_3333 -->
<!-- END:steer_appo_3333 -->

## Interpretation
Filled in after the runs. A rule that replicates across seeds shows that the *function* (e.g. a single fc unit
gating ghost hunting) is a reliable product of training, not an accident of one network.
