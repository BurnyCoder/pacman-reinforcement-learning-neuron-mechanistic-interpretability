# Preregistration addendum: frozen steering candidates (before any held-out data)

Author: Libor "Burny" Burian. Committed after the discovery phase and **before any held-out game was played**. The
machine-readable version is `results/candidates.json`; `src/pacneuron/prereg.py` refuses to run held-out phases
unless both files are committed and unchanged.

## What was looked at before freezing (discovery split only, seeds 0-99)

- Stochasticity check, the preregistered fallback, and data collection with sticky actions:
  `results/stochasticity_appo_1111*.json`, `results/collect_appo_1111_discovery*.json`.
- Exploratory audit (split-half by game), all fc units plus controls and conv3 channels: `results/audit_appo_1111.json`.
- Exploratory conjunction analysis: `results/conjunctions_appo_1111.json`.
- Open-loop screens (conv3 cell patches at the p99 dose, fc ablations): `results/screen_appo_1111.json`.
  An exploratory dose-response at max and 2x max dose was also inspected.
- Closed-loop pilots, 9 interventions x 30 discovery games (seeds 0-29): `results/pilot_appo_1111.json`.

## Deviations from the preregistration (declared before confirmatory data)

1. **Stochasticity.** Without sticky actions, same-start games stayed identical for a median of 1230 steps
   (threshold 200), so the preregistered fallback was applied: sticky actions with p = 0.25 everywhere. With sticky
   actions the median divergence is still 320 steps (above 200); no further fallback was preregistered. We
   proceed with p = 0.25. As an added robustness analysis, every H1 confidence interval is also computed with a
   bootstrap that resamples start-state clusters (games sharing an identical first state) instead of games.
2. **Sticky actions change the task.** The agents were trained without them. Under p = 0.25 the discovery mean
   score is 4082 (no sticky actions: 6384; model card: 6288). All conclusions concern the agent under sticky
   actions.
3. No other deviation. Both frozen candidates use preregistered intervention types (fc zero-ablation; conv3
   single-cell set to p99) and preregistered metrics.

## Frozen candidates (H2 family, K = 2, Holm-corrected)

| name | intervention | primary metric (predicted sign) | null distribution |
|---|---|---|---|
| `fc453_ablate` | fc unit 453 set to 0 at every step | ghosts eaten per power pill (−) | 99 random alive fc units (seed 0), each zero-ablated |
| `c3_ch18_cell56` | conv3 channel 18 at cell (5, 6) set to its discovery p99 (24.625) | approach rate to the cell's target node (−) | the other 63 conv3 channels at cell (5, 6), each set to its own p99 |

Secondary (not in the Holm family): for `fc453_ablate`, P(action ∈ {UP, UPRIGHT, UPLEFT} | nearest edible ghost
is up), predicted to decrease.

Positive control (not in the H2 family, no null): fc unit 188 clamped to its p99. Its UP logit weight is
−0.70, so it should reduce P(UP | up is open). This is mathematically a constant logit bias.

Arms on held-out seeds:
- baseline: the 100 held-out games collected with activations (unsteered), seeds 10000-10099;
- each candidate: 50 games, seeds 10000-10049;
- each null unit: 20 games, seeds 10000-10019.

Pass criteria are exactly preregistration §6:
1. predicted sign, with Holm p < 0.05 from a paired sign-flip test;
2. signed Δ above the 95th percentile of the null;
3. off-target metrics reported.

## Replication rules for agents 2222 and 3333 (H3; applied without human choice)

- `fc453_ablate`-type: the alive fc unit with the most negative discovery ablation-screen effect on P(move toward
  the nearest edible ghost). Zero-ablate it; metric ghosts per power pill (−); null of 99 random alive units.
- `c3_ch18_cell56`-type: the (channel, cell) with the most negative discovery open-loop attraction at the p99
  dose. Set it to its p99; metric approach rate to that cell's target node (−); null of the other 63 channels at
  that cell at their own p99.
- Each new agent first goes through the same discovery collection, audit and screen phases (sticky actions
  p = 0.25, seeds 0-99), then the held-out arms (seeds 10000+).
