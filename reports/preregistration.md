# Preregistration: neuron-level interpretability and steering of reward-trained Ms. Pac-Man agents

Author: Libor "Burny" Burian. Written and committed **before any neuron activation data was collected**.
An addendum (`reports/preregistration_addendum.md` + `results/candidates.json`) freezes the steering candidates
after the discovery phase and **before** any held-out data is read. `src/pacneuron/prereg.py` refuses to run
confirmatory phases unless both files are committed and unchanged.

## 1. Question

Goodfire's essay "Neural Networks Are Already Clean" (https://already-clean.goodfire.com/) argues that most
individual neurons of trained networks have "one main job" that can be read off their top-activating
examples and confirmed by steering. Does this hold for a deep RL agent trained purely from reward? Concretely:

- Are the 512 fully connected (fc) neurons of a pretrained Ms. Pac-Man agent better aligned with exact game-state
  features than random directions of the same layer (a privileged basis)?
- Is there at least one neuron or conv channel whose steering changes the agent's behaviour in a specific,
  predicted way, beyond what matched random units do?

## 2. Subjects (fixed)

- Agents: `edbeeching/atari_2B_atari_mspacman_{1111,2222,3333}` (Sample Factory 2.0 APPO, 2B env steps, reward
  only), latest checkpoint `checkpoint_p0/checkpoint_000976624_2000125952.pth`, HF revisions pinned in
  `src/pacneuron/config.py`. Seed 1111 is the primary model; 2222 and 3333 are replications.
- Architecture: Nature-CNN (conv 32/64/64, fc 512 ReLU), linear policy (9 actions) and value heads, no RNN.
- Environment: `MsPacmanNoFrameskip-v4` with the training wrapper chain. Fidelity gate (passed before this
  document): observations bit-exact and logits within 1e-5 of real Sample Factory 2.0.3 / gym 0.26.2 /
  ale-py 0.8.1 on a 446-step golden trajectory (`tests/test_fidelity.py`).
- Untrained control network: same architecture, orthogonal init (gain 1, zero bias; the run's
  `policy_initialization`), trained observation normaliser, seed 0. It is run on the same states as the agent
  but never acts.

## 3. Data

- One "game" = all lives until game over (or the ALE 108,000-frame limit). Actions are sampled from the policy
  (Sample Factory's `enjoy` default) with a per-game RNG seeded by the game seed. The same game seed in two
  conditions gives the same noop start and random draws (common random numbers).
- Discovery split: game seeds 0-99. Held-out split: game seeds 10000-10099. Held-out data is used only by
  confirmatory analyses after the addendum commit.
- Stochasticity check (first discovery analysis): for discovery games with identical noop counts, the first
  step where RAM differs. If the median over pairs exceeds 200 agent steps, all data is re-collected with ALE
  sticky actions (p = 0.25) and that setting is used everywhere.
- Steps where Ms. Pac-Man is not "in control" (she does not move between t-1, t, t+1: death animation, ready
  screen, level transition) are excluded from the audit and from per-step metrics.
- Pilot data used for engineering only (throughput, feature sanity): game seeds 900001-900024, and one game
  (seed 0) used to discover the frightened RAM state. Neither touches neuron activations.

## 4. Ground-truth features (40, from RAM + maze graph; `src/pacneuron/features.py`)

Families: player (x, y, heading, open directions, junction), danger (BFS/Euclidean distance to the nearest and
second-nearest non-edible ghost, first-step direction to it, number within 5 steps), edible (frightened flag,
frightened timer `ram[116]`, number of edible ghosts, distance and direction to the nearest edible ghost),
food (pellet/power-pill distance, direction and counts), game (maze id, life lost within 10 steps, raw reward
of the transition, ghost eaten). Agent outputs (value, entropy, action) are not features; they are reported
separately through direct logit attribution.

## 5. H1 - privileged basis ("already clean")

- Units: alive fc units (post-ReLU > 0 on at least 0.1% of in-control discovery steps), positive pole
  (post-ReLU activation).
- Single-feature fit R²₁(u, f): per-bin means of the unit's activation over 20 quantile bins of f (category
  values for heading and maze), fitted on discovery, scored on held-out (out-of-sample R²).
- Ceiling R²_all(u): `HistGradientBoostingRegressor` (scikit-learn defaults, max_iter=200, random_state=0) on
  all 40 features, fitted on at most 50,000 in-control discovery steps, scored on held-out.
- A unit is **clean** if R²_all ≥ 0.3 and max_f R²₁ / R²_all ≥ 0.7.
- Controls, scored the same way:
  - (a) 512 random orthonormal directions of the post-ReLU fc activations (primary control);
  - (b) 512 non-negative random mixtures of 2-8 alive units (Dirichlet(1) weights);
  - (c) the 512 fc units of the untrained control network.
- Test: difference in clean fraction (agent minus control). 95% CI from a game-cluster bootstrap over held-out
  games (10,000 resamples). **H1 is supported if the CI against control (a) excludes 0 in the predicted
  direction; it is falsified if the CI includes 0 or is negative.** Controls (b) and (c) are secondary. The
  threshold sensitivity of the clean fraction (R²_all and ratio thresholds swept) is reported as a curve.
- The absolute clean fraction is a lower bound, because the feature dictionary is finite.

## 6. H2 - steering (primary deliverable)

- Candidates: K ≤ 3 tuples (layer, unit, spatial cell if conv, intervention, dose, gate, primary metric,
  predicted sign). They are chosen using discovery data only: audit, open-loop patch screen, closed-loop pilots
  on discovery seeds. They are frozen in the addendum. Allowed interventions:
  - (i) set a conv3 channel at one 7x7 cell to a dose;
  - (ii) zero-ablate an fc unit;
  - (iii) clamp an fc unit to a dose.
  (iii) is mathematically a constant logit bias and counts only as a positive control, never as the H2
  "interesting" neuron.
- Dose: the unit's discovery 99th percentile of its positive activations at that site (for conv3: at that
  cell). Gate: always on, or only when frightened / not frightened.
- Primary metrics (the addendum picks exactly one per candidate):
  - **approach(c)**: over in-control steps, P(BFS distance from the player to the target lattice node c
    decreases within the next 8 steps). The target node is the main-component node closest to the conv3
    cell's receptive-field centre.
  - **ghosts_per_pill**: ghosts eaten per power pill consumed.
  - **action_given_trigger**: P(action in set A | trigger feature true).
  - **deaths_per_1k**: life losses per 1,000 in-control steps.
- Arms on held-out seeds:
  - baseline: 100 games;
  - each candidate: 50 games (seeds 10000-10049, paired with the baseline);
  - null: 20 games each (seeds 10000-10019). For conv3 candidates the null is the 63 other channels set at
    the same cell to their own dose. For fc candidates it is 99 random alive fc units with the same
    intervention type.
- Effect Δ = mean over paired games of (metric_steered - metric_baseline).
- **A candidate passes H2 if all hold:**
  1. Δ has the predicted sign, and a paired sign-flip permutation test (10,000 permutations) gives p < α
     after Holm correction across the K candidates (α = 0.05). The 95% game-bootstrap CI of Δ is reported.
  2. Specificity against the null: signed Δ exceeds the 95th percentile of the null units' signed Δ.
  3. Secondary: off-target metrics (pellets per 1k steps, deaths per 1k steps, return) are reported with
     Benjamini-Hochberg-corrected tests; large off-target damage is reported as a caveat, not a fail.
- **H2 is supported if at least one candidate passes; it is falsified if none does.**
- Sample size rationale: from the pilot, returns have a CV of about 0.3, so a 20% effect needs about 38 paired
  games at α = 0.05 and power 0.8. Event-level metrics such as approach rates average over thousands of steps
  per game and are far less noisy.

## 7. H3 - replication across seeds

For every candidate that passes H2, the addendum states a deterministic selection rule that finds the
analogous unit in seeds 2222 and 3333 from their own discovery data. Example: "the conv3 channel with the
highest discovery R²₁ for feature f at cell c". The same intervention, dose rule, metric and test (Holm over
seeds) are then applied. H3 is supported for a candidate if both seeds pass criteria 1-2.

## 8. Exploratory (labelled as such in every report)

- Negative poles (most negative fc pre-activations; not steerable downstream of the ReLU).
- Dose-response curves.
- Gating variants not frozen in the addendum.
- conv3 channel atlas.
- Training dynamics on the 36 milestone checkpoints of seed 1111.
- Any unit chosen after seeing held-out data.

## 9. Reporting

- Every frozen candidate is reported, whether it passes or fails.
- Effect sizes for confirmatory claims come from held-out data only.
- Every number in reports, README and paper is generated by `pacneuron.report` from `results/*.json`.
- Deviations from this plan are listed in `reports/00_summary.md` with reasons.
