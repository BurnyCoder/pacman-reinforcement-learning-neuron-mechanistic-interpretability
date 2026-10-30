# Experiment 3: Neuron audit — do the agent's neurons have "one main job"?

## Question and hypothesis (preregistered H1)
- **Claim tested.** Goodfire argues that most neuron poles of trained networks have one main job
  (https://already-clean.goodfire.com/).
- **Prediction.** For the 512-unit fc layer of the Ms. Pac-Man agent, the fraction of alive units well explained
  by a *single* ground-truth feature is larger than for random directions of the same layer (the
  privileged-basis test), for non-negative random mixtures of units, and for an untrained network.

## Method
- **Data.** Activations of every unit at every in-control step:
  - discovery: 100 games, game seeds 0–99;
  - held-out: 100 games, seeds 10000–10099;
  - ALE sticky actions p = 0.25 (preregistered fallback, see `00_summary.md`).
- **Score per unit.**
  - R²₁ = out-of-sample R² of the unit against bin means of one feature (20 quantile bins; categories for
    heading and maze).
  - R²_all = out-of-sample R² of `HistGradientBoostingRegressor` on all 40 features.
  - A unit has **one job** if R²_all ≥ 0.3 and max R²₁ ≥ 0.7 · R²_all.
  - Positive pole = post-ReLU activation. Negative pole = the most negative pre-activations; it is shown in the
    microscope but cannot be steered (the ReLU removes it).
- **Controls.**
  - (a) 512 random orthonormal directions of the post-ReLU fc space;
  - (b) 512 non-negative Dirichlet mixtures of 2–8 alive units;
  - (c) the fc units of an orthogonally initialised untrained network evaluated on the same states.
- **Statistics.** Game-cluster bootstrap (10,000 resamples) of the difference in one-job fraction, plus a
  robustness bootstrap over start-state clusters.
- **Splits.** Exploratory: fit on even discovery games, score on odd ones. Confirmatory: fit on all discovery
  games, score on held-out games.

## Exploratory audit (discovery split-half)
<!-- BEGIN:audit_appo_1111 -->
<!-- END:audit_appo_1111 -->

![one-job fraction, discovery](../figures/clean_fraction_discovery_appo_1111.png)

## Confirmatory H1 (held-out games)
<!-- BEGIN:h1_appo_1111 -->
<!-- END:h1_appo_1111 -->

![one-job fraction, held-out](../figures/clean_fraction_appo_1111.png)
![per-unit R²](../figures/r2_scatter_appo_1111.png)

## Exploratory follow-up: conjunctions (post hoc, discovery split-half)
Units are well predicted by all features together but not by any single one. So we asked which *pairs* of
variables explain them, using joint bins on the maze lattice:

<!-- BEGIN:conjunctions_appo_1111 -->
<!-- END:conjunctions_appo_1111 -->

Ms. Pac-Man's lattice tile combined with her heading explains fc units far better than any single feature. Many
fc neurons behave like **place × heading cells** (cf. hippocampal place cells). Ghost-related variables, even as
conjunctions, explain little. This is consistent with [Greydanus et al. 2018](https://arxiv.org/abs/1711.00138):
their Ms. Pac-Man saliency maps showed agents tracking corridors rather than ghosts.

This analysis was not preregistered and is evidence for a hypothesis to test, not a confirmed claim.

## conv3 channels
- **Test.** Within-cell AUROC of each channel for the presence of each object type in its receptive-field core.
  Location is controlled by comparing only within a cell.
- **Result.** It is weak for every channel (best <!--n:appo_1111:c3_best_obj-->0.63<!--/n-->; see the microscope site).
  conv3 channels are not clean object detectors.
- **Global states.** Some channels track global game states better. Channel 20's spatial mean is lower when ghosts
  are edible (AUROC <!--n:appo_1111:c3_ch20_frightened-->0.25<!--/n--> for frightened mode) and higher when a
  dangerous ghost is near (<!--n:appo_1111:c3_ch20_n_danger_near-->0.66<!--/n-->); exploratory, from
  `results/conv3_states_appo_1111.json`.

## Learnings
- Under a 40-feature, single-variable dictionary, the neurons of this reward-trained agent are **not** "already
  clean": most are conjunctive. Goodfire's free-text visual labels can describe conjunctions ("dog heads in
  profile"); a RAM-feature dictionary needs conjunctions to compete.
- <!--n:appo_1111:dead-->145<!--/n--> of 512 fc units never fire on discovery data (dead units). They are excluded from H1 and listed in the
  microscope.
