---
title: "Are the neurons of a reward-trained Ms. Pac-Man agent already clean? A preregistered neuron audit with causal steering"
author: "Libor \"Burny\" Burian"
date: "October 2026"
---

# Abstract

Goodfire's essay "Neural Networks Are Already Clean" argues that the individual neurons of trained vision and
language networks are largely interpretable; in its preliminary Qwen3-32B audits, about 90% of neuron poles have
"one main job". We test this claim, and whether single units can steer behaviour, in three deep RL agents that
learned Atari Ms. Pac-Man from game reward alone. They are public Sample Factory APPO agents with a Nature-CNN torso,
trained for 2B emulator frames (about 500M agent steps).

We re-implement the agents in PyTorch and check them against the original stack: observations are bit-exact and
logits agree to <!--n:fid_logit-->1.1e-05<!--/n--> on CPU (to <!--n:fid_logit_gpu-->4.5e-02<!--/n--> on the GPU used
for data collection). We decode exact game state from the console RAM, including frightened-mode registers that we
identified independently from sprite colours (<!--n:appo_1111:edible_agree-->97.1%<!--/n--> per-ghost agreement;
already documented in jev_deep_rl), and label every unit against 40 ground-truth features under a preregistered
protocol.

**Not "already clean".** On held-out games, the fraction of alive fc neurons with one main job is
<!--n:appo_1111:h1_agent-->0.8%<!--/n-->, vs <!--n:appo_1111:h1_random_dirs-->0.2%<!--/n--> for random directions and
<!--n:appo_1111:h1_untrained-->9.4%<!--/n--> for an untrained network. The preregistered privileged-basis test is
<!--n:appo_1111:h1_verdict-->supported<!--/n--> for agent 1111, but the effect is trivially small. It is
<!--n:appo_2222:h1_verdict-->not supported<!--/n--> for agent 2222 and <!--n:appo_3333:h1_verdict-->not supported<!--/n-->
for agent 3333. Exploratorily, a lookup table over Ms. Pac-Man's maze position × heading explains a median
out-of-sample R² of <!--n:appo_1111:place_heading_median-->0.20<!--/n--> per neuron, vs
<!--n:appo_1111:median_best_disc-->0.03<!--/n--> for the best single feature, which suggests conjunctive, place-like
codes.

**One unit, a no-go zone.** Two steering candidates were frozen before any held-out data. Setting a single conv3
unit (channel 18 at one spatial position) to its 99th percentile makes the agent avoid the corresponding maze
region: approach rate <!--n:appo_1111:c3_ch18_cell56:base-->0.424<!--/n--> →
<!--n:appo_1111:c3_ch18_cell56:arm-->0.361<!--/n-->, Holm p <!--n:appo_1111:c3_ch18_cell56:p-->0.0002<!--/n-->. This
effect is stronger than that of each of the 63 other channels at that position, each set to its own 99th percentile
(the preregistered H2 passes). Ablating fc neuron 453, which votes UP, reduces moving up toward edible ghosts but does
not significantly reduce ghosts eaten per power pill (H2 fails for this candidate). Its best direction label ("edible
ghost is up") does not survive confound control.

**Replication.** Frozen selection rules applied to the other two agents give the following (H3):

- no-go-zone rule: <!--n:h3:c3_ch18_cell56-->replicates<!--/n-->;
- ghost-hunting ablation rule: <!--n:h3:fc453_ablate-->does not replicate<!--/n--> (exploratory, because fc 453
  failed H2).

All tests, deviations, errata and negative results are reported.

# 1. Introduction

**Motivation.** Mechanistic interpretability asks whether the internal units of a network can be read as
human-understandable variables and whether manipulating them changes behaviour in the predicted way.

**The claim we test.** Goodfire recently argued that the neurons of trained networks are already largely
interpretable [(Cammarata, Goodfire 2026)](https://already-clean.goodfire.com/). In its preliminary audits of
Qwen3-32B, "roughly 90% of neuron poles have one main job"; it makes the same argument for vision models such as
InceptionV1 and ConvNeXt. Its evidence combines top-activating examples, feature visualisation, steering and
attribution graphs.

**Why RL is a sharp test.** Reinforcement-learning agents trained from reward alone offer two advantages:

- the environment state is fully known (for Atari, the console RAM);
- behaviour can be measured causally in closed loop.

**Gap in prior work.** Prior interpretability work on Ms. Pac-Man used:

- input saliency [(Greydanus et al. 2018)](https://arxiv.org/abs/1711.00138);
- attention maps [(Mott et al. 2019)](https://arxiv.org/abs/1906.02500);
- NMF concepts [(Bekkemoen & Langseth 2025)](https://mlanthology.org/mlj/2025/bekkemoen2025mlj-interpretable);
- a sparse autoencoder without interventions
  [(DuPlessie 2024)](https://math.mit.edu/research/highschool/primes/materials/2024/DuPlessie.pdf).

Single-channel steering of RL agents has been shown for a Procgen maze agent
[(Mini et al. 2023)](https://arxiv.org/abs/2310.08043). In a Sokoban planning agent, plans have been steered with
probe directions [(Taufeeque et al. 2024](https://arxiv.org/abs/2407.15421); [Bush et al. 2025)](https://arxiv.org/abs/2504.01871)
and with channel-level interventions [(Taufeeque et al. 2025)](https://arxiv.org/abs/2506.10138). To our knowledge,
no published work has steered individual neurons of a reward-trained Ms. Pac-Man agent.

**Contributions.**

1. A dependency-light re-implementation of the public Sample Factory Ms. Pac-Man agents with a golden fidelity
   test (bit-exact observations; logits agree to 1.1e-5 on CPU).
2. RAM ground truth for Ms. Pac-Man, including a pixel validation of the frightened-mode register `ram[116]` and the
   per-ghost edible bits (bit 7 of `ram[1+i]`), and a maze lattice graph. We found these registers independently;
   [jev_deep_rl](https://github.com/taodav/jev_deep_rl) documented them first.
3. A preregistered Goodfire-style audit of every alive fc neuron against three controls (H1).
4. Preregistered causal steering tests with matched null units on held-out games (H2), with replication
   rules for two more agents (H3).

# 2. Background and related work

- **Neuron-level interpretability.**
  - Circuits [(Olah et al. 2020)](https://distill.pub/2020/circuits/zoom-in/).
  - Polysemanticity and superposition, which motivate dictionary learning
    [(Bricken et al. 2023)](https://transformer-circuits.pub/2023/monosemantic-features/index.html).
  - Goodfire's counter-argument that neurons are mostly clean [(Cammarata, Goodfire 2026)](https://already-clean.goodfire.com/).
  - Direct logit attribution [(Elhage et al. 2021)](https://transformer-circuits.pub/2021/framework/index.html) and
    occlusion maps [(Zeiler & Fergus 2014)](https://arxiv.org/abs/1311.2901).
- **RL interpretability.**
  - An Atari model zoo with activation analysis [(Such et al. 2019)](https://arxiv.org/abs/1812.07069).
  - CoinRun attribution and model editing [(Hilton et al. 2020)](https://distill.pub/2020/understanding-rl-vision/).
  - Probe-based steering of planning in Sokoban [(Bush et al. 2025)](https://arxiv.org/abs/2504.01871).
  - Attention maps of Ms. Pac-Man agents [(Mott et al. 2019)](https://arxiv.org/abs/1906.02500).
- **Ground truth for Atari.**
  - AtariARI RAM annotations [(Anand et al. 2019)](https://arxiv.org/abs/1906.08226).
  - OCAtari object extraction [(Delfosse et al. 2023)](https://arxiv.org/abs/2306.08649).
  - Ms. Pac-Man frightened-mode registers [(jev_deep_rl, 2026)](https://github.com/taodav/jev_deep_rl).
- **Evaluation protocol.** Sticky actions [(Machado et al. 2018)](https://arxiv.org/abs/1709.06009).

# 3. Method

## 3.1 Agents and fidelity

- **Agents.** `edbeeching/atari_2B_atari_mspacman_{1111,2222,3333}`:
  - Sample Factory 2.0 APPO [(Petrenko et al. 2020)](https://arxiv.org/abs/2006.11751), an asynchronous variant
    of PPO [(Schulman et al. 2017)](https://arxiv.org/abs/1707.06347); 2B emulator frames (about 500M agent steps at
    frameskip 4), clipped reward, γ = 0.99, no RNN;
  - Nature-CNN torso [(Mnih et al. 2015)](https://www.nature.com/articles/nature14236) (conv 32/64/64; Sample
    Factory's `convnet_atari`) → fc 512 ReLU → linear policy (9 actions) and value heads.
- **Environment.** The wrapper chain is `MsPacmanNoFrameskip-v4` → no-op reset → max-and-skip 4 → episodic life →
  reward clipping → 84×84 grayscale → stack 4.
- **Fidelity.** We reproduce the wrapper chain and network in PyTorch and verify both against a golden trajectory
  generated with sample-factory 2.0.3, gym 0.26.2 and ale-py 0.8.1:
  - 0 mismatched observations in <!--n:fid_steps-->446<!--/n--> steps;
  - max logit difference <!--n:fid_logit-->1.1e-05<!--/n--> on CPU. On the GPU used for all data collection
    (PyTorch's default cuDNN TF32 convolutions) it is <!--n:fid_logit_gpu-->4.5e-02<!--/n-->. The test covers CPU
    only (Section 6).

  Appendix A has the details.

## 3.2 Ground truth

- **RAM decoding** follows OCAtari and AtariARI.
- **Frightened mode** was identified by correlating RAM bytes with blue ghost pixels and power-pill events. The
  same registers were documented earlier in [jev_deep_rl](https://github.com/taodav/jev_deep_rl), which also notes
  that only the low six bits of `ram[116]` count down.
- **Maze.** A 14×18 lattice graph built from the frame gives breadth-first-search distances and directions.
- **Features.** 40 features in five families (Appendix B). A step is excluded when Ms. Pac-Man's RAM position is
  unchanged from t−1 to t and from t to t+1. This removes death animations, ready screens and level transitions,
  and also standstills such as pushing into a wall.

## 3.3 Data and stochasticity

- **Splits.** 100 discovery games (seeds 0–99) and 100 held-out games (seeds 10000–10099) per agent.
- **Action sampling.** Actions are sampled from the policy with a per-game RNG, so steered and baseline games share
  random numbers.
- **Stochasticity fallback.** A preregistered check found that without sticky actions, same-start games stayed
  identical for a median of <!--n:appo_1111:div_nosticky-->1230<!--/n--> steps. The preregistered fallback (sticky
  actions, p = 0.25) was therefore used for all data (median divergence
  <!--n:appo_1111:div_sticky-->320<!--/n--> steps).

## 3.4 Audit (H1)

- **Per-unit score.** For each alive fc unit (active on ≥ 0.1% of in-control discovery steps; positive pole =
  post-ReLU activation):
  - R²₁, the out-of-sample R² of bin means of each single feature (20 quantile bins; features with ≤ 20 distinct
    values use their values);
  - R²_all, the out-of-sample R² of histogram gradient boosting (scikit-learn; LightGBM-style,
    [Ke et al. 2017](https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree))
    on all features.
- **One-job criterion.** R²_all ≥ 0.3 and max R²₁ ≥ 0.7 · R²_all.
- **Controls.**
  - (a) random orthonormal directions;
  - (b) non-negative mixtures of 2–8 units;
  - (c) an untrained, orthogonally initialised network [(Saxe et al. 2014)](https://arxiv.org/abs/1312.6120) on the
    same states (its alive units; init seed 0 for every agent).
- **Test.** Game-cluster bootstrap [(Field & Welsh 2007)](https://doi.org/10.1111/j.1467-9868.2007.00593.x) of the
  difference in one-job fraction. The fitted models are fixed; only the scored games are resampled. H1 is supported
  if the 95% CI against (a) excludes 0.

## 3.5 Steering (H2) and replication (H3)

- **Why intervene upstream of fc.** Because the heads are linear and there is no recurrence:
  - clamping fc unit n to v replaces its state-dependent vote h_n(s)·W_π[:, n] with the constant v·W_π[:, n], so it
    is a constant action bias wherever the unit is inactive;
  - ablating it removes the state-dependent vote;
  - conv3 interventions act through 512 ReLUs.
- **Candidate selection** (discovery data only):
  1. Exact open-loop screens, using a rank-1 update for single conv3 units.
  2. Closed-loop pilots on discovery seeds.
  3. Two candidates frozen in a committed addendum.
- **Confirmatory design** (held-out seeds):
  - 100 baseline games, 50 games per candidate, and 20 games per null unit;
  - nulls: 99 random alive fc units, or the 63 other conv3 channels at the same cell, each at its own p99;
  - pass criteria: predicted sign with a Holm-corrected [(Holm 1979)](https://www.jstor.org/stable/4615733) paired
    sign-flip p < 0.05 (10,000 permutations, so p ≥ 1/10,001;
    [Phipson & Smyth 2010](https://arxiv.org/abs/1603.05766)), **and** an effect beyond the 95th percentile of the
    null;
  - off-target metrics: two-sided tests, Benjamini–Hochberg-corrected
    [(Benjamini & Hochberg 1995)](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x);
  - approach rate: the share of in-control steps after which the BFS distance to the target node is smaller 8 agent
    steps later.
- **Replication (H3).** Deterministic selection rules transfer the procedure to agents 2222 and 3333. A rule
  replicates if its unit passes both criteria in both agents, after a further Holm correction over the two agents.
  The preregistration defines H3 only for candidates that pass H2. fc 453 failed, so its rule's replication is
  exploratory.
- **Pairing check (exploratory).** An unsteered A/A arm replays 50 held-out seeds in the steering runner. In agent
  1111, <!--n:appo_1111:aa_identical-->50 of 50<!--/n--> games match their paired baseline action for action
  (2222: <!--n:appo_2222:aa_identical-->50 of 50<!--/n-->; 3333: <!--n:appo_3333:aa_identical-->50 of 50<!--/n-->).
  So GPU batching does not break the seed pairing.

# 4. Results

## 4.1 Fidelity and ground truth
<!-- BEGIN:fidelity -->
| check | value |
|---|---|
| golden trajectory steps (incl. 2 life losses and game over) | 446 |
| steps with any observation pixel mismatch | 0 |
| max abs. logit difference (CPU) | 1.1e-05 |
| max abs. value difference (CPU) | 4.8e-07 |
| max abs. logit difference on GPU (NVIDIA GeForce RTX 5070 Laptop GPU, cuDNN TF32 on, as in data collection) | 4.5e-02 |
| reference stack | sample-factory 2.0.3, gym 0.26.2, ale-py 0.8.1, torch 1.13.1 (WSL) |
<!-- END:fidelity -->

<!-- BEGIN:ground_truth_appo_1111 -->
| validation (discovery games, sticky actions) | value |
|---|---|
| agent steps checked | 161995 (100 games) |
| RAM edible bit (bit 7 of ram[1+i]) vs blue sprite pixels, per ghost-step | 97.1% |
| frightened timer ram[116] > 0 vs any ghost edible | 100.0% |
| eaten pellets at Ms. Pac-Man's lattice node (26825 events) | 100.0% |
| steps in control (position changed between t−1 and t or t and t+1; excludes death, ready, level transitions and standstills) | 85.1% |
| distinct maze layouts seen | 4 |
| edible-bit vs blue-pixel disagreements (18585 ghost-steps): bit set but no blue pixels | 95.5% |
| ... median countdown (low 6 bits of ram[116]) at a disagreement / share at ≤ 10 (all edible ghost-steps: 7.1%) | 13 / 38.2% |
| countdown start value when a pill is eaten in normal mode (207 pills) | 38–62 |
| countdown drop per 4 agent steps (modal value, share) | 2 (97.5%) |
| frightened steps with the upper bits of ram[116] set | 18.5% |
<!-- END:ground_truth_appo_1111 -->

## 4.2 H1: one-job neurons
<!-- BEGIN:h1_appo_1111 -->
| unit set (held-out, confirmatory) | units | one-job fraction [95% CI] | median R²_all | median best single R² |
|---|---|---|---|---|
| trained agent (fc units) | 367 | 0.8% [0.8%, 0.8%] | 0.708 | 0.031 |
| random directions (control a) | 512 | 0.2% [0.2%, 0.6%] | 0.719 | 0.185 |
| non-negative unit mixtures (control b) | 512 | 1.4% [1.2%, 1.8%] | 0.720 | 0.053 |
| untrained network (control c) | 499 | 9.4% [5.2%, 11.6%] | 0.537 | 0.096 |

H1 test (agent minus random directions): 0.6% [0.2%, 0.6%]; H1 supported.
<!-- END:h1_appo_1111 -->

![](../figures/clean_fraction_appo_1111.png)

![](../figures/r2_scatter_appo_1111.png)

Robustness. The table below adds a start-state-cluster bootstrap that resamples groups of games sharing their first
in-control RAM state. There are only <!--n:appo_1111:n_start_groups-->4<!--/n--> such groups among the held-out games,
so these CIs are coarse. The threshold sweep shows that agent 1111's small advantage over random directions reverses
at a looser ratio threshold: at a ratio of 0.5, random directions score
<!--n:appo_1111:sweep_0.3_0.5:random_dirs-->8.0%<!--/n--> vs the agent's
<!--n:appo_1111:sweep_0.3_0.5:agent-->1.4%<!--/n-->.

<!-- BEGIN:h1robust_appo_1111 -->
| unit set | one-job fraction | game-bootstrap 95% CI | start-state-cluster 95% CI |
|---|---|---|---|
| trained agent (fc units) | 0.8% | [0.8%, 0.8%] | [0.8%, 0.8%] |
| random directions (control a) | 0.2% | [0.2%, 0.6%] | [0.2%, 0.4%] |
| non-negative unit mixtures (control b) | 1.4% | [1.2%, 1.8%] | [1.4%, 2.0%] |
| untrained network (control c) | 9.4% | [5.2%, 11.6%] | [4.8%, 11.0%] |

Threshold sweep, one-job fraction shown as agent / random directions (rows: R²_all threshold; columns: best-single / R²_all ratio threshold). Start-state clusters: 4.

| R²_all ≥ | ratio ≥ 0.5 | 0.6 | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|
| 0.1 | 1.6% / 8.0% | 0.8% / 1.6% | 0.8% / 0.2% | 0.0% / 0.0% | 0.0% / 0.0% |
| 0.2 | 1.6% / 8.0% | 0.8% / 1.6% | 0.8% / 0.2% | 0.0% / 0.0% | 0.0% / 0.0% |
| 0.3 | 1.4% / 8.0% | 0.8% / 1.6% | 0.8% / 0.2% | 0.0% / 0.0% | 0.0% / 0.0% |
| 0.4 | 1.1% / 8.0% | 0.8% / 1.6% | 0.8% / 0.2% | 0.0% / 0.0% | 0.0% / 0.0% |
| 0.5 | 1.1% / 8.0% | 0.8% / 1.6% | 0.8% / 0.2% | 0.0% / 0.0% | 0.0% / 0.0% |
<!-- END:h1robust_appo_1111 -->

Exploratory conjunctions (not preregistered; discovery split-half, fitted on even games and scored on odd ones). Bin
counts differ: place has up to 252 lattice nodes and place × heading up to 1,008 bins, while single features have
≤ 20 bins.
Gradient boosting on all 40 features reaches a median of <!--n:appo_1111:median_all_disc-->0.72<!--/n--> on the same
split:

<!-- BEGIN:conjunctions_appo_1111 -->
| explanatory variable(s) (exploratory, post hoc) | median R² | units with R² > 0.3 | units with R² > 0.5 |
|---|---|---|---|
| place (lattice node) | 0.127 | 11.7% | 2.5% |
| place x heading | 0.202 | 30.5% | 7.4% |
| place x frightened | 0.165 | 23.4% | 6.0% |
| pellets left (20 bins) x frightened | 0.022 | 2.7% | 1.1% |
| danger distance (20 bins) x danger direction | 0.018 | 0.3% | 0.0% |
<!-- END:conjunctions_appo_1111 -->

## 4.3 H2: steering
<!-- BEGIN:steer_appo_1111 -->
| intervention | primary metric (predicted sign) | baseline → steered | Δ [95% CI] | Holm p | above null 95th pct? | H2 |
|---|---|---|---|---|---|---|
| fc neuron 453 zero-ablated | ghosts_per_pill (−) | 0.294 → 0.228 | -0.066 [-0.192, 0.101] | 0.2280 | yes (null q95 0.013) | fail |
| conv3 channel 18 set to its p99 at cell (5,6) | approach (−) | 0.424 → 0.361 | -0.063 [-0.077, -0.050] | 0.0002 | yes (null q95 0.013) | pass |
| fc neuron 188 clamped to its p99 (constant logit bias, DLA UP -0.70) (positive control) | action_given_trigger (−) | 0.395 → 0.305 | -0.090 [-0.102, -0.079] | – | – | – |

p = 0.0001 is the smallest attainable value of the 10,000-permutation sign-flip test (1/10,001); 0.0002 is that floor after Holm over 2 tests, so it means p ≤ that value. The positive control's frozen label says "constant logit bias": clamping unit n to v shifts the logits by (v − h_n(s))·W_π[:, n], which is constant only on steps where the unit is inactive (fc 188: active on 2.1% of in-control discovery steps).
<!-- END:steer_appo_1111 -->

Off-target effects (two-sided paired sign-flip tests, Benjamini-Hochberg-corrected):

<!-- BEGIN:offtarget_appo_1111 -->
| intervention | off-target metric | Δ [95% CI] | two-sided p | BH q |
|---|---|---|---|---|
| fc neuron 453 zero-ablated | return | -1619 [-2194, -1056] | 0.0001 | 0.0001 |
| fc neuron 453 zero-ablated | deaths_per_1k | 1.14 [0.75, 1.52] | 0.0001 | 0.0001 |
| fc neuron 453 zero-ablated | pellets_per_1k | -9.23 [-17.68, -0.84] | 0.0416 | 0.0468 |
| fc neuron 453 zero-ablated | in_control | -469 [-633, -304] | 0.0001 | 0.0001 |
| conv3 channel 18 set to its p99 at cell (5,6) | return | -2490 [-2990, -2005] | 0.0001 | 0.0001 |
| conv3 channel 18 set to its p99 at cell (5,6) | deaths_per_1k | 4.17 [3.27, 5.14] | 0.0001 | 0.0001 |
| conv3 channel 18 set to its p99 at cell (5,6) | pellets_per_1k | -12.16 [-20.78, -3.83] | 0.0071 | 0.0091 |
| conv3 channel 18 set to its p99 at cell (5,6) | ghosts_per_pill | -0.05 [-0.18, 0.09] | 0.5156 | 0.5156 |
| conv3 channel 18 set to its p99 at cell (5,6) | in_control | -788 [-944, -633] | 0.0001 | 0.0001 |

p = 0.0001 is the smallest attainable value of the 10,000-permutation sign-flip test (1/10,001), so it means p ≤ that value.
<!-- END:offtarget_appo_1111 -->

![](../figures/steer_appo_1111_fc453_ablate.png)

![](../figures/steer_appo_1111_c3_ch18_cell56.png)

![](../figures/occupancy_appo_1111_c3_ch18_cell56.png)

Attribution of the candidate units: exact upstream contributions, occlusion peak, and confound-controlled label
checks (stratified AUROC within frightened × player-region strata; unique R² = drop-one loss of the
gradient-boosting ceiling):

<!-- BEGIN:attribution_appo_1111 -->
| unit | strongest upstream units (mean exact contribution) | occlusion peak (screen x, y) | label checks: single-feature R² / stratified AUROC / unique R² |
|---|---|---|---|
| fc neuron 453 zero-ablated | conv3 ch 17 (+5.1), conv3 ch 59 (+4.4), conv3 ch 1 (+4.3) | (57, 90) | dir_edible_up: 0.14 / 0.59 / 0.003; n_pellets: 0.38 / – / 0.007; n_edible: 0.29 / – / 0.002 |
| conv3 channel 18 set to its p99 at cell (5,6) | conv2 ch 40 (+6.9), conv2 ch 12 (+5.5), conv2 ch 56 (+5.1) | (137, 165) | – |
<!-- END:attribution_appo_1111 -->

![](../figures/occlusion_appo_1111_c3_ch18_cell56.png)

![](../figures/occlusion_appo_1111_fc453_ablate.png)

The no-go-zone unit depends on input inside its own receptive field: Ms. Pac-Man is within 12 px of its occlusion
peak in <!--n:appo_1111:c3_ch18_cell56:near_peak-->98.0% of its 50 top states<!--/n--> (exploratory). fc 453's best
direction label ("edible ghost is up") does not survive confound control. The unit is active mostly in frightened
mode (<!--n:appo_1111:fc453_ablate:fright_active-->88.8%<!--/n--> of its active steps vs
<!--n:appo_1111:fc453_ablate:fright_base-->40.3%<!--/n--> overall), and Ms. Pac-Man is near its occlusion peak in
<!--n:appo_1111:fc453_ablate:near_peak-->80.0% of its 50 top states<!--/n-->. Those top states largely repeat one
early-game moment (<!--n:appo_1111:fc453_ablate:top_repeat-->86.0% of them within 10 steps of step 379, from 46 games<!--/n-->),
so its occlusion map describes one situation. We therefore attribute its ablation effect to its UP vote rather than
to a ghost-direction representation.

Discovery-phase evidence that led to the candidates (exploratory):

<!-- BEGIN:screen_appo_1111 -->
| rank | conv3 channel @ cell | Δ P(move toward cell) at junctions | dose (p99) |
|---|---|---|---|
| repel 1 | ch 18 @ (5, 6) | -5.8 pp | 24.6 |
| repel 2 | ch 15 @ (0, 6) | -5.1 pp | 27.2 |
| repel 3 | ch 40 @ (4, 6) | -4.0 pp | 34.8 |
| repel 4 | ch 36 @ (5, 6) | -3.6 pp | 22.7 |
| repel 5 | ch 58 @ (5, 6) | -3.6 pp | 18.4 |
| attract 1 | ch 16 @ (6, 2) | +3.6 pp | 23.0 |
| attract 2 | ch 8 @ (1, 0) | +3.0 pp | 21.6 |
| attract 3 | ch 38 @ (5, 0) | +3.0 pp | 20.0 |
| attract 4 | ch 34 @ (6, 0) | +2.7 pp | 28.1 |
| attract 5 | ch 46 @ (6, 0) | +2.6 pp | 14.5 |
<!-- END:screen_appo_1111 -->

<!-- BEGIN:pilot_appo_1111 -->
| intervention (discovery pilot, paired games) | targeted metric | baseline → steered | Δ [95% CI] | Δ return | Δ deaths / 1k |
|---|---|---|---|---|---|
| `c3_ch20_ablate` | deaths_per_1k | 2.536 → 2.730 | 0.194 [-0.183, 0.671] | -210 | 0.19 |
| `c3_ch20_clamp_fright` | ghosts_per_pill | 0.392 → 0.213 | -0.179 [-0.292, -0.060] | -3668 | 5.28 |
| `c3_ch20_clamp` | deaths_per_1k | 2.536 → 9.071 | 6.535 [5.265, 7.759] | -4047 | 6.54 |
| `c3_ch39_ablate` | deaths_per_1k | 2.536 → 4.169 | 1.633 [0.646, 2.502] | -2566 | 1.63 |
| `c3_ch53_ablate` | deaths_per_1k | 2.536 → 2.665 | 0.129 [-0.035, 0.305] | -229 | 0.13 |
| `c3_ch18_cell56` | approach | 0.430 → 0.375 | -0.056 [-0.068, -0.044] | -3520 | 5.57 |
| `fc378_ablate` | action_given_trigger | 0.502 → 0.475 | -0.027 [-0.055, -0.000] | -1307 | 0.67 |
| `fc453_ablate` | action_given_trigger | 0.483 → 0.298 | -0.184 [-0.204, -0.164] | -2739 | 1.52 |
| `fc188_clamp` | action_given_trigger | 0.384 → 0.302 | -0.082 [-0.091, -0.072] | -3047 | 1.43 |
<!-- END:pilot_appo_1111 -->

## 4.4 H3: replication
<!-- BEGIN:h3 -->
| rule (frozen on agent 1111) | agent | selected unit | primary metric Δ [95% CI] | Holm p (within agent) | null units at least as strong | passes |
|---|---|---|---|---|---|---|
| zero-ablate the fc unit most needed for approaching edible ghosts | appo_2222 | fc neuron 379 zero-ablated | -0.101 [-0.167, -0.036] | 0.0025 | 0 of 99 | yes |
| zero-ablate the fc unit most needed for approaching edible ghosts | appo_3333 | fc neuron 351 zero-ablated | 0.190 [0.071, 0.301] | 0.9992 | 99 of 99 | no |
| set the most repelling conv3 (channel, cell) to its p99 | appo_2222 | conv3 channel 3 set to its p99 at cell (0, 0) | -0.034 [-0.043, -0.026] | 0.0002 | 0 of 63 | yes |
| set the most repelling conv3 (channel, cell) to its p99 | appo_3333 | conv3 channel 46 set to its p99 at cell (5, 6) | -0.216 [-0.232, -0.200] | 0.0002 | 0 of 63 | yes |

A rule replicates if its unit passes H2 criteria 1-2 in both agents and the within-agent Holm p-values stay < 0.05 after a further Holm correction over the two agents (preregistration §7, "Holm over seeds"). Preregistration §7 defines H3 only for candidates that passed H2; fc 453 failed H2 in agent 1111, so the fc-ablation rule's replication is exploratory. p = 0.0001 is the smallest attainable value of the 10,000-permutation sign-flip test (1/10,001); 0.0002 is that floor after Holm over 2 tests, so it means p ≤ that value.
<!-- END:h3 -->

Steering per agent (held-out):

<!-- BEGIN:steer_appo_2222 -->
| intervention | primary metric (predicted sign) | baseline → steered | Δ [95% CI] | Holm p | above null 95th pct? | H2 |
|---|---|---|---|---|---|---|
| fc neuron 379 zero-ablated | ghosts_per_pill (−) | 0.463 → 0.362 | -0.101 [-0.167, -0.036] | 0.0025 | yes (null q95 0.027) | pass |
| conv3 channel 3 set to its p99 at cell (0, 0) | approach (−) | 0.430 → 0.396 | -0.034 [-0.043, -0.026] | 0.0002 | yes (null q95 0.007) | pass |

p = 0.0001 is the smallest attainable value of the 10,000-permutation sign-flip test (1/10,001); 0.0002 is that floor after Holm over 2 tests, so it means p ≤ that value.
<!-- END:steer_appo_2222 -->

<!-- BEGIN:steer_appo_3333 -->
| intervention | primary metric (predicted sign) | baseline → steered | Δ [95% CI] | Holm p | above null 95th pct? | H2 |
|---|---|---|---|---|---|---|
| fc neuron 351 zero-ablated | ghosts_per_pill (−) | 0.660 → 0.850 | 0.190 [0.071, 0.301] | 0.9992 | no (null q95 0.019) | fail |
| conv3 channel 46 set to its p99 at cell (5, 6) | approach (−) | 0.440 → 0.224 | -0.216 [-0.232, -0.200] | 0.0002 | yes (null q95 0.012) | pass |

p = 0.0001 is the smallest attainable value of the 10,000-permutation sign-flip test (1/10,001); 0.0002 is that floor after Holm over 2 tests, so it means p ≤ that value.
<!-- END:steer_appo_3333 -->

# 5. Discussion

**Are the neurons clean?**

- **Not under single-feature labels.** With a 40-feature dictionary of single game variables, almost no fc neuron
  of these agents has one main job, and the agents are no cleaner than random directions in a meaningful way.
- **The basis is still not arbitrary.**
  - Random directions score *higher* best-single-feature R² than neurons (Fig. R²). Dense mixtures pick up slow
    global variables: pellets left is the best single feature for
    <!--n:appo_1111:best_feat_pellets:random_dirs-->472 of 512<!--/n--> random directions vs
    <!--n:appo_1111:best_feat_pellets:agent-->155 of 367<!--/n--> neurons (discovery split-half).
  - Neurons are sparse (the median alive unit is active on <!--n:appo_1111:median_active-->1.0%<!--/n--> of in-control
    steps). Exploratorily, they are conjunctive: position × heading explains them far better than any single
    feature, and position (itself a joint x × y variable) does most of that.
- **What the units are.** They resemble place cells more than "ghost detectors". Greydanus et al. report something
  similar: several partially trained A3C Ms. Pac-Man agents "appeared to be tracking corridors" rather than the ghosts
  [(Greydanus et al. 2018)](https://arxiv.org/abs/1711.00138). Attention-augmented agents, by contrast, scan the paths
  ahead for ghosts [(Mott et al. 2019)](https://arxiv.org/abs/1906.02500). Our agents are fully trained APPO agents,
  so both comparisons are loose.
- **Reconciling with Goodfire.** Goodfire's labels are free-text visual descriptions, which can express such
  conjunctions. Our result is a statement about a ground-truth *variable* basis. The conjunction analysis is post
  hoc and should be tested with a preregistered conjunctive dictionary.

**Can single units steer behaviour?**

- **Yes, causally and specifically, but not cleanly.** A single conv3 unit's activity is enough to repel the agent
  from a maze region. No other channel at that position does this at its own p99 dose, and the same selection rule
  finds such a unit in both independently trained agents (H3).
- **Steering is destructive.** The agent eats fewer pellets per step and dies more. Single-unit steering here is a
  blunt instrument, not a precise control knob.
- **Mechanism vs outcome.** The fc ablation shows a gap between mechanism and outcome. The predicted local
  behaviour ("go toward the edible ghost in direction d", a secondary metric without a null comparison) changes
  strongly in agents 1111 (<!--n:appo_1111:fc453_ablate:sec-->0.505 → 0.315 (Δ -0.190 [-0.209, -0.171])<!--/n-->) and
  3333 (<!--n:appo_3333:fc351_ablate:sec-->0.476 → 0.307 (Δ -0.169 [-0.188, -0.149])<!--/n-->), but barely in agent 2222
  (<!--n:appo_2222:fc379_ablate:sec-->0.549 → 0.526 (Δ -0.023 [-0.045, 0.001])<!--/n-->). The global outcome (ghosts
  eaten per pill) changes little in agent 1111, falls in agent 2222, and *rises* in agent 3333. Removing one
  direction's vote appears to reroute the agent rather than simply weaken ghost hunting.
- **Why the fc layer is constrained.** Because the policy head is linear and has no recurrence, clamping an fc unit
  can only swap its vote for a fixed action bias, and ablation can only remove the vote. State-dependent steering
  must enter upstream, where activations pass through the fc ReLUs.

**Methodological lessons.**

1. Bit-exact fidelity tests are cheap and necessary, and must cover the inference device. Library defaults changed
   under us (gymnasium's grayscale). Our CPU-only gate missed a <!--n:fid_logit_gpu-->4.5e-02<!--/n--> logit
   difference from cuDNN TF32 on the GPU. The A/A arm shows that it changed no sampled action in
   <!--n:appo_1111:aa_identical-->50 of 50<!--/n--> paired games (agent 1111).
2. Near-deterministic policies need injected stochasticity (sticky actions) before games can serve as
   independent samples.
3. Effects selected on a noisy outcome shrink on held-out data (winner's curse). fc 453's pilot drop in ghosts per
   pill (<!--n:appo_1111:pilot:fc453_ablate:ghosts_per_pill-->0.392 → 0.140 (Δ -0.252 [-0.331, -0.171])<!--/n-->)
   fell to Δ <!--n:appo_1111:fc453_ablate:delta-->-0.066 [-0.192, 0.101]<!--/n--> on held-out games. Event-level
   effects did not shrink: approach went from
   <!--n:appo_1111:pilot:c3_ch18_cell56:approach-->0.430 → 0.375 (Δ -0.056 [-0.068, -0.044])<!--/n--> to Δ
   <!--n:appo_1111:c3_ch18_cell56:delta-->-0.063 [-0.077, -0.050]<!--/n-->. Confirmatory effect sizes must come from
   held-out games.
4. A preregistered difference test needs a minimum effect size. Our H1 criterion was "passed" by a difference of
   <!--n:appo_1111:h1_diff-->0.6% [0.2%, 0.6%]<!--/n-->.

# 6. Limitations

- **Planned but not done.**
  - An input-pixel counterfactual for the top candidates (repainting ghosts as frightened in the observation) to
    separate ghost appearance from correlated game state.
  - A per-ghost "visible in the observation" flag (Atari sprite flicker can hide a ghost that RAM reports).
  - Training dynamics over the 36 milestone checkpoints of agent 1111.
  - Replication on agents trained with other algorithms (Stable-Baselines3 DQN/PPO).
  - The OCAtari cross-check of RAM decoding, replaced by pixel-level validation. OCAtari releases ≥ 1.1.6 pin
    numpy < 2, which is incompatible with this project's numpy ≥ 2.5, and the 1.1.5 source distribution failed to
    build in our environment.
- **Inference device.** All rollouts ran on a GPU with PyTorch's default cuDNN TF32 convolutions, which the CPU
  fidelity gate does not cover: logits differ from Sample Factory by up to
  <!--n:fid_logit_gpu-->4.5e-02<!--/n-->. The agent we analyse is therefore a very close numerical neighbour of the
  trained one, not a bit-exact copy. The A/A arm shows that the seed pairing is intact.
- **`ram[116]`.** Its upper bits change when ghosts are eaten (set on
  <!--n:appo_1111:gt_timer_upper-->18.5%<!--/n--> of frightened steps), and the frozen `fright_timer` feature used
  the raw byte. Masking it (exploratory) raises one feature's split-half R² by at most
  <!--n:appo_1111:mask_max_gain-->0.047<!--/n--> and makes no unit "clean"
  (<!--n:appo_1111:mask_newly_clean-->0<!--/n--> units).
- **Nulls and clusters.** Each null unit is estimated from 20 games against the candidate's 50, which makes the
  specificity criterion conservative. The start-state-cluster bootstrap has only
  <!--n:appo_1111:n_start_groups-->4<!--/n--> clusters.
- **Occlusion.** Maps average a unit's 50 top states. For fc 453 these mostly repeat one early-game moment, and
  occlusion cannot separate Ms. Pac-Man from the maze patch she occupies.
- **Feature dictionary.** The one-job criterion depends on a finite 40-feature dictionary of *single* variables, so
  conjunctive codes are counted as "not clean" by construction. The conjunction analysis is post hoc.
- **Sticky actions.** These change the agent's task (it was trained without them) and lower its score. All results
  describe the agent under p = 0.25.
- **Dependence between games.** Games that share a start state overlap for their first few hundred steps. We
  report a start-state-cluster bootstrap as a robustness check.
- **Scope of interventions.** Steering was tested at fixed doses (discovery p99) and only for the frozen
  intervention types; dose-response is exploratory.
- **Agents.** Three public agents of one architecture and algorithm. Generalisation to other algorithms (DQN, PPO
  with IMPALA-ResNet) is untested. The untrained control uses the same initialisation seed for all three agents.

# 7. Conclusion

**Clean?** In three reward-trained Ms. Pac-Man agents, individual neurons are not "already clean" with respect to
exact single game variables. They are sparse and, by exploratory evidence, conjunctive, place-like codes.

**Steerable?** Yes, causally and specifically. A single conv3 unit creates a no-go zone beyond all matched null
units, and its selection rule replicated in both other agents. Zero-ablating a single fc unit reduces a predicted
piece of ghost-chasing (moving up toward an edible ghost) without significantly changing ghosts eaten. These effects
come with substantial collateral damage.

**Toolkit.** The fidelity-checked models, RAM ground truth, preregistration and microscope make these agents a
convenient testbed for future work: conjunctive dictionaries, sparse autoencoders, and other algorithms.

# Appendix A. Fidelity details
See `reports/01_fidelity.md`.

# Appendix B. Feature list
See `src/pacneuron/features.py` and `reports/02_ram_ground_truth.md`.

# Appendix C. Preregistration and deviations
See `reports/preregistration.md`, `reports/preregistration_addendum.md` and `reports/00_summary.md`. The frozen
files are left unchanged; corrections to them are listed under "Errata and undeclared deviations" in
`reports/00_summary.md`.

# AI statement

This codebase and paper were written by Claude Code (Anthropic's Claude), following an exact specification from a
human author who directed the research. Human cleanup and rewriting are in progress.
