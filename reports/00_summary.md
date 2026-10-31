# Summary of all experiments

Author: Libor "Burny" Burian. Every marked number and table below is regenerated from `results/*.json` by
`uv run python -m pacneuron.pipeline --phase report`. Numbers quoted from the frozen preregistration files were written
before the results and are not regenerated.

## In one paragraph (plain language)
- **What we did.** We took video-game-playing neural networks that taught themselves Atari Ms. Pac-Man by trial and
  error (only the score as feedback, no human examples). We looked at their individual "neurons" in the way
  Goodfire inspects language and vision models.
- **How.** The game's memory tells us exactly where every ghost is, whether ghosts are edible, and how many pellets
  remain. So for each neuron we can ask: which game fact does it track? And when we turn it up or off during play,
  does the agent behave differently in the predicted way?
- **Labels.** Most neurons do not track one game fact. They respond to combinations: mostly *where Ms. Pac-Man is
  and which way she is heading*.
- **Steering.** A few units do change behaviour when steered. Setting a single unit in the last convolutional
  layer makes the agent avoid one region of the maze, like a phantom "no-go zone". This held up in all three
  agents, on games not used to choose the unit, and each time it beat all 63 other channels at the same position.
  Turning off a unit that votes "go up" makes the agent go up toward edible ghosts less often, but does not
  consistently change how many ghosts get eaten.

## Models
<!-- BEGIN:models -->
| agent | architecture | params | algorithm and objective | training data | discovery games (sticky p=0.25) | mean score ± SD | model-card score (no sticky) |
|---|---|---|---|---|---|---|---|
| [appo_1111](https://huggingface.co/edbeeching/atari_2B_atari_mspacman_1111) | Nature-CNN (SF convnet_atari): conv 32/64/64 -> fc 512 (ReLU) -> linear policy (9) + value | 1,689,258 | APPO (Sample Factory 2.0), maximise clipped game reward, gamma 0.99, no imitation | 2B emulator frames (≈500M agent steps at frameskip 4; cfg.json train_for_env_steps with summaries_use_frameskip) of interaction with MsPacmanNoFrameskip-v4 | 100 | 4082 ± 2032 | 6288 ± 1955 |
| [appo_2222](https://huggingface.co/edbeeching/atari_2B_atari_mspacman_2222) | Nature-CNN (SF convnet_atari): conv 32/64/64 -> fc 512 (ReLU) -> linear policy (9) + value | 1,689,258 | APPO (Sample Factory 2.0), maximise clipped game reward, gamma 0.99, no imitation | 2B emulator frames (≈500M agent steps at frameskip 4; cfg.json train_for_env_steps with summaries_use_frameskip) of interaction with MsPacmanNoFrameskip-v4 | 100 | 5161 ± 2056 | 9056 ± 2544 |
| [appo_3333](https://huggingface.co/edbeeching/atari_2B_atari_mspacman_3333) | Nature-CNN (SF convnet_atari): conv 32/64/64 -> fc 512 (ReLU) -> linear policy (9) + value | 1,689,258 | APPO (Sample Factory 2.0), maximise clipped game reward, gamma 0.99, no imitation | 2B emulator frames (≈500M agent steps at frameskip 4; cfg.json train_for_env_steps with summaries_use_frameskip) of interaction with MsPacmanNoFrameskip-v4 | 100 | 5056 ± 2358 | 6808 ± 1183 |
<!-- END:models -->

All three are public checkpoints by Edward Beeching on Hugging Face. They were trained with Sample Factory 2.0
APPO ([Petrenko et al. 2020](https://arxiv.org/abs/2006.11751)) purely from game reward, for 2B emulator frames
(about 500M agent steps), on `MsPacmanNoFrameskip-v4` with the standard DeepMind wrappers. No imitation learning was
involved.

## Experiments
| # | experiment | status | report |
|---|---|---|---|
| 1 | Fidelity gate: our PyTorch/Windows rebuild vs real Sample Factory | passed | [01_fidelity.md](01_fidelity.md) |
| 2 | RAM ground truth (incl. independently identified frightened-mode registers) | validated | [02_ram_ground_truth.md](02_ram_ground_truth.md) |
| 3 | Neuron audit, one-job fraction vs controls (H1) | H1 <!--n:appo_1111:h1_verdict-->supported<!--/n--> | [03_neuron_audit.md](03_neuron_audit.md) |
| 4 | Steering, frozen candidates vs matched nulls (H2) | H2 <!--n:appo_1111:h2_verdict-->supported<!--/n--> | [04_steering.md](04_steering.md) |
| 5 | Replication on agents 2222 and 3333 (H3) | see report | [05_replication.md](05_replication.md) |

Preregistration: [preregistration.md](preregistration.md) (before any data) and
[preregistration_addendum.md](preregistration_addendum.md) (frozen candidates, before held-out data).

## Key results (seed 1111)

**H1, one-job fraction on held-out games.**
- agent: <!--n:appo_1111:h1_agent-->0.8%<!--/n-->
- random directions: <!--n:appo_1111:h1_random_dirs-->0.2%<!--/n-->
- non-negative mixtures: <!--n:appo_1111:h1_nonneg_mix-->1.4%<!--/n-->
- untrained network: <!--n:appo_1111:h1_untrained-->9.4%<!--/n-->
- difference vs random directions: <!--n:appo_1111:h1_diff-->0.6% [0.2%, 0.6%]<!--/n-->

**Exploratory.** Place × heading explains a median out-of-sample R² of
<!--n:appo_1111:place_heading_median-->0.20<!--/n--> per fc unit (discovery split-half);
<!--n:appo_1111:place_heading_frac-->30.5%<!--/n--> of units exceed 0.3.

**H2, steering on held-out games (paired by seed):**
<!-- BEGIN:steer_appo_1111 -->
| intervention | primary metric (predicted sign) | baseline → steered | Δ [95% CI] | Holm p | above null 95th pct? | H2 |
|---|---|---|---|---|---|---|
| fc neuron 453 zero-ablated | ghosts_per_pill (−) | 0.294 → 0.228 | -0.066 [-0.192, 0.101] | 0.2280 | yes (null q95 0.013) | fail |
| conv3 channel 18 set to its p99 at cell (5,6) | approach (−) | 0.424 → 0.361 | -0.063 [-0.077, -0.050] | 0.0002 | yes (null q95 0.013) | pass |
| fc neuron 188 clamped to its p99 (constant logit bias, DLA UP -0.70) (positive control) | action_given_trigger (−) | 0.395 → 0.305 | -0.090 [-0.102, -0.079] | – | – | – |

p = 0.0001 is the smallest attainable value of the 10,000-permutation sign-flip test (1/10,001); 0.0002 is that floor after Holm over 2 tests, so it means p ≤ that value. The positive control's frozen label says "constant logit bias": clamping unit n to v shifts the logits by (v − h_n(s))·W_π[:, n], which is constant only on steps where the unit is inactive (fc 188: active on 2.1% of in-control discovery steps).
<!-- END:steer_appo_1111 -->

**H3, replication on agents 2222 and 3333** (frozen selection rules):
- no-go-zone rule: <!--n:h3:c3_ch18_cell56-->replicates<!--/n-->;
- ghost-hunting ablation rule: <!--n:h3:fc453_ablate-->does not replicate<!--/n-->.

Details are in [05_replication.md](05_replication.md).

## Deviations from the preregistration
1. **Stochasticity check.**
   - The preregistered check fired: without sticky actions, same-start games stayed identical for a median of
     <!--n:appo_1111:div_nosticky-->1230<!--/n--> steps.
   - We therefore applied the preregistered fallback: sticky actions p = 0.25.
   - With it the median is still <!--n:appo_1111:div_sticky-->320<!--/n--> steps, above the 200 threshold. We
     proceeded, and added a start-state-cluster bootstrap as a robustness check.
2. **Sticky actions lower the agent's score**: discovery mean <!--n:appo_1111:score_sticky-->4082<!--/n-->, vs
   <!--n:appo_1111:score_nosticky-->6384<!--/n--> without sticky actions.

## Errata and undeclared deviations (found in the fact-check audit, 2026-10-31)

The preregistration, its addendum and `results/candidates.json` are frozen and hashed by `pacneuron.prereg`, so they
are not edited. Where they or earlier reports were wrong or incomplete, the correction is here.

1. **"2B env steps" means 2B emulator frames.** The agents' `cfg.json` has `train_for_env_steps = 2e9` with
   `summaries_use_frameskip = true`, so training saw about 500M agent steps at frameskip 4.
2. **Fidelity tolerance and device.** The preregistration says logits are "within 1e-5". On CPU the maximum
   difference is <!--n:fid_logit-->1.1e-05<!--/n-->: just above 1e-5 absolute, it passes the test only through its
   relative tolerance. All data were collected on a GPU with PyTorch's default cuDNN TF32 convolutions, where the
   difference is <!--n:fid_logit_gpu-->4.5e-02<!--/n-->; the fidelity gate does not cover this. An unsteered A/A
   arm reproduces <!--n:appo_1111:aa_identical-->50 of 50<!--/n--> paired baseline games action for action (2222:
   <!--n:appo_2222:aa_identical-->50 of 50<!--/n-->; 3333: <!--n:appo_3333:aa_identical-->50 of 50<!--/n-->), so the
   seed pairing holds.
3. **Clamping is not a pure constant bias.** The preregistration (§6 iii), the addendum and the positive-control
   label call an fc clamp "a constant logit bias". Clamping unit n to v shifts the logits by (v − h_n(s))·W_π[:, n].
   That equals ablation plus the constant bias v·W_π[:, n], and it is constant only where the unit is inactive. fc
   188 is active on <!--n:appo_1111:fc188:active-->2.1%<!--/n--> of in-control discovery steps.
4. **H3 scope.** Preregistration §7 defines H3 only for candidates that pass H2. fc 453 failed H2, so the replication
   of the fc-ablation rule is exploratory. The H3 verdict also applies a further Holm correction over the two
   agents (§7 "Holm over seeds") on top of the within-agent Holm. Two details of `derive_candidates` are not in the
   frozen rule text: the secondary direction is the edible-direction feature with the highest R² for the selected
   unit, and the fc rule is not explicitly restricted to alive units (both selected units, 379 and 351, are alive).
5. **Untrained control.** Preregistration §5 says "the 512 fc units of the untrained control network". The code
   scores its alive units (499, 500 and 501 for the three agents), and it uses initialisation seed 0 for every agent.
6. **Metric operationalisations.**
   - "Approach … decreases within the next 8 steps" is implemented as "the BFS distance 8 agent steps later is
     smaller".
   - "In control" excludes every step at which Ms. Pac-Man's position is unchanged from t−1 to t and from t to t+1.
     That includes standstills such as pushing into a wall, not only death, ready and level-transition screens.
7. **Start-state clusters.** The addendum's robustness bootstrap clusters games by their first in-control RAM state.
   That gives only <!--n:appo_1111:n_start_groups-->4<!--/n--> clusters among the held-out games, so those CIs are
   very coarse. The stochasticity check groups by the reset state instead and finds more distinct start states in the
   discovery games (<!--n:appo_1111:distinct_starts-->28<!--/n-->, `results/stochasticity_appo_1111.json`).
8. **Frightened-mode registers are not new.** Bit 7 of `ram[1+i]` and `ram[116]` were identified here from pixels,
   but [jev_deep_rl](https://github.com/taodav/jev_deep_rl) documented them first (commit 3f69ca3a, 2026-09-20).
   Only the low six bits of `ram[116]` are a countdown; its upper bits are set on
   <!--n:appo_1111:gt_timer_upper-->18.5%<!--/n--> of frightened steps. The frozen `fright_timer` feature used the
   raw byte. Masking it (exploratory, `claim-checks`) raises that feature's split-half R² by at most
   <!--n:appo_1111:mask_max_gain-->0.047<!--/n--> for any unit and makes <!--n:appo_1111:mask_newly_clean-->0<!--/n-->
   units clean.
9. **The no-sticky results are reproducible again.** No code path wrote `results/*_nosticky.json` after the
   sticky-action setting was hard-coded. `STICKY=0` now does, and a re-run reproduced every committed score, step
   count and the <!--n:appo_1111:div_nosticky-->1230<!--/n-->-step median divergence exactly.
10. **"Every number is generated by `pacneuron.report`"** (preregistration §9) holds for the results tables and the
    inline-marked numbers in the README, reports and paper. It does not hold for the hand-written numbers inside the
    frozen preregistration files themselves, for design constants quoted in prose (seeds, game counts, thresholds),
    for the videos (`videos` phase) or for `paper/paper.pdf` (`scripts/build_paper.py`).

## Planned items not done

- **Planned but not done.**
  - An input-pixel counterfactual for the top candidates (repainting ghosts as frightened in the observation) to
    separate ghost appearance from correlated game state.
  - A per-ghost "visible in the observation" flag (Atari sprite flicker can hide a ghost that RAM reports).
  - Training dynamics over the 36 milestone checkpoints of agent 1111.
  - Replication on agents trained with other algorithms (Stable-Baselines3 DQN/PPO).
  - The OCAtari cross-check of RAM decoding, replaced by pixel-level validation. OCAtari releases ≥ 1.1.6 pin
    numpy < 2, which is incompatible with this project's numpy ≥ 2.5, and the 1.1.5 source distribution failed to
    build in our environment.

## Glossary
- **Reward-only RL / no imitation learning**: the agent learns from the game score alone, not from human
  demonstrations.
- **APPO**: asynchronous Proximal Policy Optimization ([Schulman et al. 2017](https://arxiv.org/abs/1707.06347);
  Sample Factory's implementation: [Petrenko et al. 2020](https://arxiv.org/abs/2006.11751)), an actor-critic
  algorithm.
- **fc unit / neuron**: one of the 512 hidden units after the conv layers; its ReLU output feeds the action and
  value heads linearly.
- **conv3 channel / cell**: one of 64 feature maps of the last conv layer; a cell is one of its 7×7 positions.
  Each cell sees a 36×36-pixel window of the 84×84 input.
- **Pole**: one side of a neuron's activation. The positive pole is "what makes it fire"; the negative pole is
  "what suppresses it".
- **Alive unit**: active (post-ReLU > 0) on at least 0.1% of in-control discovery steps.
- **One job / clean unit**: the features together explain ≥ 30% of the unit's variance (R²_all), and the best
  single feature explains ≥ 70% of that.
- **R² (out-of-sample)**: the fraction of a unit's variance predicted on games not used for fitting.
- **Privileged basis**: individual neurons are more interpretable than random directions in the same layer.
- **DLA (direct logit attribution)**: a unit's weight column in the policy head, i.e. how much it votes for each
  action ([Elhage et al. 2021](https://transformer-circuits.pub/2021/framework/index.html)).
- **Sticky actions**: with probability 0.25 the console repeats the previous action, which adds randomness
  (Machado et al. 2018).
- **Paired design / common random numbers**: steered and baseline games share the game seed (start state,
  random draws), so differences are due to the intervention. An unsteered A/A arm confirms that the same seed
  replays the same game.
- **Null units**: other units given the same kind of intervention: 99 random alive fc units for fc candidates, and
  all 63 other channels at the same cell (each at its own p99) for conv3 candidates. A specific effect must beat
  their 95th percentile.
- **Holm correction**: controls the chance of any false positive across several tested candidates
  ([Holm 1979](https://www.jstor.org/stable/4615733)).
- **In control**: Ms. Pac-Man's position changed between t−1 and t or between t and t+1. This excludes death
  animations, ready screens, level transitions and standstills.
