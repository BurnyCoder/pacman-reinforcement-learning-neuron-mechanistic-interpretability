# Summary of all experiments

Author: Libor "Burny" Burian. Every number below is regenerated from `results/*.json` by
`uv run python -m pacneuron.pipeline --phase report`.

## In one paragraph (plain language)
- **What we did.** We took video-game-playing neural networks that taught themselves Atari Ms. Pac-Man by trial and
  error (only the score as feedback, no human examples). We looked at their individual "neurons" in the way
  Goodfire inspects language and vision models.
- **How.** The game's memory tells us exactly where every ghost is, whether ghosts are edible, and how many pellets
  remain. So for each neuron we can ask: which game fact does it track? And when we turn it up or off during play,
  does the agent behave differently in the predicted way?
- **Labels.** Most neurons do not track one game fact. They respond to combinations: mostly *where Ms. Pac-Man is
  and which way she is heading*.
- **Steering.** A few units do change behaviour when steered. The confirmatory, held-out results are in the table
  below.

## Models
<!-- BEGIN:models -->
| agent | architecture | params | algorithm and objective | training data | discovery games (sticky p=0.25) | mean score ± SD | model-card score (no sticky) |
|---|---|---|---|---|---|---|---|
| [appo_1111](https://huggingface.co/edbeeching/atari_2B_atari_mspacman_1111) | Nature-CNN: conv 32/64/64 -> fc 512 (ReLU) -> linear policy (9) + value | 1,689,258 | APPO (Sample Factory 2.0), maximise clipped game reward, gamma 0.99, no imitation | 2B env steps of interaction with MsPacmanNoFrameskip-v4 | 100 | 4082 ± 2032 | 6288 ± 1955 |
<!-- END:models -->

All three are public checkpoints by Edward Beeching on Hugging Face. They were trained with Sample Factory 2.0
APPO purely from game reward, on `MsPacmanNoFrameskip-v4` with the standard DeepMind wrappers. No imitation
learning was involved.

## Experiments
| # | experiment | status | report |
|---|---|---|---|
| 1 | Fidelity gate: our PyTorch/Windows rebuild vs real Sample Factory | passed | [01_fidelity.md](01_fidelity.md) |
| 2 | RAM ground truth (incl. newly found frightened-mode addresses) | validated | [02_ram_ground_truth.md](02_ram_ground_truth.md) |
| 3 | Neuron audit, one-job fraction vs controls (H1) | H1 <!--n:appo_1111:h1_verdict-->supported<!--/n--> | [03_neuron_audit.md](03_neuron_audit.md) |
| 4 | Steering, frozen candidates vs matched nulls (H2) | H2 <!--n:appo_1111:h2_verdict-->pending<!--/n--> | [04_steering.md](04_steering.md) |
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

**Exploratory.** Place × heading explains a median R² of <!--n:appo_1111:place_heading_median-->0.20<!--/n--> per
fc unit; <!--n:appo_1111:place_heading_frac-->30.5%<!--/n--> of units exceed 0.3.

**H2, steering on held-out games (paired by seed):**
<!-- BEGIN:steer_appo_1111 -->
<!-- END:steer_appo_1111 -->

## Deviations from the preregistration
1. **Stochasticity check.**
   - The preregistered check fired: without sticky actions, same-start games stayed identical for a median of
     <!--n:appo_1111:div_nosticky-->1230<!--/n--> steps.
   - We therefore applied the preregistered fallback: sticky actions p = 0.25.
   - With it the median is still <!--n:appo_1111:div_sticky-->320<!--/n--> steps, above the 200 threshold. We
     proceeded, and added a start-state-cluster bootstrap as a robustness check.
2. **Sticky actions lower the agent's score**: discovery mean <!--n:appo_1111:score_sticky-->4082<!--/n-->, vs
   <!--n:appo_1111:score_nosticky-->6384<!--/n--> without sticky actions.

## Glossary
- **Reward-only RL / no imitation learning**: the agent learns from the game score alone, not from human
  demonstrations.
- **APPO**: asynchronous Proximal Policy Optimization, an actor-critic algorithm.
- **fc unit / neuron**: one of the 512 hidden units after the conv layers; its ReLU output feeds the action and
  value heads linearly.
- **conv3 channel / cell**: one of 64 feature maps of the last conv layer; a cell is one of its 7×7 positions.
  Each cell sees a 36×36-pixel window of the 84×84 input.
- **Pole**: one side of a neuron's activation. The positive pole is "what makes it fire"; the negative pole is
  "what suppresses it".
- **One job / clean unit**: the features together explain ≥ 30% of the unit's variance (R²_all), and the best
  single feature explains ≥ 70% of that.
- **R² (out-of-sample)**: the fraction of a unit's variance predicted on games not used for fitting.
- **Privileged basis**: individual neurons are more interpretable than random directions in the same layer.
- **DLA (direct logit attribution)**: a unit's weight column in the policy head, i.e. how much it votes for each
  action.
- **Sticky actions**: with probability 0.25 the console repeats the previous action, which adds randomness
  (Machado et al. 2018).
- **Paired design / common random numbers**: steered and baseline games share the game seed (start state,
  random draws), so differences are due to the intervention.
- **Null units**: random units given the same kind of intervention. A specific effect must beat them.
- **Holm correction**: controls the chance of any false positive across several tested candidates.
- **In control**: Ms. Pac-Man is moving. This excludes death animations, ready screens and level transitions.
