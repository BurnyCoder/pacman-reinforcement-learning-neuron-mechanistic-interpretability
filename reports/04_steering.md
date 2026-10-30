# Experiment 4: Steering single neurons and channels (preregistered H2)

## Question
Is there a neuron or channel whose steering changes the agent's behaviour in a specific, predicted way, beyond
what matched random units do?

## Structural fact that shapes the design
The agent has no RNN, and its policy head is linear on the 512 fc units. So for an fc unit n:
- **Clamping** it to v shifts the logits by exactly (v − h_n(s)) · W_π[:, n]. That is a state-independent vote
  for fixed actions, apart from the h_n(s) term (`tests/test_model.py::test_fc_clamp_equals_logit_bias`).
- **Ablating** it removes a state-dependent vote: whatever the unit adds when it fires.

Steering *upstream*, in conv3 (64 channels × 7×7 cells), acts through 512 ReLUs, so its effect can depend on the
state. Setting a single conv3 unit is an exact rank-1 update of the fc pre-activations
(`tests/test_model.py::test_rank1_conv3_patch_matches_full_forward`). This makes an open-loop screen of all
64 × 49 patches cheap.

## Discovery phase (exploratory, used only to choose candidates)
1. **Open-loop patch screen.** Each (channel, cell) is set to its discovery p99. We measure the change in the
   probability of moving toward that cell's maze location at junctions, on 4,000 recorded discovery states.
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
2. **fc ablation screen.** Zero each unit and measure the change in P(move toward the nearest edible ghost /
   dangerous ghost / pellet). The top unit for edible ghosts is fc 453. Its best direction feature is "nearest
   edible ghost is up", and it votes for UP.
3. **Closed-loop pilots** on discovery seeds 0–29, paired with the unsteered discovery games of the same seeds:
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
4. Frozen candidates and nulls: `reports/preregistration_addendum.md`, `results/candidates.json`.

## Confirmatory results (held-out seeds; paired by game seed)
<!-- BEGIN:steer_appo_1111 -->
| intervention | primary metric (predicted sign) | baseline → steered | Δ [95% CI] | Holm p | above null 95th pct? | H2 |
|---|---|---|---|---|---|---|
| fc neuron 453 zero-ablated | ghosts_per_pill (−) | 0.294 → 0.228 | -0.066 [-0.192, 0.101] | 0.2280 | yes (null q95 0.013) | fail |
| conv3 channel 18 set to its p99 at cell (5,6) | approach (−) | 0.424 → 0.361 | -0.063 [-0.077, -0.050] | 0.0002 | yes (null q95 0.013) | pass |
| fc neuron 188 clamped to its p99 (constant logit bias, DLA UP -0.70) (positive control) | action_given_trigger (−) | 0.395 → 0.305 | -0.090 [-0.102, -0.079] | – | – | – |
<!-- END:steer_appo_1111 -->

Off-target effects, i.e. how much else changes (paired Δ [95% CI]):
<!-- BEGIN:offtarget_appo_1111 -->
| intervention | Δ return | Δ deaths / 1k steps | Δ pellets / 1k steps | Δ ghosts per pill |
|---|---|---|---|---|
| fc neuron 453 zero-ablated | -1619 [-2194, -1056] | 1.14 [0.75, 1.52] | -9.23 [-17.68, -0.84] | -0.07 [-0.19, 0.10] |
| conv3 channel 18 set to its p99 at cell (5,6) | -2490 [-2990, -2005] | 4.17 [3.27, 5.14] | -12.16 [-20.78, -3.83] | -0.05 [-0.18, 0.09] |
<!-- END:offtarget_appo_1111 -->

Secondary mechanistic metric for fc 453: P(UP-ish action | nearest edible ghost is up)
<!--n:appo_1111:fc453_ablate:sec-->0.505 → 0.315 (Δ -0.190 [-0.209, -0.171])<!--/n-->.

![fc453 effect vs null](../figures/steer_appo_1111_fc453_ablate.png)
![conv3 ch18 effect vs null](../figures/steer_appo_1111_c3_ch18_cell56.png)

Videos (left: unsteered, right: steered, same game seed and sticky-action RNG):
- `figures/video_appo_1111_fc453_ablate.mp4`
- `figures/video_appo_1111_c3_ch18_cell56.mp4`, with the patched cell outlined.

![occupancy with and without the channel-18 patch](../figures/occupancy_appo_1111_c3_ch18_cell56.png)

## Interpretation (agent 1111)

### conv3 channel 18 at cell (5,6): a single unit that creates a "no-go zone" (H2 passes)
- **The intervention.** Writing the unit's own 99th-percentile activation into one position of one conv3 channel.
  The unit's receptive field covers the lower-right corridors of the maze (outlined in the video and the occupancy
  figure).
- **The effect.** The agent moves toward that location less often:
  - approach rate <!--n:appo_1111:c3_ch18_cell56:base-->0.424<!--/n--> →
    <!--n:appo_1111:c3_ch18_cell56:arm-->0.361<!--/n-->, Δ <!--n:appo_1111:c3_ch18_cell56:delta-->-0.063 [-0.077, -0.050]<!--/n-->;
  - Holm p <!--n:appo_1111:c3_ch18_cell56:p-->0.0002<!--/n-->.
- **Specificity.** The other <!--n:appo_1111:c3_ch18_cell56:null_n-->63<!--/n--> channels, set to *their*
  99th percentile at the same position, barely change approach (mean Δ
  <!--n:appo_1111:c3_ch18_cell56:null_mean-->-0.004<!--/n-->).
  <!--n:appo_1111:c3_ch18_cell56:null_ge-->0<!--/n--> of them reach channel 18's effect. The repulsion is a
  property of this channel at this place, not of perturbing conv3 there.
- **Consequences.** The agent leaves pellets in the avoided region uneaten, so levels last longer and it dies more:
  - deaths per 1k steps Δ <!--n:appo_1111:c3_ch18_cell56:deaths_per_1k-->4.2 [3.3, 5.1]<!--/n-->;
  - return Δ <!--n:appo_1111:c3_ch18_cell56:return-->-2490 [-2990, -2005]<!--/n-->.
- **What it does not show.** That channel 18 "means" danger. Its object-detection AUROCs are near chance (see the
  audit). The result is causal and behavioural ("activity here repels the agent from here"). The semantic label
  remains open.
- **Prior work.** This parallels the "cheese channel" retargeting of a Procgen maze agent
  ([Mini, Turner et al. 2023](https://arxiv.org/abs/2310.08043)), here with a repelling sign and in an Atari agent
  trained by APPO.

### fc neuron 453 ablation: a mechanistic effect that does not carry to the preregistered outcome (H2 fails)
- **The mechanism replicates.** Ablating fc 453 sharply reduces moving up when the nearest edible ghost is up:
  <!--n:appo_1111:fc453_ablate:sec-->0.505 → 0.315 (Δ -0.190 [-0.209, -0.171])<!--/n--> (secondary metric, not in the Holm family). The audit predicted
  this: the unit tracks "edible ghost is up" and votes UP.
- **The primary outcome does not.** Ghosts eaten per power pill fall only from
  <!--n:appo_1111:fc453_ablate:base-->0.294<!--/n--> to <!--n:appo_1111:fc453_ablate:arm-->0.228<!--/n-->
  (Δ <!--n:appo_1111:fc453_ablate:delta-->-0.066 [-0.192, 0.101]<!--/n-->, Holm p <!--n:appo_1111:fc453_ablate:p-->0.2280<!--/n-->).
  That is not significant, so the candidate **fails** H2 as preregistered.
- **But it is the largest effect in its layer.** <!--n:appo_1111:fc453_ablate:null_ge-->0<!--/n--> of
  <!--n:appo_1111:fc453_ablate:null_n-->99<!--/n--> random fc ablations reduce ghost eating as much (null mean
  <!--n:appo_1111:fc453_ablate:null_mean-->0.006<!--/n-->). The miss is a power problem with a noisy per-game ratio
  metric, not evidence that the unit is irrelevant.
- **Winner's curse.** The discovery pilot showed a much larger drop:
  <!--n:appo_1111:pilot:fc453_ablate:ghosts_per_pill-->0.392 → 0.140 (Δ -0.252 [-0.331, -0.171])<!--/n-->. Selecting a candidate on its discovery effect
  inflates that estimate, which is exactly why confirmatory effect sizes come only from held-out games.

### Caveats
- Steering this agent damages overall play for most interventions; even the positive control hurts. The matched
  null distribution, not the unsteered baseline, is the right yardstick for "specific".
- Doses are fixed at each unit's discovery 99th percentile. Larger doses produce larger effects in the
  open-loop screen, but leave the activation distribution. Exploratory: the strongest single-unit repulsion at
  junctions is <!--n:appo_1111:dose:p99-->-5.8 pp<!--/n--> at p99, <!--n:appo_1111:dose:max-->-11.0 pp<!--/n--> at the
  channel's max, and <!--n:appo_1111:dose:2x_max-->-40.7 pp<!--/n--> at twice the max.
- All games use sticky actions (p = 0.25). Steered and baseline games share seeds, but their trajectories diverge
  once an action differs, so per-game differences mix direct and downstream effects.
