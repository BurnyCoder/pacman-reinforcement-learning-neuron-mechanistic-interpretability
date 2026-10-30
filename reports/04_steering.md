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
<!-- END:steer_appo_1111 -->

Off-target effects, i.e. how much else changes (paired Δ [95% CI]):
<!-- BEGIN:offtarget_appo_1111 -->
<!-- END:offtarget_appo_1111 -->

Secondary mechanistic metric for fc 453: P(UP-ish action | nearest edible ghost is up)
<!--n:appo_1111:fc453_ablate:sec-->–<!--/n-->.

![fc453 effect vs null](../figures/steer_appo_1111_fc453_ablate.png)
![conv3 ch18 effect vs null](../figures/steer_appo_1111_c3_ch18_cell56.png)

Videos (left: unsteered, right: steered, same game seed and sticky-action RNG):
- `figures/video_appo_1111_fc453_ablate.mp4`
- `figures/video_appo_1111_c3_ch18_cell56.mp4`, with the patched cell outlined.

## Interpretation and caveats
- All effects are causal: interventions on the network during play, compared with the same seeds without
  intervention.
- What each confirmed effect means, and the matched-null comparison, is filled in after the run.
- Steering in this agent tends to damage overall play (return), even for random units. The null distribution,
  not the baseline, is the right yardstick for "specific".
