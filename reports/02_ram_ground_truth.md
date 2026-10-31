# Experiment 2: Ground truth from the Atari console RAM

## Why
Goodfire labels neurons by looking at top-activating examples. In an Atari game we can do better: the console's
128 bytes of RAM *are* the game state. Labels can therefore be checked against exact quantities (ghost positions,
whether ghosts are edible, remaining pellets) instead of visual impressions.

## Method
- **Known addresses** from [OCAtari](https://github.com/k4ntz/OC_Atari) `ocatari/ram/mspacman.py` (MIT) and
  [AtariARI](https://github.com/mila-iqia/atari-representation-learning) `ram_annotations.py` (MIT):
  - player / ghost x and y (`ram[10], ram[16]`; `ram[6..9], ram[12..15]`, with offsets −13 / +1);
  - heading `ram[56] & 3`;
  - power pills: bits 2–5 of `ram[117]`;
  - pellet bitmap `ram[59..100]`;
  - BCD score `ram[120..122]`.
- **Identified independently here: frightened ("edible") ghosts.** OCAtari's RAM mode and AtariARI have no address
  for this (OCAtari's vision mode detects edible ghosts by colour). We played one game, found the step at which a
  power pill's bit cleared, and correlated every RAM byte with the presence of the frightened-blue colour
  `(66, 114, 194)` in the frame. The same registers were documented earlier in
  [jev_deep_rl](https://github.com/taodav/jev_deep_rl) (`jev_rl/mspacman.py`, commit 3f69ca3a, 2026-09-20).
  - Bit 7 of `ram[1+i]` marks ghost *i* as edible. It clears when that ghost is eaten.
  - `ram[116]` is > 0 exactly while at least one ghost's edible bit is set
    (<!--n:appo_1111:gt_timer_agree-->99.98%<!--/n--> of steps).
  - Only its low six bits are a countdown (jev_deep_rl masks it with `& 63`). They start at
    <!--n:appo_1111:gt_timer_start-->38–62<!--/n--> when a pill is eaten in normal mode and fall by
    <!--n:appo_1111:gt_timer_drop-->2<!--/n--> every 4 agent steps.
  - The upper bits change when ghosts are eaten; they are set on
    <!--n:appo_1111:gt_timer_upper-->18.5%<!--/n--> of frightened steps. The study's `fright_timer` feature uses the
    raw byte (see "Learnings").
- **Maze graph.** A 14×18 lattice of pellet positions, extracted from the frame:
  - The playfield is mirrored, so pellet column j sits at x = 8+8j for j ≤ 8 and 12+8j for j ≥ 9.
  - Corridors are 12 px wide. Pellets and power pills share the wall colour, so the edge checks skip pellet rows
    and power-pill boxes.
  - All-pairs BFS distances give maze distances, the first step of a shortest path ("direction to"), and the open
    directions at Ms. Pac-Man's tile.
- **In control.** Steps where Ms. Pac-Man's position is unchanged from t−1 to t and from t to t+1 are excluded:
  death animations, ready screens, level transitions and standstills.
- **Features.** 40 features over 5 families (`src/pacneuron/features.py`):
  - player: position, heading, open directions, junction;
  - danger: distance and direction to the nearest non-edible ghost;
  - edible: frightened flag and timer, distance and direction to the nearest edible ghost;
  - food: pellets and power pills (distance, direction, count);
  - game: maze, imminent life loss, reward, ghost eaten.

## Validation (all discovery games)
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

Disagreements between the edible bit and the blue pixels are concentrated late in frightened mode.
<!--n:appo_1111:gt_dis_le10-->38.2%<!--/n--> of them occur when the countdown is ≤ 10, against
<!--n:appo_1111:gt_edible_le10-->7.1%<!--/n--> of all edible ghost-steps. In
<!--n:appo_1111:gt_bit_without_blue-->95.5%<!--/n--> of them the bit is set but no blue pixels are found. Both are
consistent with ghosts flashing white near the end of frightened mode, and with Atari sprite flicker; we did not
separate the two.

## Learnings
- No OCAtari release could be installed alongside this project. Releases ≥ 1.1.6 pin `numpy<2.0.0`, which
  conflicts with this project's numpy ≥ 2.5. The 1.1.5 source distribution failed to build in our environment: its
  `setup.py` reads a `requirements.txt` that the sdist does not include. Pixel-level validation replaced the OCAtari
  cross-check.
- `ram[116]` is not a pure timer, and the frozen `fright_timer` feature kept the raw byte. In an exploratory check
  (`claim-checks`), masking it would raise this feature's split-half R² by at most
  <!--n:appo_1111:mask_max_gain-->0.047<!--/n--> for any unit. It would make no unit "clean"
  (<!--n:appo_1111:mask_newly_clean-->0<!--/n--> units; agents 2222 and 3333:
  <!--n:appo_2222:mask_newly_clean-->0<!--/n--> and <!--n:appo_3333:mask_newly_clean-->0<!--/n-->).
- uint8 RAM arithmetic underflows (`ram[6] - 13`), so every decoder casts to int32 first.
- The first lattice attempt ignored the mirrored right half; the pellet bitmap then matched the drawn pellets in
  only 59% of cells. After the fix: 100% of non-power-pill cells (`tests/test_ground_truth.py`).
