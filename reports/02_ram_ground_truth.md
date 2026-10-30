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
- **Discovered here: frightened ("edible") ghosts.** Neither source has an address for this. We played one game,
  found the step at which a power pill's bit cleared, and correlated every RAM byte with the presence of the
  frightened-blue colour `(66, 114, 194)` in the frame:
  - `ram[116]` is a countdown timer. It restarts at about 60 when a pill is eaten, falls by 2 every 4 agent steps,
    and is > 0 exactly while any ghost is blue.
  - Bit 7 of `ram[1+i]` marks ghost *i* as edible. It clears when that ghost is eaten.
- **Maze graph.** A 14×18 lattice of pellet positions, extracted from the frame:
  - The playfield is mirrored, so pellet column j sits at x = 8+8j for j ≤ 8 and 12+8j for j ≥ 9.
  - Corridors are 12 px wide. Pellets and power pills share the wall colour, so the edge checks skip pellet rows
    and power-pill boxes.
  - All-pairs BFS distances give maze distances, the first step of a shortest path ("direction to"), and the open
    directions at Ms. Pac-Man's tile.
- **Features.** 40 features over 5 families (`src/pacneuron/features.py`):
  - player: position, heading, open directions, junction;
  - danger: distance and direction to the nearest non-edible ghost;
  - edible: frightened flag and timer, distance and direction to the nearest edible ghost;
  - food: pellets and power pills (distance, direction, count);
  - game: maze, imminent life loss, reward, ghost eaten.

## Validation (all discovery games)
<!-- BEGIN:ground_truth_appo_1111 -->
<!-- END:ground_truth_appo_1111 -->

Disagreements between the edible bit and blue pixels happen mostly at the end of frightened mode, when ghosts
flash white. The RAM bit is the cleaner signal.

## Learnings
- OCAtari's PyPI source distribution (1.1.5) fails to build: its `setup.py` reads a `requirements.txt` that is
  not packaged. Pixel-level validation replaced an OCAtari cross-check, and is the stronger test anyway.
- uint8 RAM arithmetic underflows (`ram[6] - 13`), so every decoder casts to int32 first.
- The first lattice attempt ignored the mirrored right half; the pellet bitmap then matched the drawn pellets in
  only 59% of cells. After the fix: 100% of non-power-pill cells (`tests/test_ground_truth.py`).
