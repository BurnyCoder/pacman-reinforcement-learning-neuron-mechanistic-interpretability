"""Decode Atari 2600 Ms. Pac-Man RAM into exact game state (the ground truth used to label neurons).

Local: pure functions over uint8 RAM arrays of shape [..., 128] (vectorised over time).
Global: features.py builds the neuron-labelling feature library from these.

Sources:
- OCAtari (Delfosse et al. 2023, MIT) ocatari/ram/mspacman.py: player xy = (ram[10]-13, ram[16]+1),
  ghosts xy = (ram[6..9]-13, ram[12..15]+1), orientation = ram[56] & 3, ghosts active ram[47] > 0,
  power pills present = ram[117] bits 2..5, pellet bitmap ram[59..100] (3 bytes per row x 14 rows),
  maze grid choice ram[0], score BCD ram[120..122], lives = ram[123] % 4. Sprite boxes are 9x10 px.
- AtariARI (Anand et al. 2019, MIT) atariari/benchmark/ram_annotations.py "mspacman": same x/y indices,
  dots_eaten_count = ram[119].
- Discovered in this project (reports/02_ram_ground_truth.md): ram[116] = frightened countdown timer
  (>0 exactly while ghosts are blue), ram[1+i] bit 7 = ghost i is edible (validated against blue sprite pixels).
"""

from __future__ import annotations

import numpy as np

# Sprite box sizes from OCAtari (Player/Ghost wh = 9, 10).
SPRITE_W, SPRITE_H = 9, 10
# Frightened (edible) ghost colour in the RGB frame (measured; see reports/02_ram_ground_truth.md).
FRIGHTENED_RGB = (66, 114, 194)


def _i(ram: np.ndarray) -> np.ndarray:
    """uint8 -> int32 so that subtracting offsets cannot wrap around (ram[6]-13 underflows on uint8)."""
    return ram.astype(np.int32)


def player_xy(ram: np.ndarray) -> np.ndarray:
    """Top-left screen pixel of Ms. Pac-Man's sprite, [..., 2] = (x, y)."""
    r = _i(ram)
    return np.stack([r[..., 10] - 13, r[..., 16] + 1], -1)


def ghosts_xy(ram: np.ndarray) -> np.ndarray:
    """Top-left screen pixels of the four ghosts, [..., 4, 2]. Index order = RAM order (6..9)."""
    r = _i(ram)
    return np.stack([r[..., 6:10] - 13, r[..., 12:16] + 1], -1)


def heading(ram: np.ndarray) -> np.ndarray:
    """Ms. Pac-Man orientation 0..3 (OCAtari maps it to N, E, S, W)."""
    return _i(ram)[..., 56] & 3


def ghosts_active(ram: np.ndarray) -> np.ndarray:
    """[..., 4] bool: ghost i is on the board (OCAtari: ram[47] > 0 gates all; ram[19] counts extra ghosts)."""
    r = _i(ram)
    gate = r[..., 47] > 0
    count = r[..., 19]
    idx = np.arange(4)
    return gate[..., None] & ((idx == 0) | (idx <= count[..., None]))


def ghost_edible(ram: np.ndarray) -> np.ndarray:
    """[..., 4] bool: ghost i is frightened/edible (bit 7 of ram[1+i]; discovered here, see module doc)."""
    return (_i(ram)[..., 1:5] & 128) > 0


def frightened_timer(ram: np.ndarray) -> np.ndarray:
    """Frightened countdown (ram[116]); 0 when no ghost is edible, restarts near 60 when a power pill is eaten."""
    return _i(ram)[..., 116]


def power_pills(ram: np.ndarray) -> np.ndarray:
    """[..., 4] bool presence of the 4 power pills (ram[117] bits 2..5; corners BR, TR, BL, TL per OCAtari)."""
    r = _i(ram)[..., 117]
    return np.stack([(r & m) > 0 for m in (4, 8, 16, 32)], -1)


# OCAtari power-pill top-left positions for bits 4, 8, 16, 32 (all mazes use the same corners).
POWER_PILL_XY = np.array([(148, 146), (148, 14), (8, 147), (8, 15)])

# OCAtari pellet layouts: 1 where a pellet can exist on the 14x18 lattice; GRID1 if ram[0]==0 else GRID2.
GRID1 = np.array([
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0], [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
    [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1], [1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], bool)
GRID2 = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1], [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0], [1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1], [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1], [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0], [1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1], [1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1]], bool)


def pellets(ram: np.ndarray) -> np.ndarray:
    """Pellet presence on the 14x18 lattice, [..., 14, 18] bool (port of OCAtari's bitmap decoding).

    Per row i, three bytes starting at 59+3i: byte0 bit4 -> col 17, bit6 -> col 0;
    byte1 bit j -> col 4-(j>>1) if j odd else 12-(j>>1); byte2 bit j -> col 5+(j>>1) if j even else 13+(j>>1).
    Masked by the maze's GRID (bits outside the layout are not pellets).
    """
    r = _i(ram)
    out = np.zeros((*r.shape[:-1], 14, 18), bool)
    for i in range(14):
        b0, b1, b2 = r[..., 59 + 3 * i], r[..., 60 + 3 * i], r[..., 61 + 3 * i]
        out[..., i, 17] = (b0 & 16) > 0
        out[..., i, 0] = (b0 & 64) > 0
        for j in range(8):
            c1 = 4 - (j >> 1) if j & 1 else 12 - (j >> 1)
            out[..., i, c1] = (b1 & (1 << j)) > 0
            c2 = 5 + (j >> 1) if not j & 1 else 13 + (j >> 1)
            out[..., i, c2] = (b2 & (1 << j)) > 0
    grid = np.where((r[..., 0] == 0)[..., None, None], GRID1, GRID2)
    return out & grid


def score(ram: np.ndarray) -> np.ndarray:
    """Displayed score from BCD bytes ram[122] (ten-thousands), ram[121] (hundreds), ram[120] (units)."""
    r = _i(ram)

    def bcd(b):
        return (b >> 4) * 10 + (b & 15)

    return bcd(r[..., 122]) * 10000 + bcd(r[..., 121]) * 100 + bcd(r[..., 120])


def lives(ram: np.ndarray) -> np.ndarray:
    """Remaining lives as displayed (OCAtari: ram[123] % 4)."""
    return _i(ram)[..., 123] % 4
