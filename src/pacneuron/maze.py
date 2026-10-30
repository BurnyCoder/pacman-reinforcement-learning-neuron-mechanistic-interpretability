"""Maze graph on the 14x18 pellet lattice, extracted from the RGB frame, plus BFS distances.

Local: `maze_from_rgb(rgb)` -> `Maze` (open nodes, edges, all-pairs shortest-path lengths) and helpers that map
sprite pixels to lattice nodes and give the first step of a shortest path.
Global: features.py uses this for maze (BFS) distances and directions to ghosts/pellets/power pills and for the
open-directions-at-current-tile features; metrics.py uses it for approach/avoid behavioural metrics.

Geometry (measured on frames, see reports/02_ram_ground_truth.md; pellet lattice from OCAtari ram/mspacman.py):
pellet (i, j) has top-left (8+8j, 7+12i) for j<=8 and (12+8j, 7+12i) for j>=9 (the playfield is mirrored about
x=78: col 0 at x=8 <-> col 17 at x=148, as in OCAtari's Pill positions) and size 4x2; corridors are 12 px wide around pellet rows/columns;
pellets share the wall colour, so vertical-edge checks skip pellet rows. Top border row 1 is always wall.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import shortest_path

from pacneuron.ram import POWER_PILL_XY

ROWS, COLS = 14, 18
PELLET_X = np.where(np.arange(COLS) <= 8, 8, 12) + 8 * np.arange(COLS)  # pellet top-left x per column
NODE_X = PELLET_X + 1  # pixel column through the middle of pellet column j
NODE_Y = 5 + 12 * np.arange(ROWS)  # pixel row 2 px above pellet row i (inside the corridor, off the pellets)
# Direction order used everywhere: matches action ids UP=1, RIGHT=2, LEFT=3, DOWN=4 minus 1.
DIRS = ("up", "right", "left", "down")
_DELTA = {0: (-1, 0), 1: (0, 1), 2: (0, -1), 3: (1, 0)}
UNREACHABLE = 99  # distance used when no path / no target exists


@dataclass
class Maze:
    """Lattice graph of one maze layout. `nbr[i, j, d]` = node index reached by moving in direction d, or -1."""

    open: np.ndarray  # [14, 18] bool, node has at least one corridor edge
    nbr: np.ndarray  # [14, 18, 4] int, neighbour node id per direction (-1 if wall)
    dist: np.ndarray  # [252, 252] int, BFS path length between node ids (UNREACHABLE if disconnected)
    key: bytes  # hash key (edge pattern) for caching/deduplication

    def first_steps(self, src: int, dst: int) -> np.ndarray:
        """[4] bool: directions from node src that lie on some shortest path to node dst."""
        i, j = divmod(src, COLS)
        out = np.zeros(4, bool)
        if src == dst or self.dist[src, dst] >= UNREACHABLE:
            return out
        for d in range(4):
            n = self.nbr[i, j, d]
            if n >= 0 and self.dist[n, dst] == self.dist[src, dst] - 1:
                out[d] = True
        return out


def lattice(rgb: np.ndarray) -> np.ndarray:
    """Cheap per-frame step: neighbour table [14, 18, 4] (node id per direction, -1 = wall) from one RGB frame."""
    wall = rgb[1, 80]  # top border row is wall in every maze
    notwall = ~(rgb == wall).all(-1)  # [210, 160]
    for x, y in POWER_PILL_XY:  # power pills (4x7, wall colour, fixed corners) sit in corridors, not walls
        notwall[y:y + 7, x:x + 4] = True
    node_open = notwall[NODE_Y][:, NODE_X]
    h = np.zeros((ROWS, COLS), bool)  # edge (i,j)-(i,j+1)
    for j in range(COLS - 1):
        h[:, j] = notwall[NODE_Y][:, NODE_X[j]:NODE_X[j + 1] + 1].all(1)
    v = np.zeros((ROWS, COLS), bool)  # edge (i,j)-(i+1,j), skipping pellet rows (pellets have wall colour)
    for i in range(ROWS - 1):
        ys = [y for y in range(NODE_Y[i], NODE_Y[i + 1] + 1) if y % 12 not in (7, 8)]
        v[i] = notwall[ys][:, NODE_X].all(0)
    tunnel = notwall[NODE_Y, 0] & notwall[NODE_Y, 159]  # corridor reaches both screen edges -> wraparound
    h &= node_open & np.roll(node_open, -1, axis=1)
    v &= node_open & np.roll(node_open, -1, axis=0)
    nbr = -np.ones((ROWS, COLS, 4), int)
    ids = np.arange(ROWS * COLS).reshape(ROWS, COLS)
    nbr[1:, :, 0] = np.where(v[:-1], ids[:-1], -1)  # up
    nbr[:, :-1, 1] = np.where(h[:, :-1], ids[:, 1:], -1)  # right
    nbr[:, 1:, 2] = np.where(h[:, :-1], ids[:, :-1], -1)  # left
    nbr[:-1, :, 3] = np.where(v[:-1], ids[1:], -1)  # down
    wrap = tunnel & node_open[:, 0] & node_open[:, COLS - 1]
    nbr[wrap, COLS - 1, 1] = ids[wrap, 0]
    nbr[wrap, 0, 2] = ids[wrap, COLS - 1]
    return nbr


def build_maze(nbr: np.ndarray) -> Maze:
    """Expensive step (once per distinct layout): all-pairs BFS path lengths over the lattice graph."""
    ids = np.arange(ROWS * COLS)
    src, dst = np.repeat(ids, 4), nbr.reshape(-1)
    keep = dst >= 0
    adj = coo_matrix((np.ones(keep.sum()), (src[keep], dst[keep])), shape=(ROWS * COLS,) * 2)
    d = shortest_path(adj, method="D", unweighted=True)  # BFS lengths (unweighted Dijkstra)
    d = np.where(np.isfinite(d), d, UNREACHABLE).astype(np.int16)
    node_open = (nbr >= 0).any(-1)
    return Maze(node_open, nbr, d, maze_key(nbr))


def maze_key(nbr: np.ndarray) -> bytes:
    """Hashable identity of a layout (bit-packed edge pattern)."""
    return np.packbits(nbr.ravel() >= 0).tobytes()


def maze_from_rgb(rgb: np.ndarray) -> Maze:
    """Build the lattice graph (with BFS distances) from one 210x160x3 frame."""
    return build_maze(lattice(rgb))


class MazeBank:
    """Registry of distinct maze layouts: key -> integer id, so per-step records store a small int."""

    def __init__(self):
        self.ids: dict[bytes, int] = {}
        self.mazes: list[Maze] = []

    def add(self, nbr: np.ndarray) -> int:
        key = maze_key(nbr)
        if key not in self.ids:
            self.ids[key] = len(self.mazes)
            self.mazes.append(build_maze(nbr))
        return self.ids[key]


def node_of(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Map sprite top-left pixels (OCAtari xy, 9x10 box) to the nearest lattice node id."""
    cx, cy = np.asarray(x) + 4.0, np.asarray(y) + 4.5  # sprite centre (measured offsets, reports/02)
    j = np.abs(cx[..., None] - (PELLET_X + 1.5)).argmin(-1)  # nearest pellet column centre
    i = np.clip(np.rint((cy - 7.5) / 12), 0, ROWS - 1).astype(int)
    return i * COLS + j
