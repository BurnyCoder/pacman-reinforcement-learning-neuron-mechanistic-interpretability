"""RAM decoding, maze graph and features validated against the pixels of real frames."""

import numpy as np

from pacneuron import maze as M
from pacneuron import ram as R
from pacneuron.env import replay
from pacneuron.features import FAMILY, compute_features, in_control


def test_pellet_bitmap_matches_drawn_pellets(played):
    rgb, ram = played["rgb"][20], played["ram"][20]
    wall = rgb[1, 80]
    drawn = (rgb[7 + 12 * np.arange(14)][:, M.PELLET_X] == wall).all(-1)
    mz = M.maze_from_rgb(rgb)
    pel = R.pellets(ram[None])[0]
    assert pel.sum() > 100
    check = mz.open.copy()
    pp = M.node_of(R.POWER_PILL_XY[:, 0] - 2.5, R.POWER_PILL_XY[:, 1] - 1.5)
    check.ravel()[pp] = False  # power pills are drawn in wall colour at their nodes but are not pellets
    np.testing.assert_array_equal(pel[check], drawn[check])


def test_eaten_pellets_are_at_player_node(played):
    pel, xy = R.pellets(played["ram"]), R.player_xy(played["ram"])
    node = M.node_of(xy[:, 0], xy[:, 1])
    hits = total = 0
    for t in range(1, len(pel)):
        for i, j in zip(*np.where(pel[t - 1] & ~pel[t])):
            total += 1
            hits += int(i * 18 + j in (node[t - 1], node[t]))
    assert total > 50 and hits / total > 0.98


def test_maze_is_one_component_plus_ghost_house(played):
    mz = M.maze_from_rgb(played["rgb"][20])
    nodes = np.flatnonzero(mz.open.ravel())
    main = nodes[mz.dist[nodes[0], nodes] < M.UNREACHABLE]
    assert len(main) >= 150 and len(nodes) - len(main) <= 4


def test_frightened_timer_iff_any_edible(played):
    ram = played["ram"]
    any_edible = R.ghost_edible(ram).any(1)
    assert any_edible.sum() > 20  # the seed-0 game eats a power pill within 1400 steps
    np.testing.assert_array_equal(R.frightened_timer(ram) > 0, any_edible)


def test_edible_bit_matches_blue_sprites(played):
    ram, rgb = played["ram"], played["rgb"]
    xy, edible = R.ghosts_xy(ram), R.ghost_edible(ram)
    blue = np.zeros_like(edible)
    for t in range(len(ram)):
        for g in range(4):
            x, y = xy[t, g]
            win = rgb[t, max(y - 1, 0):y + 12, max(x - 1, 0):x + 12]
            blue[t, g] = (win == R.FRIGHTENED_RGB).all(-1).sum() >= 4
    assert (blue == edible).mean() > 0.9  # disagreements: end-of-fright flashing (white) frames


def test_replay_is_deterministic(played):
    caps = {0, 100, 500}
    out = replay(0, played["action"], capture=caps, sticky=0.0)
    for t in caps:
        np.testing.assert_array_equal(out[t]["ram"], played["ram"][t])
        np.testing.assert_array_equal(out[t]["obs"], played["obs"][t])


def test_features_shapes_and_sanity(played):
    bank = M.MazeBank()
    rec = {"ram": played["ram"], "maze": np.array([bank.add(M.lattice(f)) for f in played["rgb"]]),
           "raw_reward": played["reward"], "life_lost": played["life_lost"]}
    f = compute_features(rec, bank.mazes)
    T = len(played["ram"])
    assert set(FAMILY) == set(f) and all(v.shape == (T,) for v in f.values())
    ctrl = in_control(played["ram"])
    assert 0.7 < ctrl.mean() < 1.0
    # an open direction is always available while moving, and the pellet direction is one of the open ones
    has_dir = np.stack([f[f"dir_pellet_{d}"] for d in M.DIRS], 1).astype(bool)
    opens = np.stack([f[f"open_{d}"] for d in M.DIRS], 1).astype(bool)
    assert (~has_dir | opens).all()


def test_replay_is_deterministic_with_sticky_actions(played):
    """Sticky actions draw from ALE's RNG, which env.reset(seed) seeds, so replay(seed, actions) is still exact."""
    from pacneuron.env import Game
    game, rams = Game(7, sticky=0.25), []
    for a in played["action"][:300]:
        rams.append(game.info["ram"].copy())
        game.step(int(a))
    out = replay(7, played["action"][:300], capture={0, 150, 299}, sticky=0.25)
    for t in (0, 150, 299):
        np.testing.assert_array_equal(out[t]["ram"], rams[t])
