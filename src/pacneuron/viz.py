"""Figures (matplotlib, PNG) and steering videos (MP4/GIF) generated only from results/*.json and recorded games.

Local: a small house style (reference data-viz palette: blue/orange/aqua categorical slots, blue sequential ramp,
blue<->red diverging with a gray midpoint, hairline grids, thin marks) plus one function per figure.
Global: `pacneuron.report` calls these so every figure in README/paper/site regenerates with one command.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 (validated all-pairs in the reference palette)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("bluered", ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#b83232"])
SEQUENTIAL = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#104281"])


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.edgecolor": BASE,
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "lines.linewidth": 2, "legend.frameon": False})


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def clean_fraction_bars(summary: dict, path: Path, title: str) -> None:
    """Clean ('one job') fraction of the agent's fc units vs the three controls, with 95% bootstrap CIs."""
    style()
    order = [("agent", "agent fc units"), ("random_dirs", "random directions"),
             ("nonneg_mix", "non-neg. unit mixtures"), ("untrained", "untrained network")]
    vals = [summary[k]["clean_frac"] for k, _ in order]
    errs = [summary[k].get("ci", [v, v]) for (k, _), v in zip(order, vals)]
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    y = np.arange(len(order))[::-1]
    colors = [SERIES[0]] + [MUTED] * 3  # emphasis: the agent; controls recede
    ax.barh(y, vals, color=colors, height=0.55)
    for yi, (lo, hi) in zip(y, errs):
        ax.plot([lo, hi], [yi, yi], color=INK, lw=1)
    for yi, v, (_lo, hi) in zip(y, vals, errs):
        ax.text(max(v, hi) + 0.003, yi, f"{100 * v:.1f}%", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [lbl for _, lbl in order])
    ax.set_xlabel("fraction of units with one main job (R²_all ≥ 0.3, best single feature ≥ 70% of it)")
    ax.set_xlim(0, max(1e-3, max(hi for _, hi in errs)) * 1.3)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left")
    save(fig, path)


def r2_scatter(sets: dict[str, tuple[np.ndarray, np.ndarray]], path: Path, title: str) -> None:
    """Per unit: x = R²_all (explainable at all), y = best single-feature R²; clean region = upper right wedge."""
    style()
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    for (name, (rall, rbest)), col in zip(sets.items(), SERIES):
        ax.scatter(np.clip(rall, -0.05, 1), np.clip(rbest, -0.05, 1), s=9, color=col, alpha=0.6, label=name,
                   edgecolors=SURF, linewidths=0.4)
    xs = np.linspace(0.3, 1, 10)
    ax.plot(xs, 0.7 * xs, color=INK2, lw=1)
    ax.axvline(0.3, color=INK2, lw=1)
    ax.text(0.33, 0.62, "one-job region:\nabove the line and\nright of R²_all = 0.3", fontsize=7.5, color=INK2)
    ax.set_xlabel("R²_all  (all 40 features, gradient boosting)")
    ax.set_ylabel("best single-feature R²")
    ax.set_xlim(-0.05, 1)
    ax.set_ylim(-0.05, 1)
    ax.legend(loc="upper left", fontsize=8, markerscale=1.6)
    ax.set_title(title, loc="left")
    save(fig, path)


def effect_vs_null(cands: list[dict], path: Path, title: str, xlabel: str) -> None:
    """Paired effect (95% CI, 50 games) of each candidate on its own row, and the matched null units' effects (20 games
    each, gray dots) on the row below, so that no null point is hidden behind the candidate marker. The dashed line is
    the null's 95th percentile in the predicted direction (preregistration §6 criterion 2), given as c["q95"]."""
    style()
    fig, ax = plt.subplots(figsize=(5.8, 1.1 + 1.0 * len(cands)))
    ticks, labels = [], []
    for k, c in enumerate(cands):
        y = 2 * (len(cands) - 1 - k) + 1
        null = np.asarray(c.get("null", []), float)
        if len(null):
            jitter = np.random.default_rng(k).uniform(-0.18, 0.18, len(null))
            ax.scatter(null, y - 1 + jitter, s=10, color=MUTED, alpha=0.6, label="null units (20 games each)" if k == 0 else None)
            ticks.append(y - 1)
            labels.append(f"{len(null)} null units")
        if c.get("q95") is not None:
            ax.plot([c["q95"]] * 2, [y - 1.35, y + 0.35], color=INK2, lw=1, ls="--",
                    label="null 95th percentile (predicted direction)" if k == 0 else None)
        lo, hi = c["ci"]
        ax.plot([lo, hi], [y, y], color=SERIES[0], lw=2)
        ax.scatter([c["delta"]], [y], s=40, color=SERIES[0], zorder=3, edgecolors=SURF, linewidths=1.5,
                   label="candidate (95% CI, 50 games)" if k == 0 else None)
        ticks.append(y)
        labels.append(c["label"])
    ax.axvline(0, color=BASE, lw=1)
    ax.set_yticks(ticks, labels)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.set_ylim(-0.7, 2 * len(cands) + 0.2)
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.35), fontsize=8, frameon=False, ncol=1)
    ax.set_title(title, loc="left")
    save(fig, path)


def side_by_side_video(left: list[np.ndarray], right: list[np.ndarray], path: Path, labels=("baseline", "steered"),
                       box: tuple | None = None, fps: int = 15, scale: int = 2) -> None:
    """MP4 + GIF of two synchronized replays; `box` = (x0, y0, x1, y1) screen rectangle drawn on the right panel."""
    import imageio.v2 as imageio
    n = min(len(left), len(right))
    frames = []
    for t in range(n):
        panels = []
        for k, f in enumerate((left[t], right[t])):
            im = Image.fromarray(f[:172]).resize((160 * scale, 172 * scale), Image.NEAREST)
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, 160 * scale, 14], fill=(11, 11, 11))
            dr.text((4, 2), f"{labels[k]}  t={t}", fill=(240, 239, 236))
            if box is not None and k == 1:
                x0, y0, x1, y1 = (v * scale for v in box)
                dr.rectangle([x0, y0, x1, y1], outline=(255, 255, 255), width=2)
            panels.append(np.asarray(im))
        gap = np.full((172 * scale, 6, 3), 252, np.uint8)
        frames.append(np.concatenate([panels[0], gap, panels[1]], 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path.with_suffix(".mp4"), frames, fps=fps, macro_block_size=1)
    imageio.mimsave(path.with_suffix(".gif"), frames[::3], duration=3 / fps, loop=0)  # full size, every 3rd frame


def occupancy_maps(frame: np.ndarray, base: np.ndarray, steered: np.ndarray, cell: tuple, path: Path, title: str) -> None:
    """Three panels on the maze: baseline occupancy, steered occupancy (sequential blue), and their difference
    (diverging, 0 = gray), each lattice node drawn as a dot sized/coloured by its share of in-control steps."""
    from pacneuron.maze import COLS, PELLET_X, ROWS
    style()
    xs = np.tile(PELLET_X + 2, ROWS)
    ys = np.repeat(7 + 12 * np.arange(ROWS) + 1, COLS)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.4), constrained_layout=True)
    diff = steered - base
    vmax = float(max(base.max(), steered.max()))
    dmax = float(np.abs(diff).max()) or 1.0
    i, j = cell
    box = ((8 * j + 10) * 160 / 84, (8 * i + 10) * 210 / 84, (8 * j + 26) * 160 / 84, (8 * i + 26) * 210 / 84)
    for ax, vals, cmap, lo, hi, ttl in ((axes[0], base, SEQUENTIAL, 0, vmax, "baseline"),
                                        (axes[1], steered, SEQUENTIAL, 0, vmax, "steered"),
                                        (axes[2], diff, DIVERGING, -dmax, dmax, "steered − baseline")):
        ax.imshow(frame[:172], extent=(0, 160, 172, 0), alpha=0.35)
        sc = ax.scatter(xs, ys, c=vals.ravel(), cmap=cmap, vmin=lo, vmax=hi, s=28, edgecolors="none")
        ax.add_patch(plt.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, ec=INK, lw=1.5))
        ax.set_xlim(0, 160)
        ax.set_ylim(172, 0)
        ax.axis("off")
        ax.set_title(ttl, loc="left")
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=7)
    fig.suptitle(title + "\ncolour = share of in-control steps at each maze node", x=0.01, ha="left",
                 fontweight="bold", fontsize=9)
    save(fig, path)


def cell_delta_heatmap(frame: np.ndarray, delta: np.ndarray, cell: tuple, path: Path, title: str) -> None:
    """Post-hoc specificity map: paired Δ approach rate toward the target node of every conv3 cell, drawn as 7x7
    tiles over a gray maze frame (tile = the 8-obs-px stride around the cell's receptive-field centre, mapped to the
    screen). Diverging colours centred at 0 (blue = approached less); the steered cell is outlined. If the steering
    effect were local, only tiles at and near the outlined cell would be blue."""
    style()
    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    ax.imshow(frame[:180].mean(-1), extent=(0, 160, 180, 0), cmap="gray", vmin=0, vmax=255)  # bottom tiles end ~174
    vmax = float(np.abs(delta).max()) or 1.0
    sx, sy = 160 / 84, 210 / 84
    for i in range(7):
        for j in range(7):
            x0, y0 = (8 * j + 13.5) * sx, (8 * i + 13.5) * sy
            ax.add_patch(plt.Rectangle((x0, y0), 8 * sx, 8 * sy, color=DIVERGING(0.5 + 0.5 * delta[i, j] / vmax),
                                       alpha=0.8, lw=0))
    i, j = cell
    ax.add_patch(plt.Rectangle(((8 * j + 13.5) * sx, (8 * i + 13.5) * sy), 8 * sx, 8 * sy, fill=False, ec=INK, lw=2))
    k = int(np.argmin(delta))
    for (a, b), txt in (((i, j), f"steered\n{delta[i, j]:+.3f}"), (divmod(k, 7), f"largest drop\n{delta.flat[k]:+.3f}")):
        ax.text((8 * b + 17.5) * sx, (8 * a + 17.5) * sy, txt, ha="center", va="center", fontsize=6.5, color=INK,
                bbox={"boxstyle": "round,pad=0.15", "fc": SURF, "ec": "none", "alpha": 0.85})
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-vmax, vmax))
    fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02, label="Δ approach rate (steered − baseline)")
    ax.set_xlim(0, 160)
    ax.set_ylim(180, 0)
    ax.axis("off")
    ax.set_title(title + "\npaired Δ approach toward each conv3 cell's target node\n(tiles at the cells' receptive-field "
                 "centres; the maze edges lie outside all of them)", loc="left", fontsize=9)
    save(fig, path)


def occlusion_overlay(frame: np.ndarray, occ: np.ndarray, path: Path, title: str, patch: int = 12, stride: int = 6) -> None:
    """Occlusion map (activation drop when a patch is replaced by the mean frame) over one top-activating frame.

    Patch (a, b) covers obs rows/cols [stride*a, stride*a + patch); obs 84x84 spans the full 210x160 screen.
    Diverging colours: red = occluding here lowers the unit's activation (the unit relies on it), blue = raises it.
    """
    style()
    fig, ax = plt.subplots(figsize=(3.8, 4.6))
    # gray background: the maze's own pink/blue would otherwise be confused with the red/blue diverging overlay
    ax.imshow(frame.mean(-1), extent=(0, 160, 210, 0), cmap="gray", vmin=0, vmax=255)
    n = occ.shape[0]
    acc, cnt = np.zeros((84, 84)), np.zeros((84, 84))
    for a in range(n):
        for b in range(n):
            acc[stride * a:stride * a + patch, stride * b:stride * b + patch] += occ[a, b]
            cnt[stride * a:stride * a + patch, stride * b:stride * b + patch] += 1
    m = acc / np.maximum(cnt, 1)
    vmax = float(np.abs(m).max()) or 1.0
    im = ax.imshow(m, extent=(0, 160, 210, 0), cmap=DIVERGING, vmin=-vmax, vmax=vmax, alpha=0.55)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="activation drop when occluded")
    ax.set_title(title + "\nocclusion map: mean over the unit's 50 top states,\ndrawn on its top state", loc="left",
                 fontsize=9)
    save(fig, path)
