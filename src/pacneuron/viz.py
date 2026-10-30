"""Figures (matplotlib, PNG) and steering videos (MP4/GIF) generated only from results/*.json and recorded games.

Local: a small house style (reference data-viz palette: blue/orange/aqua categorical slots, blue sequential ramp,
blue<->red diverging with a gray midpoint, hairline grids, thin marks) plus one function per figure.
Global: `pacneuron.report` calls these so every figure in README/paper/site regenerates with one command.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

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
    for yi, v in zip(y, vals):
        ax.text(v + 0.01, yi + 0.3, f"{v:.0%}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, [lbl for _, lbl in order])
    ax.set_xlabel("fraction of units with one main job (R²_all ≥ 0.3, best single feature ≥ 70% of it)")
    ax.set_xlim(0, max(1e-3, max(hi for _, hi in errs)) * 1.25)
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
    ax.text(0.62, 0.33, "clean region:\nabove line, right of 0.3", fontsize=7.5, color=INK2)
    ax.set_xlabel("R²_all  (all 40 features, gradient boosting)")
    ax.set_ylabel("best single-feature R²")
    ax.set_xlim(-0.05, 1)
    ax.set_ylim(-0.05, 1)
    ax.legend(loc="upper left", fontsize=8, markerscale=1.6)
    ax.set_title(title, loc="left")
    save(fig, path)


def effect_vs_null(cands: list[dict], path: Path, title: str, xlabel: str) -> None:
    """Paired effect (95% CI) of each candidate vs the matched null distribution of random units (gray dots)."""
    style()
    fig, ax = plt.subplots(figsize=(5.8, 0.9 + 0.7 * len(cands)))
    for k, c in enumerate(cands):
        y = len(cands) - 1 - k
        null = np.asarray(c.get("null", []), float)
        if len(null):
            jitter = np.random.default_rng(k).uniform(-0.15, 0.15, len(null))
            ax.scatter(null, y + jitter, s=10, color=MUTED, alpha=0.6, label="null units" if k == 0 else None)
        lo, hi = c["ci"]
        ax.plot([lo, hi], [y, y], color=SERIES[0], lw=2)
        ax.scatter([c["delta"]], [y], s=40, color=SERIES[0], zorder=3, edgecolors=SURF, linewidths=1.5,
                   label="candidate (95% CI)" if k == 0 else None)
    ax.axvline(0, color=BASE, lw=1)
    ax.set_yticks(range(len(cands))[::-1], [c["label"] for c in cands])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title(title, loc="left")
    save(fig, path)


def heat_on_frame(frame: np.ndarray, values: np.ndarray, path: Path, title: str, vmax: float | None = None,
                  label: str = "") -> None:
    """Overlay a 7x7 conv3-cell map (diverging, 0 = gray) on an RGB frame (cell (i,j) -> its RF core on screen)."""
    style()
    fig, ax = plt.subplots(figsize=(3.6, 4.2))
    ax.imshow(frame[:172], extent=(0, 160, 172, 0))
    vmax = vmax or float(np.abs(values).max()) or 1.0
    for i in range(7):
        for j in range(7):
            x0, x1 = (8 * j + 10) * 160 / 84, (8 * j + 26) * 160 / 84
            y0, y1 = (8 * i + 10) * 210 / 84, (8 * i + 26) * 210 / 84
            col = DIVERGING(0.5 + 0.5 * values[i, j] / vmax)
            ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, color=col, alpha=0.75, lw=0))
    ax.set_xlim(0, 160)
    ax.set_ylim(172, 0)
    ax.axis("off")
    sm = plt.cm.ScalarMappable(cmap=DIVERGING, norm=plt.Normalize(-vmax, vmax))
    fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02, label=label)
    ax.set_title(title, loc="left")
    save(fig, path)


def frame_grid(frames: list[np.ndarray], path: Path, cols: int = 4) -> None:
    from pacneuron.render import grid
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid(frames, cols=cols)).save(path)


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
    small = [np.asarray(Image.fromarray(f).resize((f.shape[1] // 2, f.shape[0] // 2))) for f in frames[::2]]
    imageio.mimsave(path.with_suffix(".gif"), small, duration=2 / fps, loop=0)
