"""Matplotlib figures (all figures are produced from data -- none are LLM-generated)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection

from .tree import Tree

# Colour-blind-safe categorical palette (Okabe-Ito) + neutral for "other"
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
NEUTRAL = "#9a9a9a"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "legend.frameon": False,
})


def method_colors(methods: list[str]) -> dict[str, str]:
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(methods)}


def plot_tree(tree: Tree, ax=None, labels: dict[str, str] | None = None, x: str = "depth",
              node_dates: dict[int, float] | None = None, top_k: int = 7, lw: float = 0.6):
    """Rectangular phylogram of a rooted tree.

    x = 'depth' (substitutions/site from root) or 'date' (needs node_dates -> decimal year).
    Tips are coloured by ``labels`` (top_k most common groups; the rest grey).
    """
    ax = ax or plt.subplots(figsize=(6, 6))[1]
    v = tree.rooted()
    ypos: dict[int, float] = {}
    for i, leaf in enumerate(v.leaves):
        ypos[leaf] = i
    for node in reversed(v.order):
        if node not in ypos:
            ys = [ypos[c] for c in v.children[node]]
            ypos[node] = 0.5 * (min(ys) + max(ys))
    if x == "date" and node_dates is None:
        raise ValueError("x='date' requires node_dates")
    xv = (lambda nd: v.depth[nd]) if x == "depth" else (lambda nd: node_dates[nd])
    segs, vsegs = [], []
    for node in v.order:
        if node != v.root:
            segs.append([(xv(v.parent[node]), ypos[node]), (xv(node), ypos[node])])
        kids = v.children.get(node, [])
        if len(kids) >= 2:
            ys = [ypos[c] for c in kids]
            vsegs.append([(xv(node), min(ys)), (xv(node), max(ys))])
    ax.add_collection(LineCollection(segs + vsegs, colors="#555555", linewidths=lw))
    if labels:
        names = v.leaf_names
        counts: dict[str, int] = {}
        for nm in names:
            counts[labels.get(nm, "NA")] = counts.get(labels.get(nm, "NA"), 0) + 1
        top = [k for k, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:top_k]]
        col = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(top)}
        xs = [xv(l) for l in v.leaves]
        ys = list(range(len(v.leaves)))
        cs = [col.get(labels.get(nm, "NA"), NEUTRAL) for nm in names]
        ax.scatter(xs, ys, s=4, c=cs, zorder=3, linewidths=0)
        for k, c in col.items():
            ax.scatter([], [], s=14, c=c, label=f"{k} (n={counts[k]})")
        ax.scatter([], [], s=14, c=NEUTRAL, label="other")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=6.5,
                  title="lineage (tips)", title_fontsize=7, handletextpad=0.2, columnspacing=1.0)
    ax.autoscale()
    ax.set_ylim(-1, len(v.leaves))
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Root-to-node distance (substitutions/site)" if x == "depth" else "Estimated date (year)")
    return ax


def plot_root_to_tip(t: np.ndarray, rtt: np.ndarray, fit, ax=None, labels=None):
    ax = ax or plt.subplots(figsize=(4.2, 3.4))[1]
    ok = np.isfinite(t)
    ax.scatter(t[ok], rtt[ok], s=8, c=PALETTE[0], alpha=0.6, linewidths=0)
    xs = np.array([np.nanmin(t) if fit.tmrca != fit.tmrca else min(np.nanmin(t), fit.tmrca), np.nanmax(t)])
    ax.plot(xs, fit.intercept + fit.slope * xs, c=PALETTE[1], lw=1.4)
    ax.set_xlabel("Sampling date (year)")
    ax.set_ylabel("Root-to-tip distance (subs/site)")
    ax.set_title(f"rate = {fit.slope:.2e} subs/site/yr   R² = {fit.r2:.2f}   n = {fit.n}", fontsize=8)
    return ax
