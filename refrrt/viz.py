"""Plotting helpers. Points are (row, col); plots use x = col, y = row."""

import matplotlib.pyplot as plt
import numpy as np


def draw_map(ax, occupancy, reference_points=None):
    ax.imshow(occupancy, cmap="gray", origin="lower", interpolation="nearest")
    if reference_points is not None:
        ax.scatter(reference_points[:, 1], reference_points[:, 0], s=12,
                   c="tab:cyan", label="reference points", zorder=3)
    rows, cols = np.nonzero(~occupancy)
    if rows.size:
        m = 30
        ax.set_xlim(cols.min() - m, cols.max() + m)
        ax.set_ylim(rows.min() - m, rows.max() + m)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def draw_tree(ax, result, color="tab:red"):
    for i, p in enumerate(result.parents):
        if p >= 0:
            a, b = result.nodes[p], result.nodes[i]
            ax.plot([a[1], b[1]], [a[0], b[0]], color=color, lw=0.8, zorder=2)
    ax.scatter(result.nodes[:, 1], result.nodes[:, 0], s=6, c=color, zorder=3)


def draw_path(ax, result, start, goal, color="tab:green"):
    if result.success:
        path = result.path
        ax.plot(path[:, 1], path[:, 0], color=color, lw=2.5, zorder=4, label="path")
        ax.scatter(path[:, 1], path[:, 0], s=18, c=color, zorder=5)
    ax.scatter([start[1]], [start[0]], marker="o", s=80, c="gold",
               edgecolors="k", zorder=6, label="start")
    ax.scatter([goal[1]], [goal[0]], marker="*", s=160, c="magenta",
               edgecolors="k", zorder=6, label="goal")


def plot_result(occupancy, reference_points, result, start, goal, title="", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    draw_map(ax, occupancy, reference_points)
    draw_tree(ax, result)
    draw_path(ax, result, start, goal)
    status = f"{len(result.path)} waypoints" if result.success else "no path"
    ax.set_title(f"{title}\n{status}, {result.iterations} iterations, {result.seconds:.3f}s")
    return ax
