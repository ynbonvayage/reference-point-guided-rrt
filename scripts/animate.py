"""Animate the tree growing and save it as a GIF.

Each frame shows the random sample (x), the nearest tree node, the search
sector around the node -> sample direction, and the tree so far. The final
path is highlighted at the end.

    python scripts/animate.py --out docs/images/demo.gif
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Wedge  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refrrt import load_map, load_reference_points, plan  # noqa: E402
from refrrt.viz import draw_map  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, nargs=2, default=[149, 796])
    ap.add_argument("--goal", type=int, nargs=2, default=[724, 218])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radius", type=float, default=100.0)
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("docs/images/demo.gif"))
    args = ap.parse_args()

    occupancy = load_map()
    refs = load_reference_points()
    result = plan(occupancy, refs, args.start, args.goal, radius=args.radius, seed=args.seed)
    if not result.success:
        sys.exit("planner failed; try another --seed")

    fig, ax = plt.subplots(figsize=(6, 5.4))
    draw_map(ax, occupancy, refs)
    ax.scatter([args.start[1]], [args.start[0]], s=90, c="gold", edgecolors="k", zorder=6)
    ax.scatter([args.goal[1]], [args.goal[0]], marker="*", s=200, c="magenta",
               edgecolors="k", zorder=6)
    title = ax.set_title("")
    edges = []
    dynamic = []
    hold = 2 * args.fps  # frames to hold the final path
    n_iter = len(result.trace)

    def edge_line(i, color, lw):
        a, b = result.nodes[result.parents[i]], result.nodes[i]
        return ax.plot([a[1], b[1]], [a[0], b[0]], color=color, lw=lw, zorder=3)[0]

    def update(frame):
        for artist in dynamic:
            artist.remove()
        dynamic.clear()

        if frame < n_iter:
            sample, nearest = result.trace[frame]
            near = result.nodes[nearest]
            size = result.history[frame]
            while len(edges) < size - 1:
                i = len(edges) + 1
                edges.append(edge_line(i, "tab:red", 1.5))
                ax.scatter([result.nodes[i][1]], [result.nodes[i][0]], s=14, c="tab:red", zorder=4)

            angle = np.degrees(np.arctan2(sample[0] - near[0], sample[1] - near[1]))
            dynamic.append(ax.add_patch(Wedge((near[1], near[0]), args.radius, angle - 60,
                                              angle + 60, color="yellow", alpha=0.35, zorder=2)))
            dynamic.append(ax.plot([near[1], sample[1]], [near[0], sample[0]], ls="--",
                                   color="orange", lw=0.8, zorder=2)[0])
            dynamic.append(ax.scatter([sample[1]], [sample[0]], marker="x", s=60,
                                      c="orange", zorder=5))
            is_goal = tuple(sample) == tuple(args.goal)
            title.set_text(f"iteration {frame + 1}   tree nodes: {size}\n"
                           f"sample: {'goal (bias)' if is_goal else 'random'}")
        else:
            path = result.path
            dynamic.append(ax.plot(path[:, 1], path[:, 0], color="tab:green", lw=3, zorder=7)[0])
            dynamic.append(ax.scatter(path[:, 1], path[:, 0], s=22, c="tab:green", zorder=8))
            title.set_text(f"path found: {len(path)} waypoints, {n_iter} iterations\n"
                           f"planning time {result.seconds * 1000:.1f} ms")
        return []

    anim = FuncAnimation(fig, update, frames=n_iter + hold, blit=False)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.88)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    print(f"saved {args.out} ({n_iter + hold} frames)")


if __name__ == "__main__":
    main()
