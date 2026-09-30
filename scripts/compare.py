"""Side-by-side: reference-point-guided RRT vs. plain RRT on the same query."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refrrt import load_map, load_reference_points, plan, plan_rrt  # noqa: E402
from refrrt.viz import plot_result  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, nargs=2, default=[149, 796])
    ap.add_argument("--goal", type=int, nargs=2, default=[724, 218])
    ap.add_argument("--rrt-step", type=float, default=1.0,
                    help="plain RRT step size in pixels")
    ap.add_argument("--rrt-iterations", type=int, default=20000,
                    help="iteration budget for the plain RRT")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    occupancy = load_map()
    refs = load_reference_points()
    ours = plan(occupancy, refs, args.start, args.goal, seed=args.seed)
    base = plan_rrt(occupancy, args.start, args.goal, seed=args.seed,
                    step=args.rrt_step, goal_tolerance=max(args.rrt_step, 1.0),
                    max_iterations=args.rrt_iterations)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_result(occupancy, refs, ours, args.start, args.goal,
                title="Reference-point-guided RRT", ax=axes[0])
    plot_result(occupancy, None, base, args.start, args.goal,
                title=f"Plain RRT (step {args.rrt_step:g})", ax=axes[1])
    axes[0].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=110)
    else:
        plt.show()


if __name__ == "__main__":
    main()
