"""Plan one path and save / show the figure.

    python scripts/run_demo.py                       # default start/goal
    python scripts/run_demo.py --start 149 796 --goal 724 218 --seed 0
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refrrt import load_map, load_reference_points, plan  # noqa: E402
from refrrt.viz import plot_result  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, nargs=2, default=[149, 796], metavar=("ROW", "COL"))
    ap.add_argument("--goal", type=int, nargs=2, default=[724, 218], metavar=("ROW", "COL"))
    ap.add_argument("--epsilon", type=float, default=0.5, help="probability of a random (non-goal) sample")
    ap.add_argument("--radius", type=float, default=100.0, help="sector radius in pixels")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None, help="save figure here instead of showing it")
    args = ap.parse_args()

    occupancy = load_map()
    refs = load_reference_points()
    result = plan(occupancy, refs, args.start, args.goal,
                  epsilon=args.epsilon, radius=args.radius, seed=args.seed)

    print(f"success={result.success} iterations={result.iterations} "
          f"time={result.seconds:.3f}s waypoints={len(result.path)}")
    for p in result.path:
        print(f"  {p[0]:4d} {p[1]:4d}")

    plot_result(occupancy, refs, result, args.start, args.goal,
                title="Reference-point-guided RRT")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=120)
    else:
        plt.show()


if __name__ == "__main__":
    main()
