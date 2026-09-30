"""Grid of planned paths for several start/goal queries."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refrrt import load_map, load_reference_points, plan  # noqa: E402
from refrrt.viz import plot_result  # noqa: E402

QUERIES = [
    ((724, 218), (149, 796)),
    ((344, 238), (206, 624)),
    ((183, 390), (181, 610)),
    ((582, 276), (344, 238)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    occupancy = load_map()
    refs = load_reference_points()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (start, goal) in zip(axes.flat, QUERIES):
        result = plan(occupancy, refs, start, goal, seed=args.seed)
        plot_result(occupancy, refs, result, start, goal,
                    title=f"{start} -> {goal}", ax=ax)
    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=100)
    else:
        plt.show()


if __name__ == "__main__":
    main()
