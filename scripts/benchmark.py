"""Time both planners over a fixed set of start/goal queries."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refrrt import load_map, load_reference_points, plan, plan_rrt  # noqa: E402

# (start, goal) as (row, col); the query set used in the original experiments.
QUERIES = [
    ((149, 796), (724, 218)),
    ((535, 244), (450, 240)),
    ((450, 244), (253, 367)),
    ((194, 243), (141, 572)),
    ((231, 387), (149, 796)),
    ((567, 304), (696, 241)),
    ((183, 390), (181, 610)),
    ((392, 292), (582, 276)),
    ((724, 218), (149, 796)),
    ((344, 238), (206, 624)),
    ((450, 240), (183, 390)),
    ((582, 276), (344, 238)),
    ((204, 657), (144, 656)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=10, help="random seeds per query")
    ap.add_argument("--rrt-step", type=float, default=1.0,
                    help="plain RRT step size in pixels")
    ap.add_argument("--rrt-iterations", type=int, default=20000)
    args = ap.parse_args()

    occupancy = load_map()
    refs = load_reference_points()

    print(f"{'start':>12} {'goal':>12} | {'ours ok':>7} {'time':>7} {'len':>7} | "
          f"{'rrt ok':>6} {'time':>7} {'len':>7}")
    totals = {"ours": [], "rrt": []}
    for start, goal in QUERIES:
        row = {}
        for name, fn in (("ours", lambda s: plan(occupancy, refs, start, goal, seed=s)),
                         ("rrt", lambda s: plan_rrt(occupancy, start, goal, seed=s,
                                                    step=args.rrt_step,
                                                    goal_tolerance=max(args.rrt_step, 1.0),
                                                    max_iterations=args.rrt_iterations))):
            rs = [fn(s) for s in range(args.runs)]
            ok = [r for r in rs if r.success]
            lengths = [np.linalg.norm(np.diff(r.path, axis=0), axis=1).sum() for r in ok]
            row[name] = (len(ok) / len(rs), np.mean([r.seconds for r in rs]),
                         np.mean(lengths) if lengths else float("nan"))
            totals[name].append(row[name])
        o, b = row["ours"], row["rrt"]
        print(f"{str(start):>12} {str(goal):>12} | {o[0]:7.0%} {o[1]:6.3f}s {o[2]:7.0f} | "
              f"{b[0]:6.0%} {b[1]:6.3f}s {b[2]:7.0f}")

    for name, rows in totals.items():
        a = np.array(rows)
        print(f"{name:>5}: success {a[:, 0].mean():.0%}, mean time {a[:, 1].mean():.3f}s, "
              f"mean path length {np.nanmean(a[:, 2]):.0f}px")


if __name__ == "__main__":
    main()
