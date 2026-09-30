"""Python planning time vs. the times recorded for the original MATLAB version.

The MATLAB times were measured with tic/toc around the whole planning loop,
which also redraws the tree and calls pause(0.01) on every iteration, so they
include plotting overhead. Python times cover the algorithm only.

Prints a Markdown table.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from refrrt import load_map, load_reference_points, plan  # noqa: E402

# (start, goal, MATLAB seconds); coordinates are MATLAB 1-based (row, col).
MATLAB_RUNS = [
    ((150, 797), (725, 219), 2.520581),
    ((536, 245), (451, 241), 0.471973),
    ((451, 245), (254, 368), 1.472255),
    ((195, 244), (142, 573), 0.916898),
    ((232, 388), (150, 797), 1.194246),
    ((568, 305), (697, 242), 2.991678),
    ((184, 391), (182, 611), 0.809278),
    ((393, 293), (583, 277), 1.001584),
    ((725, 219), (150, 797), 3.050010),
    ((345, 239), (207, 625), 1.347358),
    ((345, 239), (207, 625), 13.725946),
    ((451, 241), (184, 391), 0.996241),
    ((583, 277), (345, 239), 0.783923),
    ((205, 658), (145, 657), 6.948217),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20, help="random seeds per query")
    args = ap.parse_args()

    occupancy = load_map()
    refs = load_reference_points()

    print("| # | Start | Goal | MATLAB time (s) | Python time (ms) | Python iterations | Speed-up |")
    print("|---|---|---|---|---|---|---|")
    m_all, p_all, it_all = [], [], []
    for n, (start, goal, t_matlab) in enumerate(MATLAB_RUNS, 1):
        s0 = tuple(v - 1 for v in start)
        g0 = tuple(v - 1 for v in goal)
        rs = [plan(occupancy, refs, s0, g0, seed=s) for s in range(args.runs)]
        assert all(r.success for r in rs)
        t_py = np.mean([r.seconds for r in rs])
        its = np.mean([r.iterations for r in rs])
        m_all.append(t_matlab)
        p_all.append(t_py)
        it_all.append(its)
        print(f"| {n} | {start} | {goal} | {t_matlab:.3f} | {t_py * 1000:.2f} | {its:.0f} | "
              f"{t_matlab / t_py:,.0f}× |")
    m, p = np.mean(m_all), np.mean(p_all)
    print(f"| | **Mean** | | **{m:.3f}** | **{p * 1000:.2f}** | **{np.mean(it_all):.0f}** | "
          f"**{m / p:,.0f}×** |")


if __name__ == "__main__":
    main()
