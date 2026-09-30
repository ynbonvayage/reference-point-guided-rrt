"""Plain goal-biased RRT, used as the baseline for comparison."""

import time

import numpy as np

from .collision import segment_is_free
from .planner import PlanResult


def plan_rrt(occupancy, start, goal, *, epsilon=0.5, step=1.0,
             goal_tolerance=1.0, max_iterations=20000, seed=None):
    rng = np.random.default_rng(seed)
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    nodes = [start]
    parents = [-1]
    history = []
    t0 = time.perf_counter()

    reached = False
    iteration = 0
    while not reached and iteration < max_iterations:
        iteration += 1
        if rng.random() < epsilon:
            target = np.array([rng.integers(occupancy.shape[0]),
                               rng.integers(occupancy.shape[1])], dtype=float)
        else:
            target = goal

        tree = np.asarray(nodes)
        nearest = int(np.argmin(np.linalg.norm(tree - target, axis=1)))
        near = tree[nearest]
        direction = target - near
        length = np.linalg.norm(direction)
        if length == 0:
            continue
        new = near + direction / length * min(step, length)
        if segment_is_free(occupancy, near, new):
            nodes.append(new)
            parents.append(nearest)
            if np.linalg.norm(new - goal) <= goal_tolerance and segment_is_free(occupancy, new, goal):
                nodes.append(goal)
                parents.append(len(nodes) - 2)
                reached = True
        history.append(len(nodes))

    seconds = time.perf_counter() - t0
    nodes = np.asarray(nodes)
    parents = np.asarray(parents)
    path = []
    if reached:
        i = len(nodes) - 1
        while i != -1:
            path.append(nodes[i])
            i = parents[i]
        path.reverse()
    return PlanResult(np.asarray(path).reshape(-1, 2), nodes, parents,
                      iteration, seconds, reached, history)
