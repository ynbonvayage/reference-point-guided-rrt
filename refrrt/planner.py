"""Reference-point-guided RRT.

Classic RRT grows its tree pixel by pixel toward random samples, which is slow
on large maps, gets stuck in narrow corridors and produces a different, often
wiggly path on every run. This planner keeps the RRT loop but only lets the
tree grow onto a small, hand-picked set of *reference points*:

1. Sample a direction target: the goal with probability 1 - epsilon,
   otherwise a uniformly random cell of the map.
2. Find the tree node nearest to that sample.
3. Candidate filter ("RangeWithin"): keep the reference points that lie in a
   sector anchored at the nearest node, opening +/- `half_angle` around the
   node -> sample direction, with radius < `radius`.
4. Collision filter: drop candidates whose straight connection to the node
   crosses an obstacle.
5. Extend the tree to the closest surviving candidate.
6. Stop once the goal is added; the answer is the tree path root -> goal.

Because every choice is made among a few dozen reference points instead of
the ~1M map cells, the search is fast, the sector keeps growth pointed at the
sample, and the reference points (chosen so that neighbours see each other)
carry the tree through narrow passages.
"""

from dataclasses import dataclass, field
import time

import numpy as np

from .collision import segment_is_free


@dataclass
class PlanResult:
    path: np.ndarray               # (k, 2) waypoints from start to goal; empty if failed
    nodes: np.ndarray              # (n, 2) every tree node
    parents: np.ndarray            # (n,) parent index of each node, -1 for the root
    iterations: int
    seconds: float
    success: bool
    history: list = field(default_factory=list)  # tree size after each iteration


def _sample(rng, shape, goal, epsilon):
    if rng.random() < epsilon:
        return np.array([rng.integers(shape[0]), rng.integers(shape[1])])
    return goal


def _candidates_in_sector(refs, near, target, radius, half_angle):
    """Indices of reference points inside the sector near -> target."""
    to_target = target - near
    to_refs = refs - near
    dist = np.linalg.norm(to_refs, axis=1)
    norm = np.linalg.norm(to_target)
    if norm == 0:
        return np.array([], dtype=int)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = to_refs @ to_target / (dist * norm)
    mask = (dist > 0) & (dist < radius) & (cos >= np.cos(half_angle))
    return np.flatnonzero(mask)


def plan(occupancy, reference_points, start, goal, *, epsilon=0.5,
         radius=100.0, half_angle=np.pi / 3, max_iterations=20000, seed=None):
    """Plan a path from `start` to `goal` (both (row, col)) on `occupancy`.

    The goal is added to the reference set if it is not already in it, since
    the tree can only ever reach reference points.
    """
    rng = np.random.default_rng(seed)
    start = np.asarray(start, dtype=int)
    goal = np.asarray(goal, dtype=int)
    refs = np.asarray(reference_points, dtype=int)
    if not (refs == goal).all(axis=1).any():
        refs = np.vstack([refs, goal])

    nodes = [start]
    parents = [-1]
    in_tree = {tuple(start)}
    history = []
    t0 = time.perf_counter()

    reached = tuple(start) == tuple(goal)
    iteration = 0
    while not reached and iteration < max_iterations:
        iteration += 1
        target = _sample(rng, occupancy.shape, goal, epsilon)

        tree = np.asarray(nodes)
        nearest = int(np.argmin(np.linalg.norm(tree - target, axis=1)))
        near = tree[nearest]

        candidates = _candidates_in_sector(refs, near, target, radius, half_angle)
        free = [i for i in candidates if segment_is_free(occupancy, near, refs[i])]
        if free:
            best = min(free, key=lambda i: np.linalg.norm(refs[i] - near))
            new = refs[best]
            if tuple(new) not in in_tree:
                nodes.append(new)
                parents.append(nearest)
                in_tree.add(tuple(new))
                reached = tuple(new) == tuple(goal)
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
