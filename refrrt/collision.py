"""Collision checking on a binary occupancy grid."""

import numpy as np


def segment_is_free(occupancy, p, q):
    """Return True if the straight segment p -> q crosses no occupied cell.

    `occupancy` is a 2-D bool array (True = obstacle); points are (row, col).
    The segment is sampled once per cell along its longer axis, so thin walls
    cannot be skipped regardless of the segment's slope.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    n = int(np.ceil(np.abs(q - p).max())) + 1
    t = np.linspace(0.0, 1.0, n)[:, None]
    cells = np.rint(p + t * (q - p)).astype(int)
    rows = np.clip(cells[:, 0], 0, occupancy.shape[0] - 1)
    cols = np.clip(cells[:, 1], 0, occupancy.shape[1] - 1)
    return not occupancy[rows, cols].any()
