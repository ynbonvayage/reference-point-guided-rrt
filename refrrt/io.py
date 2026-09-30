"""Loading the map and reference points shipped in data/."""

from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_map(path=DATA_DIR / "map.npz"):
    """Binary occupancy grid, True = obstacle. Shape (rows, cols)."""
    return np.load(path)["occupancy"]


def load_reference_points(path=DATA_DIR / "reference_points.csv"):
    """(n, 2) int array of (row, col) reference points."""
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=int)
