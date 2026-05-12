"""Train/val index splits for sliding-window datasets."""
from __future__ import annotations

import numpy as np


def train_val_indices(n: int, val_ratio: float, split: str, seed: int) -> tuple[list[int], list[int]]:
    """
    split:
      temporal — first (1-val_ratio) windows in dataset order (requires windows sorted by time).
      merged_random — shuffle then take val_ratio for val.
    """
    if n < 2:
        return ([0], [0]) if n == 1 else ([], [])
    n_va = max(1, min(int(val_ratio * n), n - 1))
    if split == "temporal":
        n_tr = n - n_va
        return list(range(n_tr)), list(range(n_tr, n))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n).tolist()
    va = perm[:n_va]
    tr = perm[n_va:]
    return tr, va
