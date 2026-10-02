"""Rolling arithmetic shared by the Alpha101 and Alpha191 formulas."""

import numpy as np


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Match NumPy window reductions without a Python callback per window.

    Rank-based formulas need the same reduction order, not merely close means.
    Like pandas rolling, a full window of finite observations is required.
    Work column by column in bounded batches to avoid a rows × assets × window
    temporary allocation for a large factor universe.
    """
    rows, cols = values.shape
    window = len(weights)
    if window < 1:
        raise ValueError("rolling window must be positive")
    result = np.full((rows, cols), np.nan, dtype=float)
    if rows < window:
        return result
    weight_sum = np.sum(weights)
    for col in range(cols):
        windows = np.lib.stride_tricks.sliding_window_view(values[:, col], window)
        for start in range(0, len(windows), 4096):
            batch = windows[start : start + 4096]
            valid = np.isfinite(batch).all(axis=1)
            target = result[start + window - 1 : start + window - 1 + len(batch), col]
            target[valid] = np.sum(batch[valid] * weights, axis=1) / weight_sum
    return result
