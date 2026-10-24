"""Readouts of a leg's movement: how long the claw stays near the head, and how rhythmic it is.

Registered for stage 1d (`preregistration/embodied_stage1d_reach_versus_swing.json`) to tell a
reach that holds the leg at the head from a swing that passes near it.
"""

from __future__ import annotations

import numpy as np

DT_S = 0.001
START_S = 0.1
LAGS_S = (0.05, 0.5)


def time_near_head(distance_mm: np.ndarray, threshold_mm: float) -> float:
    """Fraction of samples with the claw closer to the antenna than `threshold_mm`."""
    return float(np.mean(np.asarray(distance_mm) < threshold_mm))


def rhythmicity(
    distance_mm: np.ndarray,
    dt_s: float = DT_S,
    start_s: float = START_S,
    lags_s: tuple[float, float] = LAGS_S,
) -> float:
    """Largest normalized autocorrelation at lags in `lags_s`, after `start_s`; 0 if flat."""
    x = np.asarray(distance_mm, dtype=float)[int(round(start_s / dt_s)) :]
    x = x - x.mean()
    energy = float(np.dot(x, x))
    if energy <= 1e-12:
        return 0.0
    lo, hi = (int(round(v / dt_s)) for v in lags_s)
    hi = min(hi, x.size - 1)
    return float(max(np.dot(x[:-k], x[k:]) / energy for k in range(lo, hi + 1)))
