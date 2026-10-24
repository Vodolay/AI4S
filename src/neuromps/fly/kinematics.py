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


MIN_PROMINENCE = 0.1


def rhythm(
    signal: np.ndarray,
    dt_s: float = DT_S,
    start_s: float = START_S,
    lags_s: tuple[float, float] = LAGS_S,
    min_prominence: float = MIN_PROMINENCE,
) -> tuple[float, float | None]:
    """Corrected rhythm measure (the audit): (score, period in s) from an autocorrelation peak.

    A peak is a local maximum of the normalized autocorrelation within `lags_s`, above 0,
    rising at least `min_prominence` above the lowest value before it. The score is the
    highest such peak and the period its lag; (0.0, None) when there is none. Unlike
    `rhythmicity`, a smooth signal without a rhythm scores 0.
    """
    x = np.asarray(signal, dtype=float)[int(round(start_s / dt_s)) :]
    x = x - x.mean()
    energy = float(np.dot(x, x))
    if energy <= 1e-12:
        return 0.0, None
    lo, hi = (int(round(v / dt_s)) for v in lags_s)
    hi = min(hi, x.size - 2)
    c = np.ones(hi + 2)
    for k in range(1, hi + 2):
        c[k] = np.dot(x[:-k], x[k:]) / energy
    trough = np.minimum.accumulate(c)
    best, period = 0.0, None
    for k in range(lo, hi + 1):
        peak = c[k] > c[k - 1] and c[k] >= c[k + 1] and c[k] > 0
        if peak and c[k] - trough[k - 1] >= min_prominence and c[k] > best:
            best, period = float(c[k]), k * dt_s
    return best, period
