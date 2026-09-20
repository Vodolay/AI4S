"""Cluster bootstrap: resample cultures, not samples.

Two conditions of one well share a baseline and a culture, so resampling rows
would treat them as independent evidence and report intervals that are too
narrow. Every interval in this project resamples **groups** (source : species :
unit) with replacement and takes all rows of each drawn group.

Intervals are percentile intervals. The point estimate is always reported next
to them, because with a few dozen groups a percentile interval is itself noisy.
"""

import logging
from collections.abc import Callable, Iterator

import numpy as np

log = logging.getLogger("neuromps.models")
DEFAULT_RESAMPLES = 1000
QUICK_RESAMPLES = 200


def resample_count(quick: bool) -> int:
    """Resamples to use; quick mode keeps `run_all.py --quick` inside its budget."""
    return QUICK_RESAMPLES if quick else DEFAULT_RESAMPLES


def group_bootstrap_indices(
    groups: np.ndarray, rng: np.random.Generator, n_resamples: int
) -> Iterator[np.ndarray]:
    """Yield row indices for each resample, drawing whole groups with replacement."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    rows_of = {name: np.flatnonzero(groups == name) for name in unique}
    for _ in range(n_resamples):
        drawn = rng.choice(unique, size=unique.size, replace=True)
        yield np.concatenate([rows_of[name] for name in drawn])


def ci95(values) -> list[float | None]:
    """Percentile 95% interval, or [None, None] when nothing could be computed."""
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return [None, None]
    return [
        round(float(np.percentile(finite, 2.5)), 4),
        round(float(np.percentile(finite, 97.5)), 4),
    ]


def group_bootstrap_metric(
    metric: Callable[[np.ndarray], float],
    groups: np.ndarray,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> dict:
    """Bootstrap a metric over groups.

    Args:
        metric: takes row indices and returns a scalar, or NaN when undefined
            for that resample (for example a class missing from the draw).
        groups: group label per row.
        n_resamples: number of resamples; recorded in the result.
        seed: seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    values = []
    for idx in group_bootstrap_indices(groups, rng, n_resamples):
        try:
            values.append(float(metric(idx)))
        except ValueError:  # a resample that a metric cannot score at all
            continue
    return {
        "ci95": ci95(values),
        "n_resamples": int(n_resamples),
        "n_valid_resamples": int(len([v for v in values if np.isfinite(v)])),
    }
