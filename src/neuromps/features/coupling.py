"""Directed coupling between compartments of a multi-chamber device.

The twin's central claim is about asymmetric connectivity, so the asset needs a
measured, stated quantity to compare against. This module computes the
cross-correlation of the two compartments' binned population rates and reports:

- ``peak_correlation``: how strongly the compartments co-vary at their best lag.
- ``lag_ms``: the lag of that peak. Positive means the first compartment leads.
- ``asymmetry``: (forward mass - backward mass) / (forward + backward), where
  mass is the correlation summed over positive and negative lags. 0 means
  symmetric coupling, +1 means all influence runs from the first compartment to
  the second. It is reported **only when the peak correlation clears a noise
  floor** of ``SIGMA_THRESHOLD`` standard errors (1/sqrt(bins)); otherwise there
  is no coupling to apportion and dividing noise by noise yields a confident
  direction out of nothing.

This is a descriptive measure, not a causal one: common drive to both
compartments produces correlation at zero lag with no asymmetry, and the measure
cannot distinguish a direct projection from a shared input that reaches one
compartment earlier.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import signal

MS_PER_S = 1000.0
SIGMA_THRESHOLD = 5.0  # standard errors above the uncorrelated noise floor


@dataclass(frozen=True)
class CouplingParams:
    """Binning and lag window for the cross-correlation."""

    bin_s: float = 0.005
    max_lag_s: float = 0.2

    def as_method_record(self) -> dict[str, object]:
        """Method fields for a schema record."""
        return {
            "coupling_method": (
                f"binned_cross_correlation(bin={self.bin_s}s,max_lag={self.max_lag_s}s)"
            )
        }


def population_rate(
    spikes: Mapping[str, np.ndarray], duration_s: float, bin_s: float
) -> tuple[np.ndarray, float]:
    """Spike counts per bin, pooled over the electrodes of one compartment."""
    n_bins = max(int(np.ceil(duration_s / bin_s)), 1)
    counts = np.zeros(n_bins)
    for times in spikes.values():
        if times.size:
            idx = np.clip((np.asarray(times) / bin_s).astype(int), 0, n_bins - 1)
            counts += np.bincount(idx, minlength=n_bins)
    return counts, bin_s


def directed_coupling(
    source: Mapping[str, np.ndarray],
    target: Mapping[str, np.ndarray],
    duration_s: float,
    params: CouplingParams | None = None,
) -> dict[str, float | None]:
    """Cross-correlation summary of source -> target coupling.

    Returns ``peak_correlation``, ``lag_ms`` (positive: source leads) and
    ``asymmetry``. All are ``None`` when either compartment is silent, and
    ``lag_ms``/``asymmetry`` are ``None`` when the peak correlation does not
    clear the noise floor.
    """
    p = params or CouplingParams()
    a, _ = population_rate(source, duration_s, p.bin_s)
    b, _ = population_rate(target, duration_s, p.bin_s)
    if a.sum() == 0 or b.sum() == 0 or a.std() == 0 or b.std() == 0:
        return {"peak_correlation": None, "lag_ms": None, "asymmetry": None}
    corr, lags = _cross_correlation(a, b, int(round(p.max_lag_s / p.bin_s)))
    peak = int(np.argmax(np.abs(corr)))
    peak_corr = float(corr[peak])
    noise_floor = SIGMA_THRESHOLD / np.sqrt(a.size)
    if abs(peak_corr) < noise_floor:
        return {"peak_correlation": peak_corr, "lag_ms": None, "asymmetry": None}
    forward = float(np.sum(np.clip(corr[lags > 0], 0, None)))
    backward = float(np.sum(np.clip(corr[lags < 0], 0, None)))
    total = forward + backward
    return {
        "peak_correlation": peak_corr,
        "lag_ms": float(lags[peak] * p.bin_s * MS_PER_S),
        "asymmetry": float((forward - backward) / total) if total > 0 else 0.0,
    }


def _cross_correlation(
    a: np.ndarray, b: np.ndarray, max_lag_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Pearson cross-correlation of two rate vectors over +/- max_lag bins.

    Uses an FFT correlation: a direct loop over lags costs ~10^8 operations per
    compartment pair, which is minutes across a whole archive. A positive lag
    means ``a`` leads ``b``.
    """
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    full = signal.correlate(b, a, mode="full", method="fft") / a.size
    centre = a.size - 1
    max_lag_bins = min(max_lag_bins, a.size - 1)
    window = full[centre - max_lag_bins : centre + max_lag_bins + 1]
    return window, np.arange(-max_lag_bins, max_lag_bins + 1)
