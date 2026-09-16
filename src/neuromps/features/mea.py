"""The single MEA feature extractor: real recordings and the digital twin both call this.

Inputs are spike times per electrode, in seconds. Outputs are exactly the
`features` fields of `schema/neuromps_schema.json`. A feature that is undefined
for the input (no active electrodes, no bursts) is ``None``, never 0.0.

Definitions are chosen to be stated, not standard: burst detection and synchrony
differ between tools, so `FeatureParams.as_method_record()` emits the method
names that the schema requires alongside the values.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import binom

SECONDS_PER_MINUTE = 60.0


@dataclass(frozen=True)
class FeatureParams:
    """Feature definitions. Defaults follow the EPA acute MEA assay where it defines one."""

    active_min_spikes_per_min: float = 5.0
    burst_max_isi_s: float = 0.1
    burst_min_spikes: int = 5
    network_burst_bin_s: float = 0.025
    network_burst_min_electrodes: int = 3
    network_burst_min_fraction: float = 0.25
    network_burst_false_positives: float = 0.05
    sttc_dt_s: float = 0.05

    def as_method_record(self) -> dict[str, object]:
        """Method fields for a schema record (see STANDARD.md section 4.2)."""
        return {
            "computed_by": "neuromps.features.mea",
            "burst_method": (
                f"maxinterval_simple(isi<={self.burst_max_isi_s}s,>={self.burst_min_spikes}spikes)"
            ),
            "network_burst_method": (
                f"binned_participation(bin={self.network_burst_bin_s}s,"
                f">=max({self.network_burst_min_electrodes},"
                f"{self.network_burst_min_fraction}*active,chance)electrodes,"
                f"chance_fp<{self.network_burst_false_positives}/recording)"
            ),
            "synchrony_method": "sttc",
            "active_electrode_threshold_spikes_per_min": self.active_min_spikes_per_min,
        }


def _bursts(times: np.ndarray, params: FeatureParams) -> list[tuple[int, int]]:
    """Index ranges [start, end] of bursts: runs of spikes with ISI <= threshold."""
    if times.size < params.burst_min_spikes:
        return []
    close = np.diff(times) <= params.burst_max_isi_s
    out, start = [], None
    for i, is_close in enumerate(close):
        if is_close and start is None:
            start = i
        elif not is_close and start is not None:
            if i - start + 1 >= params.burst_min_spikes:
                out.append((start, i))
            start = None
    if start is not None and times.size - start >= params.burst_min_spikes:
        out.append((start, times.size - 1))
    return out


def _sttc_pair(a: np.ndarray, b: np.ndarray, duration_s: float, dt: float) -> float | None:
    """Spike time tiling coefficient (Cutts & Eglen 2014) for one electrode pair."""
    if a.size == 0 or b.size == 0:
        return None
    ta = _tiled_fraction(a, duration_s, dt)
    tb = _tiled_fraction(b, duration_s, dt)
    pa = _proportion_within(a, b, dt)
    pb = _proportion_within(b, a, dt)
    if any(np.isclose(x, 1.0) for x in (pa * tb, pb * ta)):
        return None
    return 0.5 * ((pa - tb) / (1 - pa * tb) + (pb - ta) / (1 - pb * ta))


def _tiled_fraction(times: np.ndarray, duration_s: float, dt: float) -> float:
    """Fraction of the recording covered by +/- dt tiles around spikes."""
    starts = np.maximum(times - dt, 0.0)
    ends = np.minimum(times + dt, duration_s)
    order = np.argsort(starts)
    starts, ends = starts[order], ends[order]
    covered, current_end = 0.0, -np.inf
    for s, e in zip(starts, ends, strict=True):
        s = max(s, current_end)
        if e > s:
            covered += e - s
            current_end = e
    return covered / duration_s


def _proportion_within(a: np.ndarray, b: np.ndarray, dt: float) -> float:
    """Fraction of spikes in ``a`` that fall within dt of any spike in ``b``."""
    idx = np.searchsorted(b, a)
    left = np.clip(idx - 1, 0, b.size - 1)
    right = np.clip(idx, 0, b.size - 1)
    nearest = np.minimum(np.abs(a - b[left]), np.abs(a - b[right]))
    return float(np.mean(nearest <= dt))


def _network_burst_rate(
    trains: list[np.ndarray], duration_s: float, params: FeatureParams
) -> float:
    """Network events per minute: contiguous bins where enough electrodes fire together.

    This is coincidence-based, in the spirit of Axion's "network spike", not a
    rate-elevation burst detector: electrodes firing synchronously but tonically
    still count. Recorded as ``binned_participation`` in the method record.
    """
    n_bins = max(int(np.ceil(duration_s / params.network_burst_bin_s)), 1)
    threshold = max(
        params.network_burst_min_electrodes,
        int(np.ceil(params.network_burst_min_fraction * len(trains))),
        _chance_threshold(trains, duration_s, n_bins, params),
    )
    if len(trains) < threshold:
        return 0.0
    participants = np.zeros(n_bins, dtype=int)
    for t in trains:
        bins = np.unique(np.clip((t / params.network_burst_bin_s).astype(int), 0, n_bins - 1))
        participants[bins] += 1
    active_bins = participants >= threshold
    starts = np.flatnonzero(active_bins & ~np.r_[False, active_bins[:-1]])
    return len(starts) / (duration_s / SECONDS_PER_MINUTE)


def _chance_threshold(
    trains: list[np.ndarray], duration_s: float, n_bins: int, params: FeatureParams
) -> int:
    """Smallest participation count whose expected chance occurrences stay below the budget.

    Without this, independent electrodes firing at a few Hz coincide often enough
    in a 25 ms bin to produce tens of spurious "network bursts" per minute.
    Electrodes are modelled as independent Poisson processes at their own rates.
    """
    n = len(trains)
    rates = np.array([t.size / duration_s for t in trains])
    p_fire = float(np.mean(1.0 - np.exp(-rates * params.network_burst_bin_s)))
    for threshold in range(2, n + 1):
        expected = n_bins * float(binom.sf(threshold - 1, n, p_fire))
        if expected < params.network_burst_false_positives:
            return threshold
    return n + 1  # not reachable by chance control alone


def extract_features(
    spikes: Mapping[str, np.ndarray], duration_s: float, params: FeatureParams | None = None
) -> dict[str, float | int | None]:
    """Compute the core feature vector from per-electrode spike times (seconds).

    Args:
        spikes: electrode id -> spike times in seconds; order is irrelevant.
        duration_s: recording duration, used for rates and tiling.
        params: feature definitions; defaults to :class:`FeatureParams`.

    Returns:
        The `features` block of the schema. Undefined features are ``None``.
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    p = params or FeatureParams()
    trains = {k: np.sort(np.asarray(v, dtype=float)) for k, v in spikes.items()}
    min_spikes = p.active_min_spikes_per_min * duration_s / SECONDS_PER_MINUTE
    active = [t for t in trains.values() if t.size >= min_spikes and t.size > 0]
    out: dict[str, float | int | None] = dict.fromkeys(
        (
            "mfr_hz",
            "burst_rate_per_min",
            "burst_duration_s",
            "spikes_per_burst",
            "pct_spikes_in_bursts",
            "network_burst_rate_per_min",
            "synchrony_index",
            "isi_cv",
        )
    )
    out["active_electrodes"] = len(active)
    if not active:
        return out
    out["mfr_hz"] = float(np.mean([t.size / duration_s for t in active]))
    out["isi_cv"] = _mean_isi_cv(active)
    out.update(_burst_features(active, duration_s, p))
    out["network_burst_rate_per_min"] = _network_burst_rate(active, duration_s, p)
    out["synchrony_index"] = _mean_sttc(active, duration_s, p)
    return out


def _mean_isi_cv(active: list[np.ndarray]) -> float | None:
    """Mean over electrodes of the coefficient of variation of inter-spike intervals."""
    cvs = []
    for t in active:
        isi = np.diff(t)
        if isi.size >= 2 and isi.mean() > 0:
            cvs.append(isi.std(ddof=1) / isi.mean())
    return float(np.mean(cvs)) if cvs else None


def _burst_features(
    active: list[np.ndarray], duration_s: float, p: FeatureParams
) -> dict[str, float | None]:
    """Burst rate, duration, size and the share of spikes inside bursts."""
    n_bursts, durations, sizes, spikes_in_bursts, total = 0, [], [], 0, 0
    for t in active:
        total += t.size
        for start, end in _bursts(t, p):
            n_bursts += 1
            durations.append(t[end] - t[start])
            sizes.append(end - start + 1)
            spikes_in_bursts += end - start + 1
    return {
        "burst_rate_per_min": n_bursts / len(active) / (duration_s / SECONDS_PER_MINUTE),
        "burst_duration_s": float(np.mean(durations)) if durations else None,
        "spikes_per_burst": float(np.mean(sizes)) if sizes else None,
        "pct_spikes_in_bursts": 100.0 * spikes_in_bursts / total if total else None,
    }


def _mean_sttc(active: list[np.ndarray], duration_s: float, p: FeatureParams) -> float | None:
    """Mean pairwise STTC over electrode pairs."""
    values = [
        v
        for i in range(len(active))
        for j in range(i + 1, len(active))
        if (v := _sttc_pair(active[i], active[j], duration_s, p.sttc_dt_s)) is not None
    ]
    return float(np.mean(values)) if values else None
